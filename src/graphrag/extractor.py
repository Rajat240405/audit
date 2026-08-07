"""
Entity and relationship extraction via a configurable LLM provider.

The LLM backend is selected by ``GRAPHRAG_LLM_PROVIDER``:
  - ``ollama``            local Ollama (default)
  - ``groq``              Groq cloud (multi-key failover via GROQ_API_KEYS)
  - ``openai_compatible`` any OpenAI-compatible endpoint (future)

Design goals (per the production spec):
- Deterministic metadata entities (ministry, document) are added without the LLM.
- LLM-extracted entities must be *grounded*: the entity name must appear in the
  document text (normalized substring match). Ungrounded entities are dropped,
  never invented.
- Relationships must connect two grounded entities and must carry a verbatim
  evidence snippet from the document. Ungrounded edges are dropped.
- Strict JSON contract (provider-specific JSON mode); parse failures are
  retried a bounded number of times, then the document is marked failed (never
  silently coerced into fake content).

The JSON schema returned by the model is identical across providers, so the
rest of the pipeline (grounding, validation, Neo4j, embeddings) is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from pydantic import ValidationError

from src.graphrag.config import GraphRAGConfig
from src.graphrag.llm import (
    LLMBackendExhaustedError,
    build_llm_provider,
)
from src.graphrag.models import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATION_TYPES,
    DocumentRecord,
    Entity,
    EntityType,
    Relationship,
    RelationType,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an entity and relationship extractor for Indian parliamentary "
    "Question & Answer documents from the Ministry of Earth Sciences (and its "
    "predecessor, Ocean Development). Extract ONLY entities and relationships "
    "that are EXPLICITLY stated in the document text. Do not infer, guess or "
    "add world knowledge.\n"
    f"Allowed entity types: {sorted(ALLOWED_ENTITY_TYPES)}\n"
    f"Allowed relationship types: {sorted(ALLOWED_RELATION_TYPES)}\n"
    "Respond with strict JSON only, no prose, no markdown fences, matching:\n"
    '{"entities":[{"name":"...","type":"<one of the allowed types>"}],'
    '"relationships":[{"source_name":"...","source_type":"<type>",'
    '"relation":"<one of the allowed types>","target_name":"...",'
    '"target_type":"<type>"}]}\n'
    "If nothing qualifies, return {\"entities\":[],\"relationships\":[]}."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


class ExtractionError(Exception):
    """Raised when extraction definitively fails for a document."""


class EntityRelationshipExtractor:
    """Extracts grounded entities + relationships from a document via Ollama."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.provider = build_llm_provider(config)
        self.stats: dict = {
            "calls": 0,
            "retries": 0,
            "json_parse_failures": 0,
            "provider": None,  # last provider that handled a request
            "model": None,     # last model that handled a request
            "key": None,       # last masked API key
        }

    # ── audit helpers (failover observability) ─────────────────────────

    def drain_events(self) -> list[dict]:
        """Return failover events (key/model switches) since the last drain."""
        return self.provider.drain_events()

    def usage_summary(self) -> dict:
        """Per-(provider, model, masked-key) request counts."""
        return self.provider.usage_summary()

    def switch_counts(self) -> dict:
        return dict(self.provider.switch_counts)

    # ── public API ──────────────────────────────────────────────────────

    def extract(self, doc: DocumentRecord) -> tuple[list[Entity], list[Relationship]]:
        """
        Return (entities, relationships) for a document.

        Deterministic metadata entities are always included. LLM results are
        grounded before being returned. Raises ExtractionError if the LLM
        cannot produce valid JSON after the configured number of attempts.
        """
        entities, relationships, _ = self.extract_with_rejections(doc)
        return entities, relationships

    def extract_with_rejections(
        self, doc: DocumentRecord
    ) -> tuple[list[Entity], list[Relationship], dict]:
        """
        Like ``extract`` but also returns rejection details for reporting:
        (entities, relationships, rejections) where ``rejections`` is
        {"entities": [...], "relationships": [...]}.
        """
        text = f"QUESTION: {doc.question_text}\n\nANSWER: {doc.answer_text}"
        if len(text) > self.config.extract_max_chars:
            text = text[: self.config.extract_max_chars] + "\n...[truncated]"

        # Deterministic metadata entities (grounded by definition) are added
        # BEFORE relationship grounding so LLM edges can reference them.
        metadata_entities: list[Entity] = []
        if doc.ministry:
            metadata_entities.append(Entity(name=doc.ministry, type=EntityType.MINISTRY))

        payload = self._call_llm(text, context=doc.question_id)
        entities, relationships, rejections = self._parse_and_ground(
            payload, text, extra_entities=metadata_entities
        )

        # Merge metadata entities (dedupe against LLM entities).
        seen = {(e.type.value, e.name) for e in entities}
        for me in metadata_entities:
            if (me.type.value, me.name) not in seen:
                entities.append(me)
                seen.add((me.type.value, me.name))

        return entities, relationships, rejections

    # ── internals ───────────────────────────────────────────────────────

    def _call_llm(self, text: str, context: Optional[str] = None) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT:\n{text}"
        last_error: Optional[Exception] = None
        for attempt in range(self.config.extract_max_attempts):
            self.stats["calls"] += 1
            try:
                result = self.provider.generate(
                    prompt,
                    temperature=0.0,
                    max_tokens=2048,
                    timeout_seconds=self.config.llm_timeout_seconds,
                    context=context,
                )
                self.stats["provider"] = result.provider
                self.stats["model"] = result.model
                self.stats["key"] = result.key_label
                parsed = self._parse_json(result.text)
                if parsed is None:
                    self.stats["json_parse_failures"] += 1
                    raise ValueError("LLM returned unparsable JSON")
                return parsed
            except LLMBackendExhaustedError:
                # All providers/keys exhausted — propagate so the pipeline can
                # stop cleanly and save the checkpoint for later resume.
                raise
            except Exception as e:  # noqa: BLE001 - parse/transient failures
                last_error = e
                self.stats["retries"] += 1
                if attempt < self.config.extract_max_attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise ExtractionError(
            f"LLM extraction failed after {self.config.extract_max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        raw = raw.strip()
        # Strip markdown fences if the model added them despite format=json.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to salvage a trailing-garbage JSON object.
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    def _parse_and_ground(
        self,
        payload: dict,
        doc_text: str,
        extra_entities: Optional[list[Entity]] = None,
    ) -> tuple[list[Entity], list[Relationship], dict]:
        norm_doc = _normalize(doc_text)
        rejections: dict = {"entities": [], "relationships": []}

        # ── entities ────────────────────────────────────────────────────
        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()
        for extra in extra_entities or []:
            key = (extra.type.value, extra.name)
            seen.add(key)
            entities.append(extra)
        for raw in payload.get("entities", []):
            if not isinstance(raw, dict):
                rejections["entities"].append({"raw": raw, "reason": "malformed"})
                continue
            name = str(raw.get("name", "")).strip()
            etype_raw = str(raw.get("type", "")).strip()
            if etype_raw not in ALLOWED_ENTITY_TYPES:
                rejections["entities"].append(
                    {"name": name, "type": etype_raw, "reason": "type_not_allowed"}
                )
                continue  # not an allowed type -> drop (never invent)
            # Grounding guard: name must appear in the document text.
            if _normalize(name) not in norm_doc:
                rejections["entities"].append(
                    {"name": name, "type": etype_raw, "reason": "not_grounded"}
                )
                continue
            key = (etype_raw, name)
            if key in seen:
                rejections["entities"].append(
                    {"name": name, "type": etype_raw, "reason": "duplicate"}
                )
                continue
            seen.add(key)
            try:
                entities.append(Entity(name=name, type=EntityType(etype_raw)))
            except ValidationError:
                rejections["entities"].append(
                    {"name": name, "type": etype_raw, "reason": "validation"}
                )
                continue

        # ── relationships ───────────────────────────────────────────────
        # Endpoint lookup is case/whitespace-insensitive so LLM casing drift
        # (e.g. "imd" vs "IMD") does not silently drop valid edges; the stored
        # name is canonicalized to the grounded entity's name.
        entity_lookup: dict[tuple[str, str], Entity] = {
            (e.type.value, _normalize(e.name)): e for e in entities
        }
        relationships: list[Relationship] = []
        seen_rel: set[tuple] = set()
        for raw in payload.get("relationships", []):
            if not isinstance(raw, dict):
                continue
            s_name = str(raw.get("source_name", "")).strip()
            s_type = str(raw.get("source_type", "")).strip()
            rel = str(raw.get("relation", "")).strip()
            t_name = str(raw.get("target_name", "")).strip()
            t_type = str(raw.get("target_type", "")).strip()
            if rel not in ALLOWED_RELATION_TYPES:
                rejections["relationships"].append(
                    {"source": s_name, "relation": rel, "target": t_name,
                     "reason": "relation_type_not_allowed"}
                )
                continue
            src_ent = entity_lookup.get((s_type, _normalize(s_name)))
            tgt_ent = entity_lookup.get((t_type, _normalize(t_name)))
            if src_ent is None or tgt_ent is None:
                rejections["relationships"].append(
                    {"source": s_name, "relation": rel, "target": t_name,
                     "reason": "endpoint_not_grounded"}
                )
                continue  # endpoints must be grounded entities
            s_name, t_name = src_ent.name, tgt_ent.name
            # Evidence snippet: a window of the doc containing both names.
            evidence = self._find_evidence(norm_doc, s_name, t_name)
            if evidence is None:
                rejections["relationships"].append(
                    {"source": s_name, "relation": rel, "target": t_name,
                     "reason": "no_evidence"}
                )
                continue
            key = (s_type, s_name, rel, t_type, t_name)
            if key in seen_rel:
                rejections["relationships"].append(
                    {"source": s_name, "relation": rel, "target": t_name,
                     "reason": "duplicate"}
                )
                continue
            seen_rel.add(key)
            try:
                relationships.append(
                    Relationship(
                        source_type=EntityType(s_type),
                        source_name=s_name,
                        relation=RelationType(rel),
                        target_type=EntityType(t_type),
                        target_name=t_name,
                        evidence=evidence,
                    )
                )
            except ValidationError:
                rejections["relationships"].append(
                    {"source": s_name, "relation": rel, "target": t_name,
                     "reason": "validation"}
                )
                continue

        return entities, relationships, rejections

    @staticmethod
    def _find_evidence(norm_doc: str, a: str, b: str, window: int = 260) -> Optional[str]:
        na, nb = _normalize(a), _normalize(b)
        ia, ib = norm_doc.find(na), norm_doc.find(nb)
        if ia < 0 or ib < 0:
            return None
        start, end = min(ia, ib), max(ia, ib)
        snippet = norm_doc[max(0, start - 40): min(len(norm_doc), end + len(na) + 40)]
        if len(snippet) > window:
            snippet = snippet[:window] + "..."
        return snippet or None
