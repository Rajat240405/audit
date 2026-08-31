"""
Entity and relationship extraction via a configurable LLM provider.

The LLM backend is selected by ``GRAPHRAG_LLM_PROVIDER``:
  - ``ollama``            local Ollama (default)
  - ``ollama``           local Ollama (default)

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
    DocumentExtractionError,
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
    "predecessor, Ocean Development). Extract ONLY what is EXPLICITLY written "
    "in the document. Do not infer, guess, or add world knowledge.\n"
    "\n"
    "VERBATIM RULE (most important):\n"
    "- Every entity `name` must be copied EXACTLY as it appears in the document: "
    "same spelling, same casing, same abbreviation. Never expand abbreviations "
    "(e.g. keep \"IMD\", never write \"India Meteorological Department\"), never "
    "use a modern or official alternative name, never add or remove words, "
    "never title-case or normalize.\n"
    "- Only emit an entity if its exact string occurs in the document text.\n"
    "\n"
    "RELATIONSHIP RULE:\n"
    "- Only emit a relationship between two entities whose exact names BOTH "
    "appear in the document.\n"
    "- The document must explicitly support the relationship with an operative "
    "verb (e.g. implements, operates, monitors, forecasts, located in, funded "
    "by, reports to, part of, uses, related to, collaborates with).\n"
    "- Do not emit a relationship merely because both entities are mentioned "
    "without an explicit link between them.\n"
    "\n"
    "BE CONSERVATIVE:\n"
    "- When in doubt, omit. An empty result is better than a wrong one.\n"
    "- Prefer fewer, certain entities and relationships over many speculative "
    "ones.\n"
    "- Never fabricate an entity name, a relationship, or a supporting phrase.\n"
    "\n"
    f"Allowed entity types: {sorted(ALLOWED_ENTITY_TYPES)}\n"
    f"Allowed relationship types: {sorted(ALLOWED_RELATION_TYPES)}\n"
    "Respond with strict JSON only, no prose, no markdown fences, matching:\n"
    '{"entities":[{"name":"...","type":"<one of the allowed types>"}],'
    '"relationships":[{"source_name":"...","source_type":"<type>",'
    '"relation":"<one of the allowed types>","target_name":"...",'
    '"target_type":"<type>"}]}\n'
    "If nothing qualifies, return {\\\"entities\\\":[],\\\"relationships\\\":[]}.\\n"
    "JSON COMPLIANCE:\\n"
    "- Output the JSON object and NOTHING else: no prose, no explanations, no "
    "apologies, no markdown fences, no comments inside the JSON.\\n"
    "- An empty string is never a valid answer — if you find nothing to "
    "extract, output the empty JSON object "
    "({\\\"entities\\\":[],\\\"relationships\\\":[]}).\\n"
)

# JSON Schema for Groq / OpenAI-compatible Structured Outputs
# (GRAPHRAG_CHAT_RESPONSE_FORMAT=json_schema). Server-side validation makes
# json_validate_failed impossible on models that support structured outputs.
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(ALLOWED_ENTITY_TYPES)},
                },
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string"},
                    "source_type": {"type": "string", "enum": sorted(ALLOWED_ENTITY_TYPES)},
                    "relation": {"type": "string", "enum": sorted(ALLOWED_RELATION_TYPES)},
                    "target_name": {"type": "string"},
                    "target_type": {"type": "string", "enum": sorted(ALLOWED_ENTITY_TYPES)},
                },
                "required": ["source_name", "source_type", "relation", "target_name", "target_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}


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

        # Raw (un-normalized) question + answer — the SAME string the
        # verification quality checks compare entity names against. The
        # verbatim post-filter below guarantees every emitted entity is a raw
        # (case-insensitive) substring of this text, so no emitted entity can
        # be flagged as "not found verbatim".
        verbatim_text = f"{doc.question_text} {doc.answer_text}"

        # Deterministic metadata entities (grounded by definition) are added
        # BEFORE relationship grounding so LLM edges can reference them.
        metadata_entities: list[Entity] = []
        if doc.ministry:
            metadata_entities.append(Entity(name=doc.ministry, type=EntityType.MINISTRY))

        payload = self._call_llm(text, context=doc.question_id)
        entities, relationships, rejections = self._parse_and_ground(
            payload, text, extra_entities=metadata_entities,
            verbatim_text=verbatim_text,
        )

        # Merge metadata entities (dedupe against LLM entities).
        seen = {(e.type.value, e.name) for e in entities}
        for me in metadata_entities:
            if (me.type.value, me.name) not in seen:
                entities.append(me)
                seen.add((me.type.value, me.name))

        # Apply the verbatim post-filter to the FINAL list too (covers the
        # deterministic ministry entity merged above) so NO emitted entity can
        # be flagged "not found verbatim" by the verification gate.
        final_kept: list[Entity] = []
        for e in entities:
            if re.search(re.escape(e.name), verbatim_text, re.IGNORECASE):
                final_kept.append(e)
            else:
                rejections["entities"].append(
                    {"name": e.name, "type": e.type.value, "reason": "not_verbatim"}
                )
        entities = final_kept

        # Drop any relationship whose endpoint was dropped by the filter.
        valid_endpoints = {(e.type.value, e.name) for e in entities}
        relationships = [
            r for r in relationships
            if (r.source_type.value, r.source_name) in valid_endpoints
            and (r.target_type.value, r.target_name) in valid_endpoints
        ]

        return entities, relationships, rejections

    # ── internals ───────────────────────────────────────────────────────

    def _call_llm(self, text: str, context: Optional[str] = None) -> dict:
        # The system message carries the extraction rules + JSON contract; the
        # user message carries only the document. Groq/OpenAI examples use
        # this split and it keeps JSON-mode instruction handling simple.
        system_prompt = SYSTEM_PROMPT
        user_prompt = (
            "Extract entities and relationships from the DOCUMENT below.\n"
            "Respond in JSON only.\n\n"
            f"DOCUMENT:\n{text}"
        )
        # Structured Outputs (server-side schema validation) when configured;
        # otherwise legacy JSON mode.
        json_schema = (
            EXTRACTION_JSON_SCHEMA
            if self.config.chat_response_format == "json_schema"
            else None
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.config.extract_max_attempts):
            self.stats["calls"] += 1
            try:
                result = self.provider.generate(
                    user_prompt,
                    temperature=0.0,
                    max_tokens=self.config.extract_max_tokens,
                    timeout_seconds=self.config.llm_timeout_seconds,
                    context=context,
                    system=system_prompt,
                    json_schema=json_schema,
                )
                self.stats["provider"] = result.provider
                self.stats["model"] = result.model
                self.stats["key"] = result.key_label
                parsed = self._parse_json(result.text)
                if parsed is None:
                    self.stats["json_parse_failures"] += 1
                    if self.config.llm_debug:
                        logger.warning(
                            "RAW LLM OUTPUT (context=%s, provider=%s, model=%s) — "
                            "failed local JSON parse: %r",
                            context, result.provider, result.model, result.text[:4000],
                        )
                    raise ValueError("LLM returned unparsable JSON")
                return parsed
            except LLMBackendExhaustedError:
                # All providers/keys exhausted — propagate so the pipeline can
                # stop cleanly and save the checkpoint for later resume.
                raise
            except DocumentExtractionError:
                # A per-document content failure (e.g. HTTP 400
                # json_validate_failed). NOT a backend failure: no retry, no
                # key/model failover — propagate so the pipeline marks ONLY
                # this document as failed and continues.
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
        verbatim_text: Optional[str] = None,
    ) -> tuple[list[Entity], list[Relationship], dict]:
        norm_doc = _normalize(doc_text)
        rejections: dict = {"entities": [], "relationships": []}

        def _is_verbatim(name: str) -> bool:
            """True if ``name`` is a raw (case-insensitive) substring of the
            document text — the same check the verification quality gate uses.
            When no verbatim_text is supplied (legacy callers), fall back to the
            normalized doc so behaviour is unchanged."""
            haystack = verbatim_text if verbatim_text is not None else norm_doc
            return re.search(re.escape(name), haystack, re.IGNORECASE) is not None

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

        # ── verbatim post-filter (conservative) ─────────────────────────
        # Any entity that would be flagged as "not found verbatim" by the
        # verification quality gate is dropped here instead — never emitted.
        verbatim_kept: list[Entity] = []
        for e in entities:
            if _is_verbatim(e.name):
                verbatim_kept.append(e)
            else:
                rejections["entities"].append(
                    {"name": e.name, "type": e.type.value, "reason": "not_verbatim"}
                )
        entities = verbatim_kept

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
            evidence = self._find_evidence(doc_text, s_name, t_name)
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
    def _find_evidence(
        orig_doc: str,
        a: str,
        b: str,
        window: int = 420,
        context: int = 60,
    ) -> Optional[str]:
        """
        Find a verbatim evidence snippet from the ORIGINAL document text that
        contains BOTH endpoint names (case-insensitive search, original casing
        preserved).

        Conservative behaviour: if the two endpoints are farther apart than
        ``window`` characters there is no single coherent snippet containing
        both — return None so the relationship is rejected rather than emitted
        with an evidence snippet that truncates one endpoint.
        """
        def locate(name: str) -> tuple[int, int]:
            m = re.search(re.escape(name), orig_doc, re.IGNORECASE)
            return (m.start(), m.end()) if m else (-1, -1)

        sa, ea = locate(a)
        sb, eb = locate(b)
        if sa < 0 or sb < 0:
            return None
        start = min(sa, sb)
        end = max(ea, eb)
        if end - start > window:
            # Endpoints too far apart for one coherent evidence snippet.
            return None
        snippet = orig_doc[max(0, start - context): min(len(orig_doc), end + context)]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > window + 2 * context:
            snippet = snippet[: window + 2 * context].rstrip() + "..."
        return snippet or None
