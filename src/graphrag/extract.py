"""Grounded semantic extraction (Phase 2 / WS2-C).

The ONLY place LLM output reaches the graph — and it reaches it through the
project's EXISTING generation architecture: this module receives an
``LLMClient`` (built via src/generation — registry + vLLM discovery + policy;
see activation.py) and calls ``client.generate(prompt, system)``. No provider
imports, no model names, no provider configuration lives here (spec §9): if
the serving model changes, extraction follows it with zero code changes.

Validation (spec §13) — the builder NEVER blindly accepts LLM output:

    LLM JSON
      → structural validation   (allowed labels/relations, name sanity)
      → grounding               (names + evidence must appear VERBATIM in the
                                 document, casefold/whitespace-normalized —
                                 the same verbatim-trust philosophy as the
                                 server's LLM-judge gate)
      → normalization           (Phase 1 vocabulary: known surfaces →
                                 canonical "c:" keys; unknowns → stable "u:"
                                 keys, raw preserved — open world)
      → upsert                  (with per-document SUPPORTS provenance,
                                 origin="llm")

Ungrounded entities/relationships are REJECTED (and counted) — uncertainty is
preserved, never invented.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from src.models.qa_record import QARecord
from src.vocabulary import Vocabulary

from src.graphrag.models import EntityRef, GraphFact, Support, unknown_key
from src.graphrag.schema import SEMANTIC_ENDPOINTS, SEMANTIC_RELS

__all__ = ["SemanticExtractor", "ExtractionResult", "ExtractionError"]


class ExtractionError(Exception):
    """Structural/parse failure of an LLM extraction payload."""


_SYSTEM_PROMPT = """You extract structured facts from ONE parliamentary Q&A or official document about Indian ocean/atmospheric science and governance.

Return STRICT JSON only (no markdown, no commentary) with exactly this shape:
{
  "entities": [
    {"name": "<exact surface form as written in the document>",
     "type": "Organization" | "Programme" | "Facility" | "Place",
     "evidence": "<short verbatim snippet from the document containing the name>"}
  ],
  "relationships": [
    {"source": "<entity name>", "source_type": "<type>",
     "relation": "OPERATED_BY" | "LOCATED_IN" | "PART_OF" | "FUNDED_BY" | "COLLABORATES_WITH",
     "target": "<entity name>", "target_type": "<type>",
     "evidence": "<short verbatim snippet from the document supporting the relation>"}
  ]
}

Rules:
- "name" MUST be copied verbatim from the document (no paraphrasing, no translation).
- "evidence" MUST be a verbatim contiguous snippet from the document.
- Organizations: ministries, departments, institutes, agencies, committees.
- Programmes: schemes, missions, projects, initiatives, funds.
- Facilities: centres, observatories, stations, labs, radars, buoys, satellites, ships, networks.
- Place: countries, states, districts, cities, rivers, seas, oceans, coasts, regions.
- Only relationships explicitly stated or directly implied by the text.
- Do NOT extract persons (members of parliament, scientists by name).
- If nothing qualifies, return {"entities": [], "relationships": []}."""

_USER_TEMPLATE = """Document question (parliamentary context, if any):
{question}

Document text:
{text}

Extract the JSON now."""

_WS_RE = re.compile(r"\s+")
_VALID_REL_BY_ENDPOINTS: dict[tuple[str, str], set[str]] = {}
for _rel, _src, _dst in SEMANTIC_RELS:
    if _rel == "MENTIONS":
        continue
    _VALID_REL_BY_ENDPOINTS.setdefault((_src, _dst), set()).add(_rel)


def _fold(s: str) -> str:
    return _WS_RE.sub(" ", str(s or "").casefold().strip())


@dataclass
class ExtractionResult:
    """Validated output of one extraction (ready for upsert)."""

    doc_key: str
    _entity_evidence: list = field(default_factory=list)
    entities: list[EntityRef] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    llm_raw: Optional[dict] = None
    llm_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_key": self.doc_key,
            "entities": len(self.entities),
            "facts": len(self.facts),
            "rejected": len(self.rejected),
            "rejection_reasons": sorted({r["reason"] for r in self.rejected}),
        }


class SemanticExtractor:
    """Document → grounded semantic contribution, via the shared LLM client."""

    def __init__(self, llm_client, voc: Vocabulary, *, max_chars: int = 12000) -> None:
        self._client = llm_client
        self._voc = voc
        self._max_chars = max_chars
        # concept surface forms (folded) → canonical, plus canonical → label
        # (concepts are raw dicts: canonical / label / surfaces)
        self._concept_fold: dict[str, str] = {}
        self._concept_label: dict[str, str] = {}
        for c in voc.concepts:
            self._concept_label[c["canonical"]] = c.get("label") or c["canonical"]
            forms = [c.get("canonical")] + list(c.get("surfaces") or [])
            for s in forms:
                if s:
                    self._concept_fold[_fold(s)] = c["canonical"]

    # ── entrypoint ────────────────────────────────────────────────────────

    def extract(self, rec: QARecord, *, now: str) -> ExtractionResult:
        doc_key = rec.question_id
        text = self._document_text(rec)
        user = _USER_TEMPLATE.format(
            question=(rec.question_text or "")[:2000],
            text=text[: self._max_chars],
        )
        resp = self._client.generate(user, system=_SYSTEM_PROMPT)
        payload = self._parse(resp.text)
        res = ExtractionResult(doc_key=doc_key, llm_raw=payload)
        res.llm_usage = {
            "model": getattr(resp, "model", None),
            "total_tokens": getattr(resp, "total_tokens", None),
            "latency_ms": getattr(resp, "latency_ms", None),
        }
        folded_text = _fold(text)
        self._extract_entities(payload, folded_text, res)
        self._extract_relationships(payload, folded_text, res, now)
        return res

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _document_text(rec: QARecord) -> str:
        parts = [p for p in (rec.answer_text, rec.question_text) if p]
        return "\n".join(parts)

    @staticmethod
    def _parse(raw: str) -> dict:
        raw = (raw or "").strip()
        # tolerate a single markdown fence even though the prompt forbids it
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"LLM returned non-JSON payload: {e}") from e
        if not isinstance(data, dict):
            raise ExtractionError("LLM payload is not a JSON object")
        for k in ("entities", "relationships"):
            if k not in data:
                data[k] = []
            if not isinstance(data[k], list):
                raise ExtractionError(f"LLM payload field {k!r} is not a list")
        return data

    def _resolve_entity(
        self, name: str, etype: str, folded_text: str, res: ExtractionResult
    ) -> Optional[EntityRef]:
        """Structural + grounding + normalization for one entity. Returns the
        resolved EntityRef or None (with a rejection recorded)."""
        if etype not in SEMANTIC_ENDPOINTS:
            res.rejected.append({"kind": "entity", "name": name, "type": etype,
                                 "reason": "disallowed_entity_type"})
            return None
        name = str(name or "").strip()
        if not re.search(r"[A-Za-z0-9]", name) or len(name) < 2 or len(name) > 200:
            res.rejected.append({"kind": "entity", "name": name, "type": etype,
                                 "reason": "invalid_name"})
            return None
        if _fold(name) not in folded_text:
            res.rejected.append({"kind": "entity", "name": name, "type": etype,
                                 "reason": "not_grounded_in_document"})
            return None
        canon = self._concept_fold.get(_fold(name))
        if canon is not None and etype in ("Organization", "Programme"):
            # display name = the controlled vocabulary label; the observed
            # surface stays in raw (provenance)
            ref = EntityRef(label=etype, key=f"c:{canon}",
                            name=self._concept_label.get(canon, canon),
                            resolution="canonical", raw=name,
                            props={"matched_concept": canon})
        else:
            ref = EntityRef(label=etype, key=unknown_key(name), name=name,
                            resolution="unresolved", raw=name)
        return ref

    def _extract_entities(self, payload: dict, folded_text: str,
                          res: ExtractionResult) -> None:
        res._entity_evidence = []
        seen: dict[str, EntityRef] = {}
        for item in payload.get("entities", []):
            if not isinstance(item, dict):
                res.rejected.append({"kind": "entity", "name": None,
                                     "reason": "malformed_item"})
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip()
            evidence = str(item.get("evidence", "") or "").strip()
            if etype not in SEMANTIC_ENDPOINTS:
                res.rejected.append({"kind": "entity", "name": name,
                                     "type": etype,
                                     "reason": "disallowed_entity_type"})
                continue
            ref = self._resolve_entity(name, etype, folded_text, res)
            if ref is None:
                continue
            if _fold(evidence) and _fold(evidence) not in folded_text:
                res.rejected.append({"kind": "entity", "name": name,
                                     "type": etype,
                                     "reason": "evidence_not_grounded"})
                continue
            if _fold(name) in seen:
                continue  # dedupe identical surfaces
            seen[_fold(name)] = ref
            res.entities.append(ref)
            res._entity_evidence.append(evidence)

    def _extract_relationships(self, payload: dict, folded_text: str,
                               res: ExtractionResult, now: str) -> None:
        # endpoint resolution by the OBSERVED surface (what the LLM spelled),
        # not the display name (canonical concepts show the vocab label)
        by_name = {_fold(e.raw or e.name): e for e in res.entities}
        # also allow endpoint resolution against concept matches the model
        # spelled out differently — but ONLY if grounded in the document:
        for item in payload.get("relationships", []):
            if not isinstance(item, dict):
                res.rejected.append({"kind": "relationship",
                                     "reason": "malformed_item"})
                continue
            src_name = str(item.get("source", "")).strip()
            dst_name = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "")).strip()
            src_type = str(item.get("source_type", "")).strip()
            dst_type = str(item.get("target_type", "")).strip()
            evidence = str(item.get("evidence", "") or "").strip()
            if relation not in _VALID_REL_BY_ENDPOINTS.get((src_type, dst_type), set()):
                res.rejected.append(
                    {"kind": "relationship", "source": src_name,
                     "target": dst_name, "relation": relation,
                     "reason": "disallowed_relationship"})
                continue
            if _fold(evidence) not in folded_text:
                res.rejected.append(
                    {"kind": "relationship", "source": src_name,
                     "target": dst_name, "relation": relation,
                     "reason": "evidence_not_grounded"})
                continue
            src_ref = by_name.get(_fold(src_name))
            dst_ref = by_name.get(_fold(dst_name))
            if src_ref is None or dst_ref is None:
                res.rejected.append(
                    {"kind": "relationship", "source": src_name,
                     "target": dst_name, "relation": relation,
                     "reason": "endpoint_not_extracted"})
                continue
            if src_ref.label != src_type or dst_ref.label != dst_type:
                res.rejected.append(
                    {"kind": "relationship", "source": src_name,
                     "target": dst_name, "relation": relation,
                     "reason": "endpoint_type_mismatch"})
                continue
            fact = GraphFact(relation, src_ref, dst_ref, origin="llm")
            res.facts.append(fact)
            res.supports.append(Support(doc_key=res.doc_key, fact_key=fact.fact_key,
                                        evidence=evidence, origin="llm",
                                        extracted_at=now))
        # MENTIONS facts — one per grounded entity, carrying the entity's
        # own evidence snippet as the provenance text
        doc = EntityRef("Document", res.doc_key, res.doc_key)
        for e, ev in zip(res.entities, res._entity_evidence):
            fact = GraphFact("MENTIONS", doc, e, origin="llm")
            res.facts.append(fact)
            res.supports.append(Support(doc_key=res.doc_key, fact_key=fact.fact_key,
                                        evidence=ev, origin="llm",
                                        extracted_at=now))
