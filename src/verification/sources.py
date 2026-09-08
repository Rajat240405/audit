"""Evidence adapters (Phase 3 / WS3).

Converters ONLY — they normalize retrieval-mode output into the shared
verification contract. All verification logic lives in authority.py.

  * hybrid:  source dicts (the /api/verify shape) → EvidenceSet(mode=hybrid)
  * graph:   RetrievedResult list + resolved provenance → EvidenceSet(mode=graph)
             with the Document→SUPPORTS→Fact chains attached.
"""
from __future__ import annotations

from typing import Optional

from src.verification.contract import (
    MODE_GRAPH,
    MODE_HYBRID,
    EvidenceItem,
    EvidenceSet,
    GraphProvenance,
)

__all__ = ["hybrid_sources", "graph_results"]


def hybrid_sources(sources: list[dict]) -> EvidenceSet:
    """The legacy {doc_id, question, answer} dicts → normalized hybrid set."""
    items = [
        EvidenceItem(
            doc_id=str(s.get("doc_id", "")),
            question=str(s.get("question") or ""),
            answer=str(s.get("answer") or ""),
            mode=MODE_HYBRID,
        )
        for s in sources
        if s.get("doc_id")
    ]
    return EvidenceSet(items=items, mode=MODE_HYBRID)


def graph_results(results: list, capability=None,
                  limit_facts_per_result: int = 10) -> EvidenceSet:
    """Graph-mode RetrievedResult list → normalized graph evidence set.

    ``capability`` (a GraphCapability) resolves each result's
    metadata["graph"]["fact_keys"] into full provenance (fact + supports +
    documents) via the store; when omitted, only the fact keys are carried
    (the authority then flags unresolvable provenance rather than guessing).
    """
    items: list[EvidenceItem] = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        gmeta = meta.get("graph") or {}
        fact_keys = list(gmeta.get("fact_keys") or [])
        facts: list[dict] = []
        if capability is not None:
            for fk in fact_keys[:limit_facts_per_result]:
                ev = capability.evidence_for_fact(fk)
                if ev is None:
                    continue
                facts.append({
                    "fact_key": ev.fact_key,
                    "rel": ev.rel,
                    "origin": ev.origin,
                    "doc_count": ev.doc_count,
                    "source_field": (ev.supports[0] or {}).get("origin")
                    if False else _source_field_for(ev),
                    "supports": [
                        {
                            "fact_key": s.get("fact_key") or ev.fact_key,
                            "evidence": s.get("evidence"),
                            "origin": s.get("origin"),
                            "doc_key": s.get("doc_key"),
                            "extracted_at": s.get("extracted_at"),
                        }
                        for s in ev.supports
                    ],
                })
        gp = GraphProvenance(
            fact_keys=fact_keys,
            anchors=[a for a in (gmeta.get("anchors") or [])],
            facts=facts,
        )
        items.append(EvidenceItem(
            doc_id=r.doc_id,
            question=r.question or "",
            answer=r.answer or "",
            mode=MODE_GRAPH,
            graph=gp,
            stored_date=meta.get("date"),
        ))
    return EvidenceSet(items=items, mode=MODE_GRAPH)


def _source_field_for(ev) -> Optional[str]:
    """Best-effort source_field for a resolved fact (deterministic facts
    carry it on the relationship; absent for LLM facts)."""
    return getattr(ev, "source_field", None)
