"""Merge Hybrid RAG and GraphRAG evidence into ONE result list.

Both capabilities already return ``RetrievedResult`` (its docstring states it
represents a record "retrieved from either Hybrid RAG or GraphRAG"), so the
merge is a de-duplicating union — no new evidence type, no second
answer-generation path, no re-scoring of either system's results.

What must survive the merge (and is asserted by tests):

  * document identity and text (``doc_id`` / ``question`` / ``answer``);
  * hybrid sub-scores (``dense_score`` / ``bm25_score`` / ``rrf_score`` /
    ``rerank_score``) — used by the evidence panel and diagnostics;
  * graph provenance (``metadata["graph"]``: anchors, traversal path,
    fact_keys) — used by the Graph tab;
  * every other metadata key either side supplied.

When the SAME document is returned by both systems its two records are fused
so the answer LLM sees one entry carrying document evidence AND relationship
evidence, rather than a duplicate.
"""
from __future__ import annotations

from typing import Iterable, Optional

from src.retrieval.result import RetrievedResult

__all__ = ["merge_evidence", "MERGED_METHOD"]

#: retrieval_method marker for a record supported by BOTH systems.
MERGED_METHOD = "hybrid+graph"


def _merge_metadata(primary: dict | None, secondary: dict | None) -> dict:
    """Union of both metadata dicts; ``primary`` wins on scalar conflicts.

    ``graph`` is never dropped: whichever side carries traversal provenance
    keeps it, because the Graph tab renders exclusively from that key.
    """
    out = dict(secondary or {})
    for key, value in (primary or {}).items():
        if value is None and key in out:
            continue  # do not blank an existing value with None
        out[key] = value
    graph = (primary or {}).get("graph") or (secondary or {}).get("graph")
    if graph is not None:
        out["graph"] = graph
    return out


def _fuse(hybrid: RetrievedResult, graph: RetrievedResult) -> RetrievedResult:
    """Combine the two records for one document into a single entry."""
    return RetrievedResult(
        doc_id=hybrid.doc_id,
        # prefer whichever side actually carries text
        question=hybrid.question or graph.question,
        answer=hybrid.answer or graph.answer,
        score=max(float(hybrid.score or 0.0), float(graph.score or 0.0)),
        retrieval_method=MERGED_METHOD,
        metadata=_merge_metadata(hybrid.metadata, graph.metadata),
        dense_score=hybrid.dense_score if hybrid.dense_score is not None else graph.dense_score,
        bm25_score=hybrid.bm25_score if hybrid.bm25_score is not None else graph.bm25_score,
        rrf_score=hybrid.rrf_score if hybrid.rrf_score is not None else graph.rrf_score,
        rerank_score=(
            hybrid.rerank_score if hybrid.rerank_score is not None else graph.rerank_score
        ),
    )


def merge_evidence(
    hybrid_results: Optional[Iterable[RetrievedResult]],
    graph_results: Optional[Iterable[RetrievedResult]],
    *,
    limit: Optional[int] = None,
) -> list[RetrievedResult]:
    """Return one de-duplicated evidence list for the answer generator.

    Ordering is deterministic and interleaved by rank: hybrid[0], graph[0],
    hybrid[1], graph[1], … so neither system's evidence is starved when the
    downstream token budget admits only the first few documents. Records for
    the same ``doc_id`` are fused and keep the earliest position.

    ``limit`` caps the merged list; ``None`` means no cap (the generator's
    evidence budget still applies downstream).
    """
    hybrid = [r for r in (hybrid_results or []) if r is not None]
    graph = [r for r in (graph_results or []) if r is not None]

    by_doc: dict[str, RetrievedResult] = {}
    order: list[str] = []

    def _add(rec: RetrievedResult) -> None:
        key = rec.doc_id
        existing = by_doc.get(key)
        if existing is None:
            by_doc[key] = rec
            order.append(key)
            return
        # same document from the other system -> fuse, keeping hybrid as the
        # primary side so its sub-scores and text take precedence
        if existing.retrieval_method == "graph_traversal":
            by_doc[key] = _fuse(rec, existing)
        else:
            by_doc[key] = _fuse(existing, rec)

    for i in range(max(len(hybrid), len(graph))):
        if i < len(hybrid):
            _add(hybrid[i])
        if i < len(graph):
            _add(graph[i])

    merged = [by_doc[k] for k in order]
    if limit is not None:
        merged = merged[: max(0, int(limit))]
    return merged
