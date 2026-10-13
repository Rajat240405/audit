"""Retrieval score attribution and matched-unit provenance (display only).

Why this module exists
----------------------
``HybridRAGPipeline.retrieve`` collapses candidates to one per parent document.
The collapsed list keeps only the representative unit. The original dense,
BM25 and RRF candidates were then looked up by the exact PARENT id. When the
retrieval matched only a child unit such as ``{parent}_L15``, those lookups
returned ``None`` even though the child's scores existed.

This module keeps the candidate lists keyed by the actual unit id
(before collapse) and derives the parent's summary scores from them.

Rules
-----
* A parent's own score is used when the parent row itself was returned by that
  system. The summary is then labelled ``"parent"``.
* Otherwise the summary is the best score among the parent's matched units in
  that system, labelled ``"best_chunk"``. It is never presented as a score
  computed for the parent.
* ``None`` means the system did not return any matched unit for the parent.
* Scores are never compared or combined across systems, and ranking is not
  changed by anything in this module.

The module has no dependency on faiss, torch or the pipeline, so it can be
tested directly. It is display provenance only. The evidence sent to the
generator is not built here.
"""

from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Optional

from src.retrieval.result import MatchedChunk

#: Display cap on matched units per parent. ``matched_chunk_total`` keeps the real count.
MAX_MATCHED_CHUNKS = 8

SCORE_FROM_PARENT = "parent"
SCORE_FROM_BEST_CHUNK = "best_chunk"


class UnitInfo(NamedTuple):
    """Stored text and identity of one chunk-level retrieval unit."""

    text: str
    unit_type: str
    page: Optional[int]


def first_scores(pairs: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Map unit id to its score, keeping the FIRST occurrence (rank order)."""
    out: dict[str, float] = {}
    for unit_id, score in pairs:
        out.setdefault(unit_id, float(score))
    return out


def summarize_channel(
    parent_id: str,
    unit_ids: Iterable[str],
    channel: dict[str, float],
) -> tuple[Optional[float], Optional[str]]:
    """Summary score for one system: ``(score, basis)``.

    ``channel`` maps unit id to that system's score. ``unit_ids`` are the
    matched units of this parent.
    """
    if parent_id in channel:
        return channel[parent_id], SCORE_FROM_PARENT
    best: Optional[float] = None
    for unit_id in unit_ids:
        score = channel.get(unit_id)
        if score is not None and (best is None or score > best):
            best = score
    if best is None:
        return None, None
    return best, SCORE_FROM_BEST_CHUNK


def build_matched_chunks(
    parent_id: str,
    pool_units: list[tuple[str, float]],
    *,
    dense: dict[str, float],
    bm25: dict[str, float],
    rrf: dict[str, float],
    representative: Optional[str],
    rerank_score: Optional[float],
    lookup: Callable[[str], Optional[UnitInfo]],
) -> tuple[list[MatchedChunk], int]:
    """Matched units of one parent, each with its OWN scores.

    ``pool_units`` is the parent's candidates in fused (RRF) order, before
    collapse. ``representative`` is the unit the reranker scored, and only that
    unit carries ``rerank_score``.

    Order: units with stored text first, the reranked unit first among them,
    then RRF order. The parent row (no text) comes last. Returns
    ``(capped_list, total_count)``.
    """
    entries: list[tuple[tuple, MatchedChunk]] = []
    for pool_index, (unit_id, _pool_score) in enumerate(pool_units):
        if unit_id == parent_id:
            unit_type, text, page = "parent", None, None
        else:
            info = lookup(unit_id)
            if info is None:
                continue  # not a known unit: cannot be shown, never invented
            unit_type, text, page = info.unit_type, info.text, info.page
        reranked = rerank_score is not None and unit_id == representative
        entry = MatchedChunk(
            chunk_id=unit_id,
            parent_doc_id=parent_id,
            unit_type=unit_type,
            text=text,
            page=page,
            dense_score=dense.get(unit_id),
            bm25_score=bm25.get(unit_id),
            rrf_score=rrf.get(unit_id),
            rerank_score=rerank_score if reranked else None,
            reranked=reranked,
        )
        entries.append(((text is None, not reranked, pool_index), entry))
    entries.sort(key=lambda item: item[0])
    total = len(entries)
    return [entry for _, entry in entries[:MAX_MATCHED_CHUNKS]], total


def attribute_parent(
    parent_id: str,
    pool_units: list[tuple[str, float]],
    *,
    dense: dict[str, float],
    bm25: dict[str, float],
    rrf: dict[str, float],
    representative: Optional[str],
    rerank_score: Optional[float],
    lookup: Callable[[str], Optional[UnitInfo]],
) -> dict:
    """Everything a ``RetrievedResult`` needs for score attribution.

    Returns a dict whose keys match the new ``RetrievedResult`` fields:
    ``dense_score``, ``bm25_score``, ``rrf_score``, ``matched_chunks``,
    ``matched_chunk_total`` and ``score_sources``. ``rerank_score`` is NOT
    included, because the caller already sets it.
    """
    unit_ids = [unit_id for unit_id, _ in pool_units]
    dense_score, dense_basis = summarize_channel(parent_id, unit_ids, dense)
    bm25_score, bm25_basis = summarize_channel(parent_id, unit_ids, bm25)
    rrf_score, rrf_basis = summarize_channel(parent_id, unit_ids, rrf)
    matched, total = build_matched_chunks(
        parent_id,
        pool_units,
        dense=dense,
        bm25=bm25,
        rrf=rrf,
        representative=representative,
        rerank_score=rerank_score,
        lookup=lookup,
    )
    return {
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "rrf_score": rrf_score,
        "matched_chunks": matched,
        "matched_chunk_total": total,
        "score_sources": {"dense": dense_basis, "bm25": bm25_basis, "rrf": rrf_basis},
    }
