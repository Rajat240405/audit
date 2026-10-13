"""
Retrieval result model — shared by all retrieval systems.

Design Decision
---------------
We return the ORIGINAL structured record (separate question + answer),
not just the concatenated text. This is critical for:
1. The LLM generation prompt — structured Q→A format is clearer
2. Transparency — users can see exactly what was retrieved
3. Attribution — structured records are easier to cite

The concatenated text is only used for indexing, never returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchedChunk:
    """One retrieval unit that matched inside a parent document.

    A unit is a long-document chunk (``{parent}_L{n}``), a V2 table/figure
    child, or a Q/A chunk. ``text`` is the stored chunk text, never a
    reconstruction. It is ``None`` when the matched unit is the parent row
    itself, because the parent's text is the assembled evidence in
    ``RetrievedResult.answer``.

    Scores are the unit's own scores from each system that returned it.
    ``None`` means that system did not return this unit. ``rerank_score`` is
    set only on the single candidate the cross-encoder actually scored.
    These scores are not on a common scale and must not be combined.
    """

    chunk_id: str
    parent_doc_id: str
    #: "parent" | "annexure" | "question" | "answer" | "v2_table" | "v2_figure"
    unit_type: str
    text: str | None = None
    page: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    #: True for the one candidate that the reranker scored for this parent.
    reranked: bool = False


@dataclass
class RetrievedResult:
    """
    A single retrieved Q&A record with retrieval metadata.

    This represents a record retrieved from either Hybrid RAG or GraphRAG.
    The original structured fields are always returned.
    """

    doc_id: str = field(
        metadata={"description": "Unique document identifier"}
    )
    question: str = field(
        metadata={"description": "Original question text — NOT the concatenated text"}
    )
    answer: str = field(
        metadata={"description": "Original answer text — NOT the concatenated text"}
    )
    score: float = field(
        metadata={"description": "Retrieval score (system-dependent meaning)"}
    )
    retrieval_method: str = field(
        metadata={"description": "How this record was retrieved: dense, bm25, rrf_fusion, graph_traversal"}
    )
    metadata: dict = field(
        default_factory=dict,
        metadata={"description": "Additional record metadata (ministry, date, subject, etc.)"}
    )

    # Optional: scores from each sub-system (for analysis)
    dense_score: float | None = field(
        default=None,
        metadata={"description": "Score from dense vector retrieval (if applicable)"}
    )
    bm25_score: float | None = field(
        default=None,
        metadata={"description": "Score from BM25 retrieval (if applicable)"}
    )
    rrf_score: float | None = field(
        default=None,
        metadata={"description": "RRF fusion score (if applicable)"}
    )
    rerank_score: float | None = field(
        default=None,
        metadata={"description": "Cross-encoder rerank score (if applicable)"}
    )

    # Matched-unit provenance (display only; never sent to the generator).
    # ``answer`` above is still the assembled parent evidence the model reads.
    #: The units that matched inside this parent, representative first, capped.
    matched_chunks: list[MatchedChunk] = field(
        default_factory=list,
        metadata={"description": "Matched units inside this parent (display provenance)"}
    )
    #: Count of ALL matched units for this parent. May exceed len(matched_chunks).
    matched_chunk_total: int = field(
        default=0,
        metadata={"description": "Number of matched units before the display cap"}
    )
    #: Where each summary score came from: "parent" (the parent row itself)
    #: or "best_chunk" (the highest score among its matched units). Keys are
    #: "dense", "bm25" and "rrf". A value is None when no score exists.
    score_sources: dict[str, str | None] = field(
        default_factory=dict,
        metadata={"description": "Basis of dense/bm25/rrf summary scores"}
    )

    def __repr__(self) -> str:
        q_preview = self.question[:60] + "..." if len(self.question) > 60 else self.question
        return (
            f"RetrievedResult(doc_id={self.doc_id!r}, "
            f"question={q_preview!r}, "
            f"score={self.score:.4f}, "
            f"method={self.retrieval_method!r})"
        )
