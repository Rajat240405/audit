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

    def __repr__(self) -> str:
        q_preview = self.question[:60] + "..." if len(self.question) > 60 else self.question
        return (
            f"RetrievedResult(doc_id={self.doc_id!r}, "
            f"question={q_preview!r}, "
            f"score={self.score:.4f}, "
            f"method={self.retrieval_method!r})"
        )
