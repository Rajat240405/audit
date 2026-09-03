"""
Cross-encoder reranking for improved retrieval precision.

Design Decisions
----------------
1. Cross-encoder reranking is applied AFTER RRF fusion as a second stage.
   It re-evaluates the top-K candidates (e.g., 20) against the query using
   a more powerful (but slower) cross-encoder model.

2. Why cross-encoder instead of bi-encoder (already used)?
   - Bi-encoders (sentence-transformers): Encode query and document SEPARATELY.
     Fast but can't capture fine-grained query-document interactions.
   - Cross-encoders: Encode query+document TOGETHER as a pair.
     Captures word-by-word interactions. Much more accurate for reranking
     but too slow for first-stage retrieval over 3,500 docs.

3. Model choice: `cross-encoder/ms-marco-MiniLM-L-12-v2`
   - 12-layer cross-encoder trained on MS MARCO passage ranking
   - ~100-200ms per query on CPU — acceptable for our top-20 reranking
   - Much better quality than a second bi-encoder pass

4. We rescore all candidates from RRF (not just the top-K we return),
   then return the top-K from the reranked list.

5. Cross-encoder scores are NOT directly comparable to RRF or BM25 scores.
   We only use them for reordering, not for fusion with other systems.

6. The cross-encoder takes (query, document) as input and outputs a
   relevance score. We concatenate the question and answer text as the document.

References
----------
Nogueira et al. (2020). "Document Ranking with a Pretrained Sequence-to-Sequence Model."
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import CrossEncoder
import os
from pathlib import Path

from src.retrieval.result import RetrievedResult


def resolve_rerank_model() -> str:
    """Pick the cross-encoder reranker to use.

    Priority:
      1. GRAPHRAG_RERANK_MODEL env var
      2. A local ``models/bge-reranker-v2-m3`` folder (offline HPC — this is
         the multilingual BGE reranker, better than MS-MARCO for our corpus)
      3. Default ``cross-encoder/ms-marco-MiniLM-L-12-v2``
    """
    env = os.environ.get("GRAPHRAG_RERANK_MODEL")
    if env:
        return env
    from src.utils.app_paths import model_dir

    local = model_dir() / "bge-reranker-v2-m3"
    if local.exists():
        return str(local)
    return "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderReranker:
    """
    Cross-encoder reranker for re-scoring retrieved candidates.

    Takes top-K candidates from the first-stage retrieval (Hybrid RAG fusion)
    and re-ranks them using a cross-encoder model trained on passage ranking.

    Usage
    -----
    ```python
    reranker = CrossEncoderReranker()

    # Build reranker (downloads model on first run)
    candidates = [("18-1", 0.95), ("18-2", 0.87), ...]

    reranked = reranker.rerank("malaria health question", candidates, k=5)
    # Returns reranked list with new cross-encoder scores
    ```
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name for the cross-encoder.
            Default: ms-marco-MiniLM-L-12-v2 (fast, good quality)
            Alternatives: "cross-encoder/ms-marco-MiniLM-L-6-v2" (faster)
                          "cross-encoder/ms-marco-deberta-v3-base" (better, slower)
        device : str, optional
            Explicit device override.

            HPC DEVICE POLICY (query-serving phase): the GPU is reserved
            EXCLUSIVELY for the vLLM/LLM inference server. Everything in the
            retrieval path (BM25, FAISS, RRF, filtering, reranking, evidence
            assembly) is CPU-bound by design. Therefore the default is
            ALWAYS "cpu" — the reranker must never auto-select CUDA merely
            because CUDA is available. Precedence:
              1. this explicit argument
              2. RERANK_DEVICE env var (opt-in "cuda" for local
                 experimentation/tuning only)
              3. "cpu"
        max_length : int
            Maximum token length for the cross-encoder.
            Default 512 handles most Q&A pairs.
        """
        if model_name is None:
            model_name = resolve_rerank_model()
        self.model_name = model_name
        self.device = device or os.environ.get("RERANK_DEVICE", "cpu")
        if self.device != "cpu":
            print(
                f"[reranker] device={self.device!r} — non-CPU reranking. "
                f"On HPC query-serving nodes the GPU is reserved for vLLM; "
                f"unset RERANK_DEVICE to return to the CPU policy."
            )
        self.max_length = max_length
        self._model: CrossEncoder | None = None
        # GLM #5: docs that exceeded max_length in the last rerank (truncated
        # by the cross-encoder silently). Populated by rerank() so callers can
        # surface "this source was only partially considered".
        self.last_truncated_docs: set[str] = set()

    @property
    def model(self) -> CrossEncoder:
        """Lazily load the model on first access."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float] | RetrievedResult],
        k: int = 5,
        doc_texts: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Re-rank candidates using the cross-encoder.

        Parameters
        ----------
        query : str
            The user's original query.
        candidates : list[tuple[str, float] | RetrievedResult]
            Candidates to re-rank.
            Can be simple (doc_id, rrf_score) tuples or RetrievedResult objects.
        k : int
            Number of top results to return after reranking.
        doc_texts : dict[str, str], optional
            Mapping from doc_id → full document text (question + answer).
            If not provided, uses RetrievedResult.question + answer fields.

        Returns
        -------
        list[tuple[str, float]]
            Re-ranked list of (doc_id, cross_encoder_score), sorted descending.
        """
        if not candidates:
            return []

        # Build (doc_id, document_text) pairs
        doc_pairs: list[tuple[str, str]] = []
        for candidate in candidates:
            if isinstance(candidate, RetrievedResult):
                doc_id = candidate.doc_id
                if doc_texts and doc_id in doc_texts:
                    doc_text = doc_texts[doc_id]
                else:
                    doc_text = f"Question: {candidate.question}\n\nAnswer: {candidate.answer}"
            else:
                doc_id, _ = candidate
                doc_text = doc_texts.get(doc_id, "") if doc_texts else ""

            doc_pairs.append((doc_id, doc_text))

        # GLM #5: detect which docs exceed the cross-encoder token budget so we
        # can surface silent truncation. Rough token estimate (chars/4) vs
        # max_length — if a doc is longer, the encoder truncates it.
        self.last_truncated_docs = {
            doc_id for doc_id, doc_text in doc_pairs
            if len(doc_text) // 4 > self.max_length
        }

        # Build query-document pairs for cross-encoder
        query_doc_pairs = [(query, doc_text) for _, doc_text in doc_pairs]

        # Score all pairs
        scores: list[float] = self.model.predict(
            query_doc_pairs,
            show_progress_bar=False,
        )

        # Handle numpy array return type
        if hasattr(scores, "tolist") or isinstance(scores, np.ndarray):
            scores = scores.tolist()

        # Pair with doc_ids and sort by cross-encoder score
        doc_ids = [doc_id for doc_id, _ in doc_pairs]
        reranked = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked[:k]

    def rerank_with_details(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        doc_texts: dict[str, str],
        k: int = 5,
    ) -> list[tuple[str, float, float]]:
        """
        Re-rank with additional scoring details.

        Returns
        -------
        list[tuple[str, float, float]]
            (doc_id, cross_encoder_score, rrf_score) for analysis.
        """
        if not candidates:
            return []

        doc_ids = [doc_id for doc_id, _ in candidates]
        rrf_scores = {doc_id: score for doc_id, score in candidates}

        query_doc_pairs = [
            (query, doc_texts.get(doc_id, ""))
            for doc_id in doc_ids
        ]

        scores: list[float] = self.model.predict(
            query_doc_pairs,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist") or isinstance(scores, np.ndarray):
            scores = scores.tolist()

        results = [
            (doc_id, ce_score, rrf_scores.get(doc_id, 0.0))
            for doc_id, ce_score in zip(doc_ids, scores)
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name!r})"
