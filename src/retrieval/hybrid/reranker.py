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
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.retrieval.result import RetrievedResult

if TYPE_CHECKING:  # annotation only; avoids importing faiss-dependent pipeline
    from src.retrieval.hybrid.pipeline import RetrievalTimings


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


#: Environment variable for the CrossEncoder prediction batch size. Only the
#: grouping of pairs into prediction batches changes; every pair is still scored.
RERANK_BATCH_SIZE_ENV = "RERANK_BATCH_SIZE"
#: Matches the sentence-transformers library default, so unset means no change.
DEFAULT_RERANK_BATCH_SIZE = 32


def parse_rerank_batch_size(raw: object) -> int | None:
    """Return ``raw`` as a positive integer, or ``None`` if it is not one.

    Accepts a positive ``int`` or a string of decimal digits (surrounding
    whitespace allowed). Rejects bools, zero, negatives, floats and any other
    text.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str) and re.fullmatch(r"\s*[0-9]+\s*", raw):
        value = int(raw)
        return value if value > 0 else None
    return None


def _resolve_rerank_batch_size(raw: object = None) -> tuple[int, str]:
    """Return ``(batch_size, source)``; source is argument, env, default or fallback.

    Never raises. A malformed, zero or negative value falls back to the default
    and prints one warning, so a bad optional tuning setting cannot break
    queries. ``raw`` is an explicit argument. When it is ``None`` the value
    comes from ``RERANK_BATCH_SIZE``.
    """
    if raw is not None:
        value = parse_rerank_batch_size(raw)
        if value is not None:
            return value, "argument"
        print(
            f"[reranker][warning] batch_size={str(raw)[:32]!r} is not a positive "
            f"integer; using default {DEFAULT_RERANK_BATCH_SIZE}"
        )
        return DEFAULT_RERANK_BATCH_SIZE, "fallback"
    env_raw = os.environ.get(RERANK_BATCH_SIZE_ENV)
    if env_raw is None or not env_raw.strip():
        return DEFAULT_RERANK_BATCH_SIZE, "default"
    value = parse_rerank_batch_size(env_raw)
    if value is not None:
        return value, "env"
    print(
        f"[reranker][warning] {RERANK_BATCH_SIZE_ENV}={env_raw.strip()[:32]!r} is not "
        f"a positive integer; using default {DEFAULT_RERANK_BATCH_SIZE}"
    )
    return DEFAULT_RERANK_BATCH_SIZE, "fallback"


def resolve_rerank_batch_size(raw: object = None) -> int:
    """Effective CrossEncoder batch size. See ``_resolve_rerank_batch_size``."""
    return _resolve_rerank_batch_size(raw)[0]


#: Synthetic warm-up input. Fixed strings only: never corpus content, never
#: logged. Sized so the batch exercises tokenisation and the forward pass.
_WARMUP_QUERY = "synthetic warm-up query"
_WARMUP_DOC = "synthetic warm-up passage for model initialisation. " * 12


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
        batch_size: int | None = None,
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
        batch_size : int, optional
            Prediction batch size passed to ``CrossEncoder.predict``. Default
            is the ``RERANK_BATCH_SIZE`` env var, else 32. Only the grouping of
            pairs changes: every candidate is still scored.
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
        self.batch_size, batch_source = _resolve_rerank_batch_size(batch_size)
        print(
            f"[reranker] batch_size={self.batch_size} source={batch_source} "
            f"device={self.device}"
        )
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
        timing: dict | None = None,
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
        timing : dict, optional
            Out-parameter for diagnostics. Filled with ``candidate_count``,
            ``model_access_ms`` and ``model_predict_ms`` even when a step
            raises. Never changes the returned value.

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

        info = timing if timing is not None else {}
        info["candidate_count"] = len(candidates)
        info["batch_size"] = self.batch_size

        # Score all pairs. Model access (which lazily loads the weights on the
        # first call) is timed apart from prediction so the two can be told
        # apart in the logs.
        t_access = time.perf_counter()
        try:
            model = self.model
        finally:
            info["model_access_ms"] = (time.perf_counter() - t_access) * 1000
        t_predict = time.perf_counter()
        try:
            scores: list[float] = model.predict(
                query_doc_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        finally:
            info["model_predict_ms"] = (time.perf_counter() - t_predict) * 1000

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

    def warmup_inference(self, n_pairs: int = 4) -> dict:
        """Run one small synthetic prediction to warm the inference path.

        Uses FIXED synthetic text and the already-loaded model. The predictions
        are discarded: nothing is returned to callers, and no request state is
        touched (``last_truncated_docs`` is not modified, and ``rerank()`` is not
        called). Only timings and identifiers are returned, for logging.
        """
        t_access = time.perf_counter()
        model = self.model
        access_ms = (time.perf_counter() - t_access) * 1000

        pairs = [
            (_WARMUP_QUERY, f"{_WARMUP_DOC}[{i}]") for i in range(max(1, int(n_pairs)))
        ]
        t_predict = time.perf_counter()
        model.predict(  # output intentionally discarded
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        inference_ms = (time.perf_counter() - t_predict) * 1000
        return {
            "model": self.model_name,
            "device": self.device,
            "pairs": len(pairs),
            "batch_size": self.batch_size,
            "model_access_ms": access_ms,
            "inference_ms": inference_ms,
        }

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
            batch_size=self.batch_size,
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


def run_rerank_stage(
    reranker: CrossEncoderReranker,
    query: str,
    fused_results: list[tuple[str, float]],
    top_k: int,
    text_for: Callable[[str, str], str],
    timings: RetrievalTimings,
) -> list[tuple[str, float]]:
    """Stage 5 rerank, with the time split into its parts.

    ``text_for(cand_id, query)`` builds the text the cross-encoder sees for one
    candidate. Preparation, model access and prediction are timed separately
    and recorded on ``timings``. Metrics are recorded even when reranking
    raises, and the exception then propagates unchanged. The returned list is
    exactly what ``reranker.rerank`` returns.

    Logs one structured line with counts and timings only. It never includes
    the query, document text or model outputs.
    """
    t_total = time.perf_counter()
    rerank_info: dict = {}
    ok = False
    try:
        t_prep = time.perf_counter()
        try:
            rerank_texts = {
                cand_id: text_for(cand_id, query) for cand_id, _s in fused_results
            }
        finally:
            timings.candidate_text_prepare_ms = (time.perf_counter() - t_prep) * 1000

        final = reranker.rerank(
            query=query,
            candidates=fused_results,
            k=top_k,
            doc_texts=rerank_texts,
            timing=rerank_info,
        )
        ok = True
        return final
    finally:
        total_ms = (time.perf_counter() - t_total) * 1000
        timings.rerank_total_ms = total_ms
        timings.rerank_ms = total_ms
        timings.model_access_ms = float(rerank_info.get("model_access_ms", 0.0))
        timings.model_predict_ms = float(rerank_info.get("model_predict_ms", 0.0))
        timings.candidate_count = len(fused_results)
        timings.rerank_batch_size = int(rerank_info.get("batch_size", 0))
        timings.rerank_status = "ok" if ok else "failed"
        print(
            f"[rerank-timing] status={timings.rerank_status} "
            f"candidates={timings.candidate_count} top_k={top_k} "
            f"batch_size={timings.rerank_batch_size} "
            f"device={getattr(reranker, 'device', '?')} "
            f"text_prepare_ms={timings.candidate_text_prepare_ms:.1f} "
            f"model_access_ms={timings.model_access_ms:.1f} "
            f"model_predict_ms={timings.model_predict_ms:.1f} "
            f"rerank_total_ms={timings.rerank_total_ms:.1f}"
        )
