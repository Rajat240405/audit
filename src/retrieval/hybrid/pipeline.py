"""
Hybrid RAG Retrieval Pipeline.

Orchestrates: Dense vector search + BM25 search → RRF Fusion → Cross-encoder rerank.

Key Design Decisions
--------------------
1. The pipeline indexes the CONCATENATED question + answer text
   (via document_content) for both dense and BM25 retrieval.

2. After retrieval, original structured records are returned.
   The concatenated text is NEVER exposed to the user — only the
   original question and answer fields.

3. The pipeline has two modes:
   - Offline: build() — index the corpus (done once)
   - Online: retrieve() — answer user queries (done many times)

4. Latency breakdown: each sub-stage is timed independently so we can
   report exactly where time is spent (critical for the evaluation phase).

5. LLM generation is NOT part of this pipeline — it's a separate stage.
   This keeps retrieval evaluation clean (no LLM confounding variable).
   The retrieve() method returns RetrievedResult objects, not answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.models.qa_record import QARecord
from src.retrieval.hybrid.embedder import Embedder
from src.retrieval.hybrid.vector_store import FAISSVectorStore
from src.retrieval.hybrid.bm25_index import BM25Index
from src.retrieval.hybrid.fusion import RRF
from src.retrieval.hybrid.reranker import CrossEncoderReranker
from src.retrieval.result import RetrievedResult


# ─────────────────────────────────────────────────────────────────────────────
# Timing dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalTimings:
    """Per-stage latency breakdown for a retrieval operation."""

    embed_query_ms: float = 0.0
    dense_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "embed_query_ms": self.embed_query_ms,
            "dense_search_ms": self.dense_search_ms,
            "bm25_search_ms": self.bm25_search_ms,
            "rrf_fusion_ms": self.rrf_fusion_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HybridRAGPipeline:
    """
    Hybrid RAG retrieval: dense + BM25 → RRF → cross-encoder rerank.

    Indexes the concatenated document content (question + answer).
    Returns original structured records with per-stage timing.

    Usage
    -----
    Build (once):
    ```python
    pipeline = HybridRAGPipeline(records=qa_records)
    pipeline.build()
    pipeline.save("storage/hybrid_rag")
    ```

    Retrieve (many times):
    ```python
    results, timings = pipeline.retrieve("What about malaria?", top_k=5)
    ```

    Load from disk:
    ```python
    pipeline = HybridRAGPipeline()
    pipeline.load("storage/hybrid_rag")
    results, timings = pipeline.retrieve("malaria question")
    ```
    """

    def __init__(
        self,
        records: Optional[list[QARecord]] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        dense_top_k: int = 50,
        fusion_top_k: int = 20,
        rrf_k: int = 60,
        use_reranker: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        records : list[QARecord], optional
            The knowledge base records. Required for build(), optional for load().
        embedder : Embedder, optional
            Dense embedding model. Default: all-MiniLM-L6-v2 on CPU.
        vector_store : FAISSVectorStore, optional
            FAISS index. Created automatically if not provided.
        bm25_index : BM25Index, optional
            BM25 index. Created automatically if not provided.
        reranker : CrossEncoderReranker, optional
            Cross-encoder reranker. Default: ms-marco-MiniLM-L-12-v2 on CPU.
            Set use_reranker=False to skip reranking (faster, slightly less accurate).
        dense_top_k : int
            How many results to retrieve from dense vector search (before fusion).
            Default 50 is standard — captures enough candidates without being slow.
        fusion_top_k : int
            How many results to keep after RRF fusion (before reranking).
            Default 20 gives the reranker good candidates to choose from.
        rrf_k : int
            RRF smoothing parameter. Default 60 is standard.
        use_reranker : bool
            If False, skip cross-encoder reranking. Use for speed-critical paths.
        """
        self._records: Optional[list[QARecord]] = records

        # Embedding model (CPU-viable)
        self.embedder = embedder or Embedder()

        # Vector store
        self._embedding_dim = self.embedder.embedding_dim
        self.vector_store = vector_store or FAISSVectorStore(
            embedding_dim=self._embedding_dim
        )

        # BM25 index
        self.bm25_index = bm25_index or BM25Index()

        # Cross-encoder reranker (CPU-viable)
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()
        self.use_reranker = use_reranker

        # Hyperparameters
        self.dense_top_k = dense_top_k
        self.fusion_top_k = fusion_top_k
        self.rrf_k = rrf_k

        # In-memory doc lookup (doc_id → QARecord)
        self._doc_map: dict[str, QARecord] = {}

        # Cached doc texts for reranker
        self._doc_texts: dict[str, str] = {}

        # Build index immediately if records are provided
        if records:
            self.build()

    def build(self) -> None:
        """
        Build both the dense vector index and BM25 index from the knowledge base.

        This is an offline operation — run once after data ingestion.

        Raises
        ------
        ValueError
            If no records were provided to the constructor.
        """
        if not self._records:
            raise ValueError("No records provided. Pass records to constructor or call load().")

        records = self._records

        # Build doc_id → record lookup
        self._doc_map = {r.question_id: r for r in records}

        # Build doc_id → concatenated text lookup
        self._doc_texts = {
            r.question_id: r.document_content
            for r in records
        }

        # ── 1. Dense vector embeddings ──────────────────────────────────────
        doc_ids = [r.question_id for r in records]
        texts = [r.document_content for r in records]

        print(f"Generating embeddings for {len(records):,} documents...")
        t0 = time.monotonic()
        embeddings = self.embedder.embed_batch(texts, batch_size=32, show_progress=True)
        embed_time = (time.monotonic() - t0) * 1000
        print(f"  Embeddings: {embeddings.shape} in {embed_time:.0f}ms")

        self.vector_store.build(doc_ids, embeddings)

        # ── 2. BM25 index ──────────────────────────────────────────────────
        print(f"Building BM25 index for {len(records):,} documents...")
        t0 = time.monotonic()
        bm25_docs = [
            (r.question_id, r.question_text, r.answer_text)
            for r in records
        ]
        self.bm25_index.build(bm25_docs)
        bm25_time = (time.monotonic() - t0) * 1000
        print(f"  BM25 built in {bm25_time:.0f}ms")

        print(
            f"✓ Hybrid RAG index built: {len(self):,} docs, "
            f"dim={self._embedding_dim}, BM25 k1={self.bm25_index.k1}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievedResult], RetrievalTimings]:
        """
        Retrieve top-K relevant Q&A records using hybrid retrieval.

        Pipeline: embed query → dense search → BM25 search → RRF fusion
                 → cross-encoder rerank → return structured records.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int
            Number of results to return.

        Returns
        -------
        tuple[list[RetrievedResult], RetrievalTimings]
            - List of RetrievedResult objects (original question + answer, not concatenated).
            - Per-stage latency breakdown.
        """
        total_start = time.perf_counter()
        timings = RetrievalTimings()

        # ── Stage 1: Dense vector retrieval ────────────────────────────────
        t_embed = time.monotonic()
        query_embedding = self.embedder.embed(query)
        timings.embed_query_ms = (time.monotonic() - t_embed) * 1000

        t_dense = time.monotonic()
        dense_results = self.vector_store.search(query_embedding, k=self.dense_top_k)
        timings.dense_search_ms = (time.monotonic() - t_dense) * 1000

        # ── Stage 2: BM25 retrieval ──────────────────────────────────────
        t_bm25 = time.monotonic()
        bm25_results = self.bm25_index.search(query, k=self.dense_top_k)
        timings.bm25_search_ms = (time.monotonic() - t_bm25) * 1000

        # ── Stage 3: RRF Fusion ───────────────────────────────────────────
        t_rrf = time.monotonic()
        fused_results = RRF.fuse(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=self.fusion_top_k,
        )
        timings.rrf_fusion_ms = (time.monotonic() - t_rrf) * 1000

        # ── Stage 4: Cross-encoder reranking ──────────────────────────────
        if self.use_reranker and fused_results:
            t_rerank = time.monotonic()
            reranked_results = self.reranker.rerank(
                query=query,
                candidates=fused_results,
                k=top_k,
                doc_texts=self._doc_texts,
            )
            timings.rerank_ms = (time.monotonic() - t_rerank) * 1000
            final_results = reranked_results
        else:
            # Skip reranking: take top_k from RRF fusion
            final_results = fused_results[:top_k]

        # ── Stage 5: Build RetrievedResult objects ─────────────────────────
        retrieved: list[RetrievedResult] = []
        for rank, (doc_id, score) in enumerate(final_results):
            record = self._doc_map.get(doc_id)
            if record is None:
                continue  # Skip if doc not found (shouldn't happen)

            # Find the original scores from sub-systems
            dense_score = next(
                (s for d_id, s in dense_results if d_id == doc_id), None
            )
            bm25_score = next(
                (s for d_id, s in bm25_results if d_id == doc_id), None
            )
            rrf_score = next(
                (s for d_id, s in fused_results if d_id == doc_id), None
            )

            retrieved.append(RetrievedResult(
                doc_id=doc_id,
                question=record.question_text,
                answer=record.answer_text,
                score=score,
                retrieval_method="rrf_fusion",
                metadata={
                    "ministry": record.metadata.ministry,
                    "subject": record.metadata.subject,
                    "date": record.metadata.date,
                    "question_type": (
                        record.metadata.question_type.value
                        if record.metadata.question_type else None
                    ),
                },
                dense_score=dense_score,
                bm25_score=bm25_score,
                rrf_score=rrf_score,
                rerank_score=score if self.use_reranker else None,
            ))

        timings.total_ms = (time.monotonic() - total_start) * 1000

        return retrieved, timings

    def save(self, path: str | Path) -> None:
        """
        Save the complete pipeline to disk.

        Writes:
        - {path}/vector_store.index + .ids (FAISS)
        - {path}/bm25_index.pkl + .json
        - {path}/doc_map.json (doc_id → record JSON)
        - {path}/doc_texts.json (doc_id → concatenated text)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(path / "vector_store")
        self.bm25_index.save(path / "bm25_index")

        # Save doc_map (simplified — just the fields needed for retrieval)
        import orjson
        doc_map_data = {
            doc_id: record.model_dump(mode="json")
            for doc_id, record in self._doc_map.items()
        }
        with open(path / "doc_map.json", "wb") as f:
            f.write(orjson.dumps(doc_map_data))

        # Save doc texts
        with open(path / "doc_texts.json", "w", encoding="utf-8") as f:
            import json
            json.dump(self._doc_texts, f)

        print(f"✓ Saved Hybrid RAG pipeline to {path}")

    def load(self, path: str | Path) -> None:
        """
        Load a previously saved pipeline from disk.
        """
        path = Path(path)

        self.vector_store = FAISSVectorStore(
            embedding_dim=self._embedding_dim,
            index_path=str(path / "vector_store"),
        )
        self.vector_store.load(path / "vector_store")

        self.bm25_index = BM25Index()
        self.bm25_index.load(path / "bm25_index")

        # Load doc_map
        import orjson
        with open(path / "doc_map.json", "rb") as f:
            doc_map_data = orjson.loads(f.read())
        self._doc_map = {
            doc_id: QARecord.model_validate(data)
            for doc_id, data in doc_map_data.items()
        }

        # Load doc texts
        with open(path / "doc_texts.json", encoding="utf-8") as f:
            import json
            self._doc_texts = json.load(f)

        print(f"✓ Loaded Hybrid RAG pipeline from {path}")

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self._doc_map)

    def __repr__(self) -> str:
        return (
            f"HybridRAGPipeline(n_docs={len(self)}, "
            f"dense_top_k={self.dense_top_k}, "
            f"fusion_top_k={self.fusion_top_k}, "
            f"use_reranker={self.use_reranker})"
        )
