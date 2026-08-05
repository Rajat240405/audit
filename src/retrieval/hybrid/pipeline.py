"""
Hybrid RAG Retrieval Pipeline.

Orchestrates: Dense vector search + BM25 search → RRF Fusion → Cross-encoder rerank.
Supports both Document-level and Chunk-level retrieval.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from src.models.qa_record import QARecord, QAChunk, ChunkType, QARecordMetadata
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
    Can be configured for either document-level or chunk-level retrieval.
    """

    def __init__(
        self,
        records: list[QARecord] | None = None,
        embedder: Embedder | None = None,
        vector_store: FAISSVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
        dense_top_k: int = 50,
        fusion_top_k: int = 20,
        rrf_k: int = 60,
        use_reranker: bool = True,
        use_chunking: bool = False,  # Default to Document-based for backwards compatibility
    ) -> None:
        """
        Parameters
        ----------
        records : list[QARecord], optional
            The knowledge base records. Required for build(), optional for load().
        embedder : Embedder, optional
            Dense embedding model. Default: BAAI/bge-m3 on CPU.
        use_chunking : bool
            If True, split records into separate Q and A chunks during indexing.
        """
        self._records: list[QARecord] | None = records
        self.use_chunking = use_chunking

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

        # In-memory lookups
        self._doc_map: dict[str, QARecord] = {}
        self._chunk_map: dict[str, QAChunk] = {}

        # Cached index texts (used for reranker)
        self._doc_texts: dict[str, str] = {}
        self._chunk_texts: dict[str, str] = {}

        # Build index immediately if records are provided
        if records:
            self.build()

    def build(self) -> None:
        """
        Build both dense vector and BM25 indices.
        Supports either Document or Chunk slicing based on self.use_chunking.
        """
        if not self._records:
            raise ValueError("No records provided. Pass records to constructor or call load().")

        records = self._records
        print(f"[DEBUG LOG] Number of QARecords loaded: {len(records)}")

        # Always build the parent doc map first
        self._doc_map = {r.question_id: r for r in records}
        self._doc_texts = {r.question_id: r.document_content for r in records}

        # Clear chunk mapping
        self._chunk_map.clear()
        self._chunk_texts.clear()

        if self.use_chunking:
            # ── 1. Chunk Generation ──────────────────────────────────────────
            print(f"Generating chunks for {len(records):,} documents...")
            chunks_list: list[QAChunk] = []
            for r in records:
                # Question Chunk
                q_chunk = QAChunk(
                    chunk_id=f"{r.question_id}_Q",
                    parent_doc_id=r.question_id,
                    chunk_type=ChunkType.QUESTION,
                    chunk_text=f"QUESTION: {r.question_text}",
                    metadata=r.metadata
                )
                # Answer Chunk
                a_chunk = QAChunk(
                    chunk_id=f"{r.question_id}_A",
                    parent_doc_id=r.question_id,
                    chunk_type=ChunkType.ANSWER,
                    chunk_text=f"ANSWER: {r.answer_text}",
                    metadata=r.metadata
                )
                chunks_list.extend([q_chunk, a_chunk])

            self._chunk_map = {c.chunk_id: c for c in chunks_list}
            self._chunk_texts = {c.chunk_id: c.chunk_text for c in chunks_list}
            print(f"[DEBUG LOG] Number of QAChunks generated: {len(chunks_list)}")

            # Dense Index Chunks
            chunk_ids = [c.chunk_id for c in chunks_list]
            texts = [c.chunk_text for c in chunks_list]

            print(f"Generating embeddings for {len(chunks_list):,} chunks...")
            t0 = time.perf_counter()
            embeddings = self.embedder.embed_batch(texts, batch_size=8, show_progress=True)
            embed_time = (time.perf_counter() - t0) * 1000
            print(f"  Embeddings: {embeddings.shape} in {embed_time:.0f}ms")

            self.vector_store.build(chunk_ids, embeddings)
            print(f"[DEBUG LOG] Number of vectors added to FAISS: {len(self.vector_store)}")

            # BM25 Index Chunks
            t0 = time.perf_counter()
            bm25_docs = [
                (c.chunk_id, "", c.chunk_text)
                for c in chunks_list
            ]
            self.bm25_index.build(bm25_docs)
            print(f"  BM25 built in {(time.perf_counter() - t0)*1000:.0f}ms")
            print(f"[DEBUG LOG] Number of BM25 documents indexed: {len(self.bm25_index)}")

        else:
            # ── 2. Standard Document Generation ─────────────────────────────
            doc_ids = [r.question_id for r in records]
            texts = [r.document_content for r in records]

            print(f"Generating embeddings for {len(records):,} documents...")
            t0 = time.perf_counter()
            embeddings = self.embedder.embed_batch(texts, batch_size=1, show_progress=True)
            embed_time = (time.perf_counter() - t0) * 1000
            print(f"  Embeddings: {embeddings.shape} in {embed_time:.0f}ms")

            self.vector_store.build(doc_ids, embeddings)

            # BM25 Index Documents
            t0 = time.perf_counter()
            bm25_docs = [
                (r.question_id, r.question_text, r.answer_text)
                for r in records
            ]
            self.bm25_index.build(bm25_docs)
            print(f"  BM25 built in {(time.perf_counter() - t0)*1000:.0f}ms")

        print(
            f"✓ Index built successfully: {len(self):,} units indexed, "
            f"use_chunking={self.use_chunking}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievedResult], RetrievalTimings]:
        """
        Retrieve relevant results with standardized runtime logging (Phase 11).
        """
        total_start = time.perf_counter()
        timings = RetrievalTimings()

        # ── Stage 1: Dense Retrieval ───────────────────────────────────────
        t_embed = time.perf_counter()
        query_embedding = self.embedder.embed(query)
        timings.embed_query_ms = (time.perf_counter() - t_embed) * 1000

        t_dense = time.perf_counter()
        dense_results = self.vector_store.search(query_embedding, k=self.dense_top_k)
        timings.dense_search_ms = (time.perf_counter() - t_dense) * 1000
        print(f"Dense candidates : {len(dense_results)}")

        # ── Stage 2: BM25 Retrieval ────────────────────────────────────────
        t_bm25 = time.perf_counter()
        bm25_results = self.bm25_index.search(query, k=self.dense_top_k)
        timings.bm25_search_ms = (time.perf_counter() - t_bm25) * 1000
        print(f"BM25 candidates : {len(bm25_results)}")

        # ── Stage 3: RRF Fusion ─────────────────────────────────────────────
        t_rrf = time.perf_counter()
        fused_results = RRF.fuse(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=self.fusion_top_k,
        )
        timings.rrf_fusion_ms = (time.perf_counter() - t_rrf) * 1000
        print(f"RRF candidates : {len(fused_results)}")

        # ── Stage 4: Cross-Encoder Reranking ────────────────────────────────
        if self.use_reranker and fused_results:
            t_rerank = time.perf_counter()
            active_texts = self._chunk_texts if self.use_chunking else self._doc_texts
            
            reranked_results = self.reranker.rerank(
                query=query,
                candidates=fused_results,
                k=top_k,
                doc_texts=active_texts,
            )
            timings.rerank_ms = (time.perf_counter() - t_rerank) * 1000
            final_results = reranked_results
        else:
            final_results = fused_results[:top_k]
        print(f"CrossEncoder candidates : {len(final_results)}")

        # ── Stage 5: Context Assembly & Document Re-Grouping ────────────────
        retrieved: list[RetrievedResult] = []

        if self.use_chunking:
            # Group retrieved chunks by parent document ID
            grouped_chunks: dict[str, list[tuple[str, float]]] = {}
            for chunk_id, score in final_results:
                chunk = self._chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                parent_id = chunk.parent_doc_id
                if parent_id not in grouped_chunks:
                    grouped_chunks[parent_id] = []
                grouped_chunks[parent_id].append((chunk_id, score))

            print(f"Parent aggregation : {len(grouped_chunks)}")

            # Assemble merged retrieved results
            for parent_id, chunks in grouped_chunks.items():
                record = self._doc_map.get(parent_id)
                if record is None:
                    continue

                # Find which chunk types we retrieved
                q_text = ""
                a_text = ""
                highest_score = max(score for _, score in chunks)

                for chunk_id, _ in chunks:
                    chunk = self._chunk_map[chunk_id]
                    if chunk.chunk_type == ChunkType.QUESTION:
                        q_text = chunk.chunk_text
                    elif chunk.chunk_type == ChunkType.ANSWER:
                        a_text = chunk.chunk_text

                # Fallback to parent core strings if specific chunk was missed
                if not q_text:
                    q_text = f"QUESTION: {record.question_text}"
                if not a_text:
                    a_text = f"ANSWER: {record.answer_text[:2000]} ... [Truncated to fit context budget]"

                # Find standard scores from systems
                dense_score = next((s for d_id, s in dense_results if d_id in [c[0] for c in chunks]), None)
                bm25_score = next((s for d_id, s in bm25_results if d_id in [c[0] for c in chunks]), None)
                rrf_score = next((s for d_id, s in fused_results if d_id in [c[0] for c in chunks]), None)

                retrieved.append(RetrievedResult(
                    doc_id=parent_id,
                    question=q_text,
                    answer=a_text,
                    score=highest_score,
                    retrieval_method="rrf_fusion:chunk_assembly",
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
                    rerank_score=highest_score if self.use_reranker else None,
                ))
        else:
            print(f"Parent aggregation : {len(final_results)}")
            # Standard document-level construction
            for rank, (doc_id, score) in enumerate(final_results):
                record = self._doc_map.get(doc_id)
                if record is None:
                    continue

                dense_score = next((s for d_id, s in dense_results if d_id == doc_id), None)
                bm25_score = next((s for d_id, s in bm25_results if d_id == doc_id), None)
                rrf_score = next((s for d_id, s in fused_results if d_id == doc_id), None)

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

        timings.total_ms = (time.perf_counter() - total_start) * 1000
        print(f"Retrieval latency : {timings.total_ms:.2f} ms")

        return retrieved, timings

    def save(self, path: str | Path) -> None:
        """Save the complete pipeline state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(path / "vector_store")
        self.bm25_index.save(path / "bm25_index")

        # Save metadata properties
        meta = {
            "use_chunking": self.use_chunking,
            "embedding_dim": self._embedding_dim,
        }
        with open(path / "pipeline_metadata.json", "w") as f:
            json.dump(meta, f)

        # Save doc_map (portable JSON)
        import orjson
        doc_map_data = {
            doc_id: record.model_dump(mode="json")
            for doc_id, record in self._doc_map.items()
        }
        with open(path / "doc_map.json", "wb") as f:
            f.write(orjson.dumps(doc_map_data))

        # Save chunk map
        chunk_map_data = {
            chunk_id: chunk.model_dump(mode="json")
            for chunk_id, chunk in self._chunk_map.items()
        }
        with open(path / "chunk_map.json", "wb") as f:
            f.write(orjson.dumps(chunk_map_data))

        print(f"✓ Saved Hybrid RAG pipeline (use_chunking={self.use_chunking}) to {path}")

    def load(self, path: str | Path) -> None:
        """Load a serialized pipeline state from disk."""
        path = Path(path)

        # Load pipeline metadata
        if (path / "pipeline_metadata.json").exists():
            with open(path / "pipeline_metadata.json") as f:
                meta = json.load(f)
                self.use_chunking = meta.get("use_chunking", self.use_chunking)
                self._embedding_dim = meta.get("embedding_dim", self._embedding_dim)

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
        self._doc_texts = {doc_id: r.document_content for doc_id, r in self._doc_map.items()}

        # Load chunk_map
        if (path / "chunk_map.json").exists():
            with open(path / "chunk_map.json", "rb") as f:
                chunk_map_data = orjson.loads(f.read())
            self._chunk_map = {
                chunk_id: QAChunk.model_validate(data)
                for chunk_id, data in chunk_map_data.items()
            }
            self._chunk_texts = {chunk_id: c.chunk_text for chunk_id, c in self._chunk_map.items()}

        print(f"✓ Loaded Hybrid RAG pipeline (use_chunking={self.use_chunking}) from {path}")

    def __len__(self) -> int:
        """Return the number of parent indexed documents."""
        return len(self._doc_map)

    def __repr__(self) -> str:
        return (
            f"HybridRAGPipeline(n_docs={len(self)}, "
            f"use_chunking={self.use_chunking}, "
            f"use_reranker={self.use_reranker})"
        )
