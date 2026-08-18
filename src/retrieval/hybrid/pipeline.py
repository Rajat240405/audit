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
from typing import Any, Callable, Dict, List, Optional

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
        fusion_top_k: int = 50,
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
        # GLM #5b: targeted chunking for LONG documents only. Docs above this
        # length get split into ~500-char ANNEXURE chunks at index time so a
        # figure buried in a long answer can be found. Short docs stay whole
        # (the working median path is untouched).
        self.long_doc_chars = 4000
        self.long_chunk_chars = 500
        self._long_chunk_map: dict[str, QAChunk] = {}
        self._long_chunk_texts: dict[str, str] = {}

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

            # ── GLM #5b: targeted long-doc chunking ─────────────────────────
            # For docs > long_doc_chars, ALSO index ~500-char ANNEXURE chunks
            # so a figure buried in a long answer can be found. Parent docs
            # remain the primary unit; chunks are a recall boost for the
            # long tail. Short docs are untouched.
            long_recs = [r for r in records if len(r.answer_text or "") > self.long_doc_chars]
            if long_recs:
                chunk_list = []
                for r in long_recs:
                    chunk_list.extend(self._split_long_doc(r))
                self._long_chunk_map = {c.chunk_id: c for c in chunk_list}
                self._long_chunk_texts = {c.chunk_id: c.chunk_text for c in chunk_list}

                print(f"  [long-doc chunks] {len(long_recs)} long docs -> {len(chunk_list)} chunks")
                # embed + index chunks into the SAME vector store
                c_ids = [c.chunk_id for c in chunk_list]
                c_texts = [c.chunk_text for c in chunk_list]
                if c_texts:
                    c_emb = self.embedder.embed_batch(c_texts, batch_size=1, show_progress=False)
                    self.vector_store.add(c_ids, c_emb)
                # Rebuild BM25 with docs + chunks (BM25 has no incremental add;
                # the corpus is small enough to rebuild in ms).
                c_bm25 = [(c.chunk_id, "", c.chunk_text) for c in chunk_list]
                combined_bm25 = [
                    (r.question_id, r.question_text, r.answer_text) for r in records
                ] + c_bm25
                self.bm25_index.build(combined_bm25)
                print(f"  [long-doc chunks] indexed into FAISS + BM25 (rebuilt)")

        print(
            f"✓ Index built successfully: {len(self):,} units indexed, "
            f"use_chunking={self.use_chunking}"
        )

    def add_records(self, records: list) -> int:
        """INCREMENTALLY add new records to an already-built index.

        Only records whose question_id is NOT already in the doc_map are
        embedded and added:
          - FAISS: new doc embeddings via vector_store.add() (no full rebuild)
          - long-doc chunks: new chunks embedded + added the same way
          - BM25: rebuilt with docs + chunks (fast, text-only, ~ms)
        Returns the number of records actually added.
        """
        existing = set(self._doc_map.keys())
        new_recs = [r for r in records if r.question_id not in existing]
        if not new_recs:
            print("[add_records] nothing new to add")
            return 0

        print(f"[add_records] adding {len(new_recs):,} new record(s) (index already has {len(existing):,})")

        # 1. Add docs to in-memory maps
        for r in new_recs:
            self._doc_map[r.question_id] = r
        self._doc_texts = {doc_id: r.document_content for doc_id, r in self._doc_map.items()}

        # 2. Embed ONLY the new docs
        texts = [r.document_content for r in new_recs]
        doc_ids = [r.question_id for r in new_recs]
        embeddings = self.embedder.embed_batch(texts, batch_size=1, show_progress=False)
        self.vector_store.add(doc_ids, embeddings)
        print(f"[add_records] added {len(doc_ids)} embeddings to FAISS")

        # 3. Long-doc chunks for the new long records
        long_recs = [r for r in new_recs if len(r.answer_text or "") > self.long_doc_chars]
        new_chunks: list = []
        if long_recs:
            for r in long_recs:
                new_chunks.extend(self._split_long_doc(r))
            self._long_chunk_map.update({c.chunk_id: c for c in new_chunks})
            self._long_chunk_texts.update({c.chunk_id: c.chunk_text for c in new_chunks})
            if new_chunks:
                c_emb = self.embedder.embed_batch(
                    [c.chunk_text for c in new_chunks], batch_size=1, show_progress=False
                )
                self.vector_store.add([c.chunk_id for c in new_chunks], c_emb)
                print(f"[add_records] added {len(new_chunks)} long-doc chunks")

        # 4. BM25: rebuild with ALL docs + chunks (text-only, fast)
        bm25_docs = [
            (r.question_id, r.question_text, r.answer_text)
            for r in self._doc_map.values()
        ]
        bm25_docs += [(c.chunk_id, "", c.chunk_text) for c in self._long_chunk_map.values()]
        self.bm25_index.build(bm25_docs)

        print(f"[add_records] done — {len(new_recs)} added, index now has {len(self._doc_map):,} docs")
        return len(new_recs)

    def _split_long_doc(self, r: QARecord) -> list[QAChunk]:
        """Split a long document's answer into ~500-char ANNEXURE chunks."""
        chunks = []
        ans = r.answer_text or ""
        # split on paragraph boundaries first, then pack into ~500-char chunks
        paras = [p.strip() for p in ans.split("\n") if p.strip()]
        current = ""
        idx = 0
        for para in paras:
            if len(current) + len(para) + 2 > self.long_chunk_chars and current:
                chunks.append(QAChunk(
                    chunk_id=f"{r.question_id}_L{idx}",
                    parent_doc_id=r.question_id,
                    chunk_type=ChunkType.ANNEXURE,
                    chunk_text=current.strip(),
                    metadata=r.metadata,
                ))
                idx += 1
                current = para
            else:
                current = current + "\n" + para if current else para
        if current.strip():
            chunks.append(QAChunk(
                chunk_id=f"{r.question_id}_L{idx}",
                parent_doc_id=r.question_id,
                chunk_type=ChunkType.ANNEXURE,
                chunk_text=current.strip(),
                metadata=r.metadata,
            ))
        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        on_stage: Optional[Callable[[str, dict], None]] = None,
        doc_types: Optional[list[str]] = None,
        orgs: Optional[list[str]] = None,
        doc_categories: Optional[list[str]] = None,
    ) -> tuple[list[RetrievedResult], RetrievalTimings]:
        """
        Retrieve relevant results with standardized runtime logging (Phase 11).

        ``on_stage`` is an optional callback invoked as each pipeline stage
        completes (stage name -> info dict); used by the frontend SSE endpoint
        to drive the live Pipeline tab.
        """
        total_start = time.perf_counter()
        timings = RetrievalTimings()

        # ── Query expansion (Q3-class fix) ────────────────────────────────
        # Expand entity mentions with role keywords so both sides of a
        # comparison surface (e.g. "INCOIS and NIOT" also retrieves NIOT docs).
        from src.retrieval.hybrid.query_expansion import expand_query
        expanded_query = expand_query(query)

        # ── Stage 1: Embed query ───────────────────────────────────────────
        t_embed = time.perf_counter()
        if on_stage:
            on_stage("embed", {})
        # Use the expanded query for retrieval; the reranker still sees the
        # original user query for relevance scoring.
        query_embedding = self.embedder.embed(expanded_query)
        timings.embed_query_ms = (time.perf_counter() - t_embed) * 1000

        # ── Stage 2: Dense retrieval ───────────────────────────────────────
        t_dense = time.perf_counter()
        dense_results = self.vector_store.search(query_embedding, k=self.dense_top_k)
        timings.dense_search_ms = (time.perf_counter() - t_dense) * 1000
        print(f"Dense candidates : {len(dense_results)}")
        if on_stage:
            on_stage("dense", {"count": len(dense_results)})

        # ── Stage 3: BM25 retrieval ────────────────────────────────────────
        t_bm25 = time.perf_counter()
        bm25_results = self.bm25_index.search(expanded_query, k=self.dense_top_k)
        timings.bm25_search_ms = (time.perf_counter() - t_bm25) * 1000
        print(f"BM25 candidates : {len(bm25_results)}")
        if on_stage:
            on_stage("bm25", {"count": len(bm25_results)})

        # ── Stage 4: RRF fusion ────────────────────────────────────────────
        t_rrf = time.perf_counter()
        fused_results = RRF.fuse(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=self.fusion_top_k,
        )
        timings.rrf_fusion_ms = (time.perf_counter() - t_rrf) * 1000
        print(f"RRF candidates : {len(fused_results)}")
        if on_stage:
            on_stage("rrf", {"count": len(fused_results)})

        # ── Stage 4.5: Metadata filters (doc_types / orgs / categories) ──
        # Optional, applied post-RRF pre-rerank. Union WITHIN an axis, AND
        # across axes. The reranker scores only the survivors, so the answer
        # is guaranteed to come from the requested sources.
        def _record_of(candidate):
            doc_id = candidate[0]
            rec = self._doc_map.get(doc_id)
            if rec is None and self._chunk_map:
                chunk = self._chunk_map.get(doc_id)
                if chunk is not None:
                    rec = self._doc_map.get(chunk.parent_doc_id)
            if rec is None and self._long_chunk_map:
                chunk = self._long_chunk_map.get(doc_id)
                if chunk is not None:
                    rec = self._doc_map.get(chunk.parent_doc_id)
            return rec

        def _type_of(candidate) -> str | None:
            rec = _record_of(candidate)
            if rec is None or rec.metadata is None:
                return None
            return (rec.metadata.document_type or "document").lower()

        def _org_of(candidate) -> str | None:
            rec = _record_of(candidate)
            if rec is None or rec.metadata is None:
                return None
            from src.retrieval.frontend.org_tree import derive_org
            return derive_org({
                "document_type": rec.metadata.document_type,
                "subject": rec.metadata.subject,
                "source_url": rec.metadata.source_url,
                "session": rec.metadata.session,
                "question_number": rec.metadata.question_number,
                "member": rec.metadata.member,
                "question_text": rec.question_text,
                "answer_text": rec.answer_text,
            })

        def _category_of(candidate) -> str | None:
            rec = _record_of(candidate)
            if rec is None or rec.metadata is None:
                return None
            from src.retrieval.frontend.org_tree import derive_category
            return derive_category({"document_type": rec.metadata.document_type})

        if doc_types:
            allowed = set(doc_types)
            before = len(fused_results)
            fused_results = [c for c in fused_results if (_type_of(c) or "") in allowed]
            print(f"Source filter ({sorted(allowed)}): {before} -> {len(fused_results)} candidates")
            if on_stage:
                on_stage("filter", {"count": len(fused_results)})

        if orgs:
            allowed = set(orgs)
            before = len(fused_results)
            fused_results = [c for c in fused_results if (_org_of(c) or "") in allowed]
            print(f"Org filter ({sorted(allowed)}): {before} -> {len(fused_results)} candidates")
            if on_stage:
                on_stage("filter", {"count": len(fused_results)})

        if doc_categories:
            allowed = set(doc_categories)
            before = len(fused_results)
            fused_results = [c for c in fused_results if (_category_of(c) or "") in allowed]
            print(f"Category filter ({sorted(allowed)}): {before} -> {len(fused_results)} candidates")
            if on_stage:
                on_stage("filter", {"count": len(fused_results)})

        # ── Stage 5: Cross-encoder reranking ───────────────────────────────
        if self.use_reranker and fused_results:
            t_rerank = time.perf_counter()
            active_texts = dict(self._doc_texts)
            if self.use_chunking:
                active_texts.update(self._chunk_texts)
            active_texts.update(self._long_chunk_texts)
            
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
        if on_stage:
            on_stage("rerank", {"count": len(final_results)})

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
                        "document_type": record.metadata.document_type,
                    },
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    rrf_score=rrf_score,
                    rerank_score=highest_score if self.use_reranker else None,
                ))
        else:
            print(f"Parent aggregation : {len(final_results)}")
            # Chunk hits keep parent doc_id for attribution, but evidence is
            # ONLY the matched chunks (never the 400k parent body).
            grouped: dict[str, dict] = {}
            for doc_id, score in final_results:
                chunk = self._long_chunk_map.get(doc_id)
                if chunk is not None:
                    pid = chunk.parent_doc_id
                    record = self._doc_map.get(pid)
                    if record is None:
                        continue
                    g = grouped.setdefault(
                        pid, {"record": record, "score": score, "chunks": [], "chunk_ids": []}
                    )
                    if chunk.chunk_text not in g["chunks"]:
                        g["chunks"].append(chunk.chunk_text)
                        g["chunk_ids"].append(chunk.chunk_id)
                    g["score"] = max(g["score"], score)
                    continue
                record = self._doc_map.get(doc_id)
                if record is None:
                    continue
                g = grouped.setdefault(doc_id, {"record": record, "score": score, "chunks": [], "chunk_ids": []})
                g["score"] = max(g["score"], score)

            for pid, g in grouped.items():
                record = g["record"]
                long_doc = len(record.answer_text or "") > self.long_doc_chars
                if g["chunks"]:
                    evidence = "\n\n".join(g["chunks"])
                elif long_doc:
                    from src.generation.generator import extract_relevant_evidence

                    evidence = extract_relevant_evidence(
                        record.answer_text or "", query, max_chars=2000
                    )
                else:
                    evidence = record.answer_text

                dense_score = next((s for d_id, s in dense_results if d_id == pid), None)
                bm25_score = next((s for d_id, s in bm25_results if d_id == pid), None)
                rrf_score = next((s for d_id, s in fused_results if d_id == pid), None)

                retrieved.append(RetrievedResult(
                    doc_id=pid,
                    question=record.question_text,
                    answer=evidence,
                    # Task-3 fix: use THIS group's max score — previously this
                    # read the leaked loop variable (last candidate's score for
                    # every result), corrupting score-based allocation.
                    score=g["score"],
                    retrieval_method="rrf_fusion",
                    metadata={
                        "ministry": record.metadata.ministry,
                        "subject": record.metadata.subject,
                        "date": record.metadata.date,
                        "question_type": (
                            record.metadata.question_type.value
                            if record.metadata.question_type else None
                        ),
                        "document_type": record.metadata.document_type,
                        # Task 3: long-chunk provenance for Deep-mode neighbor
                        # pull-in (omitted when the parent hit directly)
                        **({"chunk_ids": list(g["chunk_ids"])} if g["chunk_ids"] else {}),
                    },
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    rrf_score=rrf_score,
                    rerank_score=g["score"] if self.use_reranker else None,
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
        from src.utils.atomic_io import dump_json_atomic, write_bytes_atomic
        import orjson

        dump_json_atomic(
            path / "pipeline_metadata.json",
            {"use_chunking": self.use_chunking, "embedding_dim": self._embedding_dim},
        )
        write_bytes_atomic(
            path / "doc_map.json",
            orjson.dumps({
                doc_id: record.model_dump(mode="json")
                for doc_id, record in self._doc_map.items()
            }),
        )
        write_bytes_atomic(
            path / "chunk_map.json",
            orjson.dumps({
                chunk_id: chunk.model_dump(mode="json")
                for chunk_id, chunk in self._chunk_map.items()
            }),
        )
        write_bytes_atomic(
            path / "long_chunk_map.json",
            orjson.dumps({
                chunk_id: chunk.model_dump(mode="json")
                for chunk_id, chunk in self._long_chunk_map.items()
            }),
        )
        self._write_build_meta(path)

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

        # P1.3 — restore long-doc chunks (missing file = older index, empty map)
        self._long_chunk_map = {}
        self._long_chunk_texts = {}
        if (path / "long_chunk_map.json").exists():
            with open(path / "long_chunk_map.json", "rb") as f:
                long_data = orjson.loads(f.read())
            self._long_chunk_map = {
                chunk_id: QAChunk.model_validate(data)
                for chunk_id, data in long_data.items()
            }
            self._long_chunk_texts = {
                chunk_id: c.chunk_text for chunk_id, c in self._long_chunk_map.items()
            }

        self._warn_if_build_meta_mismatch(path)

        print(f"✓ Loaded Hybrid RAG pipeline (use_chunking={self.use_chunking}) from {path}")

    def _write_build_meta(self, path: Path) -> None:
        """Fingerprint this index so HPC can reject embed-model / dim mismatch."""
        import hashlib
        from datetime import datetime, timezone

        hasher = hashlib.sha256()
        for doc_id in sorted(self._doc_map.keys()):
            rec = self._doc_map[doc_id]
            hasher.update(doc_id.encode("utf-8", errors="replace"))
            hasher.update(b"\0")
            hasher.update((rec.question_text or "").encode("utf-8", errors="replace"))
            hasher.update(b"\0")
            hasher.update((rec.answer_text or "")[:512].encode("utf-8", errors="replace"))
            hasher.update(b"\n")

        meta = {
            "embed_model": getattr(self.embedder, "model_name", None),
            "embed_dim": self._embedding_dim,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": len(self._doc_map),
            "long_chunk_count": len(self._long_chunk_map),
            "chunk_count": len(self._chunk_map),
            "use_chunking": self.use_chunking,
            "fusion_top_k": self.fusion_top_k,
            "rows_sha256": hasher.hexdigest(),
        }
        from src.utils.atomic_io import dump_json_atomic

        dump_json_atomic(path / "build_meta.json", meta, indent=2)

    def _warn_if_build_meta_mismatch(self, path: Path) -> None:
        meta_path = path / "build_meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        live_dim = getattr(self.embedder, "embedding_dim", self._embedding_dim)
        saved_dim = meta.get("embed_dim")
        if saved_dim is not None and live_dim is not None and int(saved_dim) != int(live_dim):
            print(
                f"[build_meta] WARNING: index embed_dim={saved_dim} "
                f"but live embedder dim={live_dim}. Rebuild the index."
            )
        saved_model = meta.get("embed_model")
        live_model = getattr(self.embedder, "model_name", None)
        if saved_model and live_model and str(saved_model) != str(live_model):
            print(
                f"[build_meta] WARNING: index built with {saved_model!r} "
                f"but live embedder is {live_model!r}."
            )

    def __len__(self) -> int:
        """Return the number of parent indexed documents."""
        return len(self._doc_map)

    def __repr__(self) -> str:
        return (
            f"HybridRAGPipeline(n_docs={len(self)}, "
            f"use_chunking={self.use_chunking}, "
            f"use_reranker={self.use_reranker})"
        )
