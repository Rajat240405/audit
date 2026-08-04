"""
Retrieval Pipeline Stage-by-Stage Diagnostic Tool

This script executes a query and intercepts the output of each pipeline sub-stage:
1. Dense Vector (FAISS) Top-K
2. Lexical (BM25) Top-K
3. Reciprocal Rank Fusion (RRF) Top-K
4. Cross-Encoder Reranked Top-K
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.retrieval.hybrid.pipeline import HybridRAGPipeline

console = Console()


def diagnose(query: str, top_k: int = 5):
    console.print(f"\n[bold cyan]=== RUNNING DIAGNOSTIC FOR QUERY: '{query}' ===[/bold cyan]\n")

    # Load Pipeline
    pipeline = HybridRAGPipeline()
    try:
        pipeline.load("storage/hybrid_rag")
    except FileNotFoundError:
        console.print("[red]Error: Index not built. Run 'retrieve build' first.[/red]")
        sys.exit(1)

    # ── Stage 1: Dense search ───────────────────────────────────────────
    query_embedding = pipeline.embedder.embed(query)
    dense_raw = pipeline.vector_store.search(query_embedding, k=pipeline.dense_top_k)

    table_dense = Table(title="Stage 1: Dense Vector (FAISS) Top Candidates", show_lines=True)
    table_dense.add_column("Rank", justify="center")
    table_dense.add_column("Doc ID", style="cyan")
    table_dense.add_column("Score", justify="right")
    table_dense.add_column("Question Excerpt", max_width=70)

    for rank, (doc_id, score) in enumerate(dense_raw[:10], start=1):
        record = pipeline._doc_map.get(doc_id)
        q_text = record.question_text[:65] + "..." if record else ""
        table_dense.add_row(str(rank), doc_id, f"{score:.4f}", q_text)

    console.print(table_dense)
    console.print()

    # ── Stage 2: BM25 search ───────────────────────────────────────────
    bm25_raw = pipeline.bm25_index.search(query, k=pipeline.dense_top_k)

    table_bm25 = Table(title="Stage 2: Lexical (BM25) Top Candidates", show_lines=True)
    table_bm25.add_column("Rank", justify="center")
    table_bm25.add_column("Doc ID", style="cyan")
    table_bm25.add_column("Score", justify="right")
    table_bm25.add_column("Question Excerpt", max_width=70)

    for rank, (doc_id, score) in enumerate(bm25_raw[:10], start=1):
        record = pipeline._doc_map.get(doc_id)
        q_text = record.question_text[:65] + "..." if record else ""
        table_bm25.add_row(str(rank), doc_id, f"{score:.4f}", q_text)

    console.print(table_bm25)
    console.print()

    # ── Stage 3: RRF Fusion ───────────────────────────────────────────
    from src.retrieval.hybrid.fusion import RRF
    fused_raw = RRF.fuse(
        [dense_raw, bm25_raw],
        k=pipeline.rrf_k,
        top_k=pipeline.fusion_top_k,
    )

    table_rrf = Table(title="Stage 3: Reciprocal Rank Fusion (RRF) Candidates", show_lines=True)
    table_rrf.add_column("Rank", justify="center")
    table_rrf.add_column("Doc ID", style="cyan")
    table_rrf.add_column("RRF Score", justify="right")
    table_rrf.add_column("Question Excerpt", max_width=70)

    for rank, (doc_id, score) in enumerate(fused_raw[:10], start=1):
        record = pipeline._doc_map.get(doc_id)
        q_text = record.question_text[:65] + "..." if record else ""
        table_rrf.add_row(str(rank), doc_id, f"{score:.4f}", q_text)

    console.print(table_rrf)
    console.print()

    # ── Stage 4: Cross-Encoder Reranker ─────────────────────────────────
    reranked_raw = pipeline.reranker.rerank(
        query=query,
        candidates=fused_raw,
        k=top_k,
        doc_texts=pipeline._doc_texts,
    )

    table_final = Table(title="Stage 4: Cross-Encoder Reranked (Final Selected Top-K)", show_lines=True)
    table_final.add_column("Rank", justify="center")
    table_final.add_column("Doc ID", style="cyan")
    table_final.add_column("Cross-Encoder Score", justify="right")
    table_final.add_column("Question Excerpt", max_width=70)

    for rank, (doc_id, score) in enumerate(reranked_raw, start=1):
        record = pipeline._doc_map.get(doc_id)
        q_text = record.question_text[:65] + "..." if record else ""
        table_final.add_row(str(rank), doc_id, f"{score:.4f}", q_text)

    console.print(table_final)
    console.print()


if __name__ == "__main__":
    query_text = "GST collection"
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    diagnose(query_text)
