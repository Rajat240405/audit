"""
Evaluation runner for Phase 4 benchmarking.
Executes benchmark queries against 4 retrieval configurations:
1. BM25 Only
2. Dense Only
3. Hybrid RAG (No reranker)
4. Hybrid RAG + Cross Encoder
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.evaluation.metrics import RetrievalMetrics


class EvaluationRunner:
    """
    Orchestrates the evaluation of a benchmark JSON file against the indexed corpus.
    """

    def __init__(self, index_dir: str = "storage/hybrid_rag") -> None:
        self.index_dir = Path(index_dir)
        self.pipeline = HybridRAGPipeline()
        self.pipeline.load(self.index_dir)

    def load_benchmark(self, path: str | Path) -> list[dict[str, Any]]:
        """Load benchmark queries from JSON."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def run_eval(self, benchmark_queries: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Run the evaluation over the specified benchmark query list.
        """
        results_by_system: dict[str, list[dict[str, Any]]] = {
            "bm25": [],
            "dense": [],
            "hybrid": [],
            "hybrid_ce": []
        }

        failures: list[dict[str, Any]] = []

        for q_idx, item in enumerate(benchmark_queries, start=1):
            query = item["query"]
            expected_id = item["expected_doc_id"]
            category = item["category"]
            difficulty = item["difficulty"]

            # ── 1. BM25 Only Run ──────────────────────────────────────────────
            t0 = time.perf_counter()
            bm25_res = self.pipeline.bm25_index.search(query, k=10)
            bm25_lat = (time.perf_counter() - t0) * 1000
            bm25_ids = [doc_id for doc_id, _ in bm25_res]
            
            results_by_system["bm25"].append({
                "query": query,
                "expected": expected_id,
                "retrieved": bm25_ids,
                "scores": [score for _, score in bm25_res],
                "latency_ms": bm25_lat,
                "category": category,
                "difficulty": difficulty
            })

            # ── 2. Dense Only Run ─────────────────────────────────────────────
            t0 = time.perf_counter()
            query_emb = self.pipeline.embedder.embed(query)
            embed_lat = (time.perf_counter() - t0) * 1000
            
            t0 = time.perf_counter()
            dense_res = self.pipeline.vector_store.search(query_emb, k=10)
            dense_lat = (time.perf_counter() - t0) * 1000
            dense_ids = [doc_id for doc_id, _ in dense_res]
            
            results_by_system["dense"].append({
                "query": query,
                "expected": expected_id,
                "retrieved": dense_ids,
                "scores": [score for _, score in dense_res],
                "latency_ms": embed_lat + dense_lat,
                "embed_latency_ms": embed_lat,
                "search_latency_ms": dense_lat,
                "category": category,
                "difficulty": difficulty
            })

            # ── 3. Hybrid Only Run ────────────────────────────────────────────
            self.pipeline.use_reranker = False
            t0 = time.perf_counter()
            hybrid_res, timings = self.pipeline.retrieve(query, top_k=10)
            hybrid_lat = (time.perf_counter() - t0) * 1000
            hybrid_ids = [r.doc_id for r in hybrid_res]
            
            results_by_system["hybrid"].append({
                "query": query,
                "expected": expected_id,
                "retrieved": hybrid_ids,
                "scores": [r.score for r in hybrid_res],
                "latency_ms": hybrid_lat,
                "timings": timings.as_dict(),
                "category": category,
                "difficulty": difficulty
            })

            # ── 4. Hybrid + Cross-Encoder Run ─────────────────────────────────
            self.pipeline.use_reranker = True
            t0 = time.perf_counter()
            ce_res, ce_timings = self.pipeline.retrieve(query, top_k=10)
            ce_lat = (time.perf_counter() - t0) * 1000
            ce_ids = [r.doc_id for r in ce_res]
            
            results_by_system["hybrid_ce"].append({
                "query": query,
                "expected": expected_id,
                "retrieved": ce_ids,
                "scores": [r.score for r in ce_res],
                "latency_ms": ce_lat,
                "timings": ce_timings.as_dict(),
                "category": category,
                "difficulty": difficulty
            })

            # ── Check for Failures in the Full System (Hybrid + CE) ───────────
            if expected_id not in ce_ids[:5]:
                # Failure details
                stage_fail = "dense+bm25"
                if expected_id in bm25_ids and expected_id not in ce_ids:
                    stage_fail = "reranker"
                elif expected_id not in bm25_ids and expected_id not in dense_ids:
                    stage_fail = "retrieval_retrieved_none"
                
                failures.append({
                    "query": query,
                    "expected_id": expected_id,
                    "expected_doc": self.pipeline._doc_texts.get(expected_id, "Document not found in pipeline index cache."),
                    "retrieved": ce_ids[:5],
                    "scores": [r.score for r in ce_res[:5]],
                    "stage": stage_fail,
                    "possible_reason": (
                        "Cross-encoder demoted target document." if stage_fail == "reranker"
                        else "Expected document was not retrieved in first-stage (Dense & BM25 both missed)."
                    )
                })

        # Compile System-wide Summary Metrics
        summary = self._compile_metrics(results_by_system)
        
        return {
            "summary": summary,
            "raw_results": results_by_system,
            "failures": failures,
            "total_queries": len(benchmark_queries)
        }

    def _compile_metrics(self, results_by_system: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Compile complete statistics and percentages across all runs."""
        compiled = {}
        for system, runs in results_by_system.items():
            n = len(runs)
            if n == 0:
                continue

            r1 = sum(RetrievalMetrics.compute_recall(r["retrieved"], r["expected"], 1) for r in runs) / n
            r3 = sum(RetrievalMetrics.compute_recall(r["retrieved"], r["expected"], 3) for r in runs) / n
            r5 = sum(RetrievalMetrics.compute_recall(r["retrieved"], r["expected"], 5) for r in runs) / n
            r10 = sum(RetrievalMetrics.compute_recall(r["retrieved"], r["expected"], 10) for r in runs) / n
            
            mrr = sum(RetrievalMetrics.compute_mrr(r["retrieved"], r["expected"]) for r in runs) / n
            ndcg5 = sum(RetrievalMetrics.compute_ndcg(r["retrieved"], r["expected"], 5) for r in runs) / n
            ndcg10 = sum(RetrievalMetrics.compute_ndcg(r["retrieved"], r["expected"], 10) for r in runs) / n
            
            avg_rank = sum(RetrievalMetrics.compute_average_rank(r["retrieved"], r["expected"]) for r in runs) / n
            
            latencies = [r["latency_ms"] for r in runs]
            mean_lat = sum(latencies) / n
            sorted_lat = sorted(latencies)
            p95_lat = sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[-1]

            # Categorized analysis
            categories = {}
            for r in runs:
                cat = r["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(RetrievalMetrics.compute_recall(r["retrieved"], r["expected"], 5))
            
            category_metrics = {
                cat: sum(scores) / len(scores) for cat, scores in categories.items()
            }

            compiled[system] = {
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr": mrr,
                "ndcg_at_5": ndcg5,
                "ndcg_at_10": ndcg10,
                "average_rank": avg_rank,
                "mean_latency_ms": mean_lat,
                "p95_latency_ms": p95_lat,
                "category_recall_at_5": category_metrics
            }

        return compiled
