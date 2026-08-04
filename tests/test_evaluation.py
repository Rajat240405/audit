"""
Unit and integration tests for Phase 4 Retrieval Evaluation Framework.
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
import pytest

from src.retrieval.evaluation.metrics import RetrievalMetrics
from src.retrieval.evaluation.runner import EvaluationRunner
from src.retrieval.evaluation.reporter import EvaluationReporter


def test_metrics_recall():
    """Verify Recall@K calculations."""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    
    # Target in top 1 -> Recall@1 and Recall@3 should be 1.0
    assert RetrievalMetrics.compute_recall(retrieved, "doc1", k=1) == 1.0
    assert RetrievalMetrics.compute_recall(retrieved, "doc1", k=3) == 1.0

    # Target in top 3 but not top 1 -> Recall@1 is 0.0, Recall@3 is 1.0
    assert RetrievalMetrics.compute_recall(retrieved, "doc2", k=1) == 0.0
    assert RetrievalMetrics.compute_recall(retrieved, "doc2", k=3) == 1.0

    # Target not in top 4 -> Recall@5 is 0.0
    assert RetrievalMetrics.compute_recall(retrieved, "doc5", k=5) == 0.0


def test_metrics_mrr():
    """Verify MRR calculations."""
    retrieved = ["doc1", "doc2", "doc3"]
    
    # Doc 1 is rank 1 -> MRR = 1.0
    assert RetrievalMetrics.compute_mrr(retrieved, "doc1") == 1.0

    # Doc 2 is rank 2 -> MRR = 0.5
    assert RetrievalMetrics.compute_mrr(retrieved, "doc2") == 0.5

    # Doc not in list -> MRR = 0.0
    assert RetrievalMetrics.compute_mrr(retrieved, "doc4") == 0.0


def test_metrics_ndcg():
    """Verify nDCG calculations."""
    retrieved = ["doc1", "doc2", "doc3"]
    
    # Doc 1 is at rank 1 -> nDCG@5 should be 1.0
    assert abs(RetrievalMetrics.compute_ndcg(retrieved, "doc1", k=5) - 1.0) < 0.01

    # Doc 2 is at rank 2 -> nDCG@5 should be 1 / log2(3) ≈ 0.63
    ndcg_doc2 = RetrievalMetrics.compute_ndcg(retrieved, "doc2", k=5)
    assert ndcg_doc2 > 0.62 and ndcg_doc2 < 0.64


def test_reporter_creates_files(tmp_path):
    """Verify EvaluationReporter generates report and files successfully."""
    reporter = EvaluationReporter(output_dir=str(tmp_path))
    
    dummy_results = {
        "summary": {
            "bm25": {
                "recall_at_1": 0.5, "recall_at_3": 0.8, "recall_at_5": 0.9, "recall_at_10": 0.9,
                "mrr": 0.65, "ndcg_at_5": 0.7, "ndcg_at_10": 0.75, "average_rank": 2.3,
                "mean_latency_ms": 1.2, "p95_latency_ms": 2.5, "category_recall_at_5": {"Test": 0.9}
            },
            "dense": {
                "recall_at_1": 0.6, "recall_at_3": 0.85, "recall_at_5": 0.95, "recall_at_10": 0.95,
                "mrr": 0.7, "ndcg_at_5": 0.78, "ndcg_at_10": 0.8, "average_rank": 1.8,
                "mean_latency_ms": 15.1, "p95_latency_ms": 22.0, "category_recall_at_5": {"Test": 0.95}
            },
            "hybrid": {
                "recall_at_1": 0.8, "recall_at_3": 0.9, "recall_at_5": 1.0, "recall_at_10": 1.0,
                "mrr": 0.85, "ndcg_at_5": 0.88, "ndcg_at_10": 0.9, "average_rank": 1.2,
                "mean_latency_ms": 16.5, "p95_latency_ms": 24.5, "category_recall_at_5": {"Test": 1.0}
            },
            "hybrid_ce": {
                "recall_at_1": 0.9, "recall_at_3": 0.95, "recall_at_5": 1.0, "recall_at_10": 1.0,
                "mrr": 0.92, "ndcg_at_5": 0.94, "ndcg_at_10": 0.95, "average_rank": 1.1,
                "mean_latency_ms": 150.2, "p95_latency_ms": 220.5, "category_recall_at_5": {"Test": 1.0}
            }
        },
        "failures": [
            {
                "query": "Where is the failure?",
                "expected_id": "failed_doc",
                "retrieved": ["doc1", "doc2"],
                "stage": "retrieval",
                "possible_reason": "Missing index key"
            }
        ],
        "total_queries": 1
    }

    reporter.generate_charts(dummy_results)
    reporter.generate_markdown_report(dummy_results)

    assert (tmp_path / "recall_comparison.png").exists()
    assert (tmp_path / "mrr_ndcg_comparison.png").exists()
    assert (tmp_path / "latency_comparison.png").exists()
    assert (tmp_path / "evaluation_report.md").exists()
