"""
Evaluation report and visualization generator for Phase 4.
Saves comparison charts and writes an automated Markdown summary report.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


class EvaluationReporter:
    """
    Renders evaluation data into formatted reports (Markdown) and visualizations (PNG charts).
    """

    def __init__(self, output_dir: str = "evaluation_results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_charts(self, results: dict[str, Any]) -> None:
        """Generate Recall, MRR, and Latency charts using Matplotlib."""
        summary = results["summary"]
        systems = list(summary.keys())
        
        # ── 1. Recall@K Comparison Chart ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r1 = [summary[sys]["recall_at_1"] * 100 for sys in systems]
        r3 = [summary[sys]["recall_at_3"] * 100 for sys in systems]
        r5 = [summary[sys]["recall_at_5"] * 100 for sys in systems]
        r10 = [summary[sys]["recall_at_10"] * 100 for sys in systems]
        
        x = np.arange(len(systems))
        width = 0.18
        
        ax.bar(x - 1.5*width, r1, width, label="Recall@1", color="#1f77b4")
        ax.bar(x - 0.5*width, r3, width, label="Recall@3", color="#aec7e8")
        ax.bar(x + 0.5*width, r5, width, label="Recall@5", color="#ff7f0e")
        ax.bar(x + 1.5*width, r10, width, label="Recall@10", color="#ffbb78")
        
        ax.set_ylabel("Recall Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_title("Retrieval Pipeline Accuracy: Recall@K Comparison", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([sys.upper() for sys in systems], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10, loc="upper left")
        
        plt.tight_layout()
        chart_path = self.output_dir / "recall_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

        # ── 2. MRR & nDCG Comparison Chart ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        mrr = [summary[sys]["mrr"] for sys in systems]
        ndcg = [summary[sys]["ndcg_at_5"] for sys in systems]
        
        x = np.arange(len(systems))
        width = 0.35
        
        ax.bar(x - width/2, mrr, width, label="MRR", color="#2ca02c")
        ax.bar(x + width/2, ndcg, width, label="nDCG@5", color="#98df8a")
        
        ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight="bold")
        ax.set_title("Mean Reciprocal Rank (MRR) & nDCG@5 Comparison", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([sys.upper() for sys in systems], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        chart_path = self.output_dir / "mrr_ndcg_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

        # ── 3. Latency Comparison Chart ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_lats = [summary[sys]["mean_latency_ms"] for sys in systems]
        p95_lats = [summary[sys]["p95_latency_ms"] for sys in systems]
        
        x = np.arange(len(systems))
        width = 0.35
        
        ax.bar(x - width/2, mean_lats, width, label="Mean Latency", color="#d62728")
        ax.bar(x + width/2, p95_lats, width, label="95th % Latency", color="#ff9896")
        
        ax.set_ylabel("Latency (milliseconds)", fontsize=12, fontweight="bold")
        ax.set_title("Mean vs 95th Percentile Retrieval Latency", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([sys.upper() for sys in systems], fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        chart_path = self.output_dir / "latency_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

    def generate_markdown_report(self, results: dict[str, Any]) -> str:
        """Compile a highly thorough Markdown report summarizing metrics and failures."""
        summary = results["summary"]
        failures = results["failures"]
        n_queries = results["total_queries"]

        parts = [
            "# Parliamentary RAG - Retrieval Pipeline Evaluation Report",
            f"**Execution Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Total Evaluation Queries**: {n_queries} unique representative queries",
            "",
            "## 1. Executive Summary & Overall Metrics",
            "This report documents the scientific evaluation of the four retrieval pipeline iterations. The metric evaluations benchmark performance independently under identical query parameters.",
            "",
            "| Metric | BM25 Only | Dense (FAISS) | Hybrid (RRF) | Hybrid + Cross-Encoder |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]

        # Populate Overall Metrics Table rows
        m_rows = [
            ("Recall@1 (Hit Rate@1)", "recall_at_1", "{:.1%}"),
            ("Recall@3", "recall_at_3", "{:.1%}"),
            ("Recall@5 (Hit Rate@5)", "recall_at_5", "{:.1%}"),
            ("Recall@10", "recall_at_10", "{:.1%}"),
            ("Mean Reciprocal Rank (MRR)", "mrr", "{:.4f}"),
            ("nDCG@5", "ndcg_at_5", "{:.4f}"),
            ("nDCG@10", "ndcg_at_10", "{:.4f}"),
            ("Average Rank", "average_rank", "{:.2f}"),
            ("Mean Latency (ms)", "mean_latency_ms", "{:.2f} ms"),
            ("95th Percentile Latency (ms)", "p95_latency_ms", "{:.2f} ms"),
        ]

        for label, key, fmt in m_rows:
            bm25_v = fmt.format(summary["bm25"][key])
            dense_v = fmt.format(summary["dense"][key])
            hybrid_v = fmt.format(summary["hybrid"][key])
            hybrid_ce_v = fmt.format(summary["hybrid_ce"][key])
            parts.append(f"| {label} | {bm25_v} | {dense_v} | {hybrid_v} | {hybrid_ce_v} |")

        parts.extend([
            "",
            "## 2. Topic Category Breakdown (Recall@5)",
            "Performance breakdown by topic domain category across each retrieval system.",
            "",
            "| Topic Category | BM25 Only | Dense (FAISS) | Hybrid (RRF) | Hybrid + Cross-Encoder |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ])

        # Category recall breakdown
        categories = list(summary["hybrid_ce"]["category_recall_at_5"].keys())
        for cat in categories:
            bm25_cat = f"{summary['bm25']['category_recall_at_5'].get(cat, 0.0):.1%}"
            dense_cat = f"{summary['dense']['category_recall_at_5'].get(cat, 0.0):.1%}"
            hybrid_cat = f"{summary['hybrid']['category_recall_at_5'].get(cat, 0.0):.1%}"
            ce_cat = f"{summary['hybrid_ce']['category_recall_at_5'].get(cat, 0.0):.1%}"
            parts.append(f"| {cat} | {bm25_cat} | {dense_cat} | {hybrid_cat} | {ce_cat} |")

        parts.extend([
            "",
            "## 3. Visualizations",
            "The following comparative charts were automatically generated and saved to the `evaluation_results/` folder:",
            "* **Recall Comparison**: `recall_comparison.png`",
            "* **MRR & nDCG Metrics**: `mrr_ndcg_comparison.png`",
            "* **Latency Profile**: `latency_comparison.png`",
            "",
            "## 4. Failure Analysis",
            f"Of the {n_queries} queries, **{len(failures)}** failed to retrieve their expected targets in the top-5 final ranks under the complete Hybrid + Cross-Encoder pipeline.",
            ""
        ])

        if failures:
            parts.extend([
                "| Failed Query | Expected ID | Top 1 Retrieved | Failure Stage | Possible Cause |",
                "| :--- | :---: | :---: | :---: | :--- |"
            ])
            for f in failures:
                ret_top = f["retrieved"][0] if f["retrieved"] else "None"
                parts.append(f"| {f['query']} | `{f['expected_id']}` | `{ret_top}` | {f['stage']} | {f['possible_reason']} |")
        else:
            parts.append("*✓ Perfect Score! All benchmark queries successfully retrieved their expected document in the top-5 final ranks.*")

        report_md = "\n".join(parts)
        
        # Save to markdown file
        with open(self.output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)
            
        return report_md


# Import datetime locally for the report header
from datetime import datetime
