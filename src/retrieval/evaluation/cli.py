"""
Phase 4 Evaluation CLI Interface.
Commands to run, compare, report, and analyze retrieval failures.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.retrieval.evaluation.runner import EvaluationRunner
from src.retrieval.evaluation.reporter import EvaluationReporter

console = Console()


@click.group()
def evaluate_cli() -> None:
    """Phase 4 — Retrieval Evaluation & Benchmarking CLI."""
    pass


@evaluate_cli.command()
@click.option(
    "--benchmark", "-b",
    default="benchmarks/default.json",
    type=str,
    help="Path to the benchmark JSON file.",
)
@click.option(
    "--output-dir", "-o",
    default="evaluation_results",
    type=str,
    help="Directory to save evaluation results.",
)
def run(benchmark: str, output_dir: str) -> None:
    """Execute the full evaluation across all four retrieval configs."""
    console.print(Panel.fit(
        "[bold cyan]Phase 4 — Executing Retrieval Benchmarks[/bold cyan]",
        border_style="cyan"
    ))

    runner = EvaluationRunner()
    
    try:
        queries = runner.load_benchmark(benchmark)
    except FileNotFoundError:
        console.print(f"[red]Error: Benchmark file not found at {benchmark}.[/red]")
        return

    console.print(f"Loaded [cyan]{len(queries):,}[/cyan] representative evaluation queries.")
    console.print("[cyan]Running metrics across BM25, Dense, Hybrid, and Cross-Encoder...[/cyan]")

    # Run Eval
    results = runner.run_eval(queries)
    
    # Save Report & Charts
    reporter = EvaluationReporter(output_dir=output_dir)
    reporter.generate_charts(results)
    report_md = reporter.generate_markdown_report(results)

    console.print("\n[bold green]✓ Benchmarks completed successfully![/bold green]")
    console.print(f"  Visualizations saved to: [cyan]{output_dir}/[/cyan]")
    console.print(f"  Markdown report saved to: [cyan]{output_dir}/evaluation_report.md[/cyan]")

    # Print quick metric comparison table
    _print_comparison_table(results["summary"])


@evaluate_cli.command()
@click.option(
    "--results-dir", "-d",
    default="evaluation_results",
    type=str,
)
def report(results_dir: str) -> None:
    """Read and display the latest generated evaluation markdown report."""
    report_path = f"{results_dir}/evaluation_report.md"
    try:
        with open(report_path, encoding="utf-8") as f:
            console.print(f.read())
    except FileNotFoundError:
        console.print(f"[red]Error: Report file not found at {report_path}. Run 'run' first.[/red]")


@evaluate_cli.command()
@click.option(
    "--benchmark", "-b",
    default="benchmarks/default.json",
    type=str,
)
def compare(benchmark: str) -> None:
    """Directly compare retrieval scores and outputs for quick verification."""
    runner = EvaluationRunner()
    try:
        queries = runner.load_benchmark(benchmark)
    except FileNotFoundError:
        console.print(f"[red]Error: Benchmark file not found at {benchmark}.[/red]")
        return

    results = runner.run_eval(queries)
    _print_comparison_table(results["summary"])


@evaluate_cli.command()
@click.option(
    "--results-dir", "-d",
    default="evaluation_results",
    type=str,
)
def failures(results_dir: str) -> None:
    """Print the detailed failure analysis of retrieval misses."""
    report_path = f"{results_dir}/evaluation_report.md"
    if not os.path.exists(report_path):
        console.print("[yellow]No report found. Running evaluations first...[/yellow]")
        return

    # Check failures in detail from a fresh run
    runner = EvaluationRunner()
    # Find default benchmark
    queries = runner.load_benchmark("benchmarks/default.json")
    results = runner.run_eval(queries)
    fails = results["failures"]

    if not fails:
        console.print("[bold green]✓ Awesome! 0 failures recorded under the complete pipeline![/bold green]")
        return

    table = Table(title="Failed Retrievals (Target Not in Top-5 Final)", show_lines=True)
    table.add_column("Query", style="yellow")
    table.add_column("Expected Doc ID", style="cyan")
    table.add_column("Failure Stage", style="red")
    table.add_column("Possible Reason")

    for f in fails:
        table.add_row(f["query"], f["expected_id"], f["stage"], f["possible_reason"])

    console.print(table)


def _print_comparison_table(summary: dict) -> None:
    """Print standard comparison table directly to console."""
    table = Table(title="Retrieval Pipeline Comparative Performance", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("BM25 Only", justify="center")
    table.add_column("Dense (FAISS)", justify="center")
    table.add_column("Hybrid (RRF)", justify="center")
    table.add_column("Hybrid + CE", justify="center")

    metrics_rows = [
        ("Recall@1 (Hit Rate@1)", "recall_at_1", "{:.1%}"),
        ("Recall@3", "recall_at_3", "{:.1%}"),
        ("Recall@5 (Hit Rate@5)", "recall_at_5", "{:.1%}"),
        ("Recall@10", "recall_at_10", "{:.1%}"),
        ("MRR", "mrr", "{:.4f}"),
        ("nDCG@5", "ndcg_at_5", "{:.4f}"),
        ("Average Rank", "average_rank", "{:.2f}"),
        ("Mean Latency", "mean_latency_ms", "{:.2f} ms"),
    ]

    for label, key, fmt in metrics_rows:
        table.add_row(
            label,
            fmt.format(summary["bm25"][key]),
            fmt.format(summary["dense"][key]),
            fmt.format(summary["hybrid"][key]),
            fmt.format(summary["hybrid_ce"][key])
        )

    console.print("\n", table, "\n")


import os

if __name__ == "__main__":
    evaluate_cli()
