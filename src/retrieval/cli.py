"""
Phase 2 CLI — Hybrid RAG query interface.

Usage:
    python -m src.retrieval.cli build              # Build indices
    python -m src.retrieval.cli query "question"   # Query the system
    python -m src.retrieval.cli interactive        # Interactive mode
"""

from __future__ import annotations

import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.data.loader import DataLoader
from src.generation.client import LLMClient
from src.generation.generator import AnswerGenerator
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.utils.project_scope import resolve_effective_ministry_filter, filter_records_by_ministry

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load the pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Files written by HybridRAGPipeline.save() that constitute a complete,
# loadable index. Used to distinguish "a real index exists" from "the
# directory exists but is empty/partial" (e.g. after a manual cleanup).
_INDEX_MARKER_FILES = (
    "pipeline_metadata.json",
    "doc_map.json",
    "vector_store.index",
    "vector_store.ids",
    "bm25_index.pkl",
    "bm25_index.json",
)


def _index_is_built(index_path: Path) -> bool:
    """Return True only if a complete, loadable pipeline was saved at index_path.

    Merely checking ``index_path.exists()`` is not enough: the directory can
    exist while being empty or partially populated (e.g. ``storage/hybrid_rag``
    created but the index files deleted). A clean build must not attempt to
    ``load()`` in that state.
    """
    if not index_path.exists() or not index_path.is_dir():
        return False
    return all((index_path / name).exists() for name in _INDEX_MARKER_FILES)


def get_pipeline(
    data_file: str | None = None,
    index_dir: str = "storage/hybrid_rag",
    force_rebuild: bool = False,
    ministry_filter: str | None = None,
    all_ministries: bool = False,
    require_index: bool = False,
) -> HybridRAGPipeline:
    """Load or build the Hybrid RAG pipeline.

    Automatically respects the project_scope configuration from ingestion.yaml
    unless an explicit override is provided.

    Parameters
    ----------
    require_index : bool
        If True, raise ``FileNotFoundError`` when no complete index exists
        (used by query-like commands: ``query``, ``interactive``, ``benchmark``).
        If False (used by ``build``), build the index from scratch when it is
        missing or empty.
    """
    index_path = Path(index_dir)
    data_path = Path(data_file) if data_file else None

    # Find latest data file if no explicit path is given.
    # Priority: enriched (if available) → processed (current Phase 1 output).
    # The processed/ dir is the canonical corpus produced by `ingest`; the
    # legacy enriched/ dir is gitignored and may not exist on fresh clones.
    if not data_path:
        for subdir in ("enriched", "processed"):
            candidates = sorted(Path("data", subdir).glob("*.jsonl"), reverse=True)
            if candidates:
                data_path = candidates[0]
                break
        if not data_path:
            raise FileNotFoundError(
                "No Phase 1 data found under data/enriched or data/processed. "
                "Run `ingest` first or pass --data <path>."
            )

    # Load from disk only if a complete index was previously saved
    # (and not forcing a rebuild).
    if not force_rebuild and _index_is_built(index_path):
        console.print(f"[cyan]Loading pipeline from {index_path}...[/cyan]")
        pipeline = HybridRAGPipeline()
        pipeline.load(index_path)
        return pipeline

    # Query-like commands must not silently build an index: they require one.
    if require_index:
        raise FileNotFoundError(
            f"No complete Hybrid RAG index found at {index_path}. "
            "Run `retrieve build` first."
        )

    # Build from scratch
    console.print(f"[cyan]Loading records from {data_path}...[/cyan]")
    records = DataLoader.load_jsonl(data_path)

    # ── Phase 12+ MoES Scope (shared utility) ──
    effective_filter = resolve_effective_ministry_filter(
        explicit_filter=ministry_filter,
        all_ministries=all_ministries,
    )

    if effective_filter:
        original_count = len(records)
        records = filter_records_by_ministry(records, effective_filter)
        console.print(f"[cyan]Ministry filter applied ({effective_filter}): {original_count:,} → {len(records):,} records[/cyan]")

    console.print(f"[green]Loaded {len(records):,} records.[/green]")

    console.print("[cyan]Building Hybrid RAG indices...[/cyan]")
    pipeline = HybridRAGPipeline(records=records)
    pipeline.save(index_path)

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Phase 2 — Hybrid RAG Query Interface."""
    pass


@cli.command()
@click.option("--data", "data_file", type=str, default=None, help="Path to enriched Q&A JSONL")
@click.option("--output", "index_dir", type=str, default="storage/hybrid_rag", help="Index output dir")
@click.option("--rebuild", is_flag=True, help="Force rebuild even if index exists")
@click.option("--ministry-filter", type=str, default=None, help="Explicit ministry filter")
@click.option("--all-ministries", is_flag=True, default=False, help="Index ALL ministries (override MoES default)")
def build(data_file: str, index_dir: str, rebuild: bool, ministry_filter: str | None, all_ministries: bool) -> None:
    """Build the Hybrid RAG indices from the knowledge base."""
    console.print(Panel.fit(
        "[bold cyan]Phase 2 — Building Hybrid RAG Index[/bold cyan]",
        border_style="cyan",
    ))
    pipeline = get_pipeline(
        data_file, index_dir, force_rebuild=rebuild,
        ministry_filter=ministry_filter, all_ministries=all_ministries
    )
    console.print(f"\n[bold green]✓ Build complete:[/bold green] {len(pipeline):,} documents indexed")


@cli.command()
@click.argument("question", type=str)
@click.option("--top-k", type=int, default=5, help="Number of results to retrieve")
@click.option("--no-rerank", is_flag=True, help="Skip cross-encoder reranking")
@click.option("--no-generate", is_flag=True, help="Skip LLM generation (retrieval only)")
@click.option("--show-prompt", is_flag=True, help="Show the full LLM prompt")
@click.option("--show-trace", is_flag=True, help="Show retrieval trace")
@click.option("--llm-model", type=str, default="qwen3:8b", help="LLM model name")
def query(
    question: str,
    top_k: int,
    no_rerank: bool,
    no_generate: bool,
    show_prompt: bool,
    show_trace: bool,
    llm_model: str,
) -> None:
    """
    Query the Hybrid RAG system with a question.

    Runs the full pipeline: retrieval → generation.
    """
    # Load pipeline (query commands require an existing, complete index)
    try:
        pipeline = get_pipeline(require_index=True)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' first: python -m src.retrieval.cli build")
        return

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    # ── Retrieval ──────────────────────────────────────────────────────────
    t_retrieval = time.monotonic()
    if no_rerank:
        pipeline.use_reranker = False
    results, timings = pipeline.retrieve(question, top_k=top_k)
    retrieval_ms = (time.monotonic() - t_retrieval) * 1000

    # Display retrieval results
    if not results:
        console.print("[yellow]No results retrieved.[/yellow]")
        return

    table = Table(title="Retrieved Q&A Records", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Doc ID", style="cyan")
    table.add_column("Ministry", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Question (excerpt)", max_width=60)

    for i, result in enumerate(results, start=1):
        q_preview = result.question[:57] + "..." if len(result.question) > 57 else result.question
        score_str = f"{result.score:.4f}"
        if result.dense_score is not None:
            score_str += f" (dense: {result.dense_score:.3f})"
        table.add_row(
            str(i),
            result.doc_id,
            result.metadata.get("ministry", "-") or "-",
            score_str,
            q_preview,
        )

    console.print(table)

    if show_trace:
        console.print("\n[bold]Retrieval Timings:[/bold]")
        for stage, ms in timings.as_dict().items():
            console.print(f"  {stage:<25} {ms:>8.2f}ms")

    console.print(f"\n[dim]Retrieval: {retrieval_ms:.0f}ms total[/dim]")

    # ── Generation ─────────────────────────────────────────────────────────
    if no_generate:
        return

    # Check if LLM is available
    llm_client = LLMClient(model=llm_model)
    if not llm_client.check_health():
        console.print("\n[yellow]⚠ LLM not available.[/yellow]")
        console.print("  Start Ollama:  ollama serve")
        console.print(f"  Pull model:    ollama pull {llm_model}")
        console.print("\n  Showing top retrieved document instead:\n")
        console.print(Panel(
            f"[bold]Question:[/bold] {results[0].question}\n\n"
            f"[bold]Answer:[/bold] {results[0].answer[:500]}...",
            title=f"[cyan]Top Result: {results[0].doc_id}[/cyan]",
        ))
        return

    console.print(f"\n[cyan]Generating answer with {llm_model}...[/cyan]")
    generator = AnswerGenerator(llm_client=llm_client)
    gen_result = generator.generate(question, results)

    console.print(Panel(
        gen_result.answer,
        title="[bold green]Generated Answer[/bold green]",
        border_style="green",
    ))

    console.print(
        f"[dim]Model: {gen_result.model} | "
        f"Tokens: {gen_result.total_tokens} | "
        f"Latency: {gen_result.generation_latency_ms:.0f}ms | "
        f"Sources: {', '.join(gen_result.sources_used)}[/dim]"
    )

    if show_prompt:
        console.print("\n[bold]Full Prompt:[/bold]")
        syntax = Syntax(gen_result.prompt[:2000] + "...", "markdown", theme="monokai")
        console.print(syntax)


@cli.command()
@click.option("--data", "data_file", type=str, default=None)
@click.option("--top-k", type=int, default=5)
def interactive(data_file: str, top_k: int) -> None:
    """Run in interactive mode — ask questions repeatedly."""
    pipeline = get_pipeline(data_file=data_file, require_index=True)
    llm_client = LLMClient()
    generator = AnswerGenerator(llm_client=llm_client)

    console.print(Panel.fit(
        "[bold cyan]Hybrid RAG Interactive Mode[/bold cyan]\n"
        "Type your question and press Enter.\n"
        "Type 'quit' or 'exit' to stop.\n"
        "Use '--no-generate' flag to skip LLM (retrieval only).",
        border_style="cyan",
    ))

    while True:
        try:
            question = console.input("\n[bold]You:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        results, timings = pipeline.retrieve(question, top_k=top_k)
        console.print(f"[dim]→ Retrieved {len(results)} results in {timings.total_ms:.0f}ms[/dim]")

        if results:
            top = results[0]
            console.print(f"[cyan]  #{top.doc_id}[/cyan] ({top.score:.3f}): {top.question[:80]}...")

        if llm_client.check_health():
            gen_result = generator.generate(question, results)
            console.print(f"[green]Answer:[/green] {gen_result.answer[:300]}...")


@cli.command()
def benchmark() -> None:
    """Benchmark retrieval latency on a set of test queries."""
    test_queries = [
        "What measures address malaria in rural areas?",
        "Status of GST collection and reforms",
        "Government steps for skill development",
        "Healthcare infrastructure in tribal areas",
        "Renewable energy targets and progress",
        "Status of metro rail projects in cities",
        "Rural employment guarantee scheme implementation",
        "Digital education initiatives in schools",
        "Water quality and drinking water coverage",
        "Road safety measures and accident data",
    ]

    pipeline = get_pipeline(require_index=True)

    console.print(Panel.fit(
        f"[bold]Hybrid RAG Benchmark[/bold]\n"
        f"{len(test_queries)} queries | {len(pipeline):,} indexed docs",
        border_style="cyan",
    ))

    timings_list: list[float] = []

    for i, query_text in enumerate(test_queries, start=1):
        t0 = time.monotonic()
        results, timings = pipeline.retrieve(query_text, top_k=5)
        elapsed_ms = (time.monotonic() - t0) * 1000
        timings_list.append(elapsed_ms)
        console.print(
            f"  {i:>2}. [{elapsed_ms:>6.0f}ms] "
            f"{len(results)} results | "
            f"Q: {query_text[:50]}..."
        )

    import statistics
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Mean latency:  {statistics.mean(timings_list):.1f}ms")
    console.print(f"  Median latency: {statistics.median(timings_list):.1f}ms")
    console.print(f"  Min latency:    {min(timings_list):.1f}ms")
    console.print(f"  Max latency:    {max(timings_list):.1f}ms")


if __name__ == "__main__":
    cli()
