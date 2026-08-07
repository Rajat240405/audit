"""
GraphRAG CLI — production Neo4j pipeline.

Commands
--------
graphrag build      Build/update the Neo4j graph from the enriched corpus
                    (resumable; runs a 10-document verification first).
graphrag rebuild    Drop the graph and rebuild from scratch.
graphrag stats      Show graph statistics (nodes, relationships, indexes).
graphrag query      Graph-aware query (entity expansion + vector search).

This is a completely separate CLI from the legacy ``graph`` (NetworkX)
command — nothing existing is modified.
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.graphrag.config import GraphRAGConfig
from src.graphrag.pipeline import GraphRAGPipeline
from src.graphrag.query import GraphRAGQuerier
from src.graphrag.verify import GraphVerifier

console = Console()

_GRAPHRAG_DEFAULT_ENRICHED = "data/enriched/enriched_*.jsonl"

# LLM backends selectable via --llm-provider (must match build_llm_provider in llm.py).
_LLM_PROVIDERS = ["ollama", "groq", "openai_compatible"]


@click.group()
def cli() -> None:
    """Neo4j GraphRAG — production graph pipeline (separate from Hybrid RAG)."""


def _load_config(
    enriched: str | None,
    checkpoint: str | None,
    embedding_model: str | None,
    ollama_model: str | None,
    limit: int | None,
    no_resume: bool,
    retry_failed: bool,
    llm_provider: str | None = None,
    llm_models: str | None = None,
    debug_one: str | None = None,
) -> GraphRAGConfig:
    base = GraphRAGConfig()
    return base.with_overrides(
        enriched_glob=enriched,
        checkpoint_path=checkpoint,
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        limit=limit,
        resume=not no_resume,
        retry_failed=retry_failed,
        llm_provider=llm_provider,
        llm_models=llm_models,
        debug_one=debug_one,
    )


@cli.command()
@click.option("--enriched", type=str, default=None, help="Glob/path to enriched JSONL")
@click.option("--checkpoint", type=str, default=None, help="Checkpoint file path")
@click.option("--embedding-model", type=str, default=None, help="Override embedding model (default BAAI/bge-m3)")
@click.option("--ollama-model", type=str, default=None, help="Override Ollama model (default qwen3:8b)")
@click.option(
    "--llm-provider",
    type=click.Choice(_LLM_PROVIDERS, case_sensitive=False),
    default=None,
    help="LLM backend for extraction (ollama | groq | openai_compatible; default ollama)",
)
@click.option(
    "--llm-models",
    type=str,
    default=None,
    help="Comma-separated model list for the active provider (e.g. 'm1,m2,m3')",
)
@click.option("--limit", type=int, default=None, help="Process at most N documents (testing)")
@click.option("--no-resume", is_flag=True, help="Ignore the checkpoint and reprocess everything")
@click.option("--retry-failed/--no-retry-failed", default=True, help="Retry failed documents on resume")
@click.option("--verify-only", is_flag=True, help="Run only the 10-document verification, then stop")
@click.option("--no-verify", is_flag=True, help="Skip the pre-build verification (use with care)")
@click.option("--debug-one", type=str, default=None, help="Question ID of the ONE document to debug (payload, est tokens, raw body, content analysis)")
def build(
    enriched: str | None,
    checkpoint: str | None,
    embedding_model: str | None,
    ollama_model: str | None,
    llm_provider: str | None,
    llm_models: str | None,
    limit: int | None,
    no_resume: bool,
    retry_failed: bool,
    verify_only: bool,
    no_verify: bool,
    debug_one: str | None,
) -> None:
    """Build (or resume) the Neo4j graph from the enriched corpus."""
    config = _load_config(
        enriched, checkpoint, embedding_model, ollama_model, limit, no_resume, retry_failed,
        llm_provider=llm_provider, llm_models=llm_models, debug_one=debug_one,
    )
    console.print(Panel.fit("[bold cyan]GraphRAG — Build (Neo4j)[/bold cyan]", border_style="cyan"))

    pipeline = GraphRAGPipeline(config)
    try:
        if not pipeline.store.ping():
            console.print("[red]Neo4j is not reachable. Is it running?[/red]")
            console.print(f"  URI: {config.neo4j_uri} | user: {config.neo4j_user}")
            sys.exit(1)

        records = pipeline.load_enriched()
        console.print(f"[cyan]Loaded {len(records):,} enriched records.[/cyan]")

        if verify_only:
            result = pipeline.verify_sample(records, n=10)
            console.print("[green]✓ Verification passed on 10 random documents.[/green]")
            _print_result(result, title="Verification")
            return

        result = pipeline.build(records, verify_first=not no_verify, n_verify=10)
        _print_result(result, title="Graph Build Complete")
    finally:
        pipeline.close()


@cli.command()
@click.option("--enriched", type=str, default=None)
@click.option("--checkpoint", type=str, default=None)
@click.option("--embedding-model", type=str, default=None)
@click.option("--ollama-model", type=str, default=None)
@click.option(
    "--llm-provider",
    type=click.Choice(_LLM_PROVIDERS, case_sensitive=False),
    default=None,
    help="LLM backend for extraction (ollama | groq | openai_compatible; default ollama)",
)
@click.option(
    "--llm-models",
    type=str,
    default=None,
    help="Comma-separated model list for the active provider (e.g. 'm1,m2,m3')",
)
@click.option("--limit", type=int, default=None)
@click.option("--debug-one", type=str, default=None, help="Question ID of the ONE document to debug")
def rebuild(
    enriched: str | None,
    checkpoint: str | None,
    embedding_model: str | None,
    ollama_model: str | None,
    llm_provider: str | None,
    llm_models: str | None,
    limit: int | None,
    debug_one: str | None,
) -> None:
    """Drop the graph and rebuild from scratch (destructive)."""
    config = _load_config(
        enriched, checkpoint, embedding_model, ollama_model, limit,
        no_resume=True, retry_failed=True,
        llm_provider=llm_provider, llm_models=llm_models, debug_one=debug_one,
    )
    console.print(Panel.fit("[bold yellow]GraphRAG — Rebuild (drops existing graph)[/bold yellow]", border_style="yellow"))

    pipeline = GraphRAGPipeline(config)
    try:
        if not pipeline.store.ping():
            console.print("[red]Neo4j is not reachable.[/red]")
            sys.exit(1)
        pipeline.store.reset_graph()
        # Fresh checkpoint: remove any previous checkpoint file.
        if config.checkpoint_file.exists():
            config.checkpoint_file.unlink()
            console.print(f"[yellow]Cleared checkpoint {config.checkpoint_file}[/yellow]")
        records = pipeline.load_enriched()
        console.print(f"[cyan]Loaded {len(records):,} enriched records.[/cyan]")
        result = pipeline.build(records, verify_first=True, n_verify=10)
        _print_result(result, title="Graph Rebuild Complete")
    finally:
        pipeline.close()


@cli.command()
@click.option("--enriched", type=str, default=None, help="Glob/path to enriched JSONL")
@click.option("--embedding-model", type=str, default=None, help="Override embedding model (default BAAI/bge-m3)")
@click.option("--ollama-model", type=str, default=None, help="Override Ollama model (default qwen3:8b)")
@click.option(
    "--llm-provider",
    type=click.Choice(_LLM_PROVIDERS, case_sensitive=False),
    default=None,
    help="LLM backend for extraction (ollama | groq | openai_compatible; default ollama)",
)
@click.option(
    "--llm-models",
    type=str,
    default=None,
    help="Comma-separated model list for the active provider (e.g. 'm1,m2,m3')",
)
@click.option("--n", type=int, default=10, help="Number of random documents to verify")
@click.option("--debug-one", type=str, default=None, help="Question ID of the ONE document to debug (payload, est tokens, raw body, content analysis)")
def verify(
    enriched: str | None,
    embedding_model: str | None,
    ollama_model: str | None,
    llm_provider: str | None,
    llm_models: str | None,
    n: int,
    debug_one: str | None,
) -> None:
    """Verify extraction quality on a sample of documents (no full build)."""
    config = _load_config(
        enriched, None, embedding_model, ollama_model, None, False, True,
        llm_provider=llm_provider, llm_models=llm_models, debug_one=debug_one,
    )
    console.print(Panel.fit("[bold cyan]GraphRAG — Verification (sample)[/bold cyan]", border_style="cyan"))

    pipeline = GraphRAGPipeline(config)
    try:
        if not pipeline.store.ping():
            console.print("[red]Neo4j is not reachable. Is it running?[/red]")
            sys.exit(1)
        records = pipeline.load_enriched()
        console.print(f"[cyan]Loaded {len(records):,} enriched records. Verifying {n} random documents...[/cyan]")
        verifier = GraphVerifier(config)
        report = verifier.run(records, n=n)
        verifier.render(report)
        grade = report.grade()
        if grade in ("Needs prompt tuning", "Poor"):
            sys.exit(2)  # non-zero exit so scripts can gate the full build
    finally:
        pipeline.close()


@cli.command()
@click.option("--checkpoint", type=str, default=None)
def stats(checkpoint: str | None) -> None:
    """Show Neo4j graph statistics."""
    config = _load_config(None, checkpoint, None, None, None, False, True)
    pipeline = GraphRAGPipeline(config)
    try:
        if not pipeline.store.ping():
            console.print("[red]Neo4j is not reachable.[/red]")
            sys.exit(1)
        s = pipeline.store.stats()
        console.print(Panel.fit("[bold cyan]GraphRAG — Graph Statistics[/bold cyan]", border_style="cyan"))

        tab = Table(title="Node Counts by Label")
        tab.add_column("Label")
        tab.add_column("Count", justify="right")
        for label in sorted(s["labels"], key=lambda l: -s["labels"][l]):
            tab.add_row(label, f"{s['labels'][label]:,}")
        console.print(tab)

        tab2 = Table(title="Relationship Counts by Type")
        tab2.add_column("Type")
        tab2.add_column("Count", justify="right")
        for rt in sorted(s["relationships"], key=lambda r: -s["relationships"][r]):
            if s["relationships"][rt]:
                tab2.add_row(rt, f"{s['relationships'][rt]:,}")
        console.print(tab2)

        console.print(f"\n[b]Total nodes:[/b] {s['total_nodes']:,}  |  [b]Total relationships:[/b] {s['total_relationships']:,}")
        vec = [i for i in s["indexes"] if i.get("type") == "VECTOR"]
        console.print(f"[b]Vector indexes:[/b] {[(i['name'], i['labelsOrTypes'], i['properties']) for i in vec]}")
        cp = pipeline.checkpoint.counts()
        console.print(f"[b]Checkpoint:[/b] {cp}")
    finally:
        pipeline.close()


@cli.command()
@click.argument("question")
@click.option("--top-k", type=int, default=10)
@click.option("--embedding-model", type=str, default=None)
@click.option("--ollama-model", type=str, default=None)
@click.option(
    "--llm-provider",
    type=click.Choice(_LLM_PROVIDERS, case_sensitive=False),
    default=None,
    help="LLM backend for entity extraction (ollama | groq | openai_compatible; default ollama)",
)
@click.option(
    "--llm-models",
    type=str,
    default=None,
    help="Comma-separated model list for the active provider (e.g. 'm1,m2,m3')",
)
@click.option("--json-output", is_flag=True, help="Emit results as JSON")
def query(
    question: str,
    top_k: int,
    embedding_model: str | None,
    ollama_model: str | None,
    llm_provider: str | None,
    llm_models: str | None,
    json_output: bool,
) -> None:
    """Graph-aware query: entity expansion + vector search."""
    config = _load_config(
        None, None, embedding_model, ollama_model, None, False, True,
        llm_provider=llm_provider, llm_models=llm_models,
    )
    querier = GraphRAGQuerier(config)
    try:
        results = querier.query(question, top_k=top_k)
        if json_output:
            click.echo(json.dumps([r.as_dict() for r in results], indent=2))
            return
        if not results:
            console.print("[yellow]No results found in the graph.[/yellow]")
            return
        table = Table(title=f"GraphRAG Results — {question[:60]}")
        table.add_column("#", style="dim")
        table.add_column("Doc ID", style="cyan")
        table.add_column("Subject")
        table.add_column("Ministry", style="dim")
        table.add_column("Date", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Matched Entities", style="dim")
        table.add_column("Via", style="dim")
        for i, r in enumerate(results, start=1):
            table.add_row(
                str(i),
                r.doc_id,
                (r.subject or "")[:45],
                r.ministry or "-",
                r.date or "-",
                f"{r.score:.3f}",
                ", ".join(r.matched_entities[:4]),
                r.via,
            )
        console.print(table)
    finally:
        querier.close()


def _print_result(result, title: str) -> None:
    d = result.to_dict()
    lines = [
        f"Documents processed : {d['documents_processed']:,}",
        f"Nodes created       : {d['nodes_created']:,}",
        f"Relationships created: {d['relationships_created']:,}",
        f"Embedding count     : {d['embedding_count']:,}",
        f"Failures            : {d['failures']}",
        f"Retries             : {d['retries']}",
        f"Build duration      : {d['duration_seconds']:.1f}s",
        f"Checkpoint counts   : {d['checkpoint']}",
        f"Skipped via checkpoint: {d['skipped_from_checkpoint']}",
    ]
    if d["failed_docs"]:
        lines.append(f"Failed docs         : {d['failed_docs'][:20]}")
    console.print(Panel.fit("\n".join(lines), title=f"[bold green]{title}[/bold green]", border_style="green"))
    if d["failures"]:
        console.print(
            f"[yellow]Warning: {d['failures']} document(s) failed. "
            "Re-run `graphrag build` to retry them (resume).[/yellow]"
        )


if __name__ == "__main__":
    cli()
