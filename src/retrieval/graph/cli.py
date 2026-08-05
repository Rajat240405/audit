"""
GraphRAG CLI Interface for Phase 5.
Provides commands to build, analyze, and traverse the NetworkX metadata graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.retrieval.graph.store import GraphStore
from src.retrieval.graph.retriever import GraphRetriever
from src.utils.project_scope import resolve_effective_ministry_filter, filter_records_by_ministry

console = Console()


@click.group()
def graph_cli() -> None:
    """Phase 5 — GraphRAG Command Line Interface."""
    pass


@graph_cli.command()
@click.option(
    "--index-dir", "-i",
    default="storage/hybrid_rag",
    help="Path to the Hybrid RAG indexing directory containing doc_map.json",
)
@click.option(
    "--output-dir", "-o",
    default="storage/graphrag",
    help="Output directory to serialize the graph.",
)
@click.option(
    "--ministry-filter",
    type=str,
    default=None,
    help="Explicit ministry filter (overrides config)",
)
@click.option(
    "--all-ministries",
    is_flag=True,
    default=False,
    help="Index ALL ministries (overrides MoES default scope)",
)
def build(index_dir: str, output_dir: str, ministry_filter: str | None, all_ministries: bool) -> None:
    """Build the metadata graph from the indexed doc map (respects project_scope)."""
    console.print(Panel.fit(
        "[bold cyan]Phase 5 — Building Metadata-Driven GraphRAG[/bold cyan]",
        border_style="cyan"
    ))

    doc_map_path = Path(index_dir) / "doc_map.json"
    if not doc_map_path.exists():
        console.print(f"[red]Error: Index doc_map.json not found at {index_dir}. Run 'retrieve build' first.[/red]")
        return

    # Load doc map
    console.print(f"Loading document metadata map from [cyan]{doc_map_path}[/cyan]...")
    with open(doc_map_path, encoding="utf-8") as f:
        doc_map = json.load(f)

    # ── Apply centralized project scope filtering (shared utility) ──
    effective_filter = resolve_effective_ministry_filter(
        explicit_filter=ministry_filter,
        all_ministries=all_ministries,
    )

    if effective_filter:
        before = len(doc_map)
        doc_map = {
            k: v for k, v in doc_map.items()
            if v.get("metadata", {}).get("ministry") and effective_filter.lower() in v.get("metadata", {}).get("ministry", "").lower()
        }
        console.print(f"[cyan]Ministry filter applied ({effective_filter}): {before:,} → {len(doc_map):,} documents[/cyan]")

    # Instantiate GraphStore and build
    store = GraphStore(storage_dir=output_dir)
    console.print(f"Constructing in-memory graph over [cyan]{len(doc_map):,}[/cyan] document nodes...")
    store.build_graph(doc_map)
    
    # Save to disk
    store.save()
    console.print(f"\n[bold green]✓ Graph successfully built and serialized to [cyan]{store.graph_file}[/cyan][/bold green]")


@graph_cli.command()
@click.option(
    "--graph-dir", "-g",
    default="storage/graphrag",
)
def stats(graph_dir: str) -> None:
    """Print comprehensive graph analytics and statistics."""
    store = GraphStore(storage_dir=graph_dir)
    try:
        store.load()
    except FileNotFoundError:
        console.print(f"[red]Error: Graph file not found at {store.graph_file}. Run 'graph build' first.[/red]")
        return

    g_stats = store.get_stats()

    console.print(Panel.fit(
        "[bold cyan]GraphRAG - Structure & Analytics Summary[/bold cyan]",
        border_style="cyan"
    ))

    # General Stats Table
    table_gen = Table(title="Core Graph Metrics", show_lines=True)
    table_gen.add_column("Metric", style="bold")
    table_gen.add_column("Value", justify="right", style="green")
    
    table_gen.add_row("Total Nodes (N)", f"{g_stats['total_nodes']:,}")
    table_gen.add_row("Total Edges (E)", f"{g_stats['total_edges']:,}")
    table_gen.add_row("Average Node Degree", f"{g_stats['average_degree']:.2f}")
    table_gen.add_row("Connected Components (Undirected)", f"{g_stats['num_components']:,}")
    console.print(table_gen)

    # Node types breakdown
    table_types = Table(title="Nodes Breakdown by Type", show_lines=True)
    table_types.add_column("Node Type", style="bold")
    table_types.add_column("Count", justify="right", style="cyan")
    for ntype, count in g_stats["node_types"].items():
        table_types.add_row(ntype, f"{count:,}")
    console.print(table_types)

    # Top Centralities
    table_top = Table(title="Top Connected Central Entities (Degree Centrality)", show_lines=True)
    table_top.add_column("Ministry", style="cyan")
    table_top.add_column("Top MPs (Degree)", style="yellow")
    table_top.add_column("Most Connected Subjects", style="magenta")

    min_list = [f"{m} ({d})" for m, d in g_stats["top_ministries"]]
    mp_list = [f"{m} ({d})" for m, d in g_stats["top_mps"]]
    sub_list = [f"{s} ({d})" for s, d in g_stats["top_subjects"]]

    max_len = max(len(min_list), len(mp_list), len(sub_list))
    for i in range(max_len):
        m_val = min_list[i] if i < len(min_list) else ""
        mp_val = mp_list[i] if i < len(mp_list) else ""
        sub_val = sub_list[i] if i < len(sub_list) else ""
        table_top.add_row(m_val, mp_val, sub_val)
    console.print(table_top)


@graph_cli.command()
@click.argument("query_text", type=str)
@click.option(
    "--graph-dir", "-g",
    default="storage/graphrag",
)
@click.option(
    "--top-k", "-k",
    default=5,
    type=int,
)
def query(query_text: str, graph_dir: str, top_k: int) -> None:
    """Query GraphRAG and retrieve adjacent document records."""
    store = GraphStore(storage_dir=graph_dir)
    try:
        store.load()
    except FileNotFoundError:
        console.print(f"[red]Error: Graph file not found. Run 'graph build' first.[/red]")
        return

    retriever = GraphRetriever(store=store)
    results = retriever.retrieve(query_text, top_k=top_k)

    console.print(f"\n[bold]GraphRAG Query:[/bold] '{query_text}'")
    
    if not results:
        console.print("[yellow]No document connections traversed for this query.[/yellow]")
        return

    table = Table(title="Graph Traversal Retrieved Documents", show_lines=True)
    table.add_column("#", justify="center")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Ministry", style="green")
    table.add_column("Subject", style="yellow")
    table.add_column("Overlap Score", justify="right")

    for idx, r in enumerate(results, start=1):
        table.add_row(
            str(idx),
            r.doc_id,
            r.metadata.get("ministry") or "-",
            r.metadata.get("subject") or "-",
            f"{r.score:.1f}"
        )
    console.print(table)


@graph_cli.command()
@click.argument("node_id", type=str)
@click.option(
    "--graph-dir", "-g",
    default="storage/graphrag",
)
def neighbors(node_id: str, graph_dir: str) -> None:
    """Print the direct neighbors of a specific node."""
    store = GraphStore(storage_dir=graph_dir)
    try:
        store.load()
    except FileNotFoundError:
        console.print(f"[red]Error: Graph file not found.[/red]")
        return

    if not store.graph.has_node(node_id):
        # Try to find fuzzy matches
        console.print(f"[yellow]Node '{node_id}' not found exactly. Searching fuzzy matches...[/yellow]")
        nodes = list(store.graph.nodes())
        matches = [n for n in nodes if node_id.lower() in n.lower()]
        if not matches:
            console.print("[red]No matching nodes found.[/red]")
            return
        node_id = matches[0]
        console.print(f"[green]Found match: '{node_id}'[/green]\n")

    attrs = store.graph.nodes[node_id]
    console.print(f"[bold]Node ID:[/bold] {node_id}")
    console.print(f"[bold]Attributes:[/bold] {attrs}")

    # Successors (outbound edges)
    successors = list(store.graph.successors(node_id))
    # Predecessors (inbound edges)
    predecessors = list(store.graph.predecessors(node_id))

    table = Table(title=f"Neighbors of '{node_id}'", show_lines=True)
    table.add_column("Direction", style="bold")
    table.add_column("Connected Node", style="cyan")
    table.add_column("Relationship", style="green")

    for succ in successors:
        edge_data = store.graph.get_edge_data(node_id, succ)
        table.add_row("Outbound (Successor)", succ, edge_data.get("relation", "-"))

    for pred in predecessors:
        edge_data = store.graph.get_edge_data(pred, node_id)
        table.add_row("Inbound (Predecessor)", pred, edge_data.get("relation", "-"))

    console.print(table)


@graph_cli.command()
@click.argument("node1", type=str)
@click.argument("node2", type=str)
@click.option(
    "--graph-dir", "-g",
    default="storage/graphrag",
)
def path(node1: str, node2: str, graph_dir: str) -> None:
    """Compute and display the shortest path between two nodes."""
    store = GraphStore(storage_dir=graph_dir)
    try:
        store.load()
    except FileNotFoundError:
        console.print(f"[red]Error: Graph file not found.[/red]")
        return

    # Check existence
    undirected_g = store.graph.to_undirected()
    
    # Helper to resolve fuzzy nodes if needed
    def resolve_fuzzy(n_id):
        if store.graph.has_node(n_id):
            return n_id
        matches = [n for n in store.graph.nodes() if n_id.lower() in n.lower()]
        return matches[0] if matches else None

    r_node1 = resolve_fuzzy(node1)
    r_node2 = resolve_fuzzy(node2)

    if not r_node1 or not r_node2:
        console.print(f"[red]Error: Could not resolve one or both nodes: '{node1}', '{node2}'[/red]")
        return

    console.print(f"Resolved path anchors: [cyan]'{r_node1}'[/cyan] → [cyan]'{r_node2}'[/cyan]")

    try:
        shortest_path = nx.shortest_path(undirected_g, r_node1, r_node2)
        console.print("\n[bold green]Shortest Connection Path found:[/bold green]")
        
        path_str = []
        for idx, step in enumerate(shortest_path):
            step_type = store.graph.nodes[step].get("type", "Unknown")
            path_str.append(f"({step} [yellow]{step_type}[/yellow])")
            
        console.print("  " + "  ──▶  ".join(path_str))
    except nx.NetworkNoPath:
        console.print("[yellow]No connection path exists between these two entities.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    graph_cli()
