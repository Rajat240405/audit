"""Canonical Neo4j GraphRAG CLI (Phase 2).

    graphrag init                 apply schema (constraints + indexes, idempotent)
    graphrag build                incremental corpus build (resumable; LLM by default)
        --deterministic-only      no LLM (metadata graph only)
        --limit N                 process at most N documents
        --semantic-backfill       ONLY documents not yet semantically
                                  extracted (--limit N then means "N documents
                                  still needing semantic extraction")
        --no-resume               ignore the checkpoint
        --prune                   withdraw docs missing from the corpus
    graphrag query "INCOIS tsunami"   graph-mode retrieval (top 5 by default)
        --top-k N
    graphrag stats                node/relationship counts
    graphrag remove <doc_key>     withdraw ONE document's graph contribution
    graphrag prune                same as `build --prune` (skips processing)

Configuration is environment-driven (see src/graphrag/config.py):
    GRAPHRAG_BACKEND=neo4j|inmemory   (default neo4j; inmemory = local validation)
    GRAPHRAG_NEO4J_URI / _USER / _PASSWORD / _DATABASE
    GRAPHRAG_CHECKPOINT, GRAPHRAG_CORPUS

On HPC the commands run as:
    python -m src.graphrag.cli init
    python -m src.graphrag.cli build            (full incremental build)
    python -m src.graphrag.cli build --prune    (after corpus deletions)
See docs/phase2/DESIGN.md §8 for the deployment notes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.graphrag.config import GraphConfigError, load_graph_config

__all__ = ["cli", "main"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("graphrag.cli")


def _store(config):
    """Backend factory shared by all commands (raises for misconfiguration —
    CLI commands fail LOUDLY; only the serving capability degrades)."""
    if config.backend == "inmemory":
        from src.graphrag.store import InMemoryGraphStore
        log.warning("inmemory backend: local validation only (no persistence)")
        return InMemoryGraphStore()
    from src.graphrag.neo4j_store import Neo4jGraphStore
    store = Neo4jGraphStore(config)
    if not store.ping():
        store.close()
        raise GraphConfigError(f"Neo4j unreachable at {config.neo4j_uri}")
    return store


def _llm(deterministic_only: bool, config=None):
    """Build the semantic-extraction client from the SHARED generation stack.

    The registry/activation/policy chain resolves provider, model, context and
    thinking exactly as for every other consumer — nothing is hardcoded here.

    The ONE GraphRAG-scoped adjustment is the OUTPUT token budget. The shared
    fast-mode profile allows 4096 completion tokens, which is right for chat
    answers but truncates extraction JSON for fact-dense documents: vLLM
    returns finish_reason="length" mid-object and the payload fails to parse
    ("LLM returned non-JSON payload"). Extraction emits one JSON record per
    entity/relationship, so its output scales with document density, not with
    answer length.

    This mutates only the LLMClient instance owned by this build process
    (resolve_active_stack constructs a fresh client per call), so the serving
    path, Hybrid RAG and every other generation consumer keep the shared
    policy value untouched.
    """
    if deterministic_only:
        return None
    from src.generation.activation import resolve_active_stack
    stack = resolve_active_stack("fast")
    policy_max_tokens = stack.client.max_tokens
    if config is not None:
        stack.client.max_tokens = int(config.extract_max_tokens)
    log.info("extraction model: provider=%s model=%s (source=%s)",
             stack.provider, stack.model, stack.source)
    log.info("extraction output budget: max_tokens=%s "
             "(shared policy default %s; GRAPHRAG_EXTRACT_MAX_TOKENS), "
             "num_ctx=%s",
             stack.client.max_tokens, policy_max_tokens, stack.plan.num_ctx)
    return stack.client


def cmd_init(config) -> int:
    store = _store(config)
    try:
        store.init_schema()
        print("schema applied (constraints + indexes, idempotent)")
        return 0
    finally:
        store.close()


def cmd_build(config, args) -> int:
    from src.graphrag.pipeline import GraphBuilder
    store = _store(config)
    try:
        if not config.corpus_path.exists():
            log.error("corpus not found: %s", config.corpus_path)
            return 1
        builder = GraphBuilder(
            store, config,
            llm_client=_llm(args.deterministic_only, config))
        if not args.deterministic_only:
            log.info("LLM extraction concurrency: %d "
                     "(GRAPHRAG_LLM_CONCURRENCY; keep <= vLLM --max-num-seqs)",
                     config.llm_concurrency)
        report = builder.run(
            config.corpus_path,
            limit=args.limit,
            deterministic_only=args.deterministic_only,
            prune=args.prune,
            resume=not args.no_resume,
            semantic_backfill=args.semantic_backfill,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.failed == 0 else 2
    finally:
        store.close()


def cmd_prune(config, args) -> int:
    from src.graphrag.pipeline import GraphBuilder
    store = _store(config)
    try:
        if not config.corpus_path.exists():
            log.error("corpus not found: %s", config.corpus_path)
            return 1
        builder = GraphBuilder(store, config, llm_client=None)
        report = builder.run(config.corpus_path, prune=True,
                             resume=not args.no_resume)
        print(json.dumps({"withdrawn": report.withdrawn}))
        return 0
    finally:
        store.close()


def cmd_query(config, args) -> int:
    from src.graphrag.capability import build_graph_capability
    cap = build_graph_capability(config)
    try:
        results = cap.retrieve(args.query, top_k=args.top_k)
        if not results:
            print("(no graph results — backend disabled or no matching anchors)")
            return 1
        for r in results:
            g = r.metadata.get("graph", {})
            print(f"── {r.doc_id}  score={r.score}  via={g.get('via')}")
            print(f"   Q: {r.question}")
            print(f"   A: {(r.answer or '')[:300]}")
            print(f"   anchors: {[a['name'] for a in g.get('anchors', [])]}")
            for fk in g.get("fact_keys", [])[:5]:
                print(f"   fact: {fk}")
        # full provenance block (Phase-3 verification input)
        print(json.dumps([e.to_dict() for e in cap.explain(results)],
                         indent=2, default=str))
        return 0
    finally:
        cap.close()


def cmd_stats(config) -> int:
    from src.graphrag.capability import build_graph_capability
    cap = build_graph_capability(config)
    try:
        print(json.dumps(cap.stats(), indent=2))
        return 0
    finally:
        cap.close()


def cmd_remove(config, args) -> int:
    import time
    store = _store(config)
    try:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = store.withdraw_document(args.doc_key, now=now)
        print(json.dumps(out))
        return 0
    finally:
        store.close()


def cli(argv=None) -> int:
    p = argparse.ArgumentParser(prog="graphrag",
                                description="Canonical Neo4j GraphRAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="apply schema (idempotent)")

    b = sub.add_parser("build", help="incremental corpus build (resumable)")
    b.add_argument("--deterministic-only", action="store_true",
                   help="no LLM — metadata graph only")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--semantic-backfill", action="store_true",
                   help="process only documents whose checkpoint extraction "
                        "mode is not 'semantic' (already-semantic documents "
                        "are skipped; --limit counts documents that still "
                        "need semantic extraction)")
    b.add_argument("--no-resume", action="store_true")
    b.add_argument("--prune", action="store_true",
                   help="withdraw docs missing from the corpus")

    q = sub.add_parser("query", help="graph-mode retrieval")
    q.add_argument("query")
    q.add_argument("--top-k", type=int, default=5)

    sub.add_parser("stats", help="graph statistics")

    rm = sub.add_parser("remove", help="withdraw one document's contribution")
    rm.add_argument("doc_key")

    pr = sub.add_parser("prune", help="withdraw docs missing from the corpus")
    pr.add_argument("--no-resume", action="store_true")

    args = p.parse_args(argv)
    try:
        config = load_graph_config()
    except GraphConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 3

    if args.cmd == "init":
        return cmd_init(config)
    if args.cmd == "build":
        return cmd_build(config, args)
    if args.cmd == "prune":
        return cmd_prune(config, args)
    if args.cmd == "query":
        return cmd_query(config, args)
    if args.cmd == "stats":
        return cmd_stats(config)
    if args.cmd == "remove":
        return cmd_remove(config, args)
    return 2


def main() -> None:  # pragma: no cover
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
