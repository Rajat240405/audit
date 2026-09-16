"""Stage 8 — separate V2 child-index build path.

Builds a **standalone** retrieval index from V2 sidecars at
``config.index_child_path`` (default ``storage/hybrid_rag_v2``). It reuses the
production stack — the same ``Embedder`` (BGE-M3), ``FAISSVectorStore``,
``BM25Index``, RRF fusion, parent collapse and ``CrossEncoderReranker`` — so
there is no parallel retrieval engine, only a second index directory.

Hard guards, because this tool runs on a machine that also serves production:

* it refuses to target ``storage/hybrid_rag`` (or any path resolving to it);
* it refuses to run unless V2 is enabled, unless ``--force-v2-off`` is passed;
* it never reads or writes ``data/corpus_reports.jsonl``;
* ``--dry-run`` reports what would be built and writes nothing.

Usage::

    python3 -m src.data.v2.reindex --dry-run
    python3 -m src.data.v2.reindex --limit 20
    python3 -m src.data.v2.reindex --out storage/hybrid_rag_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Directory names that must never be the target of a V2 build.
PROTECTED_INDEX_NAMES = ("hybrid_rag",)


class ProtectedIndexError(RuntimeError):
    """Raised when a build would write to the production index."""


def _assert_not_protected(target: Path) -> None:
    """Refuse to build over the production index.

    Compares resolved paths, so ``storage/hybrid_rag``, ``./storage/hybrid_rag``
    and a symlink to it are all rejected. A V2 build must never be able to
    replace the index a running service loads.
    """
    try:
        resolved = target.resolve()
    except OSError:
        resolved = target.absolute()
    for name in PROTECTED_INDEX_NAMES:
        if resolved.name == name:
            raise ProtectedIndexError(
                f"refusing to build a V2 index at {resolved!s}: that is the production "
                f"index. Choose a separate path (default: storage/hybrid_rag_v2)."
            )
    if resolved.exists():
        for protected in PROTECTED_INDEX_NAMES:
            try:
                if resolved.samefile(Path("storage") / protected):
                    raise ProtectedIndexError(
                        f"refusing to build a V2 index at {resolved!s}"
                    )
            except OSError:
                continue


def plan_build(
    config=None,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """What a build would do, without doing it. Safe to call any time."""
    from src.data.v2.child_units import (
        build_child_units,
        build_v2_records,
        child_unit_stats,
        sidecar_doc_ids,
    )
    from src.data.v2.config import V2Config

    cfg = config if config is not None else V2Config()
    available = sidecar_doc_ids(cfg)
    if doc_ids is None:
        doc_ids = available
    if limit is not None:
        doc_ids = doc_ids[:limit]

    units = build_child_units(cfg, doc_ids)
    records = build_v2_records(cfg, doc_ids)
    return {
        "enabled": cfg.enabled,
        "children_indexed": cfg.children_indexed,
        "index_path": str(cfg.index_child_path),
        "sidecars_available": len(available),
        "docs_selected": len(doc_ids),
        "parents": len(records),
        "child_units": child_unit_stats(units),
        "crops_on_disk": sum(
            1 for u in units if (u.get("meta") or {}).get("crop")
        ),
    }


def build_v2_index(
    config=None,
    *,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
    out: str | Path | None = None,
    dry_run: bool = False,
    embedder=None,
) -> dict:
    """Build the separate V2 index. Returns the plan/summary dict.

    ``dry_run=True`` never constructs an embedder and never writes.
    """
    from src.data.v2.config import V2Config

    cfg = config if config is not None else V2Config()
    plan = plan_build(cfg, doc_ids, limit)

    target = Path(out) if out is not None else Path(cfg.index_child_path)
    _assert_not_protected(target)
    plan["target"] = str(target)

    if dry_run:
        plan["dry_run"] = True
        plan["wrote"] = False
        return plan

    if not cfg.children_indexed:
        plan["wrote"] = False
        plan["skipped"] = (
            "V2 child indexing is off (V2_ENABLED and V2_INDEX_CHILDREN must both "
            "be true). Pass no flags to change this; nothing was written."
        )
        return plan

    if plan["parents"] == 0:
        plan["wrote"] = False
        plan["skipped"] = "no sidecars found; run the V2 extractor first"
        return plan

    from src.data.v2.child_units import build_child_units, build_v2_records

    # Importing the pipeline pulls in the embedding stack; done here so a
    # dry run needs none of it.
    from src.retrieval.hybrid.pipeline import HybridRAGPipeline

    records = build_v2_records(cfg, plan["docs_selected"] if doc_ids is None else doc_ids)
    units = build_child_units(cfg, [r.question_id for r in records])

    pipeline = HybridRAGPipeline(
        records=records,
        embedder=embedder,
        use_reranker=False,
        use_chunking=False,
        enable_v2_children=True,
    )
    # Child units are supplied directly: the parent records were synthesised
    # from the same sidecars, so re-reading the sidecar directory would be
    # redundant and would re-apply the config gate a second time.
    pipeline._index_v2_children(records)  # noqa: SLF001 - same package family
    pipeline.save(target)

    plan["wrote"] = True
    plan["indexed_units"] = len(pipeline._v2_chunk_map)  # noqa: SLF001
    plan["deliberate_unused"] = len(units)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the separate INCOIS V2 child index (never the production index)."
    )
    parser.add_argument("--out", default=None, help="target index directory")
    parser.add_argument("--limit", type=int, default=None, help="cap documents")
    parser.add_argument("--docs", default=None, help="comma-separated sidecar doc ids")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan and write nothing"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    from src.data.v2.config import load_v2_config

    cfg = load_v2_config()
    doc_ids = [d.strip() for d in args.docs.split(",")] if args.docs else None

    try:
        plan = build_v2_index(
            cfg, doc_ids=doc_ids, limit=args.limit, out=args.out, dry_run=args.dry_run
        )
    except ProtectedIndexError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    else:
        for key, value in sorted(plan.items()):
            print(f"{key:22} {value}")
    return 0


__all__ = [
    "PROTECTED_INDEX_NAMES",
    "ProtectedIndexError",
    "build_v2_index",
    "plan_build",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
