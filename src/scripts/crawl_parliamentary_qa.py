"""Parliamentary Q&A crawler — Rajya Sabha module.

Stages official Rajya Sabha Q&A records (and their English/Hindi answer
documents) into the corpus hierarchy::

    data/parliamentary-qa/rajya-sabha/
    └── session-<n>/
        ├── qa.jsonl          # QARecord-shaped rows merged by id
        ├── documents/        # byte-preserved official files (<qslno>-<lang>.<ext>)
        └── manifest.json     # sha-256 state for incremental byte-stable re-runs

Scope guard: ONLY ``--house rajya-sabha`` is implemented (argparse choices
enforce it). The Lok Sabha module and the MoES website crawler are separate
deliverables and are deliberately not implemented here.

Usage (run on a machine with internet; HPC ingests the staged corpus offline):

    # full crawl (all sessions, all configured ministries)
    python -m src.scripts.crawl_parliamentary_qa --house rajya-sabha

    # inventory only — no writes, no document downloads
    python -m src.scripts.crawl_parliamentary_qa --house rajya-sabha --dry-run

    # subset
    python -m src.scripts.crawl_parliamentary_qa --house rajya-sabha \
        --sessions 265-271 --ministry earth-sciences

    # records only (skip document downloads; answers come from inline text)
    python -m src.scripts.crawl_parliamentary_qa --house rajya-sabha --no-documents

    # list official-file artefacts pending eParlib back-fill (read-only)
    python -m src.scripts.crawl_parliamentary_qa --house rajya-sabha --print-pending-backfill

Exit codes: 0 = success, 2 = usage/config error, 3 = some sessions failed
(partial run; per-session failures are in last_run.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.scraping.config import (
    CrawlerConfigError,
    http_kwargs,
    load_config,
    output_root,
    resolve_ministries,
)
from src.scraping.http import CrawlHttpClient
from src.scraping.rs.pipeline import CrawlContext, RunOptions, pending_backfill, run


def _parse_sessions(spec: str | None) -> list[int] | None:
    """'208,212-215' → sorted unique session list; None/'all' → None."""
    if not spec or spec.strip().lower() == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"bad session range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crawl_parliamentary_qa",
        description="Rajya Sabha Q&A crawler (staging only — no indexing/embedding).",
    )
    p.add_argument("--house", required=True, choices=["rajya-sabha"],
                   help="Only 'rajya-sabha' is implemented in this build.")
    p.add_argument("--config", type=Path, default=None,
                   help="crawler config YAML (default: config/crawlers/parliamentary_qa.yaml)")
    p.add_argument("--root", type=Path, default=None,
                   help="output root override (default: <data>/parliamentary-qa/rajya-sabha)")
    p.add_argument("--sessions", default=None,
                   help="session filter, e.g. '208,212-215' or 'all' (default: config range)")
    p.add_argument("--ministry", default=None,
                   help="ministry slug filter, comma-separated (default: all configured)")
    p.add_argument("--dry-run", action="store_true",
                   help="inventory only: no writes, no document downloads")
    p.add_argument("--no-documents", dest="documents", action="store_false",
                   help="records only: skip document downloads")
    p.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                   help="keep previously failed document slots as-is (no re-attempt)")
    p.add_argument("--print-pending-backfill", action="store_true",
                   help="print failed document slots awaiting eParlib back-fill and exit")
    return p


def main(argv: list[str] | None = None, *, transport=None, sleeper=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        ministries = resolve_ministries(
            cfg, [s.strip() for s in args.ministry.split(",")] if args.ministry else None)
        sessions = _parse_sessions(args.sessions)
    except (CrawlerConfigError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    root = output_root(cfg, args.root)

    if args.print_pending_backfill:
        pending = pending_backfill(root)
        print(json.dumps({"root": str(root), "pending": pending, "count": len(pending)},
                         indent=1, ensure_ascii=True))
        return 0

    opts = RunOptions(
        sessions=sessions,
        ministry_slugs=[m["slug"] for m in ministries],
        fetch_documents=args.documents,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
    )
    http = CrawlHttpClient(transport=transport, sleeper=sleeper, **http_kwargs(cfg))
    ctx = CrawlContext(cfg=cfg, opts=opts, http=http, root=root, ministries=ministries)

    if opts.dry_run:
        print(f"[dry-run] root={root} (nothing will be written)")
    try:
        report = run(ctx)
    finally:
        http.close()

    for s in report["sessions"]:
        line = (f"  session {s['session']}: {s['status']:9s} "
                f"records={s['records']} +{s['added']} ~{s['changed']} "
                f"docs(g/p/b/m)={s['docs']['good']}/{s['docs']['partial']}/"
                f"{s['docs']['broken']}/{s['docs']['missing']}")
        if s["zero_ministries"]:
            line += f" empty={','.join(s['zero_ministries'])}"
        if s["failures"]:
            line += f" FAILURES={s['failures']}"
        print(line)
    print(json.dumps({"sessions": len(report["sessions"]),
                      "doc_totals": report["doc_totals"],
                      "http_requests": report["http_requests"],
                      "failures": report["failures"]}, indent=1))
    if not opts.dry_run:
        print(f"[done] root={root} last_run.json updated")
    return 3 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
