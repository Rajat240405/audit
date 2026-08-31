"""MoES website crawler — v2 (staging only; no indexing/embedding).

Stages ministry-owned documents from www.moes.gov.in into::

    data/.moes-website/                    (override with --root / APP_DATA_DIR)
    ├── reports/<family>/<post-slug>/      (record.json + documents/)
    │   families: annual-reports, monthly-reports, demands-for-grants  (ONLY)
    ├── press-release/<post-slug>/
    ├── attachment-map.json                (attachment-id → resolved pdf-links cache)
    └── last_run.json                      (run artefact)

Scope (fail-closed, user-approved 2026-08-26): ``reports`` (the three families
above, resolved from the LIVE taxonomy) and ``press-release`` (the whole
category) are the only valid ``--categories`` choices. ``guidelines``,
``orders-and-notices``, ``publications``, ``acts-and-policy``,
``gazette-notifications`` and the non-approved reports families
(``ncaer-reports``, ``account-at-glance``, ``achievements``,
``general-report``) are OUT — the config loader rejects them too.
Press-release documents titled "PARLIAMENT QUESTION: …" are KEPT (never
discarded as parliamentary duplicates; cross-source dedupe is a later,
reviewed integration step).

``central_documents`` (underscore) is NOT a listing category: it is the
backend's internal post_type that stores the PDF attachments of BOTH reports
and press-release posts. It is used only inside the attachment resolver
(ACF attachment id -> post-page/post?id=<id> -> central_documents post ->
pdf / pdf_hindi / pdf_both -> download) and is never crawled directly.

Usage (run on a machine with internet; HPC ingests staged corpora offline)::

    # full v2 scope (reports families + press-release)
    python -m src.scripts.crawl_moes_website

    # inventory only — no writes, no document downloads
    python -m src.scripts.crawl_moes_website --dry-run

    # dry-run + read-only attachment-resolvability probe
    # (resolves each ACF attachment id via post-page/post?id=; no file bytes)
    python -m src.scripts.crawl_moes_website --dry-run --resolve-attachments

    # subsets (narrowing only, never widening)
    python -m src.scripts.crawl_moes_website --categories reports \
        --report-families annual-reports

    # records only
    python -m src.scripts.crawl_moes_website --no-documents

Exit codes: 0 = success, 2 = usage/config error, 3 = completed with failures
(category aborts or broken document slots; details in last_run.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.scraping.moes.client import MoesApi, build_http
from src.scraping.moes.config import (
    V1_CATEGORIES,
    MoesConfigError,
    load_config,
    output_root,
)
from src.scraping.moes.pipeline import CrawlContext, RunOptions, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crawl_moes_website",
        description="MoES website crawler v1 (staging only — no indexing/embedding).",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="crawler config YAML (default: config/crawlers/moes_website.yaml)")
    p.add_argument("--root", type=Path, default=None,
                   help="output root override (default: <data>/.moes-website)")
    p.add_argument("--categories", nargs="+", default=None, choices=list(V1_CATEGORIES),
                   metavar="CATEGORY",
                   help=f"subset of {list(V1_CATEGORIES)} (default: all configured)")
    p.add_argument("--report-families", nargs="+", default=None, metavar="FAMILY",
                   help="subset of the configured reports families (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="inventory only: no writes, no document downloads")
    p.add_argument("--resolve-attachments", action="store_true",
                   help="with --dry-run: also probe attachment resolution "
                        "(read-only) so the report includes resolvability")
    p.add_argument("--no-documents", dest="documents", action="store_false",
                   help="records only: skip document downloads")
    return p


def main(argv: list[str] | None = None, *, transport=None, sleeper=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except MoesConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    root = output_root(cfg, args.root)
    opts = RunOptions(
        categories=args.categories,
        report_families=args.report_families,
        fetch_documents=args.documents,
        dry_run=args.dry_run,
        resolve_attachments=args.resolve_attachments,
    )
    http = build_http(cfg, transport=transport, sleeper=sleeper)
    api = MoesApi(cfg, http, sleeper=sleeper)
    ctx = CrawlContext(cfg=cfg, opts=opts, http=http, api=api, root=root)

    cat_hint = f"categories={args.categories}" if args.categories else "categories=all-configured"
    print(f"[start] moes-website root={root} {cat_hint}")
    if opts.dry_run:
        print(f"[dry-run] root={root} (nothing will be written)")
    try:
        try:
            report = run(ctx)
        except MoesConfigError as exc:      # scope guards fire at run time too
            print(f"[error] {exc}", file=sys.stderr)
            return 2
    except Exception as exc:                # noqa: BLE001 — e.g. taxonomy fetch failed
        print(f"[error] run aborted before any category completed: {exc}", file=sys.stderr)
        return 3
    finally:
        http.close()

    n_categories = len(report["categories"])
    for i, c in enumerate(report["categories"], 1):
        line = (f"  [{i}/{n_categories}] {c['category']:14s} {c['status']:8s} discovered={c['discovered']} "
                f"in_scope={c['in_scope']} scoped_out={len(c['scoped_out'])} "
                f"pq_titled={c['pq_titled']}")
        if c["status"] != "dry-run":
            line += (f" +{c['added']} ~{c['changed']} ={c['unchanged']} "
                     f"docs(g/p/b)={c['docs']['good']}/{c['docs']['partial']}/"
                     f"{c['docs']['broken']} skipped_ext={c['skipped_external']} "
                     f"attention={c['attention']} tombstoned={c['tombstoned']}")
        if c.get("attachments"):
            att = c["attachments"]
            line += (f" attachments(resolved/needed)={att['resolved']}/{att['needed']}"
                     f" unresolved={att['unresolved'][:8]}")
        if c.get("families"):
            line += f" families={json.dumps(c['families'], sort_keys=True)}"
        if c["failures"]:
            line += f" FAILURES={c['failures']}"
        print(line)
        for s in c["scoped_out"]:
            print(f"    scoped-out: {s['id']} {s['slug']} children={s['children']} "
                  f"({s['reason']})")
    if report.get("pq_stats"):
        print(f"  pq_stats: {json.dumps(report['pq_stats'], sort_keys=True)}")
    print(json.dumps({"categories": len(report["categories"]),
                      "http_requests": report["http_requests"],
                      "failures": report["failures"],
                      "overlap": report["overlap"]}, indent=1))
    if not opts.dry_run:
        print(f"[done] root={root} last_run.json updated")
    return 3 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
