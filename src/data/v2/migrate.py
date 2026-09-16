"""Stage 12 — controlled migration tooling. **Tooling only; it never migrates.**

This module exists so that a future migration can be *planned and audited*
before anyone runs one. It is deliberately incapable of doing the dangerous
parts:

* it never writes ``data/corpus_reports.jsonl``;
* it never writes, rebuilds or replaces ``storage/hybrid_rag``;
* it has no code path that performs a backfill. ``execute=False`` is the only
  mode, and passing ``execute=True`` raises rather than proceeding;
* every report is reproducible from the same inputs.

What it does provide:

    subset      pick documents by id, prefix, limit or page-count band
    extract     run the V2 extractor over the subset into sidecars
    compare     legacy text vs V2 ``marked.txt`` per document
    validate    spec §4 provenance invariants on every sidecar
    anomalies   flag documents whose page/table/figure ratios look wrong
    plan        the whole of the above as a dry run, writing nothing but sidecars

The anomaly ratios are the useful part for an operator: a document where 90% of
pages are ``scan`` but only 2 pages were OCR'd means the budget silently
truncated recovery, and that must be visible before a migration, not after.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.v2 import pipeline as v2_pipeline
from src.data.v2.config import V2Config
from src.data.v2.sidecar_store import SidecarStore, validate_sidecars

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Paths this tool must never write.
PROTECTED_TARGETS = (
    "data/corpus_reports.jsonl",
    "storage/hybrid_rag",
)

#: A document with more than this fraction of unusable pages is worth a look.
ANOMALY_SCAN_FRACTION = 0.5
#: OCR recovery below this fraction of scan pages suggests budget truncation.
ANOMALY_OCR_COVERAGE = 0.3
#: A document whose tables outnumber pages is suspicious.
ANOMALY_TABLES_PER_PAGE = 1.5
#: A document with no tables and no figures at all may have failed extraction.
ANOMALY_EMPTY_EXTRACTION = 0


class MigrationRefusedError(RuntimeError):
    """Raised when asked to actually execute a migration."""


@dataclass
class DocReport:
    doc_id: str
    pages: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    ocr_pages: int = 0
    tables: int = 0
    figures: int = 0
    provenance_ok: bool | None = None
    provenance_errors: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    legacy_chars: int | None = None
    v2_chars: int | None = None
    char_delta: float | None = None


@dataclass
class MigrationPlan:
    executed: bool
    docs_selected: int
    docs_with_sidecars: int
    provenance_ok: int
    provenance_failed: int
    anomalies: int
    reports: list[DocReport] = field(default_factory=list)
    protected_targets_touched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["note"] = (
            "DRY RUN. No corpus row and no index was written. Migration is not "
            "implemented in this tool by design."
        )
        return data


def _assert_no_protected_write(target: str | Path | None) -> None:
    if target is None:
        return
    resolved = str(Path(target).resolve())
    for protected in PROTECTED_TARGETS:
        if resolved.endswith(Path(protected).name) and Path(protected).name in resolved:
            raise MigrationRefusedError(
                f"refusing to write protected path {resolved!s}"
            )


def select_documents(
    config: V2Config | None = None,
    *,
    doc_ids: list[str] | None = None,
    prefix: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Choose a controlled subset of documents that already have sidecars."""
    from src.data.v2.child_units import sidecar_doc_ids

    cfg = config if config is not None else V2Config()
    available = sidecar_doc_ids(cfg)

    if doc_ids:
        wanted = set(doc_ids)
        selected = [d for d in available if d in wanted]
    elif prefix:
        selected = [d for d in available if d.startswith(prefix)]
    else:
        selected = list(available)

    if limit is not None:
        selected = selected[:limit]
    return selected


def assess_document(
    doc_id: str,
    config: V2Config | None = None,
    *,
    legacy_chars: int | None = None,
) -> DocReport:
    """Inspect one document's sidecars: counts, provenance, anomalies."""
    cfg = config if config is not None else V2Config()
    store = SidecarStore(doc_id, cfg)
    report = DocReport(doc_id=doc_id)

    pages = store.read_pages()
    tables = store.read_tables()
    figures = store.read_figures()

    report.pages = len(pages)
    for record in pages:
        report.class_counts[record.page_class] = (
            report.class_counts.get(record.page_class, 0) + 1
        )
    report.ocr_pages = sum(1 for r in pages if r.ocr_used)
    report.tables = len(tables)
    report.figures = len(figures)

    ok, errors = validate_sidecars(store.root)
    report.provenance_ok = ok
    report.provenance_errors = errors

    report.anomalies = detect_anomalies(report)

    marked = store.read_marked()
    report.v2_chars = len(marked)
    if legacy_chars is not None:
        report.legacy_chars = legacy_chars
        if legacy_chars > 0:
            report.char_delta = round((report.v2_chars - legacy_chars) / legacy_chars, 4)

    return report


def detect_anomalies(report: DocReport) -> list[str]:
    """Flag ratios that suggest extraction went wrong.

    Each anomaly is a *signal for a human*, not a verdict. The thresholds are
    deliberately loose: the goal is to surface documents an operator should
    look at, not to auto-reject them.
    """
    out: list[str] = []
    if report.pages == 0:
        out.append("no pages recorded")
        return out

    unusable = report.class_counts.get("scan", 0) + report.class_counts.get("vector", 0)
    scan_fraction = unusable / report.pages
    if scan_fraction >= ANOMALY_SCAN_FRACTION:
        coverage = report.ocr_pages / max(1, unusable)
        out.append(
            f"unusable-page fraction {scan_fraction:.0%} "
            f"({unusable}/{report.pages} pages)"
        )
        if coverage < ANOMALY_OCR_COVERAGE:
            out.append(
                f"OCR coverage {coverage:.0%} of unusable pages — page budget "
                f"likely truncated recovery (V2_OCR_PAGE_CAP)"
            )

    if report.tables / report.pages > ANOMALY_TABLES_PER_PAGE:
        out.append(
            f"{report.tables} tables over {report.pages} pages — possible "
            f"over-detection"
        )

    if report.tables == 0 and report.figures == 0:
        out.append("no tables and no figures extracted")

    if report.char_delta is not None and report.char_delta < -0.5:
        out.append(
            f"V2 text is {abs(report.char_delta):.0%} shorter than legacy — "
            f"possible extraction loss"
        )

    return out


def build_plan(
    config: V2Config | None = None,
    *,
    doc_ids: list[str] | None = None,
    prefix: str | None = None,
    limit: int | None = None,
    legacy_chars_by_doc: dict[str, int] | None = None,
    execute: bool = False,
) -> MigrationPlan:
    """Produce a full dry-run plan over a subset.

    ``execute=True`` always raises :class:`MigrationRefusedError`. There is no
    migration to execute; the flag exists so a caller cannot assume silence
    means success.
    """
    if execute:
        raise MigrationRefusedError(
            "migration execution is not implemented. This tool plans and audits "
            "only; no corpus row or index may be rewritten by it."
        )

    cfg = config if config is not None else V2Config()
    selected = select_documents(cfg, doc_ids=doc_ids, prefix=prefix, limit=limit)
    legacy_chars_by_doc = legacy_chars_by_doc or {}

    reports = [
        assess_document(doc_id, cfg, legacy_chars=legacy_chars_by_doc.get(doc_id))
        for doc_id in selected
    ]

    return MigrationPlan(
        executed=False,
        docs_selected=len(selected),
        docs_with_sidecars=len(selected),
        provenance_ok=sum(1 for r in reports if r.provenance_ok),
        provenance_failed=sum(1 for r in reports if not r.provenance_ok),
        anomalies=sum(1 for r in reports if r.anomalies),
        reports=reports,
        protected_targets_touched=[],
    )


def run_extraction(
    pdf_paths: list[Path | str],
    config: V2Config | None = None,
    *,
    write: bool = True,
) -> tuple[list, list[dict]]:
    """Extract V2 sidecars for a subset. Writes sidecars only.

    This is the one step that touches disk, and it touches only the sidecar
    directory. It cannot reach the corpus or an index.
    """
    cfg = config if config is not None else V2Config()
    _assert_no_protected_write(cfg.sidecar_dir)
    return v2_pipeline.extract_many(pdf_paths, cfg, write=write)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan and audit an INCOIS V2 migration. Never executes one."
    )
    parser.add_argument("--prefix", default=None, help="select doc ids by prefix")
    parser.add_argument("--docs", default=None, help="comma-separated doc ids")
    parser.add_argument("--limit", type=int, default=None, help="cap the subset")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="refused: migration execution is deliberately not implemented",
    )
    args = parser.parse_args(argv)

    from src.data.v2.config import load_v2_config

    cfg = load_v2_config()
    doc_ids = [d.strip() for d in args.docs.split(",")] if args.docs else None

    try:
        plan = build_plan(
            cfg,
            doc_ids=doc_ids,
            prefix=args.prefix,
            limit=args.limit,
            execute=args.execute,
        )
    except MigrationRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 2

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        data = plan.to_dict()
        for key in ("executed", "docs_selected", "provenance_ok",
                    "provenance_failed", "anomalies"):
            print(f"{key:20} {data[key]}")
        print(f"\n{data['note']}")
        for report in plan.reports:
            flags = "; ".join(report.anomalies) if report.anomalies else "ok"
            prov = "prov-ok" if report.provenance_ok else "PROV-FAIL"
            print(
                f"  {report.doc_id:44} pages={report.pages:<4} "
                f"tables={report.tables:<3} figures={report.figures:<3} "
                f"{prov:9} {flags}"
            )
    return 0


__all__ = [
    "ANOMALY_OCR_COVERAGE",
    "ANOMALY_SCAN_FRACTION",
    "ANOMALY_TABLES_PER_PAGE",
    "PROTECTED_TARGETS",
    "DocReport",
    "MigrationPlan",
    "MigrationRefusedError",
    "assess_document",
    "build_plan",
    "detect_anomalies",
    "run_extraction",
    "select_documents",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
