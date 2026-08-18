"""Audit metadata.date coverage in the live hybrid index / corpus — READ-ONLY.

Motivation: the temporal-awareness remediation (R1/R2) can only surface dates
that were actually written at ingest. Date population is ingest-path-dependent
(sansad scrape: page date; inbox QA datasets: None; inbox PDFs: None; annual
reports: filename year-range; report PDFs: filename year-or-None), so this
script measures what YOUR index actually holds BEFORE anyone relies on dates.

Stdlib-only (no faiss / sentence-transformers / pydantic needed): it parses
the plain-JSON index maps directly and never writes anything.

Usage (run where the index lives — the PC, or HPC mount):

    python -m src.scripts.audit_index_dates
    python -m src.scripts.audit_index_dates --index-dir storage/hybrid_rag
    python -m src.scripts.audit_index_dates --corpus data/corpus_reports.jsonl

Defaults: --index-dir resolves via src.utils.app_paths.index_dir()
($APP_INDEX_DIR or <project>/storage/hybrid_rag).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Keep in sync with src/generation/evidence.py::_YEAR_RE / doc_signal_year.
# (Duplicated deliberately: importing src.generation pulls the heavy
# src.retrieval chain; this script must stay stdlib-only and side-effect-free.)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _years(stamp: str | None) -> list[int]:
    return sorted(int(m.group(1)) for m in _YEAR_RE.finditer(stamp or ""))


def _format_class(stamp: str | None) -> str:
    if stamp is None or not str(stamp).strip():
        return "ABSENT"
    s = str(stamp).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "ISO YYYY-MM-DD"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return "YYYY-MM"
    if re.fullmatch(r"\d{4}", s):
        return "YYYY"
    if re.fullmatch(r"\d{4}-\d{2}\s*$", s):
        return "YYYY-YY (FY range)"
    if _YEAR_RE.search(s):
        return "other string containing a year"
    return "other / unparseable"


def _iter_records(index_dir: Path | None, corpus_jsonl: Path | None):
    """Yield (doc_id, metadata_dict). Index maps take priority when given."""
    if corpus_jsonl is not None:
        if not corpus_jsonl.exists():
            print(f"[audit] corpus file not found: {corpus_jsonl}", file=sys.stderr)
            return
        with corpus_jsonl.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[audit] {corpus_jsonl.name}:{ln} — unparseable line, skipped")
                    continue
                yield rec.get("question_id", f"line-{ln}"), rec.get("metadata") or {}
        return

    if index_dir is None:
        return
    doc_map_path = index_dir / "doc_map.json"
    if not doc_map_path.exists():
        print(f"[audit] no doc_map.json under {index_dir}", file=sys.stderr)
        return
    with doc_map_path.open("r", encoding="utf-8") as f:
        doc_map = json.load(f)  # orjson output is standard JSON
    for doc_id, rec in doc_map.items():
        yield doc_id, (rec or {}).get("metadata") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", default=None,
                    help="hybrid index dir (default: $APP_INDEX_DIR or storage/hybrid_rag)")
    ap.add_argument("--corpus", default=None,
                    help="audit a corpus JSONL instead of the index")
    ap.add_argument("--samples", type=int, default=8,
                    help="sample doc_ids to print per document_type (default 8)")
    args = ap.parse_args()

    index_dir: Path | None = None
    corpus: Path | None = None
    if args.corpus:
        corpus = Path(args.corpus).expanduser().resolve()
        print(f"[audit] source: corpus JSONL {corpus}")
    else:
        if args.index_dir:
            index_dir = Path(args.index_dir).expanduser().resolve()
        else:
            from src.utils.app_paths import index_dir as resolve_index_dir
            index_dir = resolve_index_dir()
        print(f"[audit] source: index maps under {index_dir}")

    total = 0
    dated = 0
    fmt_counts: Counter[str] = Counter()
    type_counts: dict[str, list[int]] = {}   # doc_type -> [total, dated]
    year_hist: Counter[int] = Counter()
    undated_by_type: dict[str, list[str]] = {}
    dated_by_type: dict[str, list[str]] = {}
    year_min, year_max = None, None

    for doc_id, meta in _iter_records(index_dir, corpus):
        total += 1
        doc_type = str(meta.get("document_type") or "unknown").lower()
        stamp = meta.get("date")
        cls = _format_class(stamp)
        fmt_counts[cls] += 1
        row = type_counts.setdefault(doc_type, [0, 0])
        row[0] += 1
        if cls != "ABSENT":
            dated += 1
            row[1] += 1
            dated_by_type.setdefault(doc_type, []).append(doc_id)
            ys = _years(str(stamp))
            for y in ys:
                year_hist[y] += 1
            if ys:
                year_min = ys[0] if year_min is None else min(year_min, ys[0])
                year_max = ys[-1] if year_max is None else max(year_max, ys[-1])
        else:
            undated_by_type.setdefault(doc_type, []).append(doc_id)

    if total == 0:
        print("[audit] 0 records found — check the path. Nothing else to report.")
        return 1

    pct = 100.0 * dated / total
    print("=" * 72)
    print(f"records: {total}    with date: {dated} ({pct:.1f}%)    without: {total - dated}")
    if year_min is not None:
        print(f"year span from stamps: {year_min} .. {year_max}")
    print("=" * 72)
    print("\n-- date format classes ----------------------------------------------")
    for cls, n in fmt_counts.most_common():
        print(f"  {cls:34s} {n:6d}  ({100.0 * n / total:5.1f}%)")
    print("\n-- coverage by document_type -----------------------------------------")
    for dt, (t, d) in sorted(type_counts.items()):
        print(f"  {dt:24s} {d:6d}/{t:<6d} dated  ({100.0 * d / t:5.1f}%)")
    print("\n-- signal-year histogram (top 30) ------------------------------------")
    for y, n in sorted(year_hist.items()):
        bar = "#" * max(1, int(60 * n / max(year_hist.values())))
        print(f"  {y}  {n:6d}  {bar}")
    print("\n-- samples ------------------------------------------------------------")
    for dt in sorted(type_counts):
        undated = undated_by_type.get(dt, [])[: args.samples]
        dated_s = dated_by_type.get(dt, [])[: args.samples]
        print(f"  [{dt}] undated: {undated or '—'}")
        print(f"  [{dt}] dated  : {dated_s or '—'}")
    print("\nREAD-ONLY: no files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
