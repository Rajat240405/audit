"""
Diagnose source-filter counts — why is an org (e.g. INCOIS) showing 0?

Reads data/corpus_reports.jsonl (the corpus the /api/sources catalogue reads)
and prints, with NO heavy deps:

  1. total rows + per document_type
  2. rows per derived org + category (same derivation as /api/sources)
  3. every NON-parliamentary row: question_id, document_type, subject,
     source_url (the identity fields org detection uses)
  4. whether "INCOIS" appears anywhere in subject/question_text/source_url
  5. whether data/incois_reports/ exists and how many files it holds
     (if yes but 0 corpus rows -> the reports were never ingested)

Usage:
    python -m src.scripts.diagnose_sources
    python -m src.scripts.diagnose_sources --corpus data/corpus_reports.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.retrieval.frontend.org_tree import derive_category, derive_org


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_reports.jsonl")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"[missing] {corpus} — run ingestion first")
        return

    rows = []
    for line in corpus.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass

    print(f"TOTAL ROWS        : {len(rows)}")

    dtypes: Counter = Counter()
    orgs: Counter = Counter()
    cats: Counter = Counter()
    incois_identity = 0
    for rec in rows:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        dtypes[meta.get("document_type") or "document"] += 1
        mf = dict(meta)
        mf["question_text"] = rec.get("question_text") or ""
        orgs[derive_org(mf)] += 1
        cats[derive_category(mf)] += 1
        ident = " ".join(str(x) for x in (
            meta.get("subject"), meta.get("source_url"),
            rec.get("question_text"), meta.get("title"),
        ) if x).upper()
        if "INCOIS" in ident:
            incois_identity += 1

    print("\nBY DOCUMENT_TYPE  :", dict(dtypes.most_common()))
    print("BY ORG (derived)  :", dict(orgs.most_common()))
    print("BY CATEGORY       :", dict(cats.most_common()))
    print(f"ROWS with 'INCOIS' in identity: {incois_identity}")

    print("\n── NON-PARLIAMENTARY ROWS (org detection inputs) ──")
    shown = 0
    for rec in rows:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        if str(meta.get("document_type") or "").lower() == "parliamentary_qa":
            continue
        shown += 1
        if shown > 40:
            print("  ... (truncated)")
            break
        print(f"  id={rec.get('question_id') or rec.get('doc_id')}")
        print(f"     type={meta.get('document_type')!r}")
        print(f"     subject={str(meta.get('subject'))[:70]!r}")
        print(f"     source_url={str(meta.get('source_url'))[:70]!r}")
        print(f"     question_text={str(rec.get('question_text'))[:70]!r}")
    if shown == 0:
        print("  (none — corpus has ONLY parliamentary rows)")

    # Report folders present?
    root = Path(__file__).resolve().parents[2]
    for d in ("data/incois_reports", "data/scanned_ocr", "data/moes_reports"):
        p = root / d
        if p.exists():
            n = sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".pdf", ".txt"))
            print(f"\n{d}: EXISTS ({n} pdf/txt files)")
        else:
            print(f"\n{d}: MISSING")


if __name__ == "__main__":
    main()
