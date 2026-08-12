"""
One-time corpus repair: fix document_type of mislabeled scanned annual reports
+ dedupe near-identical records (the same scanned report was ingested once
from the OCR .txt and once from the PDF, producing two records).

What it does:
  1. Any record whose subject starts with 'AR_' or 'INCOIS Annual Report' and
     document_type == 'document'  ->  annual_report
  2. Dedupe: records with the SAME (normalized) answer text keep the first,
     the duplicate is dropped.

Usage (server stopped, then rebuild):
    python -m src.scripts.repair_corpus --corpus data/corpus_reports.jsonl
    retrieve build --data data/corpus_reports.jsonl --rebuild
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_reports.jsonl")
    args = ap.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"corpus not found: {path}")
        return

    records = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue

    relabeled = 0
    for rec in records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        subj = str(meta.get("subject") or "")
        if meta.get("document_type") == "document" and (
            subj.startswith("AR_") or "INCOIS Annual Report" in subj
        ):
            meta["document_type"] = "annual_report"
            relabeled += 1

    # dedupe by normalized answer text (keep first occurrence)
    seen: set[str] = set()
    deduped = []
    dropped = 0
    for rec in records:
        key = norm(rec.get("answer_text") or rec.get("answer") or "")
        if not key:
            deduped.append(rec)
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(rec)

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in deduped) + "\n",
        encoding="utf-8",
    )
    print(f"relabeled to annual_report: {relabeled}")
    print(f"duplicates dropped: {dropped}")
    print(f"total: {len(records)} -> {len(deduped)} records -> {path}")


if __name__ == "__main__":
    main()
