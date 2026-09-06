#!/usr/bin/env python3
"""One-off correction of the verified misclassified press release (FIX A item 7).

Verified finding #8: exactly one MoES press release is typed ``annual_report``
because its headline mentions "Single Annual Report" as a TOPIC and the
content-first detector (deliberately) fired on it. The classifier is unchanged
by design; this script corrects the single corpus row as DATA.

Match key: the stable staging filename ``01-26756-eng`` inside
``metadata.source_url`` — stable across re-ingests (the content-derived
``question_id`` is NOT, since FIX A titles change question_text). The script
refuses to act unless the matched row is currently ``annual_report`` AND
carries PIB-release evidence in its text.

Safety (mirrors ``purge_hindi_rows.py``):
* Dry-run by default: prints what WOULD change. ``--apply`` writes.
* With ``--apply`` the corpus is copied to
  ``corpus_reports.jsonl.bak-type-<timestamp>`` BEFORE any change.
* Only the matched line is rewritten (re-serialized from a validated
  QARecord); every other line is written back byte-for-byte.
* Idempotent: re-running after the fix reports "already correct", exit 0.

Run AFTER the FIX A re-ingest (a forced reconversion re-types the file by
content, so correcting before re-ingesting would be overwritten), then
rebuild the index (``retrieve build --rebuild``).

Usage:
    python -m src.scripts.fix_misclassified_press_release --corpus data/corpus_reports.jsonl
    python -m src.scripts.fix_misclassified_press_release --corpus data/corpus_reports.jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from src.models.qa_record import QARecord
from src.utils.app_paths import corpus_path

TARGET_FILENAME = "01-26756-eng"
FROM_TYPE = "annual_report"
TO_TYPE = "press_release"


def _is_candidate(rec: dict) -> bool:
    url = str((rec.get("metadata") or {}).get("source_url") or "")
    return TARGET_FILENAME in url


def _has_pib_evidence(rec: dict) -> bool:
    body = f"{rec.get('question_text') or ''}\n{rec.get('answer_text') or ''}".lower()
    return "posted on:" in body


def fix_corpus(path: Path, apply: bool = False) -> tuple[str, int]:
    """Return (status, changed) where status is one of:
    'fixed' | 'already-correct' | 'not-found' | 'refused'."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    matches: list[int] = []
    for i, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001 — never touch unparseable lines
            continue
        if isinstance(rec, dict) and _is_candidate(rec):
            matches.append(i)

    if not matches:
        return "not-found", 0
    if len(matches) > 1:
        print(f"[fix-type] REFUSED: {len(matches)} rows match {TARGET_FILENAME!r} "
              f"(expected exactly 1) — inspect manually.")
        return "refused", 0

    i = matches[0]
    rec = json.loads(raw_lines[i])
    current = (rec.get("metadata") or {}).get("document_type")
    if current == TO_TYPE:
        return "already-correct", 0
    if current != FROM_TYPE or not _has_pib_evidence(rec):
        print(f"[fix-type] REFUSED: row {rec.get('question_id')} has type={current!r} "
              f"or lacks PIB evidence — not the verified row.")
        return "refused", 0

    # Validate through the project's own model before rewriting.
    fixed = QARecord.model_validate(rec)
    assert fixed.metadata is not None
    fixed.metadata.document_type = TO_TYPE
    new_line = fixed.model_dump_json()

    if not apply:
        print(f"[fix-type] DRY RUN: would set {fixed.question_id} "
              f"{FROM_TYPE} -> {TO_TYPE} (backup + write on --apply)")
        return "fixed", 0

    backup = path.with_name(f"{path.name}.bak-type-{time.strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(path, backup)
    raw_lines[i] = new_line
    path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"[fix-type] fixed {fixed.question_id}: {FROM_TYPE} -> {TO_TYPE}")
    print(f"[fix-type] backup: {backup}")
    return "fixed", 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None, help="override corpus path")
    ap.add_argument("--apply", action="store_true", help="write the fix (default: dry run)")
    args = ap.parse_args(argv)

    path = Path(args.corpus) if args.corpus else corpus_path()
    if not path.exists():
        print(f"[fix-type] ERROR: corpus not found: {path}")
        return 1
    status, _ = fix_corpus(path, apply=args.apply)
    print(f"[fix-type] status: {status}")
    return 0 if status in ("fixed", "already-correct", "not-found") else 1


if __name__ == "__main__":
    sys.exit(main())
