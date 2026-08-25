"""
COMPAT SHIM — manual ingest for INTERNAL / user-provided documents.

Status: DEPRECATED compatibility wrapper (workspace cleanup, audit §4).
This script is retained so historical muscle memory (`--check` / `--run`)
keeps working, but ALL conversion/append/move semantics now delegate to the
ONE canonical engine (`src/scripts/ingest_folder.py::ingest_folder`) — the
same implementation behind `python -m src.scripts.ingest inbox` and the
frontend /api/ingest. Previously this file carried a second, diverging copy
of the detect→convert→dedup→append flow (own converter dispatch, own
non-atomic corpus append, own processed-move); that duplication is gone.

Canonical replacement (index-aware):
    python -m src.scripts.ingest inbox        # same engine + optional
                                              # incremental index update

Workflow (unchanged contract):
    1. Drop files into  data/inbox/  (PDF, txt, json, jsonl)
    2. Run this script (or the UI "Ingest" button):
         python -m src.scripts.ingest_inbox --run
    3. New files are converted by the engine (table-aware PDF extraction,
       OCR fallback for scanned PDFs, QA/knowledge/document JSON and jsonl),
       appended to the corpus with deterministic dedup, and ingested files
       move to  data/inbox/processed/.
    4. The index is NOT rebuilt here (same as the historical script) — the
       message at the end points at the canonical index-aware command.

Modes:
    --check : list what's pending in the inbox (no changes)
    --run   : ingest everything pending

Behavioral note vs. the historical implementation: processed-move now
follows CANONICAL engine semantics exactly (converted PDF/TXT/MD move to
processed/; JSON/JSONL inputs are not moved — matching `ingest inbox` and
the UI). Everything else (what counts as ingestible, dedup, ids) is by
definition identical, because there is only one implementation.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

from src.utils.app_paths import data_dir, inbox_dir

INBOX = inbox_dir()
PROCESSED = INBOX / "processed"
LOG = data_dir() / "sync.log"

_DEPRECATION = (
    "src.scripts.ingest_inbox is deprecated (compat shim). Use "
    "`python -m src.scripts.ingest inbox` — same engine, plus the index update."
)


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pending_files() -> list[Path]:
    if not INBOX.exists():
        return []
    return [p for p in sorted(INBOX.iterdir()) if p.is_file()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="List pending inbox files")
    group.add_argument("--run", action="store_true", help="Ingest pending inbox files")
    args = ap.parse_args()

    files = pending_files()
    if not files:
        log("Inbox is empty — nothing pending.")
        return

    if args.check:
        log(f"{len(files)} pending file(s) in {INBOX}:")
        for p in files:
            log(f"  {p.name} ({p.stat().st_size//1024} KB)")
        return

    # --run — delegate to the canonical engine (single implementation).
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=1)
    print(f"  [deprecated] {_DEPRECATION}", file=sys.stderr)
    log(f"=== ingest_inbox --run ({len(files)} files) — delegating to the "
        "canonical engine (ingest_folder) ===")

    from src.scripts import ingest_folder as _engine

    res = _engine.ingest_folder(str(INBOX), move_processed=True)

    log(f"=== inbox done: {res.get('files', 0)} file(s) processed, "
        f"{res.get('added', 0)} record(s) added, {res.get('failed', 0)} failed ===")
    if res.get("added"):
        log("NEXT (index not touched by this shim): "
            "python -m src.scripts.ingest inbox   # incremental index update, "
            "or use the UI Ingest button")


if __name__ == "__main__":
    main()
