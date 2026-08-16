"""
Manual ingest for INTERNAL / user-provided documents (sir's files, scientists'
documents). These are NEVER auto-scraped — someone puts files into the inbox
folder and triggers ingest.

Workflow:
    1. Scientist/sir copies files into  data/inbox/   (PDF, txt, json, jsonl)
    2. Run this script (or click the UI "Ingest" button):
         python -m src.scripts.ingest_inbox --run
    3. Anything new in inbox is converted (PDF -> text/OCR, QA jsonl ->
       records, knowledge JSON -> records, document JSON -> records), appended
       to the corpus (deduped), and the index is rebuilt.
    4. Processed files move to  data/inbox/processed/  (so they aren't
       re-ingested next time).

Modes:
    --check : list what's pending in the inbox (no changes)
    --run   : ingest everything pending

Supported files in the inbox (auto-detected by extension + content):
    *.pdf             -> text extract, OCR fallback if scanned
    *.txt / *.md      -> document record
    *.jsonl           -> QA pairs {Question,Answer} or QARecord lines
    *.json            -> {title,content} document OR knowledge_extraction JSON
                         OR {Question,Answer} array
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.utils.app_paths import inbox_dir, corpus_path, data_dir

INBOX = inbox_dir()
PROCESSED = INBOX / "processed"
CORPUS = corpus_path()
LOG = data_dir() / "sync.log"


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


def convert_one(path: Path, out: list, seen: set[str]) -> int:
    """Convert one inbox file by delegating to convert_sirs_knowledge's
    shared converters — NO inline re-implementation (this was the source of
    the non-deterministic hash() ids). All records get deterministic ids +
    ministry=EARTH SCIENCES from _make_record."""
    from src.scripts.convert_sirs_knowledge import (
        convert_qa_dataset,
        convert_knowledge_json,
        convert_document_json,
        convert_text_file,
        convert_pdf_file,
    )

    import json as _json

    if path.suffix.lower() in (".jsonl", ".json"):
        try:
            data = _json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001 — jsonl won't parse as a whole
            data = None
        if isinstance(data, list) or path.suffix.lower() == ".jsonl":
            # QA array or jsonl of QA pairs (convert_qa_dataset handles both)
            return convert_qa_dataset(path, out, seen)
        if isinstance(data, dict):
            if "knowledge_extraction" in data:
                return convert_knowledge_json(path, out, seen)
            if data.get("content") or data.get("title"):
                return convert_document_json(path, out, seen)
            # dict-shaped QA wrapper {questions: [...]}
            if any(k in data for k in ("data", "qa", "questions")):
                return convert_qa_dataset(path, out, seen)
        return 0

    if path.suffix.lower() in (".txt", ".md"):
        return convert_text_file(path, out, seen)
    if path.suffix.lower() == ".pdf":
        return convert_pdf_file(path, out, seen)
    return 0


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

    # --run
    log(f"=== ingest_inbox --run ({len(files)} files) ===")
    out: list[dict] = []
    seen: set[str] = set()

    # seed 'seen' with existing corpus ids so we don't duplicate
    if CORPUS.exists():
        for line in open(CORPUS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("question_id"):
                    seen.add(rec["question_id"])
            except Exception:  # noqa: BLE001
                continue

    new_ids: set[str] = set()
    ok = fail = 0
    for p in files:
        try:
            n = convert_one(p, out, seen)
            if n > 0:
                ok += 1
                new_ids.add(p.name)
                log(f"  ingested {p.name} (+{n} record(s))")
            else:
                fail += 1
                log(f"  WARN {p.name}: no records extracted")
        except Exception as e:  # noqa: BLE001
            fail += 1
            log(f"  ERROR {p.name}: {e}")

    if out:
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS.open("a", encoding="utf-8") as f:
            for rec in out:
                # convert QARecord objects to plain dicts for JSONL
                if hasattr(rec, "model_dump_json"):
                    f.write(rec.model_dump_json() + "\n")
                elif hasattr(rec, "model_dump"):
                    f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"Appended {len(out)} record(s) -> {CORPUS}")

    # move successfully ingested files to processed/
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for p in files:
        if p.name in new_ids:
            shutil.move(str(p), str(PROCESSED / p.name))

    log(f"=== inbox done: {ok} ok, {fail} failed, {len(out)} records added ===")
    log("NEXT: rebuild the index (server stopped): "
        "retrieve build --data data/corpus_reports.jsonl --rebuild")


if __name__ == "__main__":
    main()
