"""
Unified "ingest from ANY folder" — the one script scientists run.

Scientists put files in ANY folder (frontend upload -> data/inbox/, or the
backend's annual_reports / incois_reports/<section> / moes_reports / a brand
new folder they created). Then ONE command:

    python -m src.scripts.ingest_folder --folder <path>

What it does, automatically:
  1. SMART TYPE DETECTION per file (detect_doc_type):
       folder name -> filename pattern (AR_/TR_/RP_/Report_) -> content header
     -> annual_report | technical_report | research_publication |
        general_report | audit_qa | document
  2. Convert (PDF -> text/OCR-aware, txt, json, jsonl QA) with the detected type
  3. Append to data/corpus_reports.jsonl (dedup by deterministic id)
  4. REBUILD the index (embeddings created) so docs are immediately queryable
     (skipped only if --no-rebuild; needs the ML env for sentence-transformers)

Also:
    --folder data/inbox           -> ingest what the UI upload saved
    --all-known                   -> scan inbox + annual_reports + incois_reports/*
                                   + moes_reports/knowledge
    --move-processed              -> move successfully ingested files to
                                   <folder>/processed (default for inbox)
    --no-rebuild                  -> skip the embedding/index step
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.models.qa_record import QARecord
from src.scripts.convert_sirs_knowledge import (
    convert_qa_dataset,
    convert_knowledge_json,
    convert_document_json,
    convert_text_file,
    convert_pdf_file,
)
from src.scripts.detect_doc_type import detect_doc_type, readable_type

# Project-root-relative paths (never CWD) — same convention as the server's
# PROJECT_ROOT, so ingest works from any working directory (CLI or in-process).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = _PROJECT_ROOT / "data" / "corpus_reports.jsonl"
LOG = _PROJECT_ROOT / "data" / "sync.log"
INDEX_DIR = str(_PROJECT_ROOT / "storage" / "hybrid_rag")

KNOWN_FOLDERS = [
    _PROJECT_ROOT / "data" / "inbox",
    _PROJECT_ROOT / "data" / "annual_reports",
    _PROJECT_ROOT / "data" / "incois_reports" / "AnnualReports",
    _PROJECT_ROOT / "data" / "incois_reports" / "Others",
    _PROJECT_ROOT / "data" / "incois_reports" / "TechnicalReports",
    _PROJECT_ROOT / "data" / "incois_reports" / "ResearchPublications",
    _PROJECT_ROOT / "data" / "moes_reports" / "knowledge",
    _PROJECT_ROOT / "data" / "scanned_ocr",
]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _peek_text(path: Path) -> str:
    """Small text sample for content-based type detection (PDF first pages)."""
    if path.suffix.lower() in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:800]
        except Exception:  # noqa: BLE001
            return ""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            parts = []
            for p in reader.pages[:3]:
                t = (p.extract_text() or "").strip()
                if t:
                    parts.append(t)
                    if sum(len(x) for x in parts) > 800:
                        break
            peek = " ".join(parts)[:800]
            if peek.strip():
                return peek
        except Exception:  # noqa: BLE001
            pass
        # scanned PDF: pypdf gives no text — OCR a page so type detection
        # sees the actual content (e.g. "ANNUAL REPORT 2028" -> annual_report)
        try:
            from src.scripts.convert_sirs_knowledge import _ocr_pdf_text

            ocr = _ocr_pdf_text(path)
            return ocr[:800]
        except Exception:  # noqa: BLE001
            return ""
    return ""


def convert_one_detected(path: Path, out: list, seen: set[str], move_after: bool) -> int:
    """Convert one file using smart type detection. Returns records added."""
    if path.suffix.lower() == ".jsonl":
        # QA pairs jsonl — always audit_qa (folder/content don't override)
        return convert_qa_dataset(path, out, seen)

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            return 0
        if isinstance(data, list):
            # QA array or document list
            if data and isinstance(data[0], dict) and ("Question" in data[0] or "question" in data[0]):
                return convert_qa_dataset(path, out, seen)
            return convert_document_json(path, out, seen)
        if isinstance(data, dict):
            if "knowledge_extraction" in data:
                return convert_knowledge_json(path, out, seen)
            if data.get("content") or data.get("title"):
                return convert_document_json(path, out, seen)
            if any(k in data for k in ("data", "qa", "questions")):
                return convert_qa_dataset(path, out, seen)
        return 0

    # PDF / TXT / MD — smart type detection
    text_peek = _peek_text(path) if path.suffix.lower() in (".pdf", ".txt", ".md") else ""
    doc_type = detect_doc_type(path, text_peek)
    if path.suffix.lower() == ".pdf":
        n = convert_pdf_file(path, out, seen, doc_type=doc_type)
    else:
        n = convert_text_file(path, out, seen, doc_type=doc_type)
    if n > 0 and move_after:
        proc = path.parent / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(proc / path.name))
    return n


def ingest_folder(folder: str, move_processed: bool = False) -> dict:
    """Convert every file in a folder, append new records to the corpus."""
    p = Path(folder)
    if not p.exists():
        log(f"[ingest_folder] folder not found: {folder}")
        return {"files": 0, "added": 0, "failed": 0}

    files = sorted(f for f in p.iterdir() if f.is_file())
    if not files:
        log(f"[ingest_folder] no files in {folder}")
        return {"files": 0, "added": 0, "failed": 0}

    log(f"[ingest_folder] scanning {folder}: {len(files)} file(s)")
    out: list = []
    seen: set[str] = set()

    # seed seen with existing corpus ids
    if CORPUS.exists():
        for line in open(CORPUS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("question_id"):
                    seen.add(r["question_id"])
            except Exception:  # noqa: BLE001
                continue

    ok = fail = 0
    types_used: dict[str, int] = {}
    # Crawl (crawl_incois_reports) writes a .txt next to each .pdf — skip the
    # .txt when its .pdf sibling exists so each report is ingested exactly
    # once (from the PDF, which keeps full fidelity).
    def _stem(f: Path) -> str:
        name = f.name
        return name.rsplit(".", 1)[0] if "." in name else name
    pdf_stems = {_stem(f) for f in files if f.suffix.lower() == ".pdf"}
    for f in files:
        if f.suffix.lower() == ".txt" and _stem(f) in pdf_stems:
            log(f"  skip {f.name} (duplicate of its .pdf)")
            continue
        try:
            before = len(out)
            n = convert_one_detected(f, out, seen, move_processed)
            if n > 0:
                ok += 1
                # log the ACTUAL record type(s) added by this file
                added_types = sorted({r.metadata.document_type for r in out[before:]})
                t = readable_type(added_types[0]) if added_types else "unknown"
                for at in added_types:
                    types_used[at] = types_used.get(at, 0) + 1
                log(f"  ingested {f.name} (+{n}, type={t})")
            else:
                fail += 1
                log(f"  WARN {f.name}: no records extracted")
        except Exception as e:  # noqa: BLE001
            fail += 1
            log(f"  ERROR {f.name}: {e}")

    if out:
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS.open("a", encoding="utf-8") as fh:
            for rec in out:
                if hasattr(rec, "model_dump_json"):
                    fh.write(rec.model_dump_json() + "\n")
                else:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"[ingest_folder] appended {len(out)} record(s) -> {CORPUS}")

    return {"files": len(files), "added": len(out), "failed": fail, "types": types_used}


def _index_exists() -> bool:
    """True if a loadable index is saved (all marker files present)."""
    idx = Path(INDEX_DIR)
    return all((idx / f).exists() for f in (
        "vector_store.index", "bm25_index.pkl", "doc_map.json", "pipeline_metadata.json",
    ))


def incremental_update() -> None:
    """Load the existing index, embed ONLY new records, add, save.

    This is the fast path — no full re-embed of the whole corpus. Only
    records whose id isn't in the index get embeddings (bge-m3), added to
    FAISS, and BM25 is rebuilt (text-only, fast)."""
    log("incremental index update (embeddings for NEW records only)...")
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    code = (
        "from src.retrieval.hybrid.pipeline import HybridRAGPipeline; "
        "from src.data.loader import DataLoader; "
        f"p = HybridRAGPipeline(); p.load({str(INDEX_DIR)!r}); "
        f"recs = DataLoader.load_jsonl({str(CORPUS)!r}); "
        "n = p.add_records(recs); "
        f"if n: p.save({str(INDEX_DIR)!r}); "
        "print(f'INCR_ADDED={n}')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=3600,
    )
    out = (r.stdout or "") + (r.stderr or "")
    tail = out[-800:].replace("\n", " | ")
    log(tail)
    if r.returncode != 0:
        log("incremental update FAILED — falling back to full rebuild")
        rebuild_index()
        return
    if "INCR_ADDED=" in out:
        added = out.split("INCR_ADDED=")[-1].split()[0]
        log(f"incremental update OK — {added} new record(s) embedded and added")
    else:
        log("incremental update OK")


def rebuild_index() -> None:
    """Full rebuild — embeddings for the ENTIRE corpus (needs ML env).
    Use for first build or --full-rebuild."""
    log("full index rebuild (embeddings for all records)...")
    # Windows cp1252 console can't print rich's unicode arrows (→) and
    # crashes the child process with UnicodeEncodeError. Force UTF-8 output
    # on the subprocess so the rebuild never dies on a console encoding issue.
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-m", "src.retrieval.cli", "build",
         "--data", str(CORPUS), "--rebuild",
         "--output", str(INDEX_DIR)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=3600,
    )
    tail = (r.stdout or "")[-600:]
    log(tail.replace("\n", " | "))
    if r.returncode != 0:
        log(f"index rebuild FAILED: {(r.stderr or '')[-300:]}")
    else:
        log("index rebuild OK — new docs are queryable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default=None, help="Any folder to ingest")
    ap.add_argument("--all-known", action="store_true",
                    help="Scan all known folders (inbox, annual, incois sections, moes)")
    ap.add_argument("--move-processed", action="store_true",
                    help="Move ingested files to <folder>/processed (default for inbox)")
    ap.add_argument("--no-rebuild", action="store_true", help="Skip index update")
    ap.add_argument("--full-rebuild", action="store_true",
                    help="Force FULL rebuild of the whole index (slow)")
    args = ap.parse_args()

    folders = []
    if args.folder:
        folders.append(args.folder)
    elif args.all_known:
        folders = KNOWN_FOLDERS
    else:
        print("Pass --folder <path> or --all-known")
        sys.exit(1)

    total_added = 0
    for folder in folders:
        move = args.move_processed or ("inbox" in str(folder).lower())
        res = ingest_folder(folder, move_processed=move)
        total_added += res["added"]

    if total_added > 0 and not args.no_rebuild:
        if args.full_rebuild or not _index_exists():
            rebuild_index()
        else:
            incremental_update()
    elif total_added == 0:
        log("nothing new ingested — no index update needed")
    else:
        log("--no-rebuild given; index NOT updated (run rebuild later)")


if __name__ == "__main__":
    main()
