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
    _DEFAULT_MINISTRY,
)
from src.scripts.detect_doc_type import detect_doc_type, readable_type

# Project-root / APP_* paths (never CWD). Same convention as the server.
from src.utils.app_paths import data_dir, index_dir, project_root

_PROJECT_ROOT = project_root()
CORPUS = data_dir() / "corpus_reports.jsonl"
LOG = data_dir() / "sync.log"
INDEX_DIR = str(index_dir())

KNOWN_FOLDERS = [
    data_dir() / "inbox",
    data_dir() / "annual_reports",
    data_dir() / "incois_reports" / "AnnualReports",
    data_dir() / "incois_reports" / "Others",
    data_dir() / "incois_reports" / "TechnicalReports",
    data_dir() / "incois_reports" / "ResearchPublications",
    data_dir() / "moes_reports" / "knowledge",
    data_dir() / "scanned_ocr",
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


def _ctx_kwargs(meta_context: dict | None) -> dict:
    """Convert a per-source meta_context into converter kwargs.

    meta_context is None for legacy flat callers (server inbox ingest,
    ingest_folder --folder): the kwargs then equal the converter defaults —
    byte-identical records to before. Hierarchical sources (src/scripts/
    ingest.py) pass {org, source, ministry, default_ministry, doc_type_hint}.
    """
    ctx = meta_context or {}
    return {
        "org": ctx.get("org"),
        "source": ctx.get("source"),
        "ministry": ctx.get("ministry"),
        "default_ministry": ctx["default_ministry"] if "default_ministry" in ctx
                            else _DEFAULT_MINISTRY,
    }


def convert_one_detected(path: Path, out: list, seen: set[str], move_after: bool,
                         meta_context: dict | None = None) -> int:
    """Convert one file using smart type detection. Returns records added."""
    ctx = _ctx_kwargs(meta_context)
    if path.suffix.lower() == ".jsonl":
        # QA pairs jsonl — always audit_qa (folder/content don't override);
        # org/source/ministry context still propagates for provenance.
        return convert_qa_dataset(path, out, seen, **ctx)

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            return 0
        if isinstance(data, list):
            # QA array or document list
            if data and isinstance(data[0], dict) and ("Question" in data[0] or "question" in data[0]):
                return convert_qa_dataset(path, out, seen, **ctx)
            return convert_document_json(path, out, seen, **ctx)
        if isinstance(data, dict):
            if "knowledge_extraction" in data:
                return convert_knowledge_json(path, out, seen, **ctx)
            if data.get("content") or data.get("title"):
                return convert_document_json(path, out, seen, **ctx)
            if any(k in data for k in ("data", "qa", "questions")):
                return convert_qa_dataset(path, out, seen, **ctx)
        return 0

    # PDF / TXT / MD — smart type detection (category hint from the source
    # registry path, e.g. moes/incois/annual_reports/ -> annual_report, sits
    # below content but above legacy folder/filename heuristics).
    text_peek = _peek_text(path) if path.suffix.lower() in (".pdf", ".txt", ".md") else ""
    doc_type = detect_doc_type(
        path, text_peek,
        category_hint=(meta_context or {}).get("doc_type_hint"),
    )
    if path.suffix.lower() == ".pdf":
        n = convert_pdf_file(path, out, seen, doc_type=doc_type, **ctx)
    else:
        n = convert_text_file(path, out, seen, doc_type=doc_type, **ctx)
    if n > 0 and move_after:
        proc = path.parent / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(proc / path.name))
    return n


def ingest_folder(folder: str, move_processed: bool = False,
                  meta_context: dict | None = None,
                  only_files: set[str] | None = None) -> dict:
    """Convert every file in a folder, append new records to the corpus.

    ``meta_context`` is an additive per-source identity for hierarchical
    ingestion (src/scripts/ingest.py): {org, source, ministry,
    default_ministry, doc_type_hint}. None (all legacy callers) produces
    byte-identical records to before.

    ``only_files`` (additive, Phase 3) scopes the run to a subset of file
    NAMES — the frontend upload flow ingests exactly the files it just
    staged instead of re-converting the whole leaf. None (default) keeps the
    whole-folder scan unchanged. The .txt-next-to-.pdf sibling skip rule
    always considers the FULL folder (a staged .txt is skipped when its PDF
    is on disk even outside the subset).
    """
    p = Path(folder)
    if not p.exists():
        log(f"[ingest_folder] folder not found: {folder}")
        return {"files": 0, "added": 0, "failed": 0}

    all_files = sorted(f for f in p.iterdir() if f.is_file())
    if only_files is not None:
        files = [f for f in all_files if f.name in only_files]
    else:
        files = all_files
    if not files:
        scope = f" (filter matched 0 of {len(all_files)} on disk)" if only_files is not None else ""
        log(f"[ingest_folder] no files in {folder}{scope}")
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
    # the sibling-skip rule consults the FULL folder (see docstring), while
    # conversion below is scoped to `files` (== all_files unless only_files)
    pdf_stems = {_stem(f) for f in all_files if f.suffix.lower() == ".pdf"}
    for f in files:
        if f.suffix.lower() == ".txt" and _stem(f) in pdf_stems:
            log(f"  skip {f.name} (duplicate of its .pdf)")
            continue
        try:
            before = len(out)
            n = convert_one_detected(f, out, seen, move_processed, meta_context)
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
        from src.utils.atomic_io import append_jsonl_atomic

        lines = []
        for rec in out:
            if hasattr(rec, "model_dump_json"):
                lines.append(rec.model_dump_json())
            else:
                lines.append(json.dumps(rec, ensure_ascii=False))
        append_jsonl_atomic(CORPUS, lines)
        log(f"[ingest_folder] appended {len(out)} record(s) -> {CORPUS}")

    return {"files": len(files), "added": len(out), "failed": fail, "types": types_used}


def _index_exists() -> bool:
    """True if a loadable index is saved (all marker files present)."""
    idx = Path(INDEX_DIR)
    return all((idx / f).exists() for f in (
        "vector_store.index", "bm25_index.pkl", "doc_map.json", "pipeline_metadata.json",
    ))


def _incremental_child_code() -> str:
    """The child-process program for the incremental index update.

    MUST be a real multi-line script: semicolon-joining is only legal between
    SIMPLE statements — ``n = p.add_records(recs); if n: ...`` is a
    SyntaxError at compile time (this exact regression shipped and was masked
    by the old auto-fallback to a full rebuild). The step sequence itself is
    unchanged: load index -> load corpus -> add_records (embeds ONLY ids not
    already indexed) -> save iff anything was added -> report INCR_ADDED.
    Paths go through repr(), which always yields a valid string literal.
    """
    return (
        "from src.retrieval.hybrid.pipeline import HybridRAGPipeline\n"
        "from src.data.loader import DataLoader\n"
        f"p = HybridRAGPipeline()\n"
        f"p.load({str(INDEX_DIR)!r})\n"
        f"recs = DataLoader.load_jsonl({str(CORPUS)!r})\n"
        "n = p.add_records(recs)\n"
        f"if n: p.save({str(INDEX_DIR)!r})\n"
        "print(f'INCR_ADDED={n}')\n"
    )


def incremental_update() -> None:
    """Load the existing index, embed ONLY new records, add, save.

    This is the fast path — no full re-embed of the whole corpus. Only
    records whose id isn't in the index get embeddings (bge-m3), added to
    FAISS, and BM25 is rebuilt (text-only, fast).

    Failure policy: a failed child run NEVER falls back to a full rebuild.
    The corpus is untouched either way (append happened before this step), so
    the safe recovery is to re-run the command (dedup makes that a no-op for
    already-appended records) or to pass --full-rebuild explicitly. Raises
    RuntimeError on failure so the CLI exits non-zero.
    """
    log("incremental index update (embeddings for NEW records only)...")
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-c", _incremental_child_code()],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=3600,
    )
    out = (r.stdout or "") + (r.stderr or "")
    tail = out[-800:].replace("\n", " | ")
    log(tail)
    if r.returncode != 0:
        log("incremental update FAILED — index NOT updated; "
            "no automatic full rebuild (use --full-rebuild explicitly)")
        raise RuntimeError(
            "incremental index update failed (see sync.log tail above). "
            "Corpus is intact (append-only); the index was left unchanged. "
            "Re-run the command (dedup makes re-runs safe) or rebuild with "
            "--full-rebuild."
        )
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
        move = args.move_processed or ("inbox" in folder)
        res = ingest_folder(folder, move_processed=move)
        total_added += res["added"]

    if total_added > 0 and not args.no_rebuild:
        try:
            if args.full_rebuild or not _index_exists():
                rebuild_index()
            else:
                incremental_update()
        except RuntimeError as e:
            log(f"[ingest_folder] ERROR: {e}")
            sys.exit(3)
    elif total_added == 0:
        log("nothing new ingested — no index update needed")
    else:
        log("--no-rebuild given; index NOT updated (run rebuild later)")


if __name__ == "__main__":
    main()
