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
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
    """Small text sample for content-based type detection (PDF first pages).

    Deliberately a cheap 3-page pypdf sampler, NOT the corpus extractor:
    detection only needs the document header, and the actual record text is
    produced later by the table-aware shared stack
    (src/data/pdf_table_extract.extract_pdf_text_with_fallback — audit IW-7).
    """
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
        # scanned PDF: pypdf gives no text — OCR the FIRST FEW pages so type
        # detection sees the actual content (e.g. "ANNUAL REPORT 2028" ->
        # annual_report). Capped at 3: the peek this feeds is truncated to
        # 800 chars anyway, and without the cap every scanned PDF was OCR'd
        # twice per run (once here, once in the real conversion) — the single
        # biggest cost in the nightly crawl.
        try:
            from src.scripts.convert_sirs_knowledge import _ocr_pdf_text

            ocr = _ocr_pdf_text(path, max_pages=3)
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


def qa_content_hash(rec) -> str:
    """Deterministic full-sha256 content hash of a QARecord.

    Canonical identity for changed-record detection (re-exported by
    src/scripts/ingest.py as ``_qa_content_hash``). Covers question_id,
    question_text, answer_text and the full metadata dump. EXCLUDES
    scraped_at (volatile — a re-crawl of unchanged content must hash
    identically) and the ``content_hash`` computed field (a
    question_text-only projection — redundant noise here).

    64-char hex (full digest — hash collisions in a 2.6k-row corpus are a
    non-issue, but the full digest keeps this safe for arbitrary growth).
    """
    d = rec.model_dump(mode="json", exclude={"scraped_at"})
    d.pop("content_hash", None)
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# English-only corpus policy
#
# MoES/INCOIS publish every press release in two language variants that land
# side by side in the same crawl folder: ``<id>-eng.pdf`` and ``<id>-hin.pdf``.
# The searchable corpus is English-only, and language is NOT a filter axis in
# this system: a record carries no language field, retrieval has no language
# filter and the UI has none either — so a Hindi record that reached the corpus
# is indistinguishable from an English one and surfaces in answers.
#
# Exclusion therefore happens at the source: language-variant files are skipped
# BEFORE conversion, so they are never OCR'd, never embedded, never indexed.
#
# Default ON. Overrides (highest precedence first):
#   * ``exclude_globs=`` argument            (source registry / direct caller)
#   * ``INGEST_EXCLUDE_GLOBS`` env var       (comma-separated; replaces defaults)
#   * ``INGEST_ALLOW_HINDI=1`` env var       (one-run escape hatch: off)
#   * ``DEFAULT_EXCLUDE_GLOBS``              (built-in English-only default)
#
# Already-ingested Hindi rows are NOT removed here — use
# ``python -m src.scripts.purge_hindi_rows`` (one-off cleanup) followed by a
# full index rebuild.
# ─────────────────────────────────────────────────────────────────────────────

# "*-hin" (extension-less) is included so ids/filenames such as
# "01-28247-hin" are covered too.
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = ("*-hin.*", "*-hin")


def _active_exclude_globs(exclude_globs: tuple[str, ...] | None = None
                          ) -> tuple[str, ...]:
    """Filename globs that must never be converted (English-only policy)."""
    if exclude_globs is not None:
        return tuple(g for g in exclude_globs if g)
    env = os.environ.get("INGEST_EXCLUDE_GLOBS")
    if env is not None:
        return tuple(g.strip() for g in env.split(",") if g.strip())
    if os.environ.get("INGEST_ALLOW_HINDI", "") in ("1", "true", "True"):
        return ()
    return DEFAULT_EXCLUDE_GLOBS


def _glob_excluded(name: str, globs: tuple[str, ...]) -> bool:
    """Case-insensitive filename glob match (``*.PDF`` == ``*.pdf``)."""
    if not globs:
        return False
    low = (name or "").lower()
    return any(fnmatch.fnmatch(low, g.lower()) for g in globs)


def _unchanged_skip_enabled() -> bool:
    """Global escape hatch: INGEST_SKIP_UNCHANGED=0 disables the OCR-cost
    guard and restores the legacy convert-every-file scan."""
    import os

    return os.environ.get("INGEST_SKIP_UNCHANGED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _parse_ts(value) -> float | None:
    """Corpus ``scraped_at`` -> epoch seconds, or None when unusable.

    Accepts the ISO strings QARecord serialises ("2026-08-28T05:38:21+00:00",
    "...Z", or naive) and datetime objects. None makes the OCR-cost guard
    CONVERT the file (fail-open: worst case is today's re-OCR behaviour).
    """
    if value is None:
        return None
    try:
        if hasattr(value, "timestamp"):          # datetime
            return float(value.timestamp())
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:                     # naive -> treat as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:  # noqa: BLE001
        return None


def _purge_stale_url_rows(stale_urls: set[str], keep_hashes: dict[str, set[str]]) -> int:
    """Remove OLD corpus rows for files whose content changed this run.

    Records-kind rows (stable question_id) are updated in place by
    ingest.py's row replacement; document-kind rows have CONTENT-DERIVED
    ids, so the previous version of a changed file would otherwise survive
    as an orphan row (stale content, still retrievable). For each changed
    source_url we keep exactly the hash set produced during this run and
    drop every older variant. Lines that fail validation are kept verbatim —
    we never delete what we cannot parse.
    """
    removed = 0
    if not CORPUS.exists():
        return 0
    kept: list[str] = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                rec = QARecord.model_validate_json(s)
                url = getattr(rec.metadata, "source_url", None) if rec.metadata else None
            except Exception:  # noqa: BLE001 — never drop what we can't parse
                kept.append(line.rstrip("\n"))
                continue
            if url is not None and url in stale_urls:
                keep = keep_hashes.get(url) or set()
                if qa_content_hash(rec) not in keep:
                    removed += 1
                    continue
            kept.append(line.rstrip("\n"))
    if removed:
        from src.utils.atomic_io import write_text_atomic
        write_text_atomic(CORPUS, "\n".join(kept) + "\n")
        log(f"[ingest_folder] removed {removed} superseded corpus row(s) "
            f"for {len(stale_urls)} changed file(s)")
    return removed


def ingest_folder(folder: str, move_processed: bool = False,
                  meta_context: dict | None = None,
                  only_files: set[str] | None = None,
                  exclude_files: set[str] | None = None,
                  seen_hashes: dict[str, str] | None = None,
                  skip_unchanged: bool = True,
                  exclude_globs: tuple[str, ...] | None = None) -> dict:
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

    ``exclude_files`` (additive, source-registry driven) is a set of exact
    file NAMES that must never be converted — crawler sidecar/metadata files
    (e.g. record.json, manifest.json) that live next to real documents in a
    staged corpus. Matching is case-insensitive; excluded files are not
    scanned, not converted, never moved. None (default) = legacy behavior.

    ``seen_hashes`` (additive, changed-record detection) is the corpus's
    last-known {source_url: content hash} map, seeded by the caller
    (ingest.py's _seed_seen_hashes_by_url). A converted file whose url is
    present with a DIFFERENT hash means the upstream file changed: it counts
    as ``changed`` (ids are content-derived, so it also lands as a new row —
    the superseded row is purged after the append). None (default) = legacy
    behavior: no change detection and ``changed`` is always 0.

    ``skip_unchanged`` (additive, OCR-cost guard) skips conversion of a file
    that is ALREADY represented in the corpus under the same ``source_url``
    and has NOT been modified since that row was written (file mtime <= the
    row's ``scraped_at``). Without it every nightly run re-OCRs every scanned
    PDF in the source folders — hours of tesseract CPU — only for the
    converter to discard the result at the id-dedup check. A file the crawler
    re-downloads (mtime refreshed) is still converted, so nothing can be
    missed; the worst case is today's behaviour. True (default) = skip.

    Escape hatch: ``INGEST_SKIP_UNCHANGED=0`` forces the legacy
    convert-everything scan for one run (e.g. after a crawler change that
    rewrites files while preserving their old mtime).

    ``exclude_globs`` (additive, English-only policy) is a tuple of
    case-insensitive filename globs that are skipped BEFORE conversion —
    never OCR'd, never embedded. Default (None) = the built-in English-only
    policy (``*-hin.*``: MoES/INCOIS publish every release twice). Language
    is NOT a metadata/filter axis in this system (records carry no language
    field and retrieval/UI have no language filter), so a Hindi file that
    reached the corpus would be indistinguishable from an English one and
    would surface in answers. See ``_active_exclude_globs`` for overrides.
    """
    _skip_unchanged = bool(skip_unchanged) and _unchanged_skip_enabled()

    p = Path(folder)
    if not p.exists():
        log(f"[ingest_folder] folder not found: {folder}")
        return {"files": 0, "added": 0, "failed": 0, "changed": 0, "unchanged": 0,
                "skipped_language": 0}

    excluded = {n.lower() for n in (exclude_files or set())}
    globs = _active_exclude_globs(exclude_globs)
    # English-only policy: language variants are dropped here, BEFORE any
    # conversion/OCR work (and before the unchanged-file skip counts them).
    all_files = sorted(
        f for f in p.iterdir()
        if f.is_file()
        and f.name.lower() not in excluded
        and not _glob_excluded(f.name, globs)
    )
    skipped_language = sum(
        1 for f in p.iterdir()
        if f.is_file() and f.name.lower() not in excluded
        and _glob_excluded(f.name, globs)
    )
    if skipped_language:
        log(
            f"[ingest_folder] english-only: skipped {skipped_language} "
            f"non-English file(s) in {folder} (globs: {', '.join(globs)})"
        )
    if only_files is not None:
        files = [f for f in all_files if f.name in only_files]
    else:
        files = all_files
    if not files:
        scope = f" (filter matched 0 of {len(all_files)} on disk)" if only_files is not None else ""
        log(f"[ingest_folder] no files in {folder}{scope}")
        return {"files": 0, "added": 0, "failed": 0, "changed": 0, "unchanged": 0,
                "skipped_language": skipped_language}

    log(f"[ingest_folder] scanning {folder}: {len(files)} file(s)")
    out: list = []
    seen: set[str] = set()

    # seed seen with existing corpus ids
    # (and, for the OCR-cost guard, the newest scraped_at per source_url)
    _ingested_ts: dict[str, float] = {}
    if CORPUS.exists():
        for line in open(CORPUS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("question_id"):
                    seen.add(r["question_id"])
                if _skip_unchanged:
                    url = (r.get("metadata") or {}).get("source_url")
                    ts = _parse_ts(r.get("scraped_at"))
                    if url and ts is not None:
                        prev = _ingested_ts.get(url)
                        if prev is None or ts > prev:
                            _ingested_ts[url] = ts
            except Exception:  # noqa: BLE001
                continue

    ok = fail = 0
    changed = 0
    unchanged = 0
    _stale_urls: set[str] = set()              # urls whose content changed
    _new_hashes: dict[str, set[str]] = {}      # url -> hashes produced this run
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
        # OCR-cost guard: a file already in the corpus that has not been
        # touched since its row was written is NOT converted again. This is
        # what stops the nightly run from re-OCRing every scanned PDF (5–14
        # min each) only to discard the text at the id-dedup check below.
        if _skip_unchanged:
            ts = _ingested_ts.get(str(f))
            if ts is not None:
                try:
                    if f.stat().st_mtime <= ts:
                        unchanged += 1
                        log(f"  skip {f.name} (unchanged since last ingest)")
                        continue
                except OSError:  # file vanished mid-scan
                    pass
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
                # Metadata-drift check (Problem 3 fix): a file whose text content
                # (and therefore question_id) is unchanged but whose metadata
                # changed (title, date, title_source, date_source from FIX-A
                # staging resolution) would be silently dropped by the
                # id-in-seen guard inside the converter. When seen_hashes is
                # active, detect this by re-converting with an empty seen and
                # comparing the full qa_content_hash against the corpus hash for
                # this source_url.
                #
                # Safety: the _skip_unchanged mtime guard already ensures
                # OCR-heavy unchanged files are never processed (skipped before
                # reaching this point). For text/embedded-text PDFs the
                # re-parse in the subprocess is cheap. The _stale_urls /
                # _purge_stale_url_rows mechanism then removes the old corpus
                # row and appends the updated record — no duplicates result.
                if seen_hashes is not None:
                    _url = str(f)
                    if _url in seen_hashes:
                        _md_tmp: list = []
                        _md_n = convert_one_detected(
                            f, _md_tmp, set(), False, meta_context
                        )
                        if _md_n > 0 and _md_tmp:
                            _md_h = qa_content_hash(_md_tmp[0])
                            if _md_h != seen_hashes.get(_url):
                                # Metadata changed — treat as changed row and
                                # schedule replacement via stale-url purge.
                                for _r in _md_tmp:
                                    out.append(_r)
                                n = _md_n
                                ok += 1
                                changed += 1
                                _stale_urls.add(_url)
                                seen_hashes[_url] = _md_h
                                _new_hashes.setdefault(_url, set()).add(_md_h)
                                log(
                                    f"  METADATA-CHANGED {f.name} "
                                    f"(metadata drift vs corpus, replacing row)"
                                )
                if n == 0:
                    fail += 1
                    log(f"  WARN {f.name}: no records extracted")
            # Changed-file detection (content-hash keyed on source_url):
            # document-kind ids are content-derived, so an updated upstream
            # file arrives under a NEW id and pure id-dedup cannot see the
            # replacement. Compare with the caller-seeded corpus hash map.
            if n > 0 and seen_hashes is not None:
                for _rec in out[before:]:
                    _md = getattr(_rec, "metadata", None)
                    _url = getattr(_md, "source_url", None) or str(f)
                    _h = qa_content_hash(_rec)
                    _old = seen_hashes.get(_url)
                    if _old is not None and _old != _h:
                        changed += 1
                        _stale_urls.add(_url)
                        log(f"  CHANGED {f.name} (content drift vs corpus)")
                    seen_hashes[_url] = _h
                    _new_hashes.setdefault(_url, set()).add(_h)
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

    if _stale_urls:
        # drop the superseded old-content rows (content-derived ids make
        # the new version a different id — replacement is url-keyed here)
        _purge_stale_url_rows(_stale_urls, _new_hashes)

    return {"files": len(files), "added": len(out), "failed": fail,
            "changed": changed, "unchanged": unchanged, "types": types_used,
            "skipped_language": skipped_language}


def _index_exists() -> bool:
    """True if a loadable index is saved (all marker files present).

    The marker set is NOT defined here: the single canonical contract lives
    in src/retrieval/hybrid/artifacts.py (audit IW-11) and is shared with the
    ``retrieve`` CLI and the ingest service.
    """
    from src.retrieval.hybrid.artifacts import index_is_complete

    return index_is_complete(INDEX_DIR)


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
