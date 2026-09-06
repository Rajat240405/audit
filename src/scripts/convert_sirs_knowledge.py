"""
Convert the "sir's system" knowledge base into this project's QARecord JSONL.

Handles the formats found on the INCOIS audit machine:
  1. FINAL_audit_qa_dataset.json          -> [{Question, Answer}, ...]   (QA pairs)
  2. Knowledge_Base/*.json                -> {knowledge_extraction: {metadata, executive_summary,
                                                key_facts, entities, tabular_data_summary}}
  3. UserKnowledge/*.json & KnowledgeBase(UserAdded)/*.json
                                           -> {title, category, source_url, file_name, content}
  4. KnowledgeBase(Scanned)/*.txt|*.pdf   -> raw text files (PDFs -> text via the shared
                                                table-aware extractor; pypdf fallback)

Output: one JSONL file in QARecord format, validated by the project's own
QARecord model, ready for `retrieve build --data <out> --rebuild`.

Usage (run on a machine that has the data):
    python -m src.scripts.convert_sirs_knowledge \
        --qa FINAL_audit_qa_dataset.json \
        --knowledge Knowledge_Base \
        --documents UserKnowledge \
        --scanned "KnowledgeBase(Scanned)" \
        --out data/sirs_processed.jsonl
Flags are optional — point it at whichever folders exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.models.qa_record import QARecord, QARecordMetadata


def _hash_id(text: str, prefix: str = "incdoc") -> str:
    """Generate a stable ID from content. Prefix: 'incdoc' (INCOIS document)
    so converted docs are clearly distinct from parliamentary IDs (18-4-3035)."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


# ── FIX A: staging-aware title/date resolution (verified findings #2/#3) ────
# MoES staging layout: <post>/documents/<file> + <post>/record.json, where
# record.json carries the authoritative CMS title + post_date (see
# src/scraping/moes/normalize.py). The folder converters previously ignored
# it and stamped file stems ("Document: 01-24173-eng", no date). Resolution
# order below; every value carries a provenance flag (title_source /
# date_source). Nothing is fabricated: unknown stays None.

_RECORD_CACHE: dict[str, dict] = {}


def _sibling_record_json(path: Path) -> dict:
    """The MoES staging record.json for this file, or {}.

    Only consulted when the file sits in a ``documents/`` leaf (the MoES
    staging convention) — flat sources (inbox, incois_reports, …) never match
    and keep byte-identical behavior. Results are cached per directory.
    """
    try:
        if path.parent.name.lower() != "documents":
            return {}
        rp = path.parent.parent / "record.json"
        key = str(rp)
        if key not in _RECORD_CACHE:
            data = json.loads(rp.read_text(encoding="utf-8", errors="ignore")) if rp.exists() else {}
            _RECORD_CACHE[key] = data if isinstance(data, dict) else {}
        return _RECORD_CACHE[key]
    except Exception:  # noqa: BLE001 — staging metadata is best-effort
        return {}


def _resolve_doc_title(path: Path, record: dict) -> tuple[str | None, str]:
    """Real title from the sibling record.json, else None (the caller keeps
    the legacy ``path.stem`` label). Returns (title, title_source)."""
    title = re.sub(r"\s+", " ", str(record.get("title") or "")).strip()
    if title:
        return title, "record.json"
    return None, "filename-stem"


_PIB_DATELINE_RE = re.compile(
    r"Posted On:\s*(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", re.IGNORECASE
)
_PIB_MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}
# Digit lookarounds (NOT \b): stems join segments with "_" (a word char), so
# "AR_2008_..." has no \b around 2008. Still rejects years embedded in longer
# digit runs (crawl timestamps like 20250807103950).
_FILENAME_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _resolve_doc_date(path: Path, text: str, record: dict) -> tuple[str | None, str | None]:
    """Best-effort date in precedence order: record.json post_date (ISO) →
    PIB dateline in the document text (→ ISO) → year embedded in the file
    stem (bare YYYY) → None. Returns (date, date_source); never fabricated.
    """
    d = str(record.get("date") or "").strip()
    if d:
        return d, "record.json"
    m = _PIB_DATELINE_RE.search((text or "")[:8000])
    if m:
        mon = _PIB_MONTHS.get(m.group(2).upper())
        day = int(m.group(1))
        if mon is not None and 1 <= day <= 31:
            return f"{m.group(3)}-{mon}-{day:02d}", "pib-dateline"
    y = _FILENAME_YEAR_RE.search(path.stem)
    if y:
        return y.group(1), "filename-year"
    return None, None


# Module-level default ministry. Set via --ministry CLI arg; callers that
# import individual converter functions inherit this automatically.
_DEFAULT_MINISTRY = "EARTH SCIENCES"


def _make_record(
    question_text: str,
    answer_text: str,
    *,
    subject: str | None = None,
    source_url: str | None = None,
    date: str | None = None,
    document_type: str = "parliamentary_qa",
    qa_id: str | None = None,
    ministry: str | None = None,
    default_ministry: str | None = _DEFAULT_MINISTRY,
    org: str | None = None,
    source: str | None = None,
    title_source: str | None = None,
    date_source: str | None = None,
) -> QARecord | None:
    """Build a record. Additive context kwargs (org/source/default_ministry)
    extend the legacy behavior without changing it: callers that pass nothing
    get exactly the historical stamps (ministry=_DEFAULT_MINISTRY, org=None).

    ``default_ministry=None`` suppresses the legacy EARTH SCIENCES fallback —
    used by hierarchical/discovered sources whose ministry is genuinely
    unknown (Earth Sciences was a temporary scope, never a global default for
    new sources).
    """
    q = _clean(question_text)
    a = _clean(answer_text)
    if len(q) < 5 or len(a) < 5:
        return None
    meta = QARecordMetadata(
        ministry=ministry or default_ministry,
        document_type=document_type,
        subject=subject or q[:120],
        date=date,
        source_url=source_url,
        question_type="unknown",
        answer_status="answered",
        org=org,
        source=source,
        title_source=title_source,
        date_source=date_source,
    )
    try:
        return QARecord(
            question_id=qa_id or _hash_id(q + "|" + a),
            question_text=q,
            answer_text=a,
            metadata=meta,
            scraped_at=datetime.now(timezone.utc),
        )
    except Exception:  # noqa: BLE001 — schema min-length (e.g. 5-9 char
        # questions) or any other validation failure: skip the record
        # instead of crashing the whole ingest run.
        return None


# ── Format 1: QA dataset [{Question, Answer}] ──────────────────────────────
def _qa_from_item(item: dict, *, org=None, source=None, ministry=None,
                  default_ministry=_DEFAULT_MINISTRY) -> QARecord | None:
    """Build an audit_qa record from a {Question, Answer} dict."""
    q = item.get("Question") or item.get("question") or item.get("question_text") or ""
    a = item.get("Answer") or item.get("answer") or item.get("answer_text") or ""
    return _make_record(q, a, document_type="audit_qa", org=org, source=source,
                        ministry=ministry, default_ministry=default_ministry)


def convert_qa_dataset(path: Path, out: list[QARecord], seen: set[str], *,
                       org=None, source=None, ministry=None,
                       default_ministry=_DEFAULT_MINISTRY) -> int:
    """Convert a QA file — either a JSON array [{Question,Answer}] (the
    scientist's FINAL_audit_qa_dataset.json) or a JSONL of QA pairs (inbox
    files). Both go through the same _make_record so ids are deterministic."""
    if not path.exists():
        return 0
    n = 0
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(item, dict):
                rec = _qa_from_item(item, org=org, source=source, ministry=ministry,
                                    default_ministry=default_ministry)
                if rec and rec.question_id not in seen:
                    seen.add(rec.question_id)
                    out.append(rec)
                    n += 1
        return n

    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("qa") or data.get("questions") or []
    for item in data:
        rec = _qa_from_item(item, org=org, source=source, ministry=ministry,
                            default_ministry=default_ministry)
        if rec and rec.question_id not in seen:
            seen.add(rec.question_id)
            out.append(rec)
            n += 1
    return n


# ── Format 2: Knowledge_Base knowledge_extraction JSON ─────────────────────
def convert_knowledge_json(path: Path, out: list[QARecord], seen: set[str], *,
                           org=None, source=None, ministry=None,
                           default_ministry=_DEFAULT_MINISTRY) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {path.name}: {e}")
        return 0
    ke = data.get("knowledge_extraction") or {}
    meta = ke.get("metadata") or {}
    title = meta.get("title") or data.get("title") or path.stem
    date = meta.get("date")
    refs = meta.get("file_reference_ids") or []
    ref_str = ", ".join(str(r) for r in refs) if refs else ""

    parts = []
    if ke.get("executive_summary"):
        parts.append(f"Executive Summary:\n{ke['executive_summary']}")
    if ke.get("key_facts"):
        facts = ke["key_facts"]
        parts.append("Key Facts:\n" + "\n".join(f"- {f}" for f in facts if f))
    ent = ke.get("entities") or {}
    if any(ent.values()):
        ent_lines = []
        for label in ("organizations", "people", "locations", "equipment"):
            vals = [v for v in (ent.get(label) or []) if v]
            if vals:
                ent_lines.append(f"{label.title()}: {', '.join(vals)}")
        if ent_lines:
            parts.append("Entities:\n" + "\n".join(ent_lines))
    if ke.get("tabular_data_summary"):
        parts.append(f"Tables:\n{json.dumps(ke['tabular_data_summary'], ensure_ascii=False)}")

    if not parts:
        return 0
    answer = "\n\n".join(parts)
    rec = _make_record(
        f"Document: {title}" + (f" ({ref_str})" if ref_str else ""),
        answer,
        subject=title,
        source_url=ref_str or None,
        date=str(date) if date else None,
        document_type="document",
        org=org, source=source, ministry=ministry, default_ministry=default_ministry,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


# ── Format 3: {title, category, content} document JSON ─────────────────────
def convert_document_json(path: Path, out: list[QARecord], seen: set[str], *,
                          org=None, source=None, ministry=None,
                          default_ministry=_DEFAULT_MINISTRY) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {path.name}: {e}")
        return 0
    if isinstance(data, list):
        n = 0
        for item in data:
            n += convert_document_dict(item, path, out, seen, org=org, source=source,
                                       ministry=ministry, default_ministry=default_ministry)
        return n
    return convert_document_dict(data, path, out, seen, org=org, source=source,
                                 ministry=ministry, default_ministry=default_ministry)


def convert_document_dict(data: dict, path: Path, out: list[QARecord], seen: set[str], *,
                          org=None, source=None, ministry=None,
                          default_ministry=_DEFAULT_MINISTRY) -> int:
    title = data.get("title") or data.get("file_name") or path.stem
    content = data.get("content") or data.get("text") or data.get("summary") or ""
    if not content:
        return 0
    url = data.get("source_url") or data.get("url") or None
    category = data.get("category")
    subject = category if category and "ministry" in category.lower() else title
    rec = _make_record(
        f"Document: {title}",
        content,
        subject=title,
        source_url=url,
        date=data.get("date") or None,
        document_type="document",
        org=org, source=source, ministry=ministry, default_ministry=default_ministry,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


# ── Format 4: scanned text / PDF files ─────────────────────────────────────
def convert_text_file(path: Path, out: list[QARecord], seen: set[str],
                      doc_type: str = "document", *, org=None, source=None,
                      ministry=None, default_ministry=_DEFAULT_MINISTRY) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text.strip()) < 10:
        return 0  # only empty/whitespace guard — short notes are valid
    # FIX A: staging title/date (record.json → dateline → filename-year → None)
    _srec = _sibling_record_json(path)
    _title, _title_source = _resolve_doc_title(path, _srec)
    _date, _date_source = _resolve_doc_date(path, text, _srec)
    _label = _title or path.stem
    rec = _make_record(
        f"Document: {_label}",
        text,
        subject=_label,
        source_url=str(path),
        date=_date,
        document_type=doc_type,
        title_source=_title_source,
        date_source=_date_source,
        org=org, source=source, ministry=ministry, default_ministry=default_ministry,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


def convert_pdf_file(path: Path, out: list[QARecord], seen: set[str],
                      doc_type: str = "document", *, org=None, source=None,
                      ministry=None, default_ministry=_DEFAULT_MINISTRY) -> int:
    # THE canonical PDF→text for folder ingestion (audit IW-7): table-aware
    # PyMuPDF first (borderless-table reconstruction), legacy pypdf as the
    # built-in fallback — one shared stack, not a second implementation.
    # PyMuPDF work runs in a child subprocess (_extract_text_subprocess) so a
    # native SIGSEGV in the MuPDF C layer cannot kill the parent ingestion process.
    try:
        text = _extract_text_subprocess(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip pdf] {path.name}: {e}")
        return 0
    if len(text.strip()) < 50:
        # scanned/image-only PDF — try OCR before giving up
        print(f"  [pdf] {path.name}: no embedded text, trying OCR...")
        text = _ocr_pdf_text(path)
        if not text.strip():
            print(f"  [skip pdf] {path.name}: no extractable text (scanned image?)")
            return 0
    # FIX A: staging title/date (record.json → dateline → filename-year → None)
    _srec = _sibling_record_json(path)
    _title, _title_source = _resolve_doc_title(path, _srec)
    _date, _date_source = _resolve_doc_date(path, text, _srec)
    _label = _title or path.stem
    rec = _make_record(
        f"Document: {_label}",
        text,
        subject=_label,
        source_url=str(path),
        date=_date,
        document_type=doc_type,
        title_source=_title_source,
        date_source=_date_source,
        org=org, source=source, ministry=ministry, default_ministry=default_ministry,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


import os as _os
import pickle as _pickle
import subprocess as _subprocess
import sys as _sys

_OCR_DPI = 200

# Project root injected into child subprocess sys.path so it can import
# src.data.pdf_table_extract without a full package install.
# __file__ = <root>/src/scripts/convert_sirs_knowledge.py  →  3 levels up.
_FITZ_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


# ── Subprocess helpers — PyMuPDF process isolation ────────────────────────────
#
# PyMuPDF's MuPDF C layer can raise SIGSEGV on certain malformed or
# rendering-edge-case PDFs.  SIGSEGV is a signal delivered to the OS process;
# a threading.Lock cannot prevent it.  The only reliable fix is to run all
# fitz operations inside a CHILD process: if the child crashes, the parent
# receives a non-zero exit code / no output and continues normally.
#
# Design:
#   _fitz_count_pages_subprocess   — fitz.open + len(doc)  → int
#   _fitz_render_page_subprocess   — fitz.open + get_pixmap → (w, h, bytes)
#   _extract_text_subprocess       — full table-aware text extraction → str
#
# All helpers communicate via pickle over stdout.  A SIGSEGV in the child
# produces returncode -11 (Linux) or similar — the parent detects this,
# logs a warning, and returns a safe empty/None result.

def _fitz_count_pages_subprocess(path: Path, timeout: int = 60) -> int:
    """Return the page count of ``path`` by opening it in a child process.

    Isolates fitz.open() behind an OS process boundary.  Returns 0 on
    crash, SIGSEGV, timeout, or when PyMuPDF is not installed.
    """
    child = (
        "import sys, pickle, os\n"
        # Suppress any text that fitz/MuPDF may write to stdout (warnings,
        # version banners) — they would corrupt the pickle payload.
        "_stdout_save = sys.stdout\n"
        "sys.stdout = open(os.devnull, 'w')\n"
        "try:\n"
        "    import fitz\n"
        f"    doc = fitz.open({str(path)!r})\n"
        "    n = len(doc); doc.close()\n"
        "except Exception:\n"
        "    n = 0\n"
        "finally:\n"
        "    sys.stdout.close()\n"
        "    sys.stdout = _stdout_save\n"
        "sys.stdout.buffer.write(pickle.dumps(n))\n"
    )
    try:
        r = _subprocess.run(
            [_sys.executable, "-c", child],
            capture_output=True, timeout=timeout,
        )
        if r.returncode == 0 and r.stdout:
            return int(_pickle.loads(r.stdout))
        # Non-zero: child crashed (SIGSEGV → exit -11) or import error
        if r.returncode != 0:
            print(
                f"  [fitz subprocess] page-count failed for {path.name} "
                f"(child exit {r.returncode})"
            )
    except _subprocess.TimeoutExpired:
        print(f"  [fitz subprocess] page-count timeout for {path.name}")
    except Exception:
        pass
    return 0


def _fitz_render_page_subprocess(
    path: Path, page_index: int, dpi: int = _OCR_DPI, timeout: int = 120,
) -> "tuple[int, int, bytes] | None":
    """Render one PDF page to RGB in a child process.

    Returns ``(width, height, rgb_samples_bytes)`` or ``None`` on crash /
    SIGSEGV / timeout / import failure.  A SIGSEGV in the MuPDF C layer
    kills only the child (exit code -11 on Linux); the parent receives
    ``None`` and continues with remaining pages.
    """
    child = (
        "import sys, pickle, os\n"
        # Suppress any text that fitz/MuPDF may write to stdout.
        "_stdout_save = sys.stdout\n"
        "sys.stdout = open(os.devnull, 'w')\n"
        "result = None\n"
        "try:\n"
        "    import fitz\n"
        f"    doc = fitz.open({str(path)!r})\n"
        f"    pix = doc[{page_index}].get_pixmap(dpi={dpi})\n"
        "    result = (pix.width, pix.height, bytes(pix.samples))\n"
        "    doc.close()\n"
        "except Exception:\n"
        "    result = None\n"
        "finally:\n"
        "    sys.stdout.close()\n"
        "    sys.stdout = _stdout_save\n"
        "sys.stdout.buffer.write(pickle.dumps(result))\n"
    )
    try:
        r = _subprocess.run(
            [_sys.executable, "-c", child],
            capture_output=True, timeout=timeout,
        )
        if r.returncode == 0 and r.stdout:
            return _pickle.loads(r.stdout)  # (w, h, bytes) or None from child
        # Non-zero returncode: SIGSEGV (-11), OOM, etc.
        if r.returncode != 0:
            print(
                f"  [fitz subprocess] page {page_index + 1} render failed "
                f"for {path.name} (child exit {r.returncode})"
            )
        return None
    except _subprocess.TimeoutExpired:
        print(
            f"  [fitz subprocess] page {page_index + 1} render timeout "
            f"for {path.name}"
        )
        return None
    except Exception:
        return None


def _extract_text_subprocess(path: Path, timeout: int = 300) -> str:
    """Run extract_pdf_text_with_fallback in a child process.

    The child imports ``src.data.pdf_table_extract`` directly — no circular
    imports (pdf_table_extract has no dependency on src.scripts) and no
    recursive subprocess spawning (pdf_table_extract never calls back here).
    Returns ``""`` on crash / SIGSEGV / timeout / import failure so callers
    can apply their normal OCR fallback.
    """
    child = (
        "import sys, pickle, os\n"
        f"sys.path.insert(0, {_FITZ_PROJECT_ROOT!r})\n"
        "from src.data.pdf_table_extract import extract_pdf_text_with_fallback\n"
        # Suppress any text that fitz/MuPDF may write to stdout before the
        # pickle payload — corruption would cause pickle.loads to fail silently.
        "_stdout_save = sys.stdout\n"
        "sys.stdout = open(os.devnull, 'w')\n"
        "text = ''\n"
        "try:\n"
        f"    data = open({str(path)!r}, 'rb').read()\n"
        "    text = extract_pdf_text_with_fallback(data)\n"
        "except Exception:\n"
        "    text = ''\n"
        "finally:\n"
        "    sys.stdout.close()\n"
        "    sys.stdout = _stdout_save\n"
        "sys.stdout.buffer.write(pickle.dumps(text or ''))\n"
    )
    try:
        r = _subprocess.run(
            [_sys.executable, "-c", child],
            capture_output=True, timeout=timeout,
        )
        if r.returncode == 0 and r.stdout:
            result = _pickle.loads(r.stdout)
            return result if isinstance(result, str) else ""
        if r.returncode != 0:
            print(
                f"  [fitz subprocess] text extraction failed for {path.name} "
                f"(child exit {r.returncode})"
            )
        return ""
    except _subprocess.TimeoutExpired:
        print(f"  [pdf subprocess] timeout extracting text from {path.name}")
        return ""
    except Exception:
        return ""


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _ocr_page(args) -> tuple[int, str]:
    """OCR one page: PyMuPDF render in a child subprocess + Tesseract in parent.

    The fitz.open + get_pixmap step runs in an isolated OS process
    (_fitz_render_page_subprocess) so a SIGSEGV in the MuPDF C layer kills
    only that child — the parent and all other page workers survive.

    Tesseract (pytesseract.image_to_string) runs in the calling thread: it
    shells out to the tesseract binary, not a C extension, so there is no
    fitz crash risk.  Tesseract parallelism across pages is fully preserved
    (the ThreadPoolExecutor in _ocr_pdf_text submits one task per page).

    Returns (page_index, text); a failed render returns '' for that page.
    """
    path, page_index = args
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return page_index, ""

    render = _fitz_render_page_subprocess(Path(path), page_index, dpi=_OCR_DPI)
    if render is None:
        # child crashed or timed out — log already emitted by the helper
        return page_index, ""

    width, height, samples = render
    try:
        img = Image.frombytes("RGB", (width, height), samples)
        t = pytesseract.image_to_string(img)
        return page_index, t or ""
    except Exception:  # noqa: BLE001
        return page_index, ""


def _ocr_workers() -> int:
    """Page-level OCR parallelism. pytesseract shells out to the tesseract
    binary, so threads scale near-linearly (the wait is a subprocess).
    Override with OCR_WORKERS; 1 disables pooling (sequential path)."""
    try:
        n = int(_os.environ.get("OCR_WORKERS", "") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        n = min(8, _os.cpu_count() or 4)
    return max(1, n)


def _ocr_pdf_text(path: Path, max_pages: int | None = None) -> str:
    """OCR a scanned PDF via subprocess PyMuPDF render + pytesseract.

    Both the page-count query and every page render run in isolated child
    subprocesses — a SIGSEGV in the MuPDF C layer kills only the relevant
    child; the parent ingestion process and all remaining page workers survive.

    Tesseract parallelism is preserved: the ThreadPoolExecutor submits one
    render+OCR task per page; each task spawns its own render subprocess then
    runs Tesseract in the calling thread (pytesseract shells out — no fitz risk).

    ``max_pages`` limits OCR to the first N pages (type-detection peek path,
    which only needs the title page). Returns '' when all page counts return 0
    or all renders fail.
    """
    n_pages = _fitz_count_pages_subprocess(path)
    if n_pages <= 0:
        return ""
    if max_pages is not None:
        n_pages = max(0, min(n_pages, int(max_pages)))
    if n_pages <= 0:
        return ""

    workers = _ocr_workers()
    pages_text: list[str] = []

    if n_pages == 1 or workers == 1:
        for i in range(n_pages):
            _, t = _ocr_page((str(path), i))
            if t.strip():
                pages_text.append(f"--- Page {i+1} (OCR) ---\n{t.strip()}")
        return "\n\n".join(pages_text)

    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, n_pages)) as pool:
            results = list(pool.map(_ocr_page, [(str(path), i) for i in range(n_pages)]))
    except Exception:  # noqa: BLE001 — never fail a conversion over pooling
        results = [_ocr_page((str(path), i)) for i in range(n_pages)]

    # pool.map preserves input order; sort anyway so output can never depend
    # on completion order.
    for i, t in sorted(results, key=lambda r: r[0]):
        if t and t.strip():
            pages_text.append(f"--- Page {i+1} (OCR) ---\n{t.strip()}")
    return "\n\n".join(pages_text)


def convert_annual_pdf(path: Path, out: list[QARecord], seen: set[str]) -> int:
    """Convert a public INCOIS annual report PDF into one document record.

    Extracts the year from the filename (AR_2023-24_... / Report_2023_...)
    so every record carries metadata.date = <year> and a subject like
    "INCOIS Annual Report 2023-24". Long-doc chunking at index time splits
    the huge text (~200 pages) into searchable chunks.

    Text extraction runs in a child subprocess (_extract_text_subprocess)
    so a SIGSEGV in the MuPDF C layer cannot kill the parent ingestion process.
    """
    m = re.search(r"(?:AR_|Report_|report_)?(\d{4}(?:-\d{2})?)", path.stem)
    year = m.group(1) if m else path.stem
    try:
        text = _extract_text_subprocess(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip annual] {path.name}: {e}")
        return 0
    if len(text.strip()) < 50:
        print(f"  [skip annual] {path.name}: no extractable text (image-only?)")
        return 0
    title = f"INCOIS Annual Report {year}"
    rec = _make_record(
        f"Document: {title}",
        text,
        subject=title,
        source_url=str(path),
        date=year,
        document_type="annual_report",
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


def convert_report_pdf(path: Path, out: list[QARecord], seen: set[str]) -> int:
    """Convert any other INCOIS report PDF (general/technical/research/news)
    into one document record with a descriptive title parsed from the
    filename (e.g. TR_ESSO-INCOIS-OMARS-TR-01(2025) -> 'INCOIS Technical
    Report OMARS-TR-01 (2025)')."""
    stem = path.stem
    # normalize the noisy filename into a readable title
    cleaned = re.sub(r"_\d{14}$", "", stem)          # strip timestamp suffix
    cleaned = re.sub(r"[_]+", " ", cleaned).strip()  # underscores -> spaces
    if cleaned.lower().startswith(("tr_", "tr ")):
        title = f"INCOIS Technical Report {cleaned[3:].strip()}"
        doc_type = "technical_report"
    elif cleaned.lower().startswith("rp "):
        title = f"INCOIS Research Publication {cleaned[3:].strip()}"
        doc_type = "research_publication"
    elif cleaned.lower().startswith("report"):
        m = re.search(r"(\d{4})", cleaned)
        yr = m.group(1) if m else ""
        title = f"INCOIS General Report {yr}".strip()
        doc_type = "general_report"
    else:
        title = f"INCOIS Document {cleaned}"
        doc_type = "document"
    # Text extraction runs in a child subprocess so a SIGSEGV in the MuPDF
    # C layer cannot kill the parent ingestion process.
    try:
        text = _extract_text_subprocess(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip report] {path.name}: {e}")
        return 0
    if len(text.strip()) < 50:
        print(f"  [skip report] {path.name}: no extractable text (image-only?)")
        return 0
    m = re.search(r"(\d{4})", path.stem)
    rec = _make_record(
        f"Document: {title}",
        text,
        subject=title,
        source_url=str(path),
        date=m.group(1) if m else None,
        document_type=doc_type,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


def scan_folder(
    folder: str,
    out: list[QARecord],
    seen: set[str],
    as_knowledge: bool = False,
    as_document: bool = False,
    include_text: bool = False,
    include_pdf: bool = False,
    as_annual: bool = False,
    as_report: bool = False,
) -> int:
    p = Path(folder)
    if not p.exists():
        print(f"  [missing] {folder} — skipped")
        return 0
    n = 0
    files = sorted(p.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        if as_annual and f.suffix.lower() == ".pdf":
            # AnnualReports folders often mix AR_* (annual) and Report_* (general)
            if f.name.lower().startswith("report_"):
                n += convert_report_pdf(f, out, seen)
            else:
                n += convert_annual_pdf(f, out, seen)
        elif as_report and f.suffix.lower() == ".pdf":
            n += convert_report_pdf(f, out, seen)
        elif as_knowledge and f.suffix.lower() == ".json":
            n += convert_knowledge_json(f, out, seen)
        elif as_document and f.suffix.lower() == ".json":
            n += convert_document_json(f, out, seen)
        elif include_text and f.suffix.lower() in (".txt", ".md"):
            n += convert_text_file(f, out, seen)
        elif include_pdf and f.suffix.lower() == ".pdf":
            n += convert_pdf_file(f, out, seen)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", default=None, help="FINAL_audit_qa_dataset.json")
    ap.add_argument("--knowledge", default=None, help="Knowledge_Base folder (knowledge_extraction JSON)")
    ap.add_argument("--documents", default=None, help="UserKnowledge / KnowledgeBase(UserAdded) folder")
    ap.add_argument("--scanned", default=None, help="KnowledgeBase(Scanned) folder (txt + pdf)")
    ap.add_argument("--annual", default=None, help="AnnualReports folder (INCOIS public annual report PDFs)")
    ap.add_argument("--reports", default=None,
                    help="Other INCOIS report folder (general/tech/research/news PDFs)")
    ap.add_argument("--out", default="data/sirs_processed.jsonl")
    ap.add_argument("--ministry", default="EARTH SCIENCES",
                    help="Ministry name stamped on all converted records "
                         "(default: EARTH SCIENCES)")
    args = ap.parse_args()

    # Set the module-level default so all converter functions inherit it.
    global _DEFAULT_MINISTRY
    _DEFAULT_MINISTRY = args.ministry

    out: list[QARecord] = []
    seen: set[str] = set()

    print("Converting sir's knowledge base -> QARecord JSONL")
    if args.qa:
        convert_qa_dataset(Path(args.qa), out, seen)
    if args.knowledge:
        n = scan_folder(args.knowledge, out, seen, as_knowledge=True)
        print(f"  [Knowledge_Base] {n} records")
    if args.documents:
        n = scan_folder(args.documents, out, seen, as_document=True)
        print(f"  [UserKnowledge/UserAdded] {n} records")
    if args.scanned:
        n = scan_folder(args.scanned, out, seen, include_text=True, include_pdf=True)
        print(f"  [Scanned] {n} records (text + extractable PDFs)")
    if args.annual:
        n = scan_folder(args.annual, out, seen, as_annual=True)
        print(f"  [AnnualReports] {n} records")
    if args.reports:
        n = scan_folder(args.reports, out, seen, as_report=True)
        print(f"  [Other reports] {n} records")

    if not out:
        print("Nothing converted — check the --paths. Exiting.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(rec.model_dump_json() + "\n")
    types = {}
    for rec in out:
        types[rec.metadata.document_type] = types.get(rec.metadata.document_type, 0) + 1
    print(f"\nWrote {len(out)} records -> {out_path}")
    print("  by type:", types)


if __name__ == "__main__":
    main()
