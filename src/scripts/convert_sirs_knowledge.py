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
    rec = _make_record(
        f"Document: {path.stem}",
        text,
        subject=path.stem,
        source_url=str(path),
        document_type=doc_type,
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
    from src.data.pdf_table_extract import extract_pdf_text_with_fallback

    try:
        text = extract_pdf_text_with_fallback(path.read_bytes())
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
    rec = _make_record(
        f"Document: {path.stem}",
        text,
        subject=path.stem,
        source_url=str(path),
        document_type=doc_type,
        org=org, source=source, ministry=ministry, default_ministry=default_ministry,
    )
    if rec and rec.question_id not in seen:
        seen.add(rec.question_id)
        out.append(rec)
        return 1
    return 0


def _ocr_pdf_text(path: Path) -> str:
    """OCR a scanned PDF via PyMuPDF render + pytesseract. Returns '' if
    tesseract/pymupdf aren't installed or OCR yields nothing."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        doc = fitz.open(str(path))
        pages_text = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            try:
                t = pytesseract.image_to_string(img)
                if t.strip():
                    pages_text.append(f"--- Page {i+1} (OCR) ---\n{t.strip()}")
            except Exception:  # noqa: BLE001
                continue
        doc.close()
        return "\n\n".join(pages_text)
    except Exception:  # noqa: BLE001
        return ""


def convert_annual_pdf(path: Path, out: list[QARecord], seen: set[str]) -> int:
    """Convert a public INCOIS annual report PDF into one document record.

    Extracts the year from the filename (AR_2023-24_... / Report_2023_...)
    so every record carries metadata.date = <year> and a subject like
    "INCOIS Annual Report 2023-24". Long-doc chunking at index time splits
    the huge text (~200 pages) into searchable chunks.
    """
    from src.data.pdf_table_extract import extract_pdf_text_with_fallback

    m = re.search(r"(?:AR_|Report_|report_)?(\d{4}(?:-\d{2})?)", path.stem)
    year = m.group(1) if m else path.stem
    try:
        text = extract_pdf_text_with_fallback(path.read_bytes())
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
    from src.data.pdf_table_extract import extract_pdf_text_with_fallback

    try:
        text = extract_pdf_text_with_fallback(path.read_bytes())
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
