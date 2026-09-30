"""Document bytes → (question_text, answer_text) extraction — LS primary text stage.

Extraction from the official answer documents is the primary content source
for ``answer_text`` — not a fallback. The original design assumed LS inline
text was always empty (the frozen workbook's ``questionText``/``answerText``
columns are), but upstream API coverage has since widened, so both candidates
now routinely exist. Which one wins is decided PER FIELD by
``src/scraping/ls/text_selection.py``: the document wins ``answer_text``
(it is the only source carrying annexure tables), inline wins
``question_text`` (the PDF-derived form carries document furniture and can be
an arbitrary one-third ratio split).

Uses the canonical table-aware and OCR-driven extraction pipeline:
- ``src.data.pdf_table_extract.extract_pdf_text`` — PyMuPDF table-aware + multi-page
  continuity + in-memory OCR for scanned pages.
- ``src.data.pdf_table_extract.split_question_answer`` — ANSWER/REPLY boundary split.
"""

from __future__ import annotations

import io
from typing import Any

from src.data.pdf_table_extract import DependencyMissingError, extract_pdf_text, split_question_answer

#: extraction failure reasons (verbatim legacy vocabulary)
Reason = str  # "scanned" | "parser_failure" | "unsupported" | "dependency_unavailable"


def _docx_text(data: bytes) -> str | None:
    """All text of an OOXML DOCX (paragraphs + table cells)."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for DOCX support. Install with: pip install python-docx"
        ) from exc
    try:
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        parts.append(cell.text)
        return "\n".join(parts)
    except Exception:
        return None


def extract_qa(body: bytes, doc_format: str) -> tuple[tuple[str, str] | None, Reason | None]:
    """Extract (question, answer) from document bytes of a sniffed format.

    Returns ``((question, answer), None)`` on success, else ``(None, reason)``
    with reason in the legacy vocabulary.
    """
    if not body:
        return None, "unsupported"
        
    if doc_format == "pdf":
        try:
            text = extract_pdf_text(body, enable_ocr=True)
        except DependencyMissingError as exc:
            return None, f"dependency_unavailable: {exc}"
        except Exception:
            return None, "parser_failure"
            
        if text is None or not text.strip():
            return None, "scanned"
    elif doc_format == "docx":
        text = _docx_text(body)
        if text is None:
            return None, "parser_failure"
        if not text.strip():
            return None, "scanned"
    else:
        return None, "unsupported"
        
    res = split_question_answer(text)
    return res, None
