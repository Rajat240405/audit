"""Magic-byte document sniffing and validity classification (design §5 step 4).

Frozen from the validation findings: official payload bytes do not always
match the URL extension (.pdf URL serving DOCX, .html URLs serving real
HTML...), missing files surface as HTTP 500, and the store contains 15-byte
truncated-PDF stubs. Classification therefore inspects BYTES, never the
filename:

- ``missing``        – upstream record has no filename (no request was made)
- ``broken``         – non-200 status / truncated stub / unopenable payload /
                       unrecognized bytes
- ``good``           – recognized format, docx|doc|html format-verified,
                       pdf opens and yields extractable text
- ``partial``        – pdf opens but contains no extractable text (scan)

Refinements known from validation but intentionally NOT reproduced here
(they are recorded in the validation workbook): legacy Kruti-Dev encoded
Hindi PDFs extract *garbage* text and classify ``good`` by this code; the
workbook carries their refined ``partial`` verdicts.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from typing import Literal

DocClass = Literal["good", "partial", "broken", "missing"]

#: payloads at or below this size that carry a %PDF- header are origin stubs
TRUNCATED_STUB_MAX_BYTES = 64

CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass
class DocFacts:
    doc_class: DocClass
    format: str                       # pdf | docx | doc | html | unknown | none
    http_status: int | None
    bytes: int
    sha256: str | None = None
    pages: int | None = None
    text_chars: int | None = None     # first-3-pages probe (pdf only)
    cause: str | None = None          # machine-readable reason for broken/missing
    note: str = ""

    def as_manifest(self) -> dict:
        out = {
            "class": self.doc_class,
            "format": self.format,
            "http": self.http_status,
            "bytes": self.bytes,
        }
        if self.sha256:
            out["sha256"] = self.sha256
        if self.cause:
            out["cause"] = self.cause
        if self.pages is not None:
            out["pages"] = self.pages
        if self.text_chars is not None:
            out["text_chars"] = self.text_chars
        return out


def sniff_format(body: bytes) -> str:
    """Identify the payload by magic bytes: pdf | docx | doc | html | unknown."""
    if body[:5] == b"%PDF-":
        return "pdf"
    if body[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except Exception:  # noqa: BLE001
            pass
        return "unknown"
    if body[:8] == CFB_MAGIC:
        return "doc"
    head = body[:1024].lstrip(b"\xef\xbb\xbf\xff\xfe\x00 \t\r\n").lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]:
        return "html"
    return "unknown"


def _pdf_probe(body: bytes) -> tuple[bool, int | None, int | None]:
    """(opens, page_count, text chars in first 3 pages) via PyMuPDF."""
    from src.data import pdf_table_extract  # lazy PyMuPDF import wrapper

    try:
        fitz = pdf_table_extract._import_fitz()  # noqa: SLF001 — shared lazy import
    except ImportError:
        # No PyMuPDF: format-level verdict only (no open/text probe).
        return False, None, None
    try:
        doc = fitz.open(stream=body, filetype="pdf")
    except Exception:  # noqa: BLE001
        return False, None, None
    try:
        pages = len(doc)
        chars = 0
        for page in doc[:3]:
            chars += len(page.get_text().strip())
        return True, pages, chars
    finally:
        doc.close()


def classify_document(
    body: bytes | None,
    *,
    http_status: int | None,
    content_type: str | None = None,  # noqa: ARG001 — kept for future hints
    no_filename: bool = False,
    transport_failed: bool = False,
) -> DocFacts:
    """Classify one downloaded (or not-downloadable) document payload."""
    if no_filename:
        return DocFacts("missing", "none", http_status, 0,
                        cause="empty-filename",
                        note="official record carries an empty filename slot")
    if transport_failed:
        return DocFacts("broken", "unknown", http_status, len(body or b""),
                        cause="request-failed",
                        note="transport error after retries")
    body = body or b""
    sha = hashlib.sha256(body).hexdigest() if body else None
    if http_status != 200:
        return DocFacts("broken", "unknown", http_status, len(body),
                        sha256=sha, cause=f"http-{http_status}",
                        note="server returned non-200 (sansad.in uses 500 for absent files)")
    fmt = sniff_format(body)
    if fmt == "pdf" and len(body) <= TRUNCATED_STUB_MAX_BYTES:
        return DocFacts("broken", "pdf", http_status, len(body), sha256=sha,
                        cause="truncated-stub",
                        note="payload is a truncated PDF stub (header only, no stream)")
    if fmt == "pdf":
        opens, pages, chars = _pdf_probe(body)
        if pages is None:  # probe unavailable (no PyMuPDF) or failed to open
            if opens is False and _pymupdf_available():
                return DocFacts("broken", "pdf", http_status, len(body), sha256=sha,
                                cause="corrupt-pdf", note="PyMuPDF could not open the PDF")
            return DocFacts("good", "pdf", http_status, len(body), sha256=sha,
                            note="format-verified only (PyMuPDF unavailable)")
        if not chars:
            return DocFacts("partial", "pdf", http_status, len(body), sha256=sha,
                            pages=pages, text_chars=0,
                            note="PDF opens but has no extractable text (image-only scan)")
        return DocFacts("good", "pdf", http_status, len(body), sha256=sha,
                        pages=pages, text_chars=chars)
    if fmt in ("docx", "doc", "html"):
        return DocFacts("good", fmt, http_status, len(body), sha256=sha,
                        note="format-verified")
    return DocFacts("broken", "unknown", http_status, len(body), sha256=sha,
                    cause="not-a-document",
                    note="payload bytes are not a recognized document format")


def _pymupdf_available() -> bool:
    try:
        from src.data import pdf_table_extract

        pdf_table_extract._import_fitz()  # noqa: SLF001
        return True
    except ImportError:
        return False


def format_extension(doc_format: str) -> str:
    return {"pdf": "pdf", "docx": "docx", "doc": "doc", "html": "html"}.get(doc_format, "bin")


@dataclass
class SlotResult:
    """Outcome for one language slot of one record (manifest material)."""

    key: str                          # "<qslno>-<lang>"
    lang: Literal["eng", "hin"]
    url: str | None
    facts: DocFacts
    path: str | None = None           # posix path relative to the session dir
    extra: dict = field(default_factory=dict)
