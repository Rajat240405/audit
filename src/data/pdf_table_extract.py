"""Unified Table-Aware PDF Text Extractor for Parliamentary Q&A (Lok Sabha & Rajya Sabha).

Design & Policy:
  1. Mandatory Dependency: Requires PyMuPDF (fitz) and Tesseract (when OCR requested).
     Raises explicit DependencyMissingError on missing dependencies. NO SILENT FALLBACK to pypdf.
  2. Multi-Page Table Stitching: Stitches tables spanning consecutive pages, prunes
     redundant repeated headers, normalizes column drift, and fuses split rows.
  3. Strict Continuity Threshold: Enforces invariant: sim < 0.85 -> NEVER MERGE (no t_idx bypass).
  4. In-Order Rendering: Preserves page reading order (no appending tables at document end).
  5. Mixed Page Support: Handles pages containing both native text and embedded raster image tables.
  6. Context-Aware Number Repair: Repairs intra-cell newline wraps without corrupting distinct numbers.
  7. Page-Boundary Row Reconstruction: Fuses wrapped split rows across page breaks.
  8. Process-Tree Hard Timeout: Runs OCR in an isolated subprocess with cross-platform termination.
  9. PicoDet -> DOTS Table Routing (opt-in via DOTS_ENABLED): when enabled, PicoDet is the
     ONLY table detector and DOTS OCR is the ONLY table extractor. A page PicoDet flags is
     owned end-to-end by DOTS and Strategies 1-3 are bypassed for that page; a page PicoDet
     clears takes the unchanged non-table path. With DOTS_ENABLED unset (the default) this
     module behaves exactly as before, byte for byte.

Routing (DOTS_ENABLED=true):

    page -> scanned? -> yes -> Tesseract OCR (unchanged; non-table scanned pages)
                     -> no  -> PicoDet -> TABLE    -> DOTS            ("dots_table")
                                       -> NO TABLE -> _render_merged  ("prose")

Page order is preserved: routing happens inside the per-page loop and every branch returns
a payload for that page, which the stitcher consumes in order.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
from statistics import median
from typing import Any, List, Optional, Tuple

_fitz_enabled = True


class DependencyMissingError(ImportError, RuntimeError):
    """Raised when a mandatory extraction dependency (PyMuPDF or Tesseract) is unavailable."""
    pass


class ExtractionTimeoutError(RuntimeError):
    """Raised when an extraction operation exceeds its hard time ceiling and is killed."""
    pass


def check_extraction_environment(ocr_required: bool = False) -> tuple[bool, str]:
    """Preflight check validating mandatory extraction dependencies."""
    if not _fitz_enabled:
        return False, "Mandatory dependency missing: PyMuPDF (fitz) is disabled."
    try:
        import fitz
    except ImportError:
        return False, "Mandatory dependency missing: PyMuPDF (fitz) is not installed."
    
    if ocr_required:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception as exc:
            return False, f"Mandatory dependency missing: Tesseract OCR is unavailable ({exc})."

    if dots_routing_enabled():
        from src.data import dots_client, table_detect

        if not table_detect.is_available():
            return False, (
                "DOTS_ENABLED=true but the PicoDet table detector is unavailable "
                "(install paddlepaddle + paddleocr, or set DOTS_ENABLED=false)."
            )
        if not dots_client.get_client().is_available():
            return False, (
                "DOTS_ENABLED=true but the DOTS vLLM endpoint is unreachable at "
                f"{dots_client.DotsConfig.from_env().base_url or '<DOTS_BASE_URL unset>'}."
            )

    return True, "All mandatory extraction dependencies are operational."


def _import_fitz():
    if not _fitz_enabled:
        raise DependencyMissingError("Extraction blocked: PyMuPDF is disabled.")
    try:
        import fitz
        return fitz
    except ImportError:
        raise DependencyMissingError("Extraction blocked: PyMuPDF is required but not installed.")


# ── PicoDet -> DOTS routing ──────────────────────────────────────────────────

#: Payload type emitted for a page extracted by DOTS. The stitcher's generic
#: branch passes any payload carrying "content" straight through in page order,
#: so no stitcher change is required.
DOTS_PAYLOAD_TYPE = "dots_table"

#: Rendering DPI recorded in extractor_version, so a DPI change is a visible
#: extraction-decision change rather than a silent one.
_DEFAULT_DPI = 200


def dots_routing_enabled() -> bool:
    """Is PicoDet -> DOTS routing switched on?

    Defaults to **false**: merely deploying this code must not change a single
    byte of extracted text. An operator opts in with DOTS_ENABLED=true.
    """
    return (os.environ.get("DOTS_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")


def current_extractor_version() -> str:
    """Identifier for the extraction *decisions* currently in force.

    Shape: ``dots-<version>/picodet-<version>/dpi<value>``, or
    ``legacy/dpi<value>`` when routing is off.

    This is extraction-decision metadata, NOT semantic content. It is stored on
    the record (``metadata.extractor_version``) purely so a changed extractor
    can trigger re-extraction, and ``qa_content_hash`` excludes it so that a
    version bump alone never marks a record as changed.
    """
    dpi = (os.environ.get("DOTS_RENDER_DPI") or "").strip() or str(_DEFAULT_DPI)
    if not dots_routing_enabled():
        return f"legacy/dpi{dpi}"
    from src.data import dots_client, table_detect

    return f"{dots_client.dots_version()}/{table_detect.detector_version()}/dpi{dpi}"


def source_sha256_of(data: bytes) -> str:
    """SHA-256 of the raw source PDF bytes.

    Stored as ``metadata.source_sha256`` and used only to decide whether
    re-extraction is needed. Like ``extractor_version`` it is excluded from
    ``qa_content_hash``: a byte-level PDF change (a re-stamped download, a new
    timestamp in the file trailer) is not by itself a semantic record change.
    """
    return hashlib.sha256(data).hexdigest()


_MIN_SERIALS = 4
_X_TOL = 6.0            # column x clustering tolerance (pt)
_ROW_ALIGN = 0.28       # content counts as a row's baseline within ±0.28·pitch
_MIN_PITCH = 8.0        # sanity guards against false-positive serial columns
_MAX_PITCH = 32.0
_LINE_MERGE_TOL = 2.0   # pt — same-visual-line baseline wobble
_OCR_PAGE_TIMEOUT_SEC = 10.0

# Strict Multi-Page Continuity Threshold: sim < 0.85 -> NEVER MERGE
_CONTINUITY_CONFIDENCE_THRESHOLD = 0.85


def clean_cell_text(text: str | None) -> str:
    """Normalize cell text. Only fuse line-wrapped decimals (e.g. '117.2\n5'), never separate numbers."""
    if text is None:
        return ""
    raw = str(text)
    if "\n" in raw or "\r" in raw:
        raw = re.sub(r'(\b\d+\.\d+)[\r\n]+(\d+\b)', r'\1\2', raw)
        raw = re.sub(r'(\b[A-Za-z]+)-[\r\n]+([a-z]+\b)', r'\1\2', raw)
    return " ".join(raw.split())


def normalize_token(s: str) -> str:
    return " ".join(re.sub(r'[^a-zA-Z0-9]', ' ', str(s or "")).lower().split())


def header_similarity(hdr1: list[str], hdr2: list[str]) -> float:
    if not hdr1 or not hdr2 or len(hdr1) != len(hdr2):
        return 0.0
    matches = sum(1 for c1, c2 in zip(hdr1, hdr2) if normalize_token(c1) == normalize_token(c2) and normalize_token(c1))
    valid_cols = sum(1 for c1 in hdr1 if normalize_token(c1))
    return matches / max(valid_cols, 1)


def render_table_as_markdown(table_data: list[list[str]]) -> str:
    """Format extracted 2D table data into a clean, syntactically valid Markdown table."""
    if not table_data or len(table_data) < 2:
        return ""
    cleaned = []
    for row in table_data:
        clean_row = [clean_cell_text(c) for c in row]
        if any(clean_row):
            cleaned.append(clean_row)
    if len(cleaned) < 2:
        return ""

    max_cols = max(len(r) for r in cleaned)
    if max_cols < 2:
        return ""

    padded = [r + [""] * (max_cols - len(r)) for r in cleaned]
    hdr = padded[0]
    sep = ["---"] * max_cols

    lines = [
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(sep) + " |"
    ]
    for r in padded[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def split_question_answer(text: str) -> tuple[str, str] | None:
    """Split extracted document text into (question, answer).
    
    Identifies the canonical parliamentary reply boundary (e.g. ANSWER, REPLY,
    MINISTER OF..., STATEMENT REFERRED TO IN REPLY TO...). Never splits on
    document header furniture such as LOK SABHA or RAJYA SABHA.
    """
    if not text or not text.strip():
        return None

    patterns = [
        r"(?i)\n\s*(?:ANSWER|REPLY|A\s*N\s*S\s*W\s*E\s*R)\s*[:\n]",
        r"(?i)\n\s*(?:THE\s+)?MINISTER\s+OF\s+[^\n]+(?:\s+STATEMENT)?\s*[:\n]",
        r"(?i)\n\s*STATEMENT\s+REFERRED\s+TO\s+IN\s+REPLY\s+TO\s+[^\n]+[:\n]",
        r"(?i)\n\s*\([a-h]\)\s*(?:to|&|\band\b)\s*\([a-h]\)\s*[:\n]",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            idx = m.start()
            return text[:idx].strip(), text[idx:].strip()

    split_idx = len(text) // 3
    return text[:split_idx].strip(), text[split_idx:].strip()


def _page_text(page) -> str:
    """Extract full structured text from a single page."""
    lines = _page_lines(page)
    if not lines:
        return ""
    runs = _serial_runs(lines)
    if not runs:
        return _render_merged(lines)
    rows = _reconstruct_rows(lines, runs[0])
    if rows is None:
        return _render_merged(lines)
    lo, hi = runs[0]["lo"], runs[0]["hi"]
    head = _render_merged([l for l in lines if l["yc"] < lo]).splitlines()
    tail = _render_merged([l for l in lines if l["yc"] > hi]).splitlines()
    return "\n".join(head + rows + tail) + "\n"


#: ``mode="dots_only"`` — every page goes straight to DOTS OCR. No PicoDet,
#: no table detection, no Tesseract, no legacy strategy, no fallback. Used by
#: the INCOIS tender ("budget") collection, where DOTS was chosen on
#: extraction quality and the alternatives were explicitly rejected.
MODE_AUTO = "auto"
MODE_DOTS_ONLY = "dots_only"


def extract_pdf_text(
    data: bytes,
    enable_ocr: bool = True,
    *,
    mode: str = MODE_AUTO,
) -> str | None:
    """Extract whole-document text with multi-page, borderless, bordered, and scanned tables.

    Raises DependencyMissingError if PyMuPDF or required OCR is unavailable.

    ``mode`` selects the per-page strategy and is keyword-only with an
    ``auto`` default, so the public seam every existing caller uses
    (``extract_pdf_text(data, enable_ocr)``) is unchanged:

    * ``auto``      — the existing behaviour, including the opt-in
      PicoDet -> DOTS table routing.
    * ``dots_only`` — DOTS OCR for every page, exclusively. Deliberately
      bypasses PicoDet, ``find_tables()``, Tesseract and the legacy
      strategies, and has NO fallback: a failure raises so the caller can
      record it rather than committing degraded text.
    """
    if mode not in (MODE_AUTO, MODE_DOTS_ONLY):
        raise ValueError(f"unknown extraction mode {mode!r}")
    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Corrupted or unreadable PDF stream: {exc}") from exc

    try:
        page_payloads = []
        for page_idx, page in enumerate(doc):
            if mode == MODE_DOTS_ONLY:
                payload = _extract_page_payload_dots_only(page, page_idx + 1)
            else:
                payload = _extract_page_payload(page, page_idx + 1, enable_ocr=enable_ocr)
            page_payloads.append(payload)

        stitched_chunks = _stitch_multipage_payloads(page_payloads)
        res = "\n\n".join(c for c in stitched_chunks if c.strip())
        return res if res.strip() else None
    finally:
        doc.close()


def extract_pdf_text_with_fallback(data: bytes, enable_ocr: bool = True) -> str:
    """Shared PDF extraction entrypoint with strict error propagation."""
    text = extract_pdf_text(data, enable_ocr=enable_ocr)
    return text if text and text.strip() else ""


def _extract_page_payload(page, page_num: int, enable_ocr: bool = True) -> dict[str, Any]:
    lines = _page_lines(page)
    native_char_count = sum(len(l["text"]) for l in lines)
    
    # Check if page is pure scanned (< 30 native text characters)
    if not lines or native_char_count < 30:
        if enable_ocr:
            from src.data.ocr_table_extract import (
                DependencyMissingError as OcrDependencyError,
                is_tesseract_available,
                ocr_page_to_structured_text,
            )
            if not is_tesseract_available():
                raise DependencyMissingError("Tesseract OCR is required for scanned PDF pages but is unavailable.")
            try:
                ocr_res = ocr_page_to_structured_text(page, timeout=_OCR_PAGE_TIMEOUT_SEC)
                if ocr_res and ocr_res.strip():
                    return {"page_num": page_num, "type": "ocr_text", "content": ocr_res, "tables": []}
            except TimeoutError as exc:
                raise ExtractionTimeoutError(f"OCR timeout exceeded on Page {page_num} (>10.0s); {exc}") from exc
            except OcrDependencyError as exc:
                raise DependencyMissingError(str(exc)) from exc
            except Exception as exc:
                raise RuntimeError(f"OCR execution failed on Page {page_num}: {exc}") from exc

        if not lines:
            return {"page_num": page_num, "type": "empty", "content": "", "tables": []}

    # PicoDet -> DOTS routing gate.
    # When enabled this REPLACES Strategies 1-3 (the competing table extractors)
    # for this page: PicoDet is the only detector and DOTS is the only table
    # extractor. A page PicoDet clears falls through to the same baseline
    # rendering Strategy 4 would have produced, so table-free documents are
    # unaffected. Scanned-page OCR above still runs first and is untouched.
    if dots_routing_enabled():
        return _extract_page_payload_dots(page, page_num, lines, native_char_count)

    # Strategy 1: Check for PyMuPDF structured vector/grid tables
    try:
        tabs = page.find_tables()
        valid_tables = []
        if tabs.tables:
            for t in tabs.tables:
                ext = t.extract()
                clean_rows = [[clean_cell_text(c) for c in r] for r in ext if any(r)]
                if len(clean_rows) >= 2 and max(len(r) for r in clean_rows) >= 2:
                    valid_tables.append({
                        "bbox": t.bbox,
                        "rows": clean_rows,
                        "cols": len(clean_rows[0]),
                        "header": clean_rows[0]
                    })

        if valid_tables:
            valid_tables.sort(key=lambda t: t["bbox"][1])
            return {
                "page_num": page_num,
                "type": "grid_tables",
                "lines": lines,
                "tables": valid_tables
            }
    except Exception:
        pass

    # Strategy 2: Check for Mixed Pages (Native text + embedded scanned/raster table image)
    if enable_ocr:
        try:
            images = page.get_images()
            if images:
                from src.data.ocr_table_extract import (
                    DependencyMissingError as OcrDependencyError,
                    is_tesseract_available,
                    ocr_image_to_structured_text,
                )
                for img_info in images:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        if r.width > 200 and r.height > 100:
                            overlapping = any(
                                (l["x0"] < r.x1 and l["x1"] > r.x0 and l["y0"] < r.y1 and l["y1"] > r.y0)
                                for l in lines
                            )
                            if not overlapping:
                                if not is_tesseract_available():
                                    raise DependencyMissingError("Tesseract OCR is required for embedded image tables but is unavailable.")
                                pix = page.get_pixmap(clip=r, dpi=300)
                                from PIL import Image
                                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                                ocr_sub = ocr_image_to_structured_text(pil_img, timeout=_OCR_PAGE_TIMEOUT_SEC)
                                if ocr_sub and ("|" in ocr_sub or len(ocr_sub) > 40):
                                    head_lines = [l for l in lines if l["yc"] < r.y0]
                                    tail_lines = [l for l in lines if l["yc"] > r.y1]
                                    head_text = _render_merged(head_lines).strip()
                                    tail_text = _render_merged(tail_lines).strip()
                                    mixed_content = "\n\n".join(p for p in [head_text, ocr_sub, tail_text] if p)
                                    return {"page_num": page_num, "type": "mixed_page", "content": mixed_content, "tables": []}
        except DependencyMissingError:
            raise
        except TimeoutError as exc:
            raise ExtractionTimeoutError(f"OCR timeout on embedded image on Page {page_num}: {exc}") from exc
        except Exception:
            pass

    # Strategy 3: Borderless serial-run geometric reconstruction
    runs = _serial_runs(lines)
    if runs:
        rows = _reconstruct_rows(lines, runs[0])
        if rows is not None:
            lo, hi = runs[0]["lo"], runs[0]["hi"]
            head = _render_merged([l for l in lines if l["yc"] < lo]).splitlines()
            tail = _render_merged([l for l in lines if l["yc"] > hi]).splitlines()
            content = "\n".join(head + rows + tail) + "\n"
            return {"page_num": page_num, "type": "serial_table", "content": content, "tables": []}

    # Strategy 4: Baseline-merged prose
    content = _render_merged(lines)
    return {"page_num": page_num, "type": "prose", "content": content, "tables": []}


def _extract_page_payload_dots_only(page, page_num: int) -> dict[str, Any]:
    """DOTS OCR for one page, exclusively — the INCOIS tender ("budget") route.

    Differs from :func:`_extract_page_payload_dots` in three ways, all of them
    required by the tender collection's contract:

    * **No detector.** PicoDet is never consulted, so there is no
      detector-fired-but-no-table case; ``detector_found_table=False`` keeps
      the corresponding validator rule from firing on a page that is
      legitimately prose.
    * **No native-text comparison.** ``native_char_count=0`` because ~92% of
      these PDFs are image-only scans with no text layer at all; comparing
      DOTS output against a zero-length native layer would be meaningless.
    * **No fallback.** Tesseract and the legacy strategies are not reachable
      from here. DOTS failures propagate so ingestion can record a failure,
      rather than quietly substituting the lower-quality extraction this
      category exists to avoid.
    """
    from src.data import dots_client

    content = dots_client.get_client().page_to_text(
        page,
        page_num,
        native_char_count=0,
        detector_found_table=False,
    )
    return {"page_num": page_num, "type": DOTS_PAYLOAD_TYPE, "content": content, "tables": []}


def _extract_page_payload_dots(page, page_num: int, lines: list[dict], native_char_count: int) -> dict[str, Any]:
    """Route one page through PicoDet and, if a table is present, DOTS.

    Contract:

    * **No table detected** -> the page renders through ``_render_merged``, which
      is byte-identical to what Strategy 4 produces today. A document with no
      tables is therefore unchanged by enabling routing.
    * **Table detected** -> DOTS owns the page outright. Its text replaces the
      page entirely; nothing is merged with a legacy table extraction.
    * **Detector unavailable / DOTS unreachable / output rejected** -> the
      exception propagates. There is deliberately no fallback to an inferior
      table extractor: a hard, diagnosable failure leaves the previous corpus
      row intact, whereas a silent downgrade would overwrite good data with
      worse data and nobody would ever know.

    The single exception is ``RULE_NO_TABLE_FOUND`` — PicoDet fired but DOTS
    found no table. That is a detector false positive, not an extraction
    failure, so the page keeps its legacy text.
    """
    from src.data import dots_client, table_detect

    # TableDetectorUnavailable and DotsUnavailable are both RuntimeError
    # subclasses and are intentionally allowed to propagate to the caller.
    if not table_detect.page_has_table(page):
        return {"page_num": page_num, "type": "prose", "content": _render_merged(lines), "tables": []}

    try:
        content = dots_client.get_client().page_to_text(
            page,
            page_num,
            native_char_count=native_char_count,
            detector_found_table=True,
        )
    except dots_client.DotsInvalidOutput as exc:
        if exc.rule == dots_client.RULE_NO_TABLE_FOUND:
            return {"page_num": page_num, "type": "prose", "content": _render_merged(lines), "tables": []}
        raise

    return {"page_num": page_num, "type": DOTS_PAYLOAD_TYPE, "content": content, "tables": []}


def _fuse_split_row(prev_row: list[str], cont_row: list[str]) -> list[str]:
    """Fuse a page-boundary split row into the preceding row without truncation or column shift."""
    fused = list(prev_row)
    for i in range(min(len(fused), len(cont_row))):
        c_text = cont_row[i].strip()
        if c_text:
            if fused[i].strip():
                fused[i] = f"{fused[i]} {c_text}".strip()
            else:
                fused[i] = c_text
    return fused


def _is_split_continuation_row(row: list[str], active_header: list[str]) -> bool:
    """Detect if a row on a continuation page is an unsplit wrapped segment of the previous row."""
    if not row or not any(row):
        return False
    if row[0].strip() == "" and any(c.strip() for c in row[1:]):
        return True
    return False


def _extract_serial_number(row: list[str]) -> int | None:
    """Extract a positive integer serial number from the first non-empty cell of a table row."""
    if not row:
        return None
    for cell in row[:2]:
        cell_str = clean_cell_text(cell)
        if not cell_str:
            continue
        m = re.match(r'^\s*\(?\s*(\d{1,5})\s*[\.\)\-]?\s*$', cell_str)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        break
    return None


def _stitch_multipage_payloads(payloads: list[dict[str, Any]]) -> list[str]:
    """Stitch multi-page tables across consecutive pages, prune repeated headers, fuse split rows, and output clean Markdown in visual order."""
    output_chunks = []
    active_multi_table = None

    for p in payloads:
        ptype = p.get("type")
        
        if ptype == "grid_tables":
            lines = p["lines"]
            tables = p["tables"]
            page_num = p["page_num"]
            
            for t_idx, t in enumerate(tables):
                t_rows = t["rows"]
                t_cols = t["cols"]
                t_header = t["header"]
                
                prev_bound = tables[t_idx - 1]["bbox"][3] if t_idx > 0 else -1.0
                curr_bound = t["bbox"][1]
                mid_lines = [l for l in lines if prev_bound < l["yc"] < curr_bound]
                mid_text = _render_merged(mid_lines).strip()
                
                is_section_boundary = bool(re.search(r'(?i)\b(?:annexure|statement|appendix)\s*[-:]?\s*(?:[ivx\d]+|[b-z])\b', mid_text))

                if active_multi_table is not None:
                    is_consec = (page_num == active_multi_table["last_page"] + 1)
                    same_cols = (t_cols == active_multi_table["cols"])
                    sim = header_similarity(active_multi_table["header"], t_header)
                    
                    # 1. Repeated header match
                    has_repeated_header = (sim >= _CONTINUITY_CONFIDENCE_THRESHOLD)

                    # 2. Headerless serial continuation match
                    is_serial_continuation = False
                    if not has_repeated_header and is_consec and same_cols and not is_section_boundary:
                        n_prev = None
                        for r in reversed(active_multi_table["rows"]):
                            sp = _extract_serial_number(r)
                            if sp is not None:
                                n_prev = sp
                                break
                        
                        if n_prev is not None and len(t_rows) > 0:
                            n_curr = _extract_serial_number(t_rows[0])
                            if n_curr is None and _is_split_continuation_row(t_rows[0], active_multi_table["header"]) and len(t_rows) > 1:
                                n_curr = _extract_serial_number(t_rows[1])
                            
                            if n_curr is not None and n_curr == n_prev + 1:
                                is_serial_continuation = True
                    
                    is_continuation = (
                        is_consec
                        and same_cols
                        and not is_section_boundary
                        and (has_repeated_header or is_serial_continuation)
                    )

                    if is_continuation:
                        if has_repeated_header:
                            if header_similarity(active_multi_table["header"], t_rows[0]) >= _CONTINUITY_CONFIDENCE_THRESHOLD:
                                incoming_rows = t_rows[1:]
                            else:
                                incoming_rows = t_rows
                        else:
                            # Headerless continuation: row 0 is a data row (e.g. Row 21)
                            incoming_rows = t_rows
                        
                        for r in incoming_rows:
                            if _is_split_continuation_row(r, active_multi_table["header"]) and len(active_multi_table["rows"]) > 1:
                                active_multi_table["rows"][-1] = _fuse_split_row(active_multi_table["rows"][-1], r)
                            else:
                                active_multi_table["rows"].append(r)
                                
                        active_multi_table["last_page"] = page_num
                        continue
                    else:
                        output_chunks.append(render_table_as_markdown(active_multi_table["rows"]))
                        active_multi_table = None

                if mid_text:
                    output_chunks.append(mid_text)

                active_multi_table = {
                    "start_page": page_num,
                    "last_page": page_num,
                    "cols": t_cols,
                    "header": t_header,
                    "rows": list(t_rows)
                }

            tb_last = tables[-1]
            after_lines = [l for l in lines if l["yc"] > tb_last["bbox"][3]]
            after_text = _render_merged(after_lines).strip()
            if after_text:
                if active_multi_table is not None:
                    output_chunks.append(render_table_as_markdown(active_multi_table["rows"]))
                    active_multi_table = None
                output_chunks.append(after_text)

        else:
            if active_multi_table is not None:
                output_chunks.append(render_table_as_markdown(active_multi_table["rows"]))
                active_multi_table = None
                
            content = p.get("content", "").strip()
            if content:
                output_chunks.append(content)

    if active_multi_table is not None:
        output_chunks.append(render_table_as_markdown(active_multi_table["rows"]))

    return output_chunks


def _render_merged(lines: list[dict]) -> str:
    merged = _merge_visual_lines(lines)
    return "\n".join(l["text"] for l in merged) + ("\n" if merged else "")


def _merge_visual_lines(lines: list[dict]) -> list[dict]:
    bands: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: l["yc"]):
        if bands and abs(l["yc"] - bands[-1][-1]["yc"]) <= _LINE_MERGE_TOL:
            bands[-1].append(l)
        else:
            bands.append([l])
    out: list[dict] = []
    for band in bands:
        band.sort(key=lambda l: l["x0"])
        fused = dict(band[0])
        fused["text"] = " ".join(l["text"] for l in band)
        fused["x1"] = max(l["x1"] for l in band)
        out.append(fused)
    return out


def _page_lines(page) -> list[dict]:
    d = page.get_text("dict")
    out: list[dict] = []
    for b in d["blocks"]:
        for ln in b.get("lines", []):
            text = " ".join(s["text"].strip() for s in ln["spans"] if s["text"].strip())
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "yc": (y0 + y1) / 2, "text": text})
    out.sort(key=lambda l: (l["yc"], l["x0"]))
    return out


def _serial_runs(lines: list[dict]) -> list[dict]:
    best: list[dict] = []
    by_band: dict[int, list[dict]] = {}
    for l in lines:
        if l["text"].isdigit() and len(l["text"]) <= 3:
            by_band.setdefault(round(l["x0"] / _X_TOL), []).append(l)
    for band_lines in by_band.values():
        band_lines = [dict(l) for l in sorted(band_lines, key=lambda l: l["yc"])]
        cur = [band_lines[0]]
        for prev, cur_l in zip(band_lines, band_lines[1:]):
            if int(cur_l["text"]) == int(prev["text"]) + 1:
                cur.append(cur_l)
            else:
                _consider_run(cur, best)
                cur = [cur_l]
        _consider_run(cur, best)
    return best


def _consider_run(candidate: list[dict], best: list[dict]) -> None:
    if len(candidate) < _MIN_SERIALS or len(candidate) <= len(best):
        return
    ys = [l["yc"] for l in candidate]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    if not gaps:
        return
    pitch = median(gaps)
    if not (_MIN_PITCH <= pitch <= _MAX_PITCH):
        return
    best.clear()
    best.append({"serials": candidate, "ys": ys, "pitch": pitch})


def _reconstruct_rows(lines: list[dict], run: dict) -> list[str] | None:
    serials, ys, p = run["serials"], run["ys"], run["pitch"]
    lo, hi = ys[0] - 0.6 * p, ys[-1] + 1.2 * p
    run["lo"], run["hi"] = lo, hi
    rows_in = [l for l in lines if lo <= l["yc"] <= hi]
    run["rows_in"] = rows_in
    if len(rows_in) < len(serials):
        return None

    xs = sorted({round(l["x0"], 1) for l in rows_in})
    cols: list[float] = []
    for x in xs:
        if not cols or x - cols[-1] > _X_TOL:
            cols.append(x)
    if len(cols) < 2:
        return None

    def col_of(l: dict) -> int:
        return min(range(len(cols)), key=lambda i: abs(l["x0"] - cols[i]))

    serial_row = {id(l): k for k, l in enumerate(serials)}
    serial_ids = set(serial_row)
    tol = _ROW_ALIGN * p
    anchors: dict[int, list[dict]] = {k: [] for k in range(len(serials))}
    seen: set[int] = set()
    for l in rows_in:
        if id(l) in serial_ids:
            anchors[serial_row[id(l)]].append(l)
            seen.add(id(l))
            continue
        for k, yc in enumerate(ys):
            if abs(l["yc"] - yc) <= tol:
                anchors[k].append(l)
                seen.add(id(l))
                break
    orphans = [l for l in rows_in if id(l) not in seen]

    station_col = max(cols)
    station_lines = sorted(
        [l for l in rows_in if abs(l["x0"] - station_col) <= _X_TOL],
        key=lambda l: l["yc"],
    )

    def _depth(text: str) -> int:
        return text.count("(") - text.count(")")

    def _station_cell_text(row: int) -> str:
        return " ".join(
            l["text"]
            for l in sorted(anchors[row], key=lambda l: (l["yc"], l["x0"]))
            if abs(l["x0"] - station_col) <= _X_TOL
        )

    def _own_orphan(o: dict, k: int) -> int:
        if k >= len(ys) - 1:
            return len(ys) - 1
        i = next((i for i, l in enumerate(station_lines) if l is o), None)
        if i is None:
            return k if abs(o["yc"] - ys[k]) <= abs(o["yc"] - ys[k + 1]) else k + 1
        do = _depth(o["text"])
        if do > 0:
            db = _depth(_station_cell_text(k + 1))
            if db < 0 and do + db == 0:
                return k + 1
        elif do < 0:
            da = _depth(_station_cell_text(k))
            if da > 0 and do + da == 0:
                return k
        gap_up = (o["y0"] - station_lines[i - 1]["y1"]) if i > 0 else 1e9
        gap_down = ((station_lines[i + 1]["y0"] - o["y1"])
                    if i + 1 < len(station_lines) else 1e9)
        if gap_down < gap_up:
            return k + 1
        return k

    owners: dict[int, list[dict]] = {k: [] for k in range(len(serials))}
    for o in orphans:
        k = max((i for i, yc in enumerate(ys) if yc <= o["yc"]), default=0)
        owners[_own_orphan(o, k)].append(o)

    rows: list[str] = []
    for k in range(len(serials)):
        cells: dict[int, list[dict]] = {}
        for l in anchors[k] + owners[k]:
            cells.setdefault(col_of(l), []).append(l)
        parts = []
        for ci in range(len(cols)):
            cl = cells.get(ci)
            if cl:
                parts.append(" ".join(c["text"] for c in
                                      sorted(cl, key=lambda c: c["yc"])))
        rowline = " ".join(parts).strip()
        if rowline:
            rows.append(rowline)
    return rows or None
