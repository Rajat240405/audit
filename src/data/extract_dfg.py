"""Demand for Grants (DfG) document-family extraction — MoES specific.

Ported from the forensically-validated prototype (``work4/extract_dfg_proto.py``,
self-tests A1–A5) plus the language/type classifier
(``work4/dfg_classifier.py``, 15/15 fixtures). Every threshold below is a
measured value from the five live DfG PDFs, not a guess.

WHY THIS IS A SEPARATE MODULE
-----------------------------
The generic path (``pdf_table_extract.extract_pdf_text_with_fallback``) is tuned
for Lok Sabha serial-numbered *borderless* annexure tables. DfG documents are a
different physical family and it fails on 4 of 5 of them:

* 02/04/05 are pure scans — the generic extractor returns "" and the caller
  falls back to eng-only OCR, which erases every Devanagari character
  (measured: 0 dev chars on 04 p11, "Fit PeN 24 - wed fags Hares" garbage).
* 01's core tables are vector outlines with no text layer.
* 03's text layer is poisoned by a broken ToUnicode map (453 mojibake lines),
  and its Devanagari ratio is 0 — which must NOT be read as "English-only".
* ``_clean()`` then flattens whatever survived (589 → 0 newlines on 01).

Three deterministic routes, chosen per page:

    ROUTE A  native text + geometric lattice  → 7-column rows
    ROUTE B  render → OCR (hin+eng)           → line rows (outline/broken pages)
    ROUTE C  render → morphological grid → per-cell OCR   (scans)

Nothing here is probabilistic or LLM-based. Correctness is established by
integer arithmetic: sum(component rows) == reported total.

This module imports nothing from the LS or RS pipelines. The only shared
primitive it reuses is ``pdf_table_extract._page_text`` (line-band reader).
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── measured constants ───────────────────────────────────────────────────────

#: Route C render DPI. 300 is the measured floor at which 04 (300-DPI source)
#: and 05 (150-DPI source) reconstruct row/column-correctly.
GRID_DPI = 300
#: Route B render DPI for flat-page OCR.
OCR_DPI = 200
#: Morphological line-detection thresholds. MUST stay fractional: 04's rules do
#: not span the full page width, so full-width thresholds find nothing.
#:
#: Calibrated at 0.08, not 0.15. Measured on 04: at 0.15 page 12 detects 29 rows
#: but page 11 — whose rules are fainter — detects only 4, silently losing the
#: opening components of the Total(07) block that page 12 closes. At 0.08 page 11
#: recovers 28 rows while page 12 is UNCHANGED at 29. 0.05 over-detects (32).
GRID_LINE_FRACTION = 0.08
#: Below this effective DPI a scan is physically digit-limited and the document
#: is FAIL-marked rather than guessed at. 02 is ~72 DPI (1151×862 image on a
#: 1151×862 pt page); 05 is ~150 and is fine.
DPI_FAIL_THRESHOLD = 100
#: Mojibake score at or above which a text layer is untrustworthy → Route B.
MOJIBAKE_THRESHOLD = 0.05
#: Devanagari chars/page above which an OCR probe proves the scan is bilingual.
BILINGUAL_SCAN_DEV_CHARS = 50
#: Fraction of code-column cells that must parse as account codes before the
#: page counts as structurally sound.
CODE_VALIDITY_DEGRADED = 0.60

# ── regexes (verbatim from the validated prototype) ──────────────────────────

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
#: amounts: >=3 digits with optional comma groups (₹-thousands cells).
#: The ``|\d{3,}`` branch is load-bearing — 3-digit amounts like 900 were the
#: last bug class found in the prototype.
AMT = re.compile(r"\d{1,3}(?:,\d{3})+|\d{3,}")
CODE_FULL = re.compile(r"(\d{4}\.\d{2}\.\d{3}\.\d{2})")
CODE_SUB = re.compile(r"\b(\d{2})[.,']\s*(\d{2})[.,']\s*(\d{2})\b")
# A block-closing row may be labelled in Latin ("Total", "कुल /Total") or purely
# in Devanagari. Measured on 04 p11 the block-06 total reads "कुला ०७ (06)" with
# no Latin word at all; matching only the Latin form merged blocks 06 and 07
# into a single carried sum and broke the cross-page reconciliation.
TOTAL_ROW = re.compile(r"कुल\s*ा?|कल\s*/|Grand\s+Total|\bTotal\b", re.I)
VOTED_ROW = re.compile(r"/\s*(?:Voted|Chargeable)\b|मतदान|भारित", re.I)
#: repeated per-page header — must be classified, never summed as a component
PAGE_HEADER = re.compile(
    r"Demand\s+No\.?\s*\d+|Detailed\s+Demands?\s+for\s+Grants|दetailed|"
    r"बजट\s+अनुमान|BUDGET\s+ESTIMATE|REVISED\s+ESTIMATE|वास्तविक|ACTUALS",
    re.I,
)
#: "Continued on next page MH 3403" style continuation bands
CONTINUATION_BAND = re.compile(r"Continued\s+on\s+next\s+page|अगले\s+पृष्ठ", re.I)
FISCAL_RE = re.compile(r"(20\d{2})\s*[-–—/]\s*(\d{2,4})")
DEMAND_NO_RE = re.compile(r"Demand\s+No\.?\s*(\d+)", re.I)
# "2025-2026" written without a slash, as in the DfG front-matter banners.
DEMAND_FY_RE = re.compile(r"(20\d{2})\s*[-–—]\s*(20\d{2}|\d{2})\b")

DEV_RANGE = ("\u0900", "\u097f")


# ── data model (normative — see spec §"Data structures") ─────────────────────

@dataclass
class DfGRow:
    """One reconstructed table row, with its cell positions preserved."""

    page: int                          # 1-based source page
    seq: int                           # row order within page
    actuals: str | None = None         # raw cell strings (₹-thousands)
    budget: str | None = None
    revised: str | None = None
    hi: str | None = None              # विवरण / HEADS (Hindi)
    code: str | None = None            # 11.01.01 / 3451.00.090.17
    en: str | None = None              # Description
    budget_next: str | None = None
    role: str = "component"            # component|subtotal|total|voted|header|band
    conf: float = 1.0                  # per-cell OCR confidence product (Route C)
    continued_from_prev_page: bool = False
    #: physical cells exactly as the route produced them, positions intact.
    #: Route A lattice widths vary page to page (measured: 6 columns on 01 p41,
    #: 7 on the Route C scans), so the canonical 7-slot view above is a
    #: projection — the arithmetic gates run on THIS, never on the projection.
    raw: list[str] = field(default_factory=list)

    def as_cells(self) -> list[str | None]:
        return [self.actuals, self.budget, self.revised, self.hi,
                self.code, self.en, self.budget_next]


@dataclass
class ArithmeticCheck:
    """One sum(component rows) == reported total constraint."""

    page: int
    label: str
    components: int
    computed: int
    reported: int
    spans_pages: bool = False

    @property
    def ok(self) -> bool:
        return self.computed == self.reported


@dataclass
class PageResult:
    page: int
    route: str                         # A | A-line | B | C
    rows: list[DfGRow] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)
    checks: list[ArithmeticCheck] = field(default_factory=list)
    verdict: str = "PASS"
    notes: list[str] = field(default_factory=list)
    dev_chars: int = 0
    grid_shape: tuple[int, int] | None = None
    code_validity: float | None = None


@dataclass
class DfGProvenance:
    """Provenance block carried into the corpus record (spec req. I)."""

    fiscal_year: str | None
    demand_no: int | None
    source_url: str | None
    page_count: int
    language_class: str
    extraction_route: dict[int, str]
    quality_verdict: str
    sha256: str
    pages: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fiscal_year": self.fiscal_year,
            "demand_no": self.demand_no,
            "source_url": self.source_url,
            "page_count": self.page_count,
            "language_class": self.language_class,
            "extraction_route": {str(k): v for k, v in sorted(self.extraction_route.items())},
            "quality_verdict": self.quality_verdict,
            "sha256": self.sha256,
            "pages": self.pages,
        }


@dataclass
class DfGDocument:
    path: str
    pages: list[PageResult] = field(default_factory=list)
    provenance: DfGProvenance | None = None
    verdict: str = "PASS"

    @property
    def all_rows(self) -> list[DfGRow]:
        return [r for p in self.pages for r in p.rows]


# ── language / encoding probes (ported from dfg_classifier.py) ───────────────

def count_devanagari(text: str) -> int:
    lo, hi = DEV_RANGE
    return sum(1 for c in text if lo <= c <= hi)


def count_letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


_MOJI_TOKEN = re.compile(
    r"\b[a-z]{1,8}\{[a-z]{1,4}\b|\bq[a-z]{0,3}6[a-z]{0,3}\b|"
    r"\b(?:qq|ffi|frfi|trfi|lri|sEsEr|iE)\b"
)


def mojibake_score(text: str) -> float:
    """Fraction of alpha tokens that look like broken-ToUnicode output.

    Measured on 03: 453 mojibake lines, score ≥ 0.05. Devanagari ratio is 0 on
    that document, so a zero ratio must never be read as "English-only" — this
    score is the only reliable signal there.
    """
    toks = re.findall(r"[A-Za-z][A-Za-z{\.]{2,}", text or "")
    if not toks:
        return 0.0
    return sum(1 for t in toks if _MOJI_TOKEN.search(t)) / len(toks)


def classify_language(
    *,
    stem: str = "",
    text_layer: str = "",
    ocr_probe_dev: int | None = None,
    scanned: bool | None = None,
) -> dict[str, Any]:
    """Deterministic language class from cheap probes (ported verbatim)."""
    if re.search(r"[-._]hin\b|[-._]hin\.", stem):
        slot = "hin"
    elif re.search(r"[-._]both\b", stem):
        slot = "both"
    elif re.search(r"[-._]eng\b", stem):
        slot = "eng"
    else:
        slot = "unknown"

    if scanned is None:
        scanned = not (text_layer or "").strip()

    if scanned:
        if ocr_probe_dev is None:
            return {"lang_class": "scanned_unprobeable", "needs_ocr": True, "slot": slot}
        if ocr_probe_dev >= BILINGUAL_SCAN_DEV_CHARS:
            return {"lang_class": "bilingual_scan", "needs_ocr": True, "slot": slot}
        return {"lang_class": "english_only_scan", "needs_ocr": True, "slot": slot}

    dev, let = count_devanagari(text_layer), count_letters(text_layer)
    ratio = dev / max(1, let)
    moji = mojibake_score(text_layer)
    if ratio >= 0.5 and dev >= 50:      # same bar as src/scripts/purge_hindi_rows.py
        out = "hindi_only"
    elif ratio >= 0.02:
        out = "bilingual_clean"
    elif slot in ("both", "hin") and dev == 0 and let > 0:
        # The staging slot declares this document bilingual, yet its text layer
        # holds no Devanagari at all. That is a broken ToUnicode map, not an
        # English document (spec req. G: "Devanagari-ratio=0 is NOT
        # English-only"). Measured on 03: 0 Devanagari chars across 47 pages,
        # full-page mojibake scores 0.031–0.050 — i.e. below the token
        # threshold, so the threshold alone would misfile it as english_only.
        # This rule keys on the document's own declared slot instead.
        out = "bilingual_broken_encoding"
    else:
        out = "bilingual_broken_encoding" if moji >= MOJIBAKE_THRESHOLD else "english_only"
    return {"lang_class": out, "needs_ocr": False, "slot": slot,
            "dev_ratio": round(ratio, 3), "mojibake": round(moji, 3)}


# ── Route A: native text + geometric lattice ─────────────────────────────────

def _cuts(xs: Sequence[float], tol: float = 6.0) -> list[float]:
    """Cluster near-duplicate rule coordinates into distinct column cuts."""
    out: list[float] = []
    for x in sorted(xs):
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


def lattice_cuts(page: Any, min_len: float = 30) -> tuple[list[float], list[float]]:
    """Vertical / horizontal rule positions from the page's vector drawings.

    Measured: 01's annex pages carry clean vertical rules (6 cuts on p41);
    03's body is borderless (≈1 cut) and must fall back to line rows.
    """
    v: list[float] = []
    h: list[float] = []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.x - p2.x) < 1 and abs(p1.y - p2.y) > min_len:
                    v.append((p1.x + p2.x) / 2)
                elif abs(p1.y - p2.y) < 1 and abs(p2.x - p1.x) > min_len:
                    h.append((p1.y + p2.y) / 2)
            elif it[0] == "re":
                r = it[1]
                if r.width < 2 and r.height > min_len:
                    v.append((r.x0 + r.x1) / 2)
                elif r.height < 2 and r.width > min_len:
                    h.append((r.y0 + r.y1) / 2)
    return _cuts(v), _cuts(h)


def page_words(page: Any) -> list[tuple[float, float, str]]:
    """Span-level (centre_x, centre_y, text) triples, reading order."""
    out: list[tuple[float, float, str]] = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                t = sp["text"].strip()
                if t:
                    x0, y0, x1, y1 = sp["bbox"]
                    out.append(((x0 + x1) / 2, (y0 + y1) / 2, t))
    return out


def rows_from_words(
    words: Sequence[tuple[float, float, str]],
    vcuts: Sequence[float],
    ytol: float = 9.0,
) -> list[list[str]]:
    """Band words into visual rows, then bucket each into a column by x."""
    ys = sorted({round(w[1], 1) for w in words})
    bands: list[list[float]] = []
    for y in ys:
        if not bands or y - bands[-1][-1] > ytol:
            bands.append([y])
        else:
            bands[-1].append(y)
    rows: list[list[str]] = []
    for b in bands:
        lo, hi = b[0] - 2, b[-1] + 2
        cells: dict[int, list[str]] = {}
        for x, y, t in words:
            if lo <= y <= hi:
                ci = sum(1 for c in vcuts if x > c)
                cells.setdefault(ci, []).append(t)
        row = [" ".join(cells.get(i, [])) for i in range(len(vcuts) + 1)]
        if any(c.strip() for c in row):
            rows.append(row)
    return rows


def attach_orphans(rows: Sequence[Sequence[str]]) -> list[list[str]]:
    """Fold wrapped continuation cells back into their owning row.

    DfG descriptions wrap across lines; a row carrying no account code and no
    serial number continues the previous coded row. Measured on 01 p41: three
    orphan amount rows, which must attach to 3451.00.090.05 for the appendix
    sum to reach 9,740,800.
    """
    out: list[list[str]] = []
    for r in rows:
        row = list(r)
        j = " ".join(row)
        if not out or not re.search(r"\d", j):
            out.append(row)
            continue
        has_code = bool(CODE_FULL.search(j) or CODE_SUB.search(j))
        if TOTAL_ROW.search(j):
            out.append(row)
            continue
        if not has_code:
            prev = out[-1]
            merged = [(pv or "") + " " + (c or "") if c else pv
                      for pv, c in zip(prev + [""] * 9, row + [""] * 9,
                                       strict=False)]
            out[-1] = merged[: max(len(prev), len(row))]
        else:
            out.append(row)
    return out


# ── Route B / C shared rendering + OCR ───────────────────────────────────────

def _render(doc: Any, pno: int, dpi: int):
    from PIL import Image

    pix = doc[pno].get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def ocr_page_image(img: Any, *, lang: str = "hin+eng", config: str = "--psm 6") -> str:
    """One OCR call. ``lang`` is always explicit — the whole point of the fix
    is that tesseract's default (eng) silently erases Devanagari."""
    import pytesseract

    return pytesseract.image_to_string(img, lang=lang, config=config) or ""


def render_ocr_page(doc: Any, pno: int, *, dpi: int = OCR_DPI,
                    lang: str = "hin+eng") -> str:
    """ROUTE B: render the page and OCR it bilingually."""
    return ocr_page_image(_render(doc, pno, dpi), lang=lang)


def _cluster(a: Sequence[int], gap: int = 8) -> list[int]:
    out: list[list[int]] = []
    for x in a:
        if not out or x - out[-1][-1] > gap:
            out.append([int(x)])
        else:
            out[-1].append(int(x))
    import numpy as np

    return [int(np.mean(g)) for g in out]


def grid_rows(doc: Any, pno: int, *, dpi: int = GRID_DPI,
              line_fraction: float = GRID_LINE_FRACTION
              ) -> tuple[list[list[str]], tuple[int, int], float]:
    """ROUTE C: morphological grid detection then per-cell OCR.

    Returns ``(rows, (n_rows, n_cols), mean_confidence)``. Numeric columns get
    English + a digit whitelist at 2× upscale; text/Hindi columns get
    ``hin+eng``. Column-role awareness is mandatory — a single flat OCR pass
    cannot associate values with labels.
    """
    import cv2
    import numpy as np
    from PIL import Image

    pix = doc[pno].get_pixmap(dpi=dpi)
    img_h, img_w, n = pix.height, pix.width, pix.n
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(img_h, img_w, n)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if n == 4 else cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, img_w // 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, img_h // 30)))
    horiz = cv2.dilate(cv2.erode(bw, hk), hk)
    vert = cv2.dilate(cv2.erode(bw, vk), vk)
    hc = np.where(horiz.sum(axis=1) > line_fraction * 255 * img_w)[0]
    vc = np.where(vert.sum(axis=0) > line_fraction * 255 * img_h)[0]
    ys, xs = _cluster(hc), _cluster(vc)
    ncol = max(0, len(xs) - 1)
    if ncol == 0 or len(ys) < 2:
        return [], (0, ncol), 0.0

    numeric_cols = {1, 2, 3, ncol - 1} if ncol >= 7 else set(range(ncol))
    rows: list[list[str]] = []
    confs: list[float] = []
    for r in range(len(ys) - 1):
        cells: list[str] = []
        for c in range(ncol):
            cc = img[max(0, ys[r] + 5):min(img_h, ys[r + 1] - 5),
                     max(0, xs[c] + 5):min(img_w, xs[c + 1] - 5)]
            if cc.size == 0 or cc.shape[0] < 14 or cc.shape[1] < 14:
                cells.append("")
                continue
            rgb = cv2.cvtColor(cc, cv2.COLOR_BGR2RGB)
            if c in numeric_cols:
                up = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                t = ocr_page_image(Image.fromarray(up), lang="eng",
                                   config="--psm 7 -c tessedit_char_whitelist=0123456789.,-")
                cells.append(t.strip().strip(".,-"))
            else:
                t = ocr_page_image(Image.fromarray(rgb), lang="hin+eng", config="--psm 6")
                cells.append(" ".join(t.split()))
        if any(cells):
            rows.append(cells)
            confs.append(_cell_confidence(cells, ncol))
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return rows, (len(ys) - 1, ncol), mean_conf


def _cell_confidence(cells: Sequence[str], ncol: int) -> float:
    """Cheap deterministic stand-in for tesseract TSV confidence.

    A cell "confidently read" is one that is non-empty and, for the columns we
    expect to be numeric or coded, actually parses as such. Used only to rank
    pages DEGRADED vs PASS — never to alter the extracted text.
    """
    if ncol <= 0:
        return 0.0
    score = 0.0
    for i, c in enumerate(cells):
        s = (c or "").strip()
        if not s:
            continue
        if i in (1, 2, 3, ncol - 1):
            score += 1.0 if re.fullmatch(r"[\d.,\- ]+", s) else 0.25
        else:
            score += 1.0
    return score / len(cells)


# ── arithmetic validation ────────────────────────────────────────────────────

def cn(s: str | None) -> int | None:
    """Cell string → int, or None. Indian grouping commas are stripped."""
    s = (s or "").replace(",", "").strip(".-| ")
    return int(s) if s.isdigit() else None


def appendix_amounts(rows: Sequence[Sequence[str]]) -> list[tuple[str, int]]:
    """(code, amount) pairs from an Appendix-XXVII-style table.

    Orphan/wrapped amounts attach to the active code block. Verified on 01 p41:
    12 pairs, Σ = 9,740,800 = the reported कुल/Total.
    """
    pairs: list[tuple[str, int]] = []
    for r in rows:
        j = " ".join(x or "" for x in r)
        if TOTAL_ROW.search(j):
            continue
        codes = [m.group(1) for m in CODE_FULL.finditer(j)]
        amounts = [int(a.replace(",", ""))
                   for a in AMT.findall(" ".join(x or "" for x in list(r)[3:]))]
        if not amounts:
            continue
        if codes:
            for k, a in enumerate(amounts):
                pairs.append((codes[min(k, len(codes) - 1)], a))
        elif pairs:
            for a in amounts:
                pairs.append((pairs[-1][0], a))
    return pairs


def reported_total(rows: Sequence[Sequence[str]]) -> int | None:
    """The largest number on the कुल/Total row — the reported grand figure."""
    for r in rows:
        if TOTAL_ROW.search(" ".join(x or "" for x in r)):
            nums = [cn(c) for c in r if cn(c)]
            if nums:
                return max(nums)
    return None


def _looks_like_code(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(r"[0-9.| '.]{5,14}", s)
                and re.search(r"\d{2}[.| ']\d{2}", s))


def _code_in_row(row: Sequence[str]) -> str | None:
    """Account code from the CODE column (index 4 of the canonical 7).

    Falls back to a position-agnostic scan when the physical row is not 7 wide
    — Route A lattices vary (6 columns on 01 p41). A bare 6-digit amount never
    matches ``_looks_like_code``, so totals rows carrying a figure in the
    Actuals column are still not mistaken for coded component rows.
    """
    if len(row) >= 5 and _looks_like_code(row[4]):
        return (row[4] or "").strip()
    for x in row:
        if _looks_like_code(x):
            return (x or "").strip()
    return None


def subhead_sum_checks(rows: Sequence[Sequence[str]], *, page: int = 0,
                       carry: list[int] | None = None
                       ) -> tuple[list[ArithmeticCheck], list[int]]:
    """Detailed-head blocks: component rows accumulate until a कुल/Total row.

    ``carry`` holds components from earlier pages, because a block routinely
    opens on one page and closes on the next. Measured on 04: Total (07) on
    p12 reports 730,000 but only 10 of its components (Σ 644,500) are on p12 —
    the block opens on p11. Per-page validation therefore reports a false
    failure, which is exactly why accumulation must cross page boundaries.

    Column-position awareness matters: on totals rows the Actuals column itself
    carries a 6-digit value, so the code must be read from the code column.
    """
    checks: list[ArithmeticCheck] = []
    comps: list[int] = list(carry or [])
    opened_earlier = bool(carry)
    for row in rows:
        j = " ".join(x or "" for x in row)
        code = _code_in_row(row)
        last = cn(row[-1]) if row else None
        if code and last is not None:
            comps.append(last)
            continue
        # Only a total row that actually carries a reported figure closes a
        # block. Scanned pages open with banner rows reading just "कुल" and
        # "Total" (04 p12 rows 6-7); resetting on those threw away the carried
        # components before the real total row was ever reached, which is what
        # made the cross-page Σ silently stop reconciling.
        closes_block = (TOTAL_ROW.search(j) and not code
                        and last is not None and last >= 1000 and comps)
        if closes_block:
                checks.append(ArithmeticCheck(
                    page=page, label="subhead_total", components=len(comps),
                    computed=sum(comps), reported=last, spans_pages=opened_earlier))
                comps = []
                opened_earlier = False
    return checks, comps


def page_code_seq(text: str) -> list[str]:
    """Account-code sequence on a page — used to prove cross-page continuity."""
    return [m.group(0).replace(" ", "") for m in CODE_SUB.finditer(text or "")]


# ── page classification / routing ────────────────────────────────────────────

def page_class(page: Any) -> dict[str, Any]:
    """Deterministic per-page routing decision (spec §"Routing key").

    Returns the route plus the measurements that produced it, so every routing
    choice in a report is auditable rather than inferred after the fact.
    """
    words = page.get_text("words")
    drawings = page.get_drawings()
    text = page.get_text()
    has_text = bool(words)
    info: dict[str, Any] = {
        "has_text_layer": has_text,
        "n_words": len(words),
        "n_drawings": len(drawings),
        "dev_chars": count_devanagari(text),
        "mojibake": round(mojibake_score(text[:800]), 4),
    }
    if has_text:
        if info["mojibake"] >= MOJIBAKE_THRESHOLD:
            info["route"] = "B"
            info["why"] = f"mojibake {info['mojibake']} >= {MOJIBAKE_THRESHOLD}"
        else:
            vc, _ = lattice_cuts(page)
            info["vcuts"] = len(vc)
            if len(vc) >= 4:
                info["route"] = "A"
                info["why"] = f"{len(vc)} vertical rules -> lattice"
            else:
                info["route"] = "A-line"
                info["why"] = f"{len(vc)} vertical rules -> borderless line rows"
    elif len(drawings) > 300:
        info["route"] = "B"
        info["why"] = f"{len(drawings)} vector items, no text layer -> outline page"
    else:
        info["route"] = "C"
        info["why"] = "no text layer, no vector art -> scan"
    return info


def effective_dpi(doc: Any, pno: int = 0) -> float | None:
    """Effective scan DPI of a page's embedded image, or None if not a scan."""
    images = doc[pno].get_images(full=True)
    if not images:
        return None
    import fitz

    pix = fitz.Pixmap(doc, images[0][0])
    rect = doc[pno].rect
    if not rect.width:
        return None
    return pix.width / (rect.width / 72.0)



def _valid_fy(token: str) -> str | None:
    """Normalise an FY token, rejecting readings that are not real FYs.

    A Demand-for-Grants FY is always ``YYYY-YY`` with ``YY == (YYYY + 1) % 100``.
    Measured on 05 (150-DPI scan) OCR returns '2024-22', '2020-24' and '2027-22'
    for the same printed string — all invalid. Rejecting them is how the
    extractor avoids inventing a fiscal year it could not read (spec req. F).
    """
    m = re.fullmatch(r"(20\d{2})\s*[-\u2013\u2014/]\s*(\d{2,4})",
                     (token or "").strip())
    if not m:
        return None
    start, end = int(m.group(1)), m.group(2)[-2:]
    return f"{start}-{end}" if f"{(start + 1) % 100:02d}" == end else None


def probe_front_matter(doc: Any, *, pages: Sequence[int] = (0, 1, 2, 3),
                       lang: str = "hin+eng") -> dict[str, Any]:
    """Recover fiscal_year / demand_no by MAJORITY VOTE over the front matter.

    A single regex hit is not trustworthy on a scan. Measured on 05 the fiscal
    year reads three different invalid ways, while the demand number is only
    reliable because all 8 sampled pages agree on it. Voting is deterministic
    and refuses to guess: an unsettled value stays None so the caller can
    supply the authoritative CMS value instead of the extractor inventing one.
    """
    dem: Counter[str] = Counter()
    fy: Counter[str] = Counter()
    for i in pages:
        if i >= doc.page_count:
            break
        text = doc[i].get_text()
        if not text.strip():
            text = render_ocr_page(doc, i, lang=lang)
        for m in DEMAND_NO_RE.finditer(text):
            dem[m.group(1)] += 1
        for m in FISCAL_RE.finditer(text):
            v = _valid_fy(m.group(0))
            if v:
                fy[v] += 1
    demand_no = fiscal_year = None
    if dem:
        top, n = dem.most_common(1)[0]
        if n * 2 > sum(dem.values()):        # strict majority, not a plurality
            demand_no = int(top)
    if fy:
        top, n = fy.most_common(1)[0]
        if n * 2 > sum(fy.values()):
            fiscal_year = top
    return {"fiscal_year": fiscal_year, "demand_no": demand_no,
            "demand_votes": dict(dem), "fy_votes": dict(fy)}


# ── row assembly ─────────────────────────────────────────────────────────────

def _classify_role(joined: str, code: str | None) -> str:
    if PAGE_HEADER.search(joined):
        return "header"
    if CONTINUATION_BAND.search(joined):
        return "band"
    if TOTAL_ROW.search(joined):
        return "total"
    if VOTED_ROW.search(joined):
        return "voted"
    if code:
        return "component"
    return "component"


def _map_to_canonical(phys: Sequence[str]) -> list[str]:
    """Project a physical row that is not 7 wide onto the canonical 7 slots.

    Positional mapping is only valid when the lattice really has 7 columns;
    otherwise the amount that belongs in ``Actuals`` lands in ``code`` and is
    silently discarded (measured: 01 p41 lost its reported total 9,740,800).
    """
    nums = [x for x in phys if x and re.fullmatch(r"[\d.,\- ]+", x)]
    code_i = next((i for i, x in enumerate(phys) if _looks_like_code(x)), -1)
    hi_i = next((i for i, x in enumerate(phys)
                 if i != code_i and x and count_devanagari(x) >= 2), -1)
    en_i = next((i for i, x in enumerate(phys)
                 if i not in (code_i, hi_i) and x and count_devanagari(x) == 0
                 and re.search(r"[A-Za-z]{3}", x)), -1)
    n = (nums + ["", "", "", ""])[:4]
    return [n[0], n[1], n[2],
            phys[hi_i] if hi_i >= 0 else "",
            phys[code_i] if code_i >= 0 else "",
            phys[en_i] if en_i >= 0 else "", n[3]]


def _row_from_cells(cells: Sequence[str], *, page: int, seq: int,
                    conf: float = 1.0) -> DfGRow:
    """Map a physical cell list onto the canonical 7-column DfG structure."""
    phys = [x or "" for x in cells]
    c = list(phys) + [""] * max(0, 7 - len(phys)) if len(phys) == 7 \
        else _map_to_canonical(phys)
    code = _code_in_row(c)
    joined = " ".join(x for x in c if x)
    return DfGRow(
        page=page, seq=seq,
        actuals=(c[0] or None), budget=(c[1] or None), revised=(c[2] or None),
        hi=(c[3] or None), code=code, en=(c[5] or None), budget_next=(c[6] or None),
        role=_classify_role(joined, code), conf=conf, raw=phys,
    )


# ── folded corpus text (structure-preserving; bypasses _clean) ───────────────

def render_folded(doc: DfGDocument) -> str:
    """Corpus ``answer_text``: page-tagged, pipe-separated, one row per line.

    Deliberately NOT flattened. The generic ``_clean()`` collapses every newline
    and multi-space column gap, which is what destroyed the current DfG corpus
    rows (6 newlines each = the ``to_text()`` separators). Callers must bypass
    it for DfG records.
    """
    out: list[str] = []
    for p in doc.pages:
        for row in p.rows:
            cells = [x or "" for x in row.as_cells()]
            line = f"[p{p.page}] "
            if row.role != "component":
                line += f"role={row.role} | "
            line += " | ".join(cells)
            if row.continued_from_prev_page:
                line += " | (cont.)"
            out.append(line.rstrip(" |"))
        for t in p.prose:
            out.append(f"[p{p.page}] {t}")
        if p.verdict != "PASS":
            out.append(f"[p{p.page}] quality={p.verdict}"
                       + (f" ({'; '.join(p.notes)})" if p.notes else ""))
    return "\n".join(out)


# ── verdicts ─────────────────────────────────────────────────────────────────

def _page_verdict(pr: PageResult, *, dpi: float | None) -> None:
    """Deterministic per-page gate. Never silently downgrades to PASS."""
    if dpi is not None and dpi < DPI_FAIL_THRESHOLD:
        pr.verdict = "FAIL"
        pr.notes.append(f"source scan ~{dpi:.0f} DPI < {DPI_FAIL_THRESHOLD}: "
                        f"digit fidelity is physically limited, not recoverable")
        return
    failed = [c for c in pr.checks if not c.ok]
    if failed:
        pr.verdict = "DEGRADED"
        for c in failed:
            pr.notes.append(
                f"arithmetic {c.label}: Σ{c.computed:,} != reported {c.reported:,}"
                f" ({c.components} components"
                f"{', block spans pages' if c.spans_pages else ''})"
            )
        return
    # The structural gates below only mean anything on a page that actually
    # holds a table. Applying them to cover/index pages marked every front page
    # of 04 DEGRADED for "code-cell validity 0%" when there is no table to have
    # codes in — a false failure that drowned the real ones.
    if not any(r.role == "component" for r in pr.rows):
        return
    if pr.code_validity is not None and pr.code_validity < CODE_VALIDITY_DEGRADED:
        pr.verdict = "DEGRADED"
        pr.notes.append(f"code-cell validity {pr.code_validity:.0%} "
                        f"< {CODE_VALIDITY_DEGRADED:.0%}")
        return
    if pr.route == "C" and pr.dev_chars == 0:
        pr.verdict = "DEGRADED"
        pr.notes.append("scan OCR recovered no Devanagari — Hindi columns suspect")


def document_verdict(doc: DfGDocument) -> str:
    """Worst page verdict wins; FAIL is never absorbed."""
    order = {"PASS": 0, "DEGRADED": 1, "FAIL": 2}
    worst = "PASS"
    for p in doc.pages:
        if order.get(p.verdict, 0) > order.get(worst, 0):
            worst = p.verdict
    return worst


# ── top-level entry point ────────────────────────────────────────────────────

def extract_dfg(
    path: str | Path,
    *,
    source_url: str | None = None,
    max_pages: int | None = None,
    routes: Iterable[str] | None = None,
    fiscal_year: str | None = None,
    demand_no: int | None = None,
) -> DfGDocument:
    """Extract one DfG PDF into structured rows, gates and provenance.

    ``routes`` restricts which routes may run (useful for tests and for
    cost control); by default all three are enabled.
    """
    import fitz

    allowed = set(routes) if routes else {"A", "A-line", "B", "C"}
    p = Path(path)
    body = p.read_bytes()
    doc = fitz.open(str(p))
    out = DfGDocument(path=str(p))
    carry: list[int] = []
    first_text = ""
    probe_text = ""

    try:
        n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(n):
            info = page_class(doc[i])
            route = info["route"] if info["route"] in allowed else "A-line"
            pr = PageResult(page=i + 1, route=route)
            dpi = effective_dpi(doc, i) if route == "C" else None

            if route in ("A", "A-line"):
                words = page_words(doc[i])
                vcuts, _ = lattice_cuts(doc[i])
                raw = rows_from_words(words, vcuts if route == "A" else [])
                raw = attach_orphans(raw)
                raw = [r for r in raw if any((x or "").strip() for x in r)]
                pr.rows = [_row_from_cells(r, page=i + 1, seq=k)
                           for k, r in enumerate(raw)]
                pr.dev_chars = info["dev_chars"]
            elif route == "B":
                text = render_ocr_page(doc, i)
                pr.dev_chars = count_devanagari(text)
                pr.prose = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pr.rows = [_row_from_cells([ln], page=i + 1, seq=k)
                           for k, ln in enumerate(pr.prose)]
            else:  # route C
                grid, shape, conf = grid_rows(doc, i)
                pr.grid_shape = shape
                pr.rows = [_row_from_cells(r, page=i + 1, seq=k, conf=conf)
                           for k, r in enumerate(grid)]
                pr.dev_chars = sum(count_devanagari(r.hi or "") for r in pr.rows)
                coded = [r for r in pr.rows if r.role == "component"]
                if coded:
                    pr.code_validity = sum(
                        1 for r in coded
                        if r.code and re.search(r"\d{2}[.]\d{2}[.]\d{2}", r.code)
                    ) / len(coded)

            raw_cells = [list(r.raw) for r in pr.rows]
            # Appendix-XXVII-style gate: a coded component list closing on a
            # कुल/Total row. This is the check that proves 01 p41 (12 components,
            # Σ 9,740,800) and is what catches a single mis-read digit.
            if route in ("A", "A-line"):
                pairs = appendix_amounts(raw_cells)
                rep = reported_total(raw_cells)
                if rep is not None and len(pairs) >= 3:
                    pr.checks.append(ArithmeticCheck(
                        page=i + 1, label="appendix_total", components=len(pairs),
                        computed=sum(a for _, a in pairs), reported=rep))
            # extend, never assign: the appendix gate above has already
            # appended to pr.checks and a tuple-assign here would discard it.
            _sub, carry = subhead_sum_checks(raw_cells, page=i + 1, carry=carry)
            pr.checks.extend(_sub)
            _page_verdict(pr, dpi=dpi)
            out.pages.append(pr)
            if i < 8:  # front matter carries the FY + Demand No.; page 1 often
                first_text += "\n" + doc[i].get_text()[:2000]   # does not
            if i == 0 and not info["has_text_layer"]:
                # Scanned front matter: one OCR probe is the only way to recover
                # the fiscal year and demand number (the staged filename is
                # NN-<wp_id>-<lang>, so it carries neither).
                probe_text = render_ocr_page(doc, 0)
        # Front-matter provenance. A single regex hit is unreliable on a scan,
        # so the values are majority-voted across the first pages. Anything the
        # vote cannot settle stays None unless the caller supplies the
        # authoritative CMS value — the extractor never invents a digit it
        # could not read (req. F). Must run before the doc is closed.
        voted = (probe_front_matter(doc)
                 if (fiscal_year is None or demand_no is None) else {})
    finally:
        doc.close()

    fy_val = fiscal_year or voted.get("fiscal_year")
    dm_val = demand_no if demand_no is not None else voted.get("demand_no")
    out.verdict = document_verdict(out)
    out.provenance = DfGProvenance(
        fiscal_year=fy_val,
        demand_no=dm_val,
        source_url=source_url,
        page_count=len(out.pages),
        language_class=classify_language(
            stem=p.stem, text_layer=first_text or probe_text)["lang_class"],
        extraction_route={pr.page: pr.route for pr in out.pages},
        quality_verdict=out.verdict,
        sha256=hashlib.sha256(body).hexdigest(),
        pages=[pr.page for pr in out.pages],
    )
    return out
