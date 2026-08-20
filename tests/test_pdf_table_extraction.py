"""Regression tests for table-aware PDF extraction (issue: malformed
annexure tables in source-document ingestion of 17-7-2936.pdf).

Forensic chain (see investigation_table_extraction/REPORT.md):
    official PDF -> RealArchiveScraper._extract_text_from_document
    -> _split_question_answer -> QARecord.answer_text -> records.jsonl

The official sansad.in PDFs render annexures as BORDERLESS tables whose
wrapped cells land on their own text line ~1 pitch below the row baseline.
pypdf's flat ``extract_text()`` (and equally PyMuPDF plain text) splices
those fragments between rows, e.g.::

    PURULIA (State Govt
    24 WEST BENGAL PURULIA Guest House)

``src/data/pdf_table_extract.py`` reconstructs rows geometrically. Wrapped-
cell ownership is decided by (1) a deterministic parenthesis-balance
override — parentheses balance inside a natural-language cell, and a wrap
partition of the cell text carries that signal — and (2) a same-column
bbox-gap heuristic with tie→UP default for signal-free cases. The override
is required because pure geometry provably cannot separate all cases on
this fixture (cross-row gap is a constant 2.20pt while within-cell
leadings are {1.96, 2.20, 2.44}pt — see investigation_table_extraction).

These tests pin (a) that the fixture genuinely exhibits the corruption
under the legacy pypdf path, (b) that the fixed extractor produces correct,
completely-ordered rows, (c) on synthetic ground-truth PDFs that the
paren rule fixes ownership in BOTH directions exactly where geometry
alone fails, and (d) that the scraper degrades gracefully to the legacy
path when PyMuPDF is unavailable.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from src.data import pdf_table_extract as pte
from src.data.pdf_table_extract import extract_pdf_text
from src.data.scraper import RealArchiveScraper

pytestmark = pytest.mark.skipif(
    pytest.importorskip("pypdf") is None, reason="pypdf required"
)

fitz = pytest.importorskip("fitz", reason="PyMuPDF required for table tests")

FIXTURE = Path(__file__).parent / "fixtures" / "17-7-2936.pdf"


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}"
    return FIXTURE.read_bytes()


def _pypdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "".join((p.extract_text() or "") for p in reader.pages)


# ---------------------------------------------------------------------------
# 1. Baseline: legacy pypdf path corrupts the tables on this fixture
# ---------------------------------------------------------------------------


def test_baseline_pypdf_exhibits_the_corruption(pdf_bytes):
    """Pin the bug: pypdf flat extraction interleaves wrapped cell fragments."""
    text = _pypdf_text(pdf_bytes)

    # Row 3 / row 24 wrapped-cell fragments spliced between rows
    assert "3 WEST BENGAL BHIRBHUM SURI\nSANTINIKETAN-" in text
    assert "PURULIA (State Govt\n24 WEST BENGAL PURULIA Guest House)" in text
    assert "SAGAR ISLAND\nKHARAGPUR( lIT" in text

    # ...and therefore the correct full rows are NOT recoverable downstream
    assert "3 WEST BENGAL BHIRBHUM SURI SANTINIKETAN-" not in text
    assert "24 WEST BENGAL PURULIA PURULIA (State Govt Guest House)" not in text


# ---------------------------------------------------------------------------
# 2. Fixed extractor: complete, correctly-ordered rows in both annexures
# ---------------------------------------------------------------------------


def test_fixed_annexure1_all_33_rows_once_in_order(pdf_bytes):
    lines = extract_pdf_text(pdf_bytes).splitlines()
    serials = [
        int(m.group(1))
        for l in lines
        if (m := re.match(r"^(\d+) WEST BENGAL\b", l))
    ]
    assert serials == list(range(1, 34))  # 33 rows, each exactly once, ordered


def test_fixed_annexure1_wrapped_rows_exact(pdf_bytes):
    lines = extract_pdf_text(pdf_bytes).splitlines()
    # rows that were corrupted under pypdf — exact full-line pins
    assert "3 WEST BENGAL BHIRBHUM SURI SANTINIKETAN-" in lines
    assert "23 WEST BENGAL NORTH TWENTY FOUR PARGANAS BASIRHAT" in lines
    assert "24 WEST BENGAL PURULIA PURULIA (State Govt Guest House)" in lines


def test_fixed_annexure1_row33_fully_reconstructed(pdf_bytes):
    """The former Kharagpur residual: paren-balance override places the
    wrapped fragment on row 33 and leaves row 32's clean row untouched."""
    lines = extract_pdf_text(pdf_bytes).splitlines()
    assert "32 WEST BENGAL SOUTH TWENTY FOUR PARGANAS SAGAR ISLAND" in lines
    assert "33 WEST BENGAL WEST MEDINIPUR KHARAGPUR( lIT CAMPUS)" in lines
    # the fragment must appear exactly once, on its owning row:
    assert sum("KHARAGPUR( lIT" in l for l in lines) == 1


# ---------------------------------------------------------------------------
# 2b. Synthetic ground truth: ownership rule correctness in both directions
# ---------------------------------------------------------------------------


def _build_synthetic_table_pdf() -> bytes:
    """8-row table with three wrapped station cells of KNOWN ownership.

    pitch=30pt, fontsize 9 (line bbox ~12pt => strictly positive bbox gaps).
    Trap geometry mirrors 17-7-2936:
      row 2: top-anchored paren cell — orphan "Dept)" sits closer to row 3
             (gap 3.49up/2.49down => bbox-gap rule alone picks DOWN, wrong)
      row 4: hyphen fragment — orphan "DELTA-" at an exact 2.99/2.99 tie
      row 7: bottom-anchored paren cell — orphan "LAMBDA (Govt" sits closer
             to row 6 (2.49/3.49 => bbox-gap alone picks UP, wrong)
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    def row(n: int, station: str) -> None:
        y = 100 + 30 * n
        for x, t in ((82, str(n)), (115, "ST"), (213, f"D{n}"), (417, station)):
            page.insert_text((x, y), t, fontsize=9, fontname="Times-Roman")

    row(1, "ONE")
    row(2, "GAMMA (State")
    page.insert_text((417, 100 + 30 * 2 + 15.5), "Dept)", fontsize=9, fontname="Times-Roman")
    row(3, "PHASE-II")
    row(4, "OMEGA")
    page.insert_text((417, 100 + 30 * 4 + 15.0), "DELTA-", fontsize=9, fontname="Times-Roman")
    row(5, "EPSILON KVK")
    row(6, "SIX")
    page.insert_text((417, 100 + 30 * 7 - 15.5), "LAMBDA (Govt", fontsize=9, fontname="Times-Roman")
    row(7, "House)")
    row(8, "EIGHT")
    try:
        return doc.tobytes()
    finally:
        doc.close()


@pytest.fixture(scope="module")
def synthetic_rows():
    text = extract_pdf_text(_build_synthetic_table_pdf())
    return [l for l in text.splitlines() if l and l[0].isdigit()]


def test_synthetic_every_row_exact(synthetic_rows):
    assert synthetic_rows == [
        "1 ST D1 ONE",
        "2 ST D2 GAMMA (State Dept)",
        "3 ST D3 PHASE-II",
        "4 ST D4 OMEGA DELTA-",
        "5 ST D5 EPSILON KVK",
        "6 ST D6 SIX",
        "7 ST D7 LAMBDA (Govt House)",
        "8 ST D8 EIGHT",
    ]


def test_synthetic_paren_override_beats_gap_trap_both_directions(synthetic_rows):
    # bottom-anchored (Kharagpur class): gap says UP(wrong) -> paren says DOWN
    assert "7 ST D7 LAMBDA (Govt House)" in synthetic_rows
    assert not any(l.startswith("6 ST D6 SIX LAMBDA") for l in synthetic_rows)
    # top-anchored mirror: gap says DOWN(wrong) -> paren mirror says UP
    assert "2 ST D2 GAMMA (State Dept)" in synthetic_rows
    assert not any(l.startswith("3 ST D3") and "Dept)" in l for l in synthetic_rows)


def test_synthetic_no_signal_tie_defaults_up(synthetic_rows):
    """Trailing-hyphen fragment with no paren signal + exact gap tie -> UP.
    Guards the row-3 (SANTINIKETAN-) class: a trailing '-' must NOT pull the
    next row's station up."""
    assert "4 ST D4 OMEGA DELTA-" in synthetic_rows
    assert "5 ST D5 EPSILON KVK" in synthetic_rows
    assert not any("DELTA- EPSILON" in l for l in synthetic_rows)


def test_fixed_annexure2_all_35_rows_once_in_order(pdf_bytes):
    lines = extract_pdf_text(pdf_bytes).splitlines()
    start = lines.index("Annexure-II")
    serials = [
        int(m.group(1))
        for l in lines[start:]
        if (m := re.match(r"^(\d+)\s", l))
    ]
    assert serials == list(range(1, 36))  # 35 rows, each exactly once, ordered
    assert "1 BANKURA KADAMDEULI" in lines
    assert "35 KOLKATA OHR 10" in lines
    assert any(l.strip() == "*****" for l in lines)  # document tail preserved


def test_fixed_annexure_headers_and_order(pdf_bytes):
    lines = extract_pdf_text(pdf_bytes).splitlines()
    i1 = lines.index("Annexure-I")
    i2 = lines.index("Annexure-II")
    first_row = lines.index("1 WEST BENGAL BANKURA BANKURA")
    assert i1 < first_row < i2


# ---------------------------------------------------------------------------
# 3. Non-table content parity: extraction must not damage unrelated text
# ---------------------------------------------------------------------------


def test_fixed_nontable_content_preserved(pdf_bytes):
    text = extract_pdf_text(pdf_bytes)
    assert "GOVERNMENT OF INDIA" in text
    assert "MINISTRY OF EARTH SCIENCES" in text
    assert re.search(r"\(a\) whether any modem automatic weather stations", text)
    assert re.search(r"\(b\)", text) and re.search(r"\(c\)", text)
    # baseline-fused header (pypdf produces it as one line too)
    assert "SNO. STATE DISTRICT STATION" in text


def test_fixed_vs_pypdf_similarity_is_surgical(pdf_bytes):
    """Outside the wrapped-row fixes, output stays pypdf-like (>95%)."""
    import difflib

    old = _pypdf_text(pdf_bytes)
    new = extract_pdf_text(pdf_bytes)
    assert difflib.SequenceMatcher(None, old, new).ratio() > 0.95


# ---------------------------------------------------------------------------
# 4. Page internals: flat pages take the untouched flat path
# ---------------------------------------------------------------------------


def test_page1_has_no_serial_run_and_uses_flat_render(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        lines = pte._page_lines(doc[0])
        assert pte._serial_runs(lines) == []  # page 1 carries no table serials
        assert pte._page_text(doc[0]) == pte._render_merged(lines)
        # baseline-fused visual line proves the merge, not geometry damage
        assert "GOVERNMENT OF INDIA" in pte._page_text(doc[0]).splitlines()
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 5. Graceful degradation: no PyMuPDF -> identical legacy behavior
# ---------------------------------------------------------------------------


def test_extract_raises_importerror_without_fitz(pdf_bytes, monkeypatch):
    monkeypatch.setattr(pte, "_fitz_enabled", False)
    with pytest.raises(ImportError):
        extract_pdf_text(pdf_bytes)


def test_scraper_falls_back_to_pypdf_without_fitz(pdf_bytes, monkeypatch):
    monkeypatch.setattr(pte, "_fitz_enabled", False)
    scraper = RealArchiveScraper.__new__(RealArchiveScraper)
    result, reason = scraper._extract_text_from_document(pdf_bytes, "pdf")
    assert reason is None and result is not None
    question, answer = result
    expected_q, expected_a = RealArchiveScraper._split_question_answer(
        _pypdf_text(pdf_bytes)
    )
    assert question == expected_q and answer == expected_a  # byte-identical legacy

    # ...and the legacy result still exhibits the (now documented) corruption
    assert "24 WEST BENGAL PURULIA Guest House)" in answer
    assert "24 WEST BENGAL PURULIA PURULIA (State Govt Guest House)" not in answer


# ---------------------------------------------------------------------------
# 6. Scraper integration & preserved failure semantics
# ---------------------------------------------------------------------------


def test_scraper_pdf_branch_returns_clean_rows(pdf_bytes):
    scraper = RealArchiveScraper.__new__(RealArchiveScraper)
    result, reason = scraper._extract_text_from_document(pdf_bytes, "pdf")
    assert reason is None and result is not None
    question, answer = result

    # ANSWER boundary split (not the 1/3-ratio fallback)
    assert question.rstrip().endswith("thereof?")
    assert "MINISTRY OF EARTH SCIENCES" in question
    assert "DR. JITENDRA SINGH" in answer

    # the retrieved-source rows that prompted this fix are now correct
    assert "24 WEST BENGAL PURULIA PURULIA (State Govt Guest House)" in answer
    assert "3 WEST BENGAL BHIRBHUM SURI SANTINIKETAN-" in answer
    assert "23 WEST BENGAL NORTH TWENTY FOUR PARGANAS BASIRHAT" in answer
    assert "35 KOLKATA OHR 10" in answer


def test_scraper_parser_failure_semantics_preserved():
    scraper = RealArchiveScraper.__new__(RealArchiveScraper)
    result, reason = scraper._extract_text_from_document(b"not a pdf at all", "pdf")
    assert result is None and reason == "parser_failure"


def test_scraper_scanned_semantics_preserved():
    """A valid PDF with no text still lands on the 'scanned' reason."""
    doc = fitz.open()
    try:
        doc.new_page()
        data = doc.tobytes()
    finally:
        doc.close()
    scraper = RealArchiveScraper.__new__(RealArchiveScraper)
    result, reason = scraper._extract_text_from_document(data, "pdf")
    assert result is None and reason == "scanned"


def test_docx_branch_untouched():
    """DOCX documents keep flowing through the original docx extractor."""
    scraper = RealArchiveScraper.__new__(RealArchiveScraper)
    result, reason = scraper._extract_text_from_document(b"PK garbage", "docx")
    # unparseable bytes -> parser_failure via _extract_text_from_docx, never PDF code
    assert result is None and reason == "parser_failure"
