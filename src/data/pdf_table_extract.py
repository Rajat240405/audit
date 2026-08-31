"""Table-aware PDF text extraction for Lok Sabha annexure tables.

Forensic context (investigation_table_extraction/REPORT.md): the official
sansad.in Q&A PDFs render annexures as BORDERLESS tables. Long cell values
wrap onto a second line that is vertically offset by ~one line pitch, so
FLAT text extraction (pypdf ``extract_text()``, and equally PyMuPDF plain
text) splices wrapped fragments into the wrong row — e.g. 17-7-2936 came out
as::

    23 WEST BENGAL NORTH TWENTY FOUR PARGANAS BASIRHAT
    PURULIA (State Govt                  <- belongs to row 24
    24 WEST BENGAL PURULIA Guest House)

This module reconstructs such tables geometrically with PyMuPDF:

  1. detect a serial-number column (>=4 consecutively numbered digit lines
     sharing an x-band with a quasi-uniform y pitch),
  2. partition content into columns by x0 clustering,
  3. group row content on the serial's own baseline,
  4. resolve wrapped-cell fragments (orphan lines) into the row they
     belong to. Two tiers, checked per orphan:
       a) PAREN-BALANCE OVERRIDE (deterministic): parentheses balance
          inside a natural-language cell, so an orphan ending with
          unmatched "(" above a row whose anchored station carries the
          matching unmatched ")" is conclusively that row's continuation
          (and the top-anchored mirror). Pure geometry provably cannot
          separate all cases — on the reference PDF the cross-row gap is
          a constant 2.20pt while within-cell leadings vary 1.96–2.44pt,
          so the geometric rule alone mis-places "KHARAGPUR( lIT".
       b) bbox-gap fallback: the tighter same-column side is the same
          cell's continuation (leading inside one cell is usually
          tighter than the gap across a row boundary); ties attach UP.

Everything else (no serial column, detection failure) falls through to
the same PyMuPDF plain-text path, so non-table ingestion output is
unchanged. No layout models, no OCR, no network.

License note: PyMuPDF is AGPL-3.0/commercial dual-licensed (already in use
under src/scripts/).
"""

from __future__ import annotations

from statistics import median

# indirection so tests can simulate "PyMuPDF missing"
_fitz_enabled = True


def _import_fitz():
    if not _fitz_enabled:
        raise ImportError("PyMuPDF disabled")
    import fitz  # lazy by design
    return fitz


_MIN_SERIALS = 4
_X_TOL = 6.0            # column x clustering tolerance (pt)
_ROW_ALIGN = 0.28       # content counts as a row's baseline within ±0.28·pitch
_MIN_PITCH = 8.0        # sanity guards against false-positive serial columns
_MAX_PITCH = 32.0
_LINE_MERGE_TOL = 2.0   # pt — same-visual-line baseline wobble:
                        # body lines are ≥ one line-height (≈11–14 pt) apart,
                        # so 2 pt only ever fuses spans of ONE printed line


def extract_pdf_text(data: bytes) -> str | None:
    """Whole-document text with borderless serial tables reconstructed.

    Returns None only when the document yields no text at all (caller treats
    that as "scanned"); raises ImportError when PyMuPDF is unavailable.
    """
    fitz = _import_fitz()
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        out = []
        for page in doc:
            out.append(_page_text(page))
        return "".join(out)
    finally:
        doc.close()


def extract_pdf_text_with_fallback(data: bytes) -> str:
    """THE shared "PDF bytes \u2192 text" entry for the folder/ingest converters
    (audit IW-7): table-aware PyMuPDF extraction.

    Returns "" when the document yields no text (scanned/image-only) so the
    caller's existing OCR handling applies exactly as before.

    Raises ``ImportError`` if PyMuPDF is unavailable (install PyMuPDF).
    Raises ``Exception`` for corrupt or unopenable documents.
    """
    text = extract_pdf_text(data)
    return text if text and text.strip() else ""


def _page_text(page) -> str:
    lines = _page_lines(page)
    if not lines:
        return ""
    runs = _serial_runs(lines)
    if not runs:
        # flat page: baseline-merged raw text (same visual-line granularity
        # as pypdf, so non-table ingestion output is unchanged)
        return _render_merged(lines)
    rows = _reconstruct_rows(lines, runs[0])
    if rows is None:
        return _render_merged(lines)

    lo, hi = runs[0]["lo"], runs[0]["hi"]
    head = _render_merged([l for l in lines if l["yc"] < lo]).splitlines()
    tail = _render_merged([l for l in lines if l["yc"] > hi]).splitlines()
    return "\n".join(head + rows + tail) + "\n"


def _render_merged(lines: list[dict]) -> str:
    """Render raw text lines with spans of one visual (same-baseline) line
    fused with a single space, preserving y,x reading order."""
    merged = _merge_visual_lines(lines)
    return "\n".join(l["text"] for l in merged) + ("\n" if merged else "")


def _merge_visual_lines(lines: list[dict]) -> list[dict]:
    # band first by baseline, then fuse each band left→right (two-pass so a
    # slightly-raised cell can't scramble the order of a shared baseline)
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
            text = " ".join(
                s["text"].strip() for s in ln["spans"] if s["text"].strip()
            )
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1,
                        "yc": (y0 + y1) / 2, "text": text})
    out.sort(key=lambda l: (l["yc"], l["x0"]))
    return out


def _serial_runs(lines: list[dict]) -> list[dict]:
    """Longest run of consecutive integers in one x-band, uniform pitch."""
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

    # column partition from content x0 clusters
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
        if id(l) in serial_ids:  # serial lines anchor to their own row
            anchors[serial_row[id(l)]].append(l)
            seen.add(id(l))
            continue
        for k, yc in enumerate(ys):
            if abs(l["yc"] - yc) <= tol:
                anchors[k].append(l)
                seen.add(id(l))
                break
    orphans = [l for l in rows_in if id(l) not in seen]

    # Wrapped-cell fragments have no serial on their baseline. Ownership:
    #
    # 1) PAREN-BALANCE OVERRIDE (deterministic, geometry-independent):
    #    parentheses balance INSIDE a natural-language cell, and a wrap
    #    partitions the cell's text. An orphan ending with unmatched "("
    #    whose below-row station anchor carries the matching unmatched ")"
    #    (depths sum to 0) is conclusively that row's continuation — and
    #    vice versa for the top-anchored mirror. Required because pure
    #    geometry provably cannot separate all cases: on 17-7-2936 the
    #    cross-row gap is a constant 2.20pt while observed within-cell
    #    leadings are {1.96, 2.20, 2.44}pt, so the bbox-gap heuristic
    #    mis-assigns row 33 ("KHARAGPUR( lIT" / "CAMPUS)").
    # 2) bbox-gap fallback: the tighter same-column side is the same cell's
    #    continuation (line leading inside one cell is usually tighter).
    #    Ties attach UP — a fragment whose own line reads complete stays
    #    with the row above (e.g. row 3's "SANTINIKETAN-").
    station_col = max(cols)  # rightmost column carries the long wrapped values
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
        if do > 0:  # e.g. "KHARAGPUR( lIT" above an anchored "CAMPUS)"
            db = _depth(_station_cell_text(k + 1))
            if db < 0 and do + db == 0:
                return k + 1
        elif do < 0:  # mirror: e.g. an orphan "Dept)" below "(State" anchor
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
