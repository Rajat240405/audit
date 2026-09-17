"""Stages 2, 4, 7 — table extraction for the experimental INCOIS V2 layer.

Ports the lattice geometry from ``v2proto/v2_extract.py`` (portrait + rotated)
and the continuation signatures from ``v2proto/continuation.py``. Thresholds and
rule order are preserved verbatim; they are what the spec's measured results
were produced with.

Three drawn-grid styles occur in this corpus and one detector handles all:

1. **long-rule lattice** — lines with ``|Δ| > 30pt`` (TR-06 style);
2. **thin-rule rects** — ``w < 2pt, h ≥ 4pt`` per-cell rules;
3. **short per-cell border segments** — where horizontal rules never reach
   ``min_len``, row bands are reconstructed from the y-endpoints of the vertical
   segments (spec §3.3; 3,003 ``l`` items on SCHEDULE-2 p223).

Determinism: pure geometry, no ML, no sampling. Byte-identical output across
repeated runs.

No OCR, no figures, no sidecar writing, no retrieval or generation imports.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.data.v2 import page_router
from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

# ── heuristics ported verbatim ────────────────────────────────────────────────

#: Row rules must be at least this long to count as a grid line.
MIN_RULE_LEN = 30.0
#: Short vertical stubs (per-cell column borders) fall in this length band.
STUB_MIN_LEN = 4.0
#: Vertical segments at least this long contribute their endpoints as row cuts.
SEGMENT_Y_MIN = 4.0
#: Row bands narrower than this are collapsed (preserves multi-line cells).
BAND_GAP = 8.0
#: Word-cluster banding gap used when there are too few horizontal cuts.
WORD_BAND_GAP = 9.0
#: Cut clustering tolerance.
CUT_TOL = 6.0
#: A page needs at least this many rects before rect edges are used as a grid.
CELL_RECT_MIN_COUNT = 8
#: Minimum endpoints before segment-endpoint row reconstruction kicks in.
SEGMENT_ENDPOINT_MIN = 6
#: Acceptance gate: a table needs at least this many rows.
MIN_ROWS = 4
#: Acceptance gate: at least this many column cuts.
MIN_VCUTS = 3
#: Fraction of rows that must look numeric (gate clause 1).
NUMERIC_ROW_FRACTION = 0.4
#: Gate clause 2, for Indian-format financial tables full of ``-`` placeholders:
#: a strong grid plus enough amount-like cells. Both parts are required — spec
#: §3.3 records that SCHEDULE-4A fails clause 1 and passes only clause 2.
STRONG_GRID_VCUTS = 6
STRONG_GRID_HCUTS = 8
STRONG_GRID_AMOUNTISH = 12

#: Table caption. Multi-level numbers (``10.4`` ≠ ``10.5``), en-dash separators
#: (``SCHEDULE – 4``) and letter suffixes (``4A``) are load-bearing: getting this
#: wrong either merges distinct tables or drops captions (spec §3.3, measured).
TABLE_CAPTION_RE = re.compile(
    r"((?:Table|SCHEDULE)\.?\s*[-.–—]?\s*\d+(?:\.\d+)*[A-Z]?)"
    r"\s*[:.\-–—]?\s*(.{0,150})",
    re.IGNORECASE,
)

_AMOUNT_RE = re.compile(r"[-+]?[\d,]+\.?\d*")
_CAPTION_NUMBER_RE = re.compile(
    r"(?:Table|SCHEDULE)\s*[-.–—]?\s*([0-9IVXA]+(?:\.[0-9]+)?[A-Z]?)", re.IGNORECASE
)
_CONTD_RE = re.compile(r"cont[dt]\.?|continued|concluded", re.IGNORECASE)


# ── geometry → grid cuts ──────────────────────────────────────────────────────


def _cluster(values: Sequence[float], tol: float = CUT_TOL) -> list[float]:
    """Collapse near-duplicate coordinates into single cut positions."""
    ordered = sorted(values)
    out: list[float] = []
    for value in ordered:
        if not out or value - out[-1] > tol:
            out.append(value)
    return out


def _accumulate_grid(
    lines: Sequence[tuple],
    rects: Sequence[Any],
    page_width: float,
    page_height: float,
    min_len: float = MIN_RULE_LEN,
) -> tuple[list[float], list[float]]:
    """Collect vertical/horizontal cut candidates from lines and rects.

    Shared by the portrait and rotated paths so the two cannot drift.
    """
    vertical: list[float] = []
    horizontal: list[float] = []
    segment_ys: list[float] = []
    stub_xs: list[float] = []

    for start, end in lines:
        dx = abs(start[0] - end[0])
        dy = abs(start[1] - end[1])
        if dx < 1 and dy > min_len:
            vertical.append((start[0] + end[0]) / 2)
            if dy >= SEGMENT_Y_MIN:
                segment_ys += [start[1], end[1]]
        elif dy < 1 and dx > min_len:
            horizontal.append((start[1] + end[1]) / 2)
        elif dx < 1 and STUB_MIN_LEN <= dy <= min_len:
            # short vertical stubs: per-cell column borders
            stub_xs += [start[0], end[0]]

    for rect in rects:
        if rect.width < 2 and rect.height >= SEGMENT_Y_MIN:
            vertical.append((rect.x0 + rect.x1) / 2)
            segment_ys += [rect.y0, rect.y1]
        elif rect.height < 2 and rect.width >= SEGMENT_Y_MIN:
            horizontal.append((rect.y0 + rect.y1) / 2)

    # cell-rect tables (filled rectangles): use rect edges as grid candidates
    if len(rects) >= CELL_RECT_MIN_COUNT:
        for rect in rects:
            if 2 < rect.width < page_width * 0.9 and 2 < rect.height < page_height * 0.5:
                vertical += [rect.x0, rect.x1]
                horizontal += [rect.y0, rect.y1]

    # segment-endpoint reconstruction (spec §3.3)
    if len(horizontal) < MIN_VCUTS and len(segment_ys) >= SEGMENT_ENDPOINT_MIN:
        horizontal += segment_ys
    if len(vertical) < MIN_VCUTS and len(stub_xs) >= SEGMENT_ENDPOINT_MIN:
        vertical += stub_xs

    return _cluster(vertical), _cluster(horizontal)


def lattice_cuts_items(
    words: Sequence[tuple],
    lines: Sequence[tuple],
    rects: Sequence[Any],
    size: tuple[float, float],
    min_len: float = MIN_RULE_LEN,
) -> tuple[list[float], list[float]]:
    """Grid cuts from **reading-frame** geometry (rotated pages)."""
    return _accumulate_grid(lines, rects, size[0], size[1], min_len)


def lattice_cuts(
    page: Any,
    config: V2Config | None = None,
    min_len: float = MIN_RULE_LEN,
    vector_ops: int | None = None,
) -> tuple[list[float], list[float], bool]:
    """Grid cuts in PDF space, with the spec §3.3 DoS guard.

    Returns ``(vcuts, hcuts, guard_skipped)``. When the cheap op-count exceeds
    the guard the bulk ``get_drawings()`` is never attempted.
    """
    # The pipeline already counted this page's drawing operators; accepting the
    # value avoids a second content-stream read per page. Omitting it preserves
    # the original standalone behaviour exactly.
    ops = vector_ops if vector_ops is not None else page_router.vector_ops_of(page)
    if not page_router.geometry_is_safe(ops):
        return [], [], True
    lines: list[tuple] = []
    rects: list[Any] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                lines.append(((item[1].x, item[1].y), (item[2].x, item[2].y)))
            elif item[0] == "re":
                rects.append(item[1])
    vertical, horizontal = _accumulate_grid(
        lines, rects, page.rect.width, page.rect.height, min_len
    )
    return vertical, horizontal, False


# ── grid cuts → rows ──────────────────────────────────────────────────────────


def _bands_from_cuts(hcuts: Sequence[float]) -> list[tuple[float, float]]:
    ordered = sorted(hcuts)
    return [
        (ordered[i], ordered[i + 1])
        for i in range(len(ordered) - 1)
        if ordered[i + 1] - ordered[i] > BAND_GAP
    ]


def _bands_from_words(words: Sequence[tuple]) -> list[tuple[float, float]]:
    ys = sorted({round(word[1], 1) for word in words})
    groups: list[list[float]] = []
    for y in ys:
        if not groups or y - groups[-1][-1] > WORD_BAND_GAP:
            groups.append([y])
        else:
            groups[-1].append(y)
    return [(group[0] - 2, group[-1] + 2) for group in groups]


def _rows_from_bands(
    words: Sequence[tuple],
    vcuts: Sequence[float],
    bands: Sequence[tuple[float, float]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for low, high in bands:
        cells: dict[int, list[str]] = {}
        for cx, cy, text, _box in words:
            if low <= cy <= high:
                index = sum(1 for cut in vcuts if cx > cut)
                cells.setdefault(index, []).append(text)
        row = [" ".join(cells.get(i, [])) for i in range(len(vcuts) + 1)]
        if any(cell.strip() for cell in row):
            rows.append(row)
    return rows


def _band_and_rows(
    words: Sequence[tuple], vcuts: Sequence[float], hcuts: Sequence[float] | None
) -> list[list[str]]:
    """Row banding: rect y-edges when available, else word clusters.

    Using rect edges keeps multi-line cells intact, which word-cluster banding
    would split (40+-char cells verified in the corpus).
    """
    bands = (
        _bands_from_cuts(hcuts) if hcuts and len(hcuts) >= 4 else _bands_from_words(words)
    )
    return _rows_from_bands(words, vcuts, bands)


def lattice_table_items(
    words: Sequence[tuple], vcuts: Sequence[float], hcuts: Sequence[float] | None = None
) -> list[list[str]]:
    """Rows from reading-frame words (rotated path)."""
    return _band_and_rows(words, vcuts, hcuts)


def page_words(page: Any) -> list[tuple]:
    """Words in PDF space as ``(center_x, center_y, text, bbox)``."""
    out: list[tuple] = []
    for block in page_router.page_text(page, "dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    x0, y0, x1, y1 = span["bbox"]
                    out.append(((x0 + x1) / 2, (y0 + y1) / 2, text, span["bbox"]))
    return out


def lattice_table(
    page: Any, vcuts: Sequence[float], hcuts: Sequence[float] | None = None
) -> list[list[str]]:
    """Rows from PDF-space words (portrait path)."""
    return _band_and_rows(page_words(page), vcuts, hcuts)


# ── acceptance gate ───────────────────────────────────────────────────────────


def numeric_row(row: Sequence[str]) -> bool:
    """A row is numeric when enough of its cells are bare numbers."""
    numeric = sum(1 for cell in row if _AMOUNT_RE.fullmatch((cell or "").strip()))
    return numeric >= max(2, len(row) // 3)


def amountish_cells(rows: Sequence[Sequence[str]]) -> int:
    """Count cells that look like amounts, including ``-`` placeholders."""
    count = 0
    for row in rows:
        for cell in row:
            text = (cell or "").strip()
            if _AMOUNT_RE.fullmatch(text) or text == "-":
                count += 1
    return count


def passes_gate(rows: Sequence[Sequence[str]], vcuts: int, hcuts: int) -> bool:
    """Spec §3.3 acceptance gate. **Both** clauses are required in the union.

    Clause 1 alone rejects Indian-format financial tables whose cells are mostly
    ``-`` placeholders; clause 2 alone would accept any dense grid. Measured:
    SCHEDULE-4A fails clause 1 and passes only clause 2.
    """
    if len(rows) < MIN_ROWS or vcuts < MIN_VCUTS:
        return False
    numeric_rows = sum(1 for row in rows if numeric_row(row))
    clause_one = numeric_rows >= max(3, int(NUMERIC_ROW_FRACTION * len(rows)))
    strong_grid = vcuts >= STRONG_GRID_VCUTS and hcuts >= STRONG_GRID_HCUTS
    clause_two = strong_grid and amountish_cells(rows) >= STRONG_GRID_AMOUNTISH
    return clause_one or clause_two


def table_caption(page_text: str) -> str | None:
    """Caption for the page's table, or ``None``.

    Takes the **last** match on the page, exactly as the reference
    implementation does. This is a known oddity — a page with two tables takes
    the later caption for both — preserved here so output stays comparable with
    the measured artifacts. Revisit only with corpus evidence.
    """
    match = None
    for found in TABLE_CAPTION_RE.finditer(page_text or ""):
        match = found
    if match is None:
        return None
    return f"{match.group(1)}: {match.group(2)}".strip()


def rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Render rows as a pipe table. The first row is treated as the header."""
    if not rows:
        return ""
    header = rows[0]
    out = [
        "| " + " | ".join(cell.strip() for cell in header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows[1:]:
        out.append("| " + " | ".join(cell.strip() for cell in row) + " |")
    return "\n".join(out)


# ── Stage 2: portrait detection ───────────────────────────────────────────────


def detect_tables(
    page: Any,
    pno: int,
    page_text: str,
    config: V2Config | None = None,
    *,
    vector_ops: int | None = None,
) -> tuple[list[dict], bool]:
    """Detect portrait ruled tables on one page.

    Returns ``(blocks, guard_skipped)``. ``pno`` is 0-based; the block records
    the 1-based page number.
    """
    if config is not None and not config.tables_active:
        return [], False

    vcuts, hcuts, guarded = lattice_cuts(page, config, vector_ops=vector_ops)
    if guarded or len(vcuts) < MIN_VCUTS:
        return [], guarded

    rows = lattice_table(page, vcuts, hcuts)
    if not passes_gate(rows, len(vcuts), len(hcuts)):
        return [], False

    numeric_rows = sum(1 for row in rows if numeric_row(row))
    block = {
        "table_id": "t1",
        "doc": None,
        "page": pno + 1,
        "bbox": [min(vcuts) - 6, 0, max(vcuts) + 6, page.rect.height],
        "ncols": len(vcuts) + 1,
        "nrows": len(rows),
        "caption": table_caption(page_text),
        "method": "vector-lattice",
        "numeric_row_fraction": round(numeric_rows / max(1, len(rows)), 2),
        "markdown": rows_to_markdown(rows),
        "rows": rows,
    }
    return [block], False


# ── Stage 4: rotated detection ────────────────────────────────────────────────


def detect_tables_rotated(
    page: Any,
    pno: int,
    page_text: str,
    config: V2Config | None = None,
    *,
    vector_ops: int | None = None,
    orientation: Any | None = None,
) -> tuple[list[dict], str, float, int]:
    """Detect tables on content-rotated pages via the reading-frame transform.

    Returns ``(blocks, orientation_label, confidence, rotation)``. Orientation is
    re-detected here rather than trusted from a caller, so the block's recorded
    ``orientation_conf`` always matches the geometry actually used.

    A rotated block **supersedes** the portrait block on the same page; the
    caller (Stage 5 pipeline) enforces that.
    """
    # Both values are already known to the pipeline for this page; passing
    # them in removes a duplicate orientation scan and op-count per page.
    if orientation is None:
        orientation = page_router.detect_orientation(page)
    if config is not None and not config.rotated_tables_active:
        return [], orientation.label, orientation.conf, 0
    if orientation.conf < page_router.ORIENTATION_CONF_GATE or orientation.rot == 0:
        return [], orientation.label, orientation.conf, orientation.rot

    _ops = vector_ops if vector_ops is not None else page_router.vector_ops_of(page)
    if not page_router.geometry_is_safe(_ops):
        return [], orientation.label, orientation.conf, orientation.rot

    words, lines, rects, size = page_router.reading_frame_items(page, orientation.rot)
    vcuts, hcuts = lattice_cuts_items(words, lines, rects, size)
    if len(vcuts) < MIN_VCUTS:
        return [], orientation.label, orientation.conf, orientation.rot

    rows = lattice_table_items(words, vcuts, hcuts)
    if not passes_gate(rows, len(vcuts), len(hcuts)):
        return [], orientation.label, orientation.conf, orientation.rot

    numeric_rows = sum(1 for row in rows if numeric_row(row))
    block = {
        "table_id": "trot",
        "doc": None,
        "page": pno + 1,
        "bbox": [min(vcuts) - 6, 0, max(vcuts) + 6, size[1]],
        "ncols": len(vcuts) + 1,
        "nrows": len(rows),
        "caption": table_caption(page_text),
        "method": f"rotated-lattice(rot{orientation.rot})",
        "orientation": orientation.label,
        "orientation_conf": orientation.conf,
        "numeric_row_fraction": round(numeric_rows / max(1, len(rows)), 2),
        "markdown": rows_to_markdown(rows),
        "rows": rows,
    }
    return [block], orientation.label, orientation.conf, orientation.rot


# ── Stage 7: multi-page continuation ─────────────────────────────────────────


def _normalise(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())[:60]


def caption_number(block: dict) -> str | None:
    """``SCHEDULE 2`` → ``2``; ``Table 10.4`` → ``10.4``; ``4A`` → ``4A``."""
    match = _CAPTION_NUMBER_RE.search(block.get("caption") or "")
    if not match:
        return None
    return match.group(1).upper().rstrip(".")


def col_sig(block: dict) -> dict | None:
    """Column signature: ncols + non-empty-count mode + last-row cell lengths."""
    rows = block.get("rows") or []
    if not rows:
        return None
    counts = [sum(1 for cell in row if cell.strip()) for row in rows]
    mode = max(set(counts), key=counts.count)
    return {
        "ncols": block.get("ncols"),
        "mode_nonempty": mode,
        "first_row_lens": [len(cell.strip()) for cell in rows[-1]][:16],
    }


def sig_similar(a: dict | None, b: dict | None) -> tuple[bool, str]:
    if not a or not b:
        return False, "missing-sig"
    if abs((a["ncols"] or 0) - (b["ncols"] or 0)) > 1:
        return False, f"ncols {a['ncols']} vs {b['ncols']}"
    left, right = a["first_row_lens"], b["first_row_lens"]
    if len(left) != len(right):
        return False, "row-lens-len"
    diff = sum(abs(x - y) for x, y in zip(left, right, strict=False)) / max(1, len(left))
    return diff < 6.0, f"row-lens-mean-diff={diff:.1f}"


def header_key(block: dict) -> str:
    rows = block.get("rows") or []
    keys = [_normalise(cell) for row in rows[:2] for cell in row if cell.strip()]
    return "|".join(keys[:6])


def should_merge(
    prev: dict, nxt: dict, page_gap_max: int = 2
) -> tuple[bool, str, dict]:
    """Deterministic merge decision **with a reason string**.

    All of these must hold: same document, page gap within ``page_gap_max``,
    equal caption number, compatible column signature, and — for a gap above 1 —
    a repeated header or an explicit ``contd``/``continued``/``concluded`` marker.

    Measured negatives that must **not** merge: Table 10.4 vs 10.5, Table 1 vs
    Table 2, SCHEDULE 2 vs 4 vs 4A, and SCHEDULE-4A p226 vs p233 (15 vs 9
    columns), which is blocked by both the column signature and the gap.
    """
    if prev.get("doc") != nxt.get("doc"):
        return False, "diff-doc", {}

    gap = abs((nxt.get("page") or 0) - (prev.get("page") or 0))
    if gap < 1 or gap > page_gap_max:
        return False, f"page-gap={gap}", {}

    left, right = caption_number(prev), caption_number(nxt)
    if not left or not right:
        return False, "no-caption-number", {}
    if left != right:
        return False, f"caption {left} != {right}", {}

    similar, why = sig_similar(col_sig(prev), col_sig(nxt))
    if not similar:
        return False, f"col-sig: {why}", {}

    header_repeat = bool(header_key(prev)) and header_key(prev) == header_key(nxt)
    contd_mark = bool(_CONTD_RE.search(prev.get("caption") or "")) or bool(
        _CONTD_RE.search(nxt.get("caption") or "")
    )
    if gap > 1 and not (header_repeat or contd_mark):
        return False, "gap>1 without header-repeat/contd", {}

    provenance = {
        "pages": sorted({prev["page"], nxt["page"]}),
        "header_repeated": header_repeat,
        "contd_marker": contd_mark,
    }
    return True, f"caption={left} {why} gap={gap}", provenance


def merge_pair(prev: dict, nxt: dict, provenance: dict) -> dict:
    """Merge a continuation into one logical block, keeping full provenance.

    Child pages keep their own blocks; the merged block is *additional* evidence
    and never replaces them (spec §3.4).
    """
    next_rows = list(nxt["rows"])
    header_repeat = provenance["header_repeated"]
    data = list(prev["rows"])
    if header_repeat:
        head_norms = {_normalise(" ".join(row)) for row in prev["rows"][:3]}
        while next_rows and _normalise(" ".join(next_rows[0])) in head_norms:
            next_rows = next_rows[1:]
    data += next_rows

    merged = dict(prev)
    merged["table_id"] = f"{prev.get('table_id')}+cont"
    merged["nrows"] = len(data)
    merged["rows"] = data
    merged["markdown"] = rows_to_markdown(data)
    merged["continuation"] = {
        "pages": provenance["pages"],
        "rows_from_pages": {
            str(prev["page"]): len(prev["rows"]),
            str(nxt["page"]): len(next_rows),
        },
        "header_repeated": header_repeat,
    }
    return merged


def merge_continuations(
    blocks: list[dict], page_gap_max: int = 2
) -> tuple[list[dict], list[dict]]:
    """Group a document's blocks by signature and merge real continuations.

    Returns ``(logical_blocks, decisions)``. ``logical_blocks`` contains the
    original blocks **plus** any merged block (additive, never destructive);
    ``decisions`` records every pairwise verdict with its reason, which is what
    makes a false-merge investigation possible.
    """
    by_caption: dict[str, list[dict]] = {}
    for block in blocks:
        number = caption_number(block)
        if number:
            by_caption.setdefault(number, []).append(block)

    merged_blocks: list[dict] = []
    decisions: list[dict] = []
    consumed: set[int] = set()

    for number, group in by_caption.items():
        ordered = sorted(group, key=lambda b: b.get("page") or 0)
        for i in range(len(ordered) - 1):
            prev, nxt = ordered[i], ordered[i + 1]
            ok, reason, provenance = should_merge(prev, nxt, page_gap_max)
            decisions.append(
                {
                    "caption_number": number,
                    "pages": [prev.get("page"), nxt.get("page")],
                    "merge": ok,
                    "reason": reason,
                }
            )
            if ok and id(prev) not in consumed and id(nxt) not in consumed:
                merged_blocks.append(merge_pair(prev, nxt, provenance))
                consumed.add(id(prev))
                consumed.add(id(nxt))

    return blocks + merged_blocks, decisions


__all__ = [
    "MIN_ROWS",
    "MIN_RULE_LEN",
    "MIN_VCUTS",
    "TABLE_CAPTION_RE",
    "amountish_cells",
    "caption_number",
    "col_sig",
    "detect_tables",
    "detect_tables_rotated",
    "header_key",
    "lattice_cuts",
    "lattice_cuts_items",
    "lattice_table",
    "lattice_table_items",
    "merge_continuations",
    "merge_pair",
    "numeric_row",
    "page_words",
    "passes_gate",
    "rows_to_markdown",
    "should_merge",
    "sig_similar",
    "table_caption",
]
