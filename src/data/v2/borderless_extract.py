"""Stage 9 — borderless (ruleless) table extraction. **Experimental, OFF.**

Ports ``v2proto/borderless.py`` including every guard. This is the one V2
detector that is *not* production-ready, and the spec's own evidence says why
(§13.1): across the measured corpus it found **0 natural positives** and
rejected 14/14 visually-confirmed lookalikes. It exists so the negative
behaviour is reproducible and so a future corpus with genuine borderless tables
has something to evaluate — not because it is trusted.

Consequently:

* ``V2_TABLE_BORDERLESS`` defaults to **false** and every feature gate is
  conjunctive with ``V2_ENABLED``;
* nothing here is enabled automatically, and no synthetic result may enable it;
* the method label is ``borderless-projection``, never ``vector-lattice`` or
  ``rotated-lattice``, so a borderless block can never be mistaken for a
  lattice table downstream;
* behaviour is fully deterministic — projection peaks are broken by coordinate,
  not by dict order.

Guards, in evaluation order (all preserved verbatim from the prototype):

    R0 orientation  non-portrait at conf >= 0.8 -> belongs to the rotated path
    R1 rules        >= 3 v-cuts or >= 3 h-cuts -> ruled, use the lattice path
    R1' DoS         op-count guard tripped -> no geometry at all
    R2 form         >= 6 dot-leader/underline runs -> a form, not a table
    R3 prose        CONTENTS heading, or >= 4 "[n]" bibliography lines
    R4 columns      projection peaks must yield >= 3 column intervals
    R5 rows         each surviving column must be populated in >= 2 y-bands,
                    and >= 3 columns must survive
    R6 acceptance   >= 4 rows AND >= 12 amount-like cells
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.data.v2 import page_router, table_extract
from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# ── guard thresholds (spec §13.1 / prototype borderless.py) ───────────────────
#: R0 — orientation confidence at which a page belongs to the rotated detector.
R0_ORIENTATION_CONF = 0.8
#: R1 — this many grid cuts means the page is ruled, not borderless.
R1_MIN_CUTS = 3
#: R2 — dot-leader / underline run.
R2_RUN_RE = re.compile(r"\.{6,}|_{4,}")
#: R2 — this many runs means a form.
R2_MIN_RUNS = 6
#: R3 — contents heading.
R3_CONTENTS_RE = re.compile(r"^\s*CONTENTS\s*$", re.MULTILINE)
#: R3 — bibliography entry.
R3_BIB_RE = re.compile(r"^\s*\[\d+\]", re.MULTILINE)
#: R3 — this many bibliography lines means a reference list.
R3_MIN_BIB_LINES = 4
#: R4 — projection bin width in points.
R4_BIN_PT = 2
#: R4 — minimum peak height, as a count.
R4_MIN_PEAK_COUNT = 4
#: R4 — peak height as a fraction of the tallest bin.
R4_PEAK_FRACTION = 0.25
#: R4 — peaks closer than this are merged.
R4_MERGE_PT = 8
#: R4 — a table needs at least this many column intervals.
R4_MIN_COLUMNS = 3
#: R5 — word-cluster band gap, matching the lattice banding.
R5_BAND_GAP = 9.0
#: R5 — a column must be populated in at least this many bands to be real.
R5_MIN_BANDS = 2
#: R5 — surviving columns required.
R5_MIN_LIVE_COLUMNS = 3
#: R6 — acceptance: minimum rows.
R6_MIN_ROWS = 4
#: R6 — acceptance: minimum amount-like cells.
R6_MIN_AMOUNTISH = 12
#: Minimum words before projection is attempted at all.
MIN_WORDS = 20

METHOD = "borderless-projection"


def guard_reasons(
    page: Any, config: V2Config | None = None
) -> tuple[list[str], tuple[list[float], list[float]]]:
    """Evaluate the rejection guards R0-R3. Returns ``(reasons, (vcuts, hcuts))``.

    An empty ``reasons`` list means the page is a borderless *candidate*; it does
    not mean a table was found (R4-R6 still apply).
    """
    reasons: list[str] = []

    orientation = page_router.detect_orientation(page)
    if orientation.conf >= R0_ORIENTATION_CONF and orientation.rot != 0:
        reasons.append(f"R0-rotated({orientation.label})")

    vcuts, hcuts, guarded = table_extract.lattice_cuts(page, config)
    if guarded:
        reasons.append("R1-DoS-guard")
    elif len(vcuts) >= R1_MIN_CUTS or len(hcuts) >= R1_MIN_CUTS:
        reasons.append(f"R1-ruled(v={len(vcuts)},h={len(hcuts)})")

    text = page.get_text("text") or ""
    if len(R2_RUN_RE.findall(text)) >= R2_MIN_RUNS:
        reasons.append("R2-form(dot/underline runs)")
    if R3_CONTENTS_RE.search(text):
        reasons.append("R3-contents")
    if len(R3_BIB_RE.findall(text)) >= R3_MIN_BIB_LINES:
        reasons.append("R3-bibliography")

    return reasons, (vcuts, hcuts)


def projection_columns(page: Any) -> tuple[list[float], list]:
    """Column start positions from a smoothed x-projection profile (R4).

    Words are binned by ``x0`` into 2pt bins, smoothed with a 3-tap kernel, and
    local maxima above ``max(4, 0.25*peak)`` become column starts. Peaks within
    ``R4_MERGE_PT`` of an earlier one are dropped, so the result is a stable,
    order-independent list.
    """
    words = [w for w in page.get_text("words") if w[4].strip()]
    if len(words) < MIN_WORDS:
        return [], words

    bins = [0] * (int(page.rect.width // R4_BIN_PT) + 2)
    for word in words:
        bins[int(word[0] // R4_BIN_PT)] += 1

    smoothed = [
        bins[i]
        + (bins[i - 1] if i else 0)
        + (bins[i + 1] if i + 1 < len(bins) else 0)
        for i in range(len(bins))
    ]
    tallest = max(smoothed) or 1
    threshold = max(R4_MIN_PEAK_COUNT, R4_PEAK_FRACTION * tallest)

    peaks: list[float] = []
    for i in range(2, len(smoothed) - 2):
        is_local_max = smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
        if smoothed[i] >= threshold and is_local_max:
            peaks.append(i * R4_BIN_PT)

    merged: list[float] = []
    for peak in peaks:
        if merged and peak - merged[-1] < R4_MERGE_PT:
            continue
        merged.append(peak)
    return merged, words


def extract_borderless(
    page: Any,
    pno: int,
    page_text: str,
    config: V2Config | None = None,
) -> tuple[dict | None, list[str]]:
    """Attempt borderless extraction on one page.

    Returns ``(block, reasons)``. ``block`` is ``None`` whenever any guard or
    acceptance rule fires, and ``reasons`` always says which — a rejection is
    evidence, not silence.
    """
    cfg = config if config is not None else V2Config()
    if not cfg.borderless_active:
        return None, ["disabled(V2_TABLE_BORDERLESS)"]

    reasons, _cuts = guard_reasons(page, cfg)
    if reasons:
        return None, reasons

    columns, words = projection_columns(page)
    if len(columns) < R4_MIN_COLUMNS:
        return None, [f"R4-columns<{R4_MIN_COLUMNS}({len(columns)})"]

    intervals = [
        (columns[i], columns[i + 1] if i + 1 < len(columns) else page.rect.width)
        for i in range(len(columns))
    ]

    # ── R5: rows from y-bands, columns must be populated repeatedly ────────
    ys = sorted({round(word[1], 1) for word in words})
    bands: list[list[float]] = []
    for y in ys:
        if not bands or y - bands[-1][-1] > R5_BAND_GAP:
            bands.append([y])
        else:
            bands[-1].append(y)
    band_ranges = [(band[0] - 2, band[-1] + 2) for band in bands]

    rows: list[list[str]] = []
    populated = [0] * len(intervals)
    for low, high in band_ranges:
        cells: list[list[str]] = [[] for _ in intervals]
        for word in words:
            cx, cy, text = word[0], word[1], word[4]
            if low <= cy <= high:
                for index, (left, right) in enumerate(intervals):
                    if left <= cx < right:
                        cells[index].append(text)
                        break
        row = [" ".join(cell) for cell in cells]
        for index, cell in enumerate(row):
            if cell.strip():
                populated[index] += 1
        if any(cell.strip() for cell in row):
            rows.append(row)

    live = [i for i, count in enumerate(populated) if count >= R5_MIN_BANDS]
    if len(live) < R5_MIN_LIVE_COLUMNS:
        return None, [f"R5-live-columns<{R5_MIN_LIVE_COLUMNS}({len(live)})"]

    rows = [[row[i] for i in live] for row in rows]
    rows = [row for row in rows if any(cell.strip() for cell in row)]

    # ── R6: acceptance ──────────────────────────────────────────────────────
    amountish = table_extract.amountish_cells(rows)
    if len(rows) < R6_MIN_ROWS or amountish < R6_MIN_AMOUNTISH:
        return None, [f"R6-rows={len(rows)},amountish={amountish}"]

    block = {
        "table_id": "tbl",
        "doc": None,
        "page": pno + 1,
        "bbox": [columns[0] - 6, 0, columns[-1] + 6, page.rect.height],
        "ncols": len(live),
        "nrows": len(rows),
        "caption": table_extract.table_caption(page_text),
        "method": METHOD,
        "numeric_row_fraction": round(
            sum(1 for row in rows if table_extract.numeric_row(row)) / max(1, len(rows)), 2
        ),
        "markdown": table_extract.rows_to_markdown(rows),
        "rows": rows,
    }
    return block, []


def is_borderless(block: dict | None) -> bool:
    """True only for a block produced by this detector."""
    return bool(block) and block.get("method") == METHOD


__all__ = [
    "METHOD",
    "R0_ORIENTATION_CONF",
    "R1_MIN_CUTS",
    "R2_MIN_RUNS",
    "R3_MIN_BIB_LINES",
    "R4_MIN_COLUMNS",
    "R5_MIN_LIVE_COLUMNS",
    "R6_MIN_AMOUNTISH",
    "R6_MIN_ROWS",
    "extract_borderless",
    "guard_reasons",
    "is_borderless",
    "projection_columns",
]
