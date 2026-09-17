"""Stage 1 — page routing primitives for the experimental INCOIS V2 layer.

Four functions, ported from the Phase-8/9 reference implementation
(``v2proto/v2_extract.py``, with the orientation variant cross-checked against
``v2proto/rot_tables.py``). Thresholds and rule order are preserved **verbatim**
from the prototype; they are the values the spec's measured results were
produced with, and nothing in this repository required changing them.

Scope: classification and geometry only. This module performs **no OCR, no table
extraction, no figure work, and no sidecar writing** — those are later stages.
It does not import torch, sentence-transformers, or any retrieval code.

Import weight: pymupdf is imported **lazily** inside the one function that needs
``fitz.Rect``. Everything else here is pure arithmetic and can be exercised with
stub page objects, which is how the coordinate transform is tested.

Determinism: no ML, no sampling, no iteration over dict/set order in a way that
affects output. Repeated calls on the same page return identical results.
"""

from __future__ import annotations

import collections
import re
from typing import TYPE_CHECKING, NamedTuple

from src.data.v2.config import V2Config

#: Attribute under which a per-page memo dict is attached to a PyMuPDF ``Page``.
_MEMO_ATTR = "_v2_page_memo"


def page_memo(page: Any) -> dict | None:
    """Per-page memo dict, or ``None`` if this page object cannot hold one.

    ``document[pno]`` returns a **fresh** ``Page`` object on every call, so a
    memo attached to the object is automatically scoped to that one page's
    processing and is discarded with the object. There is no global cache, no
    key management, no invalidation logic, and no way for one page's cached
    value to leak into another's.

    Returns ``None`` rather than raising if the object refuses the attribute,
    in which case every caller simply recomputes — caching is a pure
    optimisation and must never be load-bearing.
    """
    memo = getattr(page, _MEMO_ATTR, None)
    if memo is not None:
        return memo
    memo = {}
    try:
        setattr(page, _MEMO_ATTR, memo)
    except Exception:  # noqa: BLE001 - exotic Page implementation; skip caching
        return None
    return memo


def page_text(page: Any, fmt: str = "text", **kwargs: Any) -> Any:
    """``page.get_text(fmt, **kwargs)``, memoised per page **and per format**.

    ``"text"``, ``"dict"`` and ``"words"`` are three genuinely different
    extractions. The cache key carries the format *and* every keyword argument,
    so a ``"dict"`` result can never be handed to a caller that asked for
    ``"words"``. Measured on a 237-page native document, the pipeline was
    calling ``get_text`` ~6 times per page for ~3 distinct extractions.

    **Contract:** the memo lives on the ``Page`` object and assumes the page is
    not mutated after its first read. The extraction pipeline never mutates a
    page — there is no ``draw_*``/``insert_*`` call anywhere under ``src/`` — and
    ``document[pno]`` hands out a fresh object per page, so the memo is born and
    dies with one page's processing. Note that :func:`vector_ops_of` and
    :func:`detect_orientation` are deliberately *not* memoised: a caller may
    legitimately draw on a page and re-measure it, and a hidden cache would
    silently return a stale answer. Their duplicates are removed instead by
    passing the already-computed value into table extraction.
    """
    key = ("get_text", fmt, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
    memo = page_memo(page)
    if memo is not None and key in memo:
        return memo[key]
    value = page.get_text(fmt, **kwargs)
    if memo is not None:
        memo[key] = value
    return value

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from typing import Any

# ── heuristics ported verbatim from v2proto/v2_extract.py ─────────────────────

#: Mojibake detector. The private-use ranges are deliberately split around
#: U+F0B0-U+F0FF so that Wingdings symbol-font bullets (U+F0B7, used as a
#: list marker throughout the INCOIS annual reports) are **not** counted as
#: damage. Spec §3.1 records this as a measured Phase-6 false positive that was
#: corrected; collapsing the range back to a single ``[\ue000-\uf8ff]`` span
#: would misclassify clean pages.
MOJIBAKE_RE = re.compile(
    r"\b[a-z]{1,6}\{[a-z]"
    r"|\(cid:\d+\)"
    r"|[\ue000-\uf0a9]"
    r"|[\uf100-\uf8ff]"
    r"|\b[A-Za-z]*[a-z]\{[A-Za-z]+\b"
)

#: Page classes, in the order ``classify_page`` tests them.
PAGE_CLASSES = ("native", "mojibake", "scan", "vector", "native_low", "empty")

#: Orientation labels produced by :func:`detect_orientation`.
ORIENTATION_LABELS = (
    "portrait",
    "rot90_cw_text",
    "rot90_ccw_text",
    "rot180",
    "mixed",
    "empty",
)

#: Spec §3.1: a detected rotation is applied only when the dominant text
#: direction covers at least this fraction of the page's text characters.
ORIENTATION_CONF_GATE = 0.8

#: Spec §3.3 DoS guard. ``reading_frame_items`` must call the (expensive) bulk
#: ``page.get_drawings()``; callers check the cheap op-count first and skip
#: geometry above this threshold.
VECTOR_OP_GUARD = 400_000

#: Byte patterns counted by :func:`vector_ops_of`. Ported unchanged: note the
#: asymmetry (``c``/``m`` are counted only when newline-terminated, ``l``/``re``
#: in both forms). It is a *relative* content-volume signal compared against a
#: threshold, not an exact op census, and the thresholds were calibrated against
#: this exact expression.
_VECTOR_OP_PATTERNS = (b" l\n", b" re\n", b" c\n", b" m\n", b" l ", b" re ")

#: Minimum characters of real text before a page counts as having a usable
#: native text layer (spec §3.1).
NATIVE_CHAR_FLOOR = 50


class Orientation(NamedTuple):
    """Detected text orientation.

    A ``NamedTuple`` so the spec §3.1 contract ``("empty", 0.0, 0)`` holds by
    equality while call sites can still use named fields.
    """

    label: str
    conf: float
    rot: int


# ── S1a: content volume, without touching geometry ────────────────────────────


def vector_ops_of(page: Any) -> int:
    """Cheap count of drawing operators in a page's content stream.

    Reads ``page.read_contents()`` and counts byte patterns. This deliberately
    avoids ``page.get_drawings()``: spec §3.1 records that materialising drawing
    objects on a 1.46M-operator page hangs (measured, Phase 6). Every page is
    counted this way; only pages that pass the guard later pay for real
    geometry.

    Never raises — a page with no readable content stream counts as 0.
    """
    try:
        raw = page.read_contents()
    except Exception:  # noqa: BLE001 - unreadable stream must not break routing
        return 0
    if raw is None:
        return 0
    return sum(raw.count(pattern) for pattern in _VECTOR_OP_PATTERNS)


# ── S1b: page classification ──────────────────────────────────────────────────


def classify_page(
    page: Any,
    vector_ops: int,
    config: V2Config | None = None,
) -> tuple[str, dict[str, float | int]]:
    """Classify one page as native | mojibake | scan | vector | native_low | empty.

    Rule order and thresholds are exactly the prototype's; reordering them
    changes results (``chars >= 50`` is tested before any image or vector
    check, so a page with both a text layer and images is ``native``).

    ``vector_ops`` is passed in rather than computed here so a caller can count
    once per page and reuse it for the guard, the sidecar record, and the class.

    Returns ``(class, meta)`` where ``meta`` carries the four measured inputs.
    Note ``chars`` is the **native** text-layer length; the sidecar's ``chars``
    field is post-route (post-OCR) and is therefore not comparable for pages
    that were OCR'd.
    """
    cfg = config if config is not None else V2Config()

    text = page_text(page, "text") or ""
    chars = len(text.strip())
    lines = [line for line in text.splitlines() if line.strip()]
    moji_lines = sum(1 for line in lines if MOJIBAKE_RE.search(line))
    moji_ratio = moji_lines / max(1, len(lines))

    images = page.get_image_info(xrefs=True) or []
    image_area = sum(item["width"] * item["height"] for item in images)

    if chars >= NATIVE_CHAR_FLOOR and moji_ratio < cfg.mojibake_ratio_gate:
        page_class = "native"
    elif chars >= NATIVE_CHAR_FLOOR:
        page_class = "mojibake"
    elif images and image_area > 0:
        page_class = "scan"
    elif vector_ops >= cfg.vector_op_threshold:
        page_class = "vector"
    elif chars > 0:
        page_class = "native_low"
    else:
        page_class = "empty"

    meta = {
        "chars": chars,
        "moji_ratio": round(moji_ratio, 3),
        "n_images": len(images),
        "vector_ops": vector_ops,
    }
    return page_class, meta


def page_is_blank(page: Any, dpi: int = 30) -> bool:
    """Render-check for a page classified ``empty``: is it *really* blank?

    **[UNVALIDATED]** — deviation from the prototype, flagged deliberately. The
    reference implementation renders at 30 dpi and then discards the result
    (``v2proto/v2_extract.py:517-523`` computes ``cls_eff`` and assigns
    ``route = route if cls_eff == "empty" else route``, which is a no-op on both
    branches). Spec §3.1 and test T1 both describe a render-check that marks
    truly blank pages, so this implements one for real.

    It is kept **separate** from :func:`classify_page` so classification stays
    byte-identical to the prototype, and it is not wired into any pipeline yet.
    The blank threshold below has not been calibrated against real INCOIS pages
    because the source PDFs are not available in this environment.
    """
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        samples = bytes(pixmap.samples)
    except Exception:  # noqa: BLE001 - an unrenderable page is not "blank"
        return False
    if not samples:
        return True
    # A scanned-but-textless page still has ink; a genuinely blank one is
    # uniformly near-white. Compare every channel byte against a high floor.
    return all(value >= 250 for value in samples)


# ── S1c: orientation ──────────────────────────────────────────────────────────


def detect_orientation(page: Any) -> Orientation:
    """Char-weighted histogram of text-line direction vectors.

    Mapping (spec §3.1, prototype-verified):

    =============  ===================  =====
    dominant dir   meaning             rot
    =============  ===================  =====
    ``(1, 0)``     normal reading        0
    ``(0, -1)``    reads bottom-to-top  90
    ``(0, 1)``     reads top-to-bottom  270
    ``(-1, 0)``    upside down          180
    anything else  mixed                 0
    =============  ===================  =====

    A page with no text at all returns ``("empty", 0.0, 0)``.

    Confidence is rounded to 3 decimals to match the ``orientation_conf`` values
    recorded in the measured ``tables.jsonl`` artifacts (e.g. ``0.998``).
    """
    chars: collections.Counter = collections.Counter()
    for block in page_text(page, "dict")["blocks"]:
        for line in block.get("lines", []):
            direction = tuple(round(value, 1) for value in line["dir"])
            chars[direction] += sum(len(span["text"]) for span in line["spans"])

    if not chars:
        return Orientation("empty", 0.0, 0)

    total = sum(chars.values()) or 1
    dominant, count = chars.most_common(1)[0]
    conf = round(count / total, 3)

    if dominant == (1.0, 0.0):
        label, rot = "portrait", 0
    elif dominant == (0.0, -1.0):
        label, rot = "rot90_cw_text", 90
    elif dominant == (0.0, 1.0):
        label, rot = "rot90_ccw_text", 270
    elif dominant == (-1.0, 0.0):
        label, rot = "rot180", 180
    else:
        label, rot = "mixed", 0

    return Orientation(label, conf, rot)


def reading_rotation(orientation: Orientation) -> int:
    """The rotation to apply, honouring the spec §3.1 confidence gate.

    A low-confidence detection is **not** applied: a wrong frame scrambles a
    table far worse than leaving it unrotated, so anything below
    :data:`ORIENTATION_CONF_GATE` falls back to 0.
    """
    if orientation.label not in ("rot90_cw_text", "rot90_ccw_text", "rot180"):
        return 0
    return orientation.rot if orientation.conf >= ORIENTATION_CONF_GATE else 0


# ── S1d: reading-frame coordinate transform ───────────────────────────────────


def _map_point(x: float, y: float, rot: int, width: float, height: float) -> tuple[float, float]:
    """Pure coordinate arithmetic from PDF space into the reading frame.

    ``page.set_rotation()`` is **not** used and must not be substituted here:
    measured in Phase 9, it changes rendering but has no effect on
    ``get_text`` word coordinates, so text and geometry would end up in
    different frames.
    """
    if rot == 90:
        return (height - y, x)
    if rot == 270:
        return (y, width - x)
    if rot == 180:
        return (width - x, height - y)
    return (x, y)


def reading_frame_items(
    page: Any,
    rot: int,
) -> tuple[list[tuple], list[tuple], list, tuple[float, float]]:
    """Map words and drawing geometry into the reading frame.

    Returns ``(words, lines, rects, (W, H))`` where

    * ``words``  — ``(center_x, center_y, text, (x0, y0, x1, y1))``, reading frame
    * ``lines``  — ``((x1, y1), (x2, y2))`` endpoint pairs
    * ``rects``  — ``fitz.Rect`` objects (the Stage-2 lattice detector consumes
      ``.x0/.y0/.x1/.y1/.width/.height``)
    * ``(W, H)`` — page size in the reading frame; width and height swap for
      90/270

    Word bounding boxes are re-normalised after the transform
    (``min``/``max`` on both mapped corners) because a rotation maps a
    top-left/bottom-right pair onto a pair that is no longer ordered.

    This is the one function that must call the bulk ``page.get_drawings()`` —
    op-counting cannot yield coordinates. Callers are expected to check
    ``vector_ops_of(page) <= VECTOR_OP_GUARD`` first (spec §3.3); above that the
    page is marked ``guard_skipped`` and geometry is not attempted.
    """
    import pymupdf

    width, height = page.rect.width, page.rect.height

    def point(x: float, y: float) -> tuple[float, float]:
        return _map_point(x, y, rot, width, height)

    words: list[tuple] = []
    for word in page_text(page, "words"):
        x0, y0, x1, y1 = word[:4]
        ax, ay = point(x0, y0)
        bx, by = point(x1, y1)
        left, right = min(ax, bx), max(ax, bx)
        top, bottom = min(ay, by), max(ay, by)
        words.append(
            ((left + right) / 2, (top + bottom) / 2, word[4], (left, top, right, bottom))
        )

    lines: list[tuple] = []
    rects: list = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            kind = item[0]
            if kind == "l":
                lines.append((point(item[1].x, item[1].y), point(item[2].x, item[2].y)))
            elif kind == "re":
                rectangle = item[1]
                ax, ay = point(rectangle.x0, rectangle.y0)
                bx, by = point(rectangle.x1, rectangle.y1)
                rects.append(
                    pymupdf.Rect(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                )

    frame_width, frame_height = (height, width) if rot in (90, 270) else (width, height)
    return words, lines, rects, (frame_width, frame_height)


def geometry_is_safe(vector_ops: int) -> bool:
    """Spec §3.3 DoS guard: is bulk ``get_drawings()`` safe on this page?"""
    return vector_ops <= VECTOR_OP_GUARD


def devanagari_count(text: str) -> int:
    """Count Devanagari characters. Ported from ``v2proto/v2_extract.py::DEV``.

    Included here because it is a pure text helper with no other home; the
    Stage-11 language gate consumes it. It performs no classification itself.
    """
    return sum(1 for char in text if "\u0900" <= char <= "\u097f")


__all__ = [
    "page_memo",
    "page_text",
    "MOJIBAKE_RE",
    "NATIVE_CHAR_FLOOR",
    "ORIENTATION_CONF_GATE",
    "ORIENTATION_LABELS",
    "Orientation",
    "PAGE_CLASSES",
    "VECTOR_OP_GUARD",
    "classify_page",
    "devanagari_count",
    "detect_orientation",
    "geometry_is_safe",
    "page_is_blank",
    "reading_frame_items",
    "reading_rotation",
    "vector_ops_of",
]
