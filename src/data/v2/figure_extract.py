"""Stage 6 — figure extraction and strict caption association.

The funnel is the Phase-8/9 one, re-validated in Phase 9: raw raster objects →
drop below ``min_fig_px`` → adjacency merge (tile shards sit side by side) →
36-dpi content-hash dedup (logos repeat across pages) → candidate regions.
Measured aggregate: 118,576 raw objects → 302 candidates (0.25%); a tile page
with 4,073 objects and a 1×2px median collapses to one card.

**Caption association is strict-only.** The Phase-8 nearest-"Fig"-word heuristic
is deliberately **not** implemented: it produced 31/211 (14.7%) body-sentence
false captures such as "Figure 2 shows the spatial distribution…". The strict
rule requires punctuation immediately after the figure number, which alone
removes every one of those (spec §3.6, measured 0 violations, 6/6 hand-verified
cases). ``V2_CAPTION_STRICT=false`` therefore means *assign no captions*, never
"fall back to the broken heuristic".

A region with no qualifying caption line gets **no** caption. Captions are never
guessed (spec §3.6, requirement 7).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: A caption candidate line must START with a figure keyword, a number, and then
#: punctuation. The trailing ``[:.\-–—|]`` class is what kills the body-sentence
#: false captures; ``|`` covers journal style (``Fig. 15 | …``).
CAPTION_STRICT_RE = re.compile(
    r"^\s*(Fig(?:ure)?|Map|Plate|Chart)\.?\s*\d+\s*[:.\-–—|]", re.IGNORECASE
)

#: Minimum x-overlap fraction between caption and region.
CAPTION_OVERLAP_FLOOR = 0.25
#: Maximum vertical gap (pt) between caption edge and region edge.
CAPTION_GAP_CEILING = 150.0
#: Captions longer than this are body prose, not captions.
CAPTION_MAX_CHARS = 220
#: Merge artifacts narrower/shorter than this are dropped.
MIN_REGION_SIDE = 60.0
#: Regions at least this fraction of page width may take a caption from either
#: column zone.
FULL_WIDTH_FRACTION = 0.8
#: A crop smaller than this on either side is not written.
MIN_CROP_SIDE = 8.0

_KIND_MAP_HINT_RE = re.compile(r"map| GIS |spatial", re.IGNORECASE)


def figure_regions(
    page: Any,
    config: V2Config | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[list, dict[str, int]]:
    """Raw → size filter → adjacency merge → dedup.

    Returns ``(candidate_rects, funnel_counts)``. ``seen_hashes`` is supplied by
    the caller (one set per document) so dedup spans pages without a module
    global; pass ``None`` to disable cross-page dedup.
    """
    import pymupdf

    cfg = config if config is not None else V2Config()
    if seen_hashes is None:
        seen_hashes = set()

    infos = page.get_image_info(xrefs=True) or []
    raw = len(infos)
    big = [
        item
        for item in infos
        if item["width"] >= cfg.min_fig_px or item["height"] >= cfg.min_fig_px
    ]
    tiny_removed = raw - len(big)

    boxes = [pymupdf.Rect(item["bbox"]) for item in big]
    merged = True
    while merged:
        merged = False
        out: list = []
        while boxes:
            current = boxes.pop()
            for index, other in enumerate(boxes):
                grown = pymupdf.Rect(
                    current.x0 - cfg.fig_merge_gap,
                    current.y0 - cfg.fig_merge_gap,
                    current.x1 + cfg.fig_merge_gap,
                    current.y1 + cfg.fig_merge_gap,
                )
                if grown.intersects(other):
                    current |= other
                    boxes.pop(index)
                    merged = True
                    boxes.append(current)
                    break
            else:
                out.append(current)
        boxes = out

    candidates: list = []
    dups = 0
    for rect in boxes:
        if rect.width < MIN_REGION_SIDE or rect.height < MIN_REGION_SIDE:
            continue
        try:
            pixmap = page.get_pixmap(clip=rect, dpi=36)
            digest = hashlib.md5(pixmap.samples).hexdigest()
        except Exception:  # noqa: BLE001 - an unrenderable region is still a candidate
            digest = f"norender:{rect.x0:.1f},{rect.y0:.1f},{rect.width:.1f}"
        if digest in seen_hashes:
            dups += 1
            continue
        seen_hashes.add(digest)
        candidates.append(rect)

    funnel = {
        "raw_objects": raw,
        "tiny_removed": tiny_removed,
        "after_merge": len(boxes),
        "dups_removed": dups,
        "candidates": len(candidates),
    }
    return candidates, funnel


def classify_figure(rect: Any, page_rect: Any, page_text: str) -> str:
    """Coarse kind label from aspect ratio and page-area fraction."""
    fraction = (rect.width * rect.height) / max(
        1.0, page_rect.width * page_rect.height
    )
    if fraction > 0.75:
        kind = "page-figure"
    elif rect.width > rect.height * 1.6:
        kind = "chart/banner"
    elif rect.height > rect.width * 1.4:
        kind = "photo/panel"
    else:
        kind = "figure"
    if _KIND_MAP_HINT_RE.search((page_text or "")[:4000]):
        kind = "map?"
    return kind


def page_lines(page: Any) -> list[tuple[Any, str]]:
    """Horizontal text lines as ``(bbox, text)``.

    Only ``dir == (1.0, 0.0)`` lines are considered: rotated table text must not
    compete for figure captions.
    """
    import pymupdf

    out: list[tuple[Any, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if tuple(line["dir"]) != (1.0, 0.0):
                continue
            text = " ".join(span["text"] for span in line["spans"]).strip()
            if text:
                out.append((pymupdf.Rect(line["bbox"]), text))
    return out


def column_zones(page: Any, lines: list[tuple[Any, str]]) -> tuple[int, int | None]:
    """Detect a 2-column layout from an x-histogram valley in the middle third."""
    xs = [rect.x0 for rect, _text in lines]
    if len(xs) < 20:
        return 1, None
    bins = [0] * (int(page.rect.width // 4) + 2)
    for x in xs:
        bins[int(x // 4)] += 1
    middle = range(len(bins) // 3, 2 * len(bins) // 3)
    if not middle:
        return 1, None
    lowest = min(middle, key=lambda i: bins[i])
    if bins[lowest] <= max(bins) // 8 and max(bins) > 6:
        return 2, lowest * 4
    return 1, None


def associate_captions(
    page: Any, regions: list, config: V2Config | None = None
) -> dict[int, str]:
    """Punctuation-anchored, layout-scored, uniquely-greedy caption assignment.

    Returns ``{region_index: caption_text}``. Regions with no qualifying caption
    are absent from the mapping — captions are never guessed.

    With ``V2_CAPTION_STRICT=false`` this returns an empty mapping rather than
    falling back to the legacy heuristic (see module docstring).
    """
    cfg = config if config is not None else V2Config()
    if not cfg.figures_active or not cfg.caption_strict:
        return {}

    lines = page_lines(page)
    candidates = [
        (rect, text)
        for rect, text in lines
        if CAPTION_STRICT_RE.match(text) and len(text) <= CAPTION_MAX_CHARS
    ]
    zones, boundary = column_zones(page, lines)

    scored: list[tuple] = []
    for caption_index, (caption_rect, caption_text) in enumerate(candidates):
        for region_index, region in enumerate(regions):
            overlap = max(
                0.0,
                min(caption_rect.x1, region.x1) - max(caption_rect.x0, region.x0),
            ) / max(1.0, min(caption_rect.width, region.width))
            if overlap < CAPTION_OVERLAP_FLOOR:
                continue

            if caption_rect.y0 >= region.y1:  # caption below
                gap = caption_rect.y0 - region.y1
            elif caption_rect.y1 <= region.y0:  # caption above
                gap = region.y0 - caption_rect.y1
            else:
                gap = 0.0
            if gap > CAPTION_GAP_CEILING:
                continue

            # column guard: in a two-column layout a narrow region may only take
            # a caption from its own zone
            if (
                zones == 2
                and region.width < FULL_WIDTH_FRACTION * page.rect.width
                and (caption_rect.x0 < boundary) != (region.x0 < boundary)
            ):
                continue

            score = overlap * (1 - gap / CAPTION_GAP_CEILING)
            scored.append((score, -gap, caption_index, region_index, caption_text))

    # Best score first; ties break on smaller gap, then caption order — fully
    # deterministic, so repeated runs assign identically.
    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    used_captions: set[int] = set()
    used_regions: set[int] = set()
    assignment: dict[int, str] = {}
    for _score, _neg_gap, caption_index, region_index, caption_text in scored:
        if caption_index in used_captions or region_index in used_regions:
            continue
        used_captions.add(caption_index)
        used_regions.add(region_index)
        assignment[region_index] = caption_text
    return assignment


def nearby_context(
    lines: list[tuple[Any, str]], region: Any, limit: int = 3, max_gap: float = 120.0
) -> list[str]:
    """Up to ``limit`` nearest text lines above/below a region."""
    picked: list[tuple[float, str]] = []
    for rect, text in lines:
        if rect.y1 <= region.y0:
            gap = region.y0 - rect.y1
        elif rect.y0 >= region.y1:
            gap = rect.y0 - region.y1
        else:
            continue
        if gap <= max_gap:
            picked.append((gap, text))
    picked.sort(key=lambda item: (item[0], item[1]))
    return [text for _gap, text in picked[:limit]]


def build_figure_cards(
    page: Any,
    pno: int,
    regions: list,
    page_text: str,
    doc_id: str,
    config: V2Config | None = None,
    crop_writer: Any = None,
) -> tuple[list[dict], int]:
    """Turn candidate regions into FigureCard dicts.

    ``crop_writer(rect, index) -> path | None`` is injected by the pipeline so
    this module never touches the filesystem. Returns ``(cards, crops_made)``.
    """
    cfg = config if config is not None else V2Config()
    if not cfg.figures_active:
        return [], 0

    assignment = associate_captions(page, regions, cfg)
    lines = page_lines(page)
    cards: list[dict] = []
    crops_made = 0

    for index, region in enumerate(regions):
        caption = assignment.get(index)
        kind = classify_figure(region, page.rect, page_text)
        context = nearby_context(lines, region)

        crop_path = None
        big_enough = region.width >= MIN_CROP_SIDE and region.height >= MIN_CROP_SIDE
        if crop_writer is not None and big_enough:
            crop_path = crop_writer(region, index)
            if crop_path:
                crops_made += 1

        section = _section_hint(page_text)
        card_text = " | ".join(
            part
            for part in (
                f"Figure on page {pno + 1}",
                f"caption: {caption}" if caption else None,
                f"kind: {kind}",
                f"section: {section}" if section else None,
                f"context: {' / '.join(context)}" if context else None,
            )
            if part
        )

        cards.append(
            {
                "figure_id": f"{doc_id}#p{pno + 1}f{index + 1}",
                "doc": doc_id,
                "page": pno + 1,
                "bbox": [region.x0, region.y0, region.x1, region.y1],
                "kind": kind,
                "caption": caption,
                "context": context,
                "crop": crop_path,
                "ocr_text": None,
                "card": card_text,
            }
        )

    return cards, crops_made


def page_figure_card(
    doc_id: str,
    pno: int,
    page_class: str,
    ocr_excerpt: str | None,
    crop_path: str | None,
) -> dict:
    """A page-as-figure card for whole-page scans/vector pages."""
    kind = f"page-as-figure({page_class})"
    card_text = f"{kind} on page {pno + 1}"
    if ocr_excerpt:
        card_text += f" | ocr: {ocr_excerpt[:300]}"
    return {
        "figure_id": f"{doc_id}#p{pno + 1}pagefig",
        "doc": doc_id,
        "page": pno + 1,
        "bbox": None,
        "kind": kind,
        "caption": None,
        "context": [],
        "crop": crop_path,
        "ocr_text": ocr_excerpt,
        "card": card_text,
    }


def _section_hint(page_text: str) -> str | None:
    """First ALL-CAPS line as a coarse section label. Never guessed beyond that."""
    for line in (page_text or "").splitlines()[:25]:
        stripped = line.strip()
        if len(stripped) >= 4 and stripped.isupper() and stripped.isalpha():
            return stripped
    return None


__all__ = [
    "CAPTION_GAP_CEILING",
    "CAPTION_MAX_CHARS",
    "CAPTION_OVERLAP_FLOOR",
    "CAPTION_STRICT_RE",
    "associate_captions",
    "build_figure_cards",
    "classify_figure",
    "column_zones",
    "figure_regions",
    "nearby_context",
    "page_figure_card",
    "page_lines",
]
