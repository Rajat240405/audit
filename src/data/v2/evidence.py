"""Stage 11 — V2 evidence rendering.

Turns table blocks and figure cards into the text that reaches a generator.
This is the half of V2 that makes extraction useful: a table recovered perfectly
but rendered as an unlabelled blob of numbers is not evidence.

Two rules matter:

* **Never flatten.** A table is rendered as a labelled markdown grid so cell
  structure survives into the prompt. Flattening is precisely what the current
  path does and what loses the association between a row label and its values.
* **Never invent.** A figure card renders what was actually extracted — caption,
  kind, nearby context, OCR excerpt. If a crop exists it is named as a path for
  provenance; no description is synthesised for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.v2 import borderless_extract, language_gate

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Characters reserved for a single evidence block. Long tables are truncated
#: with an explicit marker rather than silently cut mid-cell.
MAX_TABLE_CHARS = 2400
MAX_FIGURE_CHARS = 900
TRUNCATION_MARK = "…[truncated]"


def render_table_evidence(block: dict | Any) -> str:
    """Render one table block as labelled evidence text."""
    data = _as_dict(block)
    parts: list[str] = []

    caption = data.get("caption")
    label = caption or f"Table on page {data.get('page', '?')}"
    parts.append(f"TABLE — {label}")

    method = str(data.get("method") or "")
    tags: list[str] = [f"page {data.get('page', '?')}"]
    if data.get("ncols") is not None:
        tags.append(f"{data['ncols']} cols x {data.get('nrows', '?')} rows")
    if method:
        tags.append(f"extracted: {method}")
    orientation = data.get("orientation")
    if orientation:
        tags.append(f"orientation: {orientation} (conf {data.get('orientation_conf')})")
    if borderless_extract.is_borderless(data):
        tags.append("EXPERIMENTAL borderless detection — verify before citing")
    parts.append("(" + "; ".join(tags) + ")")

    continuation = data.get("continuation")
    if continuation:
        pages = continuation.get("pages") or []
        parts.append(f"(continuation merged across pages {pages})")

    markdown = str(data.get("markdown") or "")
    if markdown:
        parts.append(_clip(markdown, MAX_TABLE_CHARS))
    else:
        parts.append("(no grid recovered)")

    return "\n".join(parts)


def render_figure_evidence(card: dict | Any) -> str:
    """Render one figure card as labelled evidence text.

    The card is the evidence. There is no vision model in this path; see
    :mod:`src.data.v2.vision_hook` for the optional, disabled-by-default
    augmentation and its hard fallback back to exactly this text.
    """
    data = _as_dict(card)
    parts: list[str] = []

    caption = data.get("caption")
    kind = data.get("kind") or "figure"
    if caption:
        header = f"FIGURE — {caption}"
    else:
        header = f"FIGURE — {kind} on page {data.get('page', '?')}"
    parts.append(header)

    tags = [f"page {data.get('page', '?')}", f"kind: {kind}"]
    parts.append("(" + "; ".join(tags) + ")")

    context = data.get("context") or []
    if context:
        parts.append("nearby text: " + " / ".join(str(c) for c in context[:3]))

    ocr_text = data.get("ocr_text")
    if ocr_text:
        parts.append("ocr: " + _clip(str(ocr_text), MAX_FIGURE_CHARS))

    crop = data.get("crop")
    if crop:
        # Provenance only. Naming the crop lets a human verify; it does not
        # imply the model saw it.
        parts.append(f"crop: {crop}")

    if not caption and not context and not ocr_text:
        parts.append("(no caption or text recovered — region only)")

    return "\n".join(parts)


def render_evidence(
    units: list[dict | Any],
    *,
    apply_language_gate: bool = True,
    config: Any = None,
) -> tuple[str, list[dict]]:
    """Render a mixed list of table/figure units into one evidence block.

    Returns ``(text, dropped)`` where ``dropped`` records what the language gate
    removed and why, so filtering is auditable rather than invisible.
    """
    blocks: list[str] = []
    dropped: list[dict] = []

    for unit in units:
        data = _as_dict(unit)
        kind = str(data.get("kind") or data.get("method") or "")
        if "figure" in kind or data.get("card") is not None:
            rendered = render_figure_evidence(data)
        else:
            rendered = render_table_evidence(data)

        if apply_language_gate:
            reason = language_gate.gate_reason(rendered, config)
            if reason:
                dropped.append(
                    {
                        "id": data.get("table_id") or data.get("figure_id") or "?",
                        "reason": reason,
                    }
                )
                continue

        blocks.append(rendered)

    return ("\n\n".join(blocks) + "\n") if blocks else "", dropped


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATION_MARK)].rstrip() + TRUNCATION_MARK


def _as_dict(obj: Any) -> dict:
    """Accept a plain dict, a dataclass, or a pydantic model."""
    if isinstance(obj, dict):
        return obj
    for attr in ("to_dict", "model_dump"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return dict(method())
            except Exception:  # noqa: BLE001 - fall through to vars()
                pass
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


__all__ = [
    "MAX_FIGURE_CHARS",
    "MAX_TABLE_CHARS",
    "render_evidence",
    "render_figure_evidence",
    "render_table_evidence",
]
