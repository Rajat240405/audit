"""Core-text construction: the production ``answer_text`` for INCOIS documents.

This is the layer that turns a V2 extraction result into the text that actually
replaces the legacy extraction in ``corpus_reports.jsonl``. It is deliberately
narrow and deliberately different from ``marked.txt``:

    marked.txt   full sidecar view — text + TABLE blocks + FIGURE references.
                 Diagnostic/experimental.
    core text    production view — text + TABLE blocks. **No figure content.**

Keeping the two separate is what makes the experimental half switchable: figure
cards and crops are captured in the same pass and stored in sidecars, but they
never reach the corpus record while ``V2_EXPERIMENTAL_INDEX`` is false. Removing
them later would mean re-extracting; capturing them now and excluding them here
means enabling experimental retrieval later needs no further PDF work.

What the core text contains:

* page-tagged native text (``[p12] …``), so page provenance survives into
  retrieval and evidence;
* OCR text on pages that needed it, in the same position as native text;
* normal extracted tables rendered as markdown grids under an explicit
  ``TABLE`` marker, so cell structure is not flattened away.

What it never contains: FIGURE lines, crop paths, figure cards, or any
experimental visual representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Marker prefixing a table block in the core text.
TABLE_MARKER = "TABLE"
#: Marker that would indicate figure content leaked into the core text. Tests
#: assert this never appears.
FIGURE_MARKER = "FIGURE"


def build_core_text(
    pages: list[Any],
    tables: list[Any],
    page_texts: dict[int, str],
    *,
    include_tables: bool = True,
    doc_id: str | None = None,
) -> str:
    """Assemble the production core text.

    ``pages`` are ``PageRecord``-like objects (need ``page``), ``tables`` are
    ``TableBlock``-like objects (need ``page``, ``table_id``, ``caption``,
    ``markdown``), and ``page_texts`` maps 1-based page number to that page's
    recovered text (native or OCR).

    Continuation-merged blocks are included: they are normal tables whose rows
    span pages, which is core content, not experimental content.
    """
    tables_by_page: dict[int, list[Any]] = {}
    if include_tables:
        for block in tables:
            tables_by_page.setdefault(int(_attr(block, "page") or 0), []).append(block)

    out: list[str] = []
    for record in pages:
        pno = int(_attr(record, "page") or 0)
        for line in (page_texts.get(pno) or "").splitlines():
            stripped = line.rstrip()
            if stripped.strip():
                out.append(f"[p{pno}] {stripped}")

        for block in tables_by_page.get(pno, []):
            markdown = str(_attr(block, "markdown") or "").strip()
            if not markdown:
                continue
            caption = _attr(block, "caption")
            header = f"[p{pno}] {TABLE_MARKER} {_attr(block, 'table_id') or ''}"
            if caption:
                header += f" — {caption}"
            out.append(header.rstrip())
            out.extend(f"[p{pno}] {ln}" for ln in markdown.splitlines() if ln.strip())

    return "\n".join(out) + "\n" if out else ""


def core_text_from_result(result: Any, *, include_tables: bool = True) -> str:
    """Convenience wrapper over a :class:`V2DocumentResult`."""
    return build_core_text(
        result.pages,
        result.tables,
        _page_texts_of(result),
        include_tables=include_tables,
        doc_id=getattr(result, "doc_id", None),
    )


def _page_texts_of(result: Any) -> dict[int, str]:
    """Recover per-page text from a result.

    ``V2DocumentResult`` keeps ``marked_text`` rather than a per-page dict, so
    the page text is re-derived from its ``[pN] `` tags. Table and figure lines
    are dropped here and re-added from the structured blocks, which is what
    guarantees no figure content can survive into the core text.
    """
    texts: dict[int, list[str]] = {}
    for line in (getattr(result, "marked_text", "") or "").splitlines():
        if not line.startswith("[p"):
            continue
        try:
            head, _, body = line.partition("] ")
            pno = int(head[2:])
        except (ValueError, IndexError):
            continue
        body_stripped = body.strip()
        # Drop sidecar-structural lines; the structured blocks are authoritative.
        if body_stripped.startswith((TABLE_MARKER, FIGURE_MARKER)):
            continue
        if body_stripped.startswith("(") and "->" in body_stripped:
            continue
        texts.setdefault(pno, []).append(body)
    return {pno: "\n".join(lines) for pno, lines in texts.items()}


def core_text_stats(text: str) -> dict[str, int]:
    """Cheap shape metrics used by the reprocessing audit."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "chars": len(text),
        "lines": len(lines),
        "pages": len({ln.partition("]")[0] for ln in lines if ln.startswith("[p")}),
        "table_blocks": sum(1 for ln in lines if f"] {TABLE_MARKER} " in ln),
        "figure_lines": sum(1 for ln in lines if f"] {FIGURE_MARKER} " in ln),
    }


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "FIGURE_MARKER",
    "TABLE_MARKER",
    "build_core_text",
    "core_text_from_result",
    "core_text_stats",
]
