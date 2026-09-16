"""Stage 8 — V2 child units built from sidecars.

A child unit is the retrievable text for one table block or one figure card. The
parent stays the document-level record; children reference it by
``parent_doc_id``. This mirrors the mechanism the retrieval pipeline already uses
for long chunks (``_long_chunk_map``), so no parallel retrieval stack is
introduced.

Child ids are deterministic and parse back to their source:

    {doc}#t{n}          nth table block in the document
    {doc}#p{page}f{i}   ith figure on a page
    {doc}#p{page}pagefig  the page-as-figure card

Child units are **additive evidence**: a table's page text stays searchable in
the parent record. Children make table and figure content retrievable at
row/caption granularity, which parent-level collapse otherwise hides.
"""

from __future__ import annotations

import re
from typing import Any

from src.data.v2.config import V2Config
from src.data.v2.sidecar_store import FigureCard, SidecarStore, TableBlock

#: Matches ``{doc}#t{n}``, ``{doc}#p{page}f{i}``, ``{doc}#p{page}pagefig``.
CHILD_ID_RE = re.compile(
    r"^(?P<doc>.+)#(?:"
    r"t(?P<table>\d+)(?P<cont>\+cont)?|"
    r"p(?P<page>\d+)(?:f(?P<fig>\d+)|pagefig)"
    r")$"
)

KIND_TABLE = "table"
KIND_TABLE_CONT = "table_continuation"
KIND_FIGURE = "figure"
KIND_PAGE_FIGURE = "page_figure"


def parse_child_id(child_id: str) -> dict[str, Any] | None:
    """Parse a child id, or return ``None`` if it is not a V2 child id."""
    match = CHILD_ID_RE.match(child_id or "")
    if not match:
        return None
    parts = match.groupdict()
    doc = parts["doc"]
    if parts["table"] is not None:
        kind = KIND_TABLE_CONT if parts["cont"] else KIND_TABLE
        return {"kind": kind, "doc": doc, "index": int(parts["table"]), "page": None}
    if child_id.endswith("pagefig"):
        return {"kind": KIND_PAGE_FIGURE, "doc": doc, "page": int(parts["page"]), "index": None}
    return {"kind": KIND_FIGURE, "doc": doc, "page": int(parts["page"]),
            "index": int(parts["fig"])}


def _table_text(block: TableBlock) -> str:
    """Table child text: caption first, then the grid.

    The caption carries the table number and title, which is what most queries
    actually contain. The markdown grid follows so cell values are matchable.
    """
    parts: list[str] = []
    if block.caption:
        parts.append(block.caption)
    parts.append(block.markdown or "")
    return "\n".join(part for part in parts if part).strip()


def _figure_text(card: FigureCard) -> str:
    """Figure child text: caption + kind + nearby context + OCR excerpt.

    A crop is deliberately **not** embedded as an image or described by a vision
    model here — that would put a model in the ingestion path. Retrieval matches
    on text; the crop path is provenance for a human.
    """
    parts: list[str] = []
    if card.caption:
        parts.append(card.caption)
    if card.kind:
        parts.append(f"[{card.kind}]")
    if card.context:
        parts.append(" ".join(card.context))
    if card.ocr_text:
        parts.append(card.ocr_text[:600])
    return " ".join(part for part in parts if part).strip()


def build_child_units(
    config: V2Config | None = None,
    doc_ids: list[str] | None = None,
    *,
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Build child units for one document or every sidecar on disk.

    Returns a list of plain dicts so this module has no dependency on
    ``QARecord`` and is trivially testable:

        {"id", "doc", "page", "kind", "text", "meta"}

    Units longer than ``max_chars`` are truncated with a marker rather than
    dropped, so a pathological table still contributes evidence.
    """
    cfg = config if config is not None else V2Config()
    limit = max_chars if max_chars is not None else cfg.child_max_chars
    root = cfg.resolved_sidecar_dir()

    if doc_ids is None:
        if not root.is_dir():
            return []
        doc_ids = sorted(p.name for p in root.iterdir() if p.is_dir())

    units: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        store = SidecarStore(doc_id, cfg)
        if not store.pages_path.is_file():
            continue
        page_of = {record.page: record.page_class for record in store.read_pages()}

        for block in store.read_tables():
            text = _table_text(block)
            if not text:
                continue
            kind = KIND_TABLE_CONT if block.continuation else KIND_TABLE
            units.append(
                {
                    "id": block.table_id,
                    "doc": doc_id,
                    "page": block.page,
                    "kind": kind,
                    "text": _clip(text, limit),
                    "meta": {
                        "caption": block.caption,
                        "method": block.method,
                        "ncols": block.ncols,
                        "nrows": block.nrows,
                        "orientation": block.orientation,
                        "continuation": block.continuation,
                    },
                }
            )

        for card in store.read_figures():
            text = _figure_text(card)
            if not text:
                continue
            units.append(
                {
                    "id": card.figure_id,
                    "doc": doc_id,
                    "page": card.page,
                    "kind": (
                        KIND_PAGE_FIGURE
                        if "page-as-figure" in (card.kind or "")
                        else KIND_FIGURE
                    ),
                    "text": _clip(text, limit),
                    "meta": {
                        "caption": card.caption,
                        "kind": card.kind,
                        "crop": card.crop,
                        "page_class": page_of.get(card.page),
                    },
                }
            )

    return units


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def child_unit_stats(units: list[dict[str, Any]]) -> dict[str, int]:
    """Counts by kind — what a migration run reports."""
    stats: dict[str, int] = {"total": len(units)}
    for unit in units:
        stats[unit["kind"]] = stats.get(unit["kind"], 0) + 1
    return stats


def sidecar_doc_ids(config: V2Config | None = None) -> list[str]:
    """Every document that already has sidecars (for backfill reporting)."""
    cfg = config if config is not None else V2Config()
    root = cfg.resolved_sidecar_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "pages.jsonl").is_file())


__all__ = [
    "CHILD_ID_RE",
    "KIND_FIGURE",
    "KIND_PAGE_FIGURE",
    "KIND_TABLE",
    "KIND_TABLE_CONT",
    "build_child_units",
    "build_v2_records",
    "child_unit_stats",
    "parse_child_id",
    "sidecar_doc_ids",
]


# ── Stage 8: parent records for the separate V2 index ────────────────────────
#
# The sidecar is keyed by PDF stem, but corpus ``question_id`` is a content hash
# (``convert_sirs_knowledge.py::_hash_id(text, "incdoc")``) and
# ``QARecordMetadata.source`` holds only ``"incois"`` — there is no stored link
# from a sidecar to a corpus row. Until ingestion is wired to stamp the real
# ``question_id`` into ``doc.json``, the V2 child index is therefore built as a
# SELF-CONTAINED index: one synthesised parent record per sidecar document, with
# child units collapsing to it. This keeps parent collapse, metadata filtering
# and reranking semantics intact inside the V2 index without ever touching the
# production index or guessing a mapping.


def build_v2_records(
    config: V2Config | None = None,
    doc_ids: list[str] | None = None,
) -> list[Any]:
    """One parent :class:`QARecord` per sidecar document.

    ``question_id`` is the sidecar doc id; ``answer_text`` is ``marked.txt``, so
    the parent carries the page-tagged document text with explicit TABLE/FIGURE
    blocks and the children carry individual tables and figures.
    """
    import json as _json

    from src.models.qa_record import QARecord, QARecordMetadata

    cfg = config if config is not None else V2Config()
    if doc_ids is None:
        doc_ids = sidecar_doc_ids(cfg)

    records: list[Any] = []
    for doc_id in doc_ids:
        store = SidecarStore(doc_id, cfg)
        if not store.pages_path.is_file():
            continue
        marked = store.read_marked()
        if not marked.strip():
            continue

        summary: dict = {}
        if store.doc_path.is_file():
            try:
                summary = _json.loads(store.doc_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a bad doc.json must not stop a build
                summary = {}

        # A future ingestion hook can stamp the real corpus id here; until then
        # the sidecar id is the parent.
        record_id = str(summary.get("record_id") or doc_id)
        document_type = str(summary.get("document_type") or "incois")

        records.append(
            QARecord(
                question_id=record_id,
                question_text=f"INCOIS V2 document: {doc_id}",
                answer_text=marked,
                metadata=QARecordMetadata(
                    document_type=document_type,
                    org="incois",
                    source="incois",
                ),
            )
        )
    return records
