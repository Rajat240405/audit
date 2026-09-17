"""Stage 5 — the V2 extraction pipeline (spec §2 stages S1-S7).

Orchestrates the router, table, OCR and figure components into one document
result, and writes the sidecars. Deliberately knows nothing about retrieval,
indexing, or generation — those consume the sidecars through
``src/data/v2/child_units.py``.

Stage order (spec §2):

    S1 page classification + orientation
    S2 routing: native keeps get_text; scan/vector/mojibake go to OCR under the
       per-document page budget, with honest skip marking
    S3 tables: portrait lattice, then rotated lattice which SUPERSEDES the
       portrait block on the same page
    S4 continuation merge (additive — child page blocks are kept)
    S5 figures: funnel -> strict captions -> cards -> deterministic crops
    S6 marked text with explicit page/table/figure structure
    S7 sidecars

Two invariants worth stating:

* Pages are fetched as ``doc[pno]`` **inside** the loop. PyMuPDF invalidates
  previously returned ``Page`` objects when later pages are edited, so holding
  references across iterations raises ``page is None`` (found in Stage 1).
* Nothing is written when ``config.enabled`` is false and ``require_enabled`` is
  left at its default, so an accidental call from a V2-off path is a no-op.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.v2 import figure_extract, ocr, page_router, table_extract
from src.data.v2.config import V2Config
from src.data.v2.sidecar_store import (
    FigureCard,
    PageRecord,
    SidecarStore,
    TableBlock,
    build_marked_text,
    validate_sidecars,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Page classes whose native text layer is usable as-is.
NATIVE_CLASSES = ("native", "native_low")
#: Page classes that need OCR to recover any text.
OCR_CLASSES = ("scan", "vector", "mojibake")
#: Classes that also get a page-as-figure card.
PAGE_FIGURE_CLASSES = ("scan", "vector")


@dataclass
class V2DocumentResult:
    """Everything one document produced."""

    doc_id: str
    pages: list[PageRecord] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    figures: list[FigureCard] = field(default_factory=list)
    marked_text: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    skipped_reason: str | None = None

    @property
    def extracted(self) -> bool:
        return self.skipped_reason is None and bool(self.pages)


def extract_document(
    pdf_path: Path | str,
    config: V2Config | None = None,
    *,
    write: bool = True,
    require_enabled: bool = True,
    crop_budget: int | None = None,
    record_id: str | None = None,
) -> V2DocumentResult:
    """Run S1-S7 over one PDF.

    ``write=False`` returns the result without touching the filesystem (used by
    the differential tests and the comparison tooling). ``require_enabled=False``
    lets the explicit opt-in reindex tool run with the master flag off; every
    other caller leaves it at the default.
    """
    import pymupdf

    cfg = config if config is not None else V2Config()
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem

    if require_enabled and not cfg.core_extraction_active:
        return V2DocumentResult(doc_id=doc_id, skipped_reason="extraction_disabled")

    started = time.perf_counter()
    store = SidecarStore(doc_id, cfg)
    budget = ocr.OcrBudget(cfg.ocr_page_cap, cfg.ocr_wall_seconds)
    crops_made = 0
    crop_limit = crop_budget if crop_budget is not None else cfg.fig_max_crops
    seen_hashes: set[str] = set()
    guard_skips = 0

    document = pymupdf.open(str(pdf_path))
    try:
        page_records: list[PageRecord] = []
        page_texts: dict[int, str] = {}
        raw_blocks: list[dict] = []
        figure_dicts: list[dict] = []
        funnels: list[dict] = []
        table_counter = 0

        # ── Phase 1 — classify, and render the pages that need OCR ────────────
        #
        # EVERY PyMuPDF access in this pipeline happens here, serially, on this
        # thread. When the OCR pool is engaged the rendered PIL images are handed
        # to workers that never receive a Document/Page/Pixmap, so MuPDF is never
        # entered concurrently and the uncontainable-SIGSEGV failure class cannot
        # arise. This is the same structural rule src/data/extract_dfg.py uses.
        #
        # A page cap or wall-clock limit is inherently sequential (the budget
        # check depends on how many pages have already succeeded), so the pool is
        # bypassed in that case and the original interleaved path runs unchanged.
        parallel = (
            cfg.ocr_workers > 0
            and budget.page_cap == 0
            and budget.wall_seconds == 0.0
        )
        prepared: list[dict] = []
        pending: list[tuple[int, Any]] = []
        for pno in range(document.page_count):
            page = document[pno]  # fetched per iteration — see module docstring
            vector_ops = page_router.vector_ops_of(page)
            page_class, meta = page_router.classify_page(page, vector_ops, cfg)
            orientation = page_router.detect_orientation(page)

            text = ""
            ocr_status = ""
            needs_ocr = page_class in OCR_CLASSES
            if page_class in NATIVE_CLASSES or page_class == "empty":
                text = page_router.page_text(page, "text") or ""
                needs_ocr = False
            elif needs_ocr and parallel:
                image, prep_status = ocr.prepare_image(page, cfg)
                if image is None:
                    ocr_status = prep_status or "error:unknown"
                    if ocr_status == "disabled":
                        budget.record_skipped("disabled")
                    needs_ocr = False
                else:
                    pending.append((pno, image))
            prepared.append(
                {
                    "vector_ops": vector_ops,
                    "page_class": page_class,
                    "meta": meta,
                    "orientation": orientation,
                    "text": text,
                    "ocr_status": ocr_status,
                    "needs_ocr": needs_ocr,
                }
            )

        # ── Phase 2 — OCR. Workers see PIL images only, never PyMuPDF. ────────
        # Results are keyed by page number, so Phase 3 restores the original
        # page order regardless of the order in which workers finished.
        ocr_texts = ocr.ocr_prepared(pending, cfg) if parallel else {}

        # ── Phase 3 — tables, figures and records, strictly in page order ─────
        for pno in range(document.page_count):
            page = document[pno]  # fetched per iteration — see module docstring
            info = prepared[pno]
            vector_ops = info["vector_ops"]
            page_class = info["page_class"]
            meta = info["meta"]
            orientation = info["orientation"]

            text = info["text"]
            ocr_used = False
            ocr_status = info["ocr_status"]
            if info["needs_ocr"]:
                if parallel:
                    text, ocr_status = ocr_texts[pno]
                else:
                    limit = budget.check()
                    if not cfg.ocr_active:
                        # Still an incompleteness: pages that need OCR did not
                        # get it. Labelled and counted, never silently
                        # "complete".
                        ocr_status = "disabled"
                        budget.record_skipped("disabled")
                    elif limit is not None:
                        # Explicit and auditable: the page is labelled with WHY
                        # it was skipped and the document is flagged incomplete
                        # below.
                        budget.record_skipped(limit)
                        ocr_status = f"skipped({limit})"
                    else:
                        text, ocr_status = ocr.ocr_page(page, cfg)
                if ocr_status == "ok":
                    ocr_used = True
                    budget.record_used()

            route = page_class
            if page_class in OCR_CLASSES:
                route = page_class + ocr.ocr_suffix(ocr_used, ocr_status.startswith("skipped("))

            page_records.append(
                PageRecord(
                    doc=doc_id,
                    page=pno + 1,
                    page_class=page_class,
                    route=route,
                    section_ctx=figure_extract._section_hint(text),  # noqa: SLF001
                    chars=len(text.strip()),
                    moji_ratio=meta["moji_ratio"],
                    n_images=meta["n_images"],
                    vector_ops=meta["vector_ops"],
                    ocr_used=ocr_used,
                    ocr_status=ocr_status,
                    orientation=orientation.label if orientation.rot else None,
                    orientation_conf=orientation.conf if orientation.rot else None,
                )
            )
            page_texts[pno + 1] = text

            # ── S3 tables ────────────────────────────────────────────────────
            if cfg.tables_active:
                if not page_router.geometry_is_safe(vector_ops):
                    guard_skips += 1
                else:
                    blocks, _guarded = table_extract.detect_tables(
                        page, pno, text, cfg, vector_ops=vector_ops
                    )
                    if cfg.rotated_tables_active:
                        rotated, _label, _conf, _rot = table_extract.detect_tables_rotated(
                            page, pno, text, cfg,
                            vector_ops=vector_ops, orientation=orientation,
                        )
                        if rotated:
                            # a rotated block SUPERSEDES the portrait block
                            blocks = rotated
                    for block in blocks:
                        table_counter += 1
                        block["table_id"] = f"{doc_id}#t{table_counter}"
                        block["doc"] = doc_id
                        raw_blocks.append(block)

            # ── S5 figures ───────────────────────────────────────────────────
            if cfg.figures_active:
                regions, funnel = figure_extract.figure_regions(page, cfg, seen_hashes)
                funnels.append(funnel)

                def _write_crop(
                    rect: Any,
                    index: int,
                    _page: int = pno,
                    _store: SidecarStore = store,
                    _src: Any = page,
                ) -> str | None:
                    nonlocal crops_made
                    if crops_made >= crop_limit:
                        return None
                    target = _store.crop_path(_page + 1, index + 1)
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _src.get_pixmap(clip=rect, dpi=cfg.fig_crop_dpi).save(str(target))
                    except Exception:  # noqa: BLE001 - a failed crop is not fatal
                        return None
                    crops_made += 1
                    return str(target)

                cards, made = figure_extract.build_figure_cards(
                    page,
                    pno,
                    regions,
                    text,
                    doc_id,
                    cfg,
                    crop_writer=_write_crop if write else None,
                )
                crops_made += made
                figure_dicts.extend(cards)

                if page_class in PAGE_FIGURE_CLASSES:
                    page_crop = None
                    if write and crops_made < crop_limit:
                        target = store.page_crop_path(pno + 1)
                        try:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            page.get_pixmap(dpi=60).save(str(target))
                            page_crop = str(target)
                            crops_made += 1
                        except Exception:  # noqa: BLE001
                            page_crop = None
                    figure_dicts.append(
                        figure_extract.page_figure_card(
                            doc_id, pno, page_class, text[:600] if ocr_used else None, page_crop
                        )
                    )
    finally:
        document.close()

    # ── S4 continuation merge (additive) ─────────────────────────────────────
    merged, decisions = table_extract.merge_continuations(
        raw_blocks, cfg.table_cont_gap
    )
    tables = [TableBlock.from_dict(block) for block in merged]
    figures = [FigureCard.from_dict(card) for card in figure_dicts]

    # ── S6 marked text ───────────────────────────────────────────────────────
    marked = build_marked_text(doc_id, page_records, tables, figures, page_texts)

    class_counts: dict[str, int] = {}
    for record in page_records:
        class_counts[record.page_class] = class_counts.get(record.page_class, 0) + 1

    summary = {
        "doc": doc_id,
        "file": pdf_path.name,
        "pages": len(page_records),
        "class_counts": class_counts,
        "ocr_pages_used": budget.used,
        "ocr_pages_skipped": budget.skipped,
        #: False means the document is NOT fully OCR'd. Consumers must treat an
        #: incomplete document as partial, never as a finished extraction.
        "ocr_complete": budget.complete,
        "ocr_limit_reason": budget.limit_reason,
        "ocr_budget": budget.as_dict(),
        "guard_skips": guard_skips,
        "tables": len(tables),
        "continuation_merges": sum(1 for d in decisions if d["merge"]),
        "continuation_decisions": len(decisions),
        "figures": len(figures),
        "crops_written": crops_made,
        "figure_funnel": _aggregate_funnel(funnels),
        "secs": round(time.perf_counter() - started, 2),
        "config": {
            k: v
            for k, v in cfg.as_flat_dict().items()
            if k
            in (
                "enabled", "tables", "table_rotation", "table_borderless",
                "figures", "index_children", "vision_at_generation", "page_routing",
                "ocr", "ocr_lang", "ocr_dpi", "ocr_page_cap",
            )
        },
    }

    result = V2DocumentResult(
        doc_id=doc_id,
        pages=page_records,
        tables=tables,
        figures=figures,
        marked_text=marked,
        summary=summary,
    )

    if record_id:
        # Permanent sidecar <-> corpus link (spec: preserve existing identity).
        summary["record_id"] = record_id

    if write:
        store.write_pages(page_records)
        store.write_tables(tables)
        store.write_figures(figures)
        store.write_marked(marked)
        store.write_doc(summary)

    return result


def _aggregate_funnel(funnels: list[dict]) -> dict[str, int]:
    total = {
        "raw_objects": 0,
        "tiny_removed": 0,
        "after_merge": 0,
        "dups_removed": 0,
        "candidates": 0,
    }
    for funnel in funnels:
        for key in total:
            total[key] += funnel.get(key, 0)
    return total


def extract_many(
    pdf_paths: list[Path | str],
    config: V2Config | None = None,
    *,
    write: bool = True,
    on_error: Any = None,
) -> tuple[list[V2DocumentResult], list[dict]]:
    """Extract several documents, never aborting the batch on one failure.

    A single unreadable PDF must not stop a migration run; failures are returned
    in the error list with the document and reason.
    """
    results: list[V2DocumentResult] = []
    errors: list[dict] = []
    for path in pdf_paths:
        try:
            results.append(extract_document(path, config, write=write))
        except Exception as exc:  # noqa: BLE001 - batch resilience
            entry = {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            errors.append(entry)
            if on_error is not None:
                on_error(entry)
    return results, errors


def verify_document(doc_id: str, config: V2Config | None = None) -> tuple[bool, list[str]]:
    """Run the spec §4 provenance check over one document's sidecars."""
    cfg = config if config is not None else V2Config()
    return validate_sidecars(cfg.resolved_sidecar_dir() / doc_id)


__all__ = [
    "NATIVE_CLASSES",
    "OCR_CLASSES",
    "PAGE_FIGURE_CLASSES",
    "V2DocumentResult",
    "extract_document",
    "extract_many",
    "verify_document",
]
