"""Stage 5 — sidecar storage and provenance validation.

One directory per document under ``<data_dir()>/v2_sidecars/<doc_id>/``:

    doc.json          summary: page classes, OCR count, funnel, timing, config
    pages.jsonl       one PageRecord per page
    tables.jsonl      one TableBlock per logical table
    figures.jsonl     one FigureCard per figure / page-figure
    marked.txt        page-tagged document text with explicit TABLE/FIGURE blocks
    assets/fig/*.png  deterministic crops: p{NNN}_f{II}.png, p{NNN}_page.png

The directory name deliberately avoids ``out/``, ``build/``, ``dist/`` and
``target/`` (spec §2 ops warning): those are common artifact-exclusion patterns
and the sidecars would silently vanish from a snapshot.

Provenance invariant (spec §4, validated 0 errors on 620 objects in Phase 9):
every figure/table id resolves to a ``(doc, page)`` present in ``pages.jsonl``,
and every non-null crop path exists on disk. :func:`validate_sidecars` enforces
exactly that.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.v2.config import V2Config

#: ``{doc}#p{page}f{i}`` / ``{doc}#p{page}pagefig`` / ``{doc}#t{i}`` (+ ``+cont``)
_CHILD_ID_RE = re.compile(r"^(?P<doc>.+)#(?:p(?P<page>\d+)(?:f(?P<fig>\d+)|pagefig)|t(?P<tbl>\d+))")

CROP_FILENAME = "p{page:03d}_f{index:02d}.png"
PAGE_CROP_FILENAME = "p{page:03d}_page.png"


@dataclass
class PageRecord:
    """One page of one document (``pages.jsonl``)."""

    doc: str
    page: int  # 1-based
    page_class: str
    route: str
    section_ctx: str | None = None
    chars: int = 0
    moji_ratio: float = 0.0
    n_images: int = 0
    vector_ops: int = 0
    ocr_used: bool = False
    ocr_status: str = ""
    orientation: str | None = None
    orientation_conf: float | None = None
    guard_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Match the measured artifact field name (``class`` is a keyword).
        data["class"] = data.pop("page_class")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageRecord:
        payload = dict(data)
        payload["page_class"] = payload.pop("class", payload.get("page_class", ""))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class TableBlock:
    """One logical table, post-continuation-merge (``tables.jsonl``)."""

    table_id: str
    doc: str
    page: int
    ncols: int
    nrows: int
    caption: str | None
    method: str
    markdown: str
    rows: list[list[str]] = field(default_factory=list)
    bbox: list[float] | None = None
    orientation: str | None = None
    orientation_conf: float | None = None
    numeric_row_fraction: float | None = None
    continuation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableBlock:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FigureCard:
    """One figure or page-figure (``figures.jsonl``)."""

    figure_id: str
    doc: str
    page: int
    kind: str
    card: str
    caption: str | None = None
    context: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    crop: str | None = None
    ocr_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FigureCard:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


class SidecarStore:
    """Writes and reads one document's sidecars.

    Nothing is created until :meth:`ensure_dir` is called, so constructing a
    store has no filesystem effect.
    """

    def __init__(self, doc_id: str, config: V2Config | None = None) -> None:
        self.doc_id = doc_id
        self.config = config if config is not None else V2Config()
        self.root = self.config.resolved_sidecar_dir() / doc_id
        self.assets_dir = self.root / "assets" / "fig"

    # ── paths ────────────────────────────────────────────────────────────────

    @property
    def pages_path(self) -> Path:
        return self.root / "pages.jsonl"

    @property
    def tables_path(self) -> Path:
        return self.root / "tables.jsonl"

    @property
    def figures_path(self) -> Path:
        return self.root / "figures.jsonl"

    @property
    def marked_path(self) -> Path:
        return self.root / "marked.txt"

    @property
    def doc_path(self) -> Path:
        return self.root / "doc.json"

    def ensure_dir(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def crop_path(self, page: int, index: int) -> Path:
        return self.assets_dir / CROP_FILENAME.format(page=page, index=index)

    def page_crop_path(self, page: int) -> Path:
        return self.assets_dir / PAGE_CROP_FILENAME.format(page=page)

    # ── writing ──────────────────────────────────────────────────────────────

    def write_pages(self, records: list[PageRecord]) -> None:
        self.ensure_dir()
        _write_jsonl(self.pages_path, [record.to_dict() for record in records])

    def write_tables(self, blocks: list[TableBlock]) -> None:
        self.ensure_dir()
        _write_jsonl(self.tables_path, [block.to_dict() for block in blocks])

    def write_figures(self, cards: list[FigureCard]) -> None:
        self.ensure_dir()
        _write_jsonl(self.figures_path, [card.to_dict() for card in cards])

    def write_marked(self, text: str) -> None:
        self.ensure_dir()
        self.marked_path.write_text(text, encoding="utf-8")

    def write_doc(self, summary: dict[str, Any]) -> None:
        self.ensure_dir()
        self.doc_path.write_text(
            json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    # ── reading ──────────────────────────────────────────────────────────────

    def read_pages(self) -> list[PageRecord]:
        return [PageRecord.from_dict(item) for item in _read_jsonl(self.pages_path)]

    def read_tables(self) -> list[TableBlock]:
        return [TableBlock.from_dict(item) for item in _read_jsonl(self.tables_path)]

    def read_figures(self) -> list[FigureCard]:
        return [FigureCard.from_dict(item) for item in _read_jsonl(self.figures_path)]

    def read_marked(self) -> str:
        return self.marked_path.read_text(encoding="utf-8") if self.marked_path.is_file() else ""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_marked_text(
    doc_id: str,
    pages: list[PageRecord],
    tables: list[TableBlock],
    figures: list[FigureCard],
    page_texts: dict[int, str],
) -> str:
    """Assemble ``marked.txt`` (spec §2 stage S6, §4 line protocol).

    Page boundaries are preserved, table blocks are represented explicitly as
    markdown, and figures are referenced by id. Text is **not** flattened into a
    single undifferentiated stream — that flattening is precisely what the
    current pipeline does and what loses cell semantics.
    """
    tables_by_page: dict[int, list[TableBlock]] = {}
    for block in tables:
        tables_by_page.setdefault(block.page, []).append(block)
    figures_by_page: dict[int, list[FigureCard]] = {}
    for card in figures:
        figures_by_page.setdefault(card.page, []).append(card)

    out: list[str] = []
    for record in pages:
        out.append(f"[p{record.page}] ({record.page_class}->{record.route})")
        for line in (page_texts.get(record.page) or "").splitlines():
            if line.strip():
                out.append(f"[p{record.page}] {line.rstrip()}")
        for block in tables_by_page.get(record.page, []):
            out.append(f"[p{record.page}] TABLE {block.table_id}:")
            out.extend(block.markdown.splitlines())
        for card in figures_by_page.get(record.page, []):
            caption = card.caption or ""
            out.append(
                f"[p{record.page}] FIGURE {card.figure_id} kind={card.kind} caption={caption}"
            )
    return "\n".join(out) + "\n"


def validate_sidecars(root: Path | str, check_crops: bool = True) -> tuple[bool, list[str]]:
    """Enforce the spec §4 provenance invariant on one sidecar directory.

    Returns ``(ok, errors)``. Checks:

    * every table/figure id parses and its page exists in ``pages.jsonl``;
    * the id's ``doc`` matches the sidecar's own doc;
    * ``marked.txt`` page-tag count equals the page count;
    * every non-null crop path exists on disk (when ``check_crops``).
    """
    directory = Path(root)
    errors: list[str] = []

    pages = [PageRecord.from_dict(item) for item in _read_jsonl(directory / "pages.jsonl")]
    if not pages:
        return False, [f"{directory}: no pages.jsonl records"]

    doc_ids = {record.doc for record in pages}
    if len(doc_ids) > 1:
        errors.append(f"pages.jsonl mixes documents: {sorted(doc_ids)}")
    known_pages = {record.page for record in pages}

    for block in [TableBlock.from_dict(i) for i in _read_jsonl(directory / "tables.jsonl")]:
        _check_child_id(block.table_id, block.doc, block.page, doc_ids, known_pages, errors)

    for card in [FigureCard.from_dict(i) for i in _read_jsonl(directory / "figures.jsonl")]:
        _check_child_id(card.figure_id, card.doc, card.page, doc_ids, known_pages, errors)
        if check_crops and card.crop and not Path(card.crop).is_file():
            errors.append(f"{card.figure_id}: crop missing on disk: {card.crop}")

    marked_path = directory / "marked.txt"
    if marked_path.is_file():
        tagged = {
            int(match.group(1))
            for match in re.finditer(r"^\[p(\d+)\]", marked_path.read_text(encoding="utf-8"), re.M)
        }
        if tagged != known_pages:
            missing = sorted(known_pages - tagged)
            extra = sorted(tagged - known_pages)
            errors.append(f"marked.txt page tags mismatch: missing={missing} extra={extra}")
    else:
        errors.append("marked.txt missing")

    return not errors, errors


def _check_child_id(
    child_id: str,
    doc: str,
    page: int,
    doc_ids: set[str],
    known_pages: set[int],
    errors: list[str],
) -> None:
    match = _CHILD_ID_RE.match(child_id)
    if not match:
        errors.append(f"unparseable child id: {child_id}")
        return
    if match.group("doc") != doc:
        errors.append(f"{child_id}: id doc {match.group('doc')!r} != record doc {doc!r}")
    if doc not in doc_ids:
        errors.append(f"{child_id}: doc {doc!r} not in pages.jsonl")
    if page not in known_pages:
        errors.append(f"{child_id}: page {page} not in pages.jsonl")


__all__ = [
    "FigureCard",
    "PageRecord",
    "SidecarStore",
    "TableBlock",
    "build_marked_text",
    "validate_sidecars",
]
