"""Feature-gated configuration for the experimental INCOIS V2 layer.

Read from the environment at **call time** (never cached), so the toggle can be
flipped per process launch, per document in a test, or per migration batch
without reloading anything. This mirrors the two precedents already in this
repository — ``DFG_PARALLEL_OCR`` (``src/data/extract_dfg.py``) and
``INGEST_EXCLUDE_GLOBS`` / ``INGEST_ALLOW_HINDI``
(``config/sources.yaml``) — both plain ``os.environ`` reads with no pydantic
settings layer in between (the repo has none).

Precedence: **environment > ``config/v2.yaml`` > the dataclass defaults**.

Malformed values fall back to the default and are reported on stdout; they never
raise. An ingestion run must not fail because someone typo'd ``V2_OCR_DPI=2oo``.

Import weight: this module itself uses only the standard library (plus ``yaml``
for the optional ``config/v2.yaml``). ``src.utils.app_paths`` is imported lazily
inside :func:`_default_sidecar_dir` / :func:`_yaml_defaults`. Note that
importing anything under ``src.data.*`` runs ``src/data/__init__.py``, which
eagerly imports ``enricher``/``loader``/``validator`` and therefore pulls in the
repo's core packages (``pydantic``, ``rich``, ``orjson``). That is pre-existing
and shared with ``src.data.extract_dfg``; what matters here is that **no OCR,
vision, or ML dependency is loaded** — ``tests/test_v2_disabled_is_inert.py``
asserts exactly that in a fresh interpreter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Same truthy vocabulary as ``src/data/extract_dfg.py::_TRUTHY`` so the two
#: feature gates behave identically for operators.
_TRUTHY = frozenset(("1", "true", "yes", "on"))
_FALSY = frozenset(("0", "false", "no", "off"))

#: Environment-variable names, kept as constants so tests and docs cannot drift
#: from the reader.
ENV_ENABLED = "V2_ENABLED"
ENV_PAGE_ROUTING = "V2_PAGE_ROUTING"
ENV_OCR = "V2_OCR"
ENV_OCR_LANG = "V2_OCR_LANG"
ENV_OCR_DPI = "V2_OCR_DPI"
ENV_OCR_PAGE_CAP = "V2_OCR_PAGE_CAP"
ENV_OCR_INVERT_LUM = "V2_OCR_INVERT_LUM"
ENV_TABLES = "V2_TABLES"
ENV_TABLE_ROTATION = "V2_TABLE_ROTATION"
ENV_TABLE_CONT_GAP = "V2_TABLE_CONT_GAP"
ENV_TABLE_BORDERLESS = "V2_TABLE_BORDERLESS"
ENV_FIGURES = "V2_FIGURES"
ENV_MIN_FIG_PX = "V2_MIN_FIG_PX"
ENV_FIG_MERGE_GAP = "V2_FIG_MERGE_GAP"
ENV_FIG_CROP_DPI = "V2_FIG_CROP_DPI"
ENV_FIG_MAX_CROPS = "V2_FIG_MAX_CROPS"
ENV_CAPTION_STRICT = "V2_CAPTION_STRICT"
ENV_HINDI_GATE_RATIO = "V2_HINDI_GATE_RATIO"
ENV_SIDECAR_DIR = "V2_SIDECAR_DIR"
ENV_INDEX_CHILDREN = "V2_INDEX_CHILDREN"
ENV_VISION_AT_GENERATION = "V2_VISION_AT_GENERATION"
ENV_RERANK_WINDOW = "V2_RERANK_WINDOW"
ENV_INDEX_CHILD_PATH = "V2_INDEX_CHILD_PATH"
ENV_CHILD_MAX_CHARS = "V2_CHILD_MAX_CHARS"
ENV_VECTOR_OP_THRESHOLD = "V2_VECTOR_OP_THRESHOLD"
ENV_MOJIBAKE_RATIO_GATE = "V2_MOJIBAKE_RATIO_GATE"
ENV_ENHANCED_EXTRACTION = "INCOIS_ENHANCED_EXTRACTION"
ENV_EXPERIMENTAL_INDEX = "V2_EXPERIMENTAL_INDEX"
ENV_OCR_WALL_SECONDS = "V2_OCR_WALL_SECONDS"
ENV_OCR_WORKERS = "V2_OCR_WORKERS"


# ── scalar parsing (fallback, never raise) ────────────────────────────────────


def _parse_bool(raw: object, default: bool, problems: list[str], name: str) -> bool:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if not value:
        return default
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    problems.append(f"{name}={raw!r} is not a boolean; using {default}")
    return default


def _parse_int(raw: object, default: int, problems: list[str], name: str) -> int:
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        problems.append(f"{name}={raw!r} is not an integer; using {default}")
        return default


def _parse_float(raw: object, default: float, problems: list[str], name: str) -> float:
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        problems.append(f"{name}={raw!r} is not a number; using {default}")
        return default


def _parse_str(raw: object, default: str) -> str:
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def _default_sidecar_dir() -> Path:
    """``<data_dir()>/v2_sidecars``.

    Resolved through ``src.utils.app_paths.data_dir()`` — **never** a literal
    ``data/v2_sidecars``. In the Singularity container ``APP_DATA_DIR=/data`` is
    a bind mount; a literal relative path would escape it and be lost under
    ``--writable-tmpfs``. The name deliberately avoids ``out/``, ``build/``,
    ``dist/`` and ``target/``, which are common artifact-exclusion patterns.
    """
    from src.utils.app_paths import data_dir

    return data_dir() / "v2_sidecars"


def _yaml_defaults() -> dict[str, object]:
    """Optional ``config/v2.yaml`` defaults. Missing/unreadable file -> {}."""
    try:
        from src.utils.app_paths import config_path

        path = config_path("v2.yaml")
        if not path.is_file():
            return {}
        import yaml

        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except Exception as exc:  # noqa: BLE001 — config must never break ingestion
        return {"__error__": f"config/v2.yaml ignored: {exc}"}
    if loaded is None:
        # An all-comments or empty YAML file is valid and simply sets nothing.
        return {}
    if not isinstance(loaded, dict):
        return {"__error__": "config/v2.yaml must be a mapping; ignored"}
    loaded.pop("version", None)
    return loaded


# ── the configuration object ──────────────────────────────────────────────────


@dataclass(frozen=True)
class V2Config:
    """Immutable V2 feature configuration.

    Defaults are chosen so that ``V2Config()`` — i.e. no environment at all —
    is exactly today's production behaviour: ``enabled=False``, and therefore
    every ``*_active`` property below is ``False`` regardless of the sub-flags.
    """

    # ── master switch for the EXPERIMENTAL layer ─────────────────────────────
    #
    # ``V2_ENABLED`` no longer means "everything V2". Enhanced core extraction
    # is now the normal INCOIS path and is gated by ``enhanced_extraction``
    # below; this flag governs the experimental half (child-unit indexing,
    # borderless tables, generation-time vision) and is conjunctive with each of
    # those feature switches.
    enabled: bool = False

    # ── CORE: enhanced INCOIS extraction replaces legacy extraction ──────────
    #
    # ON by default: this is the production extraction path for INCOIS PDFs.
    # Turning it off restores ``_extract_text_subprocess`` byte-for-byte.
    enhanced_extraction: bool = True

    # ── EXPERIMENTAL: figures/crops/child units participate in retrieval ─────
    #
    # OFF by default. Figure capture is NOT gated by this — sidecars and crops
    # are always written during enhanced extraction, so enabling experimental
    # retrieval later needs no re-extraction.
    experimental_index: bool = False

    # ── page routing / OCR (spec §6, §3.1, §3.7) ─────────────────────────────
    page_routing: bool = True
    ocr: bool = True                 # ADDITION — see module note below
    ocr_lang: str = "hin+eng"
    ocr_dpi: int = 200
    #: 0 = no cap: every page classified as needing OCR is OCR'd. A positive
    #: value is an explicit operator-imposed ceiling, and pages skipped because
    #: of it are recorded as ``skipped(page-cap)`` with ``ocr_complete=false``
    #: in doc.json — never silently dropped. The previous default of 14 was a
    #: prototype safety limit that left 108 of AR_2018's 122 scan pages
    #: un-OCR'd while the document still looked complete.
    ocr_page_cap: int = 0
    #: Wall-clock ceiling for OCR within one document, seconds. 0 = no limit.
    #: Same honesty rule: hitting it records ``skipped(wall-clock)`` and marks
    #: the document incomplete rather than pretending otherwise.
    ocr_wall_seconds: float = 0.0
    #: Page-level OCR worker threads. **0 = serial (the default, and the safe
    #: fallback).** Any positive value opts in to the bounded pool; the count
    #: actually used is further capped by CPU, RAM and the number of OCR pages.
    #: Only Tesseract runs in the workers — PyMuPDF is never touched off the
    #: main thread. Ignored when a page cap or wall-clock limit is set, because
    #: those are inherently sequential.
    ocr_workers: int = 0
    ocr_invert_lum: int = 127

    # ── tables (spec §6, §3.3-§3.5) ──────────────────────────────────────────
    tables: bool = True
    table_rotation: bool = True
    table_cont_gap: int = 2
    table_borderless: bool = False   # EXPERIMENTAL — spec §13.1, no natural positive

    # ── figures (spec §6, §3.6) ──────────────────────────────────────────────
    figures: bool = True             # ADDITION — see module note below
    min_fig_px: int = 200
    fig_merge_gap: float = 25.0
    fig_crop_dpi: int = 110
    fig_max_crops: int = 60
    caption_strict: bool = True

    # ── indexing / language gate (spec §6, §3.8, §3.9) ───────────────────────
    hindi_gate_ratio: float = 0.30
    index_children: bool = True
    rerank_window: int = 1600
    #: V2 child units live in their OWN vector store, never in the production
    #: ``storage/hybrid_rag``. A separate path means enabling V2 cannot append to
    #: or rebuild the index a running service depends on, and a bad V2 build is
    #: discarded by deleting one directory.
    index_child_path: str = "storage/hybrid_rag_v2"
    #: Upper bound on a single child unit's text length.
    child_max_chars: int = 6000

    # ── generation (spec §6, §12) ────────────────────────────────────────────
    vision_at_generation: bool = False  # EXPERIMENTAL — spec §13.7, no model exists

    # ── storage (spec §6, §2 ops warning) ────────────────────────────────────
    sidecar_dir: Path | None = None  # None -> _default_sidecar_dir() at load time

    # ── classifier heuristics (spec §3.1; "documented, tunable" per §3.3) ────
    vector_op_threshold: int = 3000
    mojibake_ratio_gate: float = 0.05

    # ── effective state ──────────────────────────────────────────────────────
    #
    # Every capability is conjunctive with the master switch, computed here so a
    # call site cannot enable a feature by testing a sub-flag alone. These are
    # the only properties production code should branch on.

    @property
    def routing_active(self) -> bool:
        """Per-page classification (cheap; no OCR implied).

        Core path: gated by ``enhanced_extraction``, NOT by the experimental
        master switch. ``enabled`` governs only experimental indexing/vision.
        """
        return self.core_extraction_active

    @property
    def ocr_active(self) -> bool:
        """OCR may run. Requires routing (that is what selects the pages)."""
        return self.routing_active and self.ocr

    @property
    def tables_active(self) -> bool:
        """Normal tables are core content, so this is a core-path gate."""
        return self.core_extraction_active and self.tables

    @property
    def rotated_tables_active(self) -> bool:
        return self.tables_active and self.table_rotation

    @property
    def borderless_active(self) -> bool:
        return self.tables_active and self.table_borderless

    @property
    def figures_active(self) -> bool:
        """Figure *capture* is part of the core pass (same PDF walk, provenance
        recorded to sidecars). Whether figures ever reach retrieval is a
        separate decision, gated by ``children_indexed``/``experimental_index``,
        so that enabling experimental retrieval later needs no re-extraction.
        """
        return self.core_extraction_active and self.figures

    @property
    def children_indexed(self) -> bool:
        """TableBlock / FigureCard units may enter the retrieval stack.

        Conjunctive: the experimental master gate, the explicit experimental
        index switch, and the child-indexing switch must all hold.
        """
        return self.enabled and self.experimental_index and self.index_children

    @property
    def core_extraction_active(self) -> bool:
        """Is enhanced core extraction the active INCOIS path?

        Independent of ``enabled``: core extraction is the production path and
        must not depend on the experimental master gate.
        """
        return self.enhanced_extraction and self.page_routing

    @property
    def vision_active(self) -> bool:
        return self.enabled and self.vision_at_generation

    def resolved_sidecar_dir(self) -> Path:
        return self.sidecar_dir if self.sidecar_dir is not None else _default_sidecar_dir()

    def as_flat_dict(self) -> dict[str, object]:
        """Serialisable view for ``doc.json`` provenance and test assertions."""
        from dataclasses import asdict

        flat = asdict(self)
        flat["sidecar_dir"] = str(self.resolved_sidecar_dir())
        for name in (
            "routing_active",
            "ocr_active",
            "tables_active",
            "rotated_tables_active",
            "borderless_active",
            "figures_active",
            "children_indexed",
            "core_extraction_active",
            "vision_active",
        ):
            flat[name] = getattr(self, name)
        return flat


def load_v2_config() -> V2Config:
    """Build a :class:`V2Config` from env over optional ``config/v2.yaml``.

    Never raises. Malformed values fall back to the default and are listed on
    stdout so a misconfigured HPC launch is diagnosable from the ingestion log.
    """
    problems: list[str] = []

    defaults = _yaml_defaults()
    if "__error__" in defaults:
        problems.append(str(defaults.pop("__error__")))

    def source(name: str) -> object:
        """Environment wins; fall back to the YAML value (possibly None)."""
        value = os.environ.get(name)
        if value is not None:
            return value
        return defaults.get(_yaml_key(name))

    cfg = V2Config(
        enabled=_parse_bool(source(ENV_ENABLED), False, problems, ENV_ENABLED),
        page_routing=_parse_bool(
            source(ENV_PAGE_ROUTING), True, problems, ENV_PAGE_ROUTING
        ),
        ocr=_parse_bool(source(ENV_OCR), True, problems, ENV_OCR),
        ocr_lang=_parse_str(source(ENV_OCR_LANG), "hin+eng"),
        ocr_dpi=_parse_int(source(ENV_OCR_DPI), 200, problems, ENV_OCR_DPI),
        ocr_page_cap=_parse_int(
            source(ENV_OCR_PAGE_CAP), 0, problems, ENV_OCR_PAGE_CAP
        ),
        ocr_invert_lum=_parse_int(
            source(ENV_OCR_INVERT_LUM), 127, problems, ENV_OCR_INVERT_LUM
        ),
        tables=_parse_bool(source(ENV_TABLES), True, problems, ENV_TABLES),
        table_rotation=_parse_bool(
            source(ENV_TABLE_ROTATION), True, problems, ENV_TABLE_ROTATION
        ),
        table_cont_gap=_parse_int(
            source(ENV_TABLE_CONT_GAP), 2, problems, ENV_TABLE_CONT_GAP
        ),
        table_borderless=_parse_bool(
            source(ENV_TABLE_BORDERLESS), False, problems, ENV_TABLE_BORDERLESS
        ),
        figures=_parse_bool(source(ENV_FIGURES), True, problems, ENV_FIGURES),
        min_fig_px=_parse_int(source(ENV_MIN_FIG_PX), 200, problems, ENV_MIN_FIG_PX),
        fig_merge_gap=_parse_float(
            source(ENV_FIG_MERGE_GAP), 25.0, problems, ENV_FIG_MERGE_GAP
        ),
        fig_crop_dpi=_parse_int(
            source(ENV_FIG_CROP_DPI), 110, problems, ENV_FIG_CROP_DPI
        ),
        fig_max_crops=_parse_int(
            source(ENV_FIG_MAX_CROPS), 60, problems, ENV_FIG_MAX_CROPS
        ),
        caption_strict=_parse_bool(
            source(ENV_CAPTION_STRICT), True, problems, ENV_CAPTION_STRICT
        ),
        hindi_gate_ratio=_parse_float(
            source(ENV_HINDI_GATE_RATIO), 0.30, problems, ENV_HINDI_GATE_RATIO
        ),
        index_children=_parse_bool(
            source(ENV_INDEX_CHILDREN), True, problems, ENV_INDEX_CHILDREN
        ),
        rerank_window=_parse_int(
            source(ENV_RERANK_WINDOW), 1600, problems, ENV_RERANK_WINDOW
        ),
        index_child_path=_parse_str(
            source(ENV_INDEX_CHILD_PATH), "storage/hybrid_rag_v2"
        ),
        child_max_chars=_parse_int(
            source(ENV_CHILD_MAX_CHARS), 6000, problems, ENV_CHILD_MAX_CHARS
        ),
        vision_at_generation=_parse_bool(
            source(ENV_VISION_AT_GENERATION), False, problems, ENV_VISION_AT_GENERATION
        ),
        sidecar_dir=_parse_sidecar_dir(source(ENV_SIDECAR_DIR)),
        vector_op_threshold=_parse_int(
            source(ENV_VECTOR_OP_THRESHOLD), 3000, problems, ENV_VECTOR_OP_THRESHOLD
        ),
        mojibake_ratio_gate=_parse_float(
            source(ENV_MOJIBAKE_RATIO_GATE), 0.05, problems, ENV_MOJIBAKE_RATIO_GATE
        ),
        enhanced_extraction=_parse_bool(
            source(ENV_ENHANCED_EXTRACTION), True, problems, ENV_ENHANCED_EXTRACTION
        ),
        experimental_index=_parse_bool(
            source(ENV_EXPERIMENTAL_INDEX), False, problems, ENV_EXPERIMENTAL_INDEX
        ),
        ocr_wall_seconds=_parse_float(
            source(ENV_OCR_WALL_SECONDS), 0.0, problems, ENV_OCR_WALL_SECONDS
        ),
        ocr_workers=_parse_int(
            source(ENV_OCR_WORKERS), 0, problems, ENV_OCR_WORKERS
        ),
    )

    for line in problems:
        print(f"[v2] config warning: {line}")

    return cfg


def _yaml_key(env_name: str) -> str:
    """``V2_TABLE_CONT_GAP`` -> ``table_cont_gap`` (the YAML spelling)."""
    prefix = "V2_"
    return env_name[len(prefix) :].lower() if env_name.startswith(prefix) else env_name.lower()


def _parse_sidecar_dir(raw: object) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return Path(value).expanduser() if value else None
