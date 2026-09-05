"""Source-aware ingestion CLI — ONE engine, many sources (flat + hierarchical).

    python -m src.scripts.ingest moes --ingest                 # whole moes/ tree
    python -m src.scripts.ingest moes/incois --ingest          # one org branch
    python -m src.scripts.ingest moes/incois/annual_reports --ingest
    python -m src.scripts.ingest parliament --ingest           # records merge
    python -m src.scripts.ingest isro --ingest                 # zero-config discovery
    python -m src.scripts.ingest all --ingest                  # registered ∪ discovered
    python -m src.scripts.ingest --list                        # show resolved sources

    (the --ingest flag is the default action: `<source>` alone also ingests)

Architecture (deliberately thin — no conversion logic lives here):

    source CLI (this file)           registry + hierarchy walker + routing
          ↓ delegates
    src/scripts/ingest_folder.py     THE engine: detect_doc_type -> convert
                                     (shared convert_sirs_knowledge
                                     converters, meta_context pass-through)
                                     -> dedup -> atomic append to
                                     data/corpus_reports.jsonl
          ↓ reused verbatim
    incremental_update()             load index, embed ONLY new records
    rebuild_index()                  FULL rebuild — --full-rebuild / first build

Hierarchical convention (config/sources.yaml, NOT code):

    data/<source>/[<org-segment>/]<category-segment>/<files>
        moes/ministry/...            -> org=moes_hq (org_map)
        moes/incois/annual_reports/x.pdf -> org=incois, document_type=annual_report
        isro/annual_reports/x.pdf    -> org=isro (default_org), annual_report

Path grammar (deterministic): walking a hierarchical source, the FIRST path
segment (relative to the source root) that matches category_map sets the
document_type hint; the first segment BEFORE it is the org slug (mapped
through org_map, lowercased). Files directly under the root use default_org
+ the engine's legacy content/filename detection. A wrongly-filed document
still self-declares: strong content headers outrank the hint
(detect_doc_type precedence: content → category_hint → legacy folder/filename).

Guarantees (same contract as ingest_folder):
  * corpus writes are append-only to data/corpus_reports.jsonl
    (deterministic ids -> scanning the same source twice adds nothing);
  * normal ingestion NEVER triggers a full re-embed of the corpus;
  * frontend inbox flow is untouched — inbox stays a flat source;
  * record ids are content hashes, so stamped metadata can never create
    duplicates, even for files moved between folders/sources.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# The engine. Reused, never re-implemented: conversion, dedup, atomic append,
# incremental/full index updates. Do NOT fork these functions into this file
# (the server imports them too — a second copy would drift).
import src.scripts.ingest_folder as _engine

# Single canonical content hash — engine-owned so the CLI (records-kind
# merge) and the engine (folders-kind conversion) can never drift apart.
# Re-exported under the underscored name as this module's public contract
# (pinned by tests/test_ingest_changed_records.py).
_qa_content_hash = _engine.qa_content_hash
from src.models.qa_record import QARecord
from src.utils.app_paths import config_path, corpus_path, data_dir, index_dir
from src.utils.atomic_io import append_jsonl_atomic

INGESTIBLE_EXTS = {".pdf", ".txt", ".md", ".json", ".jsonl"}

# Directories that must never be swallowed by discovery (registry-claimed
# roots are excluded separately, dynamically). Applies at the DATA ROOT only —
# inside a hierarchical source, path segments are data.
_ALWAYS_EXCLUDED_DIRS = {"raw", "finetune", "user-knowledge"}

# Walker pruning (any depth): move-target convention + hidden/junk dirs.
_WALK_SKIP_DIRS = {"processed", "__pycache__"}

# Legacy default ministry for FLAT registered sources without an explicit
# ministry (preserves pre-hierarchy behavior; hierarchical/discovered
# sources never inherit it).
_LEGACY_DEFAULT_MINISTRY = "EARTH SCIENCES"


def _normalize_segment(seg: str) -> str:
    """Deterministic dir-segment normalization for category_map/org_map keys."""
    return seg.strip().lower().replace("-", "_").replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Source registry (config file, with a built-in mirror as fallback)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceSpec:
    """How to ingest one logical source."""

    name: str
    kind: str  # "folders" | "records"
    folders: list[str] = field(default_factory=list)       # kind=folders
    record_dirs: list[str] = field(default_factory=list)   # kind=records
    move_processed: bool = False
    description: str = ""
    discovered: bool = False  # True when found via filesystem discovery
    # ── Hierarchy (Phase 1) ──
    hierarchical: bool = False          # walk <root>/[<org>/]<category>/<files>
    org: str | None = None              # flat source: explicit org stamp
    default_org: str | None = None      # hierarchical: org when no org segment
    org_map: dict = field(default_factory=dict)     # dir segment -> org slug
    ministry: str | None = None         # explicit ministry stamp (else default rules)
    subpath: str | None = None          # CLI subpath selector (moes/incois)
    # ── Records merge (audit IW-3) ──
    recursive: bool = False             # records: recurse record_dirs (e.g.
                                        # parliamentary-qa/rajya-sabha/session-*/)
    # ── File-name sidecar exclusion (audit IW-4) ──
    exclude_files: list[str] = field(default_factory=list)  # exact names,
                                        # case-insensitive (crawler sidecars:
                                        # record.json / manifest.json / ...)
    # ── English-only corpus policy (audit: corpus is English-only) ────────
    exclude_globs: list[str] = field(default_factory=list)  # filename globs,
                                        # case-insensitive, skipped BEFORE
                                        # conversion/OCR (default ["*-hin.*"]
                                        # when unset — see ingest_folder)
    # ── Presentation opt-out (Phase-5 integration) ──
    present_in_tree: bool = True        # hierarchical sources normally become
                                        # a ministry node in /api/sources; set
                                        # false when the source's records belong
                                        # to an EXISTING ministry's orgs instead
                                        # (e.g. moes_website -> moes/moes_hq).


@dataclass
class LeafJob:
    """One concrete folder to hand to the engine, with resolved identity."""
    folder: Path
    org: str | None
    doc_type_hint: str | None
    meta_context: dict
    exclude_files: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()


# Fallback used ONLY if config/sources.yaml is unreadable — must stay equal to
# the shipped config/sources.yaml (asserted by tests).
_BUILTIN_CATEGORY_MAP: dict[str, str] = {
    "annual_reports": "annual_report",
    "audit_reports": "audit_report",
    "research_papers": "research_publication",
    "press_release": "press_release",
    "other": "document",
}

_BUILTIN_SOURCES: dict[str, dict] = {
    "inbox": {
        "kind": "folders", "folders": ["inbox"], "move_processed": True,
        "description": "Frontend uploads / manual drop-ins.",
    },
    "parliament": {
        "kind": "records", "record_dirs": ["enriched", "processed"],
        "description": "Phase-1 pipeline output (data/enriched, data/processed).",
    },
    "moes": {
        "kind": "folders", "folders": ["moes"], "hierarchical": True,
        "ministry": "EARTH SCIENCES", "default_org": "moes_hq",
        "org_map": {"ministry": "moes_hq", "incois": "incois", "imd": "imd",
                    "iitm": "iitm", "niot": "niot"},
        "description": "Ministry of Earth Sciences knowledge tree.",
    },
    "incois": {
        "kind": "folders", "org": "incois", "ministry": "EARTH SCIENCES",
        "folders": [
            "annual_reports",
            "incois_reports/AnnualReports",
            "incois_reports/Others",
            "incois_reports/TechnicalReports",
            "incois_reports/ResearchPublications",
            "scanned_ocr",
        ],
        "description": "Legacy flat INCOIS crawler layout (+ scanned OCR).",
    },
    "moes_reports": {
        "kind": "folders", "org": "moes_hq", "ministry": "EARTH SCIENCES",
        "folders": ["moes_reports/knowledge"],
        "description": "Legacy MoES CCPS knowledge JSONs.",
    },
    "rajya_sabha": {
        "kind": "records", "record_dirs": ["parliamentary-qa/rajya-sabha"],
        "recursive": True,
        "description": "Staged Rajya Sabha Q&A (crawl_parliamentary_qa; "
                       "session-*/qa.jsonl merged recursively, ids preserved).",
    },
    "lok_sabha": {
        "kind": "records", "record_dirs": ["parliamentary-qa/lok-sabha"],
        "recursive": True,
        "description": "Staged Lok Sabha Q&A (src.scraping.ls.pipeline; "
                       "lok-*/session-*/qa.jsonl merged recursively, ids preserved).",
    },
    "moes_website": {
        "kind": "folders", "folders": [".moes-website"], "hierarchical": True,
        "ministry": "EARTH SCIENCES", "default_org": "moes_hq",
        "org_map": {"reports": "moes_hq"},
        "exclude_files": ["record.json", "manifest.json",
                          "attachment-map.json", "last_run.json"],
        "present_in_tree": False,
        "description": "MoES website staging corpus (crawl_moes_website; "
                       "documents under <category>/<post>/documents/).",
    },
}

_BUILTIN_DISCOVERY_EXCLUDES = ["processed", "enriched", "raw", "finetune", "user-knowledge",
                               "parliamentary-qa"]


def _default_config_file() -> Path:
    env = (os.environ.get("INGEST_SOURCES_CONFIG") or "").strip()
    if env:
        return Path(env).expanduser()
    return config_path("sources.yaml")


def _spec_from_dict(name: str, raw: dict) -> SourceSpec:
    return SourceSpec(
        name=name,
        kind=str(raw.get("kind") or "folders"),
        folders=list(raw.get("folders") or []),
        record_dirs=list(raw.get("record_dirs") or []),
        move_processed=bool(raw.get("move_processed", False)),
        description=str(raw.get("description") or "").strip(),
        hierarchical=bool(raw.get("hierarchical", False)),
        org=(str(raw["org"]) if raw.get("org") else None),
        default_org=(str(raw["default_org"]) if raw.get("default_org") else None),
        org_map={_normalize_segment(str(k)): str(v)
                 for k, v in (raw.get("org_map") or {}).items()},
        ministry=(str(raw["ministry"]) if raw.get("ministry") else None),
        recursive=bool(raw.get("recursive", False)),
        exclude_files=[str(f) for f in (raw.get("exclude_files") or [])],
        exclude_globs=[str(g) for g in (raw.get("exclude_globs") or [])],
        present_in_tree=bool(raw.get("present_in_tree", True)),
    )


def load_sources(config_file: str | Path | None = None
                 ) -> tuple[dict[str, SourceSpec], set[str], dict[str, str]]:
    """Load the source registry + discovery excludes + hierarchy category_map.

    Returns (sources, discovery_excludes, category_map). Falls back to the
    built-in mirror when the config file is missing/unreadable — the CLI must
    stay usable on a bare checkout.
    """
    cfg = Path(config_file) if config_file else _default_config_file()
    sources: dict[str, SourceSpec] = {}
    excludes: set[str] = set(_ALWAYS_EXCLUDED_DIRS)
    category_map: dict[str, str] = dict(_BUILTIN_CATEGORY_MAP)
    if cfg.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            for name, raw in (data.get("sources") or {}).items():
                if isinstance(raw, dict):
                    sources[str(name)] = _spec_from_dict(str(name), raw)
            disc = data.get("discovery") or {}
            excludes.update(str(d) for d in (disc.get("exclude_dirs") or []))
            # Global English-only policy: applies to every source that does
            # not declare its own exclude_globs. Empty list = opt out for
            # that source; omitted = inherit the built-in default.
            global_globs = [str(g) for g in (data.get("exclude_globs") or [])]
            if global_globs:
                for spec in sources.values():
                    if not spec.exclude_globs:
                        spec.exclude_globs = list(global_globs)
            hier = data.get("hierarchy") or {}
            file_cat = {
                _normalize_segment(str(k)): str(v)
                for k, v in (hier.get("category_map") or {}).items()
            }
            if file_cat:
                category_map = file_cat
        except Exception as e:  # noqa: BLE001 — config must never kill ingestion
            _engine.log(f"[ingest] WARN: {cfg} unreadable ({e}) — using built-in source registry")
            sources.clear()
    if not sources:
        sources = {name: _spec_from_dict(name, raw) for name, raw in _BUILTIN_SOURCES.items()}
        excludes.update(_BUILTIN_DISCOVERY_EXCLUDES)
    return sources, excludes, category_map


def _data_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    return p if p.is_absolute() else data_dir() / p


def _subtree_has_ingestible_files(root: Path) -> bool:
    """Recursive (pruned) check — hierarchical dirs hold files at depth ≥ 1."""
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in _WALK_SKIP_DIRS
            )
            for fname in filenames:
                f = Path(dirpath) / fname
                if f.is_file() and f.suffix.lower() in INGESTIBLE_EXTS:
                    return True
    except OSError:
        pass
    return False


def discover_sources(
    registered: dict[str, SourceSpec], excludes: set[str]
) -> dict[str, SourceSpec]:
    """First-level data directories not claimed by the registry that contain
    ingestible files (recursively) -> discovered HIERARCHICAL sources.

    Deterministic: sorted by directory name. A registered source's declared
    folders claim their top-level owner (`moes` claims the whole moes/
    subtree), so nesting is never double-ingested by discovery. Discovered
    sources run hierarchical with default_org=<dirname> and ministry=None
    (unknown) — Earth Sciences is never stamped by default on new sources.
    """
    root = data_dir()
    found: dict[str, SourceSpec] = {}
    if not root.exists():
        return found
    claimed: set[str] = set(registered)
    for spec in registered.values():
        for rel in list(spec.folders) + list(spec.record_dirs):
            parts = Path(rel).parts
            if parts:
                claimed.add(parts[0])
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "__pycache__":
            continue
        if d.name in claimed or d.name in excludes:
            continue
        if _subtree_has_ingestible_files(d):
            found[d.name] = SourceSpec(
                name=d.name,
                kind="folders",
                folders=[d.name],
                hierarchical=True,
                default_org=d.name,
                ministry=None,
                description="discovered data directory (unregistered hierarchical source)",
                discovered=True,
            )
    return found


def resolve_sources(
    which: str, config_file: str | Path | None = None
) -> tuple[dict[str, SourceSpec], set[str], dict[str, str]]:
    """Resolve `which` ("all" | source | source/sub/path) to an ordered
    (sources, excludes, category_map) triple.

    `moes/incois` narrows a hierarchical source to one branch (org path);
    the subpath must exist under the source root. Unknown top names fall back
    to filesystem discovery; truly unknown names error listing known sources.
    """
    registered, excludes, category_map = load_sources(config_file)

    # ── Subpath selector: source[/org[/category]] ───────────────────────────
    parts = [p for p in Path(which).parts if p not in ("", os.sep)]
    subpath: str | None = None
    if len(parts) > 1:
        which, subpath = parts[0], str(Path(*parts[1:]))

    if which.lower() == "all":
        if subpath:
            raise KeyError("'all' does not take a subpath selector")
        merged = dict(registered)
        merged.update(discover_sources(registered, excludes))
        return merged, excludes, category_map

    spec = registered.get(which)
    if spec is None:
        discovered = discover_sources(registered, excludes)
        spec = discovered.get(which)
    if spec is None:
        known = sorted(set(registered) | set(discover_sources(registered, excludes)))
        raise KeyError(
            f"unknown source '{which}'. Known sources: {', '.join(known) or '(none)'} "
            f"(or create data/{which}/ with ingestible files and retry)"
        )

    if subpath:
        if not spec.hierarchical or len(spec.folders) != 1:
            raise KeyError(f"source '{which}' is not a single-root hierarchical source; "
                           f"subpath selectors apply only to hierarchical sources")
        root = _data_path(spec.folders[0])
        if not (root / subpath).is_dir():
            raise KeyError(f"subpath not found: {root / subpath}")
        # keep the real path for filesystem comparisons; normalization is
        # applied per-segment inside resolve_path_context, never to paths
        spec = replace(spec, subpath=subpath)
    return {spec.name: spec}, excludes, category_map


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy walker — path -> identity (Phase 1 core)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_path_context(
    rel_parts: tuple[str, ...], spec: SourceSpec, category_map: dict[str, str]
) -> tuple[str | None, str | None]:
    """Path grammar: segments before the FIRST category_map hit form the org
    (first segment wins, mapped through org_map); the hit sets doc_type hint.

    Returns (org, doc_type_hint). Root-level files (no parts): default_org,
    no hint (legacy detection applies).
    """
    org: str | None = None
    hint: str | None = None
    for seg in rel_parts:
        norm = _normalize_segment(seg)
        if norm in category_map:
            hint = category_map[norm]
            break
        if org is None:
            org = spec.org_map.get(norm, norm)
    if org is None:
        org = spec.org or spec.default_org
    return org, hint


def expand_source(spec: SourceSpec, category_map: dict[str, str]) -> list[LeafJob]:
    """Walk one source into concrete leaf jobs for the engine.

    Flat sources: one job per declared folder (legacy behavior; meta_context
    carries org/ministry only when configured — inbox stays context-free so
    its records are byte-identical to the legacy default). Hierarchical
    sources: pruned top-down walk; every directory containing ingestible
    files becomes a leaf with identity from resolve_path_context.
    """
    jobs: list[LeafJob] = []

    def _ctx(org: str | None, hint: str | None) -> dict:
        return {
            "org": org,
            "source": spec.name,
            "ministry": spec.ministry,
            # Hierarchical/discovered sources with no configured ministry must
            # NOT inherit the legacy EARTH SCIENCES default (temporary scope,
            # not a global label): metadata.ministry stays None then.
            "default_ministry": spec.ministry,
            "doc_type_hint": hint,
        }

    if not spec.hierarchical:
        flat_ctx = None
        if spec.org or spec.ministry:
            flat_ctx = {
                "org": spec.org,
                "source": spec.name,
                "ministry": spec.ministry,
                # flat registered sources preserve the legacy default ministry
                "default_ministry": None if spec.ministry else _LEGACY_DEFAULT_MINISTRY,
                "doc_type_hint": None,
            }
        for rel in spec.folders:
            folder = _data_path(rel)
            jobs.append(LeafJob(folder=folder, org=spec.org,
                                doc_type_hint=None,
                                meta_context=flat_ctx or {},
                                exclude_files=tuple(spec.exclude_files),
                                exclude_globs=tuple(spec.exclude_globs)))
        return jobs

    for rel in spec.folders:
        root = _data_path(rel)
        if not root.exists():
            _engine.log(f"[ingest:{spec.name}] root not found: {root} — skipped")
            continue
        sub_target = (root / spec.subpath) if spec.subpath else None
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in _WALK_SKIP_DIRS
            )
            current = Path(dirpath)
            if sub_target is not None and not (
                current == sub_target or _is_relative_to(current, sub_target)
            ):
                continue
            files = [
                f for f in filenames
                if Path(f).suffix.lower() in INGESTIBLE_EXTS
                and (current / f).is_file()
            ]
            if not files:
                continue
            rel_parts = current.relative_to(root).parts
            org, hint = resolve_path_context(rel_parts, spec, category_map)
            jobs.append(LeafJob(folder=current, org=org, doc_type_hint=hint,
                                meta_context=_ctx(org, hint),
                                exclude_files=tuple(spec.exclude_files),
                                exclude_globs=tuple(spec.exclude_globs)))
    return jobs


def _is_relative_to(path: Path, maybe_parent: Path) -> bool:
    try:
        path.relative_to(maybe_parent)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-source ingestion (delegation only)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_engine_paths() -> None:
    """Pin the engine's import-time globals to the CURRENT app_paths."""
    _engine.CORPUS = corpus_path()
    _engine.LOG = data_dir() / "sync.log"
    _engine.INDEX_DIR = str(index_dir())


def _seed_seen_from_corpus() -> set[str]:
    """Existing corpus question_ids — deterministic cross-run dedup."""
    seen: set[str] = set()
    corpus = corpus_path()
    if corpus.exists():
        for line in corpus.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("question_id"):
                    seen.add(r["question_id"])
            except Exception:  # noqa: BLE001 — tolerate a bad historical line
                continue
    return seen


def _seed_seen_with_hash() -> dict[str, str]:
    """Existing corpus as {question_id: content hash} — changed-record
    detection for records-kind sources. Malformed lines are tolerated
    (skipped); an absent corpus yields an empty map (everything is new)."""
    hashes: dict[str, str] = {}
    corpus = corpus_path()
    if not corpus.exists():
        return hashes
    for line in corpus.open(encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        try:
            rec = QARecord.model_validate_json(s)
        except Exception:  # noqa: BLE001 — tolerate a bad historical line
            continue
        if rec.question_id:
            hashes[rec.question_id] = _qa_content_hash(rec)
    return hashes


def _seed_seen_hashes_by_url() -> dict[str, str]:
    """Existing corpus as {source_url: content hash} — changed-file detection
    for folders-kind sources (document ids are content-derived, so the
    question_id axis cannot see a re-crawled file as a replacement).
    Records without metadata.source_url (e.g. parliamentary Q&A rows) are
    excluded — they are handled by the id-keyed path instead."""
    hashes: dict[str, str] = {}
    corpus = corpus_path()
    if not corpus.exists():
        return hashes
    for line in corpus.open(encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        try:
            rec = QARecord.model_validate_json(s)
        except Exception:  # noqa: BLE001
            continue
        url = getattr(rec.metadata, "source_url", None) if rec.metadata else None
        if url:
            hashes[url] = _qa_content_hash(rec)
    return hashes


def _replace_corpus_rows(replacements: dict[str, QARecord]) -> int:
    """Atomically rewrite corpus rows whose question_id is in `replacements`.

    Changed-record write-back: the row keeps its id and position in the
    corpus; only its content is swapped for the re-crawled version. Rows we
    cannot parse are kept verbatim. A replacement id not present in the
    corpus (should not happen — the hash map is seeded from it) is appended
    so content is never lost. Returns the number of rows replaced/appended.
    """
    from src.utils.atomic_io import write_text_atomic
    corpus = corpus_path()
    if not corpus.exists() or not replacements:
        return 0
    out_lines: list[str] = []
    replaced: set[str] = set()
    for line in corpus.open(encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        try:
            rid = json.loads(s).get("question_id")
        except Exception:  # noqa: BLE001
            out_lines.append(line.rstrip("\n"))
            continue
        if rid in replacements:
            out_lines.append(replacements[rid].model_dump_json())
            replaced.add(rid)
        else:
            out_lines.append(line.rstrip("\n"))
    for rid in sorted(set(replacements) - replaced):
        out_lines.append(replacements[rid].model_dump_json())
        replaced.add(rid)
    write_text_atomic(corpus, "\n".join(out_lines) + "\n")
    return len(replaced)


def merge_record_dirs(dirs: list[str], out: list[QARecord], seen: set[str],
                      source: str | None = None, recursive: bool = False,
                      seen_hashes: dict[str, str] | None = None,
                      out_changed: list[QARecord] | None = None) -> tuple[int, int]:
    """Merge ready-made QARecord JSONL (Phase-1 parliament output, staged
    crawler corpora) into `out`.

    Ids are PRESERVED (not re-hashed): these records were produced upstream
    with stable id semantics (e.g. 18-4-3035 from the Phase-1 pipeline,
    rs-<ses>-<qno> from the Rajya Sabha staging crawler). Declaration order
    wins on duplicate ids (config lists enriched before processed — the
    richer copy survives). `source` is stamped as provenance.

    ``recursive`` (audit IW-3) descends into nested record layouts — the RS
    staging corpus keeps one qa.jsonl per session-<n>/ directory. File order
    is always sorted (deterministic); the mechanism is generic — any
    records-kind source may opt in per-registry-entry, it is not an
    RS-specific special case.

    ``seen_hashes`` (changed-record detection) maps existing corpus
    question_id -> content hash. With it, an id that is already known but
    hashes differently counts as CHANGED (record collected into
    ``out_changed`` for the caller's row-replacement write-back) instead of
    being silently skipped. None = legacy id-only behaviour (changed always
    0). The first staged occurrence of an id always wins (declaration order)
    — later same-id rows in the same run are ignored either way.

    Returns ``(added, changed)`` — the legacy int return grew a second
    element with the detection; both are plain counts.
    """
    added = 0
    changed = 0
    skipped = 0
    handled: set[str] = set()   # first staged occurrence of an id wins
    for rel in dirs:
        d = _data_path(rel)
        if not d.exists():
            continue
        files = sorted(d.rglob("*.jsonl")) if recursive else sorted(d.glob("*.jsonl"))
        for f in files:
            try:
                for line in f.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = QARecord.model_validate_json(line)
                    except Exception:  # noqa: BLE001 — count malformed/incomplete rows
                        skipped += 1
                        continue
                    if rec.question_id in handled:
                        continue
                    if rec.question_id not in seen:
                        handled.add(rec.question_id)
                        seen.add(rec.question_id)
                        if source is not None and rec.metadata is not None:
                            rec.metadata.source = source
                        out.append(rec)
                        added += 1
                        if seen_hashes is not None:
                            seen_hashes[rec.question_id] = _qa_content_hash(rec)
                    elif seen_hashes is not None:
                        h = _qa_content_hash(rec)
                        if seen_hashes.get(rec.question_id) != h:
                            handled.add(rec.question_id)
                            seen_hashes[rec.question_id] = h
                            changed += 1
                            if out_changed is not None:
                                out_changed.append(rec)
            except OSError as e:
                _engine.log(f"  [warn] {f}: {e}")
    if skipped:
        # Honest residual accounting (audit principle): rows failing QARecord
        # validation (e.g. RS records whose official answer document is still
        # missing, leaving answer_text empty) are NOT silently half-ingested
        # — the count is surfaced and the runbook points at the backfill
        # mechanism; placeholder content is never synthesized.
        _engine.log(
            f"  [warn] records merge: skipped {skipped} row(s) failing QARecord "
            "validation (e.g. empty answer_text — recover the official document "
            "via the backfill runbook; content is never faked)"
        )
    return added, changed


def ingest_source(spec: SourceSpec, move_processed: bool | None = None,
                  category_map: dict[str, str] | None = None) -> dict:
    """Ingest ONE source through the existing engine. Returns stats."""
    if spec.kind == "records":
        out: list[QARecord] = []
        out_changed: list[QARecord] = []
        seen = _seed_seen_from_corpus()
        hashes = _seed_seen_with_hash()
        added, changed = merge_record_dirs(
            spec.record_dirs, out, seen, source=spec.name,
            recursive=spec.recursive,
            seen_hashes=hashes, out_changed=out_changed,
        )
        if out_changed:
            n_rep = _replace_corpus_rows({r.question_id: r for r in out_changed})
            _engine.log(f"[ingest:{spec.name}] replaced {n_rep} changed record(s) "
                        f"in place -> {corpus_path()} (content drift detected)")
        if out:
            lines = [rec.model_dump_json() for rec in out]
            append_jsonl_atomic(corpus_path(), lines)
            _engine.log(f"[ingest:{spec.name}] appended {added} record(s) -> {corpus_path()}")
        elif not out_changed:
            _engine.log(f"[ingest:{spec.name}] no new or changed records in {spec.record_dirs}")
        return {"added": added, "changed": changed, "folders": 0}

    # kind == "folders" — expand (flat or hierarchical) into leaf jobs, each
    # handed to the proven engine path (detect -> convert -> dedup -> append).
    # Folder-level dedup against the corpus happens inside the engine per call.
    jobs = expand_source(spec, category_map or _BUILTIN_CATEGORY_MAP)
    total = files = failed = 0
    changed_total = 0
    scanned = 0
    # Changed-file detection across every leaf of this source, keyed on
    # source_url (document ids are content-derived — the id axis can't see
    # a re-crawled replacement). One seed per source run; the engine updates
    # it in place as it converts.
    seen_hashes_by_url = _seed_seen_hashes_by_url()
    # MoES ↔ Parliamentary Q&A cross-source dedup: only for the moes_website
    # source, only confirmed-duplicate filenames are added to exclude_files.
    # Empty set (safe default) preserves everything and changes nothing.
    dedup_excludes = _moes_website_dedup_excludes() if spec.name == "moes_website" else set()
    if dedup_excludes:
        _engine.log(
            f"[ingest:{spec.name}] cross-source dedup: excluding "
            f"{len(dedup_excludes)} confirmed PQ duplicate(s) at ingestion"
        )
    for job in jobs:
        if not job.folder.exists():
            _engine.log(f"[ingest:{spec.name}] folder not found: {job.folder} — skipped")
            continue
        scanned += 1
        move = False
        if not spec.hierarchical:
            move = spec.move_processed if move_processed is None else move_processed
        res = _engine.ingest_folder(
            str(job.folder),
            move_processed=move,
            meta_context=job.meta_context or None,
            exclude_files=set(job.exclude_files) | dedup_excludes or None,
            seen_hashes=seen_hashes_by_url,
            exclude_globs=job.exclude_globs or None,
        )
        total += res.get("added", 0)
        files += res.get("files", 0)
        failed += res.get("failed", 0)
        changed_total += res.get("changed", 0)
    if scanned == 0:
        _engine.log(f"[ingest:{spec.name}] no folders on disk for this source — nothing to do")
    return {"added": total, "changed": changed_total, "folders": scanned,
            "files": files, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# Index update — the reuse seam (delegates to the engine, identity-pinned)
# ─────────────────────────────────────────────────────────────────────────────

# Re-exported so tests can pin identity: the CLI MUST reuse ingest_folder's
# incremental/full index implementations, not grow its own.
embed_incremental = _engine.incremental_update
embed_full_rebuild = _engine.rebuild_index
index_is_usable = _engine._index_exists


# ─────────────────────────────────────────────────────────────────────────────
# MoES ↔ Parliamentary Q&A cross-source dedup (ingestion-only integration).
# The crawler is untouched. Only confirmed duplicates (EXACT_SHA +
# TEXTUALLY_NEAR_IDENTICAL at the calibrated threshold) are excluded from the
# moes_website source; everything uncertain/unique is preserved.
# ─────────────────────────────────────────────────────────────────────────────

_moes_dedup_cache: dict[str, object] = {}


def _moes_website_dedup_excludes() -> set[str]:
    """Filenames of confirmed-duplicate MoES PQ documents to exclude at ingest.

    Computed once per process (cached). Safe by default: missing corpora or any
    failure returns an empty set, so ingestion preserves everything rather than
    risk a false exclusion. Only applies to the `moes_website` source."""
    if "done" in _moes_dedup_cache:
        return _moes_dedup_cache["done"]  # type: ignore[return-value]
    try:
        from src.scraping.moes import dedup as _moes_dedup

        result = _moes_dedup.moes_website_dedup()
        _moes_dedup_cache["done"] = result.excluded_filenames
    except Exception:  # noqa: BLE001 — dedup is auxiliary; never fail ingestion
        _moes_dedup_cache["done"] = set()
    return _moes_dedup_cache["done"]  # type: ignore[return-value]


def choose_embed_action(total_added: int, no_rebuild: bool, full_rebuild: bool,
                        total_changed: int = 0) -> str:
    """Pure decision — mirrors ingest_folder.main()'s contract exactly:

      nothing added/changed  -> "skip"     (no index work at all)
      --no-rebuild           -> "defer"    (corpus updated; index untouched)
      --full-rebuild         -> "rebuild"  (explicit operator intent, ALWAYS —
                                            even with zero additions)
      changed > 0            -> "rebuild"  (changed rows already live in the
                                            index under old embeddings; FAISS
                                            has no in-place update — a full
                                            rebuild is the only correct path)
      no usable index        -> "rebuild"  (first build)
      otherwise              -> "incremental" (embed ONLY new records)

    ``total_changed`` defaults to 0 so legacy callers keep byte-identical
    behaviour. --no-rebuild beats the changed-trigger (operator deferral);
    --full-rebuild beats everything else (explicit intent).
    """
    if full_rebuild:
        return "rebuild"
    if total_added <= 0 and total_changed <= 0:
        return "skip"
    if no_rebuild:
        return "defer"
    if total_changed > 0:
        return "rebuild"
    if not _engine._index_exists():
        return "rebuild"
    return "incremental"


def run_embed_phase(action: str) -> None:
    if action == "rebuild":
        _engine.rebuild_index()
    elif action == "incremental":
        _engine.incremental_update()
    elif action == "defer":
        _engine.log("--no-rebuild given; index NOT updated (run "
                    "`python -m src.scripts.ingest all --ingest` without "
                    "--no-rebuild or `retrieve build --rebuild` later)")
    else:
        _engine.log("nothing new ingested — no index update needed")


def run_sources(
    specs: dict[str, SourceSpec],
    *,
    category_map: dict[str, str] | None = None,
    move_processed: bool | None = None,
    no_rebuild: bool = False,
    full_rebuild: bool = False,
) -> dict:
    """Ingest each source, then update the index exactly once at the end."""
    _sync_engine_paths()
    total_added = 0
    total_changed = 0
    per_source: dict[str, dict] = {}
    for name, spec in specs.items():
        _engine.log(f"=== ingest source: {name} (kind={spec.kind}"
                    + (", hierarchical" if spec.hierarchical else "")
                    + (f", subpath={spec.subpath}" if spec.subpath else "")
                    + (", discovered" if spec.discovered else "") + ") ===")
        res = ingest_source(spec, move_processed=move_processed, category_map=category_map)
        per_source[name] = res
        total_added += res.get("added", 0)
        total_changed += res.get("changed", 0)
    action = choose_embed_action(total_added, no_rebuild, full_rebuild,
                                 total_changed=total_changed)
    run_embed_phase(action)
    return {"added": total_added, "changed": total_changed,
            "embed": action, "sources": per_source}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_sources(config_file: str | None = None) -> None:
    registered, excludes, category_map = load_sources(config_file)
    discovered = discover_sources(registered, excludes)
    print("Registered sources (config/sources.yaml):")
    for name, spec in registered.items():
        locs = spec.folders or spec.record_dirs
        tags = []
        if spec.hierarchical:
            tags.append("hierarchical")
        if spec.move_processed:
            tags.append("move_processed")
        if spec.org:
            tags.append(f"org={spec.org}")
        if spec.ministry:
            tags.append(f"ministry={spec.ministry}")
        print(f"  {name:<14} kind={spec.kind:<8} "
              f"{'|'.join(locs) if locs else '(none)'}"
              + ("  [" + ", ".join(tags) + "]" if tags else ""))
        if spec.description:
            print(f"{'':16} {spec.description}")
    if discovered:
        print("Discovered (unregistered) hierarchical data directories:")
        for name, spec in discovered.items():
            print(f"  {name:<14} {name}/  [default_org={spec.default_org}, ministry unknown]")
    print(f"\nCategory map: {category_map}")
    print(f"Data root: {data_dir()}   Corpus: {corpus_path()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=None,
                    help="parliament | incois | moes | moes/<org>[/category] | "
                         "<discovered-source> | all")
    ap.add_argument("--ingest", action="store_true",
                    help="run ingestion for the source (default action when a source is given)")
    ap.add_argument("--list", dest="list_sources", action="store_true",
                    help="list registered + discovered sources and exit")
    ap.add_argument("--config", default=None,
                    help="override sources config (default: config/sources.yaml "
                         "or $INGEST_SOURCES_CONFIG)")
    ap.add_argument("--move-processed", action="store_true",
                    help="force moving ingested files to <folder>/processed (inbox default)")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="append to corpus but skip all index work")
    ap.add_argument("--full-rebuild", action="store_true",
                    help="FULL index rebuild afterwards (embeds EVERYTHING — explicit use only)")
    args = ap.parse_args()

    if args.list_sources or not args.source:
        _print_sources(args.config)
        if not args.source and not args.list_sources:
            print("\nPass a source name + --ingest (or `all --ingest`).")
        return

    try:
        specs, _, category_map = resolve_sources(args.source, args.config)
    except KeyError as e:
        print(f"[ingest] {e.args[0]}", file=sys.stderr)
        sys.exit(2)

    move = True if args.move_processed else None
    try:
        result = run_sources(
            specs,
            category_map=category_map,
            move_processed=move,
            no_rebuild=args.no_rebuild,
            full_rebuild=args.full_rebuild,
        )
    except RuntimeError as e:
        # incremental index update failed — the corpus append is intact and
        # append-only; the index is untouched; nothing was rebuilt silently.
        print(f"[ingest] ERROR: {e}", file=sys.stderr)
        sys.exit(3)
    print(f"[ingest] done: {result['added']} new record(s) appended"
          f", {result.get('changed', 0)} changed (replaced in corpus)"
          f"; index: {result['embed']}")


if __name__ == "__main__":
    main()
