"""Source-aware ingestion CLI — ONE engine, many sources.

    python -m src.scripts.ingest parliament --ingest
    python -m src.scripts.ingest incois --ingest
    python -m src.scripts.ingest moes --ingest
    python -m src.scripts.ingest <future-source> --ingest
    python -m src.scripts.ingest all --ingest
    python -m src.scripts.ingest --list            # show resolved sources

    (the --ingest flag is the default action: `<source>` alone also ingests)

Architecture (deliberately thin — no ingestion logic lives here):

    source CLI (this file)           registry + discovery + per-source routing
          ↓ delegates
    src/scripts/ingest_folder.py     THE engine: detect_doc_type -> convert
                                     (shared convert_sirs_knowledge
                                     converters) -> dedup -> atomic append
                                     to data/corpus_reports.jsonl
          ↓ reused verbatim
    incremental_update()             load existing index, embed ONLY new
                                     records (HybridRAGPipeline.add_records
                                     appends FAISS vectors, rebuilds the
                                     text-only BM25 side — existing vectors
                                     untouched)
    rebuild_index()                  FULL rebuild — only --full-rebuild or
                                     when no usable index exists.

Guarantees (same contract as ingest_folder):
  * corpus writes are append-only to data/corpus_reports.jsonl
    (deterministic ids -> scanning the same source twice adds nothing);
  * normal ingestion NEVER triggers a full re-embed of the corpus;
  * frontend inbox flow is untouched — `inbox` is a first-class source whose
    semantics (move-to-processed) match what server /api/ingest already does.

Sources come from config/sources.yaml (data, not code). Unknown names fall
back to filesystem discovery: a first-level, non-reserved directory under the
data root containing ingestible files is ingestible by directory name — a new
ministry needs at most a config entry, usually not even that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The engine. Reused, never re-implemented: conversion, dedup, atomic append,
# incremental/full index updates. Do NOT fork these functions into this file
# (the server imports them too — a second copy would drift).
import src.scripts.ingest_folder as _engine
from src.models.qa_record import QARecord
from src.utils.app_paths import config_path, corpus_path, data_dir, index_dir, project_root
from src.utils.atomic_io import append_jsonl_atomic

INGESTIBLE_EXTS = {".pdf", ".txt", ".md", ".json", ".jsonl"}

# Directories that must never be swallowed by discovery (registry-claimed
# roots are excluded separately, dynamically).
_ALWAYS_EXCLUDED_DIRS = {"raw", "finetune", "user-knowledge"}


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


# Fallback used ONLY if config/sources.yaml is unreadable — must stay equal to
# the shipped config/sources.yaml `sources:` section (asserted by tests).
_BUILTIN_SOURCES: dict[str, dict] = {
    "inbox": {
        "kind": "folders", "folders": ["inbox"], "move_processed": True,
        "description": "Frontend uploads / manual drop-ins.",
    },
    "parliament": {
        "kind": "records", "record_dirs": ["enriched", "processed"],
        "description": "Phase-1 pipeline output (data/enriched, data/processed).",
    },
    "incois": {
        "kind": "folders",
        "folders": [
            "annual_reports",
            "incois_reports/AnnualReports",
            "incois_reports/Others",
            "incois_reports/TechnicalReports",
            "incois_reports/ResearchPublications",
            "scanned_ocr",
        ],
        "description": "INCOIS public reports + scanned OCR text.",
    },
    "moes": {
        "kind": "folders", "folders": ["moes_reports/knowledge"],
        "description": "MoES CCPS knowledge JSONs (crawl_moes_reports).",
    },
}

_BUILTIN_DISCOVERY_EXCLUDES = ["processed", "enriched", "raw", "finetune", "user-knowledge"]


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
    )


def load_sources(config_file: str | Path | None = None) -> tuple[dict[str, SourceSpec], set[str]]:
    """Load the source registry + discovery excludes.

    Returns (sources, discovery_excludes). Falls back to the built-in mirror
    when the config file is missing/unreadable — the CLI must stay usable on a
    bare checkout.
    """
    cfg = Path(config_file) if config_file else _default_config_file()
    sources: dict[str, SourceSpec] = {}
    excludes: set[str] = set(_ALWAYS_EXCLUDED_DIRS)
    if cfg.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            for name, raw in (data.get("sources") or {}).items():
                if isinstance(raw, dict):
                    sources[str(name)] = _spec_from_dict(str(name), raw)
            disc = data.get("discovery") or {}
            excludes.update(str(d) for d in (disc.get("exclude_dirs") or []))
        except Exception as e:  # noqa: BLE001 — config must never kill ingestion
            _engine.log(f"[ingest] WARN: {cfg} unreadable ({e}) — using built-in source registry")
            sources.clear()
    if not sources:
        sources = {name: _spec_from_dict(name, raw) for name, raw in _BUILTIN_SOURCES.items()}
        excludes.update(_BUILTIN_DISCOVERY_EXCLUDES)
    return sources, excludes


def _data_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    return p if p.is_absolute() else data_dir() / p


def _dir_has_ingestible_files(d: Path) -> bool:
    try:
        return any(
            f.is_file() and f.suffix.lower() in INGESTIBLE_EXTS
            for f in d.iterdir()
        )
    except OSError:
        return False


def discover_sources(
    registered: dict[str, SourceSpec], excludes: set[str]
) -> dict[str, SourceSpec]:
    """First-level data directories that are not claimed by the registry and
    directly contain ingestible files -> discovered `folders` sources.

    Deterministic: sorted by directory name. A registered source's declared
    folders claim their path; the top-level owner (`incois_reports` for
    `incois_reports/AnnualReports`) is also claimed so nested structures are
    never double-ingested by discovery.
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
        if _dir_has_ingestible_files(d):
            found[d.name] = SourceSpec(
                name=d.name,
                kind="folders",
                folders=[d.name],
                description="discovered data directory (unregistered source)",
                discovered=True,
            )
    return found


def resolve_sources(
    which: str, config_file: str | Path | None = None
) -> tuple[dict[str, SourceSpec], set[str]]:
    """Resolve `which` ("all" or a source name) to an ordered source mapping.

    Unknown names fall back to filesystem discovery (the future-ministry
    path); a name that is neither registered nor discoverable is an error
    with the known source names listed.
    """
    registered, excludes = load_sources(config_file)
    if which.lower() == "all":
        merged = dict(registered)
        merged.update(discover_sources(registered, excludes))
        return merged, excludes
    if which in registered:
        return {which: registered[which]}, excludes
    discovered = discover_sources(registered, excludes)
    if which in discovered:
        return {which: discovered[which]}, excludes
    known = sorted(registered) + sorted(discovered)
    raise KeyError(
        f"unknown source '{which}'. Known sources: {', '.join(known) or '(none)'} "
        f"(or create data/{which}/ with ingestible files and retry)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-source ingestion (delegation only)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_engine_paths() -> None:
    """Pin the engine's import-time globals to the CURRENT app_paths.

    Same pattern the server already uses (server._run_inbox_ingest): it makes
    APP_DATA_DIR / APP_INDEX_DIR overrides authoritative for this process even
    if the env changed between import and invocation.
    """
    _engine.CORPUS = corpus_path()
    _engine.LOG = data_dir() / "sync.log"
    _engine.INDEX_DIR = str(index_dir())


def _seed_seen_from_corpus() -> set[str]:
    """Existing corpus question_ids — deterministic cross-run dedup.

    Identical rule as ingest_folder.ingest_folder's seeding (kept as a helper
    here so the records-merge path and the engine share the exact source).
    """
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


def merge_record_dirs(dirs: list[str], out: list[QARecord], seen: set[str]) -> int:
    """Merge ready-made QARecord JSONL (Phase-1 parliament output) into `out`.

    Ids are PRESERVED (not re-hashed): these records were produced by the
    Phase-1 ingestion pipeline with parliament-id semantics (e.g. 18-4-3035).
    Declaration order wins on duplicate ids (config lists enriched before
    processed — the richer copy survives).
    """
    added = 0
    for rel in dirs:
        d = _data_path(rel)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            try:
                for line in f.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = QARecord.model_validate_json(line)
                    except Exception:  # noqa: BLE001 — skip malformed lines
                        continue
                    if rec.question_id not in seen:
                        seen.add(rec.question_id)
                        out.append(rec)
                        added += 1
            except OSError as e:
                _engine.log(f"  [warn] {f}: {e}")
    return added


def ingest_source(spec: SourceSpec, move_processed: bool | None = None) -> dict:
    """Ingest ONE source through the existing engine. Returns stats."""
    if spec.kind == "records":
        out: list[QARecord] = []
        seen = _seed_seen_from_corpus()
        added = merge_record_dirs(spec.record_dirs, out, seen)
        if out:
            lines = [rec.model_dump_json() for rec in out]
            append_jsonl_atomic(corpus_path(), lines)
            _engine.log(f"[ingest:{spec.name}] appended {added} record(s) -> {corpus_path()}")
        else:
            _engine.log(f"[ingest:{spec.name}] no new records in {spec.record_dirs}")
        return {"added": added, "folders": 0}

    # kind == "folders" — the proven engine path (detect -> convert -> dedup
    # -> atomic append). Folder-level dedup against the corpus happens inside
    # the engine on every call.
    move = spec.move_processed if move_processed is None else move_processed
    total = files = failed = 0
    scanned = 0
    for rel in spec.folders:
        folder = _data_path(rel)
        if not folder.exists():
            _engine.log(f"[ingest:{spec.name}] folder not found: {folder} — skipped")
            continue
        scanned += 1
        res = _engine.ingest_folder(str(folder), move_processed=move)
        total += res.get("added", 0)
        files += res.get("files", 0)
        failed += res.get("failed", 0)
    if scanned == 0:
        _engine.log(f"[ingest:{spec.name}] no folders on disk for this source — nothing to do")
    return {"added": total, "folders": scanned, "files": files, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# Index update — the reuse seam (delegates to the engine, identity-pinned)
# ─────────────────────────────────────────────────────────────────────────────

# Re-exported so tests can pin identity: the CLI MUST reuse ingest_folder's
# incremental/full index implementations, not grow its own.
embed_incremental = _engine.incremental_update
embed_full_rebuild = _engine.rebuild_index
index_is_usable = _engine._index_exists


def choose_embed_action(total_added: int, no_rebuild: bool, full_rebuild: bool) -> str:
    """Pure decision — mirrors ingest_folder.main()'s contract exactly:

      nothing added        -> "skip"     (no index work at all)
      --no-rebuild         -> "defer"    (records appended; index untouched)
      --full-rebuild       -> "rebuild"  (explicit operator intent only)
      no usable index      -> "rebuild"  (first build)
      otherwise            -> "incremental" (embed ONLY new records)
    """
    if total_added <= 0:
        return "skip"
    if no_rebuild:
        return "defer"
    if full_rebuild or not _engine._index_exists():
        return "rebuild"
    return "incremental"


def run_embed_phase(action: str) -> None:
    if action == "rebuild":
        _engine.rebuild_index()
    elif action == "incremental":
        _engine.incremental_update()
    elif action == "defer":
        _engine.log("--no-rebuild given; index NOT updated (run `python -m src.scripts.ingest all --ingest` "
                    "without --no-rebuild or `retrieve build --rebuild` later)")
    else:
        _engine.log("nothing new ingested — no index update needed")


def run_sources(
    specs: dict[str, SourceSpec],
    *,
    move_processed: bool | None = None,
    no_rebuild: bool = False,
    full_rebuild: bool = False,
) -> dict:
    """Ingest each source, then update the index exactly once at the end."""
    _sync_engine_paths()
    total_added = 0
    per_source: dict[str, dict] = {}
    for name, spec in specs.items():
        _engine.log(f"=== ingest source: {name} (kind={spec.kind}"
                    + (", discovered" if spec.discovered else "") + ") ===")
        res = ingest_source(spec, move_processed=move_processed)
        per_source[name] = res
        total_added += res.get("added", 0)
    action = choose_embed_action(total_added, no_rebuild, full_rebuild)
    run_embed_phase(action)
    return {"added": total_added, "embed": action, "sources": per_source}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_sources(config_file: str | None = None) -> None:
    registered, excludes = load_sources(config_file)
    discovered = discover_sources(registered, excludes)
    print("Registered sources (config/sources.yaml):")
    for name, spec in registered.items():
        locs = spec.folders or spec.record_dirs
        print(f"  {name:<12} kind={spec.kind:<8} "
              f"{'|'.join(locs) if locs else '(none)'}"
              + ("  [move_processed]" if spec.move_processed else ""))
        if spec.description:
            print(f"{'':14} {spec.description}")
    if discovered:
        print("Discovered (unregistered) data directories:")
        for name, spec in discovered.items():
            print(f"  {name:<12} kind=folders {name}/")
    print(f"\nData root: {data_dir()}   Corpus: {corpus_path()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=None,
                    help="parliament | incois | moes | <discovered-source> | all")
    ap.add_argument("--ingest", action="store_true",
                    help="run ingestion for the source (default action when a source is given)")
    ap.add_argument("--list", dest="list_sources", action="store_true",
                    help="list registered + discovered sources and exit")
    ap.add_argument("--config", default=None,
                    help="override sources config (default: config/sources.yaml or $INGEST_SOURCES_CONFIG)")
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
        specs, _ = resolve_sources(args.source, args.config)
    except KeyError as e:
        print(f"[ingest] {e.args[0]}", file=sys.stderr)
        sys.exit(2)

    move = True if args.move_processed else None
    result = run_sources(
        specs,
        move_processed=move,
        no_rebuild=args.no_rebuild,
        full_rebuild=args.full_rebuild,
    )
    print(f"[ingest] done: {result['added']} new record(s) appended; "
          f"index: {result['embed']}")


if __name__ == "__main__":
    main()
