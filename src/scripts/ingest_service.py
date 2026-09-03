"""Shared ingestion SERVICE — the single seam the FastAPI frontend and the
CLI converge on (Phase 3).

                         ┌───────────────┐
                         │ CLI ingestion │   src/scripts/ingest.py
                         └───────┬───────┘   (folder/tree scans)
                                 │
                                 ▼
    THIS MODULE  ──►  resolve_upload_target()  — hierarchy validation from the
    (shared service)    │                        SAME registry the CLI uses
                        │                        (load_sources/discover_sources
                        │                        + resolve_path_context parity)
                        ▼
                 engine: src/scripts/ingest_folder.py   (convert/dedup/append —
                 SAME functions the CLI calls; nothing re-implemented here)
                        ▼
                 update_index_in_process() — in-process incremental add_records
                 (SAME HybridRAGPipeline.add_records path the CLI's
                 incremental_update() subprocess runs; the in-process variant
                 exists so the server can hot-swap its live pipeline)

There is NO second conversion pipeline: the frontend flow writes the uploaded
file INTO the hierarchical source tree (the physical convention), then calls
the identical engine entry points the CLI uses. A file staged by
`resolve_upload_target` at ``root/<org>/<category>/`` resolves — under the
Phase-1 walker — to exactly the (org, document_type) the upload stamped, so
CLI re-scans and frontend uploads can never drift apart (pinned by tests).

Upload verdicts ("new" / "duplicate" / "failed") reuse the engine's own
per-file converter (`convert_one_detected`) as a probe — granular per-file
feedback without touching converter internals. New files are converted twice
(probe + engine run); conversion is cheap relative to embedding, and dedup
guarantees the record is appended exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import src.scripts.ingest as _registry
import src.scripts.ingest_folder as _engine
from src.utils.app_paths import corpus_path, data_dir, index_dir
from src.utils.atomic_io import write_bytes_atomic  # noqa: F401  (re-used by callers)
from src.utils.labels import slug_label

# Engine's documented document_type vocabulary (detect_doc_type docstring,
# convert_sirs_knowledge converters) plus the Phase-1 additive value. The
# category_map of the ACTIVE config further constrains hierarchical uploads;
# this full set additionally validates the optional inbox hint.
_ENGINE_DOC_TYPES = {
    "annual_report", "technical_report", "research_publication",
    "general_report", "audit_qa", "document", "audit_report",
}

# Upload guard rails — same policy as server /api/upload (the frontend must
# face ONE limit, whichever entry point it uses).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
MIN_UPLOAD_BYTES = 10                 # below this the file is effectively empty

# File-name listing cap for the tree endpoint (the UI shows a preview, not a
# full document-management platform — Phase 3 §8).
_TREE_FILES_CAP = 50


class UploadValidationError(Exception):
    """Operator-facing validation failure (mapped to HTTP 400/404)."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class UploadTarget:
    """A validated upload destination: a concrete leaf folder + the EXACT
    meta_context the Phase-1 walker would derive for that path."""

    source: str
    org: str | None                # resolved org slug (None for plain inbox)
    document_type: str | None      # doc_type the path/cat hint declares
    folder: Path                   # absolute leaf dir (created by the resolver)
    rel_path: str                  # data-root-relative display path
    meta_context: dict             # handed verbatim to engine.ingest_folder
    move_processed: bool = False
    hierarchical: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label(slug: str) -> str:
    """Deterministic display label for a slug (no config surface added):
    short slugs read as acronyms (incois -> INCOIS, moes_hq -> MOES HQ).

    Implementation lives in src.utils.labels (shared with the Phase-5
    config-driven source tree); alias kept for local readability."""
    return slug_label(slug)


def _doc_type_label(doc_type: str) -> str:
    return doc_type.replace("_", " ").title()


def _inverse_category_dir(category_map: dict[str, str], doc_type: str) -> str | None:
    """category_map inverted (sorted-first wins on duplicate values)."""
    for seg in sorted(category_map):
        if category_map[seg] == doc_type:
            return seg
    return None


def _leaf_files(leaf: Path) -> tuple[int, list[str], bool]:
    """(count, names capped, truncated?) of ingestible files directly in leaf."""
    if not leaf.is_dir():
        return 0, [], False
    names = sorted(
        f.name for f in leaf.iterdir()
        if f.is_file() and f.suffix.lower() in _registry.INGESTIBLE_EXTS
    )
    return len(names), names[:_TREE_FILES_CAP], len(names) > _TREE_FILES_CAP


# ─────────────────────────────────────────────────────────────────────────────
# 1. Discovery — the authoritative hierarchy view for the frontend (and for
#    operators). Built ONLY from the active registry + the filesystem.
# ─────────────────────────────────────────────────────────────────────────────

def discover_ingest_tree(config_file: str | Path | None = None) -> dict:
    """Hierarchy view for `GET /api/ingest/targets` — registered ∪ discovered
    sources, orgs (org_map ∪ actual dirs), categories (category_map), with
    file previews for existing leaves.

    ONE authoritative source: config/sources.yaml + the data/ tree, via the
    same loaders the CLI uses. The frontend hardcodes nothing.
    """
    registered, excludes, category_map = _registry.load_sources(config_file)
    discovered = _registry.discover_sources(registered, excludes)

    doc_types = sorted(set(category_map.values()))
    sources: list[dict] = []

    all_specs = list(registered.values()) + list(discovered.values())
    for spec in all_specs:
        if spec.kind != "folders":
            continue  # records sources (parliament) are not upload targets
        entry: dict = {
            "name": spec.name,
            "label": _label(spec.name),
            "description": spec.description,
            "hierarchical": spec.hierarchical,
            "discovered": spec.discovered,
            "ministry": spec.ministry,
        }
        if not spec.hierarchical:
            # flat upload target: only the inbox line is operator-actionable
            # (move_processed marks the UI-owned flow); crawler-owned flat
            # sources are listed as unavailable so the UI can ignore them.
            entry["upload"] = spec.name == "inbox"
            entry["orgs"] = []
            sources.append(entry)
            continue

        root = _registry._data_path(spec.folders[0])
        org_dirs: dict[str, str] = {}   # org slug -> org dir segment (on disk)
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                if d.name in _registry._WALK_SKIP_DIRS:
                    continue
                norm = _registry._normalize_segment(d.name)
                if norm in category_map:
                    continue  # category leaf, not an org dir
                org_dirs[spec.org_map.get(norm, norm)] = d.name

        org_slugs = set(spec.org_map.values()) | set(org_dirs.keys())
        if spec.default_org:
            org_slugs.add(spec.default_org)
        orgs: list[dict] = []
        for slug in sorted(org_slugs):
            seg = _org_dir_segment(spec, slug, category_map)
            leaf_base = root if seg is None else root / seg
            cats = []
            for doc_type in doc_types:
                cat_dir = _inverse_category_dir(category_map, doc_type)
                leaf = leaf_base / cat_dir
                n, names, truncated = _leaf_files(leaf)
                cats.append({
                    "document_type": doc_type,
                    "label": _doc_type_label(doc_type),
                    "category_dir": cat_dir,
                    "path": (f"{spec.name}/{cat_dir}" if seg is None
                             else f"{spec.name}/{seg}/{cat_dir}"),
                    "exists": leaf.is_dir(),
                    "files": n,
                    "file_names": names,
                    "truncated": truncated,
                })
            orgs.append({"slug": slug, "label": _label(slug),
                         "dir": seg, "categories": cats})
        entry["upload"] = True
        entry["orgs"] = orgs
        sources.append(entry)

    return {
        "version": 1,
        "category_map": category_map,
        "document_types": doc_types,
        "data_root": str(data_dir()),
        "sources": sources,
    }


def _org_dir_segment(spec: _registry.SourceSpec, org_slug: str,
                     category_map: dict[str, str]) -> str | None:
    """Canonical on-disk org segment for an org slug (upload placement).

    Priority: org_map KEY (the config's canonical spelling) > first org_map
    key whose value matches > existing dir name > None (root-level placement,
    valid only for default_org — walker parity: root/<category> resolves to
    default_org)."""
    norm = _registry._normalize_segment(org_slug)
    if norm in spec.org_map:
        return norm
    for key in sorted(spec.org_map):
        if spec.org_map[key] == norm:
            return key
    root = _registry._data_path(spec.folders[0])
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and _registry._normalize_segment(d.name) == norm \
                    and norm not in category_map \
                    and d.name not in _registry._WALK_SKIP_DIRS:
                return d.name
    if spec.default_org and norm == _registry._normalize_segment(spec.default_org):
        return None
    return None  # caller already validated; None = root-level fallback


# ─────────────────────────────────────────────────────────────────────────────
# 2. Upload-target resolution + validation (operator-facing messages)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_upload_target(
    source: str | None,
    org: str | None = None,
    document_type: str | None = None,
    *,
    config_file: str | Path | None = None,
) -> UploadTarget:
    """Validate (source, org, document_type) against the ACTIVE registry and
    return the concrete leaf folder + meta_context to ingest with.

    Raises UploadValidationError with a clear operator message on any
    invalid/missing selection. Creates the leaf directory (uploads make the
    hierarchy real — the path convention IS the metadata).
    """
    if not source or not source.strip():
        raise UploadValidationError("Source is required")
    name = source.strip()

    registered, excludes, category_map = _registry.load_sources(config_file)
    spec = registered.get(name)
    if spec is None:
        discovered = _registry.discover_sources(registered, excludes)
        spec = discovered.get(name)
    if spec is None:
        known = sorted(
            s.name for s in (*registered.values(), *discovered.values())
            if s.kind == "folders" and (s.hierarchical or s.name == "inbox")
        )
        raise UploadValidationError(
            f"Unknown source '{name}'. Known upload targets: {', '.join(known) or '(none)'}",
            status_code=404,
        )
    if spec.kind != "folders":
        raise UploadValidationError(
            f"Source '{name}' is a records source ({spec.kind}); it is fed by the "
            "parliament pipeline, not by uploads."
        )

    # ── Flat source: only the inbox flow (UI-owned, move_processed) ─────────
    if not spec.hierarchical:
        if spec.name != "inbox":
            raise UploadValidationError(
                f"Source '{name}' is a crawler-managed flat source — uploads go to a "
                f"hierarchical source (e.g. moes/<org>/<category>) or to inbox."
            )
        hint = None
        if document_type and document_type.strip():
            hint = document_type.strip()
            if hint not in _ENGINE_DOC_TYPES:
                raise UploadValidationError(
                    f"Invalid document type '{hint}'. Valid: {', '.join(sorted(_ENGINE_DOC_TYPES))}"
                )
        folder = _registry._data_path(spec.folders[0])
        folder.mkdir(parents=True, exist_ok=True)
        # No context for a plain inbox upload: byte-identical records to the
        # legacy flow. org/document_type params add an additive hint only.
        ctx = None
        if hint or (org and org.strip()):
            slug = _registry._normalize_segment(org) if org and org.strip() else None
            ctx = {"org": slug, "source": spec.name, "ministry": spec.ministry,
                   # mirror expand_source's flat-source rule (legacy default)
                   "default_ministry": (None if spec.ministry
                                        else _registry._LEGACY_DEFAULT_MINISTRY),
                   "doc_type_hint": hint}
        return UploadTarget(
            source=spec.name, org=spec.org, document_type=hint,
            folder=folder, rel_path=spec.folders[0],
            meta_context=ctx or {},
            move_processed=spec.move_processed, hierarchical=False,
        )

    # ── Hierarchical source: org + document_type are authoritative ──────────
    if not org or not org.strip():
        raise UploadValidationError(f"Organization is required for source '{name}'")
    norm_org = _registry._normalize_segment(org)

    root = _registry._data_path(spec.folders[0])
    existing_dirs = set()
    if root.is_dir():
        existing_dirs = {
            _registry._normalize_segment(d.name)
            for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name not in _registry._WALK_SKIP_DIRS
        }
    known_orgs = (
        set(spec.org_map) | set(spec.org_map.values())
        | {d for d in existing_dirs if d not in category_map}
        | ({spec.default_org} if spec.default_org else set())
    )
    if norm_org not in {_registry._normalize_segment(o) for o in known_orgs if o}:
        raise UploadValidationError(
            f"Unknown organization '{org}' for source '{name}'. "
            f"Known: {', '.join(sorted(o for o in known_orgs if o))}"
        )
    org_slug = spec.org_map.get(norm_org, norm_org)

    if not document_type or not document_type.strip():
        raise UploadValidationError(f"Document type is required for source '{name}'")
    doc_type = document_type.strip()
    cat_dir = _inverse_category_dir(category_map, doc_type)
    if cat_dir is None:
        raise UploadValidationError(
            f"Invalid document type '{doc_type}'. Valid for source '{name}': "
            f"{', '.join(sorted(set(category_map.values())))} "
            "(extend hierarchy.category_map in config/sources.yaml for more)"
        )

    seg = _org_dir_segment(spec, org_slug, category_map)
    leaf = (root / cat_dir) if seg is None else (root / seg / cat_dir)
    leaf.mkdir(parents=True, exist_ok=True)
    rel = leaf.relative_to(root.parent).as_posix() if _is_under(leaf, root) else str(leaf)

    meta_context = {
        "org": org_slug,
        "source": spec.name,
        "ministry": spec.ministry,
        # mirror expand_source._ctx: a configured ministry stamps + defaults;
        # hierarchical/discovered without ministry keeps None (no legacy label)
        "default_ministry": spec.ministry,
        "doc_type_hint": doc_type,
    }
    return UploadTarget(
        source=spec.name, org=org_slug, document_type=doc_type,
        folder=leaf, rel_path=rel, meta_context=meta_context,
        move_processed=False, hierarchical=True,
    )


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Staged-upload ingestion — engine delegation + per-file verdicts
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_corpus_ids() -> set[str]:
    """Current corpus question_ids (the dedup baseline for verdicts)."""
    ids: set[str] = set()
    corpus = corpus_path()
    if corpus.exists():
        for line in corpus.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("question_id"):
                    ids.add(r["question_id"])
            except Exception:  # noqa: BLE001 — tolerate a bad historical line
                continue
    return ids


def classify_upload(path: Path, meta_context: dict, corpus_ids: set[str]) -> dict:
    """Per-file verdict using the ENGINE's own converter as the probe — no
    parallel conversion logic. One probe; the authoritative append then
    happens in ingest_folder (its corpus-seeded dedup decides for real).

    verdicts: new | duplicate | failed | skipped_duplicate_pdf
    """
    # Engine parity: a .txt next to a same-stem .pdf is skipped (crawl
    # convention — the PDF keeps full fidelity).
    if path.suffix.lower() == ".txt":
        if (path.parent / (path.name.rsplit(".", 1)[0] + ".pdf")).exists():
            return {"name": path.name, "verdict": "skipped_duplicate_pdf",
                    "records": 0,
                    "message": "Skipped — a PDF with the same name is preferred"}
    probe_out: list = []
    probe_seen: set[str] = set()
    try:
        n = _engine.convert_one_detected(path, probe_out, probe_seen, False,
                                         meta_context or None)
    except Exception as e:  # noqa: BLE001 — verdict, not a crash
        return {"name": path.name, "verdict": "failed", "records": 0,
                "message": f"Failed to read document: {e}"}
    if n == 0:
        return {"name": path.name, "verdict": "failed", "records": 0,
                "message": "No extractable content — invalid or unreadable document"}
    ids = {r.question_id for r in probe_out
           if getattr(r, "question_id", None)}
    if ids and ids.issubset(corpus_ids):
        return {"name": path.name, "verdict": "duplicate", "records": 0,
                "message": "Document already exists — skipped"}
    return {"name": path.name, "verdict": "new",
            "records": len(ids - corpus_ids),
            "message": "Document uploaded successfully"}


def ingest_uploaded_files(
    target: UploadTarget, filenames: list[str]
) -> dict:
    """Ingest previously-staged files through the ENGINE (the same
    ingest_folder the CLI drives), scoped to exactly those files.

    Returns aggregate counts + per-file verdicts. NEVER moves hierarchical
    files (the tree is the physical record); inbox semantics come from
    target.move_processed.
    """
    _engine.CORPUS = corpus_path()
    _engine.LOG = data_dir() / "sync.log"
    _engine.INDEX_DIR = str(index_dir())

    corpus_ids = snapshot_corpus_ids()
    verdicts = [
        classify_upload(target.folder / fname, target.meta_context, corpus_ids)
        for fname in filenames
    ]
    res = _engine.ingest_folder(
        str(target.folder),
        move_processed=target.move_processed,
        meta_context=target.meta_context or None,
        only_files=set(filenames),
    )
    added = res.get("added", 0)
    # A file the probe called "new" but the engine appended nothing for was
    # swallowed by the engine's own dedup in the meantime — recount honestly.
    new_files = sum(1 for v in verdicts if v["verdict"] == "new")
    dup_files = sum(1 for v in verdicts if v["verdict"] in ("duplicate", "skipped_duplicate_pdf"))
    failed_files = sum(1 for v in verdicts if v["verdict"] == "failed")
    return {
        "received": len(filenames),
        "new": new_files,
        "duplicates": dup_files,
        "failed": failed_files,
        "records_added": added,
        "engine": {"files": res.get("files", 0), "failed": res.get("failed", 0),
                   "types": res.get("types", {})},
        "files": verdicts,
        "target": {"source": target.source, "org": target.org,
                   "document_type": target.document_type, "path": target.rel_path},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Index update — in-process incremental (server hot-swap variant of the
#    engine's incremental_update() subprocess; SAME add_records semantics)
# ─────────────────────────────────────────────────────────────────────────────

# The index-completeness contract is canonical and shared (audit IW-11):
# src/retrieval/hybrid/artifacts.py — never re-declare marker lists here.
from src.retrieval.hybrid.artifacts import index_is_complete as _index_is_complete


def update_index_in_process(pipeline_factory=None):
    """Load the existing index, embed ONLY new corpus records, save; or build
    fresh when no usable index exists. Returns (n_embedded, pipeline_or_None).

    Extracted verbatim-in-spirit from server._run_inbox_ingest so the upload
    flow and the inbox flow share ONE implementation. `pipeline_factory` is a
    test seam (deterministic embedder injection); production uses
    HybridRAGPipeline.
    """
    from src.data.loader import DataLoader

    corpus = corpus_path()
    idx = index_dir()
    if not corpus.exists():
        return 0, None
    if pipeline_factory is None:
        from src.retrieval.hybrid.pipeline import HybridRAGPipeline

        pipeline_factory = HybridRAGPipeline
    records = DataLoader.load_jsonl(corpus)
    try:
        if _index_is_complete(idx):
            pipe = pipeline_factory()
            pipe.load(str(idx))
            n = int(pipe.add_records(records) or 0)
            if n:
                pipe.save(str(idx))
            _engine.log(f"[ingest] incremental update: {n} new record(s) embedded+added")
            return n, pipe
        pipe = pipeline_factory(records=records)
        pipe.save(str(idx))
        _engine.log(f"[ingest] full build with {len(records):,} records")
        return len(records), pipe
    finally:
        # HPC ingestion policy (#5): embedding models may use the GPU during
        # the build — release them when the build finishes so serving keeps
        # the GPU for vLLM. Models lazily reload (CPU) on next use of the
        # swapped-in pipeline, so this is safe. Guarded for test fakes.
        try:
            from src.retrieval.hybrid.embedder import release_embedding_models

            embedder = getattr(locals().get("pipe"), "embedder", None)
            if embedder is not None:
                embedder.release()
            release_embedding_models()
        except Exception:  # noqa: BLE001 - release must never fail ingest
            pass
