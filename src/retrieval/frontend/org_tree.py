"""Ministry -> org hierarchy + metadata derivation helpers.

CONFIGURATION-DRIVEN TREE
--------------------------
``config/sources.yaml`` is the single source of truth. This module DERIVES
the tree (and the document_type -> doc_category mapping) from it:

    sources.yaml                     ONE authoritative definition
      ├─ sources:            (registered hierarchical sources)
      └─ presentation:       (labels, extra ministries, category mappings)
                    │
                    ▼  loader below (merged; presentation labels win)
    ORG_TREE / _DT_CATEGORY          derived, refreshed on config mtime change
                    │
                    ▼
    build_sources_catalogue(records) counts ALWAYS from the indexed corpus
                    │
                    ▼
    /api/sources -> frontend (SourceFilter renders whatever arrives)

Adding a new hierarchical source to ``sources:`` (e.g. ``rajat_reports``)
makes it appear in the catalogue automatically — label derived deterministically
from the slug (``rajat_reports`` -> ``Rajat Reports`` via
``src.utils.labels.slug_label``); a ``presentation.ministries`` entry is needed
only to override the label/category hints. No Python edit, no frontend edit.

COMPATIBILITY: exported names are unchanged (``ORG_TREE``, ``_DT_CATEGORY``,
``derive_org``, ``derive_category``, ``build_sources_catalogue``,
``resolve_orgs``). If the config is missing or invalid, an exception is
raised immediately — stale snapshot fallback has been removed. A config
that is STRUCTURALLY invalid raises ``ValueError`` naming the offending key.

TREE RULE (unchanged)
---------------------
- Selecting a MINISTRY includes ALL orgs under it (like a folder tree).
- Org slugs are globally unique across ministries -> flat filter logic.
- "sansad" is a special top-level source: parliamentary Q&A is a separate
  bucket (questions ABOUT ministries, not documents PRODUCED by them).

DERIVATION (unchanged)
----------------------
derive_org()/derive_category() fill missing metadata deterministically
(document_type, subject, source_url, session, ...) so the WHOLE corpus stays
filterable without a data rewrite or index rebuild.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.app_paths import config_path
from src.utils.labels import slug_label

# ─────────────────────────────────────────────────────────────────────────────
# Shipped snapshot — used as the initial state and for label/category hints.
# This is populated at startup from sources.yaml; if the file is missing,
# startup raises immediately. Edit the YAML, not this.
# ─────────────────────────────────────────────────────────────────────────────

_SHIPPED_TREE: dict[str, dict] = {
    "sansad": {
        "name": "Sansad (Parliament Q&A)",
        "orgs": [
            {"slug": "sansad", "name": "Parliamentary Questions", "categories": ["parliamentary"]},
        ],
    },
    "moes": {
        "name": "Ministry of Earth Sciences",
        "orgs": [
            {"slug": "incois", "name": "INCOIS",
             "categories": ["annual", "scientific", "technical", "general"]},
            {"slug": "imd", "name": "IMD",
             "categories": ["annual", "monthly", "scientific"]},
            {"slug": "iitm", "name": "IITM", "categories": ["annual", "scientific"]},
            {"slug": "niot", "name": "NIOT", "categories": ["annual", "research"]},
            {"slug": "moes_hq", "name": "MoES HQ",
             "categories": ["annual", "monthly", "budget", "policy",
                            "gazette", "news", "scientific", "misc"]},
        ],
    },
    "moa": {
        "name": "Ministry of Agriculture & Farmers Welfare",
        "orgs": [
            {"slug": "moa_hq", "name": "MoA HQ",
             "categories": ["annual", "monthly", "budget", "policy"]},
        ],
    },
    "mof": {
        "name": "Ministry of Finance",
        "orgs": [
            {"slug": "mof_hq", "name": "MoF HQ", "categories": ["annual", "budget", "policy"]},
        ],
    },
}

_SHIPPED_DT_CATEGORY: dict[str, str] = {
    "parliamentary_qa": "parliamentary",
    "annual_report": "annual",
    "monthly_report": "monthly",
    "research_publication": "scientific",
    "publication": "scientific",
    "bibliometrics": "scientific",
    "technical_report": "technical",
    "general_report": "general",
    "audit_report": "audit",
    "demands_for_grants": "budget",
    "performance_budget": "budget",
    "gazette_notification": "gazette",
    "order_notice": "gazette",
    "newsletter": "news",
    "press_release": "news",
    "download": "misc",
    "document": "misc",
}

# doc_category -> UI label fallback (mirrors presentation.categories; the
# frontend keeps its own legacy map as an additional fallback for older APIs).
_SHIPPED_CATEGORY_LABELS: dict[str, str] = {
    "parliamentary": "Parliamentary Questions",
    "annual": "Annual Reports",
    "monthly": "Monthly Reports",
    "quarterly": "Quarterly Reports",
    "scientific": "Scientific / Research",
    "technical": "Technical Reports",
    "general": "General Reports",
    "audit": "Audit Reports",
    "budget": "Budget / Grants",
    "policy": "Policy Documents",
    "gazette": "Gazettes / Notices",
    "news": "Newsletters / News",
    "misc": "Misc",
}

# Derived state (populated in place so `from … import ORG_TREE` keeps working).
ORG_TREE: dict[str, dict] = {}
_DT_CATEGORY: dict[str, str] = {}
_CATEGORY_LABELS: dict[str, str] = {}

_STATE: dict = {"key": None}


def sources_config_path() -> Path:
    """Active sources registry path. Override: ``SOURCES_CONFIG`` (tests/dev)."""
    raw = (os.environ.get("SOURCES_CONFIG") or "").strip()
    return Path(raw).expanduser() if raw else config_path("sources.yaml")


def config_stamp() -> tuple[str, int, int]:
    """(path, mtime_ns, size) of the active config — used for cache keys
    (size guards coarse filesystems where mtime granularity is low; cross-
    platform note: NTFS/ext4 both fine, this is belt-and-braces)."""
    path = sources_config_path()
    try:
        st = path.stat()
        return str(path), st.st_mtime_ns, st.st_size
    except OSError:
        return str(path), 0, 0


def _read_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _require(cond: bool, key_path: str, what: str) -> None:
    if not cond:
        raise ValueError(f"sources.yaml presentation: `{key_path}` must be {what}")


def _parse_presentation(data: dict) -> tuple[dict, dict[str, str], dict[str, str]]:
    """Validate the `presentation:` block; ValueError names the offending key."""
    pres = data.get("presentation") or {}
    _require(isinstance(pres, dict), "presentation", "a mapping")
    ministries = pres.get("ministries") or {}
    _require(isinstance(ministries, dict), "presentation.ministries", "a mapping")
    for mid, entry in ministries.items():
        kp = f"presentation.ministries.{mid}"
        _require(isinstance(entry, dict), kp, "a mapping")
        _require(isinstance(entry.get("name"), str), f"{kp}.name", "a string")
        orgs = entry.get("orgs") or {}
        _require(isinstance(orgs, dict), f"{kp}.orgs", "a mapping")
        for slug, org in orgs.items():
            op = f"{kp}.orgs.{slug}"
            if isinstance(org, str):
                continue  # short form: slug: "Display Name"
            _require(isinstance(org, dict), op, "a string or mapping")
            _require(isinstance(org.get("name"), str), f"{op}.name", "a string")
            cats = org.get("categories", [])
            _require(
                isinstance(cats, list) and all(isinstance(c, str) for c in cats),
                f"{op}.categories", "a list of strings",
            )
    for section in ("categories", "doctype_categories"):
        mapping = pres.get(section) or {}
        _require(isinstance(mapping, dict), f"presentation.{section}", "a mapping")
        for k, v in mapping.items():
            _require(isinstance(v, str), f"presentation.{section}.{k}", "a string")
    return (
        ministries,
        dict(pres.get("categories") or {}),
        dict(pres.get("doctype_categories") or {}),
    )


def _registry_hierarchical_sources(path: Path) -> list:
    """Registered hierarchical folder sources (reuses the CLI loader verbatim,
    so the tree derives from the SAME registry ingestion uses)."""
    try:
        from src.scripts.ingest import load_sources

        sources, _, _ = load_sources(path)
    except Exception:  # noqa: BLE001 — registry is optional for the tree
        return []
    return [s for s in sources.values()
            if s.kind == "folders" and s.hierarchical and s.present_in_tree]


def _derive(path: Path) -> tuple[dict, dict[str, str], dict[str, str]]:
    """sources.yaml -> (tree, category_labels, dt_categories)."""
    data = _read_yaml(path)
    if data is None:
        raise ValueError(
            f"sources.yaml at {path} is present but contains no mappings. "
            "Ensure the file is valid YAML with a top-level mapping."
        )

    ministries, cat_labels, dt_cats = _parse_presentation(data)
    tree: dict[str, dict] = {}
    claimed: dict[str, str] = {}  # org slug -> ministry slug (first claim wins)

    def add_org(ministry_slug: str, slug: str, name: str, categories: list[str]) -> None:
        if slug in claimed:
            return  # org slugs are globally unique; first claim wins
        claimed[slug] = ministry_slug
        tree[ministry_slug]["orgs"].append(
            {"slug": slug, "name": name, "categories": list(categories)}
        )

    # 1) presentation ministries (config order preserved)
    for mid, entry in ministries.items():
        tree[str(mid)] = {"name": entry["name"], "orgs": []}
        for slug, org in (entry.get("orgs") or {}).items():
            if isinstance(org, str):
                add_org(str(mid), str(slug), org, [])
            else:
                add_org(str(mid), str(slug), org["name"], org.get("categories", []))

    # 2) registry-derived: each hierarchical source becomes/merges a ministry
    registry = sorted(_registry_hierarchical_sources(path), key=lambda s: s.name)
    labels = dict(_SHIPPED_CATEGORY_LABELS)
    labels.update(cat_labels)
    for spec in registry:
        mid = str(spec.name)
        if mid not in tree:
            tree[mid] = {"name": slug_label(mid), "orgs": []}
        slugs = {str(v) for v in (spec.org_map or {}).values()}
        if spec.default_org:
            slugs.add(str(spec.default_org))
        for slug in sorted(slugs):
            add_org(mid, slug, slug_label(slug), [])

    dt = dict(_SHIPPED_DT_CATEGORY)
    dt.update(dt_cats)
    return tree, labels, dt


def _apply(path: Path) -> None:
    """Rebuild derived state and swap the exported dicts ATOMICALLY.

    Each assignment is a single reference rebind (atomic under the GIL), so a
    concurrent reader — /api/sources in a worker thread, derive_category in the
    retrieval path — always sees one complete dict (old or new), never the
    transiently-empty window an in-place clear()+update() would open. No
    outside module from-imports these dicts (verified), and _derive() always
    returns freshly-built containers, so rebinding aliases nothing.
    """
    global ORG_TREE, _CATEGORY_LABELS, _DT_CATEGORY
    tree, labels, dt = _derive(path)
    ORG_TREE = tree
    _CATEGORY_LABELS = labels
    _DT_CATEGORY = dt
    _STATE["key"] = config_stamp()


def _ensure_fresh() -> None:
    """Re-derive when the config file changed (path/mtime key). Public entry
    points that read the tree call this; per-record hot paths (derive_*)
    intentionally do not (no stat syscall per record). Structural config
    errors surface HERE as ValueError (clear failure, not silent fallback)."""
    if _STATE["key"] is None or _STATE["key"] != config_stamp():
        _apply(sources_config_path())


def _load_initial() -> None:
    """Populate org-tree from sources.yaml at import time.

    Raises on any configuration error so misconfiguration is visible
    immediately rather than silently serving a stale shipped snapshot.
    """
    _ensure_fresh()


_load_initial()

# Org tokens recognized in subject/source_url text (longest first). Internal
# identity heuristics (WHO produced the doc), not configuration — see D-DTYPE
# separation note in the module docstring.
_ORG_SIGNALS: list[tuple[str, str]] = [
    ("INDIA METEOROLOGICAL", "imd"),
    ("/IMD/", "imd"),
    ("INCOIS", "incois"),
    ("INDIAN NATIONAL CENTRE FOR OCEAN INFORMATION SERVICES", "incois"),
]


def derive_org(meta: dict | None) -> str:
    """Determine which org produced a record, given its metadata dict."""
    if not meta:
        return "moes_hq"
    explicit = meta.get("org")
    if explicit:
        return str(explicit)

    dt = str(meta.get("document_type") or "").lower()
    if dt == "parliamentary_qa":
        return "sansad"

    # Parliamentary records always carry session / question_number / member.
    if (
        meta.get("session") is not None
        or meta.get("question_number") is not None
        or meta.get("member") is not None
    ):
        return "sansad"

    # IDENTITY signals only — org = WHO PRODUCED the doc, not who's mentioned
    # in its content. answer_text is deliberately EXCLUDED: INCOIS annual
    # reports repeatedly mention "India Meteorological Department", which
    # would misclassify them as IMD.
    blob = " ".join(
        str(meta.get(k) or "") for k in (
            "subject", "source_url", "collection", "question_text", "title",
        )
    ).upper()
    for token, slug in _ORG_SIGNALS:
        if token in blob:
            return slug

    return "moes_hq"


def derive_category(meta: dict | None) -> str:
    """Determine doc_category (cadence axis) from metadata (fallback mapping)."""
    if not meta:
        return "misc"
    explicit = meta.get("doc_category")
    if explicit:
        return str(explicit)
    dt = str(meta.get("document_type") or "").lower()
    return _DT_CATEGORY.get(dt, "misc")


def _record_meta_blob(rec) -> dict:
    """Normalize a QARecord or doc_map JSON dict into derivation input."""
    if rec is None:
        return {}
    if isinstance(rec, dict):
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        blob = dict(meta)
        blob["question_text"] = rec.get("question_text") or ""
        blob["answer_text"] = rec.get("answer_text") or ""
        return blob
    meta = getattr(rec, "metadata", None)
    blob = meta.model_dump() if meta is not None and hasattr(meta, "model_dump") else {}
    blob["question_text"] = getattr(rec, "question_text", "") or ""
    blob["answer_text"] = getattr(rec, "answer_text", "") or ""
    return blob


def _category_label(category: str) -> str:
    return _CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def build_sources_catalogue(records) -> dict:
    """Facet tree/types/categories from the *searchable* record set (doc_map).

    Counts ALWAYS come from the records passed in (the indexed corpus);
    configuration contributes identity/labels only and can never override a
    count. Category entries carry the configured display label (additive
    ``label`` key — older frontends ignore it).
    """
    _ensure_fresh()
    from collections import Counter

    types: Counter = Counter()
    categories: Counter = Counter()
    orgs: Counter = Counter()
    for rec in records:
        blob = _record_meta_blob(rec)
        types[blob.get("document_type") or "document"] += 1
        categories[derive_category(blob)] += 1
        orgs[derive_org(blob)] += 1

    tree: dict = {}
    known: set[str] = set()
    for mslug, m in ORG_TREE.items():
        org_list = [
            {
                "slug": o["slug"],
                "name": o["name"],
                "count": orgs.get(o["slug"], 0),
                "categories": o["categories"],
            }
            for o in m["orgs"]
        ]
        known.update(o["slug"] for o in m["orgs"])
        tree[mslug] = {
            "name": m["name"],
            "count": sum(orgs.get(o["slug"], 0) for o in m["orgs"]),
            "orgs": org_list,
        }
    extra = [
        {"slug": s, "name": slug_label(s), "count": c, "categories": []}
        for s, c in orgs.items()
        if s not in known
    ]
    if extra:
        tree["__other__"] = {
            "name": "Other sources",
            "count": sum(e["count"] for e in extra),
            "orgs": extra,
        }
    return {
        "tree": tree,
        "types": [{"type": t, "count": c} for t, c in types.most_common()],
        "categories": [
            {"category": c, "count": n, "label": _category_label(c)}
            for c, n in categories.most_common()
        ],
        "total": sum(orgs.values()),
        "source": "index",
    }


def resolve_orgs(ministry: str | None = None, orgs: list[str] | None = None) -> set[str]:
    """
    Tree-rule expansion: ministry -> all its orgs; explicit orgs are unioned.
    Returns an EMPTY set when no filter applies (means: no org restriction).
    """
    _ensure_fresh()
    selected: set[str] = set()
    if ministry and ministry != "all":
        m = ORG_TREE.get(ministry)
        if m:
            selected.update(o["slug"] for o in m["orgs"])
    if orgs:
        selected.update(orgs)
    return selected
