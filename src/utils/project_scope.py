"""
Centralized Project Scope Configuration & Filtering.

Single source of truth for MoES AI Assistant scope.
Used by ingestion, Hybrid RAG, and GraphRAG.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml


def load_project_scope(config_path: str = "config/ingestion.yaml") -> dict:
    """Load project_scope section from ingestion config."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        from src.utils.app_paths import project_root

        cfg_path = project_root() / config_path
    if not cfg_path.exists():
        return {}

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("project_scope", {})
    except Exception:
        return {}


def resolve_effective_ministry_filter(
    explicit_filter: Optional[str] = None,
    all_ministries: bool = False,
    config_path: str = "config/ingestion.yaml",
) -> Optional[str]:
    """
    Resolve the effective ministry filter using centralized project_scope config.

    Priority:
    1. --all-ministries flag → return None (index everything)
    2. explicit --ministry-filter → use it
    3. project_scope.filter_enabled → use default_ministry
    4. Otherwise → None (no filter)

    Returns the ministry string to filter on, or None.
    """
    if all_ministries:
        return None

    if explicit_filter:
        return explicit_filter

    scope = load_project_scope(config_path)
    if scope.get("filter_enabled", True):
        return scope.get("default_ministry", "Ministry of Earth Sciences")

    return None


def normalize_ministry(value: str | None) -> str:
    """Canonical ministry key for comparison (FIX B — verified finding #1).

    The corpus carries two vocabularies for the same ministry (parliament
    rows use the ``earth-sciences`` slug; converted rows use the
    ``EARTH SCIENCES`` label). This collapses both — plus the natural
    ``Ministry of X`` phrasing — to one deterministic key: lowercase, a
    leading "ministry of" stripped, all spacing/separator characters removed.
    Distinct ministries (e.g. ``ocean-development``) keep distinct keys.
    """
    if not value:
        return ""
    t = value.strip().lower()
    t = re.sub(r"^ministry\s+of\s+", "", t)
    return re.sub(r"[\s\-_]+", "", t)


def filter_records_by_ministry(records, ministry_filter: Optional[str]):
    """Apply ministry filter to a list of QARecord objects (shared helper).

    Comparison is on normalized keys (FIX B): "Ministry of Earth Sciences",
    "EARTH SCIENCES" and "earth-sciences" all match both stored vocabularies.
    """
    if not ministry_filter:
        return records

    needle = normalize_ministry(ministry_filter)
    if not needle:
        return records

    return [
        r for r in records
        if r.metadata.ministry and needle in normalize_ministry(r.metadata.ministry)
    ]


# ── English-only ingestion policy (Step 6) ───────────────────────────────────
# Document filenames that carry an explicit language suffix identifying them as
# non-English are excluded from both ingest-time corpus creation and rebuild-time
# embedding/indexing.  The naming convention is shared by the MoES website
# crawler (src/scraping/moes/documents.py) and the RS/LS document downloaders:
#   <row>-<wp_id>-eng.pdf   — English   → ingest normally
#   <row>-<wp_id>-hin.pdf   — Hindi     → exclude silently
#   <row>-<wp_id>-both.pdf  — bilingual → exclude conservatively (TODO: detect
#                                          English-only bilingual docs in future)
# Unsuffixed filenames (no trailing -eng/-hin/-both) are NEVER excluded so
# that legacy flat-source folders (inbox, annual_reports, incois_reports, …)
# continue to behave exactly as before.
_NON_ENGLISH_STEM_SUFFIXES: tuple[str, ...] = ("-hin", "-both")


def is_non_english_filename(name: str) -> bool:
    """Return True when *name* carries an explicit non-English language suffix.

    Only filenames whose *stem* (name without extension) ends with one of the
    known non-English suffixes are excluded.  Unsuffixed names are NOT touched.

    >>> is_non_english_filename("01-21697-hin.pdf")
    True
    >>> is_non_english_filename("01-21697-both.pdf")
    True
    >>> is_non_english_filename("01-21697-eng.pdf")
    False
    >>> is_non_english_filename("annual_report_2024.pdf")
    False
    """
    stem = Path(name).stem.lower()
    return any(stem.endswith(s) for s in _NON_ENGLISH_STEM_SUFFIXES)


def filter_non_english_docs(records) -> list:
    """Remove records whose source document is identified as non-English.

    Operates on the ``metadata.source_url`` field (set to ``str(path)`` by
    ``convert_pdf_file`` for all folder-ingested PDFs).  Records with no
    source_url, or whose filename carries no explicit language suffix, pass
    through unchanged — this preserves all legacy flat-source behaviour.

    Called at rebuild time (src/retrieval/cli.py) so that previously-ingested
    Hindi/bilingual records are purged from the vector and BM25 indices on the
    next full rebuild, even if they are still present in corpus_reports.jsonl.
    """
    kept = []
    removed = 0
    for r in records:
        url = (r.metadata.source_url or "") if r.metadata else ""
        filename = Path(url).name if url else ""
        if filename and is_non_english_filename(filename):
            removed += 1
        else:
            kept.append(r)
    if removed:
        import warnings
        warnings.warn(
            f"[lang-filter] excluded {removed} non-English document record(s) "
            "from index (source filenames end in -hin or -both). "
            "Run `ingest` to prevent them re-entering the corpus on the next rebuild.",
            stacklevel=2,
        )
    return kept