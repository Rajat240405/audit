"""
Centralized Project Scope Configuration & Filtering (Phase 12+)

Single source of truth for MoES AI Assistant scope.
Used by ingestion, Hybrid RAG, and GraphRAG.
"""

from __future__ import annotations

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


def filter_records_by_ministry(records, ministry_filter: Optional[str]):
    """Apply ministry filter to a list of QARecord objects (shared helper)."""
    if not ministry_filter:
        return records

    return [
        r for r in records
        if r.metadata.ministry and ministry_filter.lower() in r.metadata.ministry.lower()
    ]