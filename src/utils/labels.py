"""Deterministic slug → display-label helper.

Single home for the prettifier previously duplicated between
``src/scripts/ingest_service.py`` (upload targets) and the Phase-5
config-driven source tree (``src/retrieval/frontend/org_tree.py``).

Rule (stable, config-free): short slugs read as acronyms
(``incois`` → ``INCOIS``, ``moes_hq`` → ``MOES HQ``); longer slugs are
title-cased (``rajat_reports`` → ``Rajat Reports``). Config ``presentation:``
labels always win over this fallback.
"""

from __future__ import annotations

_ACRONYM_MAX = 7


def slug_label(slug: str) -> str:
    """Deterministic display label for a slug (no config lookup)."""
    words = str(slug or "").replace("_", " ").replace("-", " ").strip()
    if not words:
        return str(slug or "")
    return words.upper() if len(slug) <= _ACRONYM_MAX else words.title()
