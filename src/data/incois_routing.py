"""Which INCOIS folders use DOTS-OCR-only extraction — path classification.

One rule, defined once, consumed by the extractor router
(``convert_sirs_knowledge.convert_pdf_file``) and the doc-type/peek logic
(``ingest_folder.convert_one_detected``):

    Everything under ``data/incois_reports/<folder>/`` uses DOTS OCR only,
    EXCEPT the crawler-managed official report folders, which keep their
    existing extraction behaviour.

The default is DOTS-only on purpose. A deny-list means a newly added manual
folder — ``internal-docs``, ``contracts``, ``circulars``, anything — inherits
DOTS-only automatically, with no new Python condition and no config edit. The
config file exists to *opt folders out*, not in.

The folder list lives in ``config/extraction_routing.yaml`` so operations can
adjust it without a code change. If that file is missing or unreadable the
built-in defaults below apply unchanged, so extraction never silently flips
to the wrong engine because of a config problem.

Scope note: this module classifies PATHS only. It does not decide whether a
folder is ingested (``config/sources.yaml``) nor how deeply a folder is walked
(the engine lists one level; see ``ingest_folder``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── built-in defaults (used when the config file is absent/unreadable) ──────

#: Parent directory, relative to the data root, whose children are classified.
INCOIS_ROOT = "incois_reports"

#: The four crawler-managed official report folders. These keep the existing
#: extraction path. Mirrors the labels in crawl_incois_reports.SECTIONS for
#: the non-budget sections; asserted equal by tests.
DEFAULT_LEGACY_EXTRACTION_FOLDERS: tuple[str, ...] = (
    "AnnualReports",
    "Others",
    "ResearchPublications",
    "TechnicalReports",
)

#: Per-folder document_type overrides; everything else uses normal detection.
DEFAULT_DOCUMENT_TYPES: dict[str, str] = {"budget": "tender"}

_CONFIG_RELPATH = "config/extraction_routing.yaml"

_cache: dict | None = None


def _config_path() -> Path:
    override = (os.environ.get("INCOIS_ROUTING_CONFIG") or "").strip()
    if override:
        return Path(override).expanduser()
    from src.utils.app_paths import project_root

    return project_root() / _CONFIG_RELPATH


def _defaults() -> dict:
    return {
        "root": INCOIS_ROOT,
        "legacy_extraction_folders": tuple(DEFAULT_LEGACY_EXTRACTION_FOLDERS),
        "document_types": dict(DEFAULT_DOCUMENT_TYPES),
    }


def load_routing() -> dict:
    """Load and cache the routing config, falling back to built-in defaults.

    A malformed or partial file degrades per-key: any key the file does not
    supply keeps its default. A hard read/parse failure keeps all defaults.
    """
    global _cache
    if _cache is not None:
        return _cache

    cfg = _defaults()
    path = _config_path()
    try:
        if path.is_file():
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            block = (raw.get("incois") or {}) if isinstance(raw, dict) else {}
            if isinstance(block, dict):
                root = block.get("root")
                if isinstance(root, str) and root.strip():
                    cfg["root"] = root.strip()
                legacy = block.get("legacy_extraction_folders")
                if isinstance(legacy, list):
                    cfg["legacy_extraction_folders"] = tuple(
                        str(x).strip() for x in legacy if str(x).strip()
                    )
                dtypes = block.get("document_types")
                if isinstance(dtypes, dict):
                    cfg["document_types"] = {
                        str(k).strip(): str(v).strip()
                        for k, v in dtypes.items()
                        if str(k).strip() and str(v).strip()
                    }
    except Exception:  # noqa: BLE001 — config problems must not change routing
        cfg = _defaults()

    _cache = cfg
    return cfg


def reset_cache() -> None:
    """Drop the cached config (tests, and after an operator edit)."""
    global _cache
    _cache = None


# ── classification ──────────────────────────────────────────────────────────


def incois_root() -> str:
    return str(load_routing()["root"])


def legacy_extraction_folders() -> tuple[str, ...]:
    return tuple(load_routing()["legacy_extraction_folders"])


def incois_folder_of(path: str | Path) -> str | None:
    """Return the folder name directly under the INCOIS root, else None.

    ``data/incois_reports/internal-docs/a.pdf`` -> ``"internal-docs"``
    ``data/incois_reports/a.pdf``               -> None  (no subfolder)
    ``data/annual_reports/a.pdf``               -> None  (different tree)

    Matched on consecutive path parts so it is independent of the data root
    (``APP_DATA_DIR``), absolute vs relative paths, and OS separators.
    """
    root = incois_root()
    root_parts = tuple(p for p in Path(root).parts if p not in ("", os.sep))
    parts = Path(path).parts
    n = len(root_parts)
    if n == 0:
        return None
    for i in range(len(parts) - n):
        if parts[i:i + n] == root_parts:
            nxt = parts[i + n]
            # The last component is the file itself, not a folder.
            if i + n == len(parts) - 1:
                return None
            return nxt
    return None


def uses_dots_only(path: str | Path) -> bool:
    """True when *path* must be extracted with DOTS OCR and nothing else.

    DOTS-only is the DEFAULT for any folder under the INCOIS root; the four
    official crawler folders are the explicit exceptions. Anything outside the
    INCOIS root is unaffected and returns False.
    """
    folder = incois_folder_of(path)
    if folder is None:
        return False
    legacy = {f.casefold() for f in legacy_extraction_folders()}
    return folder.casefold() not in legacy


def document_type_for(path: str | Path) -> str | None:
    """Configured document_type override for *path*, or None for normal detection."""
    folder = incois_folder_of(path)
    if folder is None:
        return None
    dtypes = load_routing()["document_types"]
    if folder in dtypes:
        return dtypes[folder]
    fold = folder.casefold()
    for k, v in dtypes.items():
        if k.casefold() == fold:
            return v
    return None
