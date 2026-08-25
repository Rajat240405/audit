"""MoES website crawler config loading + fail-closed scope guards.

The frozen ``src/scraping/config.py`` is Rajya-Sabha-specific (HOUSE/ministry
shape); MoES gets its own loader here. Guards enshrined from the approved
boundary review:

- ONLY the v1 categories may appear under ``categories:`` — any other key
  fails closed at load time (exit 2 path).
- ``central-documents`` walks ONLY through ``central_documents.scopes``
  (currently the single approved ``annual-reports`` family). The legacy
  ``central_documents.enabled`` switch is gone: any ``enabled: true`` STILL
  fails closed (a stale v1 config can never silently widen scope).
- Reports families must be a non-empty mapping of family → match patterns;
  a family resolving to zero live taxonomy terms fails closed at run time
  (see ``normalize.resolve_family_terms``). Central scopes resolve against
  the live tree too (``normalize.resolve_central_family_terms``) but
  TOLERATE zero/absent child terms: live probing (2026-08-25) found the
  upstream taxonomy tree has NO central-documents node at all (the listing
  category exists without one — Annual Reports surface only as
  central_documents attachments), so per-record content evidence is the
  arbiter and stays fail-closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils import app_paths

DEFAULT_CONFIG = app_paths.config_path("crawlers", "moes_website.yaml")

#: the only categories implemented in v1 (CLI choices + config allowlist);
#: central-documents is scoped EXCLUSIVELY through central_documents.scopes
V1_CATEGORIES = ("reports", "press-release", "central-documents")


class MoesConfigError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise MoesConfigError(f"crawler config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise MoesConfigError(f"crawler config is not a mapping: {cfg_path}")
    cfg["_path"] = str(cfg_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    central = cfg.get("central_documents") or {}
    if central.get("enabled"):
        raise MoesConfigError(
            "central_documents.enabled is NOT a valid switch for the "
            "fail-closed central-documents scope: scope is declared via "
            "central_documents.scopes only — the legacy v1 blanket switch "
            "stays invalid so a stale config can never silently widen scope"
        )
    scopes = central.get("scopes") or {}
    if "central-documents" in (cfg.get("categories") or {}) and (
            not isinstance(scopes, dict) or not scopes):
        raise MoesConfigError(
            "central-documents requires central_documents.scopes — a "
            "non-empty mapping of family → {covers: [name patterns]} "
            "(the category alone never widens scope)"
        )
    for fam, fcfg in scopes.items():
        covers = (fcfg or {}).get("covers")
        if not isinstance(covers, list) or not [c for c in covers if str(c).strip()]:
            raise MoesConfigError(
                f"central_documents.scopes.{fam}.covers must be a non-empty list"
            )

    categories = cfg.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise MoesConfigError("config must define a non-empty 'categories' mapping")
    unknown = sorted(set(categories) - set(V1_CATEGORIES))
    if unknown:
        raise MoesConfigError(
            f"category(ies) outside the approved v1 scope {V1_CATEGORIES}: {unknown} "
            f"(fail-closed — guidelines/orders-and-notices/publications/acts-and-policy/"
            f"gazette-notifications/central-documents are all out of scope)"
        )
    reports = categories.get("reports")
    if reports is not None:
        families = (reports or {}).get("families")
        if not isinstance(families, dict) or not families:
            raise MoesConfigError("categories.reports must define a non-empty 'families' map")
        for fam, fcfg in families.items():
            match = (fcfg or {}).get("match")
            if not isinstance(match, list) or not [m for m in match if str(m).strip()]:
                raise MoesConfigError(f"reports family {fam!r} needs a non-empty 'match' list")

    http = cfg.get("http") or {}
    for key in ("timeout_seconds", "request_delay_seconds", "retry_backoff_seconds"):
        if key in http and float(http[key]) < 0:
            raise MoesConfigError(f"http.{key} must be >= 0")
    if not str(http.get("user_agent", "")).strip():
        raise MoesConfigError("http.user_agent is required (Akamai blocks default clients)")
    extra = http.get("extra_headers") or {}
    if not isinstance(extra, dict):
        raise MoesConfigError("http.extra_headers must be a mapping")

    langs = cfg.get("download_languages", ["english", "hindi"])
    if not isinstance(langs, list) or not langs:
        raise MoesConfigError("download_languages must be a non-empty list")
    bad_langs = sorted(set(langs) - {"english", "hindi"})
    if bad_langs:
        raise MoesConfigError(
            f"unknown download_languages {bad_langs}; allowed: english, hindi")


def resolve_categories(cfg: dict[str, Any], requested: list[str] | None) -> list[str]:
    """Requested categories (or all configured), in config order. Fail closed."""
    configured = list(cfg["categories"].keys())
    if not requested:
        return [c for c in configured if c in V1_CATEGORIES]
    wanted = list(dict.fromkeys(requested))  # dedupe, keep order
    unknown = [c for c in wanted if c not in configured or c not in V1_CATEGORIES]
    if unknown:
        raise MoesConfigError(
            f"unknown or out-of-scope categor(ies) {unknown}; v1 implements only "
            f"{sorted(set(configured) & set(V1_CATEGORIES))}"
        )
    return [c for c in configured if c in wanted]


def resolve_report_families(cfg: dict[str, Any], requested: list[str] | None) -> list[str]:
    """Requested report families (or all configured), in config order. Fail closed."""
    families = list((cfg["categories"].get("reports") or {}).get("families") or {})
    if not requested:
        return list(families)
    wanted = list(dict.fromkeys(requested))
    unknown = [f for f in wanted if f not in families]
    if unknown:
        raise MoesConfigError(
            f"unknown report famil(ies) {unknown}; configured families: {families}"
        )
    return [f for f in families if f in wanted]


def output_root(cfg: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    if cfg.get("source_root"):
        return Path(cfg["source_root"])
    return app_paths.data_dir() / str(cfg.get("output_subdir") or ".moes-website")


def http_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    http = cfg.get("http") or {}
    return {
        "timeout": float(http.get("timeout_seconds", 60)),
        "delay": float(http.get("request_delay_seconds", 1.0)),
        "retries": int(http.get("retries", 2)),
        "backoff": float(http.get("retry_backoff_seconds", 1.0)),
        "too_many_requests_backoff": float(http.get("too_many_requests_backoff_seconds", 30)),
    }


def http_headers(cfg: dict[str, Any]) -> dict[str, str]:
    http = cfg.get("http") or {}
    headers = {"User-Agent": str(http.get("user_agent", "")).strip()}
    for k, v in (http.get("extra_headers") or {}).items():
        headers[str(k)] = str(v)
    return headers


def download_languages(cfg: dict[str, Any]) -> tuple[str, ...]:
    langs = cfg.get("download_languages") or ["english", "hindi"]
    return tuple(str(x) for x in langs)


def endpoints(cfg: dict[str, Any]) -> dict[str, str]:
    ep = cfg.get("endpoints") or {}
    return {
        "counts": str(ep.get("counts", "/cms/wp-json/count-posts/all")),
        "taxonomy": str(ep.get("taxonomy", "/cms/wp-json/taxonomy/documents_category")),
        "listing": str(ep.get("listing", "/cms/wp-json/document/documents")),
        "attachment_post": str(ep.get("attachment_post", "/cms/wp-json/post-page/post?id=")),
        "robots": str(ep.get("robots", "/robots.txt")),
    }


def listing_page_size(cfg: dict[str, Any]) -> int:
    return max(1, int((cfg.get("listing") or {}).get("page_size", 100)))
