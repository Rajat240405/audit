"""Lok Sabha crawler config loading + fail-closed validation.

The frozen ``src/scraping/config.py`` is Rajya-Sabha-specific
(``SUPPORTED_HOUSES = ("rajya-sabha",)`` — not extended by design); Lok Sabha
gets its own loader here, mirroring the pattern MoES established
(``src/scraping/moes/config.py``).

Default file: ``config/crawlers/lok_sabha_qa.yaml``. Output root defaults to
``app_paths.data_dir() / "parliamentary-qa" / "lok-sabha"`` so the crawler
works on dev machines, HPC (APP_DATA_DIR) and containers unchanged.

Ministry scope model (validated live 2026-08-25):

- ``api_ministry_code`` — the sansad.in api_ls listing filter. One code may
  span ministry renames across eras (23 = EARTH SCIENCES today AND its
  pre-2006 name OCEAN DEVELOPMENT), so ``row_labels`` routes each returned
  row to its configured ministry. Row labels must be disjoint across
  ministries (fail-closed).
- ``elibrary_labels`` — exact ``f.ministry`` facet labels for the DSpace
  discover search of the legacy era.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.utils import app_paths

DEFAULT_CONFIG = app_paths.config_path("crawlers", "lok_sabha_qa.yaml")

HOUSE = "lok-sabha"
ERA_API_LS = "api_ls"
ERA_DSPACE = "dspace"

#: api_ls pageSize is honored up to 500 (validated); DSpace caps size at 100.
API_MAX_PAGE_SIZE = 500
DSPACE_MAX_PAGE_SIZE = 100


class LsConfigError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise LsConfigError(f"crawler config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise LsConfigError(f"crawler config is not a mapping: {cfg_path}")
    cfg["_path"] = str(cfg_path)
    validate_config(cfg)
    return cfg


def _norm_label(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().upper()


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("house") != HOUSE:
        raise LsConfigError(
            f"house must be {HOUSE!r} for this loader (got {cfg.get('house')!r})"
        )

    ministries = cfg.get("ministries")
    if not isinstance(ministries, list) or not ministries:
        raise LsConfigError("config must define a non-empty 'ministries' list")
    seen_slugs: set[str] = set()
    seen_labels: dict[str, str] = {}
    for m in ministries:
        if not isinstance(m, dict):
            raise LsConfigError(f"ministry entry must be a mapping: {m!r}")
        for field in ("slug", "label", "api_ministry_code",
                      "row_labels", "elibrary_labels"):
            if field not in m:
                raise LsConfigError(f"ministry entry missing {field!r}: {m!r}")
        if m["slug"] in seen_slugs:
            raise LsConfigError(f"duplicate ministry slug: {m['slug']!r}")
        seen_slugs.add(m["slug"])
        try:
            int(m["api_ministry_code"])
        except (TypeError, ValueError):
            raise LsConfigError(
                f"api_ministry_code must be an int for ministry {m['slug']!r}"
            ) from None
        for field in ("row_labels", "elibrary_labels"):
            labels = m[field]
            if not isinstance(labels, list) or not [x for x in labels if str(x).strip()]:
                raise LsConfigError(
                    f"{field} must be a non-empty label list for ministry {m['slug']!r}"
                )
        # row labels route api_ls rows to ministries — must be disjoint
        for label in m["row_labels"]:
            key = _norm_label(str(label))
            other = seen_labels.get(key)
            if other is not None and other != m["slug"]:
                raise LsConfigError(
                    f"row_label {label!r} claimed by both {other!r} and "
                    f"{m['slug']!r} (api_ls ministry codes span renames; "
                    "row-label routing must stay unambiguous)"
                )
            seen_labels[key] = m["slug"]

    eras = cfg.get("eras") or {}
    boundary = eras.get("api_ls_min_loksabha", 16)
    try:
        boundary = int(boundary)
    except (TypeError, ValueError):
        raise LsConfigError("eras.api_ls_min_loksabha must be an int") from None
    if not 2 <= boundary <= 25:
        raise LsConfigError(
            "eras.api_ls_min_loksabha must be within 2..25 "
            "(lok >= boundary → api_ls; lok < boundary → DSpace)"
        )

    api = cfg.get("api_ls") or {}
    for field in ("base_url", "referer"):
        if not str(api.get(field, "")).strip():
            raise LsConfigError(f"api_ls.{field} is required (live-validated contracts)")
    page_size = int(api.get("page_size", API_MAX_PAGE_SIZE))
    if not 1 <= page_size <= API_MAX_PAGE_SIZE:
        raise LsConfigError(f"api_ls.page_size must be within 1..{API_MAX_PAGE_SIZE}")

    elib = cfg.get("elibrary") or {}
    if not str(elib.get("base_url", "")).strip():
        raise LsConfigError("elibrary.base_url is required")
    esize = int(elib.get("search_page_size", DSPACE_MAX_PAGE_SIZE))
    if not 1 <= esize <= DSPACE_MAX_PAGE_SIZE:
        raise LsConfigError(
            f"elibrary.search_page_size must be within 1..{DSPACE_MAX_PAGE_SIZE} "
            "(server caps page size at 100 — validated)"
        )

    for field in ("loksabhas", "sessions"):
        window = cfg.get(field) or {}
        if not isinstance(window, dict):
            raise LsConfigError(f"{field} must be a mapping (min/max/exclude)")
        exclude = window.get("exclude") or []
        if not isinstance(exclude, list) or not all(isinstance(x, int) for x in exclude):
            raise LsConfigError(f"{field}.exclude must be a list of ints")

    http = cfg.get("http") or {}
    for key in ("timeout_seconds", "request_delay_seconds", "retry_backoff_seconds"):
        if key in http and float(http[key]) < 0:
            raise LsConfigError(f"http.{key} must be >= 0")
    if not str(http.get("user_agent", "")).strip():
        raise LsConfigError(
            "http.user_agent is required: sansad.in api_ls answers 403 to "
            "non-browser clients (live-validated); configure a browser UA"
        )


def resolve_ministries(cfg: dict[str, Any], slugs: list[str] | None = None) -> list[dict[str, Any]]:
    """Configured ministries, optionally filtered to the given slugs (order kept)."""
    ministries = list(cfg["ministries"])
    if not slugs:
        return ministries
    wanted = set(slugs)
    unknown = wanted - {m["slug"] for m in ministries}
    if unknown:
        raise LsConfigError(f"unknown ministry slug(s): {sorted(unknown)}")
    return [m for m in ministries if m["slug"] in wanted]


def route_ministry(
    ministries: list[dict[str, Any]], row_label: str | None
) -> dict[str, Any] | None:
    """The configured ministry whose ``row_labels`` claim this api_ls row label."""
    key = _norm_label(row_label or "")
    if not key:
        return None
    for m in ministries:
        if key in {_norm_label(str(x)) for x in m["row_labels"]}:
            return m
    return None


def output_root(cfg: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    source_root = cfg.get("source_root")
    if source_root:
        return Path(source_root)
    return app_paths.data_dir() / "parliamentary-qa" / HOUSE


def era_for(cfg: dict[str, Any], loksabha: int) -> str:
    boundary = int((cfg.get("eras") or {}).get("api_ls_min_loksabha", 16))
    return ERA_API_LS if int(loksabha) >= boundary else ERA_DSPACE


def api_endpoint(cfg: dict[str, Any]) -> tuple[str, int, str]:
    api = cfg.get("api_ls") or {}
    return (
        str(api.get("base_url", "")).rstrip("/"),
        int(api.get("page_size", API_MAX_PAGE_SIZE)),
        str(api.get("referer", "")),
    )


def elib_endpoint(cfg: dict[str, Any]) -> tuple[str, int]:
    elib = cfg.get("elibrary") or {}
    return (
        str(elib.get("base_url", "")).rstrip("/"),
        int(elib.get("search_page_size", DSPACE_MAX_PAGE_SIZE)),
    )


def window(cfg: dict[str, Any], field: str) -> tuple[int | None, int | None, list[int]]:
    w = cfg.get(field) or {}
    return w.get("min"), w.get("max"), list(w.get("exclude") or [])


def http_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    http = cfg.get("http") or {}
    return {
        "timeout": float(http.get("timeout_seconds", 60)),
        "delay": float(http.get("request_delay_seconds", 1.0)),
        "retries": int(http.get("retries", 2)),
        "backoff": float(http.get("retry_backoff_seconds", 1.0)),
    }


def http_headers(cfg: dict[str, Any]) -> dict[str, str]:
    """Browser-identity headers the live contracts demand (validated):

    - a browser ``User-Agent`` (api_ls 403s non-browser clients),
    - ``Referer`` pointing at the LS Q&A page (api_ls 403s without it).
    """
    http = cfg.get("http") or {}
    headers = {"User-Agent": str(http.get("user_agent", "")).strip()}
    referer = str((cfg.get("api_ls") or {}).get("referer", "")).strip()
    if referer:
        headers["Referer"] = referer
    for k, v in (http.get("extra_headers") or {}).items():
        headers[str(k)] = str(v)
    return headers
