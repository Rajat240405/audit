"""Crawler config loading (design §8: config-driven, never CWD-relative).

Default file: ``config/crawlers/parliamentary_qa.yaml``. Output root
defaults to ``app_paths.data_dir() / "parliamentary-qa" / <house>`` so the
crawler works on dev machines, HPC (APP_DATA_DIR) and containers unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils import app_paths

DEFAULT_CONFIG = app_paths.config_path("crawlers", "parliamentary_qa.yaml")

SUPPORTED_HOUSES = ("rajya-sabha",)


class CrawlerConfigError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise CrawlerConfigError(f"crawler config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise CrawlerConfigError(f"crawler config is not a mapping: {cfg_path}")
    cfg["_path"] = str(cfg_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    house = cfg.get("house")
    if house not in SUPPORTED_HOUSES:
        raise CrawlerConfigError(
            f"unsupported house {house!r}; this build implements only {SUPPORTED_HOUSES}"
        )
    ministries = cfg.get("ministries")
    if not isinstance(ministries, list) or not ministries:
        raise CrawlerConfigError("config must define a non-empty 'ministries' list")
    for m in ministries:
        for field in ("code", "slug", "label"):
            if field not in m:
                raise CrawlerConfigError(f"ministry entry missing {field!r}: {m!r}")


def resolve_ministries(cfg: dict[str, Any], slugs: list[str] | None = None) -> list[dict[str, Any]]:
    """Configured ministries, optionally filtered to the given slugs (order kept)."""
    ministries = list(cfg["ministries"])
    if not slugs:
        return ministries
    wanted = set(slugs)
    unknown = wanted - {m["slug"] for m in ministries}
    if unknown:
        raise CrawlerConfigError(f"unknown ministry slug(s): {sorted(unknown)}")
    return [m for m in ministries if m["slug"] in wanted]


def output_root(cfg: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    source_root = cfg.get("source_root")
    if source_root:
        return Path(source_root)
    return app_paths.data_dir() / "parliamentary-qa" / cfg["house"]


def http_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    http = cfg.get("http") or {}
    return {
        "timeout": float(http.get("timeout_seconds", 60)),
        "delay": float(http.get("request_delay_seconds", 1.0)),
        "retries": int(http.get("retries", 2)),
        "backoff": float(http.get("retry_backoff_seconds", 1.0)),
    }
