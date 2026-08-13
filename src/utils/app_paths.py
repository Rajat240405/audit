"""Configurable application filesystem roots (P0.3).

Windows/dev default: project root (same as today). HPC/container: set
APP_DATA_DIR, APP_INDEX_DIR, APP_MODEL_DIR to bind-mount paths.

Does not change storage layout — only where the existing folders live.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repository root (directory that contains ``src/`` and ``config/``)."""
    return Path(__file__).resolve().parents[2]


def _env_path(name: str) -> Path | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def data_dir() -> Path:
    """Corpus, inbox, user-knowledge. Override: ``APP_DATA_DIR``."""
    return _env_path("APP_DATA_DIR") or (project_root() / "data")


def index_dir() -> Path:
    """Hybrid RAG index directory. Override: ``APP_INDEX_DIR``."""
    return _env_path("APP_INDEX_DIR") or (project_root() / "storage" / "hybrid_rag")


def storage_dir() -> Path:
    """Parent of hybrid_rag (and sibling graphrag)."""
    return index_dir().parent


def model_dir() -> Path:
    """Local model weights (bge-m3, reranker, …). Override: ``APP_MODEL_DIR``."""
    return _env_path("APP_MODEL_DIR") or (project_root() / "models")


def corpus_path() -> Path:
    return data_dir() / "corpus_reports.jsonl"


def inbox_dir() -> Path:
    return data_dir() / "inbox"


def user_knowledge_dir() -> Path:
    return data_dir() / "user-knowledge"


def prompt_debug_path() -> Path:
    """Optional generation prompt dump. Always under APP_DATA_DIR (never CWD)."""
    return data_dir() / "generation_prompt_debug.txt"


def graph_dir() -> Path:
    """Existing GraphRAG checkpoint location (path only; implementation unchanged)."""
    return storage_dir() / "graphrag"


def config_path(*parts: str) -> Path:
    return project_root().joinpath("config", *parts)


def ensure_data_dirs() -> None:
    inbox_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    user_knowledge_dir().mkdir(parents=True, exist_ok=True)
