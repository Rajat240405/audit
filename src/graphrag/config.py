"""
GraphRAG configuration.

Single source of truth for the Neo4j GraphRAG pipeline. Every value can be
overridden via environment variables (``GRAPHRAG_*``) or CLI flags; CLI flags
take precedence.

This module is intentionally dependency-light (pydantic only) so it can be
imported anywhere without pulling in the neo4j driver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _default_ollama_url() -> str:
    """Shared Ollama endpoint resolver (local import keeps this module
    dependency-light; falls back to the upstream default on any error)."""
    try:
        from src.generation.client import ollama_base_url

        return ollama_base_url()
    except Exception:  # noqa: BLE001
        return "http://localhost:11434"


@dataclass
class GraphRAGConfig:
    """Runtime configuration for the GraphRAG pipeline."""

    # ── Neo4j connection ────────────────────────────────────────────────
    neo4j_uri: str = field(
        default_factory=lambda: _env("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: Optional[str] = field(
        default_factory=lambda: os.environ.get("GRAPHRAG_NEO4J_USER", "neo4j")
    )
    neo4j_password: Optional[str] = field(
        default_factory=lambda: os.environ.get("GRAPHRAG_NEO4J_PASSWORD", "neo4j")
    )
    neo4j_database: str = field(
        default_factory=lambda: _env("GRAPHRAG_NEO4J_DATABASE", "neo4j")
    )

    # ── Input / checkpoint paths ────────────────────────────────────────
    enriched_glob: str = field(
        default_factory=lambda: _env("GRAPHRAG_ENRICHED_GLOB", "data/enriched/enriched_*.jsonl")
    )
    checkpoint_path: str = field(
        default_factory=lambda: _env("GRAPHRAG_CHECKPOINT", "storage/graphrag/checkpoint.json")
    )

    # ── Embeddings (reuse the Hybrid RAG model: BAAI/bge-m3) ────────────
    embedding_model: str = field(
        default_factory=lambda: _env("GRAPHRAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_device: Optional[str] = field(
        default_factory=lambda: os.environ.get("GRAPHRAG_EMBEDDING_DEVICE", "cpu")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_EMBEDDING_BATCH", "16"))
    )

    # ── Local LLM (Ollama) entity/relationship extraction ───────────────
    # GRAPHRAG_OLLAMA_URL wins; otherwise follow the shared Ollama endpoint
    # resolver (OLLAMA_BASE_URL / OLLAMA_HOST) so a non-default Ollama port
    # works everywhere without a second knob.
    ollama_base_url: str = field(
        default_factory=lambda: _env("GRAPHRAG_OLLAMA_URL", "") or _default_ollama_url()
    )
    ollama_model: str = field(
        default_factory=lambda: _env("GRAPHRAG_OLLAMA_MODEL", "qwen3:8b")
    )
    ollama_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in _env(
                "GRAPHRAG_OLLAMA_MODELS",
                _env("GRAPHRAG_OLLAMA_MODEL", "qwen3:8b"),
            ).split(",")
            if m.strip()
        ]
    )
    ollama_timeout_seconds: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_OLLAMA_TIMEOUT", "300"))
    )

    # ── LLM provider selection (ollama only — local) ────────────────────
    llm_provider: str = field(
        default_factory=lambda: _env("GRAPHRAG_LLM_PROVIDER", "ollama")
    )
    llm_timeout_seconds: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_LLM_TIMEOUT", "300"))
    )
    # Generic model-list override (e.g. --llm-models "m1,m2,m3"); applies to
    # whichever provider is active. None = use the provider-specific list.
    llm_models: Optional[list[str]] = field(default=None)
    # Exponential-backoff tuning (per provider / per API key)
    llm_backoff_base: float = field(
        default_factory=lambda: float(_env("GRAPHRAG_LLM_BACKOFF_BASE", "1.0"))
    )
    llm_max_attempts_per_key: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_LLM_ATTEMPTS_PER_KEY", "3"))
    )

    extract_max_attempts: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_EXTRACT_ATTEMPTS", "3"))
    )
    # Max characters of the document sent to the LLM per extraction call.
    extract_max_chars: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_EXTRACT_MAX_CHARS", "12000"))
    )
    # Max output tokens for an extraction call (the INITIAL budget — small, so
    # we never reserve huge token counts per request). If the model truncates
    # the JSON (finish_reason=length / truncated failed_generation), the
    # provider escalates the budget automatically up to
    # extract_max_tokens_cap. Default 1536 fits extraction JSON for a single
    # document; it is NOT chain-of-thought headroom.
    extract_max_tokens: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_EXTRACT_MAX_TOKENS", "1536"))
    )
    # Upper bound for automatic budget escalation on truncation. Not reserved
    # per request — only reached when the model actually cuts output off.
    extract_max_tokens_cap: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_EXTRACT_MAX_TOKENS_CAP", "8192"))
    )
    # Chat-completions response format for Groq / OpenAI-compatible providers:
    #   json_object  -> legacy JSON mode  ({ "type": "json_object" })
    #   json_schema  -> Structured Outputs ({ "type": "json_schema", ... });
    #                   server-side schema validation — eliminates
    #                   json_validate_failed on models that support it
    #                   (qwen3.6-27b, gpt-oss-120b; llama-3.3-70b-versatile is
    #                   deprecated). Unsupported models 400 -> failover.
    chat_response_format: str = field(
        default_factory=lambda: _env("GRAPHRAG_CHAT_RESPONSE_FORMAT", "json_object")
    )
    # reasoning_format for chat providers (Groq requires "parsed" or "hidden"
    # when JSON mode is used with reasoning models). None = do not send.
    chat_reasoning_format: Optional[str] = field(
        default_factory=lambda: _env("GRAPHRAG_CHAT_REASONING_FORMAT", None) or None
    )
    # Dump full request payloads + raw response bodies to the log (debug).
    llm_debug: bool = field(
        default_factory=lambda: _env("GRAPHRAG_LLM_DEBUG", "0").lower()
        in ("1", "true", "yes")
    )
    # Enable the full per-request debug printout for EXACTLY ONE document
    # (question_id). When set, the provider prints the exact payload (no API
    # key), estimated input tokens, the raw HTTP response body, and a
    # content/empty-string analysis for that document only. Everything else
    # runs with debugging off. Overrides apply to build/rebuild/verify.
    debug_one: Optional[str] = field(
        default_factory=lambda: _env("GRAPHRAG_DEBUG_ONE", None) or None
    )

    # ── Pipeline behaviour ──────────────────────────────────────────────
    limit: Optional[int] = field(default=None)          # process at most N docs (testing)
    resume: bool = field(default=True)                  # honor the checkpoint file
    retry_failed: bool = field(default=True)            # re-attempt previously failed docs
    max_failures: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_MAX_FAILURES", "50"))
    )  # abort build if failures exceed this
    # Minimum verification grade required before a full build is allowed.
    # One of: Excellent, Good, Needs prompt tuning, Poor.
    verify_min_grade: str = field(
        default_factory=lambda: _env("GRAPHRAG_VERIFY_MIN_GRADE", "Good")
    )

    # ── Neo4j write batching ────────────────────────────────────────────
    write_batch_size: int = field(
        default_factory=lambda: int(_env("GRAPHRAG_WRITE_BATCH", "50"))
    )

    # Derived helpers
    @property
    def checkpoint_file(self) -> Path:
        return Path(self.checkpoint_path)

    def with_overrides(
        self,
        *,
        enriched_glob: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        limit: Optional[int] = None,
        resume: Optional[bool] = None,
        retry_failed: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_models: Optional[str] = None,
        debug_one: Optional[str] = None,
    ) -> "GraphRAGConfig":
        """Return a copy with CLI-level overrides applied (highest precedence)."""
        import dataclasses

        kw: dict = {}
        if enriched_glob is not None:
            kw["enriched_glob"] = enriched_glob
        if checkpoint_path is not None:
            kw["checkpoint_path"] = checkpoint_path
        if embedding_model is not None:
            kw["embedding_model"] = embedding_model
        if ollama_model is not None:
            kw["ollama_model"] = ollama_model
        if limit is not None:
            kw["limit"] = limit
        if resume is not None:
            kw["resume"] = resume
        if retry_failed is not None:
            kw["retry_failed"] = retry_failed
        if llm_provider is not None:
            kw["llm_provider"] = llm_provider
        if llm_models is not None:
            kw["llm_models"] = [m.strip() for m in llm_models.split(",") if m.strip()]
        if debug_one is not None:
            kw["debug_one"] = debug_one
        return dataclasses.replace(self, **kw)
