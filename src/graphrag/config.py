"""GraphRAG configuration (Phase 2 / WS2).

Environment-driven (spec §26). NO hardcoded production assumptions:
  * the Neo4j URI is REQUIRED for the neo4j backend (no localhost default);
  * secrets (password) come only from the environment — never committed;
  * backend selection: "neo4j" (production) or "inmemory" (local validation
    / tests only — documented, not a deployment mode).

Environment variables:
  GRAPHRAG_BACKEND         neo4j (default) | inmemory
  GRAPHRAG_NEO4J_URI       bolt://host:port  — REQUIRED for the neo4j backend
  GRAPHRAG_NEO4J_USER      default "neo4j"
  GRAPHRAG_NEO4J_PASSWORD  secret; env only
  GRAPHRAG_NEO4J_DATABASE  default "neo4j"
  GRAPHRAG_CHECKPOINT      default <storage>/graphrag/checkpoint.json
  GRAPHRAG_CORPUS          default <data>/corpus_reports.jsonl (canonical)
  GRAPHRAG_EXTRACT_MAX_CHARS   default 12000
  GRAPHRAG_EXTRACT_MAX_TOKENS  default 18000 — OUTPUT token budget for semantic
                             extraction only. The shared policy's fast-mode
                             4096 truncates fact-dense documents mid-JSON
                             (finish_reason="length"), which surfaces as
                             "LLM returned non-JSON payload". Scoped to the
                             GraphRAG extraction client; the serving path and
                             Hybrid RAG keep the shared policy value.
  GRAPHRAG_EXTRACT_ATTEMPTS    default 3
  GRAPHRAG_MAX_FAILURES        default 50 (abort build when failures exceed)
  GRAPHRAG_WRITE_BATCH       default 50
  GRAPHRAG_LLM_CONCURRENCY   default 2 — in-flight semantic extraction
                             requests. Must not exceed the vLLM server's
                             --max-num-seqs (currently 2); higher values
                             only queue on the server. Only the LLM call is
                             parallel: merging, Neo4j writes and checkpoint
                             updates stay serialized.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.app_paths import corpus_path, storage_dir

__all__ = ["GraphConfig", "GraphConfigError", "load_graph_config"]


class GraphConfigError(Exception):
    """Configuration fault — fail loudly (e.g. neo4j backend without URI)."""


@dataclass(frozen=True)
class GraphConfig:
    backend: str                      # "neo4j" | "inmemory"
    neo4j_uri: Optional[str]          # required when backend == "neo4j"
    neo4j_user: str
    neo4j_password: Optional[str]
    neo4j_database: str
    checkpoint_path: Path
    corpus_path: Path
    extract_max_chars: int
    extract_attempts: int
    max_failures: int
    write_batch_size: int
    # defaults keep every existing GraphConfig(...) call site valid
    llm_concurrency: int = 2
    # OUTPUT token budget for semantic extraction (GraphRAG-scoped)
    extract_max_tokens: int = 18000

    def validate(self) -> "GraphConfig":
        if self.backend not in ("neo4j", "inmemory"):
            raise GraphConfigError(f"unknown GRAPHRAG_BACKEND: {self.backend!r}")
        if self.backend == "neo4j" and not (self.neo4j_uri or "").strip():
            raise GraphConfigError(
                "GRAPHRAG_NEO4J_URI is not set — the neo4j backend requires an "
                "explicit bolt URI (no localhost default is assumed). Set "
                "GRAPHRAG_NEO4J_URI or use GRAPHRAG_BACKEND=inmemory for "
                "local validation."
            )
        return self

    @property
    def auth(self) -> Optional[tuple[str, str]]:
        if self.neo4j_user:
            return (self.neo4j_user, self.neo4j_password or "")
        return None


def load_graph_config(env: Optional[dict] = None) -> GraphConfig:
    """Build the config from the environment (``env`` override for tests)."""
    e = os.environ if env is None else env
    uri = (e.get("GRAPHRAG_NEO4J_URI") or "").strip() or None
    checkpoint = e.get("GRAPHRAG_CHECKPOINT") or str(
        storage_dir() / "graphrag" / "checkpoint.json")
    corpus = e.get("GRAPHRAG_CORPUS") or str(corpus_path())
    return GraphConfig(
        backend=(e.get("GRAPHRAG_BACKEND") or "neo4j").strip().lower(),
        neo4j_uri=uri,
        neo4j_user=(e.get("GRAPHRAG_NEO4J_USER") or "neo4j").strip(),
        neo4j_password=e.get("GRAPHRAG_NEO4J_PASSWORD"),
        neo4j_database=(e.get("GRAPHRAG_NEO4J_DATABASE") or "neo4j").strip(),
        checkpoint_path=Path(checkpoint),
        corpus_path=Path(corpus),
        extract_max_chars=int(e.get("GRAPHRAG_EXTRACT_MAX_CHARS") or 12000),
        extract_max_tokens=max(
            1, int(e.get("GRAPHRAG_EXTRACT_MAX_TOKENS") or 18000)),
        extract_attempts=int(e.get("GRAPHRAG_EXTRACT_ATTEMPTS") or 3),
        max_failures=int(e.get("GRAPHRAG_MAX_FAILURES") or 50),
        write_batch_size=int(e.get("GRAPHRAG_WRITE_BATCH") or 50),
        llm_concurrency=max(1, int(e.get("GRAPHRAG_LLM_CONCURRENCY") or 2)),
    ).validate()
