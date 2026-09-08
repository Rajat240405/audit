"""Canonical Neo4j GraphRAG (Phase 2).

A SECOND RETRIEVAL CAPABILITY alongside (not replacing) Hybrid RAG, backed
by Neo4j, built from the canonical corpus via the Phase 1 vocabulary, with
deterministic metadata relationships + grounded LLM semantic extraction,
multi-source provenance, and incremental (content-hash-keyed) updates.

Submodules
----------
config        : GraphConfig (env-driven; explicit Neo4j URI required)
schema        : node identities, relationship catalog, DDL
models        : open-world EntityRef / GraphFact / Support / GraphContribution
store         : GraphStore ABC + InMemoryGraphStore (tests + local validation)
neo4j_store   : Neo4jGraphStore (production backend)
deterministic : QARecord + vocabulary -> GraphContribution (no LLM)
extract       : grounded semantic extraction via the existing generation layer
pipeline      : incremental ingest (new/skip/reconcile/withdraw), checkpoint
checkpoint    : content-hash-aware resumable checkpoint
query         : lookup / 1-hop / multi-hop / temporal / provenance
capability    : graph-mode retrieval -> RetrievedResult + GraphEvidence (WS5)
cli           : `python -m src.graphrag.cli init|build|query|stats|remove`
"""

from src.graphrag.config import GraphConfig, GraphConfigError, load_graph_config
from src.graphrag.store import GraphStore, InMemoryGraphStore

__all__ = [
    "GraphConfig",
    "GraphConfigError",
    "load_graph_config",
    "GraphStore",
    "InMemoryGraphStore",
]
