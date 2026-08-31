"""
Production GraphRAG (Neo4j) — a completely separate pipeline from Hybrid RAG.

Submodules
----------
config      : GraphRAGConfig (env + CLI overrides)
models      : Entity / Relationship / DocumentRecord contracts
extractor   : grounded entity+relationship extraction via local Ollama
embeddings  : reuses the Hybrid RAG BAAI/bge-m3 embedder
neo4j_client: Neo4j driver wrapper (schema, inserts, queries, stats)
checkpoint  : resumable build checkpointing
pipeline    : build pipeline (verify → build → stats)
query       : graph-aware query (entity expansion + vector search)
cli         : `graphrag build|rebuild|stats|query`
"""

from src.graphrag.config import GraphRAGConfig
from src.graphrag.pipeline import GraphRAGPipeline

__all__ = ["GraphRAGConfig", "GraphRAGPipeline"]
