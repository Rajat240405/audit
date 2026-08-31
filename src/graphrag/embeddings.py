"""
Embeddings for GraphRAG.

Reuses the exact same embedding stack as Hybrid RAG
(``src.retrieval.hybrid.embedder.Embedder``, default ``BAAI/bge-m3``) — no new
model is introduced. Storing these vectors on ``(:Document)`` nodes lets Neo4j
vector search later be used, and keeps GraphRAG embedding space identical to
Hybrid RAG so scores stay comparable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from src.graphrag.config import GraphRAGConfig


class GraphEmbedder:
    """Thin wrapper around the Hybrid RAG embedder.

    The embedder is imported lazily so that the GraphRAG CLI does not force the
    entire Hybrid RAG stack (faiss / rank_bm25) to load just to import.
    """

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self._embedder = None  # type: ignore[assignment]

    @property
    def embedder(self):
        if self._embedder is None:
            from src.retrieval.hybrid.embedder import Embedder

            self._embedder = Embedder(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device,
            )
        return self._embedder

    @property
    def embedding_dim(self) -> int:
        return self.embedder.embedding_dim

    def embed(self, text: str) -> list[float]:
        vec = self.embedder.embed(text)
        return vec.astype(float).tolist()

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self.embedder.embed_batch(
            texts, batch_size=self.config.embedding_batch_size, show_progress=False
        )
