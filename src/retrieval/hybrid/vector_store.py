"""
FAISS vector store for dense retrieval.

Design Decisions
----------------
1. We use FAISS IndexFlatIP (Inner Product) because embeddings are
   unit-normalized — IndexFlatIP == cosine similarity.
   This is faster than IndexFlatIP with a separate normalization step.

2. We use a simple flat index (no quantization) because:
   - Our dataset is 3,500 records × 384 dimensions ≈ 5.4MB
   - Exact search is fast enough at this scale
   - No accuracy trade-off needed

3. ID mapping: FAISS only supports integer IDs internally (0, 1, 2, ...).
   We maintain an external dict: internal_id → doc_id (string).
   This lets us store string doc_ids in the retrieval result.

4. The index is built once (offline) and queried many times (online).
   We separate build() and load() for this reason.

5. We support incremental additions via add() after initial build.

6. FAISS index + id_map are saved as two separate files:
   {path}.index  — the FAISS index binary
   {path}.ids    — JSON list of doc_ids in order
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import numpy.typing as npt

from src.retrieval.result import RetrievedResult


class FAISSVectorStore:
    """
    FAISS-backed dense vector store for top-K similarity search.

    Stores document embeddings and supports fast top-K retrieval.

    Usage
    -----
    ```python
    store = FAISSVectorStore(embedding_dim=384)

    # Build from embeddings (once)
    store.build(doc_ids=["18-1", "18-2", ...], embeddings=np.array([...]))

    # Search (many times)
    results = store.search(query_embedding, k=5)
    ```
    """

    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        embedding_dim : int
            Dimension of the embedding vectors.
        index_path : str, optional
            Path to load/save the FAISS index.
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path) if index_path else None
        self._index: Optional[faiss.Index] = None
        self._doc_ids: list[str] = []

    def build(
        self,
        doc_ids: list[str],
        embeddings: npt.NDArray[np.float32],
    ) -> None:
        """
        Build the FAISS index from document embeddings.

        Parameters
        ----------
        doc_ids : list[str]
            Ordered list of document IDs corresponding to embeddings.
            doc_ids[i] is the ID for embeddings[i].
        embeddings : np.ndarray (shape: [n, embedding_dim])
            Embedding matrix. Must match doc_ids order.

        Raises
        ------
        ValueError
            If n != len(doc_ids) or embeddings.shape[1] != embedding_dim.
        """
        n = len(doc_ids)
        if n != len(embeddings):
            raise ValueError(f"doc_ids count ({n}) != embeddings count ({len(embeddings)})")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Create flat index — inner product (cosine similarity after normalization)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        # Add embeddings (must be float32, already normalized)
        self._index.add(embeddings.astype(np.float32))
        self._doc_ids = list(doc_ids)

    def add(
        self,
        doc_ids: list[str],
        embeddings: npt.NDArray[np.float32],
    ) -> None:
        """
        Add new embeddings to the existing index.

        Parameters
        ----------
        doc_ids : list[str]
            Document IDs for the new embeddings.
        embeddings : np.ndarray
            New embeddings to add.
        """
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build() first.")
        self._index.add(embeddings.astype(np.float32))
        self._doc_ids.extend(doc_ids)

    def search(
        self,
        query_embedding: npt.NDArray[np.float32],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-K most similar documents.

        Parameters
        ----------
        query_embedding : np.ndarray (shape: [embedding_dim])
            Query embedding vector. Must be unit-normalized.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (doc_id, similarity_score) sorted by score descending.
            Empty list if index is empty.
        """
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build() first.")

        # Ensure 2D array [1, dim]
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k_actual = min(k, self._index.ntotal)
        if k_actual == 0:
            return []

        scores, indices = self._index.search(
            query_embedding.astype(np.float32),
            k_actual,
        )

        # Map internal indices to doc_ids
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self._doc_ids):
                results.append((self._doc_ids[int(idx)], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        """
        Save the index and doc_ids to disk.

        Parameters
        ----------
        path : str | Path
            Base path. Two files are written:
            - {path}.index  — FAISS binary
            - {path}.ids    — JSON list of doc_ids
        """
        if self._index is None:
            raise RuntimeError("Cannot save empty index.")

        path = Path(path)
        # Save FAISS index
        faiss.write_index(self._index, str(path.with_suffix(".index")))
        # Save doc_id mapping
        with open(path.with_suffix(".ids"), "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f)

    def load(self, path: str | Path) -> None:
        """
        Load the index and doc_ids from disk.

        Parameters
        ----------
        path : str | Path
            Base path (without extension).
        """
        path = Path(path)
        index_file = path.with_suffix(".index")
        ids_file = path.with_suffix(".ids")

        if not index_file.exists() or not ids_file.exists():
            raise FileNotFoundError(f"Index files not found at {path}")

        self._index = faiss.read_index(str(index_file))
        with open(ids_file, encoding="utf-8") as f:
            self._doc_ids = json.load(f)

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return f"FAISSVectorStore(dim={self.embedding_dim}, n_docs={len(self._doc_ids)})"
