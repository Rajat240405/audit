"""
Dense embedding generation using sentence-transformers.

Design Decisions
----------------
1. We use `all-MiniLM-L6-v2` — 384 dimensions, CPU-viable, ~50ms/query.
   The RTX 3050 Mobile is not needed for this model; it runs entirely on CPU.

2. We normalize embeddings to unit length so that inner product = cosine similarity.
   This makes FAISS IndexFlatIP equivalent to cosine similarity without needing
   IndexFlatIP to do the normalization.

3. Batch encoding is used during index building for speed.
   Single encoding is used at query time.

4. Caching: the model is loaded once and reused across queries.
   This avoids the ~2-5 second model loading overhead on every query.

5. Embedding dimension is exposed so the FAISS index can be built with
   the correct dimension without hardcoding it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Dense embedding generator using sentence-transformers.

    Wraps a SentenceTransformer model with:
    - Lazy loading (model loaded on first use)
    - Unit-normalization for cosine similarity
    - Batch encoding for index building
    - Single encoding for queries

    Usage
    -----
    ```python
    embedder = Embedder()
    query_embedding = embedder.embed("What about malaria cases?")
    embeddings = embedder.embed_batch(["Q1", "Q2", "Q3"])
    ```
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name or local path.
            Default: all-MiniLM-L6-v2 (384-dim, fastest, CPU-viable)
            Alternatives: "intfloat/e5-mistral-7b-instruct" (better quality, needs GPU)
        device : str, optional
            Device: "cpu", "cuda", or None (auto-detect).
            Default None uses CPU since MiniLM-L6-v2 is faster on CPU.
        normalize : bool
            If True, L2-normalize embeddings to unit length.
            Required for IndexFlatIP to equal cosine similarity.
        """
        self.model_name = model_name
        self.device = device or "cpu"
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load the model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_embedding_dimension()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension. Loads model if needed."""
        if self._embedding_dim is None:
            _ = self.model  # Trigger lazy load
        assert self._embedding_dim is not None
        return self._embedding_dim

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """
        Encode a single text into a dense embedding vector.

        Parameters
        ----------
        text : str
            Text to encode.

        Returns
        -------
        np.ndarray (shape: [embedding_dim], dtype: float32)
            Unit-normalized embedding vector.
        """
        embedding: np.ndarray = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Ensure float32 for memory efficiency
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Encode a batch of texts into dense embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        batch_size : int
            Batch size for encoding. Higher = faster but more memory.
            Default 32 is a good balance for CPU inference.
        show_progress : bool
            Show a progress bar.

        Returns
        -------
        np.ndarray (shape: [len(texts), embedding_dim], dtype: float32)
            Matrix of unit-normalized embedding vectors.
        """
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
        )
        return embeddings.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save the model to disk for faster future loading."""
        path = Path(path)
        self.model.save(str(path))

    def __repr__(self) -> str:
        return (
            f"Embedder(model={self.model_name!r}, "
            f"device={self.device!r}, "
            f"dim={self.embedding_dim})"
        )
