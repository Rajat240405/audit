"""
Dense embedding generation using sentence-transformers.

Design Decisions
----------------
1. We use BAAI/bge-m3 — 1024 dimensions, high accuracy, CPU-viable.
   We always load the real BAAI/bge-m3 model. If loading fails, we fail-fast
   with a clear error.

2. We normalize embeddings to unit length so that inner product = cosine similarity.
   This makes FAISS IndexFlatIP equivalent to cosine similarity.

3. Batch encoding is used during index building for speed.
   Single encoding is used at query time.

4. Caching: the model is loaded once and reused across queries via a global singleton cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
import torch
import os
from pathlib import Path

# Global model singletons to prevent redundant reloading
_MODEL_CACHE: Dict[tuple[str, str], SentenceTransformer] = {}


def resolve_embed_model() -> str:
    """Pick the embedding model to use.

    Priority:
      1. GRAPHRAG_EMBED_MODEL env var (e.g. /path/to/models/bge-m3)
      2. A local ``models/bge-m3`` folder next to the repo (offline HPC)
      3. Default ``BAAI/bge-m3`` (downloads on first use)
    """
    env = os.environ.get("GRAPHRAG_EMBED_MODEL")
    if env:
        return env
    from src.utils.app_paths import model_dir

    local = model_dir() / "bge-m3"
    if local.exists():
        return str(local)
    return "BAAI/bge-m3"


class Embedder:
    """
    Dense embedding generator using sentence-transformers.
    Loads and runs the actual BAAI/bge-m3 model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name or local path.
            Default: BAAI/bge-m3 (1024-dim)
        device : str, optional
            Device: "cpu", "cuda", or None.
        normalize : bool
            If True, L2-normalize embeddings to unit length.
        """
        if model_name is None:
            model_name = resolve_embed_model()
        self.model_name = model_name
        if device is not None:
            self.device = device or "cpu"
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Embedding device: {self.device}")
        self.normalize = normalize
        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load the model. Fails fast with a clear error if resource constraints are hit."""
        global _MODEL_CACHE
        if self._model is None:
            cache_key = (self.model_name, self.device)
            if cache_key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[cache_key]
                self._embedding_dim = self._model.get_embedding_dimension()
            else:
                try:
                    # Always load the actual SentenceTransformer model
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self._embedding_dim = self._model.get_embedding_dimension()
                    _MODEL_CACHE[cache_key] = self._model
                except Exception as e:
                    # Fail-fast with clear error
                    raise RuntimeError(
                        f"Failed to load the real SentenceTransformer model '{self.model_name}' on {self.device}. "
                        f"Ensure sufficient system memory is available. Original error: {e}"
                    ) from e
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
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Encode a batch of texts into dense embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
        )
        return embeddings.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        if hasattr(self.model, "save"):
            self.model.save(str(path))

    def __repr__(self) -> str:
        return (
            f"Embedder(model={self.model_name!r}, "
            f"device={self.device!r}, "
            f"dim={self.embedding_dim})"
        )
