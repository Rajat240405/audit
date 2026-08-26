# src/retrieval/__init__.py
#
# Lazily exposes the package's public names WITHOUT importing the heavy
# retrieval stack at package-import time.
#
# Why: importing src.retrieval.hybrid.pipeline transitively imports torch,
# sentence-transformers, FAISS and rank_bm25; importing src.retrieval.graph
# imports networkx. ANY `import src.retrieval.*` (even the light
# `src.retrieval.result`, used by the generation chain) used to trigger the
# eager `from src.retrieval.hybrid.pipeline import HybridRAGPipeline` below and
# therefore loaded the whole ML/graph stack at application startup — even in
# APP_MODE=serve where no retrieval is needed until the first query.
#
# PEP 562 module __getattr__ keeps `from src.retrieval import HybridRAGPipeline`
# / `RetrievedResult` working (for any caller/tests) while deferring the
# import until the name is actually accessed.

from __future__ import annotations

import importlib

__all__ = ["HybridRAGPipeline", "RetrievedResult"]

_LAZY: dict[str, str] = {
    "HybridRAGPipeline": "src.retrieval.hybrid.pipeline",
    "RetrievedResult": "src.retrieval.result",
}


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
