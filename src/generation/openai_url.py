"""OpenAI-compatible URL joins (no doubled ``/v1``)."""

from __future__ import annotations


def _v1_root(base: str) -> str:
    """Normalize an OpenAI-compatible base to its ``/v1`` root.

    ``http://host:port``      → ``http://host:port/v1``
    ``http://host:port/v1``   → ``http://host:port/v1``
    ``http://host:port/v1/``  → ``http://host:port/v1``
    """
    root = (base or "").strip().rstrip("/")
    if not root:
        root = "http://localhost:8001"
    if root.endswith("/v1"):
        return root
    return f"{root}/v1"


def chat_completions_url(base: str) -> str:
    """Chat-completions endpoint for an OpenAI-compatible base."""
    return f"{_v1_root(base)}/chat/completions"


def models_url(base: str) -> str:
    """Model-listing endpoint (``GET /v1/models``) — what THIS server
    actually serves. Used for served-model discovery on vLLM (and any
    OpenAI-compatible server)."""
    return f"{_v1_root(base)}/models"
