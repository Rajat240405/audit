"""OpenAI-compatible URL join (no doubled ``/v1``)."""

from __future__ import annotations


def chat_completions_url(base: str) -> str:
    """Join chat completions onto an OpenAI-compatible base.

    ``http://host:port``      → ``http://host:port/v1/chat/completions``
    ``http://host:port/v1``   → ``http://host:port/v1/chat/completions``
    ``http://host:port/v1/``  → ``http://host:port/v1/chat/completions``
    """
    root = (base or "").strip().rstrip("/")
    if not root:
        root = "http://localhost:8001"
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"
