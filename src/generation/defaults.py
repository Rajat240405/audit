"""
Single source of truth for the DEFAULT LLM configuration.

Before this module, the default model + context window were stated in four
places (client.py, registry.py, server.py ACTIVE_CONFIG, cli.py) and had
already drifted (qwen2.5:7b vs qwen3:8b; num_ctx 8192/16384/32768). If you
want to change the system default, change DEFAULT_FAMILY_ID here — everything
else resolves from the registry.
"""

from __future__ import annotations

# The default model FAMILY id as registered in src.generation.registry.
# "qwen3" resolves to qwen3:8b (32768 ctx, thinking-capable) via the registry.
DEFAULT_FAMILY_ID = "qwen3"

# Fallback context window ONLY if the registry lookup fails entirely.
FALLBACK_NUM_CTX = 8192


def default_family_id() -> str:
    return DEFAULT_FAMILY_ID


def default_num_ctx() -> int:
    """Resolve the default context window from the registry; fallback
    constant if the family is somehow missing."""
    try:
        from src.generation.registry import model_registry

        fam = model_registry.get(DEFAULT_FAMILY_ID)
        if fam and getattr(fam, "context_window", None):
            return int(fam.context_window)
    except Exception:  # noqa: BLE001
        pass
    return FALLBACK_NUM_CTX


def default_model_name() -> str:
    """Resolve the default concrete model string from the registry."""
    try:
        from src.generation.registry import model_registry

        fam = model_registry.get(DEFAULT_FAMILY_ID)
        if fam and getattr(fam, "model_name", None):
            return fam.model_name
    except Exception:  # noqa: BLE001
        pass
    return "qwen3:8b"
