"""Stubs only when the real ML packages are not installed.

If sentence_transformers (or torch) is importable, leave it alone so
tests/test_hybrid_rag.py can load real BAAI/bge-m3. Never replace
SentenceTransformer with builtin object when the package exists.
"""

from __future__ import annotations

import sys
import types

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = object
    _st.CrossEncoder = object
    sys.modules["sentence_transformers"] = _st

try:
    import torch  # noqa: F401
except ImportError:
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _torch


# ── Cross-file isolation for vLLM discovery state ────────────────────────────
# Served-model discovery keeps a process-global TTL cache
# (src.generation.vllm_discovery._cache) and the server keeps the last-known
# serving limit (src.retrieval.frontend.server._LAST_SERVING_LIMIT). Tests
# that monkeypatch provider served_models wire different fake servers through
# the same process; without resetting this state, one file's fake served set
# would leak into the next file's ACTIVE_CONFIG via the per-request refresh
# hook. Reset both around every test — isolation infrastructure only; no test
# assertions are touched.
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_vllm_discovery_state():
    try:
        from src.generation import vllm_discovery

        vllm_discovery.clear_discovery_cache()
    except Exception:  # noqa: BLE001 — discovery module optional in some suites
        pass
    try:
        import src.retrieval.frontend.server as _srv

        saved_limit = _srv._LAST_SERVING_LIMIT  # noqa: SLF001
    except Exception:  # noqa: BLE001
        saved_limit = None
    yield
    try:
        from src.generation import vllm_discovery

        vllm_discovery.clear_discovery_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        import src.retrieval.frontend.server as _srv

        _srv._LAST_SERVING_LIMIT = saved_limit  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
