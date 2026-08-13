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
