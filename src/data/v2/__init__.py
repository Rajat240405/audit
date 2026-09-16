"""Experimental INCOIS V2 extraction/retrieval layer — feature-gated.

This package is **additive and optional**. Nothing in the production pipeline
imports it unless ``V2_ENABLED`` is explicitly truthy, and every entry point
must be reached through :func:`src.data.v2.config.load_v2_config` first.

Gating contract (enforced by ``tests/test_v2_disabled_is_inert.py``):

* ``V2_ENABLED`` unset or false is the production default. In that state this
  package must not be imported at all by ingestion, retrieval, or generation —
  a missing optional dependency (``pytesseract``, ``opencv``, the
  ``tesseract-ocr-hin`` language pack) therefore cannot break production.
* Importing :mod:`src.data.v2.config` adds **no extraction, OCR, ML, or vision
  dependency** — no ``pymupdf``/``fitz``, ``pytesseract``, ``cv2``, ``PIL``,
  ``numpy``, ``faiss``, ``torch`` or ``sentence_transformers``. The repo's
  always-present core packages (``pydantic``, ``rich``, ``orjson``, ``yaml``)
  still load, because ``src/data/__init__.py`` eagerly imports
  ``enricher``/``loader``/``validator``; that is pre-existing behaviour shared
  with every other ``src.data.*`` module, not something this package adds. The
  heavy extraction dependencies are imported lazily inside the extractor
  modules, never at package import time.
* Sub-capabilities are conjunctive with the master switch and are exposed as
  computed properties (``tables_active``, ``figures_active``, …) so a call site
  cannot enable a feature by checking a sub-flag alone.

Reference implementation (prototype, **not** production code) lives outside the
repository in the Phase-8/9 research package ``v2proto/``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
