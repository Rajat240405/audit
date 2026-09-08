"""Domain vocabulary & data foundation (Phase 1 / WS1).

Additive, read-only layer: controlled vocabularies (single source:
``config/vocabulary.yaml``), deterministic normalizers that preserve raw
values, and the reproducible corpus-quality report. Nothing here modifies
the canonical corpus schema, touches retrieval, or creates entities — it is
the documented foundation a future graph build (WS2) will map onto.
"""
from src.vocabulary.vocab import (
    Vocabulary,
    VocabularyEntry,
    VocabularyError,
    load_vocabulary,
)
from src.vocabulary import normalize

# NOTE: corpus_quality is NOT imported here — it is a CLI entrypoint
# (`python -m src.vocabulary.corpus_quality`); importing it from the package
# __init__ triggers a runpy warning. Import it directly where needed.

__all__ = [
    "Vocabulary",
    "VocabularyEntry",
    "VocabularyError",
    "load_vocabulary",
    "normalize",
]
