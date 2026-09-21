"""Stable document identity for incremental graph builds.

The production content hash (``qa_content_hash``) deliberately includes
``question_id``: it is the hash the corpus ingestion and the build checkpoint
have always used, and changing it would invalidate every existing entry.

That makes it the wrong tool for one specific question — "is this the same
underlying document?" — because a regenerated ``question_id`` over
byte-identical text produces a different hash and the document is treated as
changed.

This module derives an id-neutral identity from the SAME canonicalisation, so
the two hashes can never disagree about what counts as content:

* ``stable_content_hash`` — production hash with ONLY ``question_id``
  neutralised. Every other field keeps its production weight, *including
  metadata*, because metadata feeds the deterministic graph contribution and
  is therefore part of content identity for graph purposes.
* ``stable_text_hash`` — a weaker text-only digest. It is the strongest key
  that can be recomputed from a stored ``:Document`` node, so it is the
  fallback when no previous corpus is available.

Both are pure functions of the record. Neither is a replacement for
``qa_content_hash``; they answer a different question.
"""
from __future__ import annotations

import hashlib

from src.models.qa_record import QARecord
from src.scripts.ingest_folder import qa_content_hash

__all__ = ["STABLE_ID_SENTINEL", "stable_content_hash", "stable_text_hash"]

# Substituted for question_id before hashing. NUL-delimited so it cannot
# collide with a real id, and constant so every caller derives the same digest.
STABLE_ID_SENTINEL = "\x00stable\x00"


def stable_content_hash(rec: QARecord) -> str:
    """Production content hash with ONLY the volatile ``question_id`` removed.

    Metadata is retained on purpose: it feeds the deterministic contribution,
    so two records with identical text but different metadata contribute
    different facts and are NOT the same document for graph purposes.
    """
    neutral = rec.model_copy(update={"question_id": STABLE_ID_SENTINEL})
    return qa_content_hash(neutral)


def stable_text_hash(question_text: str, answer_text: str) -> str:
    """Text-only identity, recomputable from a stored ``:Document`` node.

    Weaker than :func:`stable_content_hash` — metadata changes are invisible to
    it — so it is used only when the full-fidelity key is unavailable, and
    callers must treat its matches as lower-confidence.
    """
    payload = "\x00".join([(question_text or "").strip(),
                           (answer_text or "").strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
