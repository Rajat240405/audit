"""One-shot, idempotent migration of an existing build checkpoint.

WHY THIS IS NEEDED
------------------
Checkpoints written before this change carry ``status``, ``hash`` and
``extraction`` only. The incremental fix decides whether to skip a document by
reading the *proof of incorporation* (``applied_hash`` / ``applied_extraction``)
and the *stable identity* (``stable_hash`` / ``applied_stable_hash``). Without
those fields nothing is provable, so an un-migrated checkpoint would send the
whole corpus to the LLM — exactly the full rebuild this migration prevents.

WHAT IT DOES
------------
Purely additive. For each entry it fills only the fields that are still empty:

``stable_hash``
    Recovered by matching the entry's stored ``hash`` against the current
    corpus. ``qa_content_hash`` covers question_id + text + metadata, so an
    exact hash match proves the entry was written against *this* record and
    the stable hash can be recomputed from it. A hash that matches no current
    record, or more than one, is left empty — never guessed.

``applied_hash`` / ``applied_extraction``
    * ``status == "done"`` -> taken from the entry itself. Sound because
      ``mark_done`` only runs after ``apply_contribution`` returned.
    * ``status == "failed"`` -> NOT taken from the entry, because a failure
      says nothing about the graph. Recovered only from graph evidence: the
      Document node's stored ``content_hash`` must equal the entry hash AND
      the document must have at least one ``origin == 'llm'`` SUPPORTS edge.
      Node presence alone is explicitly NOT accepted.

WHAT IT NEVER DOES
------------------
It never changes ``status``, never rewrites ``hash``, never deletes an entry,
never resets the file, and never touches the graph. Running it twice is a
no-op. An entry it cannot prove is left unproven — which sends that document to
the LLM, the safe direction.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from src.graphrag.checkpoint import EXTRACTION_SEMANTIC, GraphCheckpoint, normalize_extraction
from src.graphrag.identity import stable_content_hash
from src.scripts.ingest_folder import qa_content_hash

__all__ = ["migrate_checkpoint", "plan_migration"]

logger = logging.getLogger(__name__)


def plan_migration(ckpt: GraphCheckpoint, records, *, store=None):
    """Compute the migration WITHOUT writing anything.

    Returns ``(updates, stats)`` where ``updates`` is ``{doc_key: {field:
    value}}``. The read-only audit tool uses this so its prediction of the
    fixed build is produced by the same code the build will run, not a copy
    that can drift.
    """
    return _plan(ckpt, records, store=store)


def migrate_checkpoint(ckpt: GraphCheckpoint, records, *,
                       store=None, dry_run: bool = False) -> dict:
    """Backfill stable identity + incorporation proof on ``ckpt``.

    ``store`` is optional but strongly recommended: without it, ``failed``
    entries cannot be proven and will be re-extracted. Pass the same store the
    build will use so the evidence comes from the graph that build will read.
    """
    updates, stats = _plan(ckpt, records, store=store)
    stats["modified"] = len(updates)
    if updates and not dry_run:
        ckpt.update_fields(updates)
    logger.info("[graph] checkpoint migration: %s", stats)
    return stats


def _plan(ckpt, records, *, store=None):
    # production hash -> records, so a stored hash can be resolved back to the
    # content it was computed from (the digest itself is not invertible)
    by_hash: dict[str, list] = defaultdict(list)
    for r in records:
        by_hash[qa_content_hash(r)].append(r)

    entries = ckpt.items()
    stats = {
        "entries": len(entries),
        "stable_hash_filled": 0,
        "stable_hash_unresolvable": 0,
        "stable_hash_ambiguous": 0,
        "applied_from_done": 0,
        "applied_from_graph": 0,
        "failed_unproven": 0,
        "already_migrated": 0,
        "modified": 0,
    }

    # Only query the graph for entries that actually need graph evidence.
    needs_graph = [k for k, v in entries.items()
                   if not v.get("applied_hash") and v.get("status") == "failed"]
    evidence: dict[str, dict] = {}
    if needs_graph and store is not None:
        evidence = store.incorporated_evidence(needs_graph)
    stats["failed_checked_against_graph"] = len(needs_graph)
    stats["graph_evidence_available"] = bool(needs_graph and store is not None)

    updates: dict[str, dict] = {}
    for key, raw in entries.items():
        fields: dict = {}
        h = raw.get("hash", "")

        if not raw.get("stable_hash"):
            matches = by_hash.get(h, [])
            if len(matches) == 1:
                fields["stable_hash"] = stable_content_hash(matches[0])
                stats["stable_hash_filled"] += 1
            elif len(matches) > 1:      # duplicate question_id in the corpus
                stats["stable_hash_ambiguous"] += 1
            else:
                stats["stable_hash_unresolvable"] += 1
        stable = fields.get("stable_hash") or raw.get("stable_hash", "")

        if not raw.get("applied_hash"):
            status = raw.get("status")
            if status == "done":
                # mark_done only runs after apply_contribution succeeded
                fields["applied_hash"] = h
                fields["applied_extraction"] = normalize_extraction(
                    raw.get("extraction"))
                if stable:
                    fields["applied_stable_hash"] = stable
                stats["applied_from_done"] += 1
            elif status == "failed":
                ev = evidence.get(key)
                if (ev is not None and ev.get("content_hash") == h
                        and ev.get("has_llm_facts")):
                    # the graph itself proves this exact content is in it,
                    # with a semantic contribution attached
                    fields["applied_hash"] = h
                    fields["applied_extraction"] = EXTRACTION_SEMANTIC
                    if stable:
                        fields["applied_stable_hash"] = stable
                    stats["applied_from_graph"] += 1
                else:
                    stats["failed_unproven"] += 1
        else:
            stats["already_migrated"] += 1

        if fields:
            updates[key] = fields

    return updates, stats


def preview_migration(ckpt: GraphCheckpoint, records, *,
                      store=None) -> dict:
    """Same computation, nothing written. Safe beside a running build."""
    return migrate_checkpoint(ckpt, records, store=store, dry_run=True)
