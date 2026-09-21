"""Content-hash-aware build checkpoint (Phase 2 / WS2-E).

Resumable, crash-safe checkpoint for graph ingestion:

  * persisted per-document (atomic write: temp file + rename) — a crash at
    document N resumes at N, not 1
  * entries carry the qa_content_hash — the SAME canonical content hash the
    corpus ingestion uses (excludes scraped_at): unchanged documents are
    SKIPPED without re-extraction; changed documents are reconciled
  * format is compatible with the existing ``/api/graph/build-status``
    reader (it only reads ``documents.<key>.status``)

Entry shape::

    {"status": "done"|"failed", "hash": "<sha256>", "attempts": n,
     "facts": n, "supports": n, "updated_at": "<iso>", "last_error": ...}
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["GraphCheckpoint", "CheckpointEntry", "EXTRACTION_DETERMINISTIC",
           "EXTRACTION_SEMANTIC", "normalize_extraction"]


# Extraction modes, weakest → strongest. A document built by the stronger pass
# satisfies a request for the weaker one, never the reverse.
EXTRACTION_DETERMINISTIC = "deterministic"
EXTRACTION_SEMANTIC = "semantic"
_EXTRACTION_RANK = {EXTRACTION_DETERMINISTIC: 0, EXTRACTION_SEMANTIC: 1}


def _strong_enough(have: str | None, want: str | None) -> bool:
    """True when pass ``have`` is at least as strong as ``want``.

    An empty ``have`` means "not proven" and satisfies nothing — the safe
    direction, since it sends the document to the LLM instead of skipping it.
    """
    if not have:
        return False
    return (_EXTRACTION_RANK.get(normalize_extraction(have), 0)
            >= _EXTRACTION_RANK.get(normalize_extraction(want), 0))


def normalize_extraction(value: Optional[str]) -> str:
    """Coerce a stored extraction value to a known mode.

    BACKWARD COMPATIBILITY: checkpoint files written before this field existed
    have no ``extraction`` key. Those entries are exactly the deterministic
    builds, so a missing/unknown value reads as "deterministic" — which is both
    truthful and the safe direction (it makes the document eligible for
    semantic backfill rather than silently skipping it).
    """
    v = (value or "").strip().lower()
    return v if v in _EXTRACTION_RANK else EXTRACTION_DETERMINISTIC


@dataclass
class CheckpointEntry:
    status: str                      # "done" | "failed"
    hash: str = ""
    attempts: int = 0
    facts: int = 0
    supports: int = 0
    updated_at: str = ""
    last_error: Optional[str] = None
    # which pass produced this entry: "deterministic" | "semantic"
    extraction: str = EXTRACTION_DETERMINISTIC

    # ── proof of incorporation ────────────────────────────────────────────
    # ``status`` is MUTABLE and a transient failure overwrites "done" with
    # "failed" — which is what made one bad VLLM_BASE_URL permanently re-arm
    # thousands of already-ingested documents. These fields record what was
    # actually applied to the store and are NEVER cleared by mark_failed, so
    # the skip decision rests on evidence of incorporation rather than on the
    # most recent attempt's outcome.
    applied_hash: str = ""            # qa_content_hash last successfully applied
    applied_extraction: str = ""      # which pass applied it
    # ── stable (id-neutral) identity, see src.graphrag.identity ───────────
    stable_hash: str = ""             # stable hash of the entry's own content
    applied_stable_hash: str = ""     # stable hash of the APPLIED content

    def to_dict(self) -> dict:
        return {
            "status": self.status, "hash": self.hash, "attempts": self.attempts,
            "facts": self.facts, "supports": self.supports,
            "updated_at": self.updated_at, "last_error": self.last_error,
            "extraction": self.extraction,
            "applied_hash": self.applied_hash,
            "applied_extraction": self.applied_extraction,
            "stable_hash": self.stable_hash,
            "applied_stable_hash": self.applied_stable_hash,
        }

    def satisfies(self, mode: str) -> bool:
        """True when this entry's pass is at least as strong as ``mode``."""
        return (_EXTRACTION_RANK.get(normalize_extraction(self.extraction), 0)
                >= _EXTRACTION_RANK.get(normalize_extraction(mode), 0))


class GraphCheckpoint:
    """JSON-file checkpoint store (document key -> CheckpointEntry)."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path, *, retry_failed: bool = True,
                 max_attempts: int = 3) -> None:
        self.path = Path(path)
        self.retry_failed = retry_failed
        self.max_attempts = max_attempts
        self._data: dict[str, dict] = {}
        self._run: dict = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            docs = raw.get("documents", {}) if isinstance(raw, dict) else {}
            self._data = {k: dict(v) for k, v in docs.items() if isinstance(v, dict)}
            run = raw.get("run") if isinstance(raw, dict) else None
            self._run = dict(run) if isinstance(run, dict) else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("graph checkpoint unreadable (%s): %s — starting fresh",
                           self.path, e)
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "documents": self._data,
            "meta": {
                "done": sum(1 for v in self._data.values() if v.get("status") == "done"),
                "failed": sum(1 for v in self._data.values() if v.get("status") == "failed"),
            },
            "run": self._run,
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── queries ───────────────────────────────────────────────────────────

    def get(self, doc_key: str) -> Optional[CheckpointEntry]:
        d = self._data.get(doc_key)
        if d is None:
            return None
        return CheckpointEntry(
            status=d.get("status", ""), hash=d.get("hash", ""),
            attempts=int(d.get("attempts", 0)), facts=int(d.get("facts", 0)),
            supports=int(d.get("supports", 0)),
            updated_at=d.get("updated_at", ""), last_error=d.get("last_error"),
            # missing key (pre-existing checkpoints) → "deterministic"
            extraction=normalize_extraction(d.get("extraction")),
            # missing keys (pre-existing checkpoints) → "" = "not proven".
            # The absence of proof is the safe direction: it sends the
            # document to the LLM rather than skipping work that was never
            # shown to be done.
            applied_hash=d.get("applied_hash", ""),
            applied_extraction=normalize_extraction(d.get("applied_extraction"))
            if d.get("applied_extraction") else "",
            stable_hash=d.get("stable_hash", ""),
            applied_stable_hash=d.get("applied_stable_hash", ""),
        )

    def items(self) -> dict[str, dict]:
        """Read-only view of the raw persisted entries (for migration)."""
        return {k: dict(v) for k, v in self._data.items()}

    def update_fields(self, updates: dict[str, dict]) -> int:
        """Merge ``{doc_key: {field: value}}`` into existing entries.

        Additive by construction — only the named fields change, ``status``
        and ``hash`` are untouched unless explicitly named. One atomic write
        for the whole batch. Returns the number of entries modified.
        """
        changed = 0
        for key, fields in updates.items():
            if key not in self._data or not fields:
                continue
            self._data[key].update(fields)
            changed += 1
        if changed:
            self._save()
        return changed

    def is_done_with_hash(self, doc_key: str, content_hash: str) -> bool:
        """True iff this exact content was already ingested successfully —
        the skip decision for unchanged documents."""
        e = self.get(doc_key)
        return e is not None and e.status == "done" and e.hash == content_hash

    def incorporation_for(self, doc_key: str, content_hash: str,
                          stable_hash: str, mode: str) -> bool:
        """True iff THIS key proves ``content_hash`` is already in the store.

        Reads ``applied_hash``/``applied_extraction`` — the durable proof —
        and deliberately ignores ``status``, because a transient failure
        overwrites status without touching the graph.
        """
        e = self.get(doc_key)
        if e is None or not e.applied_hash:
            return False
        if not _strong_enough(e.applied_extraction, mode):
            return False
        return e.applied_hash == content_hash

    def churn_candidates(self, stable_hash: str, exclude: str,
                         mode: str, corpus_keys: set | None = None) -> list[str]:
        """Other keys that already hold this exact content under another id.

        ``corpus_keys`` is the set of ids in the CURRENT corpus and is
        essential, not an optimisation: a candidate that is itself still in the
        corpus is not a renamed document, it is a SECOND document that happens
        to carry identical content. Treating that as a rename would silently
        merge two distinct documents into one. Only keys absent from the
        corpus can be the source of a rename.

        Returns the full list rather than a boolean so the caller can tell
        "exactly one" (a safe re-key) from "several" (ambiguous — must go to
        the LLM). Ambiguity is never resolved into a skip.
        """
        if not stable_hash:
            return []
        out = []
        for k, v in self._data.items():
            if k == exclude or k not in self._data:
                continue
            if corpus_keys is not None and k in corpus_keys:
                continue          # still a live document, not a rename source
            if (v.get("applied_stable_hash") == stable_hash
                    and v.get("applied_hash")
                    and _strong_enough(v.get("applied_extraction", ""), mode)):
                out.append(k)
        return sorted(out)

    def needs_extraction(self, doc_key: str, content_hash: str,
                         mode: str, stable_hash: str = "",
                         corpus_keys: set | None = None) -> bool:
        """True when this document still needs the ``mode`` pass.

        Needs work when nothing proves its CURRENT content is already in the
        store at the required strength. This is the exact complement of the
        pipeline's skip/re-key decision, so the LLM prefetcher can never
        pre-compute an extraction the loop is about to skip.

        A same-content id change needs a re-key, not an extraction, so it
        returns False — but ONLY when exactly one key absent from the corpus
        claims that content. Several claimants are ambiguous and DO need the
        LLM.
        """
        if self.incorporation_for(doc_key, content_hash, stable_hash, mode):
            return False
        churn = (self.churn_candidates(stable_hash, doc_key, mode, corpus_keys)
                 if stable_hash else [])
        return len(churn) != 1

    def should_retry(self, doc_key: str) -> bool:
        e = self.get(doc_key)
        if e is None:
            return True
        if e.status == "done":
            return False
        if not self.retry_failed:
            return False
        return e.attempts < self.max_attempts

    def keys(self) -> set[str]:
        return set(self._data)

    # ── mutations (each persists atomically) ──────────────────────────────

    def mark_done(self, doc_key: str, content_hash: str, *, now: str,
                  facts: int = 0, supports: int = 0,
                  extraction: str = EXTRACTION_DETERMINISTIC,
                  stable_hash: str = "") -> None:
        """Persist a successful document.

        ``extraction`` records WHICH pass produced it. The caller must only
        pass "semantic" once the semantic contribution has actually been
        applied to the store — this is the flag semantic backfill selects on.

        ``applied_hash``/``applied_extraction`` are the durable proof of
        incorporation; ``mark_failed`` preserves them, so a later transient
        failure cannot erase the fact that this content reached the graph.
        """
        prev = self.get(doc_key)
        mode = normalize_extraction(extraction)
        self._data[doc_key] = CheckpointEntry(
            status="done", hash=content_hash,
            attempts=(prev.attempts + 1) if prev else 1,
            facts=facts, supports=supports, updated_at=now,
            extraction=mode,
            applied_hash=content_hash,
            applied_extraction=mode,
            stable_hash=stable_hash or (prev.stable_hash if prev else ""),
            applied_stable_hash=stable_hash
            or (prev.applied_stable_hash if prev else ""),
        ).to_dict()
        self._save()

    def mark_failed(self, doc_key: str, content_hash: str, error: str,
                    *, now: str, stable_hash: str = "") -> None:
        """Record a failed attempt WITHOUT destroying proven state.

        ``status`` becomes "failed" — that is correct and the UI depends on
        it — but ``applied_hash``/``applied_extraction`` are carried over
        verbatim. They are what the skip decision reads, so a transient
        failure on a document whose content is already in the graph no longer
        re-arms it for re-extraction on every subsequent run.
        """
        prev = self.get(doc_key)
        self._data[doc_key] = CheckpointEntry(
            status="failed", hash=content_hash,
            attempts=(prev.attempts + 1) if prev else 1,
            facts=prev.facts if prev else 0,
            supports=prev.supports if prev else 0,
            updated_at=now, last_error=str(error)[:500],
            # A failure never upgrades the recorded pass; keep what was already
            # proven so a failed semantic retry cannot masquerade as semantic.
            extraction=prev.extraction if prev else EXTRACTION_DETERMINISTIC,
            # ...and never downgrades it either. This is the fix: the proof
            # that the content reached the store survives the failure.
            applied_hash=prev.applied_hash if prev else "",
            applied_extraction=prev.applied_extraction if prev else "",
            stable_hash=stable_hash or (prev.stable_hash if prev else ""),
            applied_stable_hash=prev.applied_stable_hash if prev else "",
        ).to_dict()
        self._save()

    def remove(self, doc_key: str) -> None:
        if doc_key in self._data:
            del self._data[doc_key]
            self._save()

    # ── run state (build progress for /api/graph/build-status) ────────────
    #
    # The build is normally launched detached (nohup singularity exec ...), so
    # the serving process cannot observe it directly. The checkpoint file is
    # the shared source of truth: the builder records its run state here and
    # the API reads it back. Persisted in the SAME atomic write as the
    # per-document entries, so progress can never be newer than the work.
    #
    # Liveness: a build killed by SIGKILL / node reboot never writes
    # "completed" or "failed", which would strand the UI on "running" forever.
    # The reader therefore treats a stale heartbeat as "interrupted"
    # (see is_stale / state_for_reader), rather than trusting the flag.

    def begin_run(self, *, total: int, now: str, pid: int | None = None,
                  mode: str = "build") -> None:
        """Mark a build as started (resets per-run counters, keeps documents)."""
        self._run = {
            "state": "running",
            "mode": mode,
            "pid": int(pid) if pid is not None else None,
            "total": int(total),
            "processed": 0,
            "added": 0,
            "reconciled": 0,
            "skipped_unchanged": 0,
            "failed": 0,
            "current_doc": None,
            "last_doc": None,
            "started_at": now,
            "heartbeat_at": now,
            "finished_at": None,
            "error": None,
        }
        self._save()

    def update_run(self, *, now: str, **fields) -> None:
        """Update run counters + heartbeat. No-op if no run was begun."""
        if not self._run:
            return
        self._run.update(fields)
        self._run["heartbeat_at"] = now

    def end_run(self, *, now: str, state: str = "completed",
                error: str | None = None) -> None:
        if not self._run:
            return
        self._run["state"] = state
        self._run["current_doc"] = None
        self._run["heartbeat_at"] = now
        self._run["finished_at"] = now
        if error:
            self._run["error"] = str(error)[:500]
        self._save()

    def flush_run(self) -> None:
        """Persist run state without a document mutation (skip-only stretches)."""
        if self._run:
            self._save()

    def run_state(self) -> dict:
        return dict(self._run)

    def counts(self) -> dict:
        done = sum(1 for v in self._data.values() if v.get("status") == "done")
        failed = sum(1 for v in self._data.values() if v.get("status") == "failed")
        return {"done": done, "failed": failed, "total_tracked": len(self._data)}
