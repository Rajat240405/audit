"""Shared curated-knowledge store: one contribution per file, UUID-keyed.

Design constraints this module exists to satisfy
------------------------------------------------
* **Never overwrite another user's contribution.** Identity is a server-generated
  UUID, never the question slug. Two people saving the same question produce two
  independent files.
* **Never lose a record to a partial write.** Every write goes to a temporary
  file in the same directory followed by ``os.replace``, which is atomic on
  POSIX and on Windows. Mirrors ``src/graphrag/checkpoint.py:_save``.
* **The index is derived, never authoritative.** ``index.json`` and
  ``questions.f32`` can be deleted and rebuilt from ``records/``. A missing or
  corrupt index must never make saved knowledge unreadable.
* **Frozen unless edited.** Saving never mutates an existing record; only an
  explicit edit bumps ``version``/``updated_at``.

Legacy flat ``<slug>.json`` files are still *read* (see ``read_legacy``) so the
server works before, during and after migration, but nothing here writes them.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from src.utils.app_paths import user_knowledge_dir
from src.utils.atomic_io import dump_json_atomic, write_bytes_atomic

__all__ = [
    "SCHEMA_VERSION",
    "KnowledgeStore",
    "content_hash",
    "normalize_question",
    "owner_id_from",
    "trim_sources",
]

SCHEMA_VERSION = 2

#: Matches the legacy ``_normalize_q`` in server.py exactly, so tier-1 exact
#: matching keeps the behaviour existing tests already pin.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")

_ACTIVE = "active"
_ARCHIVED = "archived"


def normalize_question(text: str) -> str:
    """Lowercase, non-alphanumerics collapsed to single spaces, trimmed."""
    t = _NON_ALNUM.sub(" ", (text or "").lower())
    return _WS.sub(" ", t).strip()


def content_hash(question: str, answer: str) -> str:
    """Stable identity of a question+answer pair, for duplicate grouping.

    Uses the NORMALISED question but the raw answer, and a NUL separator so
    ``("ab", "c")`` cannot collide with ``("a", "bc")``.
    """
    payload = f"{normalize_question(question)}\x00{(answer or '').strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def owner_id_from(raw: str) -> str:
    """Opaque, stable owner key derived from a display identity.

    A hash rather than the raw name so a display rename does not orphan
    ownership, and so the id is safe to use in URLs and logs. NOT a security
    boundary — v1 identity is a manually entered name.
    """
    key = (raw or "").strip().lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def trim_sources(sources: Iterable[dict] | None) -> list[dict]:
    """Citation identity only.

    Preserved verbatim from the original implementation: full document texts
    already live in the main index, and storing them here bloats every file and
    slows every lookup.
    """
    out = []
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        out.append({
            "doc_id": s.get("doc_id") or "",
            "subject": s.get("subject") or "",
            "ministry": s.get("ministry") or "",
            "document_type": s.get("document_type") or "",
            "score": s.get("score"),
        })
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class KnowledgeStore:
    """File-backed store for shared curated knowledge."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else user_knowledge_dir()
        self.records_dir = self.root / "records"
        self.index_path = self.root / "index.json"
        self.embeddings_path = self.root / "questions.f32"
        self._lock = threading.RLock()
        self._index_cache: list[dict] | None = None
        self._index_mtime: float | None = None

    # ── paths / plumbing ───────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, dest: Path, obj: dict) -> None:
        """Atomic write via the codebase's shared helper (temp + os.replace).

        Reuses ``src.utils.atomic_io`` rather than hand-rolling it: that module
        already has tests for exactly the property this store depends on — a
        failed write leaves the previous file intact.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        dump_json_atomic(dest, obj, ensure_ascii=False, indent=1)

    def _record_path(self, knowledge_id: str) -> Path:
        # knowledge_id is server-generated UUID hex; validate anyway so a
        # crafted id can never escape the records directory.
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", knowledge_id or ""):
            raise ValueError(f"invalid knowledge_id: {knowledge_id!r}")
        return self.records_dir / f"{knowledge_id}.json"

    # ── reads ──────────────────────────────────────────────────────────────

    def get(self, knowledge_id: str) -> dict | None:
        try:
            path = self._record_path(knowledge_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def all_records(self, *, include_archived: bool = False) -> list[dict]:
        """Every record on disk, newest first. A corrupt file is skipped, not fatal."""
        if not self.records_dir.exists():
            return []
        out: list[dict] = []
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict) or not rec.get("knowledge_id"):
                continue
            if not include_archived and rec.get("lifecycle", _ACTIVE) != _ACTIVE:
                continue
            out.append(rec)
        out.sort(key=lambda r: (r.get("created_at") or "", r.get("knowledge_id") or ""))
        return out

    def active_records(self) -> list[dict]:
        """Every matchable record: v2 files, or legacy files when unmigrated."""
        recs = self.all_records()
        return recs if recs else self.legacy_as_records()

    def records_for_ids(self, ids: set) -> list[dict]:
        """Records with these ids, searching v2 then the legacy fallback."""
        found = [r for r in self.all_records() if r["knowledge_id"] in ids]
        if not found:
            found = [r for r in self.legacy_as_records() if r["knowledge_id"] in ids]
        return found

    def for_owner(self, owner_id: str, *, include_archived: bool = False) -> list[dict]:
        pool = (self.all_records(include_archived=include_archived)
                if include_archived else self.active_records())
        return [r for r in pool if r.get("owner_id") == owner_id]

    # ── index ──────────────────────────────────────────────────────────────

    def legacy_as_records(self) -> list[dict]:
        """Present legacy flat files in v2 shape, WITHOUT writing anything.

        Read-only compatibility for a deployment that has not migrated yet: it
        keeps matching working exactly as before instead of silently returning
        nothing. Migration itself stays an explicit, separate step.

        Ids are derived deterministically from the filename so they are stable
        across reads even though no v2 file exists.
        """
        out = []
        for data in self.read_legacy():
            question = str(data.get("question") or "").strip()
            answer = str(data.get("answer") or "").strip()
            if not question or not answer:
                continue
            owner_name = str(data.get("saved_by") or "").strip()
            ts = str(data.get("saved_at") or "")
            stem = str(data.get("_file") or "")[:-5] or normalize_question(question)[:32]
            kid = "legacy-" + hashlib.sha256(stem.encode("utf-8")).hexdigest()[:24]
            out.append({
                "schema_version": 1,
                "knowledge_id": kid,
                "content_hash": content_hash(question, answer),
                "question": question,
                "question_normalized": normalize_question(question),
                "answer": answer,
                "sources": trim_sources(data.get("sources")),
                "owner_id": owner_id_from(owner_name),
                "owner_name": owner_name,
                "created_at": ts, "updated_at": ts, "version": 1,
                "lifecycle": _ACTIVE, "has_embedding": False,
                "legacy": True, "_file": data.get("_file"),
            })
        return out

    def build_index(self, records: list[dict] | None = None) -> list[dict]:
        """Derive the lookup index from the records actually on disk.

        Falls back to legacy flat files when ``records/`` holds nothing, so an
        unmigrated deployment keeps working.
        """
        recs = records if records is not None else self.all_records()
        if not recs:
            recs = self.legacy_as_records()
        index = []
        for row, rec in enumerate(recs):
            index.append({
                "row": row,
                "knowledge_id": rec["knowledge_id"],
                "q_norm": (rec.get("question_normalized")
                           or normalize_question(rec.get("question", ""))),
                "content_hash": rec.get("content_hash", ""),
                "owner_id": rec.get("owner_id", ""),
                "owner_name": rec.get("owner_name", ""),
                "created_at": rec.get("created_at", ""),
                "updated_at": rec.get("updated_at", ""),
                "version": rec.get("version", 1),
                "lifecycle": rec.get("lifecycle", _ACTIVE),
                "has_embedding": bool(rec.get("has_embedding")),
            })
        return index

    def load_index(self) -> list[dict]:
        """Cached index, transparently rebuilt when missing or stale.

        A damaged index must never make knowledge unreadable — the records are
        the source of truth, so we fall back to rebuilding from disk.
        """
        with self._lock:
            try:
                mtime = self.index_path.stat().st_mtime
            except OSError:
                mtime = None
            if self._index_cache is not None and mtime == self._index_mtime:
                return self._index_cache
            index: list[dict] | None = None
            if mtime is not None:
                try:
                    loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
                        index = loaded["entries"]
                except (OSError, ValueError):
                    index = None
            if index is None:
                index = self.build_index()
                self.save_index(index)
            self._index_cache = index
            self._index_mtime = mtime
            return index

    def save_index(self, index: list[dict]) -> None:
        with self._lock:
            self._atomic_write_json(self.index_path, {
                "schema_version": SCHEMA_VERSION, "count": len(index), "entries": index,
            })
            self._index_cache = index
            try:
                self._index_mtime = self.index_path.stat().st_mtime
            except OSError:
                self._index_mtime = None

    def invalidate(self) -> None:
        with self._lock:
            self._index_cache = None
            self._index_mtime = None

    def refresh_index(self) -> list[dict]:
        with self._lock:
            index = self.build_index()
            self.save_index(index)
            return index

    # ── embeddings matrix ──────────────────────────────────────────────────

    def load_embeddings(self):
        """The saved-question embedding matrix, or None if absent/unusable."""
        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            return None
        if not self.embeddings_path.exists():
            return None
        try:
            return np.fromfile(str(self.embeddings_path), dtype=np.float32).reshape(
                self._embedding_shape()
            )
        except (OSError, ValueError):
            return None

    def _embedding_shape(self) -> tuple[int, int]:
        """Row count from the index; column count inferred from file size.

        The matrix has one row per index ENTRY (positional), regardless of
        whether that row's embedding is still current — ``has_embedding`` marks
        currency, not presence. A matrix that does not divide evenly by the
        index length is stale garbage and must be rejected, not reshaped.
        """
        rows = len(self.load_index())
        size = self.embeddings_path.stat().st_size
        if rows <= 0 or size <= 0 or (size // 4) % rows:
            raise ValueError("embedding matrix does not align with the index")
        cols = (size // 4) // rows
        return rows, cols

    def save_embeddings(self, matrix, index: list[dict]) -> None:
        """Persist the matrix and mark which index rows have a vector."""
        with self._lock:
            if matrix is None:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            matrix = matrix.astype("float32", copy=False)
            write_bytes_atomic(self.embeddings_path, matrix.tobytes())
            have = set(range(matrix.shape[0]))
            for i, entry in enumerate(index):
                entry["has_embedding"] = i in have
            self.save_index(index)

    # ── writes ─────────────────────────────────────────────────────────────

    def create(self, *, question: str, answer: str,
               sources: list[dict] | None = None,
               owner_name: str = "", owner_id: str | None = None,
               now: str | None = None) -> dict:
        """Add a NEW contribution. Never touches any existing record."""
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            raise ValueError("question and answer are required")
        ts = now or _now()
        oid = owner_id or owner_id_from(owner_name)
        record = {
            "schema_version": SCHEMA_VERSION,
            "knowledge_id": uuid.uuid4().hex,
            "content_hash": content_hash(question, answer),
            "question": question,
            "question_normalized": normalize_question(question),
            "answer": answer,
            "sources": trim_sources(sources),
            "owner_id": oid,
            "owner_name": (owner_name or "").strip(),
            "created_at": ts,
            "updated_at": ts,
            "version": 1,
            "lifecycle": _ACTIVE,
            "has_embedding": False,
            # Legacy aliases — kept so anything still reading the old shape
            # (or an unmigrated export) keeps working.
            "saved_at": ts,
            "saved_by": (owner_name or "").strip(),
        }
        with self._lock:
            self.ensure_dirs()
            self._atomic_write_json(self._record_path(record["knowledge_id"]), record)
            self.refresh_index()
        return record

    def update(self, knowledge_id: str, *, owner_id: str,
               question: str | None = None, answer: str | None = None,
               sources: list[dict] | None = None,
               now: str | None = None) -> dict:
        """Edit one record in place: SAME knowledge_id, version+1, updated_at set.

        Owner-scoped: a caller may only edit their own contribution, so an edit
        can never reach another user's identical record.
        """
        with self._lock:
            rec = self.get(knowledge_id)
            if rec is None:
                raise KeyError(f"unknown knowledge_id: {knowledge_id}")
            if rec.get("owner_id") != owner_id:
                raise PermissionError("not the owner of this contribution")
            if rec.get("lifecycle", _ACTIVE) != _ACTIVE:
                raise ValueError("cannot edit an archived contribution")
            new_q = (question or "").strip() or rec["question"]
            new_a = (answer or "").strip() or rec["answer"]
            changed = (new_q != rec["question"] or new_a != rec["answer"]
                       or (sources is not None))
            rec["question"] = new_q
            rec["answer"] = new_a
            rec["question_normalized"] = normalize_question(new_q)
            rec["content_hash"] = content_hash(new_q, new_a)
            if sources is not None:
                rec["sources"] = trim_sources(sources)
            rec["updated_at"] = now or _now()
            rec["version"] = int(rec.get("version", 1)) + (1 if changed else 0)
            if changed:
                # The question embedding no longer describes this record.
                rec["has_embedding"] = False
            self._atomic_write_json(self._record_path(knowledge_id), rec)
            self.refresh_index()
            return rec

    def archive(self, knowledge_id: str, *, owner_id: str,
                now: str | None = None) -> dict:
        """Soft-delete ONE contribution. Other owners' records are untouched."""
        with self._lock:
            rec = self.get(knowledge_id)
            if rec is None:
                raise KeyError(f"unknown knowledge_id: {knowledge_id}")
            if rec.get("owner_id") != owner_id:
                raise PermissionError("not the owner of this contribution")
            rec["lifecycle"] = _ARCHIVED
            rec["updated_at"] = now or _now()
            self._atomic_write_json(self._record_path(knowledge_id), rec)
            self.refresh_index()
            return rec

    # ── grouping ───────────────────────────────────────────────────────────

    @staticmethod
    def group_duplicates(records: list[dict]) -> list[dict]:
        """Group records sharing an exact ``content_hash`` into display cards.

        Underlying records stay separate — this is presentation only, so
        deleting one contributor never affects the others.
        """
        groups: dict[str, dict] = {}
        for rec in records:
            key = rec.get("content_hash") or rec["knowledge_id"]
            g = groups.get(key)
            if g is None:
                g = {
                    "content_hash": key,
                    "question": rec.get("question", ""),
                    "answer": rec.get("answer", ""),
                    "sources": rec.get("sources", []),
                    "contributors": [],
                    "knowledge_ids": [],
                    "created_at": rec.get("created_at", ""),
                    "updated_at": rec.get("updated_at", ""),
                }
                groups[key] = g
            g["knowledge_ids"].append(rec["knowledge_id"])
            name = rec.get("owner_name") or "unknown"
            if name not in g["contributors"]:
                g["contributors"].append(name)
            g["contributors"] = sorted(g["contributors"], key=str.lower)
            g["created_at"] = min(g["created_at"] or "", rec.get("created_at") or "")
            g["updated_at"] = max(g["updated_at"] or "", rec.get("updated_at") or "")
        out = list(groups.values())
        out.sort(key=lambda g: g["updated_at"] or "", reverse=True)
        return out

    # ── legacy read compatibility ──────────────────────────────────────────

    def read_legacy(self) -> list[dict]:
        """Read the OLD flat ``<slug>.json`` files, if any remain.

        Used for read compatibility during migration and as a fallback when
        ``records/`` is empty. Never writes, never deletes.
        """
        out = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or "question" not in data:
                continue
            data["_file"] = path.name
            out.append(data)
        return out

    def has_records(self) -> bool:
        """Any matchable knowledge at all — v2 records or legacy files."""
        if self.records_dir.exists() and any(self.records_dir.glob("*.json")):
            return True
        return bool(self.read_legacy())
