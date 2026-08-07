"""
Checkpoint manager for the GraphRAG build.

The graph build may take many hours; if interrupted it must resume from where
it stopped instead of restarting from zero. Checkpoints are persisted to a
JSON file after every successfully processed document (atomic write via
temp-file + rename) and on shutdown.

Status values:
- ``done``    : document fully inserted + embedded + checkpointed
- ``failed``  : document failed extraction/insertion (retried on resume when
                ``retry_failed`` is enabled, up to ``max_attempts``)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GraphCheckpoint:
    """Resumable checkpoint store for the graph build."""

    def __init__(self, path: str | Path, retry_failed: bool = True, max_attempts: int = 3) -> None:
        self.path = Path(path)
        self.retry_failed = retry_failed
        self.max_attempts = max_attempts
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            self._data = raw.get("documents", {}) if isinstance(raw, dict) else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load checkpoint %s: %s — starting fresh", self.path, e)
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self._data,
            "meta": {
                "done": sum(1 for v in self._data.values() if v.get("status") == "done"),
                "failed": sum(1 for v in self._data.values() if v.get("status") == "failed"),
            },
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── public API ──────────────────────────────────────────────────────

    def is_done(self, doc_id: str) -> bool:
        return self._data.get(doc_id, {}).get("status") == "done"

    def should_retry(self, doc_id: str) -> bool:
        entry = self._data.get(doc_id)
        if entry is None:
            return True
        if entry.get("status") == "done":
            return False
        if not self.retry_failed:
            return False
        return int(entry.get("attempts", 0)) < self.max_attempts

    def mark_done(self, doc_id: str) -> None:
        self._data[doc_id] = {"status": "done", "attempts": 1}
        self._save()

    def mark_failed(self, doc_id: str, error: str) -> None:
        entry = self._data.get(doc_id, {"attempts": 0})
        entry["status"] = "failed"
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_error"] = str(error)[:500]
        self._data[doc_id] = entry
        self._save()

    def counts(self) -> dict[str, int]:
        done = sum(1 for v in self._data.values() if v.get("status") == "done")
        failed = sum(1 for v in self._data.values() if v.get("status") == "failed")
        return {"done": done, "failed": failed, "total_tracked": len(self._data)}

    @property
    def path_str(self) -> str:
        return str(self.path)
