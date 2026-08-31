"""Structured append-only run history for crawlers and ingest jobs.

Records are written to ``data/run_history.jsonl`` — one JSON object per line,
appended after each completed crawl or ingest.  The file is machine-readable
so the orchestrator (``src/scripts/crawl_all.py``) can query ``ran_recently()``
without parsing free-form log text.

Schema (all fields present on every record; unknown future fields are ignored):

    {
      "ts":         "2026-08-27T06:00:00Z",  # UTC ISO-8601, completion time
      "source":     "lok_sabha",             # matches sources.yaml / schedule.yaml key
      "kind":       "crawl",                 # "crawl" | "ingest" | "crawl+ingest"
      "exit_code":  0,                       # subprocess exit code; null for in-process runs
      "duration_s": 142,                     # wall-clock seconds (float)
      "added":      12,                      # records added/downloaded
      "changed":    3,                       # records where content changed (0 if not tracked)
      "failures":   0,                       # failed slots / documents / download errors
      "ok":         true                     # false when exit_code not in the source's ok-set
    }

File location:
    ``<data_dir>/run_history.jsonl``
    Respects the ``APP_DATA_DIR`` env override — works identically on dev,
    HPC (bind-mount), and Docker (``APP_DATA_DIR=/data``).

Thread/process safety:
    Each ``append_run()`` call is a single atomic file-replace (via
    ``write_bytes_atomic``).  Concurrent callers on the same file are safe:
    the last writer wins on the atomic rename, so no entry is half-written.
    The orchestrator is single-threaded, so this is sufficient for Phase 1.
    For multi-process safety (e.g. parallel crawl jobs), file-level locking
    would be needed — not required here.

Design constraints:
    - No new dependencies: stdlib ``json``, ``datetime`` only.
    - Import path is ``src.scraping.run_log`` — consistent with the scraping
      package where per-crawler history belongs.
    - ``data_dir()`` is the single authoritative root, same as corpus_path().
    - Does NOT use sync.log (which is unstructured text, append-only, not
      machine-queryable).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.app_paths import data_dir
from src.utils.atomic_io import write_bytes_atomic

# Filename under data_dir().
RUN_HISTORY_FILENAME = "run_history.jsonl"

# Valid ``kind`` values.
KIND_CRAWL = "crawl"
KIND_INGEST = "ingest"
KIND_CRAWL_INGEST = "crawl+ingest"


def _history_path(data_root: Path | None) -> Path:
    """Return the run_history.jsonl path, using data_dir() unless overridden."""
    root = data_root if data_root is not None else data_dir()
    return root / RUN_HISTORY_FILENAME


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Read all JSON records from path; silently skip malformed lines."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            pass  # corrupt line — skip, never crash
    return records


def append_run(
    record: dict[str, Any],
    *,
    data_root: Path | None = None,
) -> None:
    """Append one run record to run_history.jsonl (atomic write).

    The caller is responsible for supplying all required fields.
    Use :func:`make_record` to build a well-formed record dict.

    Args:
        record: A dict conforming to the module-level schema.
        data_root: Override for the data directory root.  ``None`` uses
            ``data_dir()`` (i.e. ``APP_DATA_DIR`` env or project default).
    """
    path = _history_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise deterministically (sort_keys) for diff-stable files.
    line = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    # Byte-preserving append: read existing bytes, add new line, atomic replace.
    buf = bytearray()
    if path.exists():
        buf += path.read_bytes()
        if buf and not buf.endswith(b"\n"):
            buf += b"\n"
    buf += line.encode("utf-8") + b"\n"
    write_bytes_atomic(path, bytes(buf))


def make_record(
    *,
    source: str,
    kind: str,
    exit_code: int | None,
    duration_s: float,
    added: int = 0,
    changed: int = 0,
    failures: int = 0,
    ok: bool,
    ts: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a well-formed run-history record.

    Args:
        source:     Source key matching ``sources.yaml`` / ``schedule.yaml``
                    (e.g. ``"lok_sabha"``).
        kind:       One of ``KIND_CRAWL``, ``KIND_INGEST``,
                    ``KIND_CRAWL_INGEST``.
        exit_code:  Subprocess exit code, or ``None`` for in-process runs.
        duration_s: Wall-clock duration in seconds.
        added:      Records/files added this run.
        changed:    Records where content changed (0 when not tracked).
        failures:   Failed slots/documents/download errors.
        ok:         ``True`` when the run is considered successful.
        ts:         Override completion timestamp (ISO-8601 UTC).  ``None``
                    stamps ``utcnow``.
        extra:      Optional additional fields merged into the record.  Keys
                    must not shadow the standard schema fields.
    """
    rec: dict[str, Any] = {
        "ts": ts or _utcnow_iso(),
        "source": source,
        "kind": kind,
        "exit_code": exit_code,
        "duration_s": round(float(duration_s), 2),
        "added": int(added),
        "changed": int(changed),
        "failures": int(failures),
        "ok": bool(ok),
    }
    if extra:
        # Extra fields must not overwrite standard schema fields.
        for k, v in extra.items():
            if k not in rec:
                rec[k] = v
    return rec


def last_run(
    source: str,
    kind: str = KIND_CRAWL,
    *,
    data_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return the most-recent run record for ``(source, kind)``, or ``None``.

    Scans the entire history file from the end so the most-recent record is
    found without sorting.  For the expected file sizes (one line per weekly
    run per source, years of history = hundreds of lines) this is instant.

    Args:
        source:    Source key to filter on.
        kind:      Run kind to filter on.  Defaults to ``KIND_CRAWL``.
        data_root: Override for the data directory root.
    """
    records = _load_records(_history_path(data_root))
    # Walk in reverse to find the most-recent record first.
    for rec in reversed(records):
        if rec.get("source") == source and rec.get("kind") == kind:
            return rec
    return None


def ran_recently(
    source: str,
    interval_hours: float,
    *,
    kind: str = KIND_CRAWL,
    data_root: Path | None = None,
) -> bool:
    """Return ``True`` if a *successful* crawl for ``source`` completed within
    ``interval_hours`` hours of now.

    "Successful" means the record's ``ok`` field is ``True``.  A run with
    ``ok=False`` (partial failures, config error) does NOT count — the
    orchestrator must retry it.

    Args:
        source:         Source key to check.
        interval_hours: Minimum hours that must have elapsed since the last
                        successful run before another is allowed.  Pass ``0``
                        to always run.
        kind:           Run kind to filter on.  Defaults to ``KIND_CRAWL``.
        data_root:      Override for the data directory root.

    Returns:
        ``True``  → the source ran successfully recently; skip this run.
        ``False`` → the source has never run, or last run failed, or interval
                    has elapsed; proceed with the run.
    """
    if interval_hours <= 0:
        return False  # interval=0 means "always run"

    records = _load_records(_history_path(data_root))
    now = datetime.now(timezone.utc)

    # Walk from newest to oldest; stop at first matching ok record.
    for rec in reversed(records):
        if rec.get("source") != source or rec.get("kind") != kind:
            continue
        if not rec.get("ok", False):
            continue  # last run failed — must retry
        ts_str = rec.get("ts", "")
        if not ts_str:
            continue
        try:
            # Parse ISO-8601 UTC timestamp produced by _utcnow_iso().
            run_ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue  # malformed timestamp — treat as absent
        elapsed_hours = (now - run_ts).total_seconds() / 3600.0
        return elapsed_hours < interval_hours

    return False  # no successful run found
