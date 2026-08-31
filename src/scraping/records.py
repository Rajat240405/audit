"""Deterministic JSONL record emission and merge-by-id (design §5/§8).

Byte-stability contract: identical logical rows always serialize to
identical bytes (sorted keys, ASCII-safe JSON, LF endings). Re-runs reuse
the previously stored row dicts for unchanged ids, so timestamps stamped at
first creation never churn. Canonical hashing excludes volatile keys so
hash comparison answers "did the CONTENT change", not "did we re-fetch it".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

#: keys stamped at write time that must not affect content hashing
VOLATILE_KEYS: frozenset[str] = frozenset({"scraped_at"})

SEPARATORS = (",", ":")


def canonical_json(row: dict[str, Any], *, drop_volatile: bool = False) -> str:
    src = row
    if drop_volatile:
        src = {k: v for k, v in row.items() if k not in VOLATILE_KEYS}
    return json.dumps(src, sort_keys=True, separators=SEPARATORS, ensure_ascii=True)


def row_sha256(row: dict[str, Any]) -> str:
    """Content hash of a row (volatile keys excluded)."""
    return hashlib.sha256(canonical_json(row, drop_volatile=True).encode("utf-8")).hexdigest()


def serialize_row(row: dict[str, Any]) -> str:
    """Deterministic one-line serialization for qa.jsonl."""
    return canonical_json(row, drop_volatile=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def merge_by_id(
    existing: Iterable[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], str],
    sort_key: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Overlay new rows onto existing by id; preserve existing dict objects
    for unchanged ids (byte-preserving), replace changed, append new.

    Returns (merged sorted rows, stats{added, changed, unchanged}).
    """
    stats = {"added": 0, "changed": 0, "unchanged": 0}
    by_id: dict[str, dict[str, Any]] = {key(r): r for r in existing}
    for row in new_rows:
        rid = key(row)
        old = by_id.get(rid)
        if old is None:
            by_id[rid] = row
            stats["added"] += 1
        elif row_sha256(old) == row_sha256(row):
            by_id[rid] = old  # keep original bytes (incl. scraped_at)
            stats["unchanged"] += 1
        else:
            by_id[rid] = row
            stats["changed"] += 1
    merged = sorted(by_id.values(), key=sort_key)
    return merged, stats


def serialize_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [serialize_row(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
