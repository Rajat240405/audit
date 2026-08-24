"""Per-session manifest load / diff / write (design §7).

The manifest is the incremental state's single source of truth per session
folder. Writes are atomic and happen ONLY when the semantic content changed
(volatile ``generated_at`` excluded), which is what makes a no-change re-run
byte-stable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.atomic_io import write_bytes_atomic

MANIFEST_NAME = "manifest.json"

#: keys that are allowed to differ without meaning "the crawl changed"
VOLATILE_KEYS: frozenset[str] = frozenset({"generated_at"})


def _semantic(obj: dict[str, Any]) -> str:
    src = {k: v for k, v in obj.items() if k not in VOLATILE_KEYS}
    return json.dumps(src, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_semantic(manifest).encode("utf-8")).hexdigest()


def manifests_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return _semantic(a) == _semantic(b)


def load_manifest(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt manifest is treated as absent;
        return None     # the next run rebuilds it from the corpus on disk
    return data if isinstance(data, dict) else None


def write_manifest(session_dir: Path, manifest: dict[str, Any]) -> Path:
    payload = (json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True) + "\n")
    dest = session_dir / MANIFEST_NAME
    write_bytes_atomic(dest, payload.encode("utf-8"))
    return dest
