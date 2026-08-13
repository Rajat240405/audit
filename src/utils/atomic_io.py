"""Smallest Windows-safe atomic file promote (P1.6).

Write to ``<name>.tmp`` in the same directory, fsync, then ``os.replace``
onto the destination. A crash mid-write leaves the previous file intact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def write_bytes_atomic(dest: str | Path, data: bytes) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)


def write_text_atomic(dest: str | Path, text: str, encoding: str = "utf-8") -> None:
    write_bytes_atomic(dest, text.encode(encoding))


def copy_file_atomic(src: str | Path, dest: str | Path) -> None:
    src = Path(src)
    write_bytes_atomic(dest, src.read_bytes())


def append_jsonl_atomic(corpus: str | Path, new_lines: Iterable[str]) -> int:
    """Rewrite JSONL as existing lines + new lines via a temp file, then replace.

    Returns the number of new lines written. Existing file is unchanged if
    the write fails before replace.
    """
    corpus = Path(corpus)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if corpus.exists():
        existing = corpus.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    added = 0
    parts = [existing] if existing else []
    for line in new_lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts.append(line + "\n")
        added += 1
    if added == 0:
        return 0
    write_text_atomic(corpus, "".join(parts))
    return added


def dump_json_atomic(dest: str | Path, obj: object, **kwargs) -> None:
    write_text_atomic(dest, json.dumps(obj, **kwargs))
