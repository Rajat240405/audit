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
    """Append JSONL lines via a temp file, then atomically replace.

    BYTE-PRESERVING append (Windows/CRLF fix): the existing file content is
    read as raw bytes and never re-encoded — so an existing corpus keeps its
    exact bytes (LF or CRLF endings, whatever tool produced them) and only
    the new LF-terminated lines are added. Previously this read in text mode,
    which normalized CRLF → LF in memory and rewrote the entire file on every
    append (an unnecessary full rewrite; it also broke byte-level prefix
    stability for a CRLF corpus). All corpus readers use universal-newline
    text mode / ``splitlines()``, so a mixed-ending file parses identically.

    Existing file is unchanged if the write fails before replace.
    Returns the number of new lines written.
    """
    corpus = Path(corpus)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    if corpus.exists():
        buf += corpus.read_bytes()  # exact bytes — no newline normalization
        if buf and not buf.endswith(b"\n"):
            buf += b"\n"
    added = 0
    for line in new_lines:
        line = line.rstrip("\n")
        if not line:
            continue
        buf += line.encode("utf-8") + b"\n"
        added += 1
    if added == 0:
        return 0
    write_bytes_atomic(corpus, bytes(buf))
    return added


def dump_json_atomic(dest: str | Path, obj: object, **kwargs) -> None:
    write_text_atomic(dest, json.dumps(obj, **kwargs))
