#!/usr/bin/env python3
"""One-off removal of already-ingested non-English (Hindi) corpus rows.

Context
-------
The searchable corpus is English-only, but language was never a filter axis:
MoES/INCOIS publish every press release as ``<id>-eng.pdf`` AND ``<id>-hin.pdf``,
and ingestion converted both until the English-only policy landed
(``src/scripts/ingest_folder.DEFAULT_EXCLUDE_GLOBS``). Those rows are already in
``data/corpus_reports.jsonl`` and therefore already in the built index.

New files are handled by ingestion (skipped before conversion). THIS script
only removes what is already there. It does not touch the Hybrid index — a
full rebuild is required afterwards (the script prints the exact command).

What counts as a Hindi row
--------------------------
1. Filename evidence (default, deterministic, zero false positives on this
   corpus): the row's ``metadata.source_url`` or ``question_id`` matches one of
   the exclusion globs (``*-hin.*`` by default).
2. Content evidence (OPT-IN via ``--devanagari``): a few parliamentary rows
   carry Hindi ANSWER TEXT inside an English-named record (the English
   attachment was unavailable at scrape time). Those are detected by a
   conservative Devanagari ratio — at least ``--devanagari-ratio`` (default
   0.50) of the row's letters in U+0900–U+097F AND at least 50 Devanagari
   letters. Deliberately strict: it must never exclude a bilingual row.

Safety
------
* Dry-run by default: prints what WOULD be removed. ``--apply`` writes.
* With ``--apply`` the original file is copied to
  ``corpus_reports.jsonl.bak-hindi-<timestamp>`` BEFORE any change.
* Only lines identified as non-English are dropped; every other line is
  written back byte-for-byte (the original text is preserved verbatim, the
  file is never re-serialized from parsed objects).
* Never re-embeds anything — the removed rows simply stop existing, and the
  subsequent rebuild cannot embed a row that is no longer in the corpus.

Usage
-----
    python -m src.scripts.purge_hindi_rows                 # dry run
    python -m src.scripts.purge_hindi_rows --apply         # remove + back up
    python -m src.scripts.purge_hindi_rows --devanagari    # + content check
    python -m src.scripts.purge_hindi_rows --globs "*-hin.*,*-tam.*"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator

from src.utils.app_paths import corpus_path

# "*-hin" (no extension) covers document ids such as "01-28247-hin", the
# id form shown in the UI / used by some converters.
DEFAULT_GLOBS = ("*-hin.*", "*-hin")
# Minimum Devanagari letters before the ratio test is trusted at all.
MIN_DEVANAGARI_LETTERS = 50


def _devanagari_ratio(text: str) -> tuple[int, float]:
    """(devanagari letter count, share of all letters) for ``text``."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0, 0.0
    dev = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
    return dev, dev / len(letters)


def _matches_globs(value: str, globs: tuple[str, ...]) -> bool:
    low = (value or "").lower()
    return any(fnmatch.fnmatch(low, g.lower()) for g in globs)


def is_hindi_row(
    rec: dict,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
    devanagari: bool = False,
    devanagari_ratio: float = 0.5,
) -> bool:
    """True when ``rec`` is a non-English row under the active policy."""
    rid = str(rec.get("question_id") or "")
    url = str((rec.get("metadata") or {}).get("source_url") or "")
    if _matches_globs(rid, globs) or _matches_globs(url, globs):
        return True
    if devanagari:
        body = f"{rec.get('question_text') or ''}\n{rec.get('answer_text') or ''}"
        dev, ratio = _devanagari_ratio(body)
        if dev >= MIN_DEVANAGARI_LETTERS and ratio >= devanagari_ratio:
            return True
    return False


def _iter_rows(path: Path) -> Iterator[tuple[str, dict | None]]:
    """Yield (raw_line, parsed_or_None) for every non-empty corpus line."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                yield raw, json.loads(raw)
            except Exception:  # noqa: BLE001 - keep unparseable lines verbatim
                yield raw, None


def scan(
    path: Path,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
    devanagari: bool = False,
    devanagari_ratio: float = 0.5,
) -> tuple[list[str], int, int]:
    """Return (ids to remove, total rows, unparseable line count)."""
    remove: list[str] = []
    total = 0
    unparsed = 0
    for _raw, rec in _iter_rows(path):
        total += 1
        if rec is None:
            unparsed += 1
            continue
        if is_hindi_row(rec, globs, devanagari, devanagari_ratio):
            remove.append(str(rec.get("question_id") or "?"))
    return remove, total, unparsed


def purge(
    path: Path,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
    devanagari: bool = False,
    devanagari_ratio: float = 0.5,
) -> tuple[int, int, Path]:
    """Rewrite ``path`` without non-English rows. Returns (kept, removed, backup)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(path.name + f".bak-hindi-{stamp}")
    shutil.copy2(path, backup)

    kept = 0
    removed = 0
    tmp = path.with_name(path.name + f".tmp-{stamp}")
    with open(tmp, "w", encoding="utf-8") as out:
        for raw, rec in _iter_rows(path):
            if rec is not None and is_hindi_row(rec, globs, devanagari, devanagari_ratio):
                removed += 1
                continue
            out.write(raw + "\n")
            kept += 1
    tmp.replace(path)
    return kept, removed, backup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the corpus (default: dry run)")
    ap.add_argument("--globs", default=",".join(DEFAULT_GLOBS),
                    help="comma-separated filename globs (default '%(default)s')")
    ap.add_argument("--devanagari", action="store_true",
                    help="also flag rows whose text is majority-Devanagari")
    ap.add_argument("--devanagari-ratio", type=float, default=0.5,
                    help="Devanagari letter share required with --devanagari (default 0.5)")
    ap.add_argument("--corpus", default=None, help="override corpus path")
    args = ap.parse_args(argv)

    globs = tuple(g.strip() for g in args.globs.split(",") if g.strip())
    path = Path(args.corpus) if args.corpus else corpus_path()

    if not path.exists():
        print(f"corpus not found: {path}")
        return 1

    remove, total, unparsed = scan(path, globs, args.devanagari, args.devanagari_ratio)
    print(f"corpus           : {path}")
    print(f"rows total       : {total:,}")
    print(f"non-English rows : {len(remove):,}  (globs={globs}"
          f"{', devanagari>=' + str(args.devanagari_ratio) if args.devanagari else ''})")
    if unparsed:
        print(f"unparseable lines: {unparsed:,} (left untouched)")
    for rid in remove[:20]:
        print(f"   - {rid}")
    if len(remove) > 20:
        print(f"   … and {len(remove) - 20:,} more")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to remove them.")
        return 0
    if not remove:
        print("\nNothing to remove.")
        return 0

    kept, removed, backup = purge(path, globs, args.devanagari, args.devanagari_ratio)
    print(f"\nremoved {removed:,} row(s); kept {kept:,}")
    print(f"backup           : {backup}")
    print("\nNEXT STEP — the index still contains the removed rows; rebuild it:")
    print("    python -m src.scripts.ingest --rebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main())
