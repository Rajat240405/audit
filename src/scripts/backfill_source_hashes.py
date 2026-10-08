"""Backfill ``source_sha256`` / ``extractor_version`` onto existing corpus rows.

Why this exists
---------------
``needs_reextraction()`` and the folder-ingestion source-hash gate both treat a
MISSING ``source_sha256`` as "unknown — must re-extract". With 0% of legacy
rows carrying the field, switching the gate on without a backfill would re-OCR
the entire corpus on the first run. This stamps the provenance that the gate
needs, derived from the files already on disk.

Guarantees (asserted by tests)
------------------------------
* **No DOTS / OCR / PicoDet.** The PDF is opened only to hash its bytes.
* **No text is touched.** ``question_text`` / ``answer_text`` are untouched, so
  ``qa_content_hash`` is unchanged and nothing is classified as changed.
* **No FAISS rebuild, no GraphRAG pending.** This writes the corpus file and
  nothing else — no index action is computed or scheduled.

Rows whose source file is missing are left alone (absent hash keeps the safe
"re-extract" default). Rows that already carry a hash are not overwritten.

Usage
-----
    python -m src.scripts.backfill_source_hashes --dry-run
    python -m src.scripts.backfill_source_hashes
    python -m src.scripts.backfill_source_hashes --legacy-version legacy/dpi200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_LEGACY_VERSION = "legacy/dpi200"


def _version_for_row(source_url: str, legacy_version: str,
                     dots_only_version: str | None,
                     row_meta: dict | None = None) -> str:
    """The extractor version describing how THIS row was actually produced.

    A blanket ``legacy/dpi200`` would be wrong twice over for a DOTS-only
    INCOIS row: it misrepresents the extraction path, and because the
    source-hash gate re-extracts on a version mismatch it would re-OCR every
    one of those documents on the first cron after migration.

    The family is decided by the SAME routing abstraction the extractor router
    uses (``incois_routing.uses_dots_only``) — never a folder list duplicated
    here, so a future manual folder is classified correctly with no edit.
    """
    try:
        from src.data.incois_routing import uses_dots_only

        if uses_dots_only(source_url):
            if dots_only_version:
                return dots_only_version
    except Exception:  # noqa: BLE001 - fall back to the legacy stamp
        pass

    # V2 engine: answer_source on the stored row is authoritative — it records
    # which engine actually produced the text. Official INCOIS/MoES rows were
    # extracted by V2, which imports neither DOTS nor PicoDet, so stamping
    # them legacy (or PicoDet) would force a needless re-extraction.
    meta = row_meta or {}
    if meta.get("answer_source") in ("incois_v2_core", "moes_v2_core"):
        try:
            from src.data.pdf_table_extract import (
                MODE_INCOIS_V2, current_extractor_version,
            )

            return current_extractor_version(MODE_INCOIS_V2)
        except Exception:  # noqa: BLE001
            pass
    return legacy_version


def backfill(corpus: Path, legacy_version: str = DEFAULT_LEGACY_VERSION,
             dry_run: bool = False, dots_only_version: str | None = None) -> dict:
    """Stamp provenance onto rows that lack it. Returns a stats dict.

    ``dots_only_version`` defaults to the current DOTS-only version string, so
    rows that were genuinely produced by the DOTS-only route are stamped with
    the version they already match and are NOT re-extracted afterwards.
    """
    from src.scripts.ingest_folder import source_sha256_of

    if dots_only_version is None:
        try:
            from src.data.pdf_table_extract import (
                MODE_DOTS_ONLY, current_extractor_version,
            )

            dots_only_version = current_extractor_version(MODE_DOTS_ONLY)
        except Exception:  # noqa: BLE001
            dots_only_version = None

    stats = {"rows": 0, "stamped": 0, "already": 0, "missing_file": 0,
             "no_source_url": 0, "unparsable": 0,
             "stamped_dots_only": 0, "stamped_legacy": 0, "stamped_v2": 0}
    if not corpus.exists():
        return stats

    sha_cache: dict[str, str | None] = {}
    out_lines: list[str] = []

    for line in corpus.open(encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        stats["rows"] += 1
        try:
            row = json.loads(s)
        except Exception:  # noqa: BLE001 — never drop a row we cannot parse
            stats["unparsable"] += 1
            out_lines.append(s)
            continue

        meta = row.get("metadata")
        if not isinstance(meta, dict):
            stats["no_source_url"] += 1
            out_lines.append(s)
            continue

        url = meta.get("source_url")
        if not url:
            stats["no_source_url"] += 1
            out_lines.append(s)
            continue

        if meta.get("source_sha256"):
            stats["already"] += 1
            out_lines.append(s)
            continue

        if url not in sha_cache:
            p = Path(url)
            sha_cache[url] = source_sha256_of(p) if p.is_file() else None
        sha = sha_cache[url]
        if sha is None:
            stats["missing_file"] += 1
            out_lines.append(s)
            continue

        meta["source_sha256"] = sha
        if not meta.get("extractor_version"):
            _ver = _version_for_row(url, legacy_version, dots_only_version, meta)
            meta["extractor_version"] = _ver
            if dots_only_version and _ver == dots_only_version:
                stats["stamped_dots_only"] += 1
            elif _ver.startswith("incois_v2@"):
                stats["stamped_v2"] += 1
            else:
                stats["stamped_legacy"] += 1
        stats["stamped"] += 1
        # Re-serialise the raw dict, NOT a re-validated model: this must not
        # recompute or normalise anything else on the row.
        out_lines.append(json.dumps(row, ensure_ascii=False))

    if stats["stamped"] and not dry_run:
        from src.utils.atomic_io import write_text_atomic

        write_text_atomic(corpus, "\n".join(out_lines) + "\n")

    return stats


def main() -> None:
    from src.utils.app_paths import corpus_path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None, help="corpus jsonl (default: app corpus)")
    ap.add_argument("--legacy-version", default=DEFAULT_LEGACY_VERSION,
                    help=f"extractor_version for legacy rows (default {DEFAULT_LEGACY_VERSION})")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else corpus_path()
    stats = backfill(corpus, args.legacy_version, args.dry_run)

    print(f"[backfill] corpus: {corpus}")
    print(f"  rows scanned          : {stats['rows']}")
    print(f"  stamped               : {stats['stamped']}")
    print(f"    as DOTS-only        : {stats['stamped_dots_only']}")
    print(f"    as INCOIS/MoES V2   : {stats['stamped_v2']}")
    print(f"    as legacy           : {stats['stamped_legacy']}")
    print(f"  already had a hash    : {stats['already']}")
    print(f"  source file missing   : {stats['missing_file']}")
    print(f"  no source_url         : {stats['no_source_url']}")
    if stats["unparsable"]:
        print(f"  unparsable (kept)     : {stats['unparsable']}")
    print("  NOTE: no DOTS/OCR ran, no index rebuild, no GraphRAG work.")
    if args.dry_run:
        print("  [dry-run] nothing written")


if __name__ == "__main__":
    main()
