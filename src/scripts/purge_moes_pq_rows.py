"""One-time historical cleanup: remove MoES parliamentary-question rows.

WHAT THIS DOES
--------------
``data/corpus_reports.jsonl`` accumulated MoES *website* press releases titled
``PARLIAMENT QUESTION: …`` before a discovery-time gate existed. Those are
PIB-format reply summaries, not the authoritative parliamentary record; the
authoritative Lok Sabha / Rajya Sabha answers are separate rows that MUST stay.

This script removes the MoES PQ-derived rows and nothing else.

WHAT THIS DOES *NOT* DO
-----------------------
* It never touches Lok Sabha, Rajya Sabha or legacy ``parliamentary-qa`` rows.
  They are classified KEEP structurally, *before* any title rule is evaluated.
* It never touches INCOIS or unrelated MoES rows.
* It does not touch the retrieval index. Run the rebuild separately (§ below).
* It does not delete anything from the MoES staging tree — resurrection is
  prevented at ingestion by ``ingest._moes_website_dedup_excludes``.

CLASSIFICATION
--------------
A row is a *candidate* only if it is MoES-website-derived:
``metadata.source == "moes_website"`` OR ``.moes-website/`` appears in
``metadata.source_url``. (``metadata.source`` alone is not enough: the
``incdoc-`` id prefix is shared with INCOIS and every other folder-converted
source, so the prefix cannot be used as the key.)

Then:
  REMOVE — candidate AND tier-1 title match AND not LS/RS
  REVIEW — candidate AND (tier-2 title OR PQ-looking slug OR no usable title)
  KEEP   — everything else, unconditionally including all parliamentary rows

Safety and determinism
----------------------
* Dry run by default. ``--apply`` writes.
* Kept lines are copied through **verbatim** — the file is reassembled from the
  original line bytes, never re-serialised, so untouched rows stay
  byte-identical.
* Atomic write (temp file in the same directory → ``os.replace``).
* Timestamped backup taken before any write.
* Hard validation gate: refuses to apply if any protected-source count would
  change, or if ``removed != before - after``.
* Idempotent: a second run finds no candidates and rewrites nothing.

Usage
-----
    python -m src.scripts.purge_moes_pq_rows                 # dry run
    python -m src.scripts.purge_moes_pq_rows --apply         # remove + back up
    python -m src.scripts.purge_moes_pq_rows --corpus PATH --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.scraping.moes.pq_gate import (
    DETECTOR_VERSION,
    pq_slug_signal,
    pq_title_tier,
)
from src.utils.app_paths import corpus_path

MIGRATION_ID = "moes-pq-cleanup"
MIGRATION_VERSION = "1.0.0"

#: Sources that own genuine parliamentary records. KEEP, unconditionally.
PROTECTED_SOURCES = frozenset({"parliamentary-qa", "lok_sabha", "rajya_sabha",
                               "parliament", "sansad"})
#: Id shapes that are parliamentary by construction.
PROTECTED_ID_RE = re.compile(r"^(?:ls|rs)-\d", re.IGNORECASE)
LEGACY_PARL_ID_RE = re.compile(r"^\d+-\d+-\d+$")

REMOVE, REVIEW, KEEP = "REMOVE", "REVIEW", "KEEP"


# ─────────────────────────────────────────────────────────────────────────────
# classification
# ─────────────────────────────────────────────────────────────────────────────

def _meta(row: dict[str, Any]) -> dict[str, Any]:
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def _title_evidence(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """(title, field it came from) — or (None, None) when no usable title exists.

    ``metadata.subject`` holds the real title verbatim for rows converted with
    a staging ``record.json``; ``question_text`` carries the same title in the
    ``"Document: <title>"`` form. ``title_source == "filename-stem"`` means the
    "title" was only ever the file stem, which is not title evidence at all.
    """
    m = _meta(row)
    if m.get("title_source") == "filename-stem":
        return None, None
    val = str(m.get("subject") or "").strip()
    if val:
        return val, "metadata.subject"
    q = str(row.get("question_text") or "").strip()
    if q.startswith("Document: "):
        val = q[len("Document: "):].strip()
        if val:
            return val, "question_text"
    return None, None


def _slug_from_url(url: str) -> str:
    """Post slug segment of a staged MoES path, for slug corroboration."""
    m = re.search(r"\.moes-website/[^/]+/([^/]+)/documents/", url or "")
    return m.group(1) if m else ""


def is_moes_website_row(row: dict[str, Any]) -> bool:
    """MoES-website origin. Source field first, staged-path as the fallback."""
    m = _meta(row)
    if str(m.get("source") or "") == "moes_website":
        return True
    return ".moes-website/" in str(m.get("source_url") or "")


def is_parliamentary_row(row: dict[str, Any]) -> bool:
    """Genuine LS / RS / legacy parliamentary record. Structural, not textual."""
    m = _meta(row)
    if str(m.get("source") or "") in PROTECTED_SOURCES:
        return True
    if str(m.get("org") or "") == "sansad":
        return True
    qid = str(row.get("question_id") or "")
    return bool(PROTECTED_ID_RE.match(qid) or LEGACY_PARL_ID_RE.match(qid))


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one corpus row. Pure and deterministic.

    Order matters: the parliamentary check runs FIRST so no title heuristic can
    ever reach a genuine LS/RS record, whatever its title says.
    """
    if is_parliamentary_row(row):
        return {"state": KEEP, "rule": "protected-parliamentary-source",
                "tier": 0, "signal": "none", "matched_field": None,
                "matched_text": None, "confidence": "high"}

    if not is_moes_website_row(row):
        return {"state": KEEP, "rule": "not-moes-website-origin",
                "tier": 0, "signal": "none", "matched_field": None,
                "matched_text": None, "confidence": "high"}

    title, field = _title_evidence(row)
    tier = pq_title_tier(title)

    if tier == 1:
        return {"state": REMOVE, "rule": "moes-origin ∧ tier1-title",
                "tier": 1, "signal": "title-tier1", "matched_field": field,
                "matched_text": title, "confidence": "high"}

    url = str(_meta(row).get("source_url") or "")
    if tier == 2 or pq_slug_signal(url) or pq_slug_signal(_slug_from_url(url)):
        return {"state": REVIEW, "rule": "moes-origin ∧ ambiguous-signal",
                "tier": tier, "signal": "title-tier2" if tier == 2 else "slug",
                "matched_field": field, "matched_text": title,
                "confidence": "low"}

    if title is None:
        return {"state": REVIEW, "rule": "moes-origin ∧ no-usable-title",
                "tier": 0, "signal": "no-title", "matched_field": None,
                "matched_text": None, "confidence": "low"}

    return {"state": KEEP, "rule": "moes-origin ∧ no-pq-signal",
            "tier": 0, "signal": "none", "matched_field": field,
            "matched_text": title, "confidence": "high"}


# ─────────────────────────────────────────────────────────────────────────────
# scanning / writing
# ─────────────────────────────────────────────────────────────────────────────

def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        yield from enumerate(f, 1)


def scan(path: Path) -> dict[str, Any]:
    """Classify every row. Returns kept/removed/review line + evidence sets."""
    kept: list[str] = []
    removed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_source_before: dict[str, int] = {}
    unparsable: list[int] = []

    for n, raw in iter_lines(path):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:  # noqa: BLE001 — never rewrite a corpus we can't parse
            unparsable.append(n)
            continue
        src = str(_meta(row).get("source") or "<none>")
        by_source_before[src] = by_source_before.get(src, 0) + 1
        verdict = classify_row(row)
        counts[verdict["state"]] = counts.get(verdict["state"], 0) + 1
        title, _ = _title_evidence(row)
        entry = {
            "record_id": row.get("question_id"),
            "source": _meta(row).get("source"),
            "org": _meta(row).get("org"),
            "document_type": _meta(row).get("document_type"),
            "title": title,
            "source_url": _meta(row).get("source_url"),
            "staged_filename": Path(
                str(_meta(row).get("source_url") or "")).name or None,
            "detector": "moes_pq_title_tier1",
            "detector_version": DETECTOR_VERSION,
            "rule": verdict["rule"],
            "evidence": {
                "tier": verdict["tier"],
                "signal": verdict["signal"],
                "matched_field": verdict["matched_field"],
                "matched_text": verdict["matched_text"],
                "corpus_line": n,
            },
            "classification": verdict["state"],
            "confidence": verdict["confidence"],
            "content_hash": row.get("content_hash"),
            "line_sha256": hashlib.sha256(
                raw.rstrip("\n").encode("utf-8")).hexdigest(),
        }
        if verdict["state"] == REMOVE:
            removed.append(entry)
        elif verdict["state"] == REVIEW:
            review.append(entry)
        else:
            kept.append(raw if raw.endswith("\n") else raw + "\n")

    return {"kept": kept, "removed": removed, "review": review,
            "counts": counts, "by_source_before": by_source_before,
            "unparsable": unparsable}


def preflight(path: Path, result: dict[str, Any]) -> list[str]:
    """Blocking conditions. Any entry returned means: do not write."""
    problems: list[str] = []
    if not path.is_file():
        problems.append(f"corpus not found: {path}")
    if result["unparsable"]:
        problems.append(
            f"{len(result['unparsable'])} unparseable line(s) at "
            f"{result['unparsable'][:5]} — refusing to rewrite a corpus that "
            f"cannot be fully parsed")
    b = result["by_source_before"]
    if not any(k in b for k in PROTECTED_SOURCES) and b:
        problems.append(
            "no protected parliamentary source present — wrong corpus path?")
    return problems


def _source_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, raw in iter_lines(path):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        src = str(_meta(row).get("source") or "<none>")
        out[src] = out.get(src, 0) + 1
    return out


def validate_invariants(path: Path, result: dict[str, Any],
                        before_counts: dict[str, int]) -> list[str]:
    """Post-write invariants. Hard stop if any fail."""
    after = _source_counts(path)
    problems: list[str] = []
    for src in sorted(set(before_counts) | set(after)):
        if src == "moes_website":
            continue                       # the only source allowed to shrink
        if before_counts.get(src, 0) != after.get(src, 0):
            problems.append(
                f"protected source count changed: {src} "
                f"{before_counts.get(src, 0)} -> {after.get(src, 0)}")
    total_after = sum(after.values())
    expected = sum(before_counts.values()) - len(result["removed"])
    if total_after != expected:
        problems.append(
            f"row arithmetic mismatch: expected {expected}, found {total_after}")
    return problems


def purge(path: Path, result: dict[str, Any], run_id: str) -> Path:
    """Atomically rewrite the corpus without the REMOVE rows. Returns backup."""
    backup = path.with_name(f"{path.name}.bak-pq-{run_id}")
    shutil.copy2(path, backup)

    tmp = path.with_name(f".{path.name}.tmp-{run_id}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.writelines(result["kept"])        # verbatim — untouched rows unchanged
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)                   # atomic on the same filesystem
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)                   # persist the rename itself
        finally:
            os.close(dfd)
    except OSError:
        pass
    return backup


def write_manifests(out_dir: Path, run_id: str, result: dict[str, Any],
                    envelope: dict[str, Any], *, dry_run: bool) -> dict[str, Path]:
    prefix = "dry-run-" if dry_run else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "removed": out_dir / f"{prefix}removed-{run_id}.jsonl",
        "review": out_dir / f"{prefix}review-{run_id}.jsonl",
        "run": out_dir / f"{prefix}run-{run_id}.json",
    }
    paths["removed"].write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in result["removed"]),
        encoding="utf-8")
    paths["review"].write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in result["review"]),
        encoding="utf-8")
    paths["run"].write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, help="default: app_paths.corpus_path()")
    ap.add_argument("--out-dir", default=None,
                    help="manifest directory (default: data/migrations/pq_cleanup)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the corpus (default: dry run)")
    args = ap.parse_args(argv)

    from src.utils.app_paths import data_dir
    path = Path(args.corpus) if args.corpus else corpus_path()
    out_dir = Path(args.out_dir) if args.out_dir \
        else data_dir() / "migrations" / "pq_cleanup"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    if not path.is_file():
        print(f"corpus not found: {path}")
        return 2

    result = scan(path)
    before_counts = result["by_source_before"]

    problems = preflight(path, result)
    if problems:
        print("PREFLIGHT FAILED — nothing written:")
        for p in problems:
            print(f"  ✗ {p}")
        return 2

    envelope = {
        "migration_id": MIGRATION_ID, "migration_version": MIGRATION_VERSION,
        "detector_version": DETECTOR_VERSION, "run_id": run_id,
        "dry_run": not args.apply, "corpus_path": str(path),
        "records_before": sum(before_counts.values()),
        "removed": len(result["removed"]), "kept": len(result["kept"]),
        "review": len(result["review"]),
        "records_after": sum(before_counts.values()) - len(result["removed"]),
        "by_source_before": dict(sorted(before_counts.items())),
        "argv": list(argv or []), "started_at": run_id,
    }

    print(f"corpus              : {path}")
    print(f"records             : {envelope['records_before']}")
    print(f"  REMOVE (MoES PQ)  : {len(result['removed'])}")
    print(f"  REVIEW (ambiguous): {len(result['review'])}")
    print(f"  KEEP              : {len(result['kept'])}")
    print(f"by source (before)  : {envelope['by_source_before']}")

    paths = write_manifests(out_dir, run_id, result, envelope, dry_run=not args.apply)
    print(f"manifests           : {paths['removed'].parent}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to remove them.")
        return 0

    backup = purge(path, result, run_id)
    problems = validate_invariants(path, result, before_counts)
    if problems:
        print("\nVALIDATION FAILED — restoring backup:")
        for p in problems:
            print(f"  ✗ {p}")
        shutil.copy2(backup, path)
        print(f"  restored from {backup}")
        return 3

    envelope["corpus_sha256_after"] = hashlib.sha256(
        path.read_bytes()).hexdigest()
    envelope["backup_path"] = str(backup)
    envelope["by_source_after"] = dict(sorted(_source_counts(path).items()))
    envelope["finished_at"] = datetime.now(UTC).isoformat()
    paths["run"].write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\napplied. removed {len(result['removed'])} row(s).")
    print(f"backup              : {backup}")
    print(f"records after       : {envelope['records_after']}")
    print("NEXT: rebuild the index from the cleaned corpus, then verify:")
    print(f"  python -m src.retrieval.cli build --data {path} --rebuild "
          f"--output <index_dir>")
    print(f"  python -m src.scripts.verify_moes_pq_cleanup "
          f"--manifest {paths['removed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
