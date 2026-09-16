"""Controlled, auditable reprocessing of already-ingested INCOIS documents.

Replaces the partial legacy extraction in ``corpus_reports.jsonl`` with enhanced
core text, **without changing record identity**.

Why this is a separate tool rather than a bulk re-ingest:

* ``question_id`` is a content hash, so better text mints a new id and orphans
  the existing record from every index/filter/citation referencing it. This tool
  reads the id from the corpus and writes it back unchanged.
* A bulk corpus rewrite is not auditable. This tool is per-document, dry-run by
  default, backs up before writing, writes atomically, and logs every decision
  to JSONL — including the no-ops.
* Rollback is a file operation on the backup, not a re-extraction.

Usage::

    python -m src.data.v2.reprocess --dry-run                # report only
    python -m src.data.v2.reprocess --apply --pdf path.pdf   # one document
    python -m src.data.v2.reprocess --rollback <run-id>      # restore backup

Safety rails: ``--apply`` is required to write; more than ``BULK_GUARD``
documents additionally requires ``--confirm-bulk``; and a target corpus that
does not exist is refused rather than created.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.atomic_io import write_bytes_atomic

#: Writing more than this many records at once needs an explicit override.
BULK_GUARD = 5


@dataclass
class ReprocessOutcome:
    """One document's outcome. Always produced, even for no-ops."""

    pdf: str
    status: str  # ok | unchanged | not-in-corpus | extraction-failed | dry-run
    record_id: str | None = None
    old_chars: int = 0
    new_chars: int = 0
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def delta(self) -> int:
        return self.new_chars - self.old_chars

    @property
    def ratio(self) -> float:
        return round(self.new_chars / self.old_chars, 2) if self.old_chars else 0.0

    def as_dict(self) -> dict:
        return {
            "pdf": self.pdf,
            "status": self.status,
            "record_id": self.record_id,
            "old_chars": self.old_chars,
            "new_chars": self.new_chars,
            "delta_chars": self.delta,
            "ratio": self.ratio,
            "reason": self.reason,
            "detail": self.detail,
        }


def load_corpus(path: Path | str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def find_record(records: list[dict], pdf_path: Path | str) -> dict | None:
    """Locate a corpus record by PDF filename (``metadata.source_url``)."""
    name = Path(str(pdf_path)).name
    for record in records:
        metadata = record.get("metadata") or {}
        source = metadata.get("source_url")
        if source and Path(str(source)).name == name:
            return record
    return None


def _extract_core_text(
    pdf_path: Path, record_id: str | None, config=None
) -> tuple[str | None, str, dict]:
    from src.data.v2.config import load_v2_config
    from src.data.v2.core_text import core_text_from_result, core_text_stats
    from src.data.v2.pipeline import extract_document

    # The env-loaded config, not bare dataclass defaults: an operator's
    # INCOIS_ENHANCED_EXTRACTION / V2_OCR_* settings must reach this path.
    cfg = config if config is not None else load_v2_config()
    result = extract_document(pdf_path, cfg, write=True, record_id=record_id)
    if not result.extracted:
        return None, f"not-extracted({result.skipped_reason})", {}
    text = core_text_from_result(result)
    stats = core_text_stats(text)
    stats["ocr_complete"] = bool(result.summary.get("ocr_complete", True))
    stats["ocr_limit_reason"] = result.summary.get("ocr_limit_reason")
    stats["doc_id"] = result.doc_id
    if stats["figure_lines"]:
        # Defensive: core text must never carry figure content.
        return None, "internal-error:figure-lines-in-core-text", stats
    return text, "ok", stats


def reprocess_pdf(
    pdf_path: Path | str,
    *,
    corpus_path: Path | str,
    dry_run: bool = True,
    log_path: Path | str | None = None,
    record_id: str | None = None,
    min_gain: int = 0,
    config=None,
) -> ReprocessOutcome:
    """Reprocess one document. ``dry_run=True`` (default) never writes."""
    pdf = Path(pdf_path)
    corpus = Path(corpus_path)

    if not corpus.is_file():
        outcome = ReprocessOutcome(pdf.name, "error", reason="corpus-missing")
        _log(log_path, outcome, dry_run)
        return outcome
    if not pdf.is_file():
        outcome = ReprocessOutcome(pdf.name, "error", reason="pdf-missing")
        _log(log_path, outcome, dry_run)
        return outcome

    if record_id is None:
        from src.data.v2.identity import build_identity_map

        mapping, _ = build_identity_map(corpus)
        record_id = mapping.get(pdf.name)

    records = load_corpus(corpus)
    record = find_record(records, pdf)
    if record is None:
        # Not previously ingested: nothing to *re*process. Ingesting it is a
        # different, intentional operation (the normal crawler path).
        outcome = ReprocessOutcome(
            pdf.name, "not-in-corpus", record_id=record_id,
            reason="no corpus record with this source_url filename",
        )
        _log(log_path, outcome, dry_run)
        return outcome

    question_id = record.get("question_id")
    old_text = record.get("answer_text") or ""
    try:
        new_text, reason, stats = _extract_core_text(pdf, record_id or question_id, config)
    except Exception as e:  # noqa: BLE001 - one bad PDF must not abort a run
        outcome = ReprocessOutcome(
            pdf.name, "extraction-failed", record_id=record_id,
            old_chars=len(old_text), reason=f"{type(e).__name__}: {str(e)[:120]}",
        )
        _log(log_path, outcome, dry_run)
        return outcome

    if new_text is None:
        outcome = ReprocessOutcome(
            pdf.name, "extraction-failed", record_id=record_id,
            old_chars=len(old_text), reason=reason, detail=stats,
        )
        _log(log_path, outcome, dry_run)
        return outcome

    # Identity is preserved by construction: the existing question_id is written
    # back verbatim, so no index/filter/citation is invalidated.
    record["question_id"] = question_id
    record["answer_text"] = new_text
    metadata = record.setdefault("metadata", {})
    metadata["record_id"] = record_id or question_id
    metadata["answer_source"] = "incois_v2_core"
    metadata["answer_text_source"] = "enhanced:v2_core"
    metadata["core_text_stats"] = stats

    outcome = ReprocessOutcome(
        pdf.name,
        "dry-run" if dry_run else "ok",
        record_id=record_id or question_id,
        old_chars=len(old_text),
        new_chars=len(new_text),
        reason=reason,
        detail=stats,
    )

    if len(new_text.strip()) - len(old_text.strip()) < min_gain:
        outcome.status = "unchanged"
        outcome.reason = f"gain below min_gain={min_gain}"
        _log(log_path, outcome, dry_run)
        return outcome

    if dry_run:
        _log(log_path, outcome, dry_run)
        return outcome

    write_corpus(corpus, records)
    _log(log_path, outcome, dry_run)
    return outcome


def write_corpus(corpus: Path | str, records: list[dict]) -> None:
    """Atomically replace the corpus. Backup happens in ``backup_corpus``."""
    corpus = Path(corpus)
    payload = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    write_bytes_atomic(corpus, payload.encode("utf-8"))


def backup_corpus(corpus: Path | str, backup_dir: Path | str | None = None) -> Path:
    """Copy the corpus to a timestamped backup. Returns the backup path."""
    corpus = Path(corpus)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest_dir = Path(backup_dir) if backup_dir else corpus.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{corpus.stem}.{stamp}.jsonl"
    shutil.copy2(corpus, dest)
    return dest


def rollback(corpus: Path | str, backup: Path | str) -> None:
    """Restore a backup over the corpus (atomic)."""
    backup = Path(backup)
    if not backup.is_file():
        raise FileNotFoundError(f"backup not found: {backup}")
    write_bytes_atomic(Path(corpus), backup.read_bytes())


def discover_incois_pdfs(corpus: Path | str) -> list[Path]:
    """Every PDF filename the corpus claims for INCOIS, in corpus order."""
    out: list[Path] = []
    for record in load_corpus(corpus):
        metadata = record.get("metadata") or {}
        if (metadata.get("org") or "").lower() != "incois":
            continue
        source = metadata.get("source_url")
        if source:
            out.append(Path(str(source)))
    return out


def _log(log_path: Path | str | None, outcome: ReprocessOutcome, dry_run: bool) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": "dry-run" if dry_run else "apply"}
    entry.update(outcome.as_dict())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None, help="corpus_reports.jsonl path")
    parser.add_argument("--pdf", action="append", default=None,
                        help="PDF path (repeatable); omit to consider all INCOIS")
    parser.add_argument("--apply", action="store_true",
                        help="actually write (default is dry-run)")
    parser.add_argument("--min-gain", type=int, default=0,
                        help="only write when the new text gains this many chars")
    parser.add_argument("--confirm-bulk", action="store_true",
                        help=f"required to apply more than {BULK_GUARD} documents")
    parser.add_argument("--log", default=None, help="JSONL audit log path")
    parser.add_argument("--rollback", default=None, metavar="BACKUP",
                        help="restore this backup over the corpus and exit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from src.data.v2.identity import default_corpus_path

    corpus = Path(args.corpus) if args.corpus else default_corpus_path()

    if args.rollback:
        rollback(corpus, args.rollback)
        print(f"rolled back {corpus} from {args.rollback}")
        return 0

    targets = [Path(p) for p in args.pdf] if args.pdf else discover_incois_pdfs(corpus)
    if args.apply and len(targets) > BULK_GUARD and not args.confirm_bulk:
        print(
            f"refusing to apply to {len(targets)} documents without "
            f"--confirm-bulk (guard = {BULK_GUARD}). Use --pdf to target "
            f"individual documents, which is the intended workflow.",
            file=sys.stderr,
        )
        return 2

    backup: Path | None = None
    if args.apply:
        backup = backup_corpus(corpus)
        print(f"backup: {backup}")

    outcomes = [
        reprocess_pdf(
            t, corpus_path=corpus, dry_run=not args.apply,
            log_path=args.log, min_gain=args.min_gain,
        )
        for t in targets
    ]

    report: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "corpus": str(corpus),
        "backup": str(backup) if backup else None,
        "documents": len(outcomes),
        "by_status": {s: sum(1 for o in outcomes if o.status == s)
                      for s in sorted({o.status for o in outcomes})},
        "total_old_chars": sum(o.old_chars for o in outcomes),
        "total_new_chars": sum(o.new_chars for o in outcomes),
        "results": [o.as_dict() for o in outcomes],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        for o in outcomes:
            print(
                f"  {o.status:18s} {o.pdf[:52]:52s} "
                f"{o.old_chars:>8d} -> {o.new_chars:>8d} ({o.ratio}x) {o.reason}"
            )
        print(f"total: {report['total_old_chars']} -> {report['total_new_chars']} chars")
        print(f"mode: {report['mode']}  statuses: {report['by_status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "BULK_GUARD",
    "ReprocessOutcome",
    "backup_corpus",
    "discover_incois_pdfs",
    "find_record",
    "load_corpus",
    "main",
    "reprocess_pdf",
    "rollback",
    "write_corpus",
]
