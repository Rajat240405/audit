"""Controlled, auditable reprocessing of already-ingested corpus documents.

Replaces the partial legacy extraction in ``corpus_reports.jsonl`` with enhanced
core text, **without changing record identity**.

Scope is selected by ``--org`` (default ``incois``, the original behaviour):
only corpus records whose ``metadata.org`` matches the selected organisation are
considered, so an MoES run cannot touch INCOIS records and vice versa. DfG and
parliamentary records carry different orgs and are never selected by any mode.
Each org has a profile that decides how its text is produced and how the result
is labelled; MoES is extracted through the *same* adapter the ingest path uses
(``src.scripts.convert_sirs_knowledge.enhanced_core_text``), so reprocessing and
ingestion cannot drift apart.

Why this is a separate tool rather than a bulk re-ingest:

* ``question_id`` is a content hash, so better text mints a new id and orphans
  the existing record from every index/filter/citation referencing it. This tool
  reads the id from the corpus and writes it back unchanged.
* A bulk corpus rewrite is not auditable. This tool is per-document, dry-run by
  default, backs up before writing, writes atomically, and logs every decision
  to JSONL — including the no-ops.
* Rollback is a file operation on the backup, not a re-extraction.

Usage::

    Dry-run is the DEFAULT: with no --apply nothing is written, no backup is
    made, and every document is still reported. --dry-run may be passed to say
    so explicitly.

    # INCOIS (default; identical to the original tool)
    python -m src.data.v2.reprocess --dry-run --json
    python -m src.data.v2.reprocess --apply --pdf path.pdf

    # MoES — dry-run first, always
    python -m src.data.v2.reprocess --org moes_hq --dry-run --json
    python -m src.data.v2.reprocess --org moes_hq --apply --pdf path.pdf

    python -m src.data.v2.reprocess --rollback <backup>       # restore backup

Safety rails: ``--apply`` is required to write; more than ``BULK_GUARD``
documents additionally requires ``--confirm-bulk``; and a target corpus that
does not exist is refused rather than created. Nothing here rebuilds an index —
that stays a separate, explicit operator step.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.v2.identity import INCOIS_ORG, MOES_ORG
from src.utils.atomic_io import write_bytes_atomic

#: Writing more than this many records at once needs an explicit override.
BULK_GUARD = 5


@dataclass(frozen=True)
class OrgProfile:
    """How one organisation's records are re-extracted and labelled.

    Adding an organisation is a one-line registry entry plus (if it does not use
    the plain V2 core path) an extraction branch — never a change to the safety
    rails, which are org-independent by construction.
    """

    #: Value matched against ``metadata.org`` (compared lowercased).
    slug: str
    #: Written to ``metadata.answer_source`` on success.
    answer_source: str
    #: Extract through the MoES adapter in ``convert_sirs_knowledge.py`` rather
    #: than calling the V2 pipeline directly. This is what keeps reprocessing
    #: and ingestion on one code path, including the MoES sidecar directory.
    via_adapter: bool = False
    #: ``metadata.document_type`` values that carry this org but are produced by
    #: a DIFFERENT dedicated pipeline, and so must never be reprocessed here.
    #:
    #: Demands-for-Grants is the live case: it is crawled under the MoES website
    #: source, so its records carry ``org="moes_hq"``, but ingestion routes it by
    #: ``extraction_profile`` to ``src/data/extract_dfg.py`` and stamps
    #: ``document_type="demands_for_grants"``. Running the V2 core engine over
    #: those records would silently replace dedicated structured output with
    #: generic core text — so the org match alone is not sufficient, and this
    #: exclusion is what makes it sufficient.
    exclude_document_types: frozenset = frozenset()


_ORG_PROFILES: dict[str, OrgProfile] = {
    INCOIS_ORG: OrgProfile(
        slug=INCOIS_ORG,
        answer_source="incois_v2_core",
        via_adapter=False,
    ),
    MOES_ORG: OrgProfile(
        slug=MOES_ORG,
        # Same value the ingest adapter stamps, so a reprocessed record is
        # indistinguishable from a freshly ingested one.
        answer_source="moes_v2_core",
        via_adapter=True,
        exclude_document_types=frozenset({"demands_for_grants"}),
    ),
}

#: The organisation considered when ``--org`` is omitted — the original
#: behaviour of this tool, preserved exactly.
DEFAULT_ORG = INCOIS_ORG


def org_profile(org: str) -> OrgProfile:
    """The profile for ``org``. Raises ``KeyError`` for an unknown org.

    Refusing an unknown org (rather than silently selecting zero records) turns
    a typo into an immediate, legible failure instead of a run that reports
    "nothing to do" and looks like success.
    """
    key = (org or "").strip().lower()
    if key not in _ORG_PROFILES:
        raise KeyError(
            f"unknown org {org!r}; known orgs: {', '.join(sorted(_ORG_PROFILES))}"
        )
    return _ORG_PROFILES[key]


def record_in_scope(record: dict, profile: OrgProfile) -> bool:
    """Is this corpus record one that ``profile`` is allowed to reprocess?

    Two conditions, both required: the record's org matches, and its
    ``document_type`` is not one the profile excludes. Parliamentary records are
    rejected by the first condition (their sources declare no org, so
    ``metadata.org`` is absent); Demands-for-Grants is rejected by the second.
    """
    metadata = record.get("metadata") or {}
    if (metadata.get("org") or "").lower() != profile.slug:
        return False
    return (metadata.get("document_type") or "") not in profile.exclude_document_types


@dataclass
class ReprocessOutcome:
    """One document's outcome. Always produced, even for no-ops."""

    pdf: str
    #: ``ok`` | ``unchanged`` | ``not-in-corpus`` | ``out-of-scope``
    #: | ``extraction-failed`` | ``dry-run`` | ``error``
    status: str
    record_id: str | None = None
    old_chars: int = 0
    new_chars: int = 0
    reason: str = ""
    detail: dict = field(default_factory=dict)
    #: Set only when ``resolve_source_pdf`` had to re-anchor the stored
    #: ``source_url`` onto this machine's data root — so the audit log records
    #: which file was actually read, not just which one was named.
    resolved_path: str | None = None

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
            "resolved_path": self.resolved_path,
        }


def load_corpus(path: Path | str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _source_name(source_url: str) -> str:
    """The filename of a stored ``source_url``, splitting on EITHER separator.

    ``Path.name`` only understands the *host's* separator, so on POSIX a corpus
    written on Windows (``E:\\audit3\\...\\01-24173-eng.pdf``) yields the whole
    string as its "name" and never matches anything. Records are located by
    filename, so that comparison has to be platform-independent in both
    directions — a corpus is routinely reprocessed from a different OS than the
    one that ingested it.
    """
    segs = _stored_segments(source_url)
    return segs[-1] if segs else ""


def find_record(records: list[dict], pdf_path: Path | str) -> dict | None:
    """Locate a corpus record by PDF filename (``metadata.source_url``)."""
    name = _source_name(str(pdf_path))
    for record in records:
        metadata = record.get("metadata") or {}
        source = metadata.get("source_url")
        if source and _source_name(str(source)) == name:
            return record
    return None


#: A leading drive specifier ("E:") in a stored Windows path. Split out so it is
#: not treated as a directory component when re-anchoring.
_DRIVE_RE = re.compile(r"^[A-Za-z]:$")


def _stored_segments(source_url: str) -> list[str]:
    """Split a stored ``source_url`` on either separator, dropping anchors.

    Both ``/`` and ``\\`` are split regardless of the host platform, because the
    value was written by whatever machine did the ingest — a corpus built in the
    HPC container holds ``/data/...`` while one built on a workstation holds
    ``E:\\audit3\\data\\...``, and either may be reprocessed from the other.
    """
    segs = [s for s in re.split(r"[\\/]+", str(source_url)) if s]
    return [s for s in segs if not _DRIVE_RE.match(s)]


def resolve_source_pdf(
    source_url: Path | str | None, *, data_root: Path | str | None = None
) -> Path | None:
    """Resolve a stored ``metadata.source_url`` to a file that exists **now**.

    ``source_url`` records wherever the crawl put the file *at ingest time*,
    which need not be reachable from the machine doing the reprocessing: a
    corpus ingested inside the HPC container stores ``/data/.moes-website/...``
    while the same staging tree on a workstation lives at
    ``E:\\audit3\\data\\.moes-website\\...``. Resolving the stored string
    verbatim therefore reports ``pdf-missing`` for documents that are plainly
    present. ``identity.build_identity_map`` already keys on the filename for
    exactly this reason; this applies the same reasoning to opening the file.

    Resolution order, first hit wins:

    1. the stored path verbatim — so any corpus that resolves today behaves
       identically, and an explicit ``--pdf`` argument is honoured as given;
    2. progressively shorter suffixes of the stored path re-anchored on the
       current data root, **longest first**, so the most specific existing
       match wins and a bare filename is never matched on its own.

    Returns ``None`` when nothing exists. Read-only: it never creates, moves or
    guesses a file, and it requires at least a parent directory plus the
    filename to exist, so it cannot silently pick an unrelated document.
    """
    if not source_url:
        return None
    stored = Path(str(source_url))
    if stored.is_file():
        return stored

    from src.utils.app_paths import data_dir

    root = Path(data_root) if data_root is not None else data_dir()
    segs = _stored_segments(source_url)
    # Stop at 2 segments: parent + filename. Anchoring on the filename alone
    # would be ambiguous across staging subtrees.
    for i in range(len(segs) - 1):
        candidate = root.joinpath(*segs[i:])
        if candidate.is_file():
            return candidate
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


def _extract_via_adapter(pdf_path: Path) -> tuple[str | None, str, dict]:
    """Extract through the MoES ingest adapter — one code path, not a second.

    Deliberately calls ``convert_sirs_knowledge.enhanced_core_text`` — the exact
    function the MoES ingest branch uses — instead of re-implementing extraction
    here, so reprocessing and ingestion cannot drift apart: same engine, same
    ``v2_sidecars_moes`` provenance root, same status strings.

    Note on identity: ``enhanced_core_text`` resolves its own ``record_id`` from
    the adapter's merged identity map (which reads the *default* corpus path),
    so a ``--corpus`` override does not redirect that sidecar naming. It cannot
    affect record identity here, because this tool writes the corpus's existing
    ``question_id`` back verbatim regardless of what the sidecar is called.
    """
    # Lazy import: convert_sirs_knowledge pulls in the ingestion stack, and this
    # module stays importable without it (mirrors the other lazy imports here).
    from src.data.v2.core_text import core_text_stats
    from src.scripts.convert_sirs_knowledge import (
        _moes_sidecar_dir,
        enhanced_core_text,
    )

    text, status, summary = enhanced_core_text(
        pdf_path, return_summary=True, sidecar_dir=_moes_sidecar_dir()
    )
    if text is None:
        # No legacy fallback here on purpose: reprocessing replaces text only
        # when the enhanced path succeeds. Falling back would silently swap
        # better text for worse and report it as an improvement.
        return None, status, {}

    stats = core_text_stats(text)
    summary = summary or {}
    stats["ocr_complete"] = bool(summary.get("ocr_complete", True))
    stats["ocr_limit_reason"] = summary.get("ocr_limit_reason")
    stats["ocr_pages_skipped"] = summary.get("ocr_pages_skipped")
    stats["adapter_status"] = status
    if stats["figure_lines"]:
        # Defensive: core text must never carry figure content.
        return None, "internal-error:figure-lines-in-core-text", stats
    return text, "ok", stats


#: Trailing marker added by the ingest adapter when OCR was capped. Matched so
#: re-running never stacks a second marker onto the same subject.
_INCOMPLETE_SUFFIX_RE = re.compile(r"\s+—\s+INCOMPLETE-OCR\(\d+p\)$")


def _incomplete_subject(subject: str, pages_skipped: int | None) -> str:
    """The subject, with exactly one ``INCOMPLETE-OCR`` marker as appropriate.

    Normalises rather than appends: a document that was capped on an earlier run
    and extracts completely now gets its marker cleared, and one that is still
    capped keeps a single, current marker. An incompletely-extracted document is
    never left labelled as though it were complete.
    """
    base = _INCOMPLETE_SUFFIX_RE.sub("", subject or "").rstrip()
    if pages_skipped:
        return f"{base} — INCOMPLETE-OCR({pages_skipped}p)"
    return base


def reprocess_pdf(
    pdf_path: Path | str,
    *,
    corpus_path: Path | str,
    dry_run: bool = True,
    log_path: Path | str | None = None,
    record_id: str | None = None,
    min_gain: int = 0,
    config=None,
    org: str = DEFAULT_ORG,
) -> ReprocessOutcome:
    """Reprocess one document. ``dry_run=True`` (default) never writes.

    ``org`` selects the extraction path and provenance labels. It does **not**
    widen the search: the corpus record is still located by PDF filename, so
    passing ``--pdf`` for a document belonging to another org is reported as
    ``not-in-corpus`` rather than silently rewriting it under the wrong
    organisation's labels.
    """
    pdf = Path(pdf_path)
    corpus = Path(corpus_path)
    # Display/identity name, split on either separator: a Windows source_url
    # read on POSIX has no host-native separator to split on.
    name = _source_name(str(pdf_path))
    # Validated up front so a bad --org fails before any file is touched.
    profile = org_profile(org)

    if not corpus.is_file():
        outcome = ReprocessOutcome(name, "error", reason="corpus-missing")
        _log(log_path, outcome, dry_run)
        return outcome

    # The stored source_url names the *ingest-time* location, which need not be
    # reachable from this machine (a corpus built in the HPC container stores
    # /data/... while the same tree here lives under APP_DATA_DIR). Resolve it
    # before concluding the document is absent — but only by finding the real
    # file, never by guessing one.
    resolved = resolve_source_pdf(pdf)
    if resolved is None:
        outcome = ReprocessOutcome(
            name, "error", reason="pdf-missing",
            detail={"stored_source_url": str(pdf)},
        )
        _log(log_path, outcome, dry_run)
        return outcome
    reanchored = resolved != pdf
    pdf = resolved

    if record_id is None:
        from src.data.v2.identity import build_identity_map

        mapping, _ = build_identity_map(corpus, org=profile.slug)
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

    if not record_in_scope(record, profile):
        # Present in the corpus, but not this mode's business — a DfG record
        # named under --org moes_hq, or any record whose org differs from the
        # selected one. Refuse explicitly instead of rewriting it under the
        # wrong profile's labels; the record is left byte-identical.
        meta = record.get("metadata") or {}
        outcome = ReprocessOutcome(
            pdf.name, "out-of-scope", record_id=record_id,
            old_chars=len(record.get("answer_text") or ""),
            reason=(
                f"org={meta.get('org')!r} "
                f"document_type={meta.get('document_type')!r} is not "
                f"reprocessable under --org {profile.slug}"
            ),
        )
        _log(log_path, outcome, dry_run)
        return outcome

    question_id = record.get("question_id")
    old_text = record.get("answer_text") or ""
    try:
        if profile.via_adapter:
            new_text, reason, stats = _extract_via_adapter(pdf)
        else:
            new_text, reason, stats = _extract_core_text(
                pdf, record_id or question_id, config
            )
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
    metadata["answer_source"] = profile.answer_source
    metadata["answer_text_source"] = f"enhanced:{reason}"
    metadata["core_text_stats"] = stats

    # Mirror the ingest adapter's honesty rule, and only for the orgs whose
    # ingest path has it: a capped OCR run must not leave the record looking
    # complete. INCOIS ingestion does not mark subjects, so neither does this —
    # keeping INCOIS output byte-identical to the original tool.
    if profile.via_adapter:
        skipped = (
            stats.get("ocr_pages_skipped")
            if not stats.get("ocr_complete", True)
            else None
        )
        existing = metadata.get("subject") or ""
        if skipped or _INCOMPLETE_SUFFIX_RE.search(existing):
            metadata["subject"] = _incomplete_subject(existing, skipped)

    outcome = ReprocessOutcome(
        pdf.name,
        "dry-run" if dry_run else "ok",
        record_id=record_id or question_id,
        old_chars=len(old_text),
        new_chars=len(new_text),
        reason=reason,
        detail=stats,
        # Recorded only when it differs from what the corpus named, so the log
        # shows which file was actually read after re-anchoring.
        resolved_path=str(pdf) if reanchored else None,
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


def discover_org_pdfs(corpus: Path | str, org: str = DEFAULT_ORG) -> list[Path]:
    """Every PDF filename the corpus claims for ``org``, in corpus order.

    Selection is ``record_in_scope``: an exact (case-insensitive) match on
    ``metadata.org`` **and** not one of the profile's excluded document types.
    Parliamentary records carry no ``metadata.org`` (their sources declare no
    org) and so are never selected by any mode; Demands-for-Grants records do
    carry ``org="moes_hq"`` — they are crawled by the MoES website source — and
    are excluded by document type so the MoES mode cannot reach them.
    """
    profile = org_profile(org)
    out: list[Path] = []
    for record in load_corpus(corpus):
        if not record_in_scope(record, profile):
            continue
        source = (record.get("metadata") or {}).get("source_url")
        if source:
            out.append(Path(str(source)))
    return out


def discover_incois_pdfs(corpus: Path | str) -> list[Path]:
    """INCOIS discovery — the original entry point, behaviour unchanged."""
    return discover_org_pdfs(corpus, INCOIS_ORG)


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
    parser.add_argument(
        "--org", default=DEFAULT_ORG,
        choices=sorted(_ORG_PROFILES),
        help=(
            "organisation whose records are considered "
            f"(default: {DEFAULT_ORG}, the original behaviour). Only corpus "
            "records whose metadata.org matches are selected, so one "
            "organisation's run can never rewrite another's records. "
            f"{MOES_ORG} extracts through the MoES ingest adapter."
        ),
    )
    parser.add_argument("--pdf", action="append", default=None,
                        help="PDF path (repeatable); omit to consider every "
                             "record belonging to --org")
    parser.add_argument("--apply", action="store_true",
                        help="actually write (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="explicit no-op: dry-run is already the default. "
                             "Accepted so the intent is stated on the command "
                             "line and in shell history; conflicts with --apply")
    parser.add_argument("--min-gain", type=int, default=0,
                        help="only write when the new text gains this many chars")
    parser.add_argument("--confirm-bulk", action="store_true",
                        help=f"required to apply more than {BULK_GUARD} documents")
    parser.add_argument("--log", default=None, help="JSONL audit log path")
    parser.add_argument("--rollback", default=None, metavar="BACKUP",
                        help="restore this backup over the corpus and exit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        print("--apply and --dry-run are contradictory; pass only one.",
              file=sys.stderr)
        return 2

    from src.data.v2.identity import default_corpus_path

    corpus = Path(args.corpus) if args.corpus else default_corpus_path()

    if args.rollback:
        rollback(corpus, args.rollback)
        print(f"rolled back {corpus} from {args.rollback}")
        return 0

    targets = (
        [Path(p) for p in args.pdf] if args.pdf
        else discover_org_pdfs(corpus, args.org)
    )
    if args.apply and len(targets) > BULK_GUARD and not args.confirm_bulk:
        print(
            f"refusing to apply to {len(targets)} documents without "
            f"--confirm-bulk (guard = {BULK_GUARD}). Use --pdf to target "
            f"individual documents, which is the intended workflow.",
            file=sys.stderr,
        )
        return 2
    if not targets:
        print(
            f"no corpus records with metadata.org={args.org!r} to consider "
            f"(corpus: {corpus})",
            file=sys.stderr,
        )
        return 0

    backup: Path | None = None
    if args.apply:
        backup = backup_corpus(corpus)
        print(f"backup: {backup}")

    outcomes = [
        reprocess_pdf(
            t, corpus_path=corpus, dry_run=not args.apply,
            log_path=args.log, min_gain=args.min_gain, org=args.org,
        )
        for t in targets
    ]

    report: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "org": args.org,
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
        print(f"mode: {report['mode']}  org: {report['org']}  "
              f"statuses: {report['by_status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "BULK_GUARD",
    "DEFAULT_ORG",
    "OrgProfile",
    "ReprocessOutcome",
    "backup_corpus",
    "discover_incois_pdfs",
    "discover_org_pdfs",
    "find_record",
    "load_corpus",
    "main",
    "org_profile",
    "record_in_scope",
    "reprocess_pdf",
    "resolve_source_pdf",
    "rollback",
    "write_corpus",
]
