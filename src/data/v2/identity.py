"""Corpus identity mapping: PDF filename → existing ``question_id``.

Enhanced extraction must **preserve** the identity of an already-ingested INCOIS
document. That is not automatic: ``convert_sirs_knowledge._make_record`` derives
``question_id = _hash_id(question + "|" + answer)`` — a content hash — so better
extraction would mint a new id and orphan the existing record from every index,
filter and citation that refers to it.

The mechanism to avoid that already exists in the codebase:

    convert_sirs_knowledge.py:184   question_id = qa_id or _hash_id(q + "|" + a)

and every INCOIS record already carries its source file:

    metadata.source_url = "data/incois_reports/AnnualReports/AR_2008_....pdf"

Verified on the real corpus: **73/73 INCOIS records have ``source_url`` set**, and
all 73 are ``Document: <stem>`` records. So the mapping is read directly from the
corpus — no legacy re-extraction, no guessing, no schema change.

This module never writes the corpus. It only reads it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Metadata ``org`` value that marks INCOIS records.
INCOIS_ORG = "incois"

#: Metadata ``org`` value that marks MoES (HQ website) records.
MOES_ORG = "moes_hq"


def default_corpus_path() -> Path:
    """The corpus the running service reads."""
    try:
        from src.utils.app_paths import corpus_path

        return Path(corpus_path())
    except Exception:  # noqa: BLE001 - fall back to the conventional location
        return Path("data/corpus_reports.jsonl")


def build_identity_map(
    corpus: Path | str | None = None,
    *,
    org: str = INCOIS_ORG,
) -> tuple[dict[str, str], list[dict]]:
    """Map ``PDF filename → existing question_id`` from the corpus.

    Returns ``(mapping, problems)``. ``problems`` records anything that would
    make an identity decision unsafe — a record with no ``source_url``, or two
    different ids claiming the same filename — so a reprocessing run can refuse
    rather than guess.

    The key is the **filename** (not the full path) because the stored
    ``source_url`` reflects wherever the crawl put the file at ingest time, which
    need not match the path being reprocessed now.
    """
    path = Path(corpus) if corpus is not None else default_corpus_path()
    mapping: dict[str, str] = {}
    problems: list[dict] = []
    seen: dict[str, str] = {}

    if not path.is_file():
        return {}, [{"kind": "corpus-missing", "path": str(path)}]

    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append({"kind": "bad-json", "line": lineno})
                continue

            metadata = record.get("metadata") or {}
            if (metadata.get("org") or "").lower() != org:
                continue

            question_id = record.get("question_id")
            source_url = metadata.get("source_url")
            if not source_url:
                problems.append(
                    {"kind": "no-source-url", "line": lineno, "question_id": question_id}
                )
                continue

            name = Path(str(source_url)).name
            previous = seen.get(name)
            if previous is not None and previous != question_id:
                problems.append(
                    {
                        "kind": "conflicting-ids",
                        "filename": name,
                        "question_ids": [previous, question_id],
                    }
                )
                continue

            seen[name] = question_id
            mapping[name] = question_id

    return mapping, problems


def question_id_for(pdf_path: Path | str, mapping: dict[str, str]) -> str | None:
    """The existing corpus id for a PDF, or ``None`` if it was never ingested.

    ``None`` is meaningful: it means this document is new to the corpus, so
    content-hash identity is correct for it. Callers must not substitute a
    fabricated id.
    """
    return mapping.get(Path(str(pdf_path)).name)


def save_identity_map(mapping: dict[str, str], dest: Path | str) -> Path:
    """Persist the map so a reprocessing run is reproducible and auditable."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_identity_map(src: Path | str) -> dict[str, str]:
    path = Path(src)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stamp_record_id(store: Any, record_id: str, extra: dict | None = None) -> dict:
    """Write ``record_id`` into a sidecar's ``doc.json``.

    This is the permanent sidecar ↔ corpus link. It is additive: existing keys
    are preserved, so re-stamping after a later reprocess cannot lose provenance.
    """
    payload: dict = {}
    if store.doc_path.is_file():
        try:
            payload = json.loads(store.doc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["record_id"] = record_id
    if extra:
        payload.update(extra)
    store.write_doc(payload)
    return payload


__all__ = [
    "INCOIS_ORG",
    "MOES_ORG",
    "build_identity_map",
    "default_corpus_path",
    "load_identity_map",
    "question_id_for",
    "save_identity_map",
    "stamp_record_id",
]
