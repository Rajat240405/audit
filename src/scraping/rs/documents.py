"""Document slot planning + byte-preserving download (design §5 step 4, §6).

Two slots per record: English (``files`` → sansad.in/getFile/annex/...) and
Hindi (``hindifiles`` → .../qhindi/...). The upstream record's own
``eng_file_name`` / ``hindi_file_name`` decides whether a slot exists at all
(31 MoES records carry an empty filename — classified ``missing`` with NO
request, exactly as validated).

Downloaded bytes go to ``.staging/<key>.part`` first and are promoted
atomically; an existing identical destination (sha-256 match) is never
rewritten, and incomplete ``.part`` files from a crashed run are simply
re-fetched. Sidecar names are ``<qslno>-<lang>.<sniffed-ext>`` because the
same filename is shared across languages in 732/866 records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.scraping.formats import SlotResult, classify_document, format_extension
from src.scraping.http import CrawlHttpClient, HttpTransportError
from src.utils.atomic_io import write_bytes_atomic

STAGING_DIRNAME = ".staging"
DOCUMENTS_DIRNAME = "documents"


@dataclass
class Slot:
    lang: str               # "eng" | "hin"
    filename: str | None    # None => upstream record has no file for this slot
    url: str | None


def plan_slots(raw: dict[str, Any]) -> list[Slot]:
    eng_name = (raw.get("eng_file_name") or "").strip() or None
    hin_name = (raw.get("hindi_file_name") or "").strip() or None
    return [
        Slot("eng", eng_name, raw.get("files") or None),
        Slot("hin", hin_name, raw.get("hindifiles") or None),
    ]


def process_slot(
    http: CrawlHttpClient,
    slot: Slot,
    session_dir: Path,
    qslno: int,
) -> tuple[SlotResult, bytes | None, bool]:
    """Download & classify one slot.

    Returns (SlotResult, body|None, dest_written). ``body`` is kept so the
    caller can run PDF text extraction without re-reading the promoted file.
    ``dest_written`` marks whether the destination file bytes changed on
    disk in THIS run (drives the session's updated/unchanged status).
    """
    key = f"{qslno}-{slot.lang}"
    if slot.filename is None:
        facts = classify_document(None, http_status=None, no_filename=True)
        return SlotResult(key=key, lang=slot.lang, url=slot.url, facts=facts), None, False

    body: bytes | None
    try:
        resp = http.get(slot.url)
        body, status = resp.body, resp.status
        facts = classify_document(body, http_status=status)
    except HttpTransportError:
        body, status = None, None
        facts = classify_document(None, http_status=None, transport_failed=True)
    result = SlotResult(key=key, lang=slot.lang, url=slot.url, facts=facts)

    if facts.doc_class not in ("good", "partial"):
        return result, body, False

    ext = format_extension(facts.format)
    docs_dir = session_dir / DOCUMENTS_DIRNAME
    staging_dir = session_dir / STAGING_DIRNAME
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir / f"{key}.part"
    dest = docs_dir / f"{key}.{ext}"

    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == facts.sha256:
        # identical bytes already on disk — never rewrite (byte-preserving)
        result.path = f"{DOCUMENTS_DIRNAME}/{dest.name}"
        _cleanup_staging(staging)
        return result, body, False

    staging.write_bytes(body)
    docs_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(dest, staging.read_bytes())
    _cleanup_staging(staging)
    result.path = f"{DOCUMENTS_DIRNAME}/{dest.name}"
    return result, body, True


def _cleanup_staging(staging: Path) -> None:
    try:
        staging.unlink()
    except FileNotFoundError:
        pass
    try:
        staging.parent.rmdir()  # leaves tree clean when the last .part is gone
    except OSError:
        pass
