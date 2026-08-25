"""MoES document slots: attachment resolution, download, classification,
byte-preserving promote.

Attachment resolution contract (live-verified 2026-08-24, 247/247 in-scope
ids): the site's wp/v2/media listing omits every attached file, so ACF
``file:[id]`` rows are resolved via the site's own generic post fetcher
``GET /cms/wp-json/post-page/post?id=<id>`` whose ``acf_data`` inlines the
file objects:

- ``pdf``       → English file        (lang suffix ``-eng``)
- ``pdf_hindi`` → Hindi file          (``-hin``, per download_languages)
- ``pdf_both``  → single bilingual    (``-both``, always fetched)

Resolved posts of type ``revision`` are followed through their own first ACF
file row (bounded chain, cycle-safe) — a defensive path; the v1 census needed
zero hops. A chain ending without any pdf object is the MoES-specific failure
class ``attachment-missing-upstream`` (kept from the 30007-era defect); it is
NEVER cached, so failed resolutions self-heal on re-runs.

The ``.staging/`` + atomic-promote + hash-skip idiom below is an intentional
SMALL DUPLICATION of ``src/scraping/rs/documents.py::process_slot`` (approved
boundary review D5): the RS crawler is frozen — keep this as the attribution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scraping.formats import DocFacts, classify_document, format_extension
from src.scraping.http import HttpTransportError
from src.utils.atomic_io import write_bytes_atomic

STAGING_DIRNAME = ".staging"
DOCUMENTS_DIRNAME = "documents"

#: ACF inline object key -> our lang code. "both" = single bilingual file.
_PDF_FIELDS = {"pdf": "eng", "pdf_hindi": "hin", "pdf_both": "both"}
#: config download_languages -> slot langs they gate (both is never gated)
_LANG_GATES = {"english": "eng", "hindi": "hin"}

_RESOLVE_CHAIN_CAP = 5


@dataclass
class DocsOutcome:
    entries: list[dict[str, Any]] = field(default_factory=list)   # fetched/failed slots
    failed: list[dict[str, Any]] = field(default_factory=list)    # class=broken
    skipped_external: list[dict[str, Any]] = field(default_factory=list)
    attention: list[dict[str, Any]] = field(default_factory=list)  # empty-file-row
    bytes_changed: bool = False                                   # any dest write


def slot_key(record: dict[str, Any], row: int, lang: str | None = None) -> str:
    base = f"{record['wp_id']}-{row:02d}"
    return f"{base}-{lang}" if lang else base


def doc_basename(record: dict[str, Any], row: int, lang: str, ext: str) -> str:
    """Stable content-independent name: row index + wp id + lang. (Post slugs
    CAN be renamed upstream; embedding them in filenames would orphan bytes on
    a rename. The human-readable slug lives in the record directory name.)"""
    return f"{row + 1:02d}-{record['wp_id']}-{lang}.{ext}"


def _cleanup_staging(staging: Path) -> None:  # duplicated from rs/documents.py (D5)
    try:
        staging.unlink()
    except FileNotFoundError:
        pass
    try:
        staging.parent.rmdir()  # leaves tree clean when the last .part is gone
    except OSError:
        pass


def _promote(record_dir: Path, name: str, body: bytes, sha256: str) -> bool:
    """Byte-preserving promote (duplicated idiom from rs/documents.py, D5)."""
    docs_dir = record_dir / DOCUMENTS_DIRNAME
    staging_dir = record_dir / STAGING_DIRNAME
    dest = docs_dir / name
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        _cleanup_staging(staging_dir / f"{name}.part")
        return False
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir / f"{name}.part"
    staging.write_bytes(body)
    docs_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(dest, staging.read_bytes())
    _cleanup_staging(staging)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# attachment-id resolution (post-page/post?id= contract)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_links(posts_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """acf_data.pdf/pdf_hindi/pdf_both objects -> {lang: link-info}."""
    acf = posts_obj.get("acf_data") or {}
    links: dict[str, dict[str, Any]] = {}
    for field_, lang in _PDF_FIELDS.items():
        obj = acf.get(field_)
        if not isinstance(obj, dict):
            continue
        url = str(obj.get("url") or "")
        if not url.startswith("http"):
            continue
        links[lang] = {
            "url": url,
            "file_id": obj.get("id") or obj.get("ID"),
            "filename": obj.get("filename"),
            "filesize": obj.get("filesize"),
            "mime_type": obj.get("mime_type"),
        }
    return links


def _first_row_id(posts_obj: dict[str, Any]) -> int | None:
    rows = (posts_obj.get("acf_data") or {}).get("file") or []
    if rows and isinstance(rows[0], dict):
        fid = rows[0].get("file")
        if isinstance(fid, list) and fid:
            try:
                return int(fid[0])
            except (TypeError, ValueError):
                return None
    return None


def resolve_attachment(api, attachment_id: int, *, cap: int = _RESOLVE_CHAIN_CAP
                       ) -> tuple[dict[str, dict[str, Any]], str | None]:
    """attachment id -> pdf links, following revision chains (bounded).

    Returns (links, None) when resolvable, else ({}, reason) with reason in
    {empty-response, no-pdf-object, cycle, max-depth}. Transport/HTTP failures
    propagate as HttpApiError/HttpTransportError (the pipeline maps them to
    the slot-level cause ``attachment-resolve-failed``).
    """
    current = attachment_id
    seen: set[int] = set()
    for _ in range(cap):
        if current in seen:
            return {}, "cycle"
        seen.add(current)
        posts_obj = api.attachment_post(current)
        if not posts_obj:
            return {}, "empty-response"
        links = extract_pdf_links(posts_obj)
        if links:
            return links, None
        nxt = _first_row_id(posts_obj)
        if posts_obj.get("post_type") == "revision" and nxt:
            current = nxt
            continue
        return {}, "no-pdf-object"
    return {}, "max-depth"


# ─────────────────────────────────────────────────────────────────────────────
# slot processing
# ─────────────────────────────────────────────────────────────────────────────

def _entry(record: dict[str, Any], row: int, facts: DocFacts, *,
           lang: str | None, url: str | None, attachment_id: int | None,
           file_id: Any, path: str | None, duplicate_of: str | None,
           note_extra: str = "") -> dict[str, Any]:
    out = {
        "key": slot_key(record, row, lang),
        "record_id": record["id"],
        "row": row,
        "lang": lang,
        "attachment_id": attachment_id,
        "file_id": file_id,
        "url": url,
        "path": path,
        "duplicate_of": duplicate_of,
    }
    m = facts.as_manifest()
    if note_extra and m.get("note"):
        m["note"] = f"{m['note']}; {note_extra}"
    elif note_extra:
        m["note"] = note_extra
    out.update(m)
    return out


def process_record_documents(
    api,
    record: dict[str, Any],
    record_dir: Path,
    attachments: dict[int, dict[str, Any]],
    sha_seen: dict[str, str],
    *,
    languages: tuple[str, ...] = ("english", "hindi"),
) -> DocsOutcome:
    """Resolve/download/classify every file row of one record.

    ``attachments[aid]`` is the pipeline's runtime resolution view: either a
    links dict from the cache (``{"eng": {...}, "hin": {...}, "both": {...}}``)
    or a negative marker ``{"error": cause, "note": str}`` for ids that
    failed to resolve this run (never cached — retried next run).
    """
    wanted = {_LANG_GATES[l] for l in languages if l in _LANG_GATES}
    wanted.add("both")  # single bilingual files are never language-gated
    outcome = DocsOutcome()
    for frow in record.get("files") or []:
        row = int(frow["row"])
        attachment_id = frow.get("attachment_id")
        external_url = frow.get("external_url")

        if attachment_id is None and external_url:
            outcome.skipped_external.append({
                "record_id": record["id"], "row": row,
                "external_url": external_url, "type": frow.get("type"),
            })
            continue
        if attachment_id is None:
            outcome.attention.append({
                "type": "empty-file-row", "record_id": record["id"], "row": row,
                "note": "ACF file row carries neither attachment id nor external link",
            })
            continue

        aid = int(attachment_id)
        resolved = attachments.get(aid)
        if resolved is None or "error" in resolved:
            # unresolved this run (transient resolve errors are retried next run)
            if resolved is None:
                cause, note = "attachment-missing-upstream", \
                    "attachment id not resolved (no crawler resolution attempted)"
            else:
                cause, note = str(resolved["error"]), str(resolved.get("note") or "")
            facts = DocFacts("broken", "none", None, 0, cause=cause, note=note)
            entry = _entry(record, row, facts, lang=None, url=None,
                           attachment_id=aid, file_id=None, path=None,
                           duplicate_of=None)
            outcome.entries.append(entry)
            outcome.failed.append(entry)
            continue

        for lang in ("eng", "hin", "both"):
            link = resolved.get(lang)
            if not link or lang not in wanted:
                continue
            url = str(link["url"])
            try:
                resp = api.fetch_bytes(url)
                body, status = resp.body, resp.status
                facts = classify_document(body, http_status=status)
            except HttpTransportError:
                body, status = None, None
                facts = classify_document(None, http_status=None, transport_failed=True)

            path = None
            duplicate_of = None
            if facts.doc_class in ("good", "partial"):
                name = doc_basename(record, row, lang, format_extension(facts.format))
                if _promote(record_dir, name, body or b"", facts.sha256 or ""):
                    outcome.bytes_changed = True
                path = f"{DOCUMENTS_DIRNAME}/{name}"
                if facts.sha256:
                    my_key = slot_key(record, row, lang)
                    first = sha_seen.get(facts.sha256)
                    if first is None or first == my_key:
                        # first-ever sighting, or our own prior entry on a
                        # retry run — (re)claim, never self-flag as duplicate
                        sha_seen[facts.sha256] = my_key
                    else:
                        duplicate_of = first

            entry = _entry(record, row, facts, lang=lang, url=url,
                           attachment_id=aid, file_id=link.get("file_id"),
                           path=path, duplicate_of=duplicate_of)
            outcome.entries.append(entry)
            if facts.doc_class == "broken":
                outcome.failed.append(entry)
    return outcome
