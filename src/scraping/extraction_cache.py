"""Content-addressed PDF-extraction cache — incremental nightly processing.

WHY THIS EXISTS
---------------
INCOIS folder sources already skip unchanged PDFs: ``ingest_folder`` gates on
``metadata.source_sha256`` + ``extractor_version`` before calling any
extractor. **LS and RS get none of that** — they are ``kind: records`` sources
(``config/sources.yaml``), so the folder gate never runs for them, and
``ls/extract.py::extract_qa`` / ``rs/pipeline.py::apply_answer_fallback`` call
``extract_pdf_text`` unconditionally on every nightly crawl. Every fetched LS/RS
PDF is therefore re-run through PicoDet -> DOTS each night even when neither the
document nor the extractor changed.

This module closes that gap at the two extraction chokepoints without
restructuring either pipeline.

KEY
---
``sha256(pdf_bytes)`` + ``extractor_version``. Content-addressed, so it is
immune to filename changes, re-downloads, mtime churn and path moves, and a
re-published PDF with identical bytes is correctly treated as unchanged.
Changing the extractor (DOTS on/off, PicoDet threshold, DPI, model) changes the
version and therefore the key — no stale text is ever reused across a pipeline
change.

SAFETY PROPERTIES
-----------------
* **Success is committed only after validation.** ``record_success`` refuses to
  store empty/whitespace output, so a failed or partial extraction can never be
  cached as a skippable success.
* **Failures are recorded but never skippable.** ``lookup`` returns a hit only
  for ``status == "ok"``; a failed document is retried on the next run, with
  its attempt count and last error preserved for diagnosis.
* **Atomic, append-only manifest.** Last entry per key wins. A crash mid-write
  cannot corrupt earlier entries.
* **Read-only on miss.** Nothing is mutated unless an extraction actually ran.
* **Never fatal.** Any cache error degrades to "extract normally" — the cache
  is an optimisation, never a correctness dependency.

CONTROLLED REPROCESSING
-----------------------
``EXTRACTION_FORCE_REPROCESS=1``  bypass lookups for one run (re-extract all).
``EXTRACTION_CACHE_ENABLED=0``    disable entirely (restores pre-cache behaviour).
Because the key embeds ``extractor_version``, an extractor upgrade *naturally*
invalidates only what it affects — there is no silent corpus-wide re-OCR.

LAYOUT
------
``<data>/extraction_cache/manifest.jsonl``   index, one JSON object per event
``<data>/extraction_cache/text/<sha>.txt``   extracted text, content-addressed
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST_NAME = "manifest.jsonl"
TEXT_DIRNAME = "text"
STATUS_OK = "ok"
STATUS_FAILED = "failed"


# ── configuration ───────────────────────────────────────────────────────────


def cache_enabled() -> bool:
    return (os.environ.get("EXTRACTION_CACHE_ENABLED", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def force_reprocess() -> bool:
    """EXTRACTION_FORCE_REPROCESS=1 — controlled, explicit re-extraction."""
    return (os.environ.get("EXTRACTION_FORCE_REPROCESS", "") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def cache_dir() -> Path:
    override = (os.environ.get("EXTRACTION_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    from src.utils.app_paths import data_dir

    return data_dir() / "extraction_cache"


def current_extractor_version() -> str:
    """Version of the extraction pipeline LS/RS actually use (the auto route)."""
    try:
        from src.data.pdf_table_extract import MODE_AUTO, current_extractor_version as _v

        return _v(MODE_AUTO) or "unknown"
    except Exception:  # noqa: BLE001 - never fail a crawl over provenance
        return "unknown"


# ── identity ────────────────────────────────────────────────────────────────


def content_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def cache_key(content_hash: str, extractor_version: str) -> str:
    """Stable key. Both components matter: same bytes + different extractor
    must be a MISS, otherwise a pipeline upgrade would silently reuse old text."""
    return hashlib.sha256(
        f"{content_hash}|{extractor_version}".encode("utf-8")
    ).hexdigest()


# ── entries ─────────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    key: str
    content_sha256: str
    extractor_version: str
    status: str
    doc_id: str | None = None
    source_url: str | None = None
    output_sha256: str | None = None
    output_chars: int = 0
    processed_at: str = ""
    attempts: int = 0
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {
            "key": self.key, "content_sha256": self.content_sha256,
            "extractor_version": self.extractor_version, "status": self.status,
            "doc_id": self.doc_id, "source_url": self.source_url,
            "output_sha256": self.output_sha256, "output_chars": self.output_chars,
            "processed_at": self.processed_at, "attempts": self.attempts,
            "last_error": self.last_error,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @staticmethod
    def from_json(d: dict) -> "CacheEntry":
        return CacheEntry(
            key=d.get("key", ""), content_sha256=d.get("content_sha256", ""),
            extractor_version=d.get("extractor_version", ""),
            status=d.get("status", STATUS_FAILED), doc_id=d.get("doc_id"),
            source_url=d.get("source_url"), output_sha256=d.get("output_sha256"),
            output_chars=int(d.get("output_chars") or 0),
            processed_at=d.get("processed_at", ""), attempts=int(d.get("attempts") or 0),
            last_error=d.get("last_error"), extra=d.get("extra") or {},
        )


# ── manifest I/O (append-only, last entry per key wins) ─────────────────────


_index: dict[str, CacheEntry] | None = None
_index_path: Path | None = None


def _manifest_path() -> Path:
    return cache_dir() / MANIFEST_NAME


def _load_index(force: bool = False) -> dict[str, CacheEntry]:
    global _index, _index_path
    path = _manifest_path()
    if _index is not None and _index_path == path and not force:
        return _index
    idx: dict[str, CacheEntry] = {}
    try:
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = CacheEntry.from_json(json.loads(line))
                    except Exception:  # noqa: BLE001 - tolerate a torn line
                        continue
                    if e.key:
                        idx[e.key] = e          # later entry supersedes earlier
    except OSError:
        idx = {}
    _index, _index_path = idx, path
    return idx


def reset_cache_state() -> None:
    """Drop the in-process index (tests, and after an external edit)."""
    global _index, _index_path
    _index, _index_path = None, None


def _append(entry: CacheEntry) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    idx = _load_index()
    idx[entry.key] = entry


def _text_path(content_hash: str, extractor_version: str) -> Path:
    return cache_dir() / TEXT_DIRNAME / f"{cache_key(content_hash, extractor_version)}.txt"


# ── public API ──────────────────────────────────────────────────────────────


def lookup(body: bytes, extractor_version: str | None = None) -> Optional[str]:
    """Cached extraction text for these exact bytes + extractor, else None.

    Returns a hit ONLY for a validated success. Failures, unknown keys, a
    disabled cache and ``EXTRACTION_FORCE_REPROCESS`` all return None so the
    caller extracts normally.
    """
    if not body or not cache_enabled() or force_reprocess():
        return None
    ver = extractor_version or current_extractor_version()
    ch = content_sha256(body)
    entry = _load_index().get(cache_key(ch, ver))
    if entry is None or entry.status != STATUS_OK:
        return None
    try:
        text = _text_path(ch, ver).read_text(encoding="utf-8")
    except OSError:
        return None                      # text file lost -> re-extract
    if not text.strip():
        return None                      # defensive: never serve empty as ok
    if entry.output_sha256 and hashlib.sha256(
            text.encode("utf-8")).hexdigest() != entry.output_sha256:
        return None                      # fingerprint mismatch -> re-extract
    return text


def record_success(body: bytes, text: str, *, extractor_version: str | None = None,
                   doc_id: str | None = None, source_url: str | None = None) -> bool:
    """Commit a VALIDATED extraction. Returns True when stored.

    Refuses empty/whitespace output, so a partial or failed run can never be
    cached as a skippable success.
    """
    if not cache_enabled() or not body:
        return False
    if not text or not text.strip():
        return False
    ver = extractor_version or current_extractor_version()
    ch = content_sha256(body)
    try:
        tp = _text_path(ch, ver)
        tp.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name per writer. A fixed ``<key>.tmp`` would be shared by
        # two processes extracting the SAME pdf concurrently (e.g. a manual
        # crawl overlapping cron — crawl_all's lock does not cover that), and
        # their interleaved writes would be os.replace'd into place together.
        tmp = tp.with_name(f"{tp.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, tp)          # atomic: text exists before the index entry
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)   # never leave a partial temp behind
        prev = _load_index().get(cache_key(ch, ver))
        _append(CacheEntry(
            key=cache_key(ch, ver), content_sha256=ch, extractor_version=ver,
            status=STATUS_OK, doc_id=doc_id, source_url=source_url,
            output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            output_chars=len(text),
            processed_at=datetime.now(timezone.utc).isoformat(),
            attempts=(prev.attempts + 1) if prev else 1,
        ))
        return True
    except OSError:
        return False                     # cache is an optimisation, never fatal


def record_failure(body: bytes, reason: str, *, extractor_version: str | None = None,
                   doc_id: str | None = None, source_url: str | None = None) -> None:
    """Record a failed attempt. NEVER makes the document skippable."""
    if not cache_enabled() or not body:
        return
    ver = extractor_version or current_extractor_version()
    ch = content_sha256(body)
    try:
        prev = _load_index().get(cache_key(ch, ver))
        _append(CacheEntry(
            key=cache_key(ch, ver), content_sha256=ch, extractor_version=ver,
            status=STATUS_FAILED, doc_id=doc_id, source_url=source_url,
            processed_at=datetime.now(timezone.utc).isoformat(),
            attempts=(prev.attempts + 1) if prev else 1,
            last_error=str(reason)[:300],
        ))
    except OSError:
        pass


def stats() -> dict[str, Any]:
    """Operator summary — read-only."""
    idx = _load_index(force=True)
    ok = sum(1 for e in idx.values() if e.status == STATUS_OK)
    versions: dict[str, int] = {}
    for e in idx.values():
        versions[e.extractor_version] = versions.get(e.extractor_version, 0) + 1
    return {
        "manifest": str(_manifest_path()),
        "entries": len(idx),
        "ok": ok,
        "failed": len(idx) - ok,
        "by_extractor_version": versions,
        "enabled": cache_enabled(),
        "force_reprocess": force_reprocess(),
    }
