"""Document slot planning, resolution chains and byte-preserving download.

Per record two independent slots: ``eng`` and ``hin``. Link classes (from
REPORT.md §2/§6 — both eras live-validated 2026-08-25):

- **annex**   — ``sansad.in/getFile/lsapps/loksabhaquestions/{annex,qhindi}/
  <dir>/<file>?source=lsapps``. On non-200 the random-suffix variant
  (``AU2973_ciCkhd.pdf``) is retried once with the suffix stripped
  (``AU2973.pdf``); config ``policy.annex_suffix_retry``.
- **dspace-handle** — ``elibrary.sansad.in/handle/<prefix>/<suffix>``,
  resolved via :func:`src.scraping.dspace.resolve_handle` (REST ladder →
  handle-page HTML fallback; bitstream ``/content`` serves anonymously).
- **blank**   — empty/whitespace link: the upstream record was never
  published online. Slot class ``missing`` with cause
  ``document-not-published``; NO request is made and the slot is parked
  permanently (covers the 69 blank-link OCEAN DEVELOPMENT rows the operator
  declared unrecoverable; also the 573 sparse Hindi links).

Storage is content-keyed, not record-keyed: files land in
``<session_dir>/documents/`` as ``annex-<dir>-<stem>.<ext>`` /
``qhindi-<dir>-<stem>.<ext>`` / ``dspace-<prefix>-<suffix>-<lang>.<ext>``
(falls back to ``<record-id>-<lang>.<ext>`` for unrecognized links). One
fetch → one stored document → N record references (grouped-answer annexes),
via the per-session URL cache; on-disk sha-256 adoption keeps re-runs and
cross-record sharing byte-stable. Downloaded bytes stage in
``.staging/<key>.part`` and promote atomically.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scraping.dspace import handle_parts, resolve_handle
from src.scraping.formats import DocFacts, SlotResult, classify_document, format_extension
from src.scraping.http import CrawlHttpClient, HttpTransportError
from src.scraping.ls.discovery import RawLsQuestion
from src.utils.atomic_io import write_bytes_atomic

STAGING_DIRNAME = ".staging"
DOCUMENTS_DIRNAME = "documents"


def _mkdir_with_retry(path: Path, *, retries: int = 2, delay: float = 0.2) -> None:
    """``path.mkdir(parents=True, exist_ok=True)`` with a short retry/backoff.

    Google Drive File Stream on Windows virtualises its filesystem through a
    local agent that processes directory-creation requests asynchronously.
    This produces a reproducible two-error sequence when creating a new leaf
    directory whose parent was itself just created in the same run:

    * **WinError 3** (``ERROR_PATH_NOT_FOUND``) — ``CreateDirectoryW`` fires
      before the agent has flushed the parent's entry into the virtual FS
      table.  Python maps this to ``FileNotFoundError``, which
      ``Path.mkdir(parents=True)`` handles by recursing upward — but the
      grandparent may already be registered, giving:
    * **WinError 183** (``ERROR_ALREADY_EXISTS``) for the grandparent, which
      Python's ``mkdir`` suppresses *only* when ``is_dir()`` returns ``True``
      at that instant.  If the agent hasn't completed its flush yet,
      ``is_dir()`` can return ``False``, so the error propagates and kills
      the crawl mid-session.

    The fix: after any ``OSError`` from ``mkdir``, check whether the path is
    now a directory (the race may have resolved between the raise and this
    check).  If yes, we're done.  If not, sleep briefly and retry up to
    ``retries`` times before re-raising the original exception.  Two retries
    at 200 ms each (≤ 400 ms total) are enough for Drive's agent to flush
    in practice; the delay never fires on real filesystems because ``mkdir``
    succeeds on the first attempt.
    """
    exc: OSError | None = None
    for attempt in range(1 + retries):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError as e:
            exc = e
            if path.is_dir():
                # The race resolved between the raise and this check:
                # the directory exists now — treat as success.
                return
            if attempt < retries:
                time.sleep(delay)
    raise exc  # type: ignore[misc]  # retries >= 0 so exc is always set

#: random-suffix pattern on annex basenames: "AU2973_ciCkhd.pdf" -> "_ciCkhd"
SUFFIX_RE = re.compile(r"_[A-Za-z0-9]{4,8}(\.[A-Za-z0-9]{1,5})(\?.*)?$")

ANNEX_PATH_RE = re.compile(r"loksabhaquestions/(annex|qhindi)/([^?#]+)")

LinkKind = str  # "annex" | "dspace-handle" | "blank" | "other"


@dataclass
class Slot:
    lang: str               # "eng" | "hin"
    url: str | None
    kind: LinkKind


@dataclass
class SlotOutcome:
    """Process result for one slot (facts + payload + manifest extras)."""

    result: SlotResult
    body: bytes | None
    written: bool
    extra: dict[str, Any] = field(default_factory=dict)   # deterministic manifest hints


def classify_link(url: str | None) -> LinkKind:
    s = (url or "").strip()
    if not s:
        return "blank"
    if handle_parts(s):
        return "dspace-handle"
    if "sansad.in/getFile/" in s or "loksabhaquestions" in s:
        return "annex"
    return "other"


def plan_slots(q: RawLsQuestion) -> list[Slot]:
    return [
        Slot("eng", q.eng_url, classify_link(q.eng_url)),
        Slot("hin", q.hin_url, classify_link(q.hin_url)),
    ]


def _blank_facts() -> DocFacts:
    return DocFacts(
        "missing", "none", None, 0,
        cause="document-not-published",
        note="upstream record carries no document link (never published "
             "online; parked permanently — never retried or recovered)",
    )


def document_key(url: str, kind: LinkKind, lang: str, record_id: str) -> str:
    """Stable content key for the stored file name (minus extension)."""
    if kind == "annex":
        m = ANNEX_PATH_RE.search(url)
        if m:
            section, rest = m.group(1), m.group(2).strip("/")
            stem = rest.rsplit(".", 1)[0] if "." in rest.rsplit("/", 1)[-1] else rest
            return f"{section}-{stem.replace('/', '-')}"
    if kind == "dspace-handle":
        parts = handle_parts(url)
        if parts:
            prefix, suffix = parts[1].split("/", 1)
            return f"dspace-{prefix}-{suffix}-{lang}"
    return f"{record_id}-{lang}"


def strip_random_suffix(url: str) -> str | None:
    """'…/AU2973_ciCkhd.pdf?source=lsapps' → '…/AU2973.pdf?source=lsapps'.

    None when the basename carries no recognizable random suffix (the retry
    is attempted at most once, and only for annex links)."""
    m = ANNEX_PATH_RE.search(url)
    if not m:
        return None
    basename = m.group(2).rsplit("/", 1)[-1]
    stripped = SUFFIX_RE.search(basename)
    if not stripped:
        return None
    return url.replace(basename, SUFFIX_RE.sub(r"\1", basename), 1)


class UrlCache:
    """Per-session URL → (facts, body, rel_path) — one fetch, one stored
    document, N record references (grouped-answer annexes)."""

    def __init__(self) -> None:
        self._hits: dict[str, tuple[DocFacts, bytes | None, str | None, dict[str, Any]]] = {}

    def get(self, url: str) -> tuple[DocFacts, bytes | None, str | None, dict[str, Any]] | None:
        return self._hits.get(url)

    def put(self, url: str, facts: DocFacts, body: bytes | None,
            rel_path: str | None, extra: dict[str, Any]) -> None:
        self._hits[url] = (facts, body, rel_path, dict(extra))


def _fetch_annex(http: CrawlHttpClient, url: str, *,
                 suffix_retry: bool) -> tuple[bytes | None, int | None, dict[str, Any]]:
    """GET an annex URL, with the one-shot random-suffix retry. Returns
    (body, status, extra). Statuses are data (the host answers 500 for
    missing files); transport failures surface as (None, None, …)."""
    extra: dict[str, Any] = {}
    try:
        resp = http.get(url)
    except HttpTransportError:
        return None, None, extra
    if resp.status == 200:
        return resp.body, resp.status, extra
    if suffix_retry:
        stripped = strip_random_suffix(url)
        if stripped and stripped != url:
            extra["suffix_retried"] = True
            try:
                retry = http.get(stripped)
            except HttpTransportError:
                return None, None, extra
            extra["retry_url"] = stripped
            if retry.status == 200:
                return retry.body, retry.status, extra
            # keep the FIRST response's status as the cause evidence — the
            # random-suffix form is the canonical publication URL
    return resp.body, resp.status, extra


def _fetch_dspace(http: CrawlHttpClient, url: str
                  ) -> tuple[bytes | None, int | None, dict[str, Any]]:
    """Resolve a handle to its bitstream content URL and GET it."""
    extra: dict[str, Any] = {}
    resolved = resolve_handle(http, url)
    if resolved.method:
        extra["dspace_resolution"] = resolved.method
    if not resolved.content_url:
        extra["resolve_error"] = resolved.error
        return None, None, extra                    # resolution failed → broken
    extra["resolved_url"] = resolved.content_url
    try:
        resp = http.get(resolved.content_url)
    except HttpTransportError:
        return None, None, extra
    return resp.body, resp.status, extra


def process_slot(
    http: CrawlHttpClient,
    q: RawLsQuestion,
    slot: Slot,
    session_dir: Path,
    record_id: str,
    *,
    suffix_retry: bool = True,
    cache: UrlCache | None = None,
) -> SlotOutcome:
    """Resolve, download, classify and stage one slot.

    Mirrors the RS contract (stage .part → atomic promote; sha-256 identical
    destination never rewritten) with the LS additions: blank-link parking,
    annex suffix retry, DSpace handle resolution and URL-level sharing.
    """
    key = f"{record_id}-{slot.lang}"
    url = slot.url

    if slot.kind == "blank":
        return SlotOutcome(
            SlotResult(key=key, lang=slot.lang, url=url, facts=_blank_facts()),
            None, False,
        )

    # one fetch per URL per session — grouped-answer annexes resolve to the
    # same stored file for every referencing record
    cached = cache.get(url) if cache is not None and url else None
    if cached is not None:
        facts, body, rel_path, extra = cached
        return SlotOutcome(
            SlotResult(key=key, lang=slot.lang, url=url, facts=facts,
                       path=rel_path, extra=dict(extra)),
            body, False, dict(extra),
        )

    extra: dict[str, Any] = {}
    if slot.kind == "dspace-handle":
        body, status, extra = _fetch_dspace(http, url)
    elif slot.kind == "annex":
        body, status, extra = _fetch_annex(http, url, suffix_retry=suffix_retry)
    else:  # "other" — unrecognized shape; still GET it once (bytes decide)
        try:
            resp = http.get(url)
            body, status = resp.body, resp.status
        except HttpTransportError:
            body, status = None, None

    if extra.get("resolve_error"):
        # handle could not be resolved to any bitstream — no document fetch
        # ever happened; the resolution error IS the slot's cause
        facts = DocFacts(
            "broken", "unknown", None, 0,
            cause=str(extra["resolve_error"]),
            note="dspace handle resolved to no document bitstream "
                 "(REST ladder and HTML fallback both exhausted)",
        )
    elif status is None:
        facts = classify_document(None, http_status=None, transport_failed=True)
    else:
        facts = classify_document(body or b"", http_status=status)
    rel_path: str | None = None
    written = False

    if facts.doc_class in ("good", "partial"):
        key_base = document_key(url, slot.kind, slot.lang, record_id)
        ext = format_extension(facts.format)
        docs_dir = session_dir / DOCUMENTS_DIRNAME
        staging_dir = session_dir / STAGING_DIRNAME
        _mkdir_with_retry(staging_dir)
        staging = staging_dir / f"{key_base}.part"
        dest = docs_dir / f"{key_base}.{ext}"

        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == facts.sha256:
            rel_path = f"{DOCUMENTS_DIRNAME}/{dest.name}"   # adopt — never rewrite
            _cleanup_staging(staging)
        else:
            staging.write_bytes(body)
            _mkdir_with_retry(docs_dir)
            write_bytes_atomic(dest, staging.read_bytes())
            _cleanup_staging(staging)
            rel_path = f"{DOCUMENTS_DIRNAME}/{dest.name}"
            written = True

    if cache is not None and url:
        cache.put(url, facts, body, rel_path, extra)
    return SlotOutcome(
        SlotResult(key=key, lang=slot.lang, url=url, facts=facts,
                   path=rel_path, extra=dict(extra)),
        body, written, extra,
    )


def _cleanup_staging(staging: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        staging.unlink()
    # leaves the tree clean when the last .part is gone
    with contextlib.suppress(OSError):
        staging.parent.rmdir()
