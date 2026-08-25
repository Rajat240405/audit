"""DSpace 7 (elibrary.sansad.in) bitstream resolution — shared house-crawler helper.

Behavior ported 1:1 from the proven production code in
``src/data/scraper.py`` (``RealArchiveScraper._pick_document_bitstream`` /
``_resolve_dspace_handle``), which stays byte-identical and untouched — the
legacy archive scraper and the RS backfill keep their own copies for now
(deliberate: zero risk to shipping paths; the copies are documented mirrors,
not hidden duplicates).

Mechanical differences only:

- HTTP goes through the crawler framework's :class:`CrawlHttpClient`
  (any-status GET; transport retries live inside the client), so outcomes are
  returned as a :class:`ResolveResult` instead of printed via rich console.
- The resolution METHOD is reported (``rest`` | ``html``) so manifests can
  record which ladder rung produced the document URL.

Live-frozen contracts (validated 2026-08-25, anonymous access):

- ``/server/api/pid/find`` / ``core/items`` / ``/metadata`` answer 401 or
  empty for anonymous clients → the REST ladder degrades gracefully.
- ``/server/api/core/bitstreams/<uuid>/content`` serves the real PDF
  anonymously (200).
- The handle HTML page (200) carries a ``citation_pdf_url`` meta tag and/or a
  ``bitstreams/<uuid>/download`` link — the working anonymous path.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from src.scraping.http import CrawlHttpClient, HttpTransportError

HANDLE_URL_RE = re.compile(r"(https?://[^/]+)/handle/(\d+/\d+)")
UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
CITATION_PDF_RE = re.compile(r'citation_pdf_url"\s+content="([^"]+)"', re.I)
BITSTREAM_DOWNLOAD_RE = re.compile(
    r"bitstreams/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/download"
)


@dataclass
class ResolveResult:
    """Outcome of resolving one DSpace handle URL to a document content URL."""

    content_url: str | None
    error: str | None               # None on success; machine-readable otherwise
    method: str | None = None       # "rest" | "html" — which rung produced the URL
    trail: list[dict[str, Any]] = field(default_factory=list)  # per-step evidence


def pick_document_bitstream(bitstreams: list[dict]) -> str | None:
    """Choose the primary document bitstream of a DSpace item.

    Preference ladder (verbatim from RealArchiveScraper): English PDF →
    English DOCX → Hindi PDF/DOCX; derived files (.txt, .jpg, …) ignored.
    Returns the bitstream UUID or None.
    """
    def score(bs: dict) -> tuple | None:
        name = (bs.get("name") or "").lower()
        if name.endswith(".pdf"):
            kind = 0
        elif name.endswith(".docx"):
            kind = 1
        else:
            return None  # derived file, not a document
        hindi = 1 if "hindi" in name else 0
        return (hindi, kind, name)

    scored = [
        (score(bs), bs.get("uuid"))
        for bs in bitstreams
        if score(bs) is not None and bs.get("uuid")
    ]
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored else None


def handle_parts(url: str) -> tuple[str, str] | None:
    """(base, "prefix/suffix") of a DSpace handle URL, or None."""
    m = HANDLE_URL_RE.match(url or "")
    return (m.group(1), m.group(2)) if m else None


def content_url(base: str, uuid: str) -> str:
    return f"{base}/server/api/core/bitstreams/{uuid}/content"


def _get(http: CrawlHttpClient, url: str, trail: list[dict[str, Any]]) -> tuple[int, bytes] | None:
    """One ladder step; None on transport failure (recorded, never raised)."""
    try:
        resp = http.get(url)
    except HttpTransportError as exc:
        trail.append({"url": url, "error": f"transport: {exc}"[:200]})
        return None
    trail.append({"url": url, "status": resp.status})
    return resp.status, resp.body


def _json(body: bytes) -> Any | None:
    import json

    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001 — corrupt JSON rung: degrade, never raise
        return None


def resolve_handle(http: CrawlHttpClient, url: str) -> ResolveResult:
    """Resolve a DSpace handle page to its document content URL.

    Ladder (both rungs anonymous-safe; verbatim strategy from the legacy code):
      1. REST: pid/find → item → bundles → bitstreams → pick → /content URL.
         Any non-200 (incl. the observed anonymous 401/empty) degrades to 2.
      2. HTML: handle page ``citation_pdf_url`` meta, else the
         ``bitstreams/<uuid>/download`` link.
    """
    parts = handle_parts(url)
    if parts is None:
        return ResolveResult(None, "invalid-handle-url")
    base, _handle = parts
    trail: list[dict[str, Any]] = []

    # ── rung 1: REST API ────────────────────────────────────────────────────
    # legacy parity: pid/find?id=<handle-url>; anonymous answers non-200/empty
    # today (validated 2026-08-25) — kept for parity + future-auth use.
    find_url = f"{base}/server/api/pid/find?id={urllib.parse.quote(url, safe='')}"
    step = _get(http, find_url, trail)
    if step is not None and step[0] == 200:
        data = _json(step[1])
        item_uuid = data.get("uuid") if isinstance(data, dict) else None
        if item_uuid:
            step = _get(http, f"{base}/server/api/core/items/{item_uuid}/bundles", trail)
            if step is not None and step[0] == 200:
                bundles = (_json(step[1]) or {}).get("_embedded", {}).get("bundles", [])
                for bundle in bundles:
                    buuid = bundle.get("uuid")
                    if not buuid:
                        continue
                    step = _get(
                        http,
                        f"{base}/server/api/core/bundles/{buuid}/bitstreams",
                        trail,
                    )
                    if step is None or step[0] != 200:
                        continue
                    bitstreams = (
                        (_json(step[1]) or {}).get("_embedded", {}).get("bitstreams", [])
                    )
                    uuid = pick_document_bitstream(bitstreams)
                    if uuid:
                        return ResolveResult(
                            content_url(base, uuid), None, method="rest", trail=trail
                        )

    # ── rung 2: handle-page HTML fallback (the proven anonymous path) ───────
    step = _get(http, url, trail)
    if step is not None and step[0] == 200:
        text = step[1].decode("utf-8", errors="ignore")
        m = CITATION_PDF_RE.search(text)
        if m:
            uuid = UUID_RE.search(m.group(1))
            if uuid:
                return ResolveResult(
                    content_url(base, uuid.group(1)), None, method="html", trail=trail
                )
        m = BITSTREAM_DOWNLOAD_RE.search(text)
        if m:
            return ResolveResult(
                content_url(base, m.group(1)), None, method="html", trail=trail
            )

    return ResolveResult(None, "no-dspace-bitstream", trail=trail)
