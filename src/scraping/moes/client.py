"""MoES JSON-layer client (Akamai posture + 429 policy).

Two boundary-review decisions implemented here:

- **Zero-touch headers (D4):** the frozen ``CrawlHttpClient`` accepts an
  injected ``httpx.Client`` (``client=`` hook). MoES builds one carrying the
  browser-like header set (UA + Referer + apikey + Content-Type — plain UAs
  are 403-blocked by Akamai, verified 2026-08-24) instead of editing
  ``src/scraping/http.py``.
- **MoES policy layer (§1.8):** the frozen client has no 429 handling (HTTP
  statuses are data). ``MoesApi`` adds the crawl-level policy above it: a
  single 30 s backoff on 429, then the request fails as data
  (``HttpApiError``); categories/files translate that into abort/broken.

GET-only. Pagination is deterministic (page-ascending to exhaustion).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from src.scraping.http import CrawlHttpClient, HttpApiError, HttpResponse
from src.scraping.moes.config import (
    endpoints,
    http_headers,
    http_kwargs,
    listing_page_size,
)


def build_http(
    cfg: dict[str, Any],
    *,
    transport: httpx.BaseTransport | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> CrawlHttpClient:
    """Frozen polite client wired with the MoES header set via client injection."""
    kw = http_kwargs(cfg)
    inner = httpx.Client(
        timeout=kw["timeout"],
        follow_redirects=True,
        headers=http_headers(cfg),
        transport=transport,
    )
    return CrawlHttpClient(
        timeout=kw["timeout"],
        delay=kw["delay"],
        retries=kw["retries"],
        backoff=kw["backoff"],
        client=inner,          # injected — headers are ours; http.py untouched
        sleeper=sleeper,
    )


class MoesApi:
    """Typed access to the MoES public JSON layer (plus the 429 policy)."""

    def __init__(
        self,
        cfg: dict[str, Any],
        http: CrawlHttpClient,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.http = http
        self._sleeper = sleeper or time.sleep
        self._base = str(cfg.get("site", "https://www.moes.gov.in")).rstrip("/")
        self._ep = endpoints(cfg)
        self.listing_page_size = listing_page_size(cfg)
        self._tmr_backoff = http_kwargs(cfg)["too_many_requests_backoff"]

    # ── core request with 429 policy ──────────────────────────────────────────
    def _get(self, path_or_url: str) -> HttpResponse:
        url = path_or_url if path_or_url.startswith("http") else self._base + path_or_url
        resp = self.http.get(url)
        if resp.status == 429:
            self._sleeper(self._tmr_backoff)   # one backoff, then fail as data
            resp = self.http.get(url)
        return resp

    def _get_json(self, path: str) -> Any:
        resp = self._get(path)
        if resp.status != 200:
            raise HttpApiError(f"GET {resp.url} -> HTTP {resp.status}")
        try:
            return httpx.Response(200, content=resp.body).json()
        except Exception as exc:  # noqa: BLE001
            raise HttpApiError(f"GET {resp.url} -> invalid JSON: {exc}") from exc

    # ── endpoints ─────────────────────────────────────────────────────────────
    def counts(self) -> dict[str, int]:
        """count-posts/all, normalized to {post_type: n}."""
        raw = self._get_json(self._ep["counts"])
        items = raw if isinstance(raw, list) else []
        return {str(i.get("post_type")): int(i.get("posts", 0))
                for i in items if isinstance(i, dict) and i.get("post_type")}

    def taxonomy(self) -> list[dict[str, Any]]:
        raw = self._get_json(self._ep["taxonomy"])
        if isinstance(raw, dict):
            terms = raw.get("documents_category") or raw.get("terms") or []
        elif isinstance(raw, list):
            terms = raw
        else:
            terms = []
        return [t for t in terms if isinstance(t, dict)]

    def listing_page_url(self, category: str, page: int) -> str:
        return (f"{self._ep['listing']}?document_category={category}"
                f"&limit={self.listing_page_size}&page={page}")

    def listing_posts(self, category: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """All posts of one parent category, page-ascending to exhaustion.

        Returns (posts, stats{pages_seen, total_items}).
        """
        posts: list[dict[str, Any]] = []
        page = 0
        total_pages = None
        total_items = None
        while True:
            page += 1
            data = self._get_json(self.listing_page_url(category, page))
            if not isinstance(data, dict):
                raise HttpApiError(f"listing {category} page {page}: unexpected payload")
            batch = [p for p in (data.get("posts") or []) if isinstance(p, dict)]
            posts.extend(batch)
            total_items = data.get("total_items", total_items)
            total_pages = data.get("total_pages", total_pages)
            if total_pages is not None and page >= int(total_pages):
                break
            if not batch:  # safety: empty page without total_pages contract
                break
        stats = {"pages_seen": page, "total_items": total_items}
        return posts, stats

    def attachment_post(self, attachment_id: int) -> dict[str, Any]:
        """Resolve one ACF attachment id via the generic post fetcher.

        Live contract (2026-08-24): returns the resolved post object in
        ``posts`` — a ``central_documents`` post carrying inline
        ``acf_data.pdf`` / ``pdf_hindi`` / ``pdf_both`` objects with direct
        /static/uploads/ URLs, or a ``revision`` (caller may follow its own
        acf file rows), or an empty dict when the id does not resolve.
        """
        raw = self._get_json(f"{self._ep['attachment_post']}{int(attachment_id)}")
        if isinstance(raw, dict) and isinstance(raw.get("posts"), dict):
            return raw["posts"]
        if isinstance(raw, dict) and raw.get("posts") in (None, [], ""):
            return {}
        raise HttpApiError(f"attachment {attachment_id}: unexpected payload shape")

    def fetch_bytes(self, url: str) -> HttpResponse:
        """Document bytes (GET; status is data — classified downstream)."""
        return self._get(url)

    def robots(self) -> tuple[int, str]:
        resp = self._get(self._ep["robots"])
        return resp.status, resp.body.decode("utf-8", errors="replace")[:500]
