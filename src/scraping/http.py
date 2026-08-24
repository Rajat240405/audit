"""Polite HTTP client for crawlers (design §3 failure grammar).

Rules frozen from live validation (2026-08-23):

- GET-only (HEAD returns 403 on sansad.in — never used).
- HTTP status codes are DATA, not exceptions: the file host answers 500
  instead of 404 for missing documents. ``get()`` therefore returns the
  response for ANY status; only transport-level failures (DNS, connect,
  timeout) are retried (``retries`` attempts, linear backoff) and then
  raise :class:`HttpTransportError`.
- Politeness: a fixed ``delay`` after every request (config-driven;
  injectable clock for deterministic offline tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

DEFAULT_UA = "INCOIS-AuditPro-ParliamentaryQA/1.0 (+https://github.com/Rajat240405/audit)"


class HttpTransportError(RuntimeError):
    """Raised when all transport-level attempts for a request failed."""


class HttpApiError(RuntimeError):
    """Raised when a JSON API returns a non-200 status or invalid JSON."""


@dataclass
class HttpResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str | None:
        return self.headers.get("content-type")


class CrawlHttpClient:
    """GET-only polite client with injectable transport and clock."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        delay: float = 1.0,
        retries: int = 2,
        backoff: float = 1.0,
        user_agent: str = DEFAULT_UA,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.delay = delay
        self.retries = retries
        self.backoff = backoff
        self._sleeper = sleeper or time.sleep
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
            transport=transport,
        )
        self.request_count = 0

    def _pace(self) -> None:
        if self.delay > 0:
            self._sleeper(self.delay)

    def get(self, url: str) -> HttpResponse:
        """Return the response for any HTTP status; retry transport errors only."""
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.get(url)
                self.request_count += 1
                self._pace()
                return HttpResponse(
                    url=str(resp.url),
                    status=resp.status_code,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=resp.content,
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:  # noqa: PERF203
                last_exc = exc
                if attempt < self.retries:
                    self._sleeper(self.backoff * (attempt + 1))
        raise HttpTransportError(f"GET failed after {self.retries + 1} attempts: {url}: {last_exc}")

    def get_json(self, url: str) -> Any:
        """GET and parse JSON; raise HttpApiError on non-200 or invalid JSON."""
        resp = self.get(url)
        if resp.status != 200:
            raise HttpApiError(f"GET {url} -> HTTP {resp.status}")
        try:
            return httpx.Response(200, content=resp.body).json()
        except Exception as exc:  # noqa: BLE001
            raise HttpApiError(f"GET {url} -> invalid JSON: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CrawlHttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
