"""In-process sliding-window rate limiting (Phase 2).

Scope note: the application is **single-process / single-Uvicorn-worker by
design** (see ``src/retrieval/frontend/server.py: start_server`` —
``ACTIVE_CONFIG`` and friends are process globals). An in-memory limiter is
therefore sufficient and honest for this deployment, and it is documented as
such rather than implied to be cluster-wide.

It is deliberately **not** the only defence: the durable limits that must
survive a restart (OTP expiry, attempt counts, resend cooldown) live in the
database and are enforced by ``src/auth/service.py``. This module only bounds
request *volume*.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque

from src.auth.errors import RateLimited

__all__ = ["RateLimiter", "get_limiter", "reset_limiter"]

_Bucket = deque  # of monotonically increasing timestamps


class RateLimiter:
    """Thread-safe sliding-window limiter keyed by an arbitrary string."""

    def __init__(self, clock=time.monotonic) -> None:
        self._hits: dict[str, _Bucket] = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock

    def check(self, key: str, *, limit: int, window_seconds: int) -> None:
        """Record a hit and raise :class:`RateLimited` when over the limit.

        The hit is recorded **before** the limit is evaluated, so a caller that
        catches the error and retries does not get extra attempts.
        """
        if limit <= 0:
            raise RateLimited("requests are disabled for this action", retry_after=window_seconds)
        now = self._clock()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            bucket.append(now)
            if len(bucket) > limit:
                retry_after = math.ceil(bucket[0] + window_seconds - now)
                raise RateLimited(
                    "too many attempts — please wait and try again",
                    retry_after=retry_after,
                )

    def hits(self, key: str) -> int:
        """Number of recorded hits currently inside the window (tests/ops)."""
        with self._lock:
            return len(self._hits.get(key, ()))

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter


def reset_limiter() -> None:
    """Test seam."""
    _limiter.clear()
