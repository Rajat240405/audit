"""Bounded concurrency primitives shared by the document-processing paths.

Three things live here so that no pipeline invents its own policy:

* :func:`dots_semaphore` — the ONE process-wide limit on simultaneous DOTS
  inference requests. Every DOTS call acquires it, whichever pipeline started
  it, so tender + LS + RS cannot multiply into 4+4+4. The DOTS server runs
  with ``--max-num-seqs 4``; the application-side limit matches it.
* :func:`dots_executor` — the shared page worker pool. Pages are the unit of
  work, submitted to one global queue, so whichever worker frees up first
  picks up the next page — regardless of which document it belongs to.
* :func:`bounded_map` — ordered, bounded parallel map used by the LS and RS
  document/slot paths.

Determinism rule observed everywhere: workers may COMPLETE out of order, but
every result is reassembled by input index before anything is persisted.

Worker counts are operator-configurable by environment variable and default
to 4. Setting a count to 1 takes a genuinely sequential code path (no pool, no
futures), so "workers=1" reproduces the previous behaviour exactly.
"""

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# ── configuration ───────────────────────────────────────────────────────────

DEFAULT_DOTS_CONCURRENCY = 4
DEFAULT_LS_DOCUMENT_WORKERS = 4
DEFAULT_RS_DOCUMENT_WORKERS = 4


def env_workers(name: str, default: int) -> int:
    """Read a worker count from the environment, clamped to >= 1."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(1, val)


def dots_concurrency() -> int:
    """Max simultaneous DOTS inference requests, process-wide (DOTS_CONCURRENCY)."""
    return env_workers("DOTS_CONCURRENCY", DEFAULT_DOTS_CONCURRENCY)


def ls_document_workers() -> int:
    """Max simultaneous LS document/slot operations (LS_DOCUMENT_WORKERS)."""
    return env_workers("LS_DOCUMENT_WORKERS", DEFAULT_LS_DOCUMENT_WORKERS)


def rs_document_workers() -> int:
    """Max simultaneous RS document/slot operations (RS_DOCUMENT_WORKERS)."""
    return env_workers("RS_DOCUMENT_WORKERS", DEFAULT_RS_DOCUMENT_WORKERS)


# ── the single global DOTS gate ─────────────────────────────────────────────

_dots_sem: threading.BoundedSemaphore | None = None
_dots_sem_size: int | None = None
_dots_sem_lock = threading.Lock()

# Observability for benchmarking: peak simultaneous DOTS requests actually seen.
_dots_inflight = 0
_dots_peak = 0
_dots_stat_lock = threading.Lock()


def dots_semaphore() -> threading.BoundedSemaphore:
    """The process-wide DOTS limiter (created once, sized by DOTS_CONCURRENCY)."""
    global _dots_sem, _dots_sem_size
    want = dots_concurrency()
    with _dots_sem_lock:
        if _dots_sem is None or _dots_sem_size != want:
            _dots_sem = threading.BoundedSemaphore(want)
            _dots_sem_size = want
        return _dots_sem


class dots_slot:
    """Context manager holding one global DOTS slot for its whole body.

    Wraps the entire request including its retries — a retry is still an
    in-flight request as far as the server is concerned.
    """

    def __enter__(self) -> "dots_slot":
        global _dots_inflight, _dots_peak
        dots_semaphore().acquire()
        with _dots_stat_lock:
            _dots_inflight += 1
            _dots_peak = max(_dots_peak, _dots_inflight)
        return self

    def __exit__(self, *exc: object) -> bool:
        global _dots_inflight
        with _dots_stat_lock:
            _dots_inflight -= 1
        try:
            dots_semaphore().release()
        except ValueError:  # pragma: no cover - resized mid-flight
            pass
        return False


def dots_stats() -> dict[str, int]:
    """Current and peak in-flight DOTS requests (benchmark instrumentation)."""
    with _dots_stat_lock:
        return {"inflight": _dots_inflight, "peak": _dots_peak,
                "limit": dots_concurrency()}


def reset_dots_stats() -> None:
    global _dots_inflight, _dots_peak
    with _dots_stat_lock:
        _dots_inflight = 0
        _dots_peak = 0


# ── the shared page worker pool ─────────────────────────────────────────────

_dots_pool: ThreadPoolExecutor | None = None
_dots_pool_size: int | None = None
_dots_pool_lock = threading.Lock()


def dots_executor() -> ThreadPoolExecutor:
    """Shared pool backing the global DOTS page queue.

    One pool for the whole process: pages from any document land in the same
    queue, so a free worker takes the next page rather than idling until its
    own document finishes.
    """
    global _dots_pool, _dots_pool_size
    want = dots_concurrency()
    with _dots_pool_lock:
        if _dots_pool is None or _dots_pool_size != want:
            if _dots_pool is not None:
                _dots_pool.shutdown(wait=False)
            _dots_pool = ThreadPoolExecutor(
                max_workers=want, thread_name_prefix="dots-page")
            _dots_pool_size = want
        return _dots_pool


def shutdown_dots_executor() -> None:
    global _dots_pool, _dots_pool_size
    with _dots_pool_lock:
        if _dots_pool is not None:
            _dots_pool.shutdown(wait=True)
        _dots_pool = None
        _dots_pool_size = None


atexit.register(shutdown_dots_executor)


# ── ordered bounded map ─────────────────────────────────────────────────────


def bounded_map(fn: Callable[[T], R], items: Sequence[T], workers: int,
                *, thread_name_prefix: str = "work") -> list[R]:
    """Apply *fn* over *items* with at most *workers* in flight.

    Results are returned in INPUT order regardless of completion order. If
    several items raise, the exception from the LOWEST index is the one
    re-raised, so a failing run reports the same error every time.

    ``workers <= 1`` runs a plain loop — same call order, same exception
    timing, no threads created.
    """
    seq = list(items)
    if not seq:
        return []
    if workers <= 1 or len(seq) == 1:
        return [fn(item) for item in seq]

    results: list[R] = [None] * len(seq)  # type: ignore[list-item]
    errors: dict[int, BaseException] = {}
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=thread_name_prefix) as pool:
        futures: dict[Future, int] = {
            pool.submit(fn, item): idx for idx, item in enumerate(seq)
        }
        for fut, idx in futures.items():
            try:
                results[idx] = fut.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[idx] = exc
    if errors:
        raise errors[min(errors)]
    return results
