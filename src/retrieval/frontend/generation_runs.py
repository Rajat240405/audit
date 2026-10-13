"""Server-owned generation runs.

A generation is a *run*, not a property of the HTTP response that started it.

  * The run's frames (SSE frames, exactly as the browser receives them) are
    produced by a worker thread that is NOT tied to any client connection.
    A closed tab, a reload, a dropped proxy connection or a second viewer
    cannot stop or restart the generation.
  * Every frame is kept in order. Any client can subscribe from frame index
    ``N``: it receives frames ``N..end`` already produced, then the live tail,
    until the run finishes. This is how a reloaded tab or a second tab
    reconnects without starting a second inference request.
  * Only ``cancel()`` (the explicit Stop button) cancels a run. Subscriber
    disconnects are never interpreted as cancellation.
  * ``status`` is the server's own truth: running, done, cancelled, failed or
    incomplete. Clients use it to decide what a run's end means.

Runs are kept in memory for ``retention_s`` after they finish. A server restart
forgets them; the client then reports the answer as lost, never as resumed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Run ids are chosen by the browser (so it can persist them before the first
# byte arrives) or generated here. Restricting the alphabet keeps them safe to
# embed in URLs and in logs.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

HEARTBEAT = ": ping\n\n"
_DONE_FRAME = 'data: {"type": "done"}\n\n'

RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"
INCOMPLETE = "incomplete"
TERMINAL = frozenset({DONE, CANCELLED, FAILED, INCOMPLETE})


def valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_RE.match(run_id or ""))


@dataclass
class GenerationRun:
    run_id: str
    frames: list[str] = field(default_factory=list)
    status: str = RUNNING
    cancel_requested: bool = False
    finished_at: float | None = None
    cond: threading.Condition = field(default_factory=threading.Condition)
    cancel_flag: threading.Event = field(default_factory=threading.Event)

    def status_dict(self) -> dict:
        with self.cond:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "frames": len(self.frames),
                "cancel_requested": self.cancel_requested,
            }


class RunStore:
    """Registry of runs. Thread-safe. One process = one registry."""

    def __init__(self, retention_s: float = 7200.0, max_runs: int = 256,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._runs: dict[str, GenerationRun] = {}
        self._lock = threading.Lock()
        self._retention_s = retention_s
        self._max_runs = max_runs
        self._clock = clock

    def get(self, run_id: str) -> GenerationRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def get_or_start(self, run_id: str,
                     make_frames: Callable[[], Iterator[str]]) -> tuple[GenerationRun, bool]:
        """Return ``(run, created)``. Starts the worker only when ``created``."""
        with self._lock:
            self._prune_locked()
            existing = self._runs.get(run_id)
            if existing is not None:
                return existing, False
            run = GenerationRun(run_id=run_id)
            self._runs[run_id] = run
        thread = threading.Thread(target=_work, args=(self, run, make_frames),
                                  name=f"gen-run-{run_id[:12]}", daemon=True)
        thread.start()
        return run, True

    def cancel(self, run_id: str) -> GenerationRun | None:
        run = self.get(run_id)
        if run is None:
            return None
        with run.cond:
            if run.status == RUNNING:
                run.cancel_requested = True
        run.cancel_flag.set()
        return run

    def prune(self) -> None:
        with self._lock:
            self._prune_locked()

    def _prune_locked(self) -> None:
        now = self._clock()
        for rid, run in list(self._runs.items()):
            if run.finished_at is not None and now - run.finished_at > self._retention_s:
                del self._runs[rid]
        # Bound memory: drop the oldest finished runs first.
        if len(self._runs) > self._max_runs:
            finished = sorted(
                (r for r in self._runs.values() if r.finished_at is not None),
                key=lambda r: r.finished_at or 0.0,
            )
            for run in finished[: len(self._runs) - self._max_runs]:
                self._runs.pop(run.run_id, None)

    def _now(self) -> float:
        return self._clock()


def _work(store: RunStore, run: GenerationRun, make_frames: Callable[[], Iterator[str]]) -> None:
    """Worker thread: produce frames for the run until it ends or is cancelled."""
    frames_iter = None
    outcome = INCOMPLETE
    saw_done = False
    try:
        frames_iter = make_frames()
        for frame in frames_iter:
            with run.cond:
                run.frames.append(frame)
                run.cond.notify_all()
            if frame == _DONE_FRAME:
                saw_done = True
            if run.cancel_flag.is_set():
                break
        if run.cancel_flag.is_set() and not saw_done:
            outcome = CANCELLED
        elif saw_done:
            outcome = DONE
        else:
            outcome = INCOMPLETE
    except Exception as exc:  # noqa: BLE001 - the run must always reach a terminal state
        # Full detail goes to the server log only. Clients get a fixed status.
        log.error("generation run %s failed: %s", run.run_id, type(exc).__name__)
        outcome = FAILED
    finally:
        if frames_iter is not None and hasattr(frames_iter, "close"):
            try:
                frames_iter.close()  # closes the upstream LLM stream on Stop
            except Exception:  # noqa: BLE001
                log.exception("closing generation run %s failed", run.run_id)
        with run.cond:
            run.status = outcome
            run.finished_at = store._now()
            run.cond.notify_all()


def wait_finished(run: GenerationRun, timeout: float = 10.0) -> None:
    """Block until the run reaches a terminal status (used by tests)."""
    deadline = time.monotonic() + timeout
    with run.cond:
        while run.status == RUNNING:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"run {run.run_id} still running")
            run.cond.wait(remaining)


def stream_frames(run: GenerationRun, start: int = 0,
                  heartbeat_s: float = 15.0) -> Iterator[str]:
    """Yield frames ``start..`` for one subscriber, then the live tail.

    Ends when the run has finished and every frame has been yielded. A
    subscriber that stops iterating (closed tab) affects nothing else.
    """
    idx = max(0, int(start))
    while True:
        with run.cond:
            if idx >= len(run.frames) and run.status == RUNNING:
                run.cond.wait(heartbeat_s)
            batch = run.frames[idx:]
            finished = run.status != RUNNING
        if batch:
            for frame in batch:
                yield frame
            idx += len(batch)
            continue
        if finished:
            return
        yield HEARTBEAT
