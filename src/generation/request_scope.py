"""Request-scoped isolation for shared generation state (thread-local overlay).

WHY THIS EXISTS
---------------
The server owns ONE module-level ``llm_client`` and ONE ``generator``. Chat
handlers bind request-specific settings onto them before generating::

    _apply_execution_plan(...)   # llm_client.max_tokens / .think / .model ...
    generator.plan = plan
    generator.system_prompt = base + tone_hint

``/api/chat/stream`` and ``/api/chat`` are declared with ``def`` (not
``async def``), so FastAPI runs them in a THREADPOOL — several requests execute
those mutations concurrently against the same objects. A Deep request that
binds ``max_tokens=12288, think=True`` and then yields during retrieval can be
overwritten by a concurrent Fast request (``4096, think=False``) and will
generate with the wrong budget. The same applies to the
``generator.system_prompt`` tone suffix, which is saved/restored around the
request and can be restored to another request's value.

DESIGN (deliberately minimal — no architecture change)
------------------------------------------------------
``scoped_attr`` is a descriptor that keeps the existing attribute API exactly
as-is (``client.max_tokens`` reads/writes still work everywhere) but redirects
writes to a THREAD-LOCAL overlay while a request scope is active:

    inside ``with request_scope():``   → read/write the calling thread's value
    outside a scope                    → read/write the shared/global value

That distinction is important and intentional:

  * chat handlers wrap their work in ``request_scope()`` → fully isolated;
  * the ``/api/provider`` admin switch and boot-time configuration run OUTSIDE
    a scope → they keep mutating the shared defaults for every future request,
    preserving the existing global-switch contract.

Threads that never enter a scope (CLI, GraphRAG build, tests) see the plain
attribute behaviour they always had.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

__all__ = ["request_scope", "in_request_scope", "scoped_attr"]

_LOCAL = threading.local()


def _overrides() -> dict | None:
    """Per-thread override map, or None when no scope is active."""
    return getattr(_LOCAL, "overrides", None)


def in_request_scope() -> bool:
    return getattr(_LOCAL, "depth", 0) > 0


@contextmanager
def request_scope():
    """Isolate per-request mutations of scoped attributes to this thread.

    Re-entrant: nested scopes share the outermost overlay, so a handler that
    calls a helper which also opens a scope keeps one consistent view.
    """
    depth = getattr(_LOCAL, "depth", 0)
    if depth == 0:
        _LOCAL.overrides = {}
    _LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _LOCAL.depth = depth
        if depth == 0:
            # drop the overlay so the thread reverts to shared values and the
            # references held here cannot keep objects alive
            _LOCAL.overrides = None


class scoped_attr:
    """Instance attribute with a thread-local overlay while in a request scope.

    Reads fall back to the shared value when this thread has not set its own,
    so a scope only ever *adds* isolation — it never hides configuration that
    was applied globally at boot or by the provider switch.
    """

    __slots__ = ("_name", "_store")

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self._store = f"_shared_{name}" if name else ""

    def __set_name__(self, owner, name: str) -> None:
        if self._name is None:
            self._name = name
            self._store = f"_shared_{name}"

    def _key(self, obj) -> tuple:
        return (id(obj), self._name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        ov = _overrides()
        if ov is not None:
            key = self._key(obj)
            if key in ov:
                return ov[key]
        try:
            return obj.__dict__[self._store]
        except KeyError:
            raise AttributeError(self._name) from None

    def __set__(self, obj, value: Any) -> None:
        ov = _overrides()
        if ov is not None:
            ov[self._key(obj)] = value
        else:
            obj.__dict__[self._store] = value

    def __delete__(self, obj) -> None:
        ov = _overrides()
        if ov is not None:
            ov.pop(self._key(obj), None)
            return
        obj.__dict__.pop(self._store, None)
