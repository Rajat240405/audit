"""Lazy PostgreSQL engine and session management (Phase 1).

Design rules — both are load-bearing for this repository:

1. **No connection at import.** ``src/retrieval/frontend/server.py`` runs
   ``ensure_data_dirs()`` at import time, and ``tests/conftest.py`` installs an
   autouse fixture that imports it. An eager engine would therefore open sockets
   in every pytest process and in every CLI invocation (``python -m
   src.scripts.ingest``, the Alembic CLI, ...). The engine is created on first
   *use* and memoised.

2. **No automatic migration.** Nothing here runs ``alembic upgrade head``.
   Migrations are an explicit, observable operator step — see
   ``docs/database-setup.md``. This keeps ``APP_MODE=serve`` containers
   immutable by default and stops a bad migration from bricking a running
   serving container.

``create_engine()`` is itself lazy (it connects on first use), so even calling
:func:`get_engine` does not open a socket. The laziness here is about *when the
engine object is built*, which is what keeps import side effects out.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.config import DatabaseConfig, load_database_config

__all__ = [
    "get_db",
    "get_engine",
    "get_session_factory",
    "reset_engine",
    "session_scope",
]

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

# Reentrant, and never held across a call into get_engine(). Both matter:
# get_session_factory() needs the engine, and acquiring a plain Lock twice on
# the same thread would self-deadlock (reproduced during development).
_lock = threading.RLock()


def get_engine(config: DatabaseConfig | None = None) -> Engine:
    """Return the process-wide engine, creating it on first call.

    Args:
        config: use an already-built configuration. ``None`` (the normal case)
            loads it from the environment.

    Raises:
        DatabaseConfigError: if the environment is incomplete or invalid. This
            is deliberate: a misconfigured deployment fails here, loudly.
    """
    global _engine
    if _engine is None:
        with _lock:
            # Double-checked: another thread may have built it while we waited.
            if _engine is None:
                cfg = config if config is not None else load_database_config()
                cfg.validate()
                _engine = create_engine(
                    cfg.url,
                    echo=cfg.echo,
                    pool_size=cfg.pool_size,
                    max_overflow=cfg.max_overflow,
                    # Pre-ping matters on HPC: the database runs in a SEPARATE
                    # container/SIF and restarts independently of the app, so
                    # pooled connections go stale without this.
                    pool_pre_ping=cfg.pool_pre_ping,
                    # Fail fast rather than hanging when the DB is unreachable.
                    connect_args={"connect_timeout": cfg.connect_timeout_seconds},
                )
    return _engine


def get_session_factory(config: DatabaseConfig | None = None) -> sessionmaker[Session]:
    """Return the memoised ``sessionmaker`` bound to :func:`get_engine`.

    The engine is resolved **before** :data:`_lock` is taken, so the two locks
    are never nested.
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine(config)
        with _lock:
            if _session_factory is None:
                _session_factory = sessionmaker(
                    bind=engine,
                    autoflush=False,
                    expire_on_commit=False,
                )
    return _session_factory


def reset_engine() -> None:
    """Dispose the engine and forget it, so the next call rebuilds from config.

    Intended for tests and for a deliberate reconfiguration. Safe to call when
    no engine exists.
    """
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            try:
                _engine.dispose()
            finally:
                _engine = None
        _session_factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager: commit on success, roll back on error, always close.

    For CLI/background use. Request handlers should use :func:`get_db` instead.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Nothing uses this yet — Phase 1 adds no routes. It exists so Phase 4/5 can
    write ``Depends(get_db)`` without redesigning the session layer.

    The session is **not** committed here: the caller owns its transaction.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
