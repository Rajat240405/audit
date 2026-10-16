"""PostgreSQL foundation (Phase 1).

Environment-driven configuration, a **lazy** connection engine and Alembic's
metadata target. Nothing in this package connects to PostgreSQL at import time
— see ``src/db/session.py``.

Phase 1 deliberately contains NO authentication, user, session, OTP,
password-reset, chat-history or admin code. The only table is ``app_meta``
(``src/db/models.py``), which exists so the foundation can be *verified* end to
end: connect, migrate up, write one row, migrate down.

Public surface:

    from src.db import (
        DatabaseConfig, DatabaseConfigError, load_database_config,
        mask_database_url, get_engine, reset_engine,
        get_session_factory, get_db, Base,
    )

``src/retrieval/frontend/server.py`` does **not** import this package yet; that
wiring is Phase 4 / Phase 5 work.
"""

from __future__ import annotations

from src.db.config import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MAX_OVERFLOW,
    DEFAULT_POOL_SIZE,
    DEFAULT_PORT,
    SQLALCHEMY_SCHEME,
    DatabaseConfig,
    DatabaseConfigError,
    load_database_config,
    mask_database_url,
)
from src.db.models import AppMeta, Base
from src.db.session import (
    get_db,
    get_engine,
    get_session_factory,
    reset_engine,
    session_scope,
)

__all__ = [
    "AppMeta",
    "Base",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OVERFLOW",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PORT",
    "DatabaseConfig",
    "DatabaseConfigError",
    "SQLALCHEMY_SCHEME",
    "get_db",
    "get_engine",
    "get_session_factory",
    "load_database_config",
    "mask_database_url",
    "reset_engine",
    "session_scope",
]
