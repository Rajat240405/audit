"""PostgreSQL configuration (Phase 1 foundation).

Environment-driven, mirroring the convention already established by
``src/graphrag/config.py``:

  * every value comes from the environment — nothing is hard-coded;
  * secrets (the password) come **only** from the environment and are never
    committed, never logged and never echoed in an exception;
  * a required value has NO default, so a misconfiguration fails loudly at
    configuration time instead of silently at first request;
  * ``validate()`` raises :class:`DatabaseConfigError` for any fault.

This module performs **no I/O**. Importing it never touches the network and
never imports ``src.retrieval.frontend.server`` (whose import-time
``ensure_data_dirs()`` side effects would otherwise ride along).

Environment variables
---------------------
Either supply a complete URL::

    DATABASE_URL        Full SQLAlchemy URL. Wins over the parts below.

Or supply the parts (all five are then read)::

    DATABASE_HOST       default "localhost"
    DATABASE_PORT       default 5432
    DATABASE_NAME       REQUIRED — no default (fails loudly)
    DATABASE_USER       REQUIRED — no default (fails loudly)
    DATABASE_PASSWORD   REQUIRED — no default; env/`.env` only, never committed

Optional tuning::

    DATABASE_ECHO              "1"/"true" to log every statement (default off)
    DATABASE_POOL_SIZE         default 5
    DATABASE_MAX_OVERFLOW      default 10
    DATABASE_CONNECT_TIMEOUT    seconds, default 10

Scheme handling: ``postgresql://`` and ``postgres://`` are normalised to
``postgresql+psycopg://`` so a plain URL can never silently select the psycopg2
dialect (psycopg2 is NOT a dependency of this project). Any other backend
(``sqlite``, ``mysql``, ...) is rejected.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy.engine.url import URL, make_url

__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OVERFLOW",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PORT",
    "SQLALCHEMY_SCHEME",
    "DatabaseConfig",
    "DatabaseConfigError",
    "load_database_config",
    "mask_database_url",
]

#: The only driver this project installs (psycopg 3).
SQLALCHEMY_SCHEME = "postgresql+psycopg"

#: Accepted plain-URL aliases, normalised to :data:`SQLALCHEMY_SCHEME`.
_SCHEME_ALIASES = frozenset({"postgres", "postgresql"})

DEFAULT_PORT = 5432
DEFAULT_HOST = "localhost"
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class DatabaseConfigError(Exception):
    """Configuration fault — raised loudly, never swallowed.

    Messages name the *variable* that is wrong. They never echo the password.
    """


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _require(parts: Mapping[str, str], name: str) -> str:
    """Return a required variable or fail loudly naming it."""
    raw = _clean(parts.get(name))
    if not raw:
        raise DatabaseConfigError(
            f"{name} is required and is not set. "
            f"Set {name} in your .env (local) or .env.hpc (HPC); "
            f"see .env.example and docs/database-setup.md."
        )
    return raw


def _int_env(parts: Mapping[str, str], name: str, default: int) -> int:
    raw = _clean(parts.get(name))
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DatabaseConfigError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < 0:
        raise DatabaseConfigError(f"{name} must be >= 0, got {value}.")
    return value


def _bool_env(parts: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _clean(parts.get(name)).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise DatabaseConfigError(
        f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}."
    )


def _normalise_scheme(raw: str, *, source: str) -> str:
    """Force the psycopg 3 dialect for any plain PostgreSQL URL."""
    scheme, sep, rest = raw.partition("://")
    if not sep:
        raise DatabaseConfigError(
            f"{source} must be a SQLAlchemy URL of the form "
            f"'{SQLALCHEMY_SCHEME}://user:password@host:port/database' "
            f"(got a value with no '://' scheme separator)."
        )
    candidate = scheme.strip().lower()
    if candidate in _SCHEME_ALIASES:
        return f"{SQLALCHEMY_SCHEME}://{rest}"
    if candidate == SQLALCHEMY_SCHEME:
        return raw
    raise DatabaseConfigError(
        f"{source} must be a PostgreSQL URL (scheme '{SQLALCHEMY_SCHEME}', "
        f"'postgresql' or 'postgres'), got {candidate + '://'!r}. "
        f"This deployment is PostgreSQL-only."
    )


def mask_database_url(url: str) -> str:
    """Return ``url`` with the password replaced by ``***``.

    Used for every log line, every exception message and ``repr()``. The raw
    URL is needed by the engine, so it is kept on :class:`DatabaseConfig` in a
    field that is excluded from ``repr`` and only ever surfaced through
    :meth:`DatabaseConfig.safe_url`.
    """
    try:
        parsed = make_url(url)
    except Exception:  # noqa: BLE001 - never let masking itself explode
        # Deliberately does NOT echo the input: an unparsable URL may be a
        # mistyped credential and must not be copied into a log.
        return "<unparsable database url>"
    return parsed.render_as_string(hide_password=True)


@dataclass(frozen=True, repr=False)
class DatabaseConfig:
    """Validated PostgreSQL connection settings.

    ``url`` holds the REAL connection string (password included) because the
    engine needs it. It is excluded from the generated ``repr`` and replaced by
    :meth:`safe_url` in the hand-written one, so a stray ``print(config)``
    cannot leak a credential.
    """

    url: str
    echo: bool = False
    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW
    pool_pre_ping: bool = True
    connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS

    def __repr__(self) -> str:  # never leak the password
        return (
            f"DatabaseConfig(url={self.safe_url()!r}, echo={self.echo!r}, "
            f"pool_size={self.pool_size!r}, max_overflow={self.max_overflow!r}, "
            f"pool_pre_ping={self.pool_pre_ping!r}, "
            f"connect_timeout_seconds={self.connect_timeout_seconds!r})"
        )

    def __str__(self) -> str:  # never leak the password
        return self.safe_url()

    def safe_url(self) -> str:
        """The connection URL with the password masked — safe to log."""
        return mask_database_url(self.url)

    def validate(self) -> DatabaseConfig:
        """Check the URL is a complete PostgreSQL/psycopg URL. Fail loudly."""
        try:
            parsed = make_url(self.url)
        except Exception as exc:  # noqa: BLE001
            # The exception text can embed the raw URL, so only the class name
            # is reported. Nothing here echoes the input.
            raise DatabaseConfigError(
                f"database URL could not be parsed (SQLAlchemy raised "
                f"{type(exc).__name__}). Check DATABASE_URL or the "
                f"DATABASE_HOST/PORT/NAME/USER/PASSWORD parts."
            ) from exc

        backend = parsed.get_backend_name()
        if backend != "postgresql":
            raise DatabaseConfigError(
                f"database URL must use the PostgreSQL backend, got {backend!r}."
            )
        driver = parsed.get_driver_name()
        if driver != "psycopg":
            raise DatabaseConfigError(
                f"database URL must use the psycopg 3 driver "
                f"('{SQLALCHEMY_SCHEME}://'), got driver {driver!r}."
            )
        if not parsed.database:
            raise DatabaseConfigError(
                "database URL has no database name. Set DATABASE_NAME "
                "(or put /<database> at the end of DATABASE_URL)."
            )
        if not parsed.username:
            raise DatabaseConfigError(
                "database URL has no username. Set DATABASE_USER "
                "(or put <user>: in DATABASE_URL)."
            )
        if not parsed.password:
            raise DatabaseConfigError(
                "database URL has no password. Set DATABASE_PASSWORD in your "
                "env file (.env locally, .env.hpc on HPC) — never in tracked "
                "source. See docs/database-setup.md."
            )
        return self


def load_database_config(env: Mapping[str, str] | None = None) -> DatabaseConfig:
    """Build and validate :class:`DatabaseConfig` from ``env`` (default ``os.environ``).

    Raises :class:`DatabaseConfigError` on any fault — there is no partial or
    "best effort" configuration.
    """
    parts = os.environ if env is None else env

    raw_url = _clean(parts.get("DATABASE_URL"))
    if raw_url:
        url = _normalise_scheme(raw_url, source="DATABASE_URL")
    else:
        # No URL: compose from the parts, percent-encoding each component
        # safely. A password containing '@', '/' or ':' must not break the URL.
        name = _require(parts, "DATABASE_NAME")
        user = _require(parts, "DATABASE_USER")
        password = _require(parts, "DATABASE_PASSWORD")
        host = _clean(parts.get("DATABASE_HOST")) or DEFAULT_HOST
        port = _int_env(parts, "DATABASE_PORT", DEFAULT_PORT)
        url = URL.create(
            SQLALCHEMY_SCHEME,
            username=user,
            password=password,
            host=host,
            port=port,
            database=name,
        ).render_as_string(hide_password=False)

    config = DatabaseConfig(
        url=url,
        echo=_bool_env(parts, "DATABASE_ECHO", False),
        pool_size=_int_env(parts, "DATABASE_POOL_SIZE", DEFAULT_POOL_SIZE),
        max_overflow=_int_env(parts, "DATABASE_MAX_OVERFLOW", DEFAULT_MAX_OVERFLOW),
        pool_pre_ping=True,
        connect_timeout_seconds=_int_env(
            parts, "DATABASE_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
    )
    return config.validate()
