"""Alembic environment — wires migrations to src.db.config / src.db.session.

Two deliberate properties:

1. **It never imports ``src.retrieval.frontend.server``.** That module runs
   ``ensure_data_dirs()`` and applies torch thread limits at import time.
   Only ``src.db.config`` (pure, no I/O) and ``src.db.models`` (SQLAlchemy only)
   are imported here.

2. **The password never reaches Alembic's config or logs.** Alembic's
   ``sqlalchemy.url`` option is set to the MASKED url (used for offline ``--sql``
   rendering, where no credential is needed); the real connection is made
   through our own lazy engine in :func:`run_migrations_online`.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig

from sqlalchemy import Engine

from alembic import context
from src.db.config import DatabaseConfigError, load_database_config
from src.db.models import Base
from src.db.session import get_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_GUIDANCE = (
    "Set DATABASE_URL, or the DATABASE_HOST / DATABASE_PORT / DATABASE_NAME / "
    "DATABASE_USER / DATABASE_PASSWORD parts, in your environment. See "
    ".env.example and docs/database-setup.md. Credentials belong in .env "
    "(local) or .env.hpc (HPC) — never in tracked source."
)


def _resolve_config():
    """Load database config, converting a config fault into a clean exit."""
    try:
        return load_database_config()
    except DatabaseConfigError as exc:
        print(f"alembic: database configuration error: {exc}\n{_GUIDANCE}",
              file=sys.stderr)
        raise SystemExit(2) from exc


def run_migrations_offline() -> None:
    """Render SQL without connecting (``alembic upgrade head --sql``).

    The masked URL is sufficient: DDL rendering needs the dialect, not the
    credential. This is also what keeps the password out of generated .sql.
    """
    cfg = _resolve_config()
    context.configure(
        url=cfg.safe_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations through the project's lazy engine."""
    cfg = _resolve_config()
    # Keep Alembic's own option masked; the engine below holds the real URL.
    config.set_main_option("sqlalchemy.url", cfg.safe_url())
    connectable: Engine = get_engine(cfg)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
