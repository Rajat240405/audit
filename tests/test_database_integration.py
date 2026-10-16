"""Phase 1 — live PostgreSQL connectivity and migration tests.

These tests are **skipped** unless a dedicated test database is configured via::

    DATABASE_URL_TEST=postgresql+psycopg://user:password@host:5432/incois_audit_test

Two safety guards, both deliberate:

1. The variable is a *separate* one (``DATABASE_URL_TEST``), never
   ``DATABASE_URL``. Pointing the test suite at your development database
   requires an explicit, separate act.
2. The database **name must contain "test"**. If it does not, every destructive
   test in this file is skipped. This is what prevents a typo from running
   ``alembic downgrade base`` against real data.

Run against a local Docker database with, for example::

    docker compose --profile db up -d
    DATABASE_URL_TEST='postgresql+psycopg://incois:change-me@localhost:5433/incois_audit_test' \
        python -m pytest tests/test_database_integration.py -v

Migration tests below DO run ``downgrade base`` and are therefore destructive —
but only ever against the database named by ``DATABASE_URL_TEST``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from src.db import AppMeta, get_engine, get_session_factory, reset_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"

TEST_URL_ENV = "DATABASE_URL_TEST"

#: The database name must contain this, or destructive tests are skipped.
REQUIRED_NAME_FRAGMENT = "test"


def _require_test_database_url() -> str:
    """Return the test URL, or skip the whole module with a clear reason."""
    raw = (os.environ.get(TEST_URL_ENV) or "").strip()
    if not raw:
        pytest.skip(
            f"{TEST_URL_ENV} is not set — skipping live PostgreSQL tests. "
            f"See docs/database-setup.md and tests/test_database_config.py "
            f"(which runs without a database)."
        )
    try:
        parsed = make_url(raw)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TEST_URL_ENV} is not a parsable URL ({type(exc).__name__}).")

    name = parsed.database or ""
    if REQUIRED_NAME_FRAGMENT not in name.lower():
        pytest.skip(
            f"refusing to run destructive migration tests: database name "
            f"{name!r} does not contain {REQUIRED_NAME_FRAGMENT!r}. "
            f"Point {TEST_URL_ENV} at a dedicated *_test database."
        )
    return raw


def _alembic(which: str, target: str) -> None:
    """Run an Alembic command against DATABASE_URL (set by the caller)."""
    from alembic.config import Config

    from alembic import command

    reset_engine()  # never reuse a pooled connection across a schema change
    cfg = Config(str(ALEMBIC_INI))
    getattr(command, which)(cfg, target)
    reset_engine()


@pytest.fixture(scope="module", autouse=True)
def test_database_url() -> str:
    """Configure DATABASE_URL for Alembic and leave the schema upgraded."""
    url = _require_test_database_url()
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    reset_engine()

    # Start from a known state: `downgrade base` is a no-op on a fresh DB.
    _alembic("downgrade", "base")
    _alembic("upgrade", "head")

    yield url

    _alembic("upgrade", "head")  # leave the test database usable
    reset_engine()
    if previous is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous


@pytest.fixture()
def engine(test_database_url: str) -> Engine:
    reset_engine()
    return get_engine()


def _table_exists(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = :t AND table_schema = current_schema())"
                ),
                {"t": table},
            ).scalar()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity
# ─────────────────────────────────────────────────────────────────────────────


def test_select_one_succeeds(engine: Engine):
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_server_reports_postgresql(engine: Engine):
    with engine.connect() as conn:
        version = str(conn.execute(text("SELECT version()")).scalar())
    assert "PostgreSQL" in version


def test_driver_is_psycopg3(engine: Engine):
    assert engine.dialect.driver == "psycopg"


def test_pool_pre_ping_is_enabled(engine: Engine):
    """HPC: the database SIF restarts independently of the application SIF."""
    assert engine.pool._pre_ping is True


# ─────────────────────────────────────────────────────────────────────────────
# Schema after `alembic upgrade head`
# ─────────────────────────────────────────────────────────────────────────────


def test_upgrade_head_creates_app_meta(engine: Engine):
    assert _table_exists(engine, "app_meta")


def test_alembic_version_records_the_head(engine: Engine):
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    heads = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads()
    assert len(heads) == 1

    with engine.connect() as conn:
        current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert current == heads[0] == "0003_auth_sessions_reset"


def test_app_meta_columns_match_the_model(engine: Engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_name = 'app_meta'"
            )
        ).fetchall()
    columns = {r[0]: (r[1], r[2]) for r in rows}
    assert columns["key"][0] == "character varying"
    assert columns["value"][0] == "text"
    assert columns["updated_at"][0] == "timestamp with time zone"
    assert columns["updated_at"][1] == "NO"


# ─────────────────────────────────────────────────────────────────────────────
# Data round trip (always rolled back)
# ─────────────────────────────────────────────────────────────────────────────


def test_raw_insert_select_rolls_back(engine: Engine):
    session = get_session_factory()()
    try:
        session.execute(text("INSERT INTO app_meta (key, value) VALUES ('probe', 'v1')"))
        assert (
            session.execute(text("SELECT value FROM app_meta WHERE key = 'probe'")).scalar()
            == "v1"
        )
        session.rollback()
        assert (
            session.execute(
                text("SELECT count(*) FROM app_meta WHERE key = 'probe'")
            ).scalar()
            == 0
        )
    finally:
        session.close()


def test_orm_model_round_trip_and_server_default(engine: Engine):
    """Proves src/db/models.py and the migration agree, including now()."""
    session = get_session_factory()()
    try:
        session.add(AppMeta(key="orm_probe", value="hello"))
        session.commit()
        session.refresh(session.get(AppMeta, "orm_probe"))  # pull the DB default
        obj = session.get(AppMeta, "orm_probe")
        assert obj.value == "hello"
        assert obj.updated_at is not None, "server_default now() did not apply"

        obj.value = "changed"
        session.commit()
        session.refresh(obj)
        assert obj.value == "changed"

        session.delete(obj)
        session.commit()
        assert session.get(AppMeta, "orm_probe") is None
    finally:
        session.rollback()
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Migration reversibility (destructive — guarded by DATABASE_URL_TEST)
# ─────────────────────────────────────────────────────────────────────────────


def test_downgrade_base_drops_app_meta(engine: Engine, test_database_url: str):
    _alembic("downgrade", "base")
    reset_engine()
    try:
        assert not _table_exists(get_engine(), "app_meta")
    finally:
        _alembic("upgrade", "head")  # restore for any later test
    assert _table_exists(get_engine(), "app_meta")


def test_full_downgrade_upgrade_round_trip(engine: Engine, test_database_url: str):
    """Migrations must be reversible and repeatable, not one-shot."""
    for _ in range(2):
        _alembic("downgrade", "base")
        reset_engine()
        assert not _table_exists(get_engine(), "app_meta")
        _alembic("upgrade", "head")
        reset_engine()
        assert _table_exists(get_engine(), "app_meta")


def test_upgrade_head_is_idempotent(test_database_url: str):
    """Running `upgrade head` twice must be a no-op, not an error."""
    _alembic("upgrade", "head")
    _alembic("upgrade", "head")
    reset_engine()
    assert _table_exists(get_engine(), "app_meta")
