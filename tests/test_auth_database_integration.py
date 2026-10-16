"""Phase 2 — signup against a REAL PostgreSQL database.

These tests are SKIPPED unless ``DATABASE_URL_TEST`` is set and names a database
whose name contains ``test``. That guard is deliberate: the migration tests
below create and drop tables, and they must never be pointed at development or
production data by accident.

    DATABASE_URL_TEST=postgresql+psycopg://incois:change-me@localhost:5433/incois_audit_test \
        python -m pytest tests/test_auth_database_integration.py -v

Everything that can be verified without a server is verified in
``test_auth_service.py`` (SQLite) and ``test_auth_migrations.py`` (offline DDL).
"""

from __future__ import annotations

import os

import pytest
from _auth_helpers import (  # noqa: F401  (autouse fixture must be imported)
    _reset_auth_state,
    code_from_mail,
    code_from_message,
    good_password,
    make_config,
)
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.auth import models as am
from src.auth.mailer import CapturingMailer
from src.auth.passwords import verify_password
from src.auth.service import (
    complete_password_reset,
    complete_signup,
    current_user,
    login,
    start_password_reset,
    start_signup,
    verify_signup_otp,
)
from src.db import get_engine, reset_engine

TEST_URL_ENV = "DATABASE_URL_TEST"
REQUIRED_NAME_FRAGMENT = "test"


def _require_test_database_url() -> str:
    raw = (os.environ.get(TEST_URL_ENV) or "").strip()
    if not raw:
        pytest.skip(
            f"{TEST_URL_ENV} is not set — skipping live PostgreSQL signup tests. "
            f"These need a real, throwaway PostgreSQL database."
        )
    try:
        parsed = make_url(raw)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TEST_URL_ENV} is not a parsable URL ({type(exc).__name__}).")

    name = parsed.database or ""
    if REQUIRED_NAME_FRAGMENT not in name.lower():
        pytest.skip(
            f"refusing to run: database name {name!r} does not contain "
            f"{REQUIRED_NAME_FRAGMENT!r}. Point {TEST_URL_ENV} at a dedicated "
            f"*_test database."
        )
    return raw


@pytest.fixture(scope="module", autouse=True)
def test_database_url() -> str:
    url = _require_test_database_url()
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    reset_engine()
    yield url
    reset_engine()
    if previous is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous


@pytest.fixture()
def engine(test_database_url: str) -> Engine:
    reset_engine()
    return get_engine()


@pytest.fixture()
def session(engine: Engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        # Leave nothing behind, whatever the test did.
        db.rollback()
        db.execute(text("DELETE FROM email_otp"))
        db.execute(text("DELETE FROM users"))
        db.commit()
        db.close()


# ── the server really is PostgreSQL ──────────────────────────────────────────


def test_the_server_is_postgresql(engine: Engine):
    assert engine.dialect.name == "postgresql"


# ── the migrated schema ──────────────────────────────────────────────────────


def test_the_signup_tables_exist(engine: Engine):
    tables = set(inspect(engine).get_table_names())
    assert {"users", "email_otp"} <= tables


def test_users_columns_and_constraints(engine: Engine):
    insp = inspect(engine)
    columns = {c["name"] for c in insp.get_columns("users")}
    assert {
        "id",
        "email",
        "email_normalized",
        "password_hash",
        "email_verified",
        "is_active",
        "created_at",
        "updated_at",
    } <= columns

    uniques = insp.get_unique_constraints("users")
    assert any("email_normalized" in u["column_names"] for u in uniques)


def test_email_otp_columns(engine: Engine):
    columns = {c["name"] for c in inspect(engine).get_columns("email_otp")}
    assert {
        "id",
        "email_normalized",
        "purpose",
        "code_hash",
        "salt",
        "expires_at",
        "attempts",
        "max_attempts",
        "consumed_at",
        "created_at",
        "last_sent_at",
        "resend_count",
    } <= columns


def test_the_unique_constraint_is_enforced_by_the_database(session):
    session.add(
        am.User(
            id=am.new_id(),
            email="dup@incois.gov.in",
            email_normalized="dup@incois.gov.in",
            password_hash="x",
        )
    )
    session.commit()

    session.add(
        am.User(
            id=am.new_id(),
            email="DUP@incois.gov.in",
            email_normalized="dup@incois.gov.in",
            password_hash="y",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_timestamp_columns_are_timezone_aware(engine: Engine):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'email_otp' AND column_name = 'expires_at'"
            )
        ).one()
    assert row.data_type == "timestamp with time zone"


# ── the real signup flow, on PostgreSQL ──────────────────────────────────────


def test_a_full_signup_flow_persists_to_postgresql(session):
    config = make_config()
    mailer = CapturingMailer()

    start_signup(session, email="Real.User@incois.gov.in", config=config, mailer=mailer)
    assert len(mailer.sent) == 1

    verified = verify_signup_otp(
        session,
        email="real.user@incois.gov.in",
        code=code_from_mail(mailer),
        config=config,
    )
    done = complete_signup(
        session, signup_token=verified.signup_token, password=good_password(), config=config
    )
    assert done.email_verified is True

    user = session.scalars(
        select(am.User).where(am.User.email_normalized == "real.user@incois.gov.in")
    ).one()
    assert user.email_verified is True
    assert verify_password(good_password(), user.password_hash, config=config) is True


def test_the_flow_works_through_a_committed_transaction(session):
    """The challenge must survive a commit — a different request reads it."""
    config = make_config()
    mailer = CapturingMailer()
    start_signup(session, email="commit@incois.gov.in", config=config, mailer=mailer)
    code = code_from_mail(mailer)

    session.commit()  # simulates the next HTTP request

    assert (
        verify_signup_otp(
            session, email="commit@incois.gov.in", code=code, config=config
        ).signup_token
        is not None
    )


# ── Phase 3: sessions, on real PostgreSQL ───────────────────────────────────


def test_the_sessions_table_exists(engine: Engine):
    assert "sessions" in set(inspect(engine).get_table_names())


def test_sessions_columns(engine: Engine):
    columns = {c["name"] for c in inspect(engine).get_columns("sessions")}
    assert {
        "id",
        "user_id",
        "token_hash",
        "created_at",
        "last_seen_at",
        "expires_at",
        "revoked_at",
        "user_agent",
    } <= columns


def test_sessions_token_hash_is_unique(engine: Engine):
    """The lookup key must be unique — a collision would be a session hijack."""
    insp = inspect(engine)
    uniques = {c["column_names"][0] for u in [insp.get_unique_constraints("sessions")] for c in [u[0]]} if False else set()
    uniques = {
        col
        for u in insp.get_unique_constraints("sessions")
        for col in u["column_names"]
    }
    assert "token_hash" in uniques


def test_sessions_have_a_cascading_foreign_key_to_users(engine: Engine):
    fks = inspect(engine).get_foreign_keys("sessions")
    assert any(
        fk["referred_table"] == "users" and fk["referred_columns"] == ["id"]
        for fk in fks
    ), f"expected sessions.user_id -> users.id, got {fks}"


def test_timestamps_are_timezone_aware(engine: Engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'sessions' AND column_name IN "
                "('expires_at', 'last_seen_at', 'revoked_at')"
            )
        ).all()
    assert len(rows) == 3
    assert {r.data_type for r in rows} == {"timestamp with time zone"}


def test_a_full_signup_login_reset_flow_persists(session):
    """Phase 2 -> Phase 3 end to end, with commits between every step."""
    config = make_config()
    mailer = CapturingMailer()

    # Phase 2: create the account.
    start_signup(session, email="Phase3@incois.gov.in", config=config, mailer=mailer)
    verified = verify_signup_otp(
        session,
        email="phase3@incois.gov.in",
        code=code_from_mail(mailer),
        config=config,
    )
    complete_signup(
        session, signup_token=verified.signup_token, password=good_password(), config=config
    )
    session.commit()

    # Phase 3: sign in.
    signed_in = login(session, email="phase3@incois.gov.in", password=good_password(), config=config)
    session.commit()
    assert current_user(session, session_token=signed_in.session_token, config=config) is not None

    # Phase 3: reset the password; the session must die.
    start_password_reset(session, email="phase3@incois.gov.in", config=config, mailer=mailer)
    session.commit()
    reset_code = code_from_message(mailer.last)
    result = complete_password_reset(
        session,
        email="phase3@incois.gov.in",
        code=reset_code,
        new_password="Tide-Quiet-77-K9",
        config=config,
    )
    session.commit()

    assert result.sessions_revoked == 1
    assert current_user(session, session_token=signed_in.session_token, config=config) is None
    assert (
        login(session, email="phase3@incois.gov.in", password="Tide-Quiet-77-K9", config=config).email
        == "phase3@incois.gov.in"
    )
