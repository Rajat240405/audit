"""Shared helpers for the Phase 2 (signup / email OTP) test-suite.

Named with a leading underscore so pytest does not collect it as a test module.
Importing it only imports project code — no database server, no SMTP server and
no network.

SQLite in-memory stands in for PostgreSQL so the *service logic* can be tested
without a database server. This is a deliberate trade-off: it proves the
service code is correct, but it does NOT prove the PostgreSQL schema or the
Alembic migration is correct. Those checks live in
``test_auth_database_integration.py`` and stay skipped unless
``DATABASE_URL_TEST`` names a throwaway database.

Argon2 parameters are lowered by default here (a real 64 MiB / 3-pass hash
costs ~150 ms and this suite hashes dozens of times). The production defaults
are asserted directly by a config test instead, so lowering them here cannot
hide a weakened default.
"""

from __future__ import annotations

import re
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.auth import models as _auth_models  # noqa: F401  (registers User/EmailOtp)
from src.auth.config import AuthConfig, load_auth_config
from src.auth.mailer import CapturingMailer
from src.auth.rate_limit import reset_limiter
from src.auth.service import complete_signup, start_signup, verify_signup_otp
from src.db.models import Base

TEST_SECRET = "phase2-test-secret-not-a-real-key-0123456789abcdef"
DEV_EMAIL = "dev.admin@gmail.com"

#: Real Argon2id defaults are slow; these keep the suite fast. The production
#: defaults are asserted in test_auth_security.py::test_production_defaults.
FAST_ARGON2 = {
    "AUTH_ARGON2_TIME_COST": "1",
    "AUTH_ARGON2_MEMORY_COST": "8192",
    "AUTH_ARGON2_PARALLELISM": "1",
}


def base_env(**overrides: object) -> dict[str, str]:
    """A complete, deterministic environment — never reads ``os.environ``."""
    env = {
        "AUTH_SECRET": TEST_SECRET,
        "DEV_ADMIN_EMAIL": DEV_EMAIL,
        "APP_ENV": "development",
        **FAST_ARGON2,
    }
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def make_config(**overrides: object) -> AuthConfig:
    """Build an AuthConfig from an explicit env mapping (no os.environ)."""
    return load_auth_config(base_env(**overrides))


def new_engine():
    """A private in-memory SQLite engine. StaticPool keeps one connection."""
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture()
def sqlite_session():
    """A Session over a fresh, empty in-memory database (schema created)."""
    engine = new_engine()
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_auth_state():
    """Rate limits are process-global; keep tests isolated from each other."""
    reset_limiter()
    yield
    reset_limiter()


@contextmanager
def temporary_session():
    """Non-fixture variant, for tests that need two independent sessions."""
    engine = new_engine()
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def code_from_message(message, length: int = 6) -> str:
    """Extract the code from one captured message."""
    match = re.search(rf"\b\d{{{length}}}\b", message.body)
    assert match, f"no {length}-digit code found in the mail body"
    return match.group(0)


def code_from_mail(mailer: CapturingMailer, length: int = 6) -> str:
    """Extract the code from the most recent captured message.

    Going through the mail body — rather than reading ``dev_otp`` — proves the
    code is actually delivered to the user, not merely generated.
    """
    message = mailer.last  # `last` is a property, not a method
    assert message is not None, "no mail was captured"
    return code_from_message(message, length)


def good_password() -> str:
    """A password that satisfies the default policy."""
    return "K7-Quiet-Tide-92"


def register_user(
    session,
    config: AuthConfig,
    mailer: CapturingMailer,
    email: str = "boss@incois.gov.in",
    password: str | None = None,
):
    """Run the whole Phase 2 signup flow and return the completed account.

    Phase 3 tests need an existing, verified account to sign in to.
    """
    start_signup(session, email=email, config=config, mailer=mailer)
    verified = verify_signup_otp(
        session, email=email, code=code_from_mail(mailer), config=config
    )
    return complete_signup(
        session,
        signup_token=verified.signup_token,
        password=password or good_password(),
        config=config,
    )
