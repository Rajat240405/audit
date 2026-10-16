"""
Phase 3 regression — the signup allowlist must not gate *login*.

Reported symptom
----------------
A registered user submits correct credentials to ``POST /api/auth/login`` and
gets::

    HTTP 403 {"detail": {"code": "email_not_allowed",
                         "message": "this email address is not allowed to sign up"}}

Root cause
----------
``src/auth/service.py::login`` normalised the submitted address with
``validate_signup_email()`` — the *signup* validator, which enforces the
allowlist (``@incois.gov.in`` plus one ``DEV_ADMIN_EMAIL``) and raises
``email_not_allowed`` (403). The allowlist is a policy about who may *create*
an account; using it as an authentication check means anyone whose address is
not on it can never sign in, even with a valid, verified account and the right
password. It also silently locks out existing users if the allowlist is
reconfigured after they registered.

The fix normalises with ``normalize_email()`` instead: same canonicalisation
(so the lookup, the rate-limit key and the stored address agree) and the same
``email_invalid`` rejection for malformed input, but no allowlist gate.

What these tests pin down
-------------------------
Login side (the bug):

  * a registered, verified, active account can sign in even when its address is
    NOT on the allowlist — 200, not 403;
  * login never answers ``email_not_allowed``;
  * wrong password -> 401 ``invalid_credentials`` (generic, as before);
  * unknown address -> 401 ``invalid_credentials`` (no enumeration);
  * a malformed address is still rejected with 400 ``email_invalid`` — the fix
    did not remove validation, only the allowlist gate;
  * case/whitespace normalisation is preserved;
  * rate limiting still applies to login.

Signup side (the guard-rail — the allowlist must keep working):

  * an unapproved address is still refused with 403 ``email_not_allowed``;
  * the organisational domain and the single dev address are still accepted.

Only the auth router is mounted, so this runs without the ML dependency set.
Nothing here touches a real database or an SMTP server, and no credential or
secret is printed.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# tests/ is on sys.path when pytest runs with rootdir=<repo>
sys.path.insert(0, os.path.dirname(__file__))

from src.auth import models  # noqa: E402,F401  (registers users/otp/sessions)
from src.auth.mailer import CapturingMailer  # noqa: E402
from src.auth.rate_limit import reset_limiter  # noqa: E402
from src.auth.router import router as auth_router  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import get_db  # noqa: E402

GOOD_PASSWORD = "Tide-Quiet-77-K9"
WRONG_PASSWORD = "Not-The-Right-Pass-42"

#: A Gmail address — deliberately NOT ``@incois.gov.in``. It is allowlisted
#: during signup purely by being named as DEV_ADMIN_EMAIL, which lets the test
#: revoke that status afterwards and model "an existing account whose address
#: has fallen off the allowlist".
ACCOUNT = "registered.user@gmail.com"

#: What DEV_ADMIN_EMAIL is set to once the account exists. The account is then
#: no longer allowlisted — exactly the reported scenario.
OTHER_DEV_EMAIL = "someone.else@incois.gov.in"

BASE_ENV: dict[str, str] = {
    "AUTH_SECRET": "phase3-login-allowlist-secret-not-a-real-key-0123456789",
    "APP_ENV": "development",
    # Real Argon2id defaults are slow; this suite only needs them to be valid.
    "AUTH_ARGON2_TIME_COST": "1",
    "AUTH_ARGON2_MEMORY_COST": "8192",
    "AUTH_ARGON2_PARALLELISM": "1",
}


def _env(**overrides: object) -> dict[str, str]:
    env = dict(BASE_ENV)
    env["DEV_ADMIN_EMAIL"] = ACCOUNT
    env.update({key: str(value) for key, value in overrides.items()})
    return env


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Apply the base environment for each test only — never at import time."""
    for key, value in _env().items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The limiter is process-global; keep tests isolated from one another."""
    reset_limiter()
    yield
    reset_limiter()


@pytest.fixture()
def world(monkeypatch):
    """A fresh app with one registered, verified, active account.

    The account is created while its address IS allowlisted (it is the
    configured DEV_ADMIN_EMAIL), which is the only way to obtain a password.
    Tests then revoke that status with :func:`revoke_allowlist` to reproduce
    the reported condition.
    """
    import src.auth.router as router_mod

    mailer = CapturingMailer()
    monkeypatch.setattr(router_mod, "get_mailer", lambda config=None: mailer)

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def _db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = _db
    client = TestClient(app)

    # Sign up: start -> read the code out of the captured mail -> verify -> complete.
    client.post("/api/auth/signup/start", json={"email": ACCOUNT})
    code = re.search(r"\b(\d{6})\b", mailer.last.body).group(1)
    verified = client.post(
        "/api/auth/signup/verify", json={"email": ACCOUNT, "otp": code}
    )
    assert verified.status_code == 200, verified.text
    completed = client.post(
        "/api/auth/signup/complete",
        json={
            "signup_token": verified.json()["signup_token"],
            "password": GOOD_PASSWORD,
        },
    )
    assert completed.status_code == 200, completed.text

    return {"client": client, "mailer": mailer, "monkeypatch": monkeypatch}


def revoke_allowlist(world) -> None:
    """Make the registered account's address no longer allowlisted."""
    world["monkeypatch"].setenv("DEV_ADMIN_EMAIL", OTHER_DEV_EMAIL)


def login(client, email: str = ACCOUNT, password: str = GOOD_PASSWORD):
    return client.post(
        "/api/auth/login", json={"email": email, "password": password}
    )


def error_code(response) -> str | None:
    detail = response.json().get("detail")
    return detail.get("code") if isinstance(detail, dict) else None


# ── the reported bug ────────────────────────────────────────────────────────


def test_registered_account_can_sign_in_when_not_allowlisted(world):
    """The regression: an existing account signs in despite being off-list.

    Before the fix this returned 403 ``email_not_allowed``.
    """
    revoke_allowlist(world)
    response = login(world["client"])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "signed_in"
    assert response.json()["email"] == ACCOUNT


def test_login_never_answers_email_not_allowed(world):
    """Login must not answer with a signup-policy error, on any input."""
    revoke_allowlist(world)
    client = world["client"]
    for email, password in (
        (ACCOUNT, GOOD_PASSWORD),      # correct credentials
        (ACCOUNT, WRONG_PASSWORD),     # wrong password
        ("nobody@incois.gov.in", GOOD_PASSWORD),  # unknown address
    ):
        response = login(client, email, password)
        assert error_code(response) != "email_not_allowed", (
            f"login returned the signup allowlist error for {email!r}: "
            f"{response.status_code} {response.text}"
        )


def test_login_still_issues_the_session_cookie(world):
    """The Phase 3 session-cookie contract survives the fix."""
    revoke_allowlist(world)
    response = login(world["client"])
    assert response.status_code == 200
    raw = response.headers["set-cookie"]
    assert "auditpro_session=" in raw
    assert "HttpOnly" in raw
    assert "Path=/" in raw
    assert "samesite=lax" in raw.lower()


def test_authenticated_after_login(world):
    """``GET /api/auth/me`` succeeds with the cookie login issued."""
    revoke_allowlist(world)
    client = world["client"]
    assert login(client).status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["authenticated"] is True
    assert me.json()["user"]["email"] == ACCOUNT


# ── generic failure behaviour must be unchanged ─────────────────────────────


def test_wrong_password_still_returns_generic_401(world):
    revoke_allowlist(world)
    response = login(world["client"], ACCOUNT, WRONG_PASSWORD)
    assert response.status_code == 401
    assert error_code(response) == "invalid_credentials"


def test_unknown_address_still_returns_generic_401(world):
    """No account enumeration: unknown address == wrong password."""
    response = login(world["client"], "ghost@incois.gov.in", GOOD_PASSWORD)
    assert response.status_code == 401
    assert error_code(response) == "invalid_credentials"


def test_malformed_email_is_still_rejected_with_400(world):
    """The fix removed the allowlist gate, not email validation."""
    response = login(world["client"], "not-an-email", GOOD_PASSWORD)
    assert response.status_code == 400
    assert error_code(response) == "email_invalid"


def test_login_still_normalizes_case_and_whitespace(world):
    """Normalisation is what the lookup and the rate-limit key depend on."""
    revoke_allowlist(world)
    response = login(world["client"], "  Registered.User@GMAIL.COM  ", GOOD_PASSWORD)
    assert response.status_code == 200, response.text
    assert response.json()["email"] == ACCOUNT


def test_login_rate_limiting_still_works(world):
    """The limiter is still applied at login, before any credential check."""
    world["monkeypatch"].setenv("AUTH_LOGIN_LIMIT", "2")
    client = world["client"]
    for _ in range(2):
        assert login(client, ACCOUNT, WRONG_PASSWORD).status_code == 401
    limited = login(client, ACCOUNT, WRONG_PASSWORD)
    assert limited.status_code == 429, limited.text
    assert error_code(limited) == "rate_limited"


# ── the allowlist must still gate SIGNUP ────────────────────────────────────


def test_signup_still_rejects_an_unapproved_address(world):
    """Signup restrictions for unapproved addresses remain intact."""
    client = world["client"]
    response = client.post(
        "/api/auth/signup/start", json={"email": "random.person@hotmail.com"}
    )
    assert response.status_code == 403
    assert error_code(response) == "email_not_allowed"


def test_signup_still_accepts_the_organisational_domain(world):
    client = world["client"]
    response = client.post(
        "/api/auth/signup/start", json={"email": "new.staff@incois.gov.in"}
    )
    assert response.status_code == 200, response.text


def test_signup_still_accepts_the_configured_dev_address(world):
    """DEV_ADMIN_EMAIL keeps its one-address exception."""
    revoke_allowlist(world)
    client = world["client"]
    response = client.post(
        "/api/auth/signup/start", json={"email": OTHER_DEV_EMAIL}
    )
    assert response.status_code == 200, response.text


def test_successful_login_does_not_create_an_account(world):
    """Login must never auto-provision a user (no account creation on login)."""
    revoke_allowlist(world)
    client = world["client"]
    before = client.post("/api/auth/signup/start", json={"email": "ghost@incois.gov.in"})
    assert before.status_code == 200, before.text
    # A login attempt for an address that has started but never completed
    # signup must fail — starting signup is not an account.
    response = login(client, "ghost@incois.gov.in", GOOD_PASSWORD)
    assert response.status_code == 401
    assert error_code(response) == "invalid_credentials"
