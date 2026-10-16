"""
Phase 3 session-cookie HTTP test — does a new tab inherit the session?

The question behind the "signing in does not persist across tabs" report is
whether the server issues a cookie that a *different* client can present. This
module answers it at the HTTP layer, with the real auth router, real cookies and
a real database (in-memory SQLite):

  * log in as client 1, capture the cookie the server actually sets;
  * build a BRAND NEW client that carries only that cookie value — it shares no
    client-side or server-side state with client 1, which is what a second
    browser tab looks like from the server's point of view;
  * assert the second client is authenticated.

It also pins the cookie attributes that decide whether a browser will store and
resend it at all (HttpOnly, Path, SameSite, no Domain), and that a revoked or
forged cookie is rejected.

Only the auth router is mounted: ``src.retrieval.frontend.server`` needs the full
ML dependency set, which ``test_auth_phase3_e2e_http.py`` skips without it.
Mounting the router alone keeps this runnable anywhere and keeps the assertion
focused on the cookie contract.

Nothing here touches a real database or an SMTP server.
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

# tests/ is on sys.path when pytest runs with rootdir=/home/user/audit
sys.path.insert(0, os.path.dirname(__file__))

from src.auth import models  # noqa: E402,F401  (registers users/otp/sessions)
from src.auth.config import load_auth_config  # noqa: E402
from src.auth.mailer import CapturingMailer  # noqa: E402
from src.auth.rate_limit import reset_limiter  # noqa: E402
from src.auth.router import router as auth_router  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import get_db  # noqa: E402

GOOD_PASSWORD = "Tide-Quiet-77-K9"
EMAIL = "dev.admin@gmail.com"

#: The auth configuration under test.
#:
#: Deliberately NOT written into ``os.environ`` at import time. The router
#: builds its config from the environment on every request, so seeding it here
#: and applying it with ``monkeypatch`` keeps this module from leaking settings
#: into every other test in the process — including any that assert on the
#: *absence* of an auth config.
TEST_ENV: dict[str, str] = {
    "AUTH_SECRET": "phase3-cookie-test-secret-not-a-real-key-0123456789",
    "DEV_ADMIN_EMAIL": "dev.admin@gmail.com",
    "APP_ENV": "development",
    # Real Argon2id defaults are slow; this suite only needs them to be valid.
    "AUTH_ARGON2_TIME_COST": "1",
    "AUTH_ARGON2_MEMORY_COST": "8192",
    "AUTH_ARGON2_PARALLELISM": "1",
}


def _env(**overrides: object) -> dict[str, str]:
    """A complete auth environment, with `overrides` applied on top."""
    env = dict(TEST_ENV)
    env.update({key: str(value) for key, value in overrides.items()})
    return env


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Apply TEST_ENV for the duration of each test, and only then."""
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The limiter is process-global. Without this, the second test's signup is
    refused by the resend cooldown, no mail is captured, and the fixture fails
    for a reason that has nothing to do with cookies."""
    reset_limiter()
    yield
    reset_limiter()


@pytest.fixture()
def world(monkeypatch):
    """A fresh app: empty in-memory DB, capturing mailer, seeded user."""
    import src.auth.router as router_mod

    config = load_auth_config(_env())
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

    # Create the account: signup is the only way to get a password set.
    code = f"{_start_signup(client, mailer)}"
    verify = client.post("/api/auth/signup/verify", json={"email": EMAIL, "otp": code})
    client.post(
        "/api/auth/signup/complete",
        json={"signup_token": verify.json()["signup_token"], "password": GOOD_PASSWORD},
    )

    return {"app": app, "client": client, "config": config, "mailer": mailer}


def _start_signup(client, mailer) -> str:
    client.post("/api/auth/signup/start", json={"email": EMAIL})
    return re.search(r"\b(\d{6})\b", mailer.last.body).group(1)


def _login(client) -> str:
    """Log in and return the raw session cookie value."""
    res = client.post("/api/auth/login", json={"email": EMAIL, "password": GOOD_PASSWORD})
    assert res.status_code == 200, res.text
    value = client.cookies.get("auditpro_session")
    assert value, "no session cookie was issued"
    return value


# ── the cookie the server actually sets ─────────────────────────────────────


def test_login_sets_a_shareable_host_only_cookie(world):
    client = world["client"]
    res = client.post("/api/auth/login", json={"email": EMAIL, "password": GOOD_PASSWORD})
    assert res.status_code == 200

    raw = res.headers["set-cookie"]
    assert "auditpro_session=" in raw
    assert "HttpOnly" in raw, "the session must not be readable from JavaScript"
    assert "Path=/" in raw, "a narrower Path would hide it from /api/auth/me"
    assert "samesite=lax" in raw.lower(), "Lax still sends it on same-origin navigation"
    # No Domain: a host-only cookie. Setting Domain would widen the scope.
    assert not re.search(r"Domain=", raw, re.I), "cookie must stay host-only"


def test_the_cookie_is_not_flagged_secure_over_local_http(world):
    """In development the Secure flag must be off, or plain HTTP cannot use it."""
    client = world["client"]
    res = client.post("/api/auth/login", json={"email": EMAIL, "password": GOOD_PASSWORD})
    assert "Secure" not in res.headers["set-cookie"]


def test_production_sets_the_secure_flag(monkeypatch, world):
    """...and on in production. If you serve production over plain HTTP, the
    browser will drop the cookie — that is a deployment fix (use HTTPS, or set
    AUTH_SESSION_COOKIE_SECURE=false for a trusted LAN), not a code change."""
    import src.auth.router as router_mod

    monkeypatch.setattr(
        router_mod, "_config", lambda: load_auth_config(_env(APP_ENV="production"))
    )

    client = world["client"]
    res = client.post("/api/auth/login", json={"email": EMAIL, "password": GOOD_PASSWORD})
    assert res.status_code == 200
    assert "Secure" in res.headers["set-cookie"]


# ── the second-tab question ─────────────────────────────────────────────────


def test_a_second_client_presenting_the_same_cookie_is_authenticated(world):
    """A brand-new client — no shared state — gets the same session. This is
    what a second browser tab looks like to the server."""
    cookie_value = _login(world["client"])

    tab2 = TestClient(world["app"], cookies={"auditpro_session": cookie_value})
    res = tab2.get("/api/auth/me")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["authenticated"] is True
    assert body["user"]["email"] == EMAIL


def test_the_second_client_never_had_to_log_in(world):
    """The second tab authenticates from the cookie alone.

    Every path the second client requests is recorded, so the claim is checked
    against what it actually sent rather than assumed: it must hit
    /api/auth/me, and must never touch /api/auth/login — no credential is ever
    exchanged in the second tab.
    """
    cookie_value = _login(world["client"])

    requested: list[str] = []
    tab2 = TestClient(world["app"], cookies={"auditpro_session": cookie_value})
    tab2.event_hooks["request"].append(lambda request: requested.append(request.url.path))

    res = tab2.get("/api/auth/me")
    assert res.status_code == 200, res.text
    assert res.json()["authenticated"] is True

    assert "/api/auth/login" not in requested, f"second tab logged in: {requested}"
    assert requested == ["/api/auth/me"], f"unexpected requests: {requested}"


def test_logout_in_one_client_revokes_it_for_the_other(world):
    """The session is server-side state, so ending it ends it everywhere."""
    cookie_value = _login(world["client"])
    tab2 = TestClient(world["app"], cookies={"auditpro_session": cookie_value})
    assert tab2.get("/api/auth/me").status_code == 200

    out = world["client"].post("/api/auth/logout", json={})
    assert out.status_code == 200
    assert out.json()["revoked"] is True

    after = tab2.get("/api/auth/me")
    assert after.status_code == 401
    assert after.json()["detail"]["code"] == "not_authenticated"


def test_a_forged_cookie_is_rejected(world):
    _login(world["client"])
    forger = TestClient(world["app"], cookies={"auditpro_session": "not-a-real-token"})
    res = forger.get("/api/auth/me")
    assert res.status_code == 401


def test_an_expired_session_is_rejected(world, monkeypatch):
    """A session past its hard TTL must not authenticate.

    `expires_at` is stamped onto the session row when it is created, so the short
    TTL has to be in force *at login time*. Lowering the setting afterwards does
    not retroactively expire a live session — which is the correct behaviour,
    and worth pinning so nobody "fixes" it the other way.
    """
    import time

    import src.auth.router as router_mod

    monkeypatch.setattr(
        router_mod,
        "_config",
        lambda: load_auth_config(_env(AUTH_SESSION_TTL_SECONDS="1")),
    )
    cookie_value = _login(world["client"])

    # Alive first — proves the 401 below is expiry, not a bad cookie.
    live = TestClient(world["app"], cookies={"auditpro_session": cookie_value}).get(
        "/api/auth/me"
    )
    assert live.status_code == 200, live.text

    time.sleep(1.2)

    expired = TestClient(world["app"], cookies={"auditpro_session": cookie_value}).get(
        "/api/auth/me"
    )
    assert expired.status_code == 401
    assert expired.json()["detail"]["code"] == "not_authenticated"


def test_a_failed_login_issues_no_cookie(world):
    client = TestClient(world["app"])
    res = client.post(
        "/api/auth/login", json={"email": EMAIL, "password": "wrong-password"}
    )
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "invalid_credentials"
    assert "set-cookie" not in res.headers
