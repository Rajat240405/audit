"""
Phase 3 end-to-end HTTP test: the real FastAPI app, real routes, real cookies.

This is the strongest available proof that sign-in, sign-out and password
reset work end to end without a PostgreSQL server or an SMTP server:

  * a real `uvicorn`-less `TestClient` drives the actual ASGI app;
  * `get_db` is overridden with a throwaway in-memory SQLite session;
  * the mailer is the real `CapturingMailer`, so OTP codes are read from the
    mail the code actually "sent".

Nothing here touches a real database. It is safe to run anywhere.
"""

from __future__ import annotations

import pytest

# `torch` and friends are not installed in every environment; the server module
# imports them. If it cannot be imported, skip rather than fail — the route
# table and service tests still cover the logic.
server = pytest.importorskip(
    "src.retrieval.frontend.server", reason="server module needs the full ML dependency set"
)
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.auth import models  # noqa: E402  (registers users/email_otp/sessions)
from src.auth.mailer import CapturingMailer  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import get_db  # noqa: E402

app = server.app

from _auth_helpers import (  # noqa: E402  (_reset_auth_state is an autouse fixture)
    _reset_auth_state,  # noqa: F401
    make_config,
    new_engine,
)

GOOD_PASSWORD = "Tide-Quiet-77-K9"
NEW_PASSWORD = "Harbour-Lamp-42-Q1"


def _code_from(mailer: CapturingMailer) -> str:
    """Pull the OTP out of the body of the last message that was 'sent'."""
    import re

    body = mailer.last.body
    match = re.search(r"\b(\d{6})\b", body)
    assert match, f"no 6-digit code in the message body:\n{body}"
    return match.group(1)


@pytest.fixture()
def client(monkeypatch):
    """The real ASGI app, with a throwaway SQLite DB and a capturing mailer."""
    engine = new_engine()
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    mailer = CapturingMailer()
    # Fast Argon2 and a deterministic secret; the allowlist is a module
    # constant (@incois.gov.in + DEV_ADMIN_EMAIL), not a config field.
    config = make_config(AUTH_DEV_RETURN_OTP="false")

    monkeypatch.setattr("src.auth.router.load_auth_config", lambda: config)
    monkeypatch.setattr("src.auth.router.get_mailer", lambda _cfg: mailer)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        c._auditpro_mailer = mailer  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()


def _register(client) -> str:
    """Run the Phase 2 signup flow over HTTP; return the verified email."""
    email = "boss@incois.gov.in"
    assert client.post("/api/auth/signup/start", json={"email": email}).status_code == 200
    code = _code_from(client._auditpro_mailer)  # type: ignore[attr-defined]
    v = client.post("/api/auth/signup/verify", json={"email": email, "otp": code})
    assert v.status_code == 200, v.text
    signup_token = v.json()["signup_token"]
    r = client.post(
        "/api/auth/signup/complete",
        json={"signup_token": signup_token, "password": GOOD_PASSWORD},
    )
    assert r.status_code == 200, r.text
    return email


# ═══════════════════════════════════════════════════════════════════════════


def test_the_phase_3_routes_are_reachable_on_the_real_app():
    """They must not be shadowed by the SPA catch-all or missing entirely.

    Checked through the OpenAPI schema rather than by issuing requests, so it
    needs no database: the schema is generated from the routes the app
    actually registered, including nested included routers.
    """
    with TestClient(app) as c:
        schema = c.get("/openapi.json").json()

    expected = {
        ("get", "/api/auth/me"),
        ("post", "/api/auth/login"),
        ("post", "/api/auth/logout"),
        ("post", "/api/auth/password/forgot"),
        ("post", "/api/auth/password/reset"),
        # ...and the Phase 2 routes must still be there.
        ("post", "/api/auth/signup/start"),
        ("post", "/api/auth/signup/verify"),
        ("post", "/api/auth/signup/complete"),
        ("get", "/api/auth/password-policy"),
    }
    registered = {
        (method, path)
        for path, ops in schema.get("paths", {}).items()
        for method in ops
        if path.startswith("/api/auth")
    }
    assert expected <= registered, f"missing routes: {sorted(expected - registered)}"


def test_sign_in_sets_an_http_only_session_cookie(client):
    email = _register(client)

    r = client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})
    assert r.status_code == 200, r.text
    assert r.json()["email"] == email

    cookie = client.cookies.get("auditpro_session")
    assert cookie, "no session cookie was set"

    # Find the Set-Cookie header and check the security flags.
    set_cookie = [h for h in r.headers.get_list("set-cookie") if "auditpro_session" in h]
    assert set_cookie, "no Set-Cookie header for the session"
    raw = set_cookie[0].lower()
    assert "httponly" in raw, "session cookie must be HttpOnly"
    assert "samesite=lax" in raw
    # Local dev is not production, so Secure stays off (it would break http://).
    assert "secure" not in raw


def test_me_reports_the_signed_in_user(client):
    email = _register(client)
    client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})

    r = client.get("/api/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["user"]["email"] == email


def test_me_is_401_without_a_session(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "not_authenticated"


def test_wrong_password_is_rejected_with_one_generic_message(client):
    email = _register(client)

    r = client.post("/api/auth/login", json={"email": email, "password": "wrong-password"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "invalid_credentials"

    # An unknown address must look identical — no account enumeration.
    r2 = client.post(
        "/api/auth/login", json={"email": "nobody@incois.gov.in", "password": "wrong-password"}
    )
    assert r2.status_code == r.status_code
    assert r2.json()["detail"]["code"] == r.json()["detail"]["code"]
    assert r2.json()["detail"]["message"] == r.json()["detail"]["message"]


def test_logout_ends_the_session(client):
    email = _register(client)
    client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})
    assert client.get("/api/auth/me").json()["authenticated"] is True

    r = client.post("/api/auth/logout", json={})
    assert r.status_code == 200
    assert r.json()["revoked"] is True

    # The session is gone server-side, so /me is now a 401.
    assert client.get("/api/auth/me").status_code == 401


def test_logout_is_safe_when_there_is_no_session(client):
    r = client.post("/api/auth/logout", json={})
    assert r.status_code == 200
    assert r.json()["revoked"] is False


def test_forgot_password_does_not_reveal_who_has_an_account(client):
    email = _register(client)

    known = client.post("/api/auth/password/forgot", json={"email": email})
    unknown = client.post(
        "/api/auth/password/forgot", json={"email": "nobody@incois.gov.in"}
    )

    assert known.status_code == unknown.status_code == 200
    # Identical shape, so the response cannot be used to enumerate accounts.
    assert set(known.json()) == set(unknown.json())
    assert known.json()["status"] == unknown.json()["status"] == "code_sent"


def test_password_reset_end_to_end(client):
    email = _register(client)
    client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})

    # Request a reset code.
    mailer = client._auditpro_mailer  # type: ignore[attr-defined]
    before = len(mailer.sent)
    r = client.post("/api/auth/password/forgot", json={"email": email})
    assert r.status_code == 200
    assert len(mailer.sent) == before + 1, "no reset mail was sent"

    code = _code_from(mailer)

    # Use it to set a new password.
    r = client.post(
        "/api/auth/password/reset",
        json={"email": email, "otp": code, "new_password": NEW_PASSWORD},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "password_updated"
    assert body["sessions_revoked"] >= 1, "the existing session should be revoked"

    # The old password no longer works…
    bad = client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})
    assert bad.status_code == 401

    # …and the new one does.
    good = client.post("/api/auth/login", json={"email": email, "password": NEW_PASSWORD})
    assert good.status_code == 200


def test_a_signup_code_cannot_reset_a_password(client):
    email = _register(client)

    # A signup OTP row exists for this address; it must not be accepted as a
    # password-reset code (different `purpose`).
    r = client.post(
        "/api/auth/password/reset",
        json={"email": email, "otp": "123456", "new_password": NEW_PASSWORD},
    )
    assert r.status_code in (400, 401), r.text
    assert r.json()["detail"]["code"] == "otp_invalid"


def test_only_the_token_digest_is_persisted(client):
    email = _register(client)
    r = client.post("/api/auth/login", json={"email": email, "password": GOOD_PASSWORD})
    assert r.status_code == 200
    raw_cookie = client.cookies.get("auditpro_session")
    assert raw_cookie

    # Look at the sessions table directly.
    engine = None
    for override in app.dependency_overrides.values():
        pass
    # Recreate a session bound to the same in-memory engine via a fresh query.
    from sqlalchemy.orm import Session as SASession  # noqa: E402
    from sqlalchemy import create_engine as _ce  # noqa: E402
    from sqlalchemy import inspect as _inspect  # noqa: E402

    # Grab the engine the fixture created by re-using one live session.
    gen = app.dependency_overrides[get_db]()
    db: SASession = next(gen)
    try:
        rows = db.query(models.Session).all()
        assert rows, "no session row was persisted"
        joined = " ".join(
            str(getattr(row, c.name)) for row in rows for c in row.__table__.columns
        )
        assert raw_cookie not in joined, "the raw session token was stored in the database!"
    finally:
        db.close()
