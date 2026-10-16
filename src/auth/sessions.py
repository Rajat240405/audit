"""Phase 3 — server-side sessions.

How a session works
-------------------

1. On a successful login the server generates an opaque, high-entropy token
   (:func:`new_session_token`) and stores **only its HMAC** in PostgreSQL
   (:class:`src.auth.models.Session`). The cookie carries the token; the
   database carries the digest. Read access to the database alone is therefore
   not enough to resume somebody's session — the same reasoning as storing a
   password hash instead of a password.

2. The cookie is looked up by that digest, so the raw token exists only in the
   user's browser and in the request that carries it.

3. A session stops being valid when **any** of these becomes true:

   * ``revoked_at`` is set — explicit logout, or a password reset revoking
     every session for the account (the right response to a suspected
     compromise);
   * ``expires_at`` has passed — an absolute ceiling that is never extended,
     however active the user is;
   * ``last_seen_at`` is older than ``session_idle_timeout_seconds``.

   The first two are enforced without needing a background job: an expired row
   is simply never returned by :func:`resolve_session`.

This module deliberately contains **no FastAPI imports**. Cookie construction
belongs to the router (``src/auth/router.py``), which keeps this logic
testable on its own.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.auth import models as models_mod
from src.auth.config import AuthConfig

# Renamed on import: the SQLAlchemy session parameter is also called `session`,
# and shadowing it here would be genuinely confusing.
AuthSession = models_mod.Session

__all__ = [
    "AuthSession",
    "create_session",
    "hash_session_token",
    "new_session_token",
    "resolve_session",
    "revoke_all_sessions_for_user",
    "revoke_session",
    "session_is_live",
    "SESSION_TOKEN_BYTES",
]

#: 32 bytes -> 256 bits of entropy. A session token is never guessable.
SESSION_TOKEN_BYTES = 32


def new_session_token() -> str:
    """An opaque session token from the CSPRNG. Never stored, only hashed."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str, *, key: bytes) -> str:
    """HMAC-SHA256 of the cookie token, keyed by a subkey of ``AUTH_SECRET``.

    Domain-separated with a ``session:`` prefix so a session token can never
    collide with an OTP or signup-token digest even under a reused key.
    """
    return hmac.new(key, f"session:{token}".encode(), hashlib.sha256).hexdigest()


def session_is_live(row: AuthSession, config: AuthConfig, now: datetime | None = None) -> bool:
    """True when the row is not revoked, not expired and not idle-timed-out."""
    current = now or models_mod.utcnow()
    if row.revoked_at is not None:
        return False
    expires_at = models_mod.as_utc(row.expires_at)
    if expires_at is None or expires_at <= current:
        return False
    last_seen = models_mod.as_utc(row.last_seen_at)
    if last_seen is not None:
        idle = (current - last_seen).total_seconds()
        if idle > config.session_idle_timeout_seconds:
            return False
    return True


def create_session(
    session: Session,
    *,
    user_id: str,
    config: AuthConfig,
    user_agent: str | None = None,
) -> tuple[str, AuthSession]:
    """Mint a session and commit it. Returns ``(raw_token, row)``.

    The raw token is returned exactly once, to be written to the cookie. It is
    not recoverable afterwards.
    """
    token = new_session_token()
    now = models_mod.utcnow()
    row = AuthSession(
        id=models_mod.new_id(),
        user_id=user_id,
        token_hash=hash_session_token(token, key=config.session_key()),
        created_at=now,
        last_seen_at=now,
        expires_at=now + timedelta(seconds=config.session_ttl_seconds),
        revoked_at=None,
        # Truncated: a header is attacker-controlled and unbounded.
        user_agent=(user_agent or None),
    )
    session.add(row)
    session.commit()
    return token, row


def resolve_session(
    session: Session,
    *,
    token: str | None,
    config: AuthConfig,
    now: datetime | None = None,
) -> AuthSession | None:
    """Look up a live session by its cookie token, or return ``None``.

    Every rejection is intentionally indistinguishable from the outside: an
    unknown token, a revoked one, an expired one and an idle-timed-out one all
    produce ``None``.

    On success ``last_seen_at`` is refreshed, which is what makes the idle
    timeout a *sliding* window rather than a second absolute deadline.
    """
    if not token:
        return None

    digest = hash_session_token(token, key=config.session_key())
    row = (
        session.execute(select(AuthSession).where(AuthSession.token_hash == digest))
        .scalars()
        .first()
    )
    if row is None:
        return None

    current = now or models_mod.utcnow()
    if not session_is_live(row, config, now=current):
        return None

    row.last_seen_at = current
    session.commit()
    return row


def revoke_session(
    session: Session,
    *,
    token: str | None,
    config: AuthConfig,
) -> bool:
    """Revoke one session (logout). Returns False if there was nothing live.

    Idempotent by design: logging out twice, or logging out with an
    already-dead cookie, is not an error.
    """
    if not token:
        return False

    digest = hash_session_token(token, key=config.session_key())
    row = (
        session.execute(select(AuthSession).where(AuthSession.token_hash == digest))
        .scalars()
        .first()
    )
    if row is None or row.revoked_at is not None:
        return False

    row.revoked_at = models_mod.utcnow()
    session.commit()
    return True


def revoke_all_sessions_for_user(session: Session, *, user_id: str) -> int:
    """Revoke every live session for an account. Returns how many were killed.

    Used after a password reset: whoever changed the password may have done so
    because the old one leaked, so every other device must be signed out.
    """
    now = models_mod.utcnow()
    rows = (
        session.execute(
            select(AuthSession).where(
                AuthSession.user_id == user_id,
                AuthSession.revoked_at.is_(None),
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        row.revoked_at = now
    session.commit()
    return len(rows)
