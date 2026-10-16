"""Phase 2 tables: accounts and email OTP challenges.

Column types are deliberately **portable** (String / Text / Integer / Boolean /
DateTime) so the service layer can be exercised against SQLite in-memory in
tests as well as against PostgreSQL. Behaviour that depends on the backend:

* PostgreSQL returns timezone-aware datetimes; SQLite returns naive ones.
  Always read timestamps through :func:`as_utc` and always write
  :func:`utcnow`.

Phase 2 added the signup tables; Phase 3 adds the session table. There is
deliberately still **no** role column, no permission table and no chat-history
table — those are later phases.

Password reset reuses :class:`EmailOtp` with a different ``purpose`` rather than
adding a second code table: the challenge lifecycle (expiry, single use, attempt
cap, resend cooldown, keyed digest) is identical, and the purpose scopes the
HMAC so a signup code can never be replayed against the reset endpoint.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models import Base

__all__ = [
    "EmailOtp",
    "Session",
    "User",
    "as_utc",
    "new_id",
    "utcnow",
    "OTP_PURPOSE_PASSWORD_RESET",
    "OTP_PURPOSE_SIGNUP",
]

#: Purpose recorded on every OTP row. Scopes the HMAC and the lookup, so a code
#: issued for one flow is meaningless in the other.
OTP_PURPOSE_SIGNUP = "signup"
OTP_PURPOSE_PASSWORD_RESET = "password_reset"


def new_id() -> str:
    """Primary key. A plain UUID4 string, so it is portable across backends."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Timezone-aware UTC 'now' — what every timestamp column is written with."""
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """Normalise a timestamp read from any backend to UTC-aware.

    PostgreSQL hands back aware datetimes; SQLite hands back naive ones (having
    dropped the offset). Treating a naive value as UTC is correct because
    :func:`utcnow` is the only thing that ever writes these columns.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class User(Base):
    """An account. Created only after the email has been OTP-verified."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    #: As entered, trimmed. Display/audit only — never used for lookup.
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Canonical (trimmed, lowercased). Unique — this is the identity.
    email_normalized: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    #: Encoded Argon2id hash. Never a plaintext password.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    #: True for every account created by this flow (verification precedes it).
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=func.true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EmailOtp(Base):
    """One OTP challenge. Stores only a keyed digest, never the code."""

    __tablename__ = "email_otp"
    __table_args__ = (Index("ix_email_otp_email_purpose", "email_normalized", "purpose"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email_normalized: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False, default=OTP_PURPOSE_SIGNUP)
    #: HMAC-SHA256 hex over (purpose, email, salt, code) keyed by AUTH_SECRET.
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Per-record random salt, so identical codes hash differently across rows.
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    #: Set once the code is used (or superseded). Enforces single use.
    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: Drives the resend cooldown.
    last_sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resend_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Session(Base):
    """One server-side sign-in.

    The cookie carries an opaque, high-entropy token. Only its HMAC is stored
    here, so read access to the database alone is not enough to replay a live
    session — the same reasoning as storing a password hash rather than a
    password.

    A session ends when any of these is true:
      * ``revoked_at`` is set   — explicit logout (or reset-password revocation);
      * ``expires_at`` has passed — absolute lifetime, never extended;
      * ``last_seen_at`` is older than the idle timeout — inactivity.
    """

    __tablename__ = "sessions"
    # `unique=True` on token_hash already yields a unique constraint (and, in
    # PostgreSQL, a unique index of the same name) through the naming
    # convention — declaring a matching Index() as well would collide by name.
    __table_args__ = (Index("ix_sessions_user_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    #: Owning account. CASCADE keeps the database tidy if a user row is removed.
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    #: HMAC-SHA256 hex of the cookie token. Unique — this is the lookup key.
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: Refreshed on every successful validation; drives the idle timeout.
    #: Deliberately has NO onupdate — it must only move when the session is used,
    #: not when it is revoked.
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: Absolute expiry. Never slid forward — a session has a hard ceiling.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: Set on logout and on password reset. Null means the session is live.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Diagnostics only — never used for authentication decisions.
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
