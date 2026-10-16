"""auth sessions: sessions table (Phase 3)

Phase 3 — sign in, sign out, and password reset by email OTP.

Depends on 0002_signup_email_otp (Phase 2) so a fresh database always has
app_meta and users/email_otp first.

What this migration adds
------------------------
* ``sessions`` — server-side sign-in state.

What it deliberately does NOT add
---------------------------------
* No new table for password-reset codes. Reset reuses ``email_otp`` with
  ``purpose = 'password_reset'``, so the challenge lifecycle (expiry, single
  use, attempt cap, keyed digest) is defined in exactly one place and the
  ``purpose`` column already scopes the HMAC.
* No role, permission or chat-history table (Phase 4 / Phase 5).
* No change to ``users`` or ``email_otp`` — both already carry everything
  Phase 3 needs.

Nothing here is destructive: it creates one table and two indexes. No existing
column is altered or dropped, so applying it to a database that already has
real signup data preserves every row.

Revision ID: 0003_auth_sessions_password_reset
Revises: 0002_signup_email_otp
Create Date: 2026-10-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_auth_sessions_reset"
down_revision: str | None = "0002_signup_email_otp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # No onupdate: last_seen_at must move only when the session is used,
        # not when it is revoked. The service sets it explicitly.
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sessions")),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_sessions_user_id"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("token_hash", name=op.f("uq_sessions_token_hash")),
    )
    # token_hash is UNIQUE via the constraint above, which PostgreSQL
    # implements as a unique index of the same name. Creating a second index
    # here would fail with "relation already exists".
    op.create_index(op.f("ix_sessions_user_id"), "sessions", ["user_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_sessions_user_id"), table_name="sessions")
    op.drop_table("sessions")
