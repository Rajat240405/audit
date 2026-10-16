"""signup: users + email_otp

Phase 2 — signup and email OTP verification.

Depends on 0001_create_app_meta (the Phase 1 foundation) so a fresh database
always has the app_meta table first.

Only the tables required for signup, email verification and account creation
are created. Deliberately absent — later phases:
  * sessions / login state          (Phase 3)
  * roles and role assignment       (Phase 4)
  * password-reset columns          (Phase 3)
  * conversations / messages        (Phase 5)

Notes on the schema:
  * ``users.email_normalized`` is UNIQUE — this is the real duplicate guard and
    is what makes a concurrent double signup fail rather than create two rows.
  * ``email_otp.code_hash`` stores an HMAC-SHA256 digest, never the code.
  * ``email_otp.salt`` is per-record, so the same code hashes differently on
    every row.
  * Types are PostgreSQL-compatible and portable; ``server_default`` uses
    ``now()`` / ``false()`` / ``true()`` so SQLite-based service tests behave
    the same way.

Revision ID: 0002_signup_email_otp
Revises: 0001_create_app_meta
Create Date: 2026-10-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_signup_email_otp"
down_revision: str | None = "0001_create_app_meta"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("email_normalized", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "email_verified",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email_normalized", name=op.f("uq_users_email_normalized")),
    )
    op.create_index(op.f("ix_users_email_normalized"), "users", ["email_normalized"])

    op.create_table(
        "email_otp",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email_normalized", sa.String(length=255), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("salt", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_sent_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resend_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_email_otp")),
    )
    op.create_index(
        "ix_email_otp_email_purpose", "email_otp", ["email_normalized", "purpose"]
    )


def downgrade() -> None:
    op.drop_index("ix_email_otp_email_purpose", table_name="email_otp")
    op.drop_table("email_otp")
    op.drop_index(op.f("ix_users_email_normalized"), table_name="users")
    op.drop_table("users")
