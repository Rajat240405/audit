"""create app_meta

Phase 1 PostgreSQL foundation.

Deliberately the ONLY table in this revision: it exists so the foundation can be
verified end to end (connect -> upgrade -> insert/select -> downgrade) without
pre-empting Phase 2 (users) or Phase 5 (conversations / messages).

The primary key is named explicitly (``pk_app_meta``) to match the naming
convention declared on ``src.db.models.Base`` — without a stable name Alembic
cannot reliably diff or drop the constraint in a later revision.

Revision ID: 0001_create_app_meta
Revises:
Create Date: 2026-10-14

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_create_app_meta"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_meta",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_app_meta")),
    )


def downgrade() -> None:
    op.drop_table("app_meta")
