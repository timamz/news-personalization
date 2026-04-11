"""add user_spec to subscriptions and conversation_summary to users

Revision ID: a8f2e3b1c4d5
Revises: 7d2fe63af1fd
Create Date: 2026-04-11 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8f2e3b1c4d5"
down_revision: str | None = "7d2fe63af1fd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("user_spec", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "users",
        sa.Column("conversation_summary", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("users", "conversation_summary")
    op.drop_column("subscriptions", "user_spec")
