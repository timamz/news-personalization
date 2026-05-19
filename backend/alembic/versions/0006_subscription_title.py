"""Add title column to subscriptions.

Short human-readable label authored by the conversational agent at
subscription creation time. Used for display in subscription summaries
so the agent can identify subs by name rather than by parsing user_spec.

Revision ID: 0006_subscription_title
Revises: 0005_subscription_paused_at
Create Date: 2026-05-19 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_subscription_title"
down_revision: str | Sequence[str] | None = "0005_subscription_paused_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("title", sa.String(120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "title")
