"""Add a paused_at timestamp to subscriptions.

A non-NULL ``paused_at`` marks a subscription as temporarily stopped
by the user. All polling, scheduling, and delivery code must skip
stopped subscriptions, and the active-subscription cap must not count
them. This is distinct from ``is_active`` (the soft-delete marker):
metadata is preserved and the user can resume later.

Revision ID: 0005_subscription_paused_at
Revises: 0004_llm_usage
Create Date: 2026-05-15 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_subscription_paused_at"
down_revision: str | Sequence[str] | None = "0004_llm_usage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "paused_at")
