"""add_event_subscriptions

Revision ID: d62e4e1b6f7a
Revises: c3bb4f1a0e66
Create Date: 2026-03-03 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d62e4e1b6f7a"
down_revision: str | Sequence[str] | None = "c3bb4f1a0e66"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("delivery_mode", sa.String(length=16), nullable=True),
    )
    op.execute(
        """
        UPDATE subscriptions
        SET delivery_mode = 'digest'
        WHERE delivery_mode IS NULL
        """
    )
    op.alter_column(
        "subscriptions",
        "delivery_mode",
        existing_type=sa.String(length=16),
        nullable=False,
    )

    op.add_column("news_items", sa.Column("event_title", sa.Text(), nullable=True))
    op.add_column("news_items", sa.Column("event_summary", sa.Text(), nullable=True))
    op.add_column(
        "news_items",
        sa.Column("event_starts_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("news_items", "event_starts_at")
    op.drop_column("news_items", "event_summary")
    op.drop_column("news_items", "event_title")
    op.drop_column("subscriptions", "delivery_mode")
