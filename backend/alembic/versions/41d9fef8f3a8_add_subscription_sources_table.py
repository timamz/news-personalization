"""add_subscription_sources_table

Revision ID: 41d9fef8f3a8
Revises: 9f31d4a2b6c0
Create Date: 2026-02-27 16:05:00.000000

"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "41d9fef8f3a8"
down_revision: str | Sequence[str] | None = "9f31d4a2b6c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "subscription_sources",
        sa.Column("subscription_id", sa.UUID(), nullable=False),
        sa.Column("feed_id", sa.UUID(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["feed_id"], ["rss_feeds.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subscription_id",
            "feed_id",
            name="uq_subscription_source_subscription_feed",
        ),
    )

    # Backfill existing subscriptions with currently active sources to keep
    # behavior stable after introducing fixed source sets.
    bind = op.get_bind()
    subscription_ids = list(bind.execute(sa.text("SELECT id FROM subscriptions")).scalars())
    active_feed_ids = list(
        bind.execute(sa.text("SELECT id FROM rss_feeds WHERE is_active = TRUE")).scalars()
    )
    if not subscription_ids or not active_feed_ids:
        return

    rows: list[dict[str, object]] = []
    now = datetime.now(UTC)
    for subscription_id in subscription_ids:
        for feed_id in active_feed_ids:
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "created_at": now,
                    "subscription_id": subscription_id,
                    "feed_id": feed_id,
                }
            )

    op.bulk_insert(
        sa.table(
            "subscription_sources",
            sa.column("id", sa.UUID()),
            sa.column("created_at", sa.DateTime(timezone=True)),
            sa.column("subscription_id", sa.UUID()),
            sa.column("feed_id", sa.UUID()),
        ),
        rows,
    )


def downgrade() -> None:
    op.drop_table("subscription_sources")
