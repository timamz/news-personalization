"""Add subscription_source stats columns

Revision ID: bad8baaa0487
Revises: c0d1e2f3a4b5
Create Date: 2026-04-18 23:14:02.713845

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "bad8baaa0487"
down_revision: str | Sequence[str] | None = "c0d1e2f3a4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscription_sources",
        sa.Column(
            "contributed_last_30_digests",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "subscription_sources",
        sa.Column(
            "contribution_rate",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "subscription_sources",
        sa.Column(
            "digests_since_last_contribution",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "subscription_sources",
        sa.Column("item_cosine_p50", sa.Float(), nullable=True),
    )
    op.add_column(
        "subscription_sources",
        sa.Column("item_cosine_p90", sa.Float(), nullable=True),
    )
    op.add_column(
        "subscription_sources",
        sa.Column("item_cosine_std", sa.Float(), nullable=True),
    )
    op.add_column(
        "subscription_sources",
        sa.Column("stats_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscription_sources", "stats_updated_at")
    op.drop_column("subscription_sources", "item_cosine_std")
    op.drop_column("subscription_sources", "item_cosine_p90")
    op.drop_column("subscription_sources", "item_cosine_p50")
    op.drop_column("subscription_sources", "digests_since_last_contribution")
    op.drop_column("subscription_sources", "contribution_rate")
    op.drop_column("subscription_sources", "contributed_last_30_digests")
