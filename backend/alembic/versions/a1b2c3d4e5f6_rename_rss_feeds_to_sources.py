"""rename rss_feeds to sources, feed_id to source_id

Revision ID: a1b2c3d4e5f6
Revises: 7d2fe63af1fd
Create Date: 2026-03-22 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "7d2fe63af1fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table("rss_feeds", "sources")

    op.alter_column("news_items", "feed_id", new_column_name="source_id")
    op.alter_column("subscription_sources", "feed_id", new_column_name="source_id")

    op.drop_constraint(
        "uq_subscription_source_subscription_feed",
        "subscription_sources",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_subscription_source_subscription_source",
        "subscription_sources",
        ["subscription_id", "source_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_subscription_source_subscription_source",
        "subscription_sources",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_subscription_source_subscription_feed",
        "subscription_sources",
        ["subscription_id", "source_id"],
    )

    op.alter_column("subscription_sources", "source_id", new_column_name="feed_id")
    op.alter_column("news_items", "source_id", new_column_name="feed_id")

    op.rename_table("sources", "rss_feeds")
