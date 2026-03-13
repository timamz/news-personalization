"""remove label fields add prompt summaries

Revision ID: 7c4f6f5d8e21
Revises: 3b7d5f2a91c4
Create Date: 2026-03-13 23:40:00.000000

"""

from collections.abc import Sequence

import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c4f6f5d8e21"
down_revision: str | Sequence[str] | None = "3b7d5f2a91c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("prompt_summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "rss_feeds",
        sa.Column("source_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "rss_feeds",
        sa.Column(
            "source_description_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE subscriptions
            SET prompt_summary = NULLIF(BTRIM(raw_prompt), '')
            WHERE prompt_summary IS NULL
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE subscriptions
            SET prompt_summary = 'News subscription'
            WHERE prompt_summary IS NULL OR BTRIM(prompt_summary) = ''
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE rss_feeds
            SET source_description = CASE
                WHEN BTRIM(COALESCE(title, '')) <> '' THEN title || ' (' || url || ')'
                ELSE url
            END
            WHERE source_description IS NULL
            """
        )
    )

    op.alter_column("subscriptions", "prompt_summary", nullable=False)
    op.alter_column("rss_feeds", "source_description", nullable=False)

    op.drop_column("subscriptions", "topics_embedding")
    op.drop_column("subscriptions", "topics")
    op.drop_column("rss_feeds", "topic_embedding")
    op.drop_column("rss_feeds", "topic_tags")


def downgrade() -> None:
    op.add_column(
        "rss_feeds",
        sa.Column("topic_tags", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )
    op.add_column(
        "rss_feeds",
        sa.Column(
            "topic_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )
    op.add_column(
        "subscriptions",
        sa.Column("topics", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )
    op.add_column(
        "subscriptions",
        sa.Column(
            "topics_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )

    op.drop_column("rss_feeds", "source_description_embedding")
    op.drop_column("rss_feeds", "source_description")
    op.drop_column("subscriptions", "prompt_summary")

    op.alter_column("subscriptions", "topics", server_default=None)
    op.alter_column("rss_feeds", "topic_tags", server_default=None)
