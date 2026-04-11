"""add topic_embedding to subscriptions

Revision ID: b9c3d4e5f6a7
Revises: a8f2e3b1c4d5
Create Date: 2026-04-11 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "b9c3d4e5f6a7"
down_revision: str | None = "a8f2e3b1c4d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("subscriptions", sa.Column("topic_embedding", Vector(1536), nullable=True))
    op.execute(
        "UPDATE subscriptions SET topic_embedding = canonical_prompt_embedding "
        "WHERE canonical_prompt_embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "topic_embedding")
