"""add_subscription_query_embeddings

Revision ID: 0b3e8f9c7a11
Revises: f1c7b1d3a8e2
Create Date: 2026-03-06 18:10:00.000000

"""

from collections.abc import Sequence

import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0b3e8f9c7a11"
down_revision: str | Sequence[str] | None = "f1c7b1d3a8e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column(
            "raw_prompt_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )
    op.add_column(
        "subscriptions",
        sa.Column(
            "topics_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "topics_embedding")
    op.drop_column("subscriptions", "raw_prompt_embedding")
