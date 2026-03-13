"""add canonical prompt to subscriptions

Revision ID: cc2e4b7a11d5
Revises: 7c4f6f5d8e21
Create Date: 2026-03-14 00:40:00.000000

"""

from collections.abc import Sequence

import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cc2e4b7a11d5"
down_revision: str | Sequence[str] | None = "7c4f6f5d8e21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("canonical_prompt", sa.Text(), nullable=True),
    )
    op.add_column(
        "subscriptions",
        sa.Column(
            "canonical_prompt_embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE subscriptions
            SET canonical_prompt = raw_prompt,
                canonical_prompt_embedding = raw_prompt_embedding
            WHERE canonical_prompt IS NULL
            """
        )
    )
    op.alter_column("subscriptions", "canonical_prompt", nullable=False)


def downgrade() -> None:
    op.drop_column("subscriptions", "canonical_prompt_embedding")
    op.drop_column("subscriptions", "canonical_prompt")
