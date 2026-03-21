"""drop raw_prompt_embedding from subscriptions

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-22 12:01:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("subscriptions", "raw_prompt_embedding")


def downgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("raw_prompt_embedding", Vector(1536), nullable=True),
    )
