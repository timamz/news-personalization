"""add user timezone

Revision ID: 3b7d5f2a91c4
Revises: 0b3e8f9c7a11
Create Date: 2026-03-13 21:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3b7d5f2a91c4"
down_revision: str | Sequence[str] | None = "0b3e8f9c7a11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("timezone", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "timezone")
