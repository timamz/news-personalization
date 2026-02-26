"""add_last_digest_scheduled_at

Revision ID: 5a1b71e42001
Revises: 79205137a13b
Create Date: 2026-02-26 18:55:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5a1b71e42001"
down_revision: str | Sequence[str] | None = "79205137a13b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("last_digest_scheduled_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "last_digest_scheduled_at")
