"""make_subscription_schedule_optional

Revision ID: c3bb4f1a0e66
Revises: 41d9fef8f3a8
Create Date: 2026-02-27 16:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3bb4f1a0e66"
down_revision: str | Sequence[str] | None = "41d9fef8f3a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "subscriptions",
        "schedule_cron",
        existing_type=sa.String(length=100),
        nullable=True,
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE subscriptions
        SET schedule_cron = '0 8 * * *'
        WHERE schedule_cron IS NULL
        """
    )
    op.alter_column(
        "subscriptions",
        "schedule_cron",
        existing_type=sa.String(length=100),
        nullable=False,
    )
