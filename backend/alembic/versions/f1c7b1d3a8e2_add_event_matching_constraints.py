"""add_event_matching_constraints

Revision ID: f1c7b1d3a8e2
Revises: d62e4e1b6f7a
Create Date: 2026-03-04 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f1c7b1d3a8e2"
down_revision: str | Sequence[str] | None = "d62e4e1b6f7a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("event_matching_mode", sa.String(length=32), nullable=True),
    )
    op.execute(
        """
        UPDATE subscriptions
        SET event_matching_mode = 'basic'
        WHERE event_matching_mode IS NULL
        """
    )
    op.alter_column(
        "subscriptions",
        "event_matching_mode",
        existing_type=sa.String(length=32),
        nullable=False,
    )

    op.add_column(
        "subscriptions",
        sa.Column(
            "event_constraints",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE subscriptions
        SET event_constraints = '[]'::jsonb
        WHERE event_constraints IS NULL
        """
    )
    op.alter_column(
        "subscriptions",
        "event_constraints",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "event_constraints")
    op.drop_column("subscriptions", "event_matching_mode")
