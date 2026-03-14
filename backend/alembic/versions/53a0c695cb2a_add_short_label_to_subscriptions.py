"""add_short_label_to_subscriptions

Revision ID: 53a0c695cb2a
Revises: cc2e4b7a11d5
Create Date: 2026-03-14 22:01:14.312034

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '53a0c695cb2a'
down_revision: Union[str, Sequence[str], None] = 'cc2e4b7a11d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'subscriptions',
        sa.Column('short_label', sa.String(length=30), nullable=False, server_default=''),
    )
    op.alter_column('subscriptions', 'short_label', server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('subscriptions', 'short_label')
