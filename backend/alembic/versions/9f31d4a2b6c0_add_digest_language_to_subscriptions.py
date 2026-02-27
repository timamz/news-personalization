"""add_digest_language_to_subscriptions

Revision ID: 9f31d4a2b6c0
Revises: 5a1b71e42001
Create Date: 2026-02-27 13:24:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f31d4a2b6c0"
down_revision: str | Sequence[str] | None = "5a1b71e42001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("digest_language", sa.String(length=16), nullable=True),
    )
    op.execute(
        """
        UPDATE subscriptions
        SET digest_language = CASE
            WHEN raw_prompt ~ '[А-Яа-яЁё]' THEN 'ru'
            ELSE 'en'
        END
        WHERE digest_language IS NULL
        """
    )
    op.alter_column(
        "subscriptions",
        "digest_language",
        existing_type=sa.String(length=16),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "digest_language")
