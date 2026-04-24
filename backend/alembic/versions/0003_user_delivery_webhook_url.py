"""Add a default delivery webhook URL to users.

The backend stores a generic per-user webhook target so frontends can
register their delivery endpoint once and the digest / notification
pipelines can fall back to it when a subscription does not carry its own
override.

Revision ID: 0003_user_delivery_webhook_url
Revises: 0002_removed_at_timestamptz
Create Date: 2026-04-20 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_user_delivery_webhook_url"
down_revision: str | Sequence[str] | None = "0002_removed_at_timestamptz"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("delivery_webhook_url", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "delivery_webhook_url")
