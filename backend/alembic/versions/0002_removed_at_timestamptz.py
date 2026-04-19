"""Make source_removal_log.removed_at timezone-aware.

The model declared ``removed_at`` as a bare ``Mapped[datetime]``, which
SQLAlchemy mapped to ``TIMESTAMP WITHOUT TIME ZONE``. Code inserts
``datetime.now(UTC)`` (offset-aware), which asyncpg refuses to bind into
a naive column. Bring the column in line with every other timestamp in
the schema (TIMESTAMPTZ, like ``created_at``).

Revision ID: 0002_removed_at_timestamptz
Revises: 0001_baseline
Create Date: 2026-04-19 02:30:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_removed_at_timestamptz"
down_revision: str | Sequence[str] | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE source_removal_log "
        "ALTER COLUMN removed_at TYPE TIMESTAMP WITH TIME ZONE "
        "USING removed_at AT TIME ZONE 'UTC'"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE source_removal_log "
        "ALTER COLUMN removed_at TYPE TIMESTAMP WITHOUT TIME ZONE "
        "USING removed_at AT TIME ZONE 'UTC'"
    )
