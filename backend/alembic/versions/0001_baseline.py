"""Baseline schema: create everything from the current SQLAlchemy models.

Replaces the entire accumulated migration history. Runs against an empty
database and installs the full schema as declared in
``news_service.models`` in a single pass. The pgvector extension is
installed first because several columns use the ``Vector`` type.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-04-19 12:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

from news_service.models import Base

revision: str = "0001_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
