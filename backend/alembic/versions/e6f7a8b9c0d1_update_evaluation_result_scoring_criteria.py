"""Update evaluation_results scoring: replace coverage+dedup+quality with format+conciseness

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-04-15 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: str | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evaluation_results",
        sa.Column("format_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_results",
        sa.Column("conciseness_score", sa.Float(), nullable=True),
    )

    # Migrate existing data: map old scores to new columns
    op.execute(
        """
        UPDATE evaluation_results
        SET format_score = quality_score,
            conciseness_score = dedup_score
        """
    )

    op.alter_column("evaluation_results", "format_score", nullable=False)
    op.alter_column("evaluation_results", "conciseness_score", nullable=False)

    # Recalculate overall as average of 3 scores
    op.execute(
        """
        UPDATE evaluation_results
        SET overall_score = ROUND(
            (relevance_score + format_score + conciseness_score) / 3.0, 2
        )
        """
    )

    op.drop_column("evaluation_results", "coverage_score")
    op.drop_column("evaluation_results", "dedup_score")
    op.drop_column("evaluation_results", "quality_score")


def downgrade() -> None:
    op.add_column(
        "evaluation_results",
        sa.Column("quality_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_results",
        sa.Column("dedup_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_results",
        sa.Column("coverage_score", sa.Float(), nullable=True),
    )

    op.execute(
        """
        UPDATE evaluation_results
        SET quality_score = format_score,
            dedup_score = conciseness_score,
            coverage_score = 3.0
        """
    )

    op.alter_column("evaluation_results", "quality_score", nullable=False)
    op.alter_column("evaluation_results", "dedup_score", nullable=False)
    op.alter_column("evaluation_results", "coverage_score", nullable=False)

    op.drop_column("evaluation_results", "conciseness_score")
    op.drop_column("evaluation_results", "format_score")
