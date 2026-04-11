"""add pipeline_events and evaluation_results tables

Revision ID: c4d5e6f7a8b9
Revises: b9c3d4e5f6a7
Create Date: 2026-04-11 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "b9c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pipeline_events",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False, index=True),
        sa.Column("pipeline_type", sa.String(32), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column(
            "subscription_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=True,
            index=True,
        ),
        sa.Column("input_summary", sa.dialects.postgresql.JSON, nullable=True),
        sa.Column("output_summary", sa.dialects.postgresql.JSON, nullable=True),
        sa.Column("token_usage", sa.dialects.postgresql.JSON, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("model_name", sa.String(64), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False, index=True),
        sa.Column(
            "subscription_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=False,
            index=True,
        ),
        sa.Column("delivery_type", sa.String(16), nullable=False),
        sa.Column("relevance_score", sa.Float, nullable=False),
        sa.Column("coverage_score", sa.Float, nullable=False),
        sa.Column("dedup_score", sa.Float, nullable=False),
        sa.Column("quality_score", sa.Float, nullable=False),
        sa.Column("overall_score", sa.Float, nullable=False),
        sa.Column("judge_model", sa.String(64), nullable=False),
        sa.Column("verdict", sa.String(16), nullable=False, server_default="PASS"),
    )


def downgrade() -> None:
    op.drop_table("evaluation_results")
    op.drop_table("pipeline_events")
