"""Add llm_usage ledger table.

Stores one row per completed or failed LLM / embedding call served by
the service. Rows are written from the global LiteLLM success/failure
callback so every model dispatch is accounted for regardless of which
agent path produced it.

Revision ID: 0004_llm_usage
Revises: 0003_user_delivery_webhook_url
Create Date: 2026-04-24 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004_llm_usage"
down_revision: str | Sequence[str] | None = "0003_user_delivery_webhook_url"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("agent", sa.Text(), nullable=True),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("subscription_id", UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False, server_default="litellm"),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("call_type", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(14, 10), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_llm_usage_agent", "llm_usage", ["agent"])
    op.create_index("ix_llm_usage_run_id", "llm_usage", ["run_id"])
    op.create_index("ix_llm_usage_user_id", "llm_usage", ["user_id"])
    op.create_index("ix_llm_usage_subscription_id", "llm_usage", ["subscription_id"])
    op.create_index("ix_llm_usage_created_at", "llm_usage", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_created_at", table_name="llm_usage")
    op.drop_index("ix_llm_usage_subscription_id", table_name="llm_usage")
    op.drop_index("ix_llm_usage_user_id", table_name="llm_usage")
    op.drop_index("ix_llm_usage_run_id", table_name="llm_usage")
    op.drop_index("ix_llm_usage_agent", table_name="llm_usage")
    op.drop_table("llm_usage")
