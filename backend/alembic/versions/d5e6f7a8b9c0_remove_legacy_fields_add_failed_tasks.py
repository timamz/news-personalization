"""Remove legacy subscription fields, add failed_tasks table

Remove canonical_prompt, canonical_prompt_embedding, prompt_summary, short_label
from subscriptions. These are superseded by user_spec as the single source of truth.

Add failed_tasks table for dead letter queue (records Celery tasks that failed
after all retries).

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-04-14 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "d5e6f7a8b9c0"
down_revision: str | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- Copy canonical_prompt into user_spec where user_spec is empty ---
    op.execute(
        """
        UPDATE subscriptions
        SET user_spec = '## Topic' || E'\\n' || canonical_prompt
        WHERE (user_spec IS NULL OR user_spec = '')
          AND canonical_prompt IS NOT NULL
          AND canonical_prompt != ''
        """
    )

    # --- Copy canonical_prompt_embedding to topic_embedding where missing ---
    op.execute(
        """
        UPDATE subscriptions
        SET topic_embedding = canonical_prompt_embedding
        WHERE topic_embedding IS NULL
          AND canonical_prompt_embedding IS NOT NULL
        """
    )

    # --- Drop legacy columns ---
    op.drop_column("subscriptions", "canonical_prompt")
    op.drop_column("subscriptions", "canonical_prompt_embedding")
    op.drop_column("subscriptions", "prompt_summary")
    op.drop_column("subscriptions", "short_label")

    # --- Create failed_tasks table ---
    op.create_table(
        "failed_tasks",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("task_name", sa.Text(), nullable=False),
        sa.Column("task_args", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("task_kwargs", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("exception_type", sa.Text(), nullable=False),
        sa.Column("exception_message", sa.Text(), nullable=False),
        sa.Column("traceback", sa.Text(), nullable=False, server_default=""),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("failed_tasks")

    op.add_column(
        "subscriptions",
        sa.Column("short_label", sa.String(30), nullable=False, server_default=""),
    )
    op.add_column(
        "subscriptions",
        sa.Column("prompt_summary", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "subscriptions",
        sa.Column("canonical_prompt_embedding", Vector(1536), nullable=True),
    )
    op.add_column(
        "subscriptions",
        sa.Column("canonical_prompt", sa.Text(), nullable=False, server_default=""),
    )

    # Restore canonical_prompt from user_spec topic line
    op.execute(
        """
        UPDATE subscriptions
        SET canonical_prompt = raw_prompt
        WHERE canonical_prompt = ''
        """
    )
