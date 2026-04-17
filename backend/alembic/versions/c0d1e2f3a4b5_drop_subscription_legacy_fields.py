"""Drop subscriptions.raw_prompt and subscriptions.format_instructions

Folds any ``format_instructions`` content into the ``user_spec`` markdown
under a ``## Preferences`` section before dropping the columns, so no
existing user guidance is lost. ``raw_prompt`` is dropped unconditionally
since its content is always equal-or-subset of ``user_spec`` topic.

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
Create Date: 2026-04-17 19:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c0d1e2f3a4b5"
down_revision: str | None = "b9c0d1e2f3a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE subscriptions
        SET user_spec = CASE
            WHEN COALESCE(user_spec, '') = '' THEN
                '## Topic' || chr(10) || COALESCE(raw_prompt, '')
                || chr(10) || chr(10) || '## Preferences' || chr(10)
                || COALESCE(NULLIF(format_instructions, ''), 'brief summary')
            WHEN user_spec NOT LIKE '%## Preferences%' THEN
                user_spec || chr(10) || chr(10) || '## Preferences' || chr(10)
                || COALESCE(NULLIF(format_instructions, ''), 'brief summary')
            ELSE user_spec
        END
        """
    )
    op.drop_column("subscriptions", "raw_prompt")
    op.drop_column("subscriptions", "format_instructions")


def downgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("raw_prompt", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "subscriptions",
        sa.Column(
            "format_instructions",
            sa.Text(),
            nullable=False,
            server_default="brief summary",
        ),
    )
    op.alter_column("subscriptions", "raw_prompt", server_default=None)
    op.alter_column("subscriptions", "format_instructions", server_default=None)
