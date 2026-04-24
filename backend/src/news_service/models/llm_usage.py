"""Per-call ledger of every LLM and web-search request served by the service.

Rows are inserted by the global LiteLLM success/failure callback registered
in ``core.llm_usage``. Every agentic or one-shot model call lands here with
provider-reported token counts, a derived USD cost (computed from the
static pricing table in settings), and attribution columns that tie the
call back to a specific agent, subscription, user, or benchmark run.

Example usage::

    row = LLMUsage(
        agent="digest_writer",
        model="openai/gpt-5.4-nano",
        call_type="chat",
        prompt_tokens=1200,
        completion_tokens=340,
        cost_usd=Decimal("0.000520"),
    )
    session.add(row)
"""

import uuid
from decimal import Decimal

from sqlalchemy import Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class LLMUsage(UUIDPrimaryKey, TimestampMixin, Base):
    """One row per completed (or failed) LLM / embedding / search call."""

    __tablename__ = "llm_usage"

    agent: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    provider: Mapped[str] = mapped_column(Text, nullable=False, default="litellm")
    model: Mapped[str] = mapped_column(Text, nullable=False)
    call_type: Mapped[str] = mapped_column(Text, nullable=False)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 10), nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
