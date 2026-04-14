"""Evaluation result model for tracking digest/event quality over time."""

import uuid

from sqlalchemy import Float, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class EvaluationResult(UUIDPrimaryKey, TimestampMixin, Base):
    """Records quality scores from the LLM-as-Judge for each delivery.

    Enables tracking quality trends over time, detecting degradation,
    and producing charts for analysis.

    Usage:
        result = EvaluationResult(
            trace_id="abc123",
            subscription_id=sub.id,
            delivery_type="digest",
            relevance_score=4.0,
            format_score=3.5,
            conciseness_score=5.0,
            overall_score=4.2,
            judge_model="openai/gpt-5.4-nano",
        )
    """

    __tablename__ = "evaluation_results"

    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    delivery_type: Mapped[str] = mapped_column(String(16), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    format_score: Mapped[float] = mapped_column(Float, nullable=False)
    conciseness_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    judge_model: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False, default="PASS")
