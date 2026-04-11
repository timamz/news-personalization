"""Pipeline event model for observability and trace replay."""

import uuid

from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class PipelineEvent(UUIDPrimaryKey, TimestampMixin, Base):
    """Records a single step in a pipeline execution for observability.

    Each pipeline run shares a trace_id, enabling full replay of
    what happened during digest generation, event assessment, or
    source discovery.

    Usage:
        event = PipelineEvent(
            trace_id="abc123",
            pipeline_type="digest",
            agent_name="DigestPlanner",
            event_type="llm_call",
            subscription_id=sub.id,
            input_summary={"user_spec": "..."},
            output_summary={"plan": "..."},
            token_usage={"prompt": 500, "completion": 120},
            latency_ms=1200,
        )
    """

    __tablename__ = "pipeline_events"

    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_type: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    input_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    token_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
