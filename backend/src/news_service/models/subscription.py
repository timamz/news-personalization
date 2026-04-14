import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class Subscription(UUIDPrimaryKey, TimestampMixin, Base):
    """User subscription for news monitoring.

    The user_spec field is the single source of truth for user intent. It is a
    markdown document with sections like ## Topic, ## Sources, ## Preferences.
    All pipeline agents read user_spec to understand what the user wants.

    The raw_prompt preserves the user's original input verbatim for audit/history.
    The topic_embedding is derived from user_spec and used for vector similarity
    queries in the digest candidate pipeline.

    Example usage::

        sub = Subscription(
            user_id=user.id,
            raw_prompt="AI news daily",
            user_spec="## Topic\\nAI news daily",
            delivery_mode="digest",
            schedule_cron="0 8 * * *",
            digest_language="en",
        )
    """

    __tablename__ = "subscriptions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    topic_embedding = mapped_column(Vector(1536), nullable=True)
    user_spec: Mapped[str] = mapped_column(Text, nullable=False, default="")
    delivery_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="digest")
    schedule_cron: Mapped[str | None] = mapped_column(String(100), nullable=True)
    format_instructions: Mapped[str] = mapped_column(Text, nullable=False, default="brief summary")
    digest_language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    delivery_webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_digest_scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    user: Mapped["User"] = relationship(back_populates="subscriptions")  # noqa: F821
    sent_items: Mapped[list["SentItem"]] = relationship(  # noqa: F821
        back_populates="subscription", cascade="all, delete-orphan"
    )
    source_links: Mapped[list["SubscriptionSource"]] = relationship(  # noqa: F821
        back_populates="subscription",
        cascade="all, delete-orphan",
    )
