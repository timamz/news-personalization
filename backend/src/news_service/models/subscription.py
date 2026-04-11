import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class Subscription(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "subscriptions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    canonical_prompt_embedding = mapped_column(Vector(1536), nullable=True)
    topic_embedding = mapped_column(Vector(1536), nullable=True)
    prompt_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    short_label: Mapped[str] = mapped_column(String(30), nullable=False, default="")
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
