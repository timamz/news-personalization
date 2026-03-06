import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class Subscription(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "subscriptions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_prompt_embedding = mapped_column(Vector(1536), nullable=True)
    topics: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    topics_embedding = mapped_column(Vector(1536), nullable=True)
    delivery_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="digest")
    event_matching_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="basic")
    event_constraints: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
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
