import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class Subscription(UUIDPrimaryKey, TimestampMixin, Base):
    """User subscription for news monitoring.

    ``user_spec`` is the single source of truth for every LLM-facing field:
    topic, preferences, format guidance, exclusions, reflector observations.
    It is a markdown document with ``## Topic``, ``## Preferences`` and
    related sections; pipelines read it directly or extract pieces via
    ``extract_topic`` / ``parse_user_spec`` from ``models.user_spec``.

    All remaining columns are dispatch/retrieval concerns that do not
    require LLM interpretation: ``delivery_mode``, ``schedule_cron``,
    ``digest_language``, ``delivery_webhook_url``, ``topic_embedding``,
    ``is_active``, and the two ``last_*_at`` bookkeeping timestamps.

    Example usage::

        sub = Subscription(
            user_id=user.id,
            user_spec="## Topic\\nAI news daily\\n\\n## Preferences\\nBrief summary",
            delivery_mode="digest",
            schedule_cron="0 8 * * *",
            digest_language="en",
        )
    """

    __tablename__ = "subscriptions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    topic_embedding = mapped_column(Vector(1536), nullable=True)
    user_spec: Mapped[str] = mapped_column(Text, nullable=False, default="")
    delivery_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="digest")
    schedule_cron: Mapped[str | None] = mapped_column(String(100), nullable=True)
    digest_language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    delivery_webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_digest_scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_reflected_at: Mapped[datetime | None] = mapped_column(
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
