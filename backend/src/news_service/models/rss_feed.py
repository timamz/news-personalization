from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class RssFeed(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "rss_feeds"

    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    topic_tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    topic_embedding = mapped_column(Vector(1536), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    subscriber_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    news_items: Mapped[list["NewsItem"]] = relationship(  # noqa: F821
        back_populates="feed", cascade="all, delete-orphan"
    )
    subscription_links: Mapped[list["SubscriptionSource"]] = relationship(  # noqa: F821
        back_populates="feed",
        cascade="all, delete-orphan",
    )
