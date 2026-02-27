import uuid

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class SubscriptionSource(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "subscription_sources"
    __table_args__ = (
        UniqueConstraint(
            "subscription_id",
            "feed_id",
            name="uq_subscription_source_subscription_feed",
        ),
    )

    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    feed_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rss_feeds.id", ondelete="CASCADE"),
        nullable=False,
    )

    subscription: Mapped["Subscription"] = relationship(back_populates="source_links")  # noqa: F821
    feed: Mapped["RssFeed"] = relationship(back_populates="subscription_links")  # noqa: F821
