import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, UUIDPrimaryKey


class SentItem(UUIDPrimaryKey, Base):
    __tablename__ = "sent_items"
    __table_args__ = (UniqueConstraint("subscription_id", "news_item_id", name="uq_sent_item"),)

    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False
    )
    news_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("news_items.id", ondelete="CASCADE"), nullable=False
    )
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    subscription: Mapped["Subscription"] = relationship(back_populates="sent_items")  # noqa: F821
    news_item: Mapped["NewsItem"] = relationship(back_populates="sent_items")  # noqa: F821
