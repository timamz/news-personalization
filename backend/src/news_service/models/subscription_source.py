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
            "source_id",
            name="uq_subscription_source_subscription_source",
        ),
    )

    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )

    is_user_specified: Mapped[bool] = mapped_column(default=False, server_default="false")

    subscription: Mapped["Subscription"] = relationship(back_populates="source_links")  # noqa: F821
    source: Mapped["Source"] = relationship(back_populates="subscription_links")  # noqa: F821
