import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, UniqueConstraint
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

    contributed_last_30_digests: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    contribution_rate: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )
    digests_since_last_contribution: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    item_cosine_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    item_cosine_p90: Mapped[float | None] = mapped_column(Float, nullable=True)
    item_cosine_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    stats_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    subscription: Mapped["Subscription"] = relationship(back_populates="source_links")  # noqa: F821
    source: Mapped["Source"] = relationship(back_populates="subscription_links")  # noqa: F821
