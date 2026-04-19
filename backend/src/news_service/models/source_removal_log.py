"""Tracks sources removed by the reflector so discovery avoids re-adding them.

The reflector hard-deletes SubscriptionSource rows when removing dead sources.
This log preserves the removal history so the discovery pipeline can see what
was recently removed and why, letting the LLM decide whether to re-discover.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class SourceRemovalLog(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "source_removal_log"

    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    removed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    removal_reason: Mapped[str] = mapped_column(Text, default="", server_default="")
