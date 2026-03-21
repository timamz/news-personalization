import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, UUIDPrimaryKey


class NewsItem(UUIDPrimaryKey, Base):
    __tablename__ = "news_items"

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    embedding = mapped_column(Vector(1536), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    news_source: Mapped["Source"] = relationship(back_populates="news_items")  # noqa: F821
    sent_items: Mapped[list["SentItem"]] = relationship(  # noqa: F821
        back_populates="news_item", cascade="all, delete-orphan"
    )
