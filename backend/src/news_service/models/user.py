from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class User(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "users"

    api_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    timezone: Mapped[str | None] = mapped_column(String(255), nullable=True)
    conversation_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    subscriptions: Mapped[list["Subscription"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )
