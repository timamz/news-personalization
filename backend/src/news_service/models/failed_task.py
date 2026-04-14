"""Dead letter table — records Celery tasks that failed after all retries."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from news_service.models.base import Base, TimestampMixin, UUIDPrimaryKey


class FailedTask(UUIDPrimaryKey, TimestampMixin, Base):
    """Stores failed Celery tasks for inspection and manual retry.

    Example usage::

        failed = FailedTask(
            task_name="news_service.tasks.deliver_digest.deliver_digest",
            task_args='["abc-123"]',
            task_kwargs="{}",
            exception_type="ValueError",
            exception_message="LLM returned empty response",
            traceback="Traceback (most recent call last): ...",
            retries=3,
        )
        session.add(failed)
    """

    __tablename__ = "failed_tasks"

    task_name: Mapped[str] = mapped_column(Text, nullable=False)
    task_args: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    task_kwargs: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    exception_type: Mapped[str] = mapped_column(Text, nullable=False)
    exception_message: Mapped[str] = mapped_column(Text, nullable=False)
    traceback: Mapped[str] = mapped_column(Text, nullable=False, default="")
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
