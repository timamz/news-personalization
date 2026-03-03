import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select

from news_service.db.session import get_task_session
from news_service.models.subscription import Subscription
from news_service.services.scheduler import is_schedule_due
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

DELIVER_DIGEST_TASK = "news_service.tasks.deliver_digest.deliver_digest"


@celery_app.task(name="news_service.tasks.schedule_digests.schedule_due_digests")
def schedule_due_digests() -> dict:
    return asyncio.run(_schedule_due_digests())


async def _schedule_due_digests(now: datetime | None = None) -> dict:
    current_time = _truncate_to_minute(now or datetime.now(UTC))
    queued = 0
    invalid_cron = 0

    async with get_task_session() as session:
        result = await session.execute(
            select(Subscription).where(
                Subscription.is_active.is_(True),
                Subscription.delivery_mode == "digest",
            )
        )
        subscriptions = list(result.scalars().all())

        for subscription in subscriptions:
            if subscription.delivery_mode != "digest":
                continue
            if not subscription.schedule_cron:
                continue

            last_run_at = subscription.last_digest_scheduled_at or subscription.created_at
            try:
                due = is_schedule_due(
                    subscription.schedule_cron,
                    last_run_at=last_run_at,
                    now=current_time,
                )
            except ValueError:
                invalid_cron += 1
                logger.exception(
                    "Invalid cron expression '%s' for subscription %s",
                    subscription.schedule_cron,
                    subscription.id,
                    extra={"subscription_id": str(subscription.id)},
                )
                continue

            if not due:
                continue

            celery_app.send_task(DELIVER_DIGEST_TASK, args=[str(subscription.id)])
            subscription.last_digest_scheduled_at = current_time
            queued += 1

        await session.commit()

    logger.info(
        "Digest scheduling scan complete: checked=%d queued=%d invalid_cron=%d",
        len(subscriptions),
        queued,
        invalid_cron,
    )
    return {"checked": len(subscriptions), "queued": queued, "invalid_cron": invalid_cron}


def _truncate_to_minute(value: datetime) -> datetime:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.replace(second=0, microsecond=0)
