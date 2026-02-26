import logging

from celery.schedules import crontab

from news_service.models.subscription import Subscription
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def parse_cron_to_celery(cron_expr: str) -> crontab:
    """Convert a 5-field cron expression to a Celery crontab."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): {cron_expr}")

    minute, hour, day_of_month, month_of_year, day_of_week = parts
    return crontab(
        minute=minute,
        hour=hour,
        day_of_month=day_of_month,
        month_of_year=month_of_year,
        day_of_week=day_of_week,
    )


def register_delivery_schedule(subscription: Subscription) -> None:
    """Add a Celery Beat schedule entry for a subscription's digest delivery."""
    schedule_name = f"deliver-digest-{subscription.id}"
    celery_crontab = parse_cron_to_celery(subscription.schedule_cron)

    celery_app.conf.beat_schedule[schedule_name] = {
        "task": "news_service.tasks.deliver_digest.deliver_digest",
        "schedule": celery_crontab,
        "args": [str(subscription.id)],
    }
    logger.info(
        "Registered delivery schedule '%s' with cron '%s'",
        schedule_name,
        subscription.schedule_cron,
        extra={"subscription_id": str(subscription.id)},
    )


def remove_delivery_schedule(subscription: Subscription) -> None:
    schedule_name = f"deliver-digest-{subscription.id}"
    celery_app.conf.beat_schedule.pop(schedule_name, None)
    logger.info(
        "Removed delivery schedule '%s'",
        schedule_name,
        extra={"subscription_id": str(subscription.id)},
    )
