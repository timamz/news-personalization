from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from celery.schedules import crontab
from croniter import croniter


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


def is_schedule_due(
    cron_expr: str,
    *,
    last_run_at: datetime,
    now: datetime | None = None,
    timezone_name: str = "UTC",
) -> bool:
    """Return True when cron expression is due at the provided moment."""
    timezone = ZoneInfo(timezone_name)
    current_time = _as_timezone(now or datetime.now(UTC), timezone)
    schedule = parse_cron_to_celery(cron_expr)
    schedule.nowfun = lambda: current_time
    due, _next_check_seconds = schedule.is_due(_as_timezone(last_run_at, timezone))
    return due


def next_cron_match(
    cron_expr: str,
    *,
    after: datetime,
    timezone_name: str = "UTC",
) -> datetime:
    """Return the next datetime (UTC) at which the cron expression fires strictly after `after`."""
    timezone = ZoneInfo(timezone_name)
    base = _as_timezone(after, timezone)
    itr = croniter(cron_expr, base)
    return itr.get_next(datetime).astimezone(UTC)


def _as_timezone(value: datetime, timezone: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).astimezone(timezone)
    return value.astimezone(timezone)
