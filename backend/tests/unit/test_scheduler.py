import logging
from datetime import UTC, datetime

import pytest

from news_service.services.scheduler import is_schedule_due, parse_cron_to_celery

logging.disable(logging.CRITICAL)


def test_parse_daily_morning_cron() -> None:
    result = parse_cron_to_celery("0 8 * * *")
    assert str(result) == "<crontab: 0 8 * * * (m/h/dM/MY/d)>", (
        "parse did not produce correct daily morning crontab"
    )


def test_parse_every_third_day() -> None:
    result = parse_cron_to_celery("0 8 */3 * *")
    assert str(result) == "<crontab: 0 8 */3 * * (m/h/dM/MY/d)>", (
        "parse did not produce correct every-third-day crontab"
    )


def test_parse_saturday_morning() -> None:
    result = parse_cron_to_celery("0 8 * * 6")
    assert str(result) == "<crontab: 0 8 * * 6 (m/h/dM/MY/d)>", (
        "parse did not produce correct Saturday morning crontab"
    )


def test_parse_invalid_cron_raises() -> None:
    with pytest.raises(ValueError):
        parse_cron_to_celery("0 8 *")


def test_parse_every_15_minutes() -> None:
    result = parse_cron_to_celery("*/15 * * * *")
    assert str(result) == "<crontab: */15 * * * * (m/h/dM/MY/d)>", (
        "parse did not produce correct 15-minute crontab"
    )


def test_schedule_due_when_next_run_reached() -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is True, (
        "schedule was not detected as due when next run time was reached"
    )


def test_schedule_not_due_before_scheduled_time() -> None:
    now = datetime(2026, 2, 26, 7, 59, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is False, (
        "schedule was incorrectly detected as due before scheduled time"
    )


def test_schedule_not_due_when_already_scheduled_this_tick() -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is False, (
        "schedule was incorrectly detected as due when already scheduled this tick"
    )


def test_schedule_uses_user_timezone() -> None:
    now = datetime(2026, 2, 26, 6, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
    result = is_schedule_due(
        "0 9 * * *",
        last_run_at=last_run,
        now=now,
        timezone_name="Europe/Berlin",
    )
    assert result is True, "schedule did not account for user timezone when checking due status"
