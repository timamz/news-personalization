from datetime import UTC, datetime

import pytest

from news_service.services.scheduler import is_schedule_due, parse_cron_to_celery


def test_parse_daily_morning_cron():
    result = parse_cron_to_celery("0 8 * * *")
    assert str(result) == "<crontab: 0 8 * * * (m/h/dM/MY/d)>"


def test_parse_every_third_day():
    result = parse_cron_to_celery("0 8 */3 * *")
    assert str(result) == "<crontab: 0 8 */3 * * (m/h/dM/MY/d)>"


def test_parse_saturday_morning():
    result = parse_cron_to_celery("0 8 * * 6")
    assert str(result) == "<crontab: 0 8 * * 6 (m/h/dM/MY/d)>"


def test_parse_invalid_cron_raises():
    with pytest.raises(ValueError, match="Invalid cron expression"):
        parse_cron_to_celery("0 8 *")


def test_parse_every_15_minutes():
    result = parse_cron_to_celery("*/15 * * * *")
    assert str(result) == "<crontab: */15 * * * * (m/h/dM/MY/d)>"


def test_schedule_due_when_next_run_reached():
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is True


def test_schedule_not_due_before_scheduled_time():
    now = datetime(2026, 2, 26, 7, 59, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is False


def test_schedule_not_due_when_already_scheduled_this_tick():
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=now) is False


def test_schedule_uses_user_timezone():
    now = datetime(2026, 2, 26, 6, 0, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)

    assert is_schedule_due(
        "0 9 * * *",
        last_run_at=last_run,
        now=now,
        timezone_name="Europe/Berlin",
    ) is True
