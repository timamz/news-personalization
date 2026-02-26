import pytest

from news_service.services.scheduler import parse_cron_to_celery


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
