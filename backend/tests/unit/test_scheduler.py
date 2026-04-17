import logging
from datetime import UTC, datetime

import pytest

from news_service.services.scheduler import is_schedule_due, parse_cron_to_celery

logging.disable(logging.CRITICAL)


@pytest.mark.parametrize(
    "expression",
    ["0 8 * * *", "0 8 */3 * *", "0 8 * * 6", "*/15 * * * *"],
)
def test_parse_cron_to_celery_accepts_valid_expressions(expression: str) -> None:
    result = parse_cron_to_celery(expression)
    assert str(result) == f"<crontab: {expression} (m/h/dM/MY/d)>"


def test_parse_cron_to_celery_raises_on_invalid_expression() -> None:
    with pytest.raises(ValueError):
        parse_cron_to_celery("0 8 *")


def test_is_schedule_due_fires_when_next_run_reached_and_stays_quiet_before_it() -> None:
    due = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    not_yet = datetime(2026, 2, 26, 7, 59, tzinfo=UTC)
    last_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)

    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=due) is True
    assert is_schedule_due("0 8 * * *", last_run_at=last_run, now=not_yet) is False
    # Already scheduled at this exact tick
    assert is_schedule_due("0 8 * * *", last_run_at=due, now=due) is False


def test_is_schedule_due_uses_user_timezone() -> None:
    now = datetime(2026, 2, 26, 6, 0, tzinfo=UTC)  # 07:00 Berlin (DST off)
    last_run = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
    assert (
        is_schedule_due(
            "0 9 * * *",
            last_run_at=last_run,
            now=now,
            timezone_name="Europe/Berlin",
        )
        is True
    ), "user timezone was not applied when evaluating the cron boundary"
