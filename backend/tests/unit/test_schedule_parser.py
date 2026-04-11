"""Tests for parse_schedule_preference function."""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.schedule_parser import (
    ParsedSchedule,
    parse_schedule_preference,
)

logging.disable(logging.CRITICAL)


def _mock_completion(schedule_cron: str | None) -> MagicMock:
    parsed = ParsedSchedule(schedule_cron=schedule_cron) if schedule_cron is not None else None
    msg = MagicMock()
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestParseSchedulePreference:
    @pytest.mark.asyncio
    async def test_returns_cron_from_llm_output(self, mocker) -> None:
        cron_value = f"0 {uuid.uuid4().int % 24} * * *"
        mocker.patch(
            "news_service.agents.schedule_parser.chat_completion",
            new=AsyncMock(return_value=_mock_completion(cron_value)),
        )

        result = await parse_schedule_preference("каждое утро в 8")

        assert result == cron_value, f"did not return expected cron string, got {result!r}"

    @pytest.mark.asyncio
    async def test_raises_value_error_when_llm_returns_none(self, mocker) -> None:
        completion = _mock_completion(None)
        completion.choices[0].message.parsed = None
        mocker.patch(
            "news_service.agents.schedule_parser.chat_completion",
            new=AsyncMock(return_value=completion),
        )

        with pytest.raises(ValueError):
            await parse_schedule_preference(f"каждый день {uuid.uuid4().hex[:4]}")
