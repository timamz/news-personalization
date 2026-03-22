"""Tests for parse_schedule_preference function."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.schedule_parser import (
    ParsedSchedule,
    parse_schedule_preference,
)

logging.disable(logging.CRITICAL)


def _mock_completion(schedule_cron: str | None) -> MagicMock:
    """Build a fake OpenAI completion with parsed output."""
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
    async def test_returns_cron_from_llm_output(self) -> None:
        cron_value = "0 8 * * *"
        fake_completion = _mock_completion(cron_value)
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

        with patch(
            "news_service.agents.schedule_parser._client",
            mock_client,
        ):
            result = await parse_schedule_preference("каждое утро в 8")

        assert result == cron_value, f"did not return expected cron string, got {result!r}"

    @pytest.mark.asyncio
    async def test_raises_value_error_when_llm_returns_none(self) -> None:
        fake_completion = _mock_completion(None)
        fake_completion.choices[0].message.parsed = None
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

        with (
            patch(
                "news_service.agents.schedule_parser._client",
                mock_client,
            ),
            pytest.raises(ValueError),
        ):
            await parse_schedule_preference("каждый день в полдень")
