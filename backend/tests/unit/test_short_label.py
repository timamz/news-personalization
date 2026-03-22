"""Tests for generate_short_label function."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.short_label import (
    ShortLabel,
    generate_short_label,
)

logging.disable(logging.CRITICAL)


def _mock_completion(label: str | None) -> MagicMock:
    """Build a fake OpenAI completion with parsed ShortLabel output."""
    parsed = ShortLabel(label=label) if label is not None else None
    msg = MagicMock()
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestGenerateShortLabel:
    @pytest.mark.asyncio
    async def test_returns_label_from_llm_output(self) -> None:
        expected_label = "Тех новости"
        fake_completion = _mock_completion(expected_label)
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

        with patch(
            "news_service.agents.short_label._client",
            mock_client,
        ):
            result = await generate_short_label("Новости технологий и ИИ")

        assert result == expected_label, f"did not return expected label, got {result!r}"

    @pytest.mark.asyncio
    async def test_raises_value_error_when_llm_returns_none(self) -> None:
        fake_completion = _mock_completion(None)
        fake_completion.choices[0].message.parsed = None
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

        with (
            patch(
                "news_service.agents.short_label._client",
                mock_client,
            ),
            pytest.raises(ValueError),
        ):
            await generate_short_label("Криптовалюты и блокчейн")
