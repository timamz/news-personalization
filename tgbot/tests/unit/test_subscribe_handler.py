from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import subscribe


def _mock_message(telegram_id: int, text: str):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        text=text,
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_cmd_subscribe_prompts_for_description(monkeypatch):
    message = _mock_message(telegram_id=123, text="")
    state = SimpleNamespace(set_state=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    await subscribe.cmd_subscribe(message, state)

    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_prompt)
    message.answer.assert_awaited_once()
    assert "Describe what news you want" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_process_prompt_hides_technical_details(monkeypatch):
    message = _mock_message(telegram_id=123, text="AI news daily")
    state = SimpleNamespace(clear=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    create_subscription = AsyncMock(
        return_value=SimpleNamespace(
            id="sub-1",
            topics=["ai", "technology"],
            schedule_cron="0 8 * * *",
            format_instructions="brief summary",
        )
    )
    monkeypatch.setattr(subscribe.backend, "create_subscription", create_subscription)

    await subscribe.process_prompt(message, state)

    create_subscription.assert_awaited_once()
    assert message.answer.await_count == 2

    confirmation_text = message.answer.await_args_list[1].args[0]
    assert "Topics: ai, technology" in confirmation_text
    assert "Schedule:" not in confirmation_text
    assert "Format:" not in confirmation_text
