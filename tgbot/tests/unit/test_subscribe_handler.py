from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import subscribe


def _mock_message(telegram_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        text=text,
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_process_prompt_with_channel_asks_for_scope(monkeypatch):
    message = _mock_message(telegram_id=123, text="Следи за @gonzo_ml каждое утро")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    await subscribe.process_prompt(message, state)

    state.update_data.assert_awaited_once_with(
        prompt="Следи за @gonzo_ml каждое утро",
        telegram_channels=["gonzo_ml"],
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_scope_choice)
    assert message.answer.await_count == 1
    first_button = message.answer.await_args.kwargs["reply_markup"].inline_keyboard[0][0]
    assert "Only these channels" in first_button.text


@pytest.mark.asyncio
async def test_process_prompt_without_channels_asks_if_user_has_sources(monkeypatch):
    message = _mock_message(telegram_id=123, text="ML обзоры каждое утро")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    await subscribe.process_prompt(message, state)

    state.update_data.assert_awaited_once_with(
        prompt="ML обзоры каждое утро",
        telegram_channels=[],
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_source_knowledge)
    assert message.answer.await_count == 1
    assert "Do you already have specific Telegram channels" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_scope_choice_creates_subscription_with_preferences(monkeypatch):
    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        data=subscribe.SCOPE_WITH_DISCOVERY,
        message=SimpleNamespace(answer=AsyncMock()),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={"prompt": "ML daily", "telegram_channels": ["gonzo_ml"]}),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    create_subscription = AsyncMock(
        return_value=SimpleNamespace(
            id="sub-1",
            topics=["машинное обучение"],
            schedule_cron="0 8 * * *",
            format_instructions="brief summary",
        )
    )
    monkeypatch.setattr(subscribe.backend, "create_subscription", create_subscription)

    await subscribe.handle_scope_choice(callback, state)

    callback.answer.assert_awaited_once()
    create_subscription.assert_awaited_once_with(
        "api-key",
        "ML daily",
        "http://tgbot:8001/deliver/123",
        fixed_telegram_channels=["gonzo_ml"],
        include_discovered_sources=True,
    )
    assert callback.message.answer.await_count == 2
    confirmation_text = callback.message.answer.await_args_list[1].args[0]
    assert "Subscription created!" in confirmation_text
    assert "Schedule:" not in confirmation_text
    assert "Format:" not in confirmation_text
    state.clear.assert_awaited_once()
