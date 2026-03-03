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


def _mock_callback(telegram_id: int, data: str) -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        message=SimpleNamespace(answer=AsyncMock()),
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_process_prompt_with_explicit_schedule_goes_to_source_scope(monkeypatch):
    message = _mock_message(telegram_id=123, text="Следи за @gonzo_ml каждый день в 9")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                topics=["ml"],
                delivery_mode="digest",
                schedule_cron="0 9 * * *",
                schedule_was_explicit=True,
                format_instructions="brief summary",
                digest_language="ru",
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.update_data.assert_any_await(
        prompt="Следи за @gonzo_ml каждый день в 9",
        delivery_mode="digest",
        schedule_cron_override="0 9 * * *",
        manual_only=False,
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_scope_choice)
    assert message.answer.await_count == 1
    assert (
        "Should this digest be limited only to these channels?"
        in message.answer.await_args.args[0]
    )


@pytest.mark.asyncio
async def test_process_prompt_without_schedule_asks_schedule_decision(monkeypatch):
    message = _mock_message(telegram_id=123, text="Хочу новости по ML")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                topics=["ml"],
                delivery_mode="digest",
                schedule_cron=None,
                schedule_was_explicit=False,
                format_instructions="brief summary",
                digest_language="ru",
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_schedule_decision)
    assert message.answer.await_count == 1
    prompt_text = message.answer.await_args.args[0]
    assert "Do you want this digest to be delivered automatically" in prompt_text


@pytest.mark.asyncio
async def test_process_prompt_with_event_mode_skips_schedule(monkeypatch):
    message = _mock_message(telegram_id=123, text="Notify me when a new episode is announced")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                topics=["tv"],
                delivery_mode="event",
                schedule_cron=None,
                schedule_was_explicit=False,
                format_instructions="brief summary",
                digest_language="en",
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.update_data.assert_any_await(
        prompt="Notify me when a new episode is announced",
        delivery_mode="event",
        schedule_cron_override=None,
        manual_only=False,
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_source_knowledge)
    assert message.answer.await_count == 1
    assert "specific Telegram channels for these notifications" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_schedule_decision_no_continues_source_flow(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.SCHEDULE_ENABLE_NO)
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={"prompt": "Следи за @gonzo_ml"}),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )

    await subscribe.handle_schedule_decision(callback, state)

    callback.answer.assert_awaited_once()
    state.update_data.assert_any_await(schedule_cron_override=None, manual_only=True)
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_scope_choice)


@pytest.mark.asyncio
async def test_handle_scope_choice_creates_manual_only_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.SCOPE_WITH_DISCOVERY)
    state = SimpleNamespace(
        get_data=AsyncMock(
            return_value={
                "prompt": "ML daily",
                "telegram_channels": ["gonzo_ml"],
                "delivery_mode": "digest",
                "schedule_cron_override": None,
                "manual_only": True,
            }
        ),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    create_subscription = AsyncMock(
        return_value=SimpleNamespace(
            id="sub-1",
            topics=["машинное обучение"],
            schedule_cron=None,
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
        schedule_cron_override=None,
        manual_only=True,
        delivery_mode="digest",
    )
    assert callback.message.answer.await_count == 2
    assert "Subscription created!" in callback.message.answer.await_args_list[1].args[0]
    state.clear.assert_awaited_once()
