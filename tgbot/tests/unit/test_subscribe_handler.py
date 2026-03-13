from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ParseMode

from tgbot.handlers import subscribe
from tgbot.language import LanguagePreference
from tgbot.webhook_server import delivery_webhook_url


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


@pytest.fixture(autouse=True)
def _mock_ui_language(monkeypatch) -> None:
    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))


@pytest.mark.asyncio
async def test_process_prompt_with_explicit_schedule_goes_to_source_scope(monkeypatch):
    message = _mock_message(telegram_id=123, text="Следи за @gonzo_ml каждый день в 9")
    state = SimpleNamespace(
        update_data=AsyncMock(),
        get_data=AsyncMock(
            return_value={
                "prompt": "Следи за @gonzo_ml каждый день в 9",
                "delivery_mode": "digest",
                "schedule_cron_override": "0 9 * * *",
                "manual_only": False,
                "schedule_was_explicit": True,
                "digest_language_override": "ru",
            }
        ),
        set_state=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="fixed", code="ru")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                prompt_summary="Следить за @gonzo_ml каждый день в 9",
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
        schedule_was_explicit=True,
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_scope_choice)
    assert message.answer.await_count == 1
    assert "Should this digest be limited only to these sources?" in (
        message.answer.await_args.args[0]
    )


@pytest.mark.asyncio
async def test_process_prompt_without_schedule_asks_schedule_decision(monkeypatch):
    message = _mock_message(telegram_id=123, text="Хочу новости по ML")
    state = SimpleNamespace(
        update_data=AsyncMock(),
        get_data=AsyncMock(
            return_value={
                "prompt": "Хочу новости по ML",
                "delivery_mode": "digest",
                "schedule_cron_override": None,
                "manual_only": False,
                "schedule_was_explicit": False,
                "digest_language_override": "ru",
            }
        ),
        set_state=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="fixed", code="ru")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                prompt_summary="Новости по ML",
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
    keyboard = message.answer.await_args.kwargs["reply_markup"]
    assert keyboard.inline_keyboard[1][0].text == "Back"


@pytest.mark.asyncio
async def test_process_prompt_with_event_mode_skips_schedule(monkeypatch):
    message = _mock_message(telegram_id=123, text="Notify me when a new episode is announced")
    state = SimpleNamespace(
        update_data=AsyncMock(),
        get_data=AsyncMock(
            return_value={
                "prompt": "Notify me when a new episode is announced",
                "delivery_mode": "event",
                "schedule_cron_override": None,
                "manual_only": False,
                "schedule_was_explicit": False,
                "digest_language_override": "en",
            }
        ),
        set_state=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="fixed", code="en")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                prompt_summary="New episode announcements",
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
        schedule_was_explicit=False,
    )
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_source_knowledge)
    assert message.answer.await_count == 1
    assert "specific sources for these notifications" in message.answer.await_args.args[0]


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
async def test_process_prompt_with_ask_mode_prompts_for_subscription_language(monkeypatch):
    message = _mock_message(telegram_id=123, text="Хочу новости по ML")
    state = SimpleNamespace(update_data=AsyncMock(), set_state=AsyncMock())

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="ask", code=None)),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "parse_subscription_prompt",
        AsyncMock(
            return_value=SimpleNamespace(
                prompt_summary="Новости по ML",
                delivery_mode="digest",
                schedule_cron=None,
                schedule_was_explicit=False,
                format_instructions="brief summary",
                digest_language="ru",
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_language_choice)
    assert "Choose the language for this subscription" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_back_from_schedule_input_returns_to_schedule_decision() -> None:
    callback = _mock_callback(telegram_id=123, data=subscribe.BACK)
    state = SimpleNamespace(
        get_state=AsyncMock(return_value=subscribe.SubscribeFlow.waiting_for_schedule_input.state),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
    )

    await subscribe.handle_back(callback, state)

    callback.answer.assert_awaited_once()
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_schedule_decision)
    assert (
        "Do you want this digest to be delivered automatically"
        in callback.message.answer.await_args.args[0]
    )


@pytest.mark.asyncio
async def test_handle_back_from_channels_input_returns_to_source_knowledge() -> None:
    callback = _mock_callback(telegram_id=123, data=subscribe.BACK)
    state = SimpleNamespace(
        get_state=AsyncMock(return_value=subscribe.SubscribeFlow.waiting_for_channels_input.state),
        get_data=AsyncMock(return_value={"delivery_mode": "event"}),
        set_state=AsyncMock(),
    )

    await subscribe.handle_back(callback, state)

    callback.answer.assert_awaited_once()
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_source_knowledge)
    assert "specific sources for these notifications" in callback.message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_scope_choice_creates_manual_only_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.SCOPE_WITH_DISCOVERY)
    state = SimpleNamespace(
        get_data=AsyncMock(
            return_value={
                "prompt": "ML daily",
                "telegram_channels": ["gonzo_ml"],
                "twitter_accounts": ["openai"],
                "delivery_mode": "digest",
                "schedule_cron_override": None,
                "manual_only": True,
                "digest_language_override": "ru",
            }
        ),
        update_data=AsyncMock(),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    create_subscription = AsyncMock(
        return_value=SimpleNamespace(
            id="sub-1",
            prompt_summary="ML daily",
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
        delivery_webhook_url(123),
        fixed_telegram_channels=["gonzo_ml"],
        fixed_twitter_accounts=["openai"],
        include_discovered_sources=True,
        schedule_cron_override=None,
        manual_only=True,
        delivery_mode="digest",
        digest_language="ru",
    )
    assert callback.message.answer.await_count == 2
    assert "Subscription created!" in callback.message.answer.await_args_list[1].args[0]
    state.clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_show_scope_choice_formats_x_accounts_without_https(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.BACK)
    state = SimpleNamespace(
        get_data=AsyncMock(
            return_value={
                "delivery_mode": "digest",
                "telegram_channels": ["gonzo_ml"],
                "reddit_subreddits": ["python"],
                "twitter_accounts": ["openai"],
                "scope_channels_origin": "prompt",
            }
        ),
        set_state=AsyncMock(),
    )

    await subscribe._show_scope_choice_step(callback, state)

    text = callback.message.answer.await_args.args[0]
    assert "x.com/openai" in text
    assert "https://x.com/openai" not in text


@pytest.mark.asyncio
async def test_handle_scope_choice_for_event_subscription_offers_recent_events(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.SCOPE_ONLY_PROVIDED)
    state = SimpleNamespace(
        get_data=AsyncMock(
            return_value={
                "prompt": "Notify me when concerts are announced",
                "telegram_channels": ["music_channel"],
                "delivery_mode": "event",
                "schedule_cron_override": None,
                "manual_only": False,
                "digest_language_override": "en",
            }
        ),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "create_subscription",
        AsyncMock(
            return_value=SimpleNamespace(
                id="sub-evt",
                prompt_summary="Concert announcements",
                schedule_cron=None,
                format_instructions="brief summary",
            )
        ),
    )

    await subscribe.handle_scope_choice(callback, state)

    assert callback.message.answer.await_count == 3
    assert "Subscription created!" in callback.message.answer.await_args_list[1].args[0]
    assert (
        callback.message.answer.await_args_list[2].args[0]
        == "Would you like to see what you might have missed in the last 7 days?"
    )
    state.update_data.assert_any_await(created_subscription_id="sub-evt")
    state.set_state.assert_awaited_once_with(
        subscribe.SubscribeFlow.waiting_for_recent_events_decision
    )
    state.clear.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_recent_events_decision_yes_sends_backfill(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.RECENT_EVENTS_YES)
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={"created_subscription_id": "sub-evt"}),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "list_recent_events",
        AsyncMock(
            return_value=SimpleNamespace(
                news_item_ids=["news-1", "news-2"],
                subject="Recent events you may have missed",
                body="- Demo concert\n- Another concert",
            )
        ),
    )
    acknowledge_recent_events = AsyncMock()
    monkeypatch.setattr(
        subscribe.backend,
        "acknowledge_recent_events",
        acknowledge_recent_events,
    )

    await subscribe.handle_recent_events_decision(callback, state)

    callback.answer.assert_awaited_once()
    assert callback.message.answer.await_count == 2
    assert (
        callback.message.answer.await_args_list[0].args[0]
        == "Checking what you might have missed in the last 7 days..."
    )
    assert (
        callback.message.answer.await_args_list[1].args[0]
        == "Recent events you may have missed\n\n• Demo concert\n• Another concert"
    )
    assert callback.message.answer.await_args_list[0].kwargs["parse_mode"] == ParseMode.HTML
    assert callback.message.answer.await_args_list[0].kwargs["disable_web_page_preview"] is True
    assert callback.message.answer.await_args_list[1].kwargs["parse_mode"] == ParseMode.HTML
    assert callback.message.answer.await_args_list[1].kwargs["disable_web_page_preview"] is True
    acknowledge_recent_events.assert_awaited_once_with(
        "api-key",
        "sub-evt",
        ["news-1", "news-2"],
    )
    state.clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_back_from_recent_events_deletes_created_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=123, data=subscribe.BACK)
    state = SimpleNamespace(
        get_state=AsyncMock(
            return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
        ),
        get_data=AsyncMock(
            return_value={
                "created_subscription_id": "sub-evt",
                "creation_back_target": "scope_choice",
                "delivery_mode": "event",
                "telegram_channels": ["music_channel"],
                "scope_channels_origin": "manual",
            }
        ),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    delete_subscription = AsyncMock()
    monkeypatch.setattr(subscribe.backend, "delete_subscription", delete_subscription)

    await subscribe.handle_back(callback, state)

    callback.answer.assert_awaited_once()
    delete_subscription.assert_awaited_once_with("api-key", "sub-evt")
    state.update_data.assert_awaited_once_with(created_subscription_id=None)
    state.set_state.assert_awaited_once_with(subscribe.SubscribeFlow.waiting_for_scope_choice)
    assert "Should these notifications be limited only to these sources?" in (
        callback.message.answer.await_args.args[0]
    )
