from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscribe
from tgbot.language import LanguagePreference


def _mock_message(telegram_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        text=text,
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        ),
    )


def _mock_callback(telegram_id: int, data: str) -> SimpleNamespace:
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=42,
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        message=msg,
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        ),
    )


def _mock_state(data=None):
    base = {"_menu_msg_id": 42}
    if data:
        base.update(data)
    return SimpleNamespace(
        get_data=AsyncMock(return_value=base),
        get_state=AsyncMock(return_value=None),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _mock_ui_language(monkeypatch) -> None:
    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))


@pytest.mark.asyncio
async def test_process_prompt_with_explicit_schedule_goes_to_source_scope(monkeypatch):
    message = _mock_message(telegram_id=111, text="AI news every morning")
    # Pre-populate state with what update_data would set, so _continue_after_prompt
    # can read the prompt and delivery_mode
    state = _mock_state(
        data={
            "prompt": "AI news every morning",
            "delivery_mode": "digest",
            "schedule_cron_override": "0 8 * * *",
            "schedule_was_explicit": True,
            "digest_language_override": "en",
        }
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
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                schedule_was_explicit=True,
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    # Should set source knowledge state (via _continue_with_source_flow)
    state.set_state.assert_awaited()


@pytest.mark.asyncio
async def test_process_prompt_without_schedule_asks_schedule_decision(monkeypatch):
    message = _mock_message(telegram_id=111, text="AI news")
    state = _mock_state(
        data={
            "prompt": "AI news",
            "delivery_mode": "digest",
            "schedule_cron_override": None,
            "schedule_was_explicit": False,
            "digest_language_override": "en",
        }
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
                delivery_mode="digest",
                schedule_cron=None,
                schedule_was_explicit=False,
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_schedule_decision)


@pytest.mark.asyncio
async def test_process_prompt_with_event_mode_skips_schedule(monkeypatch):
    message = _mock_message(telegram_id=111, text="Notify me about new Severance episodes")
    state = _mock_state(
        data={
            "prompt": "Notify me about new Severance episodes",
            "delivery_mode": "event",
            "schedule_cron_override": None,
            "schedule_was_explicit": False,
            "digest_language_override": "en",
        }
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
                delivery_mode="event",
                schedule_cron=None,
                schedule_was_explicit=False,
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    # Should go to source flow, not schedule
    state.set_state.assert_awaited()
    # Should NOT be schedule_decision
    set_calls = [str(c) for c in state.set_state.await_args_list]
    assert not any("schedule_decision" in c for c in set_calls)


@pytest.mark.asyncio
async def test_handle_schedule_decision_no_continues_source_flow(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.SCHEDULE_ENABLE_NO)
    state = _mock_state(data={"prompt": "AI news", "delivery_mode": "digest", "_menu_msg_id": 42})

    await subscribe.handle_schedule_decision(callback, state)

    state.set_state.assert_awaited()


@pytest.mark.asyncio
async def test_process_prompt_with_ask_mode_prompts_for_subscription_language(monkeypatch):
    message = _mock_message(telegram_id=111, text="AI news")
    state = _mock_state(data={"_menu_msg_id": 42})

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
                delivery_mode="digest",
                schedule_cron=None,
                schedule_was_explicit=False,
            )
        ),
    )

    await subscribe.process_prompt(message, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_language_choice)


@pytest.mark.asyncio
async def test_handle_back_from_schedule_input_returns_to_schedule_decision(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.BACK)
    state = _mock_state(data={"_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_schedule_input.state
    )

    await subscribe.handle_back(callback, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_schedule_decision)


@pytest.mark.asyncio
async def test_handle_back_from_channels_input_returns_to_source_knowledge(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.BACK)
    state = _mock_state(data={"delivery_mode": "digest", "_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_channels_input.state
    )

    await subscribe.handle_back(callback, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_source_knowledge)


@pytest.mark.asyncio
async def test_handle_scope_choice_creates_manual_only_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.SCOPE_ONLY_PROVIDED)
    state = _mock_state(
        data={
            "prompt": "AI news",
            "delivery_mode": "digest",
            "schedule_cron_override": None,
            "manual_only": True,
            "telegram_channels": ["test_channel"],
            "reddit_subreddits": [],
            "twitter_accounts": [],
            "_menu_msg_id": 42,
        }
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    create_sub = AsyncMock(return_value=SimpleNamespace(id="sub-new", prompt_summary="AI news"))
    monkeypatch.setattr(subscribe.backend, "create_subscription", create_sub)

    await subscribe.handle_scope_choice(callback, state)

    create_sub.assert_awaited_once()
    call_kwargs = create_sub.await_args.kwargs
    assert call_kwargs["fixed_telegram_channels"] == ["test_channel"]
    assert call_kwargs["manual_only"] is True
    assert call_kwargs["delivery_mode"] == "digest"


@pytest.mark.asyncio
async def test_show_scope_choice_formats_x_accounts_without_https(monkeypatch):
    callback = _mock_callback(telegram_id=111, data="unused")
    state = _mock_state(
        data={
            "delivery_mode": "digest",
            "telegram_channels": [],
            "reddit_subreddits": [],
            "twitter_accounts": ["OpenAI"],
            "scope_channels_origin": "manual",
            "_menu_msg_id": 42,
        }
    )

    await subscribe._show_scope_choice_step(callback, state)

    # edit_menu edits via bot.edit_message_text since mock isn't real CallbackQuery
    callback.bot.edit_message_text.assert_awaited_once()
    text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert "x.com/OpenAI" in text
    assert "https://" not in text


@pytest.mark.asyncio
async def test_handle_scope_choice_for_event_subscription_offers_recent_events(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.SCOPE_ONLY_PROVIDED)
    state = _mock_state(
        data={
            "prompt": "Concert announcements",
            "delivery_mode": "event",
            "schedule_cron_override": None,
            "manual_only": None,
            "telegram_channels": [],
            "reddit_subreddits": [],
            "twitter_accounts": [],
            "_menu_msg_id": 42,
        }
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "create_subscription",
        AsyncMock(return_value=SimpleNamespace(id="sub-evt", prompt_summary="Concerts")),
    )

    await subscribe.handle_scope_choice(callback, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_recent_events_decision)


@pytest.mark.asyncio
async def test_handle_recent_events_decision_yes_sends_backfill(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.RECENT_EVENTS_YES)
    state = _mock_state(data={"created_subscription_id": "sub-evt", "_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "list_recent_events",
        AsyncMock(
            return_value=SimpleNamespace(
                subject="Recent events",
                body="Event 1\nEvent 2",
                news_item_ids=["n1", "n2"],
            )
        ),
    )
    monkeypatch.setattr(subscribe.backend, "acknowledge_recent_events", AsyncMock())

    with patch("tgbot.handlers.menu.show_subscription_list", new_callable=AsyncMock):
        await subscribe.handle_recent_events_decision(callback, state)

    subscribe.backend.list_recent_events.assert_awaited_once_with("api-key", "sub-evt")
    subscribe.backend.acknowledge_recent_events.assert_awaited_once()
    state.clear.assert_awaited()


@pytest.mark.asyncio
async def test_handle_back_from_recent_events_deletes_created_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=subscribe.BACK)
    state = _mock_state(
        data={
            "created_subscription_id": "sub-evt",
            "creation_back_target": "source_knowledge",
            "delivery_mode": "event",
            "_menu_msg_id": 42,
        }
    )
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    delete_sub = AsyncMock()
    monkeypatch.setattr(subscribe.backend, "delete_subscription", delete_sub)

    await subscribe.handle_back(callback, state)

    delete_sub.assert_awaited_once_with("api-key", "sub-evt")
    state.set_state.assert_awaited()
