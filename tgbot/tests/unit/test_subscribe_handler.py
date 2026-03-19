from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscribe
from tgbot.language import LanguagePreference


def _mock_status_msg():
    """A mock message object that supports edit_text and delete."""
    return SimpleNamespace(
        message_id=100,
        edit_text=AsyncMock(),
        delete=AsyncMock(),
        chat=SimpleNamespace(id=111),
    )


def _mock_message(telegram_id: int, text: str) -> SimpleNamespace:
    status_msg = _mock_status_msg()
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        text=text,
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(return_value=status_msg),
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


async def _mock_stream_simple(agent_message="What schedule?", conv_id="conv-123", **kwargs):
    """Async generator that yields a simple done event (no tool calls)."""
    yield {
        "event": "done",
        "conversation_id": conv_id,
        "agent_message": agent_message,
        "status": kwargs.get("status", "in_progress"),
        "finalized_config": kwargs.get("finalized_config"),
    }


async def _mock_stream_with_tools(agent_message="Verified!", conv_id="conv-123"):
    """Async generator that yields status events then done."""
    yield {"event": "status", "status_message": "@durov"}
    yield {
        "event": "done",
        "conversation_id": conv_id,
        "agent_message": agent_message,
        "status": "in_progress",
    }


async def _mock_create_stream(sub_id="sub-new", prompt_summary="AI news digest", **extra):
    """Async generator that yields status then done for subscription creation."""
    yield {"event": "status", "status_message": "Analyzing your request..."}
    yield {
        "event": "done",
        "subscription": {
            "id": sub_id,
            "prompt_summary": prompt_summary,
            "delivery_mode": extra.get("delivery_mode", "digest"),
            "schedule_cron": extra.get("schedule_cron"),
            "format_instructions": extra.get("format_instructions", "brief summary"),
            "digest_language": extra.get("digest_language", "en"),
            "short_label": extra.get("short_label", ""),
        },
    }


@pytest.fixture(autouse=True)
def _mock_ui_language(monkeypatch) -> None:
    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))


# ---------- First message starts conversation via stream ----------


@pytest.mark.asyncio
async def test_first_message_starts_conversation(monkeypatch):
    message = _mock_message(telegram_id=111, text="AI news every morning")
    state = _mock_state()

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="fixed", code="en")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "get_current_user",
        AsyncMock(return_value=SimpleNamespace(timezone="UTC")),
    )

    monkeypatch.setattr(
        subscribe.backend,
        "start_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_simple(
            agent_message="How often would you like updates?",
        ),
    )

    await subscribe.process_chat_message(message, state)

    state.update_data.assert_awaited()


# ---------- Status message is sent and edited to final response ----------


@pytest.mark.asyncio
async def test_status_message_edited_to_final(monkeypatch):
    """Verify: status message sent first, then edited to agent response."""
    message = _mock_message(telegram_id=111, text="every morning")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_simple(
            agent_message="Great, updates every morning!",
        ),
    )

    await subscribe.process_chat_message(message, state)

    # First call: status message
    message.bot.send_message.assert_awaited_once()
    # Status message was edited to final response
    status_msg = message.bot.send_message.return_value
    status_msg.edit_text.assert_awaited_once()
    edit_kwargs = status_msg.edit_text.await_args.kwargs
    assert edit_kwargs["text"] == "Great, updates every morning!"


# ---------- Status edits during tool calls ----------


@pytest.mark.asyncio
async def test_status_updates_during_tool_calls(monkeypatch):
    """Verify status message is edited when tool call events arrive."""
    message = _mock_message(telegram_id=111, text="add @durov")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_with_tools(),
    )

    await subscribe.process_chat_message(message, state)

    status_msg = message.bot.send_message.return_value
    # edit_text called at least twice: once for status update, once for final
    assert status_msg.edit_text.await_count >= 2


# ---------- Continued message uses stream endpoint ----------


@pytest.mark.asyncio
async def test_continued_message_uses_stream(monkeypatch):
    message = _mock_message(telegram_id=111, text="every morning")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    stream_calls = []

    async def mock_stream(api_key, conv_id, msg):
        stream_calls.append((api_key, conv_id, msg))
        async for ev in _mock_stream_simple():
            yield ev

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        mock_stream,
    )

    await subscribe.process_chat_message(message, state)

    assert stream_calls == [("api-key", "conv-123", "every morning")]


# ---------- Finalized config triggers subscription creation ----------


@pytest.mark.asyncio
async def test_finalized_config_creates_digest_subscription(monkeypatch):
    message = _mock_message(telegram_id=111, text="yes, discover sources")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "AI news digest",
        "short_label": "AI News",
        "delivery_mode": "digest",
        "schedule_cron": "0 8 * * *",
        "manual_only": False,
        "format_instructions": "brief summary",
        "digest_language": "en",
        "fixed_telegram_channels": [],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": True,
    }

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_simple(
            agent_message="Ready!",
            status="ready",
            finalized_config=config,
        ),
    )

    create_calls = []

    async def mock_create_stream(*a, **kw):
        create_calls.append(kw)
        async for ev in _mock_create_stream(prompt_summary="AI news digest"):
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create_stream)

    await subscribe.process_chat_message(message, state)

    assert len(create_calls) == 1
    assert create_calls[0]["delivery_mode"] == "digest"
    assert create_calls[0]["schedule_cron_override"] == "0 8 * * *"
    assert create_calls[0]["include_discovered_sources"] is True


# ---------- Event subscription shows recent events prompt ----------


@pytest.mark.asyncio
async def test_finalized_event_subscription_offers_recent_events(monkeypatch):
    message = _mock_message(telegram_id=111, text="yes")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "Concert alerts",
        "short_label": "Concerts",
        "delivery_mode": "event",
        "event_matching_mode": "basic",
        "schedule_cron": None,
        "manual_only": False,
        "format_instructions": "brief summary",
        "digest_language": "en",
        "fixed_telegram_channels": [],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": True,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_simple(
            agent_message="Ready!",
            status="ready",
            finalized_config=config,
        ),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "create_subscription_stream",
        lambda *a, **kw: _mock_create_stream(
            sub_id="sub-evt", prompt_summary="Concerts", delivery_mode="event"
        ),
    )

    await subscribe.process_chat_message(message, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_recent_events_decision)


# ---------- Recent events flow ----------


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
async def test_handle_recent_events_empty_shows_message_with_back_button(monkeypatch):
    """When no recent events, show the empty message with a back button instead of navigating."""
    callback = _mock_callback(telegram_id=111, data=subscribe.RECENT_EVENTS_YES)
    state = _mock_state(data={"created_subscription_id": "sub-evt", "_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
    )

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "list_recent_events",
        AsyncMock(return_value=None),
    )

    await subscribe.handle_recent_events_decision(callback, state)

    subscribe.backend.list_recent_events.assert_awaited_once_with("api-key", "sub-evt")
    state.clear.assert_awaited()
    # Should edit message with empty text AND a keyboard (back button), not navigate away
    callback.bot.edit_message_text.assert_awaited()
    edit_kwargs = callback.bot.edit_message_text.call_args.kwargs
    assert edit_kwargs.get("reply_markup") is not None


# ---------- Cancel cleans up conversation ----------


@pytest.mark.asyncio
async def test_cancel_cleans_up_conversation(monkeypatch):
    callback = _mock_callback(telegram_id=111, data="subscribe:cancel")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscribe.backend, "cancel_subscription_conversation", AsyncMock())

    with patch("tgbot.handlers.menu.show_main_menu", new_callable=AsyncMock):
        await subscribe.handle_cancel(callback, state)

    subscribe.backend.cancel_subscription_conversation.assert_awaited_once_with(
        "api-key", "conv-123"
    )
    state.clear.assert_awaited()


# ---------- Menu text is ignored ----------


@pytest.mark.asyncio
async def test_menu_text_is_ignored(monkeypatch):
    message = _mock_message(telegram_id=111, text="📋 Menu")
    state = _mock_state()

    await subscribe.process_chat_message(message, state)

    # No state changes
    state.set_state.assert_not_awaited()


# ---------- Fixed sources in config are forwarded ----------


@pytest.mark.asyncio
async def test_fixed_sources_forwarded_to_create(monkeypatch):
    message = _mock_message(telegram_id=111, text="ok")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "Tech news from specific channels",
        "short_label": "Tech",
        "delivery_mode": "digest",
        "schedule_cron": None,
        "manual_only": True,
        "format_instructions": "brief summary",
        "digest_language": "en",
        "fixed_telegram_channels": ["durov"],
        "fixed_reddit_subreddits": ["technology"],
        "fixed_twitter_accounts": ["openai"],
        "include_discovered_sources": False,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _mock_stream_simple(
            agent_message="Ready!",
            status="ready",
            finalized_config=config,
        ),
    )

    create_calls = []

    async def mock_create_stream(*a, **kw):
        create_calls.append(kw)
        async for ev in _mock_create_stream(sub_id="sub-1", prompt_summary="Tech news"):
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create_stream)

    await subscribe.process_chat_message(message, state)

    assert len(create_calls) == 1
    assert create_calls[0]["fixed_telegram_channels"] == ["durov"]
    assert create_calls[0]["fixed_reddit_subreddits"] == ["technology"]
    assert create_calls[0]["fixed_twitter_accounts"] == ["openai"]
    assert create_calls[0]["include_discovered_sources"] is False
    assert create_calls[0]["manual_only"] is True


# ---------- Error event shows failure ----------


@pytest.mark.asyncio
async def test_stream_error_event_shows_failure(monkeypatch):
    message = _mock_message(telegram_id=111, text="test")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    async def mock_error_stream(*a, **kw):
        yield {"event": "error", "detail": "something went wrong"}

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        mock_error_stream,
    )

    await subscribe.process_chat_message(message, state)

    # Status message should be deleted
    status_msg = message.bot.send_message.return_value
    status_msg.delete.assert_awaited_once()
    # Error reply should be sent
    assert message.bot.send_message.await_count == 2  # status + error reply
