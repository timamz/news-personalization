import logging
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscribe
from tgbot.language import LanguagePreference

logging.disable(logging.CRITICAL)


def _make_status_msg() -> SimpleNamespace:
    return SimpleNamespace(
        message_id=random.randint(100, 999),
        edit_text=AsyncMock(),
        delete=AsyncMock(),
        chat=SimpleNamespace(id=random.randint(100, 999)),
    )


def _make_message(telegram_id: int, text: str) -> SimpleNamespace:
    status_msg = _make_status_msg()
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


def _make_callback(telegram_id: int, data: str) -> SimpleNamespace:
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        edit_reply_markup=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=random.randint(10, 100),
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        message=msg,
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(return_value=_make_status_msg()),
        ),
    )


def _make_state(data: dict | None = None) -> SimpleNamespace:
    base = {"_menu_msg_id": random.randint(10, 100)}
    if data:
        base.update(data)
    return SimpleNamespace(
        get_data=AsyncMock(return_value=base),
        get_state=AsyncMock(return_value=None),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )


async def _stream_simple(
    agent_message: str = "What schedule?", conv_id: str = "conv-123", **kwargs
):
    yield {
        "event": "done",
        "conversation_id": conv_id,
        "agent_message": agent_message,
        "status": kwargs.get("status", "in_progress"),
        "finalized_config": kwargs.get("finalized_config"),
    }


async def _stream_with_tools(
    agent_message: str = "\u041f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u043e!",
    conv_id: str = "conv-123",
):
    yield {"event": "status", "status_message": "@durov"}
    yield {
        "event": "done",
        "conversation_id": conv_id,
        "agent_message": agent_message,
        "status": "in_progress",
    }


async def _create_stream(
    sub_id: str = "sub-new",
    prompt_summary: str = "\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0418\u0418",
    **extra,
):
    yield {
        "event": "status",
        "status_message": "\u0410\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0443\u0435\u043c...",
    }
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


@pytest.mark.asyncio
async def test_first_message_starts_conversation(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(
        telegram_id=tid,
        text="\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0418\u0418",
    )
    state = _make_state()

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="fixed", code="ru")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "get_current_user",
        AsyncMock(return_value=SimpleNamespace(timezone="UTC")),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "start_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="\u041a\u0430\u043a \u0447\u0430\u0441\u0442\u043e?"
        ),
    )

    await subscribe.process_chat_message(message, state)

    state.update_data.assert_awaited(), "first message did not update state data"


@pytest.mark.asyncio
async def test_status_message_edited_to_final_response(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    final_text = f"\u041e\u0442\u0432\u0435\u0442-{random.randint(1, 999)}"
    message = _make_message(
        telegram_id=tid, text="\u043a\u0430\u0436\u0434\u043e\u0435 \u0443\u0442\u0440\u043e"
    )
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(agent_message=final_text),
    )

    await subscribe.process_chat_message(message, state)

    status_msg = message.bot.send_message.return_value
    edit_kwargs = status_msg.edit_text.await_args.kwargs
    assert edit_kwargs["text"] == final_text, (
        "status message was not edited to the final agent response"
    )


@pytest.mark.asyncio
async def test_status_updates_during_tool_calls(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u0434\u043e\u0431\u0430\u0432\u044c @durov")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_with_tools(),
    )

    await subscribe.process_chat_message(message, state)

    status_msg = message.bot.send_message.return_value
    assert status_msg.edit_text.await_count >= 2, (
        "status message was not edited at least twice during tool calls"
    )


@pytest.mark.asyncio
async def test_continued_message_uses_conversation_stream(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    conv_id = f"conv-{random.randint(100, 999)}"
    user_text = (
        f"\u043a\u0430\u0436\u0434\u043e\u0435 \u0443\u0442\u0440\u043e {random.randint(1, 99)}"
    )
    message = _make_message(telegram_id=tid, text=user_text)
    state = _make_state(data={"conversation_id": conv_id})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    stream_calls: list[tuple] = []

    async def mock_stream(api_key, cid, msg):
        stream_calls.append((api_key, cid, msg))
        async for ev in _stream_simple():
            yield ev

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        mock_stream,
    )

    await subscribe.process_chat_message(message, state)

    assert stream_calls == [("api-key", conv_id, user_text)], (
        "continued message did not use the correct conversation stream parameters"
    )


@pytest.mark.asyncio
async def test_finalized_config_creates_digest_subscription(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(
        telegram_id=tid,
        text="\u0434\u0430, \u043d\u0430\u0439\u0434\u0438",
    )
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "\u0414\u0430\u0439\u0434\u0436\u0435\u0441\u0442 \u0418\u0418",
        "short_label": "AI",
        "delivery_mode": "digest",
        "schedule_cron": "0 8 * * *",
        "manual_only": False,
        "format_instructions": "\u043a\u0440\u0430\u0442\u043a\u043e",
        "digest_language": "ru",
        "fixed_telegram_channels": [],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": True,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="\u0413\u043e\u0442\u043e\u0432\u043e!",
            status="ready",
            finalized_config=config,
        ),
    )

    create_calls: list[dict] = []

    async def mock_create_stream(*a, **kw):
        create_calls.append(kw)
        async for ev in _create_stream():
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create_stream)

    await subscribe.process_chat_message(message, state)

    assert create_calls[0]["delivery_mode"] == "digest", (
        "finalized config did not trigger digest subscription creation"
    )


@pytest.mark.asyncio
async def test_finalized_config_passes_schedule_cron(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u0434\u0430")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    cron = f"0 {random.randint(6, 11)} * * *"
    config = {
        "prompt_summary": "News",
        "short_label": "N",
        "delivery_mode": "digest",
        "schedule_cron": cron,
        "manual_only": False,
        "format_instructions": "brief",
        "digest_language": "en",
        "fixed_telegram_channels": [],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": True,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="Ready!", status="ready", finalized_config=config
        ),
    )

    create_calls: list[dict] = []

    async def mock_create(*a, **kw):
        create_calls.append(kw)
        async for ev in _create_stream():
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create)

    await subscribe.process_chat_message(message, state)

    assert create_calls[0]["schedule_cron_override"] == cron, (
        "finalized config did not pass the schedule_cron to create_subscription_stream"
    )


@pytest.mark.asyncio
async def test_finalized_event_subscription_offers_recent_events(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u0434\u0430")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "\u041a\u043e\u043d\u0446\u0435\u0440\u0442\u044b",
        "short_label": "Concerts",
        "delivery_mode": "event",
        "schedule_cron": None,
        "manual_only": False,
        "format_instructions": "\u043a\u0440\u0430\u0442\u043a\u043e",
        "digest_language": "ru",
        "fixed_telegram_channels": [],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": True,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="\u0413\u043e\u0442\u043e\u0432\u043e!",
            status="ready",
            finalized_config=config,
        ),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "create_subscription_stream",
        lambda *a, **kw: _create_stream(
            sub_id="sub-evt",
            prompt_summary="\u041a\u043e\u043d\u0446\u0435\u0440\u0442\u044b",
            delivery_mode="event",
        ),
    )

    await subscribe.process_chat_message(message, state)

    state.set_state.assert_awaited_with(
        subscribe.SubscribeFlow.waiting_for_recent_events_decision,
    )


@pytest.mark.asyncio
async def test_handle_recent_events_decision_yes_acknowledges(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    callback = _make_callback(telegram_id=tid, data=subscribe.RECENT_EVENTS_YES)
    state = _make_state(data={"created_subscription_id": "sub-evt", "_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
    )

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    async def mock_recent_stream(*a, **kw):
        yield {
            "event": "status",
            "status_message": "\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430...",
        }
        yield {
            "event": "done",
            "preview": {
                "news_item_ids": ["n1", "n2"],
                "subject": "\u0421\u043e\u0431\u044b\u0442\u0438\u044f",
                "body": "\u0421\u043e\u0431\u044b\u0442\u0438\u0435 1",
            },
        }

    monkeypatch.setattr(
        subscribe.backend,
        "list_recent_events_stream",
        mock_recent_stream,
    )
    monkeypatch.setattr(subscribe.backend, "acknowledge_recent_events", AsyncMock())

    with patch("tgbot.handlers.menu.show_subscription_list", new_callable=AsyncMock):
        await subscribe.handle_recent_events_decision(callback, state)

    subscribe.backend.acknowledge_recent_events.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_recent_events_empty_clears_state(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    callback = _make_callback(telegram_id=tid, data=subscribe.RECENT_EVENTS_YES)
    state = _make_state(data={"created_subscription_id": "sub-evt", "_menu_msg_id": 42})
    state.get_state = AsyncMock(
        return_value=subscribe.SubscribeFlow.waiting_for_recent_events_decision.state
    )

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    async def mock_empty_stream(*a, **kw):
        yield {
            "event": "status",
            "status_message": "\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430...",
        }
        yield {"event": "done", "preview": None}

    monkeypatch.setattr(
        subscribe.backend,
        "list_recent_events_stream",
        mock_empty_stream,
    )

    await subscribe.handle_recent_events_decision(callback, state)

    state.clear.assert_awaited(), "handle_recent_events_decision with no events did not clear state"


@pytest.mark.asyncio
async def test_cancel_cleans_up_conversation(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    conv_id = f"conv-{random.randint(100, 999)}"
    callback = _make_callback(telegram_id=tid, data="subscribe:cancel")
    state = _make_state(data={"conversation_id": conv_id})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))
    cancel_conv = AsyncMock()
    monkeypatch.setattr(subscribe.backend, "cancel_subscription_conversation", cancel_conv)

    with patch("tgbot.handlers.menu.show_main_menu", new_callable=AsyncMock):
        await subscribe.handle_cancel(callback, state)

    cancel_conv.assert_awaited_once_with("api-key", conv_id)


@pytest.mark.asyncio
async def test_menu_text_is_ignored() -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\U0001f4cb Menu")
    state = _make_state()

    await subscribe.process_chat_message(message, state)

    state.set_state.assert_not_awaited(), "menu text was not ignored"


@pytest.mark.asyncio
async def test_fixed_sources_forwarded_to_create(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u043e\u043a")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "\u0422\u0435\u0445 \u043d\u043e\u0432\u043e\u0441\u0442\u0438",
        "short_label": "Tech",
        "delivery_mode": "digest",
        "schedule_cron": None,
        "manual_only": True,
        "format_instructions": "\u043a\u0440\u0430\u0442\u043a\u043e",
        "digest_language": "ru",
        "fixed_telegram_channels": ["durov"],
        "fixed_reddit_subreddits": ["technology"],
        "fixed_twitter_accounts": ["openai"],
        "include_discovered_sources": False,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="\u0413\u043e\u0442\u043e\u0432\u043e!",
            status="ready",
            finalized_config=config,
        ),
    )

    create_calls: list[dict] = []

    async def mock_create(*a, **kw):
        create_calls.append(kw)
        async for ev in _create_stream(sub_id="sub-1"):
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create)

    await subscribe.process_chat_message(message, state)

    assert create_calls[0]["fixed_telegram_channels"] == ["durov"], (
        "fixed telegram channels were not forwarded to create"
    )


@pytest.mark.asyncio
async def test_fixed_sources_manual_only_forwarded(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u043e\u043a")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    config = {
        "prompt_summary": "Tech",
        "short_label": "T",
        "delivery_mode": "digest",
        "schedule_cron": None,
        "manual_only": True,
        "format_instructions": "brief",
        "digest_language": "en",
        "fixed_telegram_channels": ["durov"],
        "fixed_reddit_subreddits": [],
        "fixed_twitter_accounts": [],
        "include_discovered_sources": False,
    }
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        lambda *a, **kw: _stream_simple(
            agent_message="Ready!", status="ready", finalized_config=config
        ),
    )

    create_calls: list[dict] = []

    async def mock_create(*a, **kw):
        create_calls.append(kw)
        async for ev in _create_stream(sub_id="sub-1"):
            yield ev

    monkeypatch.setattr(subscribe.backend, "create_subscription_stream", mock_create)

    await subscribe.process_chat_message(message, state)

    assert create_calls[0]["manual_only"] is True, "manual_only flag was not forwarded to create"


@pytest.mark.asyncio
async def test_stream_error_event_deletes_status_message(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid, text="\u0442\u0435\u0441\u0442")
    state = _make_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    async def mock_error_stream(*a, **kw):
        yield {"event": "error", "detail": "\u043e\u0448\u0438\u0431\u043a\u0430"}

    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation_stream",
        mock_error_stream,
    )

    await subscribe.process_chat_message(message, state)

    status_msg = message.bot.send_message.return_value
    status_msg.delete.assert_awaited_once(), "error event did not delete the status message"
