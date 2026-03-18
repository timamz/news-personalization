from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.client import ConversationChoiceInfo, ConversationTurnInfo
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


def _conversation_turn(
    *,
    conversation_id: str = "conv-123",
    agent_message: str = "What schedule?",
    status: str = "in_progress",
    choices: list[ConversationChoiceInfo] | None = None,
    finalized_config: dict | None = None,
) -> ConversationTurnInfo:
    return ConversationTurnInfo(
        conversation_id=conversation_id,
        agent_message=agent_message,
        status=status,
        choices=choices,
        finalized_config=finalized_config,
    )


@pytest.fixture(autouse=True)
def _mock_ui_language(monkeypatch) -> None:
    monkeypatch.setattr(subscribe, "get_ui_language", AsyncMock(return_value="en"))


# ---------- First message starts conversation ----------


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

    turn = _conversation_turn(
        agent_message="How often would you like to receive the digest?",
        choices=[
            ConversationChoiceInfo(label="Every morning", value="every morning"),
            ConversationChoiceInfo(label="Manual only", value="manual only"),
        ],
    )
    monkeypatch.setattr(
        subscribe.backend,
        "start_subscription_conversation",
        AsyncMock(return_value=turn),
    )

    await subscribe.process_chat_message(message, state)

    subscribe.backend.start_subscription_conversation.assert_awaited_once()
    state.update_data.assert_awaited()


# ---------- Continued message relays to backend ----------


@pytest.mark.asyncio
async def test_continued_message_relays_to_backend(monkeypatch):
    message = _mock_message(telegram_id=111, text="every morning")
    state = _mock_state(data={"conversation_id": "conv-123"})

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    turn = _conversation_turn(agent_message="Do you know specific sources?")
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation",
        AsyncMock(return_value=turn),
    )

    await subscribe.process_chat_message(message, state)

    subscribe.backend.continue_subscription_conversation.assert_awaited_once_with(
        "api-key", "conv-123", "every morning"
    )


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
    turn = _conversation_turn(
        agent_message="Your subscription is ready!",
        status="ready",
        finalized_config=config,
    )
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation",
        AsyncMock(return_value=turn),
    )

    create_sub = AsyncMock(
        return_value=SimpleNamespace(id="sub-new", prompt_summary="AI news digest")
    )
    monkeypatch.setattr(subscribe.backend, "create_subscription", create_sub)

    await subscribe.process_chat_message(message, state)

    create_sub.assert_awaited_once()
    call_kwargs = create_sub.await_args.kwargs
    assert call_kwargs["delivery_mode"] == "digest"
    assert call_kwargs["schedule_cron_override"] == "0 8 * * *"
    assert call_kwargs["include_discovered_sources"] is True


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
    turn = _conversation_turn(
        agent_message="Ready!",
        status="ready",
        finalized_config=config,
    )
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation",
        AsyncMock(return_value=turn),
    )
    monkeypatch.setattr(
        subscribe.backend,
        "create_subscription",
        AsyncMock(return_value=SimpleNamespace(id="sub-evt", prompt_summary="Concerts")),
    )

    await subscribe.process_chat_message(message, state)

    state.set_state.assert_awaited_with(subscribe.SubscribeFlow.waiting_for_recent_events_decision)


# ---------- Choice callback sends value as message ----------


@pytest.mark.asyncio
async def test_choice_callback_relays_value(monkeypatch):
    callback = _mock_callback(telegram_id=111, data=f"{subscribe.CONV_CHOICE_PREFIX}every morning")
    state = _mock_state(data={"conversation_id": "conv-123"})
    state.get_state = AsyncMock(return_value=subscribe.SubscribeFlow.chatting.state)

    monkeypatch.setattr(subscribe, "ensure_api_key", AsyncMock(return_value="api-key"))

    turn = _conversation_turn(agent_message="Got it, every morning!")
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation",
        AsyncMock(return_value=turn),
    )

    await subscribe.handle_conversation_choice(callback, state)

    subscribe.backend.continue_subscription_conversation.assert_awaited_once_with(
        "api-key", "conv-123", "every morning"
    )


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
    turn = _conversation_turn(
        agent_message="Ready!",
        status="ready",
        finalized_config=config,
    )
    monkeypatch.setattr(
        subscribe.backend,
        "continue_subscription_conversation",
        AsyncMock(return_value=turn),
    )

    create_sub = AsyncMock(return_value=SimpleNamespace(id="sub-1", prompt_summary="Tech news"))
    monkeypatch.setattr(subscribe.backend, "create_subscription", create_sub)

    await subscribe.process_chat_message(message, state)

    create_sub.assert_awaited_once()
    call_kwargs = create_sub.await_args.kwargs
    assert call_kwargs["fixed_telegram_channels"] == ["durov"]
    assert call_kwargs["fixed_reddit_subreddits"] == ["technology"]
    assert call_kwargs["fixed_twitter_accounts"] == ["openai"]
    assert call_kwargs["include_discovered_sources"] is False
    assert call_kwargs["manual_only"] is True
