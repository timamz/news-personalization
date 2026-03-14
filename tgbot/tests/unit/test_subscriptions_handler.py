from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscriptions
from tgbot.menu_utils import E_CANCEL_EDIT, E_CONFIRM, S_CONFIRM_DEL, S_DELETE


def _make_bot():
    return SimpleNamespace(
        edit_message_text=AsyncMock(),
        delete_message=AsyncMock(),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
    )


def _mock_callback(telegram_id: int, data: str):
    bot = _make_bot()
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        delete=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=42,
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        answer=AsyncMock(),
        message=msg,
        bot=bot,
    )


def _mock_message(telegram_id: int):
    bot = _make_bot()
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        answer=AsyncMock(),
        bot=bot,
    )


def _mock_state(data=None):
    base = {"_menu_msg_id": 42}
    if data:
        base.update(data)
    return SimpleNamespace(
        get_data=AsyncMock(return_value=base),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _mock_ui_language(monkeypatch) -> None:
    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))


@pytest.mark.asyncio
async def test_handle_send_now_queues_digest(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="s:now:sub-2")
    state = _mock_state()

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    send_now = AsyncMock(return_value={"task_id": "task-123", "status": "queued"})
    monkeypatch.setattr(subscriptions.backend, "send_now", send_now)

    await subscriptions.handle_send_now(callback, state)

    send_now.assert_awaited_once_with("api-key", "sub-2")
    callback.answer.assert_awaited_once_with("📤 Digest queued.")


@pytest.mark.asyncio
async def test_handle_delete_shows_confirmation(monkeypatch):
    callback = _mock_callback(telegram_id=222, data=f"{S_DELETE}sub-2")
    state = _mock_state()

    await subscriptions.handle_delete(callback, state)

    callback.answer.assert_awaited_once()
    # edit_menu edits via bot.edit_message_text since mock isn't real CallbackQuery
    callback.bot.edit_message_text.assert_awaited_once()
    text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert "Are you sure" in text
    keyboard = callback.bot.edit_message_text.await_args.kwargs["reply_markup"]
    buttons = keyboard.inline_keyboard[0]
    assert "Yes, delete" in buttons[0].text
    assert buttons[0].callback_data == f"{S_CONFIRM_DEL}sub-2"


@pytest.mark.asyncio
async def test_handle_confirm_delete_calls_backend(monkeypatch):
    callback = _mock_callback(telegram_id=222, data=f"{S_CONFIRM_DEL}sub-2")
    state = _mock_state()

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    delete_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "delete_subscription", delete_sub)

    with patch("tgbot.handlers.menu.show_subscription_list", new_callable=AsyncMock):
        await subscriptions.handle_confirm_delete(callback, state)

    delete_sub.assert_awaited_once_with("api-key", "sub-2")
    callback.answer.assert_awaited_once_with("🗑 Subscription deleted.")


@pytest.mark.asyncio
async def test_handle_set_language_updates_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="s:slng:sub-2:ru")
    state = _mock_state()

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_sub)
    sub = SimpleNamespace(
        id="sub-2", prompt_summary="AI news", delivery_mode="digest", digest_language="ru"
    )
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_set_language(callback, state)

    update_sub.assert_awaited_once_with("api-key", "sub-2", digest_language="ru")
    callback.answer.assert_awaited_once_with("Language updated to Russian.")


@pytest.mark.asyncio
async def test_handle_disable_schedule_updates_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="s:dsch:sub-2")
    state = _mock_state()

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_sub)
    sub = SimpleNamespace(
        id="sub-2", prompt_summary="AI news", delivery_mode="digest", digest_language="en"
    )
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_disable_schedule(callback, state)

    update_sub.assert_awaited_once_with("api-key", "sub-2", schedule_cron=None)
    callback.answer.assert_awaited_once_with("🚫 Automatic schedule disabled.")


@pytest.mark.asyncio
async def test_process_sources_edit_appends_subscription_sources(monkeypatch):
    message = _mock_message(telegram_id=123)
    message.text = "@gonzo_ml r/python"
    state = _mock_state(data={"subscription_id": "sub-5"})

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    append_sources = AsyncMock(return_value=SimpleNamespace(added_sources_count=2))
    monkeypatch.setattr(subscriptions.backend, "append_subscription_sources", append_sources)

    await subscriptions.process_sources_edit(message, state)

    append_sources.assert_awaited_once_with(
        "api-key",
        "sub-5",
        fixed_telegram_channels=["gonzo_ml"],
        fixed_reddit_subreddits=["python"],
        fixed_twitter_accounts=[],
    )


@pytest.mark.asyncio
async def test_handle_confirm_request_edit_applies_proposal(monkeypatch):
    callback = _mock_callback(telegram_id=222, data=f"{E_CONFIRM}sub-2")
    state = _mock_state(
        data={
            "proposed_canonical_prompt": "Track major anime episode releases.",
            "proposed_prompt_summary": "Major anime episode releases",
            "proposed_format_instructions": "brief summary",
        }
    )

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    apply_edit = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "apply_subscription_edit", apply_edit)

    with patch("tgbot.handlers.menu.show_subscription_detail", new_callable=AsyncMock):
        await subscriptions.handle_confirm_request_edit(callback, state)

    apply_edit.assert_awaited_once_with(
        "api-key",
        "sub-2",
        canonical_prompt="Track major anime episode releases.",
        prompt_summary="Major anime episode releases",
        format_instructions="brief summary",
    )
    state.clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_cancel_request_edit_returns_to_edit_menu(monkeypatch):
    callback = _mock_callback(telegram_id=222, data=f"{E_CANCEL_EDIT}sub-2")
    state = _mock_state()

    sub = SimpleNamespace(
        id="sub-2", prompt_summary="AI news", delivery_mode="digest", digest_language="en"
    )
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_cancel_request_edit(callback, state)

    callback.answer.assert_awaited_once()
    state.clear.assert_awaited_once()
