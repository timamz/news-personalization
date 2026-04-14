import logging
import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscriptions
from tgbot.menu_utils import EDIT_CANCEL, S_CONFIRM_DEL, S_DELETE, S_EDIT

logging.disable(logging.CRITICAL)


def _make_bot() -> SimpleNamespace:
    return SimpleNamespace(
        edit_message_text=AsyncMock(),
        delete_message=AsyncMock(),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=random.randint(50, 200))),
    )


def _make_callback(telegram_id: int, data: str) -> SimpleNamespace:
    bot = _make_bot()
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        delete=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=random.randint(10, 100),
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        answer=AsyncMock(),
        message=msg,
        bot=bot,
    )


def _make_state(data: dict | None = None) -> SimpleNamespace:
    base = {"_menu_msg_id": random.randint(10, 100)}
    if data:
        base.update(data)
    return SimpleNamespace(
        get_data=AsyncMock(return_value=base),
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_handle_send_now_calls_backend(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"s:now:{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    send_now = AsyncMock(return_value={"task_id": f"task-{uuid.uuid4().hex}", "status": "queued"})
    monkeypatch.setattr(subscriptions.backend, "send_now", send_now)

    await subscriptions.handle_send_now(callback, state)

    send_now.assert_awaited_once_with("api-key", sub_id)


@pytest.mark.asyncio
async def test_handle_delete_shows_confirmation_with_correct_callback(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{S_DELETE}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))

    await subscriptions.handle_delete(callback, state)

    callback.bot.edit_message_text.assert_awaited_once()
    text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert "Are you sure" in text, "handle_delete did not show a confirmation prompt"
    keyboard = callback.bot.edit_message_text.await_args.kwargs["reply_markup"]
    buttons = keyboard.inline_keyboard[0]
    assert buttons[0].callback_data == f"{S_CONFIRM_DEL}{sub_id}", (
        "delete confirmation button did not have the correct callback data"
    )


@pytest.mark.asyncio
async def test_handle_confirm_delete_calls_backend(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{S_CONFIRM_DEL}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    delete_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "delete_subscription", delete_sub)

    with patch("tgbot.handlers.menu.show_subscription_list", new_callable=AsyncMock):
        await subscriptions.handle_confirm_delete(callback, state)

    delete_sub.assert_awaited_once_with("api-key", sub_id)


@pytest.mark.asyncio
async def test_handle_set_language_updates_subscription(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"s:slng:{sub_id}:ru")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_sub)

    with patch("tgbot.handlers.menu.show_subscription_detail", new_callable=AsyncMock):
        await subscriptions.handle_set_language(callback, state)

    update_sub.assert_awaited_once_with("api-key", sub_id, digest_language="ru")


@pytest.mark.asyncio
async def test_handle_edit_menu_enters_conversation_state_and_stores_subscription_id(
    monkeypatch,
) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{S_EDIT}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="ru"))

    await subscriptions.handle_edit_menu(callback, state)

    state.set_state.assert_awaited_once_with(subscriptions.EditConversation.chatting)
    call_kwargs = state.update_data.await_args.kwargs
    assert call_kwargs.get("subscription_id") == sub_id, (
        "handle_edit_menu did not store subscription_id in state"
    )


@pytest.mark.asyncio
async def test_handle_edit_cancel_clears_state(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=EDIT_CANCEL)
    state = _make_state(data={"subscription_id": sub_id, "conversation_id": "conv-123"})

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscriptions.backend, "cancel_subscription_conversation", AsyncMock())

    with patch("tgbot.handlers.menu.show_subscription_detail", new_callable=AsyncMock):
        await subscriptions.handle_edit_cancel(callback, state)

    state.clear.assert_awaited_once(), ("handle_edit_cancel did not clear the state")
