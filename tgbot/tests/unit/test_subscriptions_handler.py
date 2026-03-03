from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import subscriptions


def _mock_message(telegram_id: int):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        answer=AsyncMock(),
    )


def _mock_callback(telegram_id: int, data: str):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        answer=AsyncMock(),
        message=SimpleNamespace(edit_text=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_cmd_list_adds_send_now_button(monkeypatch):
    message = _mock_message(telegram_id=111)
    sub = SimpleNamespace(
        id="sub-1",
        topics=["ai"],
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        format_instructions="brief summary",
    )

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.cmd_list(message)

    message.answer.assert_awaited_once()
    message_text = message.answer.await_args.args[0]
    assert message_text == "Topics: ai\nType: Digest"

    kwargs = message.answer.await_args.kwargs
    keyboard = kwargs["reply_markup"]
    buttons = keyboard.inline_keyboard[0]
    assert buttons[0].text == "Send now"
    assert buttons[0].callback_data == "send_now:sub-1"
    assert buttons[1].text == "Delete"
    assert buttons[1].callback_data == "delete_sub:sub-1"


@pytest.mark.asyncio
async def test_cmd_list_hides_send_now_for_event_subscription(monkeypatch):
    message = _mock_message(telegram_id=111)
    sub = SimpleNamespace(
        id="sub-9",
        topics=["concerts"],
        delivery_mode="event",
        schedule_cron=None,
        format_instructions="brief summary",
    )

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.cmd_list(message)

    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == "Topics: concerts\nType: Event notifications"

    keyboard = message.answer.await_args.kwargs["reply_markup"]
    buttons = keyboard.inline_keyboard[0]
    assert len(buttons) == 1
    assert buttons[0].text == "Delete"
    assert buttons[0].callback_data == "delete_sub:sub-9"


@pytest.mark.asyncio
async def test_handle_send_now_queues_digest(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="send_now:sub-2")

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    send_now = AsyncMock(return_value={"task_id": "task-123", "status": "queued"})
    monkeypatch.setattr(subscriptions.backend, "send_now", send_now)

    await subscriptions.handle_send_now(callback)

    send_now.assert_awaited_once_with("api-key", "sub-2")
    callback.answer.assert_awaited_once_with("Digest queued.")
