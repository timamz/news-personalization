from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import subscriptions
from tgbot.webhook_server import delivery_webhook_url


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
        message=SimpleNamespace(answer=AsyncMock(), edit_text=AsyncMock()),
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
    assert buttons[1].text == "Edit"
    assert buttons[1].callback_data == "edit_sub:digest:sub-1"
    assert buttons[2].text == "Delete"
    assert buttons[2].callback_data == "delete_sub:sub-1"


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
    assert len(buttons) == 2
    assert buttons[0].text == "Edit"
    assert buttons[0].callback_data == "edit_sub:event:sub-9"
    assert buttons[1].text == "Delete"
    assert buttons[1].callback_data == "delete_sub:sub-9"


@pytest.mark.asyncio
async def test_handle_send_now_queues_digest(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="send_now:sub-2")

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    send_now = AsyncMock(return_value={"task_id": "task-123", "status": "queued"})
    monkeypatch.setattr(subscriptions.backend, "send_now", send_now)

    await subscriptions.handle_send_now(callback)

    send_now.assert_awaited_once_with("api-key", "sub-2")
    callback.answer.assert_awaited_once_with("Digest queued.")


@pytest.mark.asyncio
async def test_handle_edit_menu_shows_digest_edit_actions():
    callback = _mock_callback(telegram_id=222, data="edit_sub:digest:sub-2")

    await subscriptions.handle_edit_menu(callback)

    callback.answer.assert_awaited_once()
    callback.message.answer.assert_awaited_once()
    keyboard = callback.message.answer.await_args.kwargs["reply_markup"]
    first_row = keyboard.inline_keyboard[0]
    second_row = keyboard.inline_keyboard[1]
    assert first_row[0].text == "Change schedule"
    assert first_row[0].callback_data == "edit_sched:sub-2"
    assert first_row[1].text == "Disable schedule"
    assert first_row[1].callback_data == "disable_sched:sub-2"
    assert second_row[0].text == "Change format"
    assert second_row[1].text == "Deliver here"


@pytest.mark.asyncio
async def test_handle_disable_schedule_updates_subscription(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="disable_sched:sub-2")

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_subscription = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_subscription)

    await subscriptions.handle_disable_schedule(callback)

    update_subscription.assert_awaited_once_with("api-key", "sub-2", schedule_cron=None)
    callback.answer.assert_awaited_once_with("Automatic schedule disabled.")


@pytest.mark.asyncio
async def test_process_format_edit_updates_subscription(monkeypatch):
    message = _mock_message(telegram_id=123)
    message.text = "bullet list with links"
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={"subscription_id": "sub-5"}),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_subscription = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_subscription)

    await subscriptions.process_format_edit(message, state)

    update_subscription.assert_awaited_once_with(
        "api-key",
        "sub-5",
        format_instructions="bullet list with links",
    )
    message.answer.assert_awaited_once_with("Format updated.")
    state.clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_deliver_here_updates_webhook(monkeypatch):
    callback = _mock_callback(telegram_id=222, data="deliver_here:sub-2")

    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_subscription = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_subscription)

    await subscriptions.handle_deliver_here(callback)

    update_subscription.assert_awaited_once_with(
        "api-key",
        "sub-2",
        delivery_webhook_url=delivery_webhook_url(222),
    )
    callback.answer.assert_awaited_once_with("Delivery updated to this chat.")
