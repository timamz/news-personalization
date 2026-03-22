import logging
import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tgbot.handlers import subscriptions
from tgbot.menu_utils import E_CANCEL_EDIT, E_CONFIRM, S_CONFIRM_DEL, S_DELETE

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


def _make_message(telegram_id: int) -> SimpleNamespace:
    bot = _make_bot()
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        answer=AsyncMock(),
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
async def test_handle_delete_shows_confirmation(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{S_DELETE}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))

    await subscriptions.handle_delete(callback, state)

    callback.bot.edit_message_text.assert_awaited_once()
    text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert "Are you sure" in text, "handle_delete did not show a confirmation prompt"


@pytest.mark.asyncio
async def test_handle_delete_confirmation_button_has_correct_callback_data(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{S_DELETE}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))

    await subscriptions.handle_delete(callback, state)

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
    sub = SimpleNamespace(
        id=sub_id,
        prompt_summary="Новости ИИ",
        canonical_prompt="Новости ИИ каждый день",
        raw_prompt="Новости ИИ каждый день",
        delivery_mode="digest",
        digest_language="ru",
    )
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_set_language(callback, state)

    update_sub.assert_awaited_once_with("api-key", sub_id, digest_language="ru")


@pytest.mark.asyncio
async def test_handle_disable_schedule_updates_subscription(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"s:dsch:{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    update_sub = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "update_subscription", update_sub)
    sub = SimpleNamespace(
        id=sub_id,
        prompt_summary="Новости",
        canonical_prompt="Новости каждый день",
        raw_prompt="Новости каждый день",
        delivery_mode="digest",
        digest_language="en",
    )
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_disable_schedule(callback, state)

    update_sub.assert_awaited_once_with("api-key", sub_id, schedule_cron=None)


@pytest.mark.asyncio
async def test_process_sources_edit_appends_subscription_sources(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    message = _make_message(telegram_id=tid)
    message.text = "@gonzo_ml r/python"
    state = _make_state(data={"subscription_id": sub_id})

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    append_sources = AsyncMock(return_value=SimpleNamespace(added_sources_count=2))
    monkeypatch.setattr(subscriptions.backend, "append_subscription_sources", append_sources)

    await subscriptions.process_sources_edit(message, state)

    append_sources.assert_awaited_once_with(
        "api-key",
        sub_id,
        fixed_telegram_channels=["gonzo_ml"],
        fixed_reddit_subreddits=["python"],
        fixed_twitter_accounts=[],
    )


@pytest.mark.asyncio
async def test_handle_confirm_request_edit_applies_proposal(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    canonical = f"\u041f\u0440\u043e\u043c\u043f\u0442-{uuid.uuid4().hex[:6]}"
    summary = f"\u0420\u0435\u0437\u044e\u043c\u0435-{uuid.uuid4().hex[:6]}"
    fmt = f"\u0424\u043e\u0440\u043c\u0430\u0442-{uuid.uuid4().hex[:6]}"
    callback = _make_callback(telegram_id=tid, data=f"{E_CONFIRM}{sub_id}")
    state = _make_state(
        data={
            "proposed_canonical_prompt": canonical,
            "proposed_prompt_summary": summary,
            "proposed_format_instructions": fmt,
        }
    )

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    apply_edit = AsyncMock()
    monkeypatch.setattr(subscriptions.backend, "apply_subscription_edit", apply_edit)

    with patch("tgbot.handlers.menu.show_subscription_detail", new_callable=AsyncMock):
        await subscriptions.handle_confirm_request_edit(callback, state)

    apply_edit.assert_awaited_once_with(
        "api-key",
        sub_id,
        canonical_prompt=canonical,
        prompt_summary=summary,
        format_instructions=fmt,
    )


@pytest.mark.asyncio
async def test_handle_cancel_request_edit_clears_state(monkeypatch) -> None:
    tid = random.randint(1000, 9999)
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"
    callback = _make_callback(telegram_id=tid, data=f"{E_CANCEL_EDIT}{sub_id}")
    state = _make_state()

    monkeypatch.setattr(subscriptions, "get_ui_language", AsyncMock(return_value="en"))
    sub = SimpleNamespace(
        id=sub_id,
        prompt_summary="Новости",
        canonical_prompt="Новости каждый день",
        raw_prompt="Новости каждый день",
        delivery_mode="digest",
        digest_language="en",
    )
    monkeypatch.setattr(subscriptions, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(subscriptions.backend, "list_subscriptions", AsyncMock(return_value=[sub]))

    await subscriptions.handle_cancel_request_edit(callback, state)

    state.clear.assert_awaited_once(), "handle_cancel_request_edit did not clear the state"
