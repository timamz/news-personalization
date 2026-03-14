from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import start
from tgbot.language import LanguagePreference


def _mock_message(telegram_id: int):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        ),
    )


def _mock_callback(telegram_id: int, data: str):
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=42,
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        answer=AsyncMock(),
        message=msg,
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
        update_data=AsyncMock(),
        set_state=AsyncMock(),
        clear=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _mock_storage(monkeypatch) -> None:
    from tgbot.handlers import menu

    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(menu, "get_ui_language", AsyncMock(return_value="en"))


@pytest.mark.asyncio
async def test_cmd_start_prompts_for_ui_language_when_missing(monkeypatch) -> None:
    message = _mock_message(telegram_id=123)
    state = _mock_state()

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value=None))

    await start.cmd_start(message, state)

    # Should show welcome + persistent keyboard first
    message.answer.assert_awaited_once()
    # edit_menu should edit the stored menu message or send a new one
    bot_calls = list(message.bot.edit_message_text.await_args_list) + list(
        message.bot.send_message.await_args_list
    )
    assert bot_calls, "Expected edit_menu to call bot.edit_message_text or send_message"
    any_lang_text = any("language" in str(c).lower() or "язык" in str(c).lower() for c in bot_calls)
    assert any_lang_text


@pytest.mark.asyncio
async def test_handle_ui_language_choice_prompts_subscription_language_when_missing(
    monkeypatch,
) -> None:
    callback = _mock_callback(telegram_id=123, data=start.UI_LANGUAGE_RU)
    state = _mock_state(
        data={
            "setup_next_action": "subscribe",
            "setup_require_subscription_after_ui": True,
        }
    )

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_language_preference", AsyncMock(return_value=None))
    monkeypatch.setattr(start, "save_ui_language", AsyncMock())

    await start.handle_ui_language_choice(callback, state)

    start.save_ui_language.assert_awaited_once_with(123, "api-key", "ru")
    # Should show subscription language selection via edit_menu (bot.edit_message_text)
    callback.bot.edit_message_text.assert_awaited_once()
    edit_text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert "подписок" in edit_text.lower() or "subscription" in edit_text.lower()


@pytest.mark.asyncio
async def test_handle_subscription_language_choice_updates_existing_subscriptions(
    monkeypatch,
) -> None:
    callback = _mock_callback(telegram_id=123, data=start.SUBSCRIPTION_LANGUAGE_FIXED_RU)
    state = _mock_state(data={"setup_next_action": "menu"})

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(
        start,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="ask", code=None)),
    )
    monkeypatch.setattr(start, "save_language_preference", AsyncMock())
    monkeypatch.setattr(
        start.backend,
        "get_current_user",
        AsyncMock(return_value=SimpleNamespace(timezone="UTC")),
    )
    monkeypatch.setattr(
        start.backend,
        "list_subscriptions",
        AsyncMock(return_value=[SimpleNamespace(id="sub-1", digest_language="en")]),
    )
    update_subscription = AsyncMock()
    monkeypatch.setattr(start.backend, "update_subscription", update_subscription)

    await start.handle_subscription_language_choice(callback, state)

    update_subscription.assert_awaited_once_with("api-key", "sub-1", digest_language="ru")
    # Confirmation message sent via _answer (event.message.answer)
    callback.message.answer.assert_awaited()
    all_texts = " ".join(str(c) for c in callback.message.answer.await_args_list)
    assert "Обновлено" in all_texts or "Updated" in all_texts


@pytest.mark.asyncio
async def test_handle_subscription_language_choice_prompts_timezone_when_missing(
    monkeypatch,
) -> None:
    callback = _mock_callback(telegram_id=123, data=start.SUBSCRIPTION_LANGUAGE_FIXED_EN)
    state = _mock_state(data={"setup_next_action": "menu"})

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(
        start,
        "get_language_preference",
        AsyncMock(return_value=LanguagePreference(mode="ask", code=None)),
    )
    monkeypatch.setattr(start, "save_language_preference", AsyncMock())
    monkeypatch.setattr(start.backend, "list_subscriptions", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        start.backend,
        "get_current_user",
        AsyncMock(return_value=SimpleNamespace(timezone=None)),
    )

    await start.handle_subscription_language_choice(callback, state)

    state.set_state.assert_awaited_once_with(start.SetupFlow.waiting_for_timezone_city)
