import logging
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import start
from tgbot.language import LanguagePreference

logging.disable(logging.CRITICAL)


def _make_message(telegram_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        answer=AsyncMock(),
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(
                return_value=SimpleNamespace(message_id=random.randint(50, 200))
            ),
        ),
    )


def _make_callback(telegram_id: int, data: str) -> SimpleNamespace:
    msg = SimpleNamespace(
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        chat=SimpleNamespace(id=telegram_id),
        message_id=random.randint(10, 100),
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        data=data,
        answer=AsyncMock(),
        message=msg,
        bot=SimpleNamespace(
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
            send_message=AsyncMock(
                return_value=SimpleNamespace(message_id=random.randint(50, 200))
            ),
        ),
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
async def test_cmd_start_shows_language_prompt_when_ui_language_missing(monkeypatch) -> None:
    from tgbot.handlers import menu

    tid = random.randint(1000, 9999)
    message = _make_message(telegram_id=tid)
    state = _make_state()

    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value=None))
    monkeypatch.setattr(menu, "get_ui_language", AsyncMock(return_value=None))
    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value=f"key-{tid}"))

    await start.cmd_start(message, state)

    bot_calls = list(message.bot.edit_message_text.await_args_list) + list(
        message.bot.send_message.await_args_list
    )
    any_lang_text = any(
        "language" in str(c).lower() or "\u044f\u0437\u044b\u043a" in str(c).lower()
        for c in bot_calls
    )
    assert any_lang_text, "cmd_start did not show language selection when ui_language is missing"


@pytest.mark.asyncio
async def test_handle_ui_language_choice_saves_language_and_shows_subscription_selection(
    monkeypatch,
) -> None:
    from tgbot.handlers import menu

    tid = random.randint(1000, 9999)
    callback = _make_callback(telegram_id=tid, data=start.UI_LANGUAGE_RU)
    state = _make_state(
        data={
            "setup_next_action": "subscribe",
            "setup_require_subscription_after_ui": True,
        }
    )

    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(menu, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value=f"key-{tid}"))
    monkeypatch.setattr(start, "get_language_preference", AsyncMock(return_value=None))
    save_ui = AsyncMock()
    monkeypatch.setattr(start, "save_ui_language", save_ui)

    await start.handle_ui_language_choice(callback, state)

    save_ui.assert_awaited_once_with(tid, f"key-{tid}", "ru")
    callback.bot.edit_message_text.assert_awaited_once()
    edit_text = callback.bot.edit_message_text.await_args.kwargs["text"]
    assert (
        "\u043f\u043e\u0434\u043f\u0438\u0441\u043e\u043a" in edit_text.lower()
        or "subscription" in edit_text.lower()
    ), "handle_ui_language_choice did not show subscription language selection"


@pytest.mark.asyncio
async def test_handle_subscription_language_choice_updates_existing_subscriptions(
    monkeypatch,
) -> None:
    from tgbot.handlers import menu

    tid = random.randint(1000, 9999)
    callback = _make_callback(telegram_id=tid, data=start.SUBSCRIPTION_LANGUAGE_FIXED_RU)
    state = _make_state(data={"setup_next_action": "menu"})

    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(menu, "get_ui_language", AsyncMock(return_value="ru"))
    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value=f"key-{tid}"))
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
    sub_id = f"sub-{random.randint(100, 999)}"
    monkeypatch.setattr(
        start.backend,
        "list_subscriptions",
        AsyncMock(return_value=[SimpleNamespace(id=sub_id, digest_language="en")]),
    )
    update_sub = AsyncMock()
    monkeypatch.setattr(start.backend, "update_subscription", update_sub)

    await start.handle_subscription_language_choice(callback, state)

    update_sub.assert_awaited_once_with(f"key-{tid}", sub_id, digest_language="ru")


@pytest.mark.asyncio
async def test_handle_subscription_language_choice_prompts_timezone_when_missing(
    monkeypatch,
) -> None:
    from tgbot.handlers import menu

    tid = random.randint(1000, 9999)
    callback = _make_callback(telegram_id=tid, data=start.SUBSCRIPTION_LANGUAGE_FIXED_EN)
    state = _make_state(data={"setup_next_action": "menu"})

    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(menu, "get_ui_language", AsyncMock(return_value="en"))
    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value=f"key-{tid}"))
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

    state.set_state.assert_awaited_once_with(
        start.SetupFlow.waiting_for_timezone_city,
    )
