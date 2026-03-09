from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.filters import Command

from tgbot.handlers import start
from tgbot.language import LanguagePreference
from tgbot.ui_text import t


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
        message=SimpleNamespace(answer=AsyncMock()),
    )


def test_cmd_help_uses_command_filter() -> None:
    help_handler = next(
        handler
        for handler in start.router.message.handlers
        if handler.callback.__name__ == "cmd_help"
    )

    command_filters = [
        filter_obj.callback
        for filter_obj in help_handler.filters
        if isinstance(filter_obj.callback, Command)
    ]

    assert command_filters
    assert command_filters[0].commands == ("help",)


@pytest.mark.asyncio
async def test_cmd_start_prompts_for_ui_language_when_missing(monkeypatch) -> None:
    message = _mock_message(telegram_id=123)
    state = SimpleNamespace(clear=AsyncMock(), update_data=AsyncMock())

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_ui_language", AsyncMock(return_value=None))

    await start.cmd_start(message, state)

    state.clear.assert_awaited_once()
    state.update_data.assert_awaited_once_with(
        setup_next_action="welcome",
        setup_require_subscription_after_ui=True,
    )
    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == t("en", "ui_language_initial")
    assert message.answer.await_args.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_handle_ui_language_choice_prompts_subscription_language_when_missing(
    monkeypatch,
) -> None:
    callback = _mock_callback(telegram_id=123, data=start.UI_LANGUAGE_RU)
    state = SimpleNamespace(
        get_data=AsyncMock(
            return_value={
                "setup_next_action": "subscribe",
                "setup_require_subscription_after_ui": True,
            }
        ),
        update_data=AsyncMock(),
        clear=AsyncMock(),
    )

    monkeypatch.setattr(start, "ensure_api_key", AsyncMock(return_value="api-key"))
    monkeypatch.setattr(start, "get_language_preference", AsyncMock(return_value=None))
    monkeypatch.setattr(start, "save_ui_language", AsyncMock())

    await start.handle_ui_language_choice(callback, state)

    start.save_ui_language.assert_awaited_once_with(123, "api-key", "ru")
    state.clear.assert_not_awaited()
    callback.message.answer.assert_awaited_once()
    assert callback.message.answer.await_args.args[0] == t("ru", "subscription_language_initial")


@pytest.mark.asyncio
async def test_handle_subscription_language_choice_updates_existing_subscriptions(
    monkeypatch,
) -> None:
    callback = _mock_callback(telegram_id=123, data=start.SUBSCRIPTION_LANGUAGE_FIXED_RU)
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={"setup_next_action": "welcome"}),
        update_data=AsyncMock(),
        clear=AsyncMock(),
    )

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
        "list_subscriptions",
        AsyncMock(return_value=[SimpleNamespace(id="sub-1", digest_language="en")]),
    )
    update_subscription = AsyncMock()
    monkeypatch.setattr(start.backend, "update_subscription", update_subscription)

    await start.handle_subscription_language_choice(callback, state)

    update_subscription.assert_awaited_once_with(
        "api-key",
        "sub-1",
        digest_language="ru",
    )
    state.update_data.assert_awaited_once_with(
        setup_next_action=None,
        setup_require_subscription_after_ui=False,
    )
    state.clear.assert_awaited_once()
    assert (
        "Обновлено существующих подписок: 1." in callback.message.answer.await_args_list[0].args[0]
    )
    assert callback.message.answer.await_args_list[1].args[0] == t("ru", "welcome")
