"""Subscription actions: send now, edit, delete — entered from menu detail view."""

import logging

from aiogram import Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient, ConversationTurnInfo
from tgbot.handlers.subscribe import _delete_status, _edit_status, _edit_to_final, _send_status
from tgbot.language import UILanguage
from tgbot.menu_utils import (
    E_LANGUAGE,
    E_SET_LANG,
    EDIT_CANCEL,
    M_SUB,
    S_CANCEL_DEL,
    S_CONFIRM_DEL,
    S_DELETE,
    S_EDIT,
    S_SEND_NOW,
    back_button,
    edit_menu,
)
from tgbot.storage import get_language_preference, get_ui_language
from tgbot.ui_text import interface_language_name, t
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()


class EditConversation(StatesGroup):
    chatting = State()


# ---------- Send now ----------


@router.callback_query(lambda c: c.data and c.data.startswith(S_SEND_NOW))
async def handle_send_now(callback: CallbackQuery, state: FSMContext) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(S_SEND_NOW) :]
    lang = await _ui_lang(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer(t(lang, "registration_failed"))
        return

    try:
        await backend.send_now(api_key, subscription_id)
        await callback.answer(t(lang, "digest_queued"))
    except Exception:
        logger.exception("Failed to queue digest for subscription %s", subscription_id)
        await callback.answer(t(lang, "digest_queue_failed"))


# ---------- Edit (conversational) ----------


@router.callback_query(lambda c: c.data and c.data.startswith(S_EDIT))
async def handle_edit_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(S_EDIT) :]
    lang = await _ui_lang(callback.from_user.id)

    await state.set_state(EditConversation.chatting)
    await state.update_data(subscription_id=subscription_id, conversation_id=None)

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_cancel"),
                    callback_data=f"{EDIT_CANCEL}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_change_language"),
                    callback_data=f"{E_LANGUAGE}{subscription_id}",
                ),
            ],
        ]
    )
    await edit_menu(callback, state, t(lang, "edit_prompt"), keyboard)


@router.message(EditConversation.chatting)
async def process_edit_message(message: types.Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if text == "\U0001f4cb Menu":
        return
    if not text:
        return

    telegram_id = message.from_user.id
    lang = await _ui_lang(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _reply(message, t(lang, "registration_failed"))
        await state.clear()
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await state.clear()
        return

    conversation_id = state_data.get("conversation_id")
    status_msg = await _send_status(message, t(lang, "status_thinking"))

    try:
        if conversation_id:
            stream = backend.continue_subscription_conversation_stream(
                api_key, conversation_id, text
            )
        else:
            user_info = await backend.get_current_user(api_key)
            language_pref = await get_language_preference(telegram_id)
            user_language = language_pref.code if language_pref else None
            stream = backend.start_subscription_conversation_stream(
                api_key,
                text,
                user_language=user_language,
                user_timezone=user_info.timezone,
                mode="edit",
                subscription_id=subscription_id,
            )

        turn_data: dict | None = None
        async for event_data in stream:
            match event_data.get("event"):
                case "status":
                    key = event_data.get("status_key", "status_thinking")
                    skip = {"event", "status_key"}
                    kwargs = {k: v for k, v in event_data.items() if k not in skip}
                    await _edit_status(status_msg, t(lang, key, **kwargs))
                case "done":
                    turn_data = event_data
                    if not conversation_id:
                        await state.update_data(conversation_id=turn_data["conversation_id"])
                case "error":
                    await _delete_status(status_msg)
                    await _show_edit_error(message, state, lang, subscription_id)
                    return
    except Exception:
        logger.exception(
            "Edit conversation API call failed for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await _delete_status(status_msg)
        await _show_edit_error(message, state, lang, subscription_id)
        return

    if turn_data is None:
        await _delete_status(status_msg)
        await _show_edit_error(message, state, lang, subscription_id)
        return

    turn = ConversationTurnInfo(
        conversation_id=turn_data["conversation_id"],
        agent_message=turn_data["agent_message"],
        status=turn_data["status"],
        finalized_config=turn_data.get("finalized_config"),
    )

    if turn.status == "ready" and turn.finalized_config:
        await _apply_edit_config(
            message, state, lang, subscription_id, turn.finalized_config, status_msg
        )
        return

    cancel_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_cancel"),
                    callback_data=f"{EDIT_CANCEL}{subscription_id}",
                ),
            ],
        ]
    )
    await _edit_to_final(status_msg, turn.agent_message, cancel_keyboard)


async def _apply_edit_config(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    lang: UILanguage,
    subscription_id: str,
    config: dict,
    status_msg: types.Message | None,
) -> None:
    telegram_id = _telegram_id(event)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.apply_subscription_edit_config(api_key, subscription_id, config)
    except Exception:
        logger.exception(
            "Failed to apply edit config for subscription %s",
            subscription_id,
        )
        await _delete_status(status_msg)
        await _show_edit_error(event, state, lang, subscription_id)
        return

    await _delete_status(status_msg)
    await state.clear()

    from tgbot.handlers.menu import show_subscription_detail

    await _reply(event, t(lang, "edit_subscription_updated"))
    await show_subscription_detail(event, state, subscription_id)


async def _show_edit_error(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    lang: UILanguage,
    subscription_id: str,
) -> None:
    await state.clear()
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[back_button(lang, f"{M_SUB}{subscription_id}")]],
    )
    await edit_menu(event, state, t(lang, "edit_subscription_failed"), keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_CANCEL))
async def handle_edit_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(EDIT_CANCEL) :]

    state_data = await state.get_data()
    conversation_id = state_data.get("conversation_id")
    if conversation_id:
        try:
            api_key = await ensure_api_key(callback.from_user.id, backend)
            await backend.cancel_subscription_conversation(api_key, conversation_id)
        except Exception:
            logger.debug("Failed to cancel edit conversation %s", conversation_id)

    await state.clear()

    from tgbot.handlers.menu import show_subscription_detail

    await show_subscription_detail(callback, state, subscription_id)


# ---------- Edit language ----------


@router.callback_query(lambda c: c.data and c.data.startswith(E_LANGUAGE))
async def handle_edit_language(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_LANGUAGE) :]
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_russian"),
                    callback_data=f"{E_SET_LANG}{subscription_id}:ru",
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_english"),
                    callback_data=f"{E_SET_LANG}{subscription_id}:en",
                ),
            ],
            [back_button(lang, f"{M_SUB}{subscription_id}")],
        ]
    )
    await edit_menu(
        callback,
        state,
        t(lang, "edit_subscription_language_prompt"),
        keyboard,
    )


@router.callback_query(lambda c: c.data and c.data.startswith(E_SET_LANG))
async def handle_set_language(callback: CallbackQuery, state: FSMContext) -> None:
    payload = callback.data[len(E_SET_LANG) :]
    subscription_id, digest_language = payload.rsplit(":", maxsplit=1)
    telegram_id = callback.from_user.id
    lang = await _ui_lang(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(api_key, subscription_id, digest_language=digest_language)
        await callback.answer(
            t(
                lang,
                "subscription_language_updated",
                language=interface_language_name(lang, digest_language),
            )
        )
    except Exception:
        logger.exception("Failed to update language for subscription %s", subscription_id)
        await callback.answer(t(lang, "subscription_language_update_failed"))
        return

    from tgbot.handlers.menu import show_subscription_detail

    await show_subscription_detail(callback, state, subscription_id)


# ---------- Delete ----------


@router.callback_query(lambda c: c.data and c.data.startswith(S_DELETE))
async def handle_delete(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(S_DELETE) :]
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_confirm_delete"),
                    callback_data=f"{S_CONFIRM_DEL}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_cancel"),
                    callback_data=f"{S_CANCEL_DEL}{subscription_id}",
                ),
            ]
        ]
    )
    await edit_menu(callback, state, t(lang, "confirm_delete_prompt"), keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(S_CONFIRM_DEL))
async def handle_confirm_delete(callback: CallbackQuery, state: FSMContext) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(S_CONFIRM_DEL) :]
    lang = await _ui_lang(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer(t(lang, "registration_failed"))
        return

    try:
        await backend.delete_subscription(api_key, subscription_id)
        await callback.answer(t(lang, "subscription_deleted"))
    except Exception:
        logger.exception("Failed to delete subscription %s", subscription_id)
        await callback.answer(t(lang, "subscription_delete_failed"))
        return

    # Return to subscription list
    from tgbot.handlers.menu import show_subscription_list

    await show_subscription_list(callback, state)


@router.callback_query(lambda c: c.data and c.data.startswith(S_CANCEL_DEL))
async def handle_cancel_delete(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(S_CANCEL_DEL) :]
    # Return to subscription detail
    from tgbot.handlers.menu import show_subscription_detail

    await show_subscription_detail(callback, state, subscription_id)


# ---------- Helpers ----------


async def callback_answer_via_edit(
    message: types.Message,
    state: FSMContext,
    text: str,
    subscription_id: str,
) -> None:
    """Show a brief result then navigate back to the subscription detail."""
    lang = await _ui_lang(message.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{M_SUB}{subscription_id}")],
        ]
    )
    await edit_menu(message, state, text, keyboard)


def _telegram_id(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_lang(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"


async def _reply(
    event: types.Message | CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Send a new message in the chat."""
    chat_id, bot = _resolve_chat(event)
    if chat_id and bot:
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def _resolve_chat(
    event: types.Message | CallbackQuery,
) -> tuple[int | None, object | None]:
    if hasattr(event, "chat"):
        return event.chat.id, event.bot
    if hasattr(event, "message") and event.message:
        return event.message.chat.id, event.bot
    return None, None
