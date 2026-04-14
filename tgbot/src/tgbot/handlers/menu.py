"""Central menu handler — main menu, settings, help, subscription list & detail."""

import contextlib
import logging

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.language import UILanguage
from tgbot.menu_utils import (
    E_LANGUAGE,
    M_HELP,
    M_MAIN,
    M_NEW,
    M_SET_LANG,
    M_SET_SUB_LANG,
    M_SET_TZ,
    M_SETTINGS,
    M_SUB,
    M_SUBS,
    S_DELETE,
    S_EDIT,
    S_SEND_NOW,
    back_button,
    edit_menu,
    main_menu_keyboard,
    send_new_menu,
)
from tgbot.storage import get_ui_language
from tgbot.ui_text import interface_language_name, t
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

MENU_TEXT = "📋 Menu"


# ---------- Menu button (reply keyboard text) ----------


@router.message(F.text == MENU_TEXT)
async def handle_menu_button(message: types.Message, state: FSMContext) -> None:
    """User tapped the persistent 📋 Menu button."""
    # Delete user's "📋 Menu" text message to keep chat clean
    with contextlib.suppress(Exception):
        await message.delete()

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    from tgbot.handlers.start import ensure_user_setup

    if not await ensure_user_setup(
        message, state, api_key=api_key, next_action="menu", reset_state=True
    ):
        return

    lang = await _ui_lang(message)
    await send_new_menu(message, state, t(lang, "menu_title"), main_menu_keyboard(lang))


# ---------- Main menu ----------


@router.callback_query(lambda c: c.data == M_MAIN)
async def handle_main_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.set_state(None)
    lang = await _ui_lang(callback)
    await edit_menu(callback, state, t(lang, "menu_title"), main_menu_keyboard(lang))


async def show_main_menu(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    """Public entry for showing the main menu (called from start.py etc.)."""
    lang = await _ui_lang(event)
    await send_new_menu(event, state, t(lang, "menu_title"), main_menu_keyboard(lang))


# ---------- Help ----------


@router.callback_query(lambda c: c.data == M_HELP)
async def handle_help(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    lang = await _ui_lang(callback)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[back_button(lang, M_MAIN)]])
    await edit_menu(callback, state, t(lang, "help_text"), keyboard)


# ---------- Settings ----------


@router.callback_query(lambda c: c.data == M_SETTINGS)
async def handle_settings(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await _show_settings(callback, state)


async def _show_settings(event: types.Message | CallbackQuery, state: FSMContext) -> None:
    lang = await _ui_lang(event)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_interface_language"),
                    callback_data=M_SET_LANG,
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t(lang, "button_sub_language_setting"),
                    callback_data=M_SET_SUB_LANG,
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t(lang, "button_timezone_setting"),
                    callback_data=M_SET_TZ,
                ),
            ],
            [back_button(lang, M_MAIN)],
        ]
    )
    await edit_menu(event, state, t(lang, "settings_title"), keyboard)


# ---------- Subscription list ----------


@router.callback_query(lambda c: c.data == M_SUBS)
async def handle_subscription_list(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await show_subscription_list(callback, state)


async def show_subscription_list(event: types.Message | CallbackQuery, state: FSMContext) -> None:
    """Show the subscription list in the menu message."""
    lang = await _ui_lang(event)
    telegram_id = _telegram_id(event)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        subs = await backend.list_subscriptions(api_key)
    except Exception:
        logger.exception("Failed to list subscriptions for telegram_id=%d", telegram_id)
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[back_button(lang, M_MAIN)]])
        await edit_menu(event, state, t(lang, "failed_load_subscriptions"), keyboard)
        return

    if not subs:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=t(lang, "button_create_one"),
                        callback_data=M_NEW,
                    )
                ],
                [back_button(lang, M_MAIN)],
            ]
        )
        text = f"{t(lang, 'subscriptions_title')}\n\n{t(lang, 'subscriptions_empty_hint')}"
        await edit_menu(event, state, text, keyboard)
        return

    buttons: list[list[InlineKeyboardButton]] = []
    for sub in subs:
        label = sub.display_label
        if len(label) > 28:
            label = label[:25] + "..."
        buttons.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"{M_SUB}{sub.id}",
                )
            ]
        )
    buttons.append([back_button(lang, M_MAIN)])
    await edit_menu(
        event,
        state,
        t(lang, "subscriptions_title"),
        InlineKeyboardMarkup(inline_keyboard=buttons),
    )


# ---------- Subscription detail ----------


@router.callback_query(lambda c: c.data and c.data.startswith(M_SUB))
async def handle_subscription_detail(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(M_SUB) :]
    await show_subscription_detail(callback, state, subscription_id)


async def show_subscription_detail(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    subscription_id: str,
) -> None:
    """Show a single subscription's detail view with action buttons."""
    lang = await _ui_lang(event)
    telegram_id = _telegram_id(event)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        subs = await backend.list_subscriptions(api_key)
    except Exception:
        logger.exception("Failed to load subscription %s", subscription_id)
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[back_button(lang, M_SUBS)]])
        await edit_menu(event, state, t(lang, "failed_load_subscriptions"), keyboard)
        return

    sub = next((s for s in subs if s.id == subscription_id), None)
    if sub is None:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[back_button(lang, M_SUBS)]])
        await edit_menu(event, state, t(lang, "subscription_deleted"), keyboard)
        return

    mode_label = t(lang, "type_event") if sub.delivery_mode == "event" else t(lang, "type_digest")
    text = t(
        lang,
        "subscription_detail",
        prompt_summary=sub.display_label,
        canonical_prompt=sub.topic,
        type=mode_label,
        language=interface_language_name(lang, sub.digest_language),
    )

    buttons: list[list[InlineKeyboardButton]] = []
    if sub.delivery_mode == "digest":
        buttons.append(
            [
                InlineKeyboardButton(
                    text=t(lang, "button_send_now"),
                    callback_data=f"{S_SEND_NOW}{sub.id}",
                ),
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(lang, "button_edit"),
                callback_data=f"{S_EDIT}{sub.id}",
            ),
            InlineKeyboardButton(
                text=t(lang, "button_change_language"),
                callback_data=f"{E_LANGUAGE}{sub.id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(lang, "button_delete"),
                callback_data=f"{S_DELETE}{sub.id}",
            ),
        ]
    )
    buttons.append([back_button(lang, M_SUBS)])
    await edit_menu(event, state, text, InlineKeyboardMarkup(inline_keyboard=buttons))


# ---------- Helpers ----------


def _telegram_id(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_lang(event: types.Message | CallbackQuery) -> UILanguage:
    return await get_ui_language(_telegram_id(event)) or "en"
