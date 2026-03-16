"""Shared utilities for menu-based UI navigation."""

import contextlib
import logging

from aiogram import types
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from tgbot.language import UILanguage
from tgbot.ui_text import t

logger = logging.getLogger(__name__)

MENU_MSG_KEY = "_menu_msg_id"

# --- Callback data constants ---

# Main menu navigation
M_MAIN = "m:main"
M_SUBS = "m:subs"
M_SUB = "m:sub:"  # + subscription_id
M_NEW = "m:new"
M_SETTINGS = "m:set"
M_HELP = "m:help"

# Settings
M_SET_LANG = "m:sl"
M_SET_SUB_LANG = "m:ssl"
M_SET_TZ = "m:stz"

# Subscription actions
S_SEND_NOW = "s:now:"
S_EDIT = "s:edit:"
S_DELETE = "s:del:"
S_CONFIRM_DEL = "s:cdel:"
S_CANCEL_DEL = "s:xdel:"

# Edit actions
E_SCHEDULE = "s:esch:"
E_DISABLE_SCHED = "s:dsch:"
E_LANGUAGE = "s:elng:"
E_SET_LANG = "s:slng:"  # + subscription_id:lang
E_REQUEST = "s:ereq:"
E_SOURCES = "s:esrc:"
E_CONFIRM = "s:ecnf:"
E_REVISE = "s:erev:"
E_CANCEL_EDIT = "s:excl:"

# Subscribe flow
SUB_CANCEL = "sub:cancel"


def persistent_keyboard() -> ReplyKeyboardMarkup:
    """The always-visible reply keyboard with the Menu button."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📋 Menu")]],
        resize_keyboard=True,
        is_persistent=True,
    )


async def edit_menu(
    event: types.Message | types.CallbackQuery,
    state: FSMContext,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Edit the menu message in-place, or send a new one if editing fails."""
    # For callback queries, try editing the callback's message directly
    if isinstance(event, types.CallbackQuery) and event.message:
        try:
            await event.message.edit_text(text, reply_markup=reply_markup)
            await state.update_data(**{MENU_MSG_KEY: event.message.message_id})
            return
        except TelegramBadRequest as e:
            if "message is not modified" in str(e):
                return
            logger.debug("Failed to edit callback message: %s", e)
        except Exception:
            logger.debug("Failed to edit callback message")

    # For text messages or failed edits: try editing the stored menu message
    chat_id, bot = _resolve_chat(event)
    if not chat_id or not bot:
        return

    state_data = await state.get_data()
    old_msg_id = state_data.get(MENU_MSG_KEY)

    if old_msg_id:
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=old_msg_id,
                reply_markup=reply_markup,
            )
            return
        except TelegramBadRequest as e:
            if "message is not modified" in str(e):
                return
        except Exception:
            pass
        # Couldn't edit — delete old and send new
        with contextlib.suppress(Exception):
            await bot.delete_message(chat_id=chat_id, message_id=old_msg_id)

    msg = await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    await state.update_data(**{MENU_MSG_KEY: msg.message_id})


async def send_new_menu(
    event: types.Message | types.CallbackQuery,
    state: FSMContext,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Always delete the old menu message and send a fresh one at the bottom."""
    chat_id, bot = _resolve_chat(event)
    if not chat_id or not bot:
        return

    state_data = await state.get_data()
    old_msg_id = state_data.get(MENU_MSG_KEY)
    if old_msg_id:
        with contextlib.suppress(Exception):
            await bot.delete_message(chat_id=chat_id, message_id=old_msg_id)

    msg = await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    await state.update_data(**{MENU_MSG_KEY: msg.message_id})


def main_menu_keyboard(lang: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_subscriptions"),
                    callback_data=M_SUBS,
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_new_subscription"),
                    callback_data=M_NEW,
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t(lang, "button_settings"),
                    callback_data=M_SETTINGS,
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_help"),
                    callback_data=M_HELP,
                ),
            ],
        ]
    )


def back_button(lang: UILanguage, target: str = M_MAIN) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=t(lang, "button_back"), callback_data=target)


def cancel_button(lang: UILanguage) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=t(lang, "button_cancel"), callback_data=SUB_CANCEL)


def _resolve_chat(
    event: types.Message | types.CallbackQuery,
) -> tuple[int | None, object | None]:
    if hasattr(event, "chat"):
        return event.chat.id, event.bot
    if hasattr(event, "message") and event.message:
        return event.message.chat.id, event.bot
    return None, None
