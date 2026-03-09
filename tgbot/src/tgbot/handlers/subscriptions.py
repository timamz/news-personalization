import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.handlers import start as start_handler
from tgbot.language import UILanguage
from tgbot.storage import get_ui_language
from tgbot.ui_text import interface_language_name, t
from tgbot.user_registry import ensure_api_key
from tgbot.webhook_server import delivery_webhook_url

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

DELETE_PREFIX = "delete_sub:"
SEND_NOW_PREFIX = "send_now:"
EDIT_PREFIX = "edit_sub:"
EDIT_SCHEDULE_PREFIX = "edit_sched:"
DISABLE_SCHEDULE_PREFIX = "disable_sched:"
EDIT_FORMAT_PREFIX = "edit_fmt:"
EDIT_LANGUAGE_PREFIX = "edit_lang:"
SET_LANGUAGE_PREFIX = "set_lang:"
DELIVER_HERE_PREFIX = "deliver_here:"


class EditFlow(StatesGroup):
    waiting_for_schedule = State()
    waiting_for_format = State()


@router.message(Command("list"))
async def cmd_list(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    if not await start_handler.ensure_user_setup(
        message,
        state,
        api_key=api_key,
        next_action=None,
        reset_state=False,
    ):
        return

    ui_language = await _ui_language_or_default(telegram_id)
    try:
        subs = await backend.list_subscriptions(api_key)
    except Exception:
        logger.exception("Failed to list subscriptions for telegram_id=%d", telegram_id)
        await message.answer(t(ui_language, "failed_load_subscriptions"))
        return

    if not subs:
        await message.answer(t(ui_language, "no_subscriptions"))
        return

    for sub in subs:
        topics_str = ", ".join(sub.topics)
        mode_label = (
            t(ui_language, "type_event")
            if sub.delivery_mode == "event"
            else t(ui_language, "type_digest")
        )
        text = t(
            ui_language,
            "subscription_card",
            topics=topics_str,
            type=mode_label,
            language=interface_language_name(ui_language, sub.digest_language),
        )
        buttons = []
        if sub.delivery_mode == "digest":
            buttons.append(
                InlineKeyboardButton(
                    text=t(ui_language, "button_send_now"),
                    callback_data=f"{SEND_NOW_PREFIX}{sub.id}",
                )
            )
        buttons.append(
            InlineKeyboardButton(
                text=t(ui_language, "button_edit"),
                callback_data=f"{EDIT_PREFIX}{sub.delivery_mode}:{sub.id}",
            )
        )
        buttons.append(
            InlineKeyboardButton(
                text=t(ui_language, "button_delete"),
                callback_data=f"{DELETE_PREFIX}{sub.id}",
            )
        )
        keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
        await message.answer(text, reply_markup=keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_PREFIX))
async def handle_edit_menu(callback: CallbackQuery) -> None:
    await callback.answer()
    delivery_mode, subscription_id = _parse_mode_and_subscription_id(callback.data, EDIT_PREFIX)
    if callback.message is None:
        return
    ui_language = await _ui_language_or_default(callback.from_user.id)
    await callback.message.answer(
        t(ui_language, "edit_menu_prompt"),
        reply_markup=_edit_keyboard(subscription_id, delivery_mode, ui_language),
    )


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_LANGUAGE_PREFIX))
async def handle_edit_language(callback: CallbackQuery) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_LANGUAGE_PREFIX)
    if callback.message is None:
        return
    ui_language = await _ui_language_or_default(callback.from_user.id)
    await callback.message.answer(
        t(ui_language, "edit_subscription_language_prompt"),
        reply_markup=_language_keyboard(subscription_id, ui_language),
    )


@router.callback_query(lambda c: c.data and c.data.startswith(SET_LANGUAGE_PREFIX))
async def handle_set_language(callback: CallbackQuery) -> None:
    subscription_id, digest_language = _parse_subscription_language_callback(callback.data)
    telegram_id = callback.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(
            api_key,
            subscription_id,
            digest_language=digest_language,
        )
        await callback.answer(
            t(
                ui_language,
                "subscription_language_updated",
                language=interface_language_name(ui_language, digest_language),
            )
        )
    except Exception:
        logger.exception("Failed to update language for subscription %s", subscription_id)
        await callback.answer(t(ui_language, "subscription_language_update_failed"))


@router.callback_query(lambda c: c.data and c.data.startswith(SEND_NOW_PREFIX))
async def handle_send_now(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(SEND_NOW_PREFIX) :]
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer(t(ui_language, "registration_failed"))
        return

    try:
        await backend.send_now(api_key, subscription_id)
        await callback.answer(t(ui_language, "digest_queued"))
    except Exception:
        logger.exception("Failed to queue digest for subscription %s", subscription_id)
        await callback.answer(t(ui_language, "digest_queue_failed"))


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_SCHEDULE_PREFIX))
async def handle_edit_schedule(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_SCHEDULE_PREFIX)
    await state.set_state(EditFlow.waiting_for_schedule)
    await state.update_data(subscription_id=subscription_id)
    if callback.message:
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await callback.message.answer(t(ui_language, "edit_schedule_prompt"))


@router.message(EditFlow.waiting_for_schedule)
async def process_schedule_edit(message: types.Message, state: FSMContext) -> None:
    schedule_text = (message.text or "").strip()
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not schedule_text:
        await message.answer(t(ui_language, "describe_schedule"))
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await message.answer(t(ui_language, "edit_session_expired"))
        await state.clear()
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        schedule_cron = await backend.parse_schedule(api_key, schedule_text)
        await backend.update_subscription(
            api_key,
            subscription_id,
            schedule_cron=schedule_cron,
        )
        await message.answer(t(ui_language, "schedule_updated"))
    except Exception:
        logger.exception("Failed to update schedule for subscription %s", subscription_id)
        await message.answer(t(ui_language, "schedule_update_failed"))
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(DISABLE_SCHEDULE_PREFIX))
async def handle_disable_schedule(callback: CallbackQuery) -> None:
    subscription_id = _subscription_id_from_callback(callback.data, DISABLE_SCHEDULE_PREFIX)
    telegram_id = callback.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(api_key, subscription_id, schedule_cron=None)
        await callback.answer(t(ui_language, "schedule_disabled"))
    except Exception:
        logger.exception("Failed to disable schedule for subscription %s", subscription_id)
        await callback.answer(t(ui_language, "schedule_disable_failed"))


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_FORMAT_PREFIX))
async def handle_edit_format(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_FORMAT_PREFIX)
    await state.set_state(EditFlow.waiting_for_format)
    await state.update_data(subscription_id=subscription_id)
    if callback.message:
        await callback.message.answer(
            t(await _ui_language_or_default(callback.from_user.id), "edit_format_prompt")
        )


@router.message(EditFlow.waiting_for_format)
async def process_format_edit(message: types.Message, state: FSMContext) -> None:
    format_instructions = (message.text or "").strip()
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not format_instructions:
        await message.answer(t(ui_language, "edit_format_empty"))
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await message.answer(t(ui_language, "edit_session_expired"))
        await state.clear()
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(
            api_key,
            subscription_id,
            format_instructions=format_instructions,
        )
        await message.answer(t(ui_language, "format_updated"))
    except Exception:
        logger.exception("Failed to update format for subscription %s", subscription_id)
        await message.answer(t(ui_language, "format_update_failed"))
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(DELIVER_HERE_PREFIX))
async def handle_deliver_here(callback: CallbackQuery) -> None:
    subscription_id = _subscription_id_from_callback(callback.data, DELIVER_HERE_PREFIX)
    telegram_id = callback.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(
            api_key,
            subscription_id,
            delivery_webhook_url=_webhook_url_for_chat(telegram_id),
        )
        await callback.answer(t(ui_language, "delivery_updated_here"))
    except Exception:
        logger.exception("Failed to update delivery target for subscription %s", subscription_id)
        await callback.answer(t(ui_language, "delivery_update_failed"))


@router.callback_query(lambda c: c.data and c.data.startswith(DELETE_PREFIX))
async def handle_delete(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(DELETE_PREFIX) :]
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer(t(ui_language, "registration_failed"))
        return

    try:
        await backend.delete_subscription(api_key, subscription_id)
        await callback.answer(t(ui_language, "subscription_deleted"))
        if callback.message:
            await callback.message.edit_text(t(ui_language, "subscription_deleted"))
    except Exception:
        logger.exception("Failed to delete subscription %s", subscription_id)
        await callback.answer(t(ui_language, "subscription_delete_failed"))


def _edit_keyboard(
    subscription_id: str,
    delivery_mode: str,
    ui_language: UILanguage,
) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    if delivery_mode == "digest":
        buttons.append(
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_change_schedule"),
                    callback_data=f"{EDIT_SCHEDULE_PREFIX}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_disable_schedule"),
                    callback_data=f"{DISABLE_SCHEDULE_PREFIX}{subscription_id}",
                ),
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(ui_language, "button_change_language"),
                callback_data=f"{EDIT_LANGUAGE_PREFIX}{subscription_id}",
            ),
            InlineKeyboardButton(
                text=t(ui_language, "button_change_format"),
                callback_data=f"{EDIT_FORMAT_PREFIX}{subscription_id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(ui_language, "button_deliver_here"),
                callback_data=f"{DELIVER_HERE_PREFIX}{subscription_id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(ui_language, "button_delete"),
                callback_data=f"{DELETE_PREFIX}{subscription_id}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _parse_mode_and_subscription_id(callback_data: str | None, prefix: str) -> tuple[str, str]:
    if callback_data is None:
        return "digest", ""
    payload = callback_data[len(prefix) :]
    delivery_mode, subscription_id = payload.split(":", maxsplit=1)
    return delivery_mode, subscription_id


def _subscription_id_from_callback(callback_data: str | None, prefix: str) -> str:
    if callback_data is None:
        return ""
    return callback_data[len(prefix) :]


def _language_keyboard(subscription_id: str, ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_russian"),
                    callback_data=f"{SET_LANGUAGE_PREFIX}{subscription_id}:ru",
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_english"),
                    callback_data=f"{SET_LANGUAGE_PREFIX}{subscription_id}:en",
                ),
            ]
        ]
    )


def _parse_subscription_language_callback(callback_data: str | None) -> tuple[str, str]:
    if callback_data is None:
        return "", "en"
    payload = callback_data[len(SET_LANGUAGE_PREFIX) :]
    subscription_id, digest_language = payload.split(":", maxsplit=1)
    return subscription_id, digest_language


def _webhook_url_for_chat(chat_id: int) -> str:
    return delivery_webhook_url(chat_id)


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"
