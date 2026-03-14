import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.handlers import start as start_handler
from tgbot.language import UILanguage
from tgbot.source_parser import parse_source_tokens
from tgbot.storage import get_ui_language
from tgbot.ui_text import interface_language_name, t
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

DELETE_PREFIX = "delete_sub:"
CONFIRM_DELETE_PREFIX = "confirm_del:"
CANCEL_DELETE_PREFIX = "cancel_del:"
SEND_NOW_PREFIX = "send_now:"
EDIT_PREFIX = "edit_sub:"
EDIT_SCHEDULE_PREFIX = "edit_sched:"
DISABLE_SCHEDULE_PREFIX = "disable_sched:"
EDIT_REQUEST_PREFIX = "edit_req:"
EDIT_LANGUAGE_PREFIX = "edit_lang:"
SET_LANGUAGE_PREFIX = "set_lang:"
ADD_SOURCES_PREFIX = "add_sources:"
EDIT_CONFIRM_PREFIX = "edit_confirm:"
EDIT_REVISE_PREFIX = "edit_revise:"
EDIT_CANCEL_PREFIX = "edit_cancel:"


class EditFlow(StatesGroup):
    waiting_for_schedule = State()
    waiting_for_request = State()
    waiting_for_sources = State()


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
        mode_label = (
            t(ui_language, "type_event")
            if sub.delivery_mode == "event"
            else t(ui_language, "type_digest")
        )
        text = t(
            ui_language,
            "subscription_card",
            prompt_summary=sub.prompt_summary,
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


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_REQUEST_PREFIX))
async def handle_edit_request(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_REQUEST_PREFIX)
    await state.set_state(EditFlow.waiting_for_request)
    await state.update_data(
        subscription_id=subscription_id,
        draft_canonical_prompt=None,
        draft_format_instructions=None,
        proposed_canonical_prompt=None,
        proposed_prompt_summary=None,
        proposed_format_instructions=None,
    )
    if callback.message:
        await callback.message.answer(
            t(await _ui_language_or_default(callback.from_user.id), "edit_request_prompt")
        )


@router.callback_query(lambda c: c.data and c.data.startswith(ADD_SOURCES_PREFIX))
async def handle_add_sources(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, ADD_SOURCES_PREFIX)
    await state.set_state(EditFlow.waiting_for_sources)
    await state.update_data(subscription_id=subscription_id)
    if callback.message:
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await callback.message.answer(t(ui_language, "edit_sources_prompt"))


@router.message(EditFlow.waiting_for_request)
async def process_request_edit(message: types.Message, state: FSMContext) -> None:
    change_request = (message.text or "").strip()
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not change_request:
        await message.answer(t(ui_language, "edit_request_empty"))
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
        proposal = await backend.propose_subscription_edit(
            api_key,
            subscription_id,
            change_request=change_request,
            draft_canonical_prompt=state_data.get("draft_canonical_prompt"),
            draft_format_instructions=state_data.get("draft_format_instructions"),
        )
        await state.update_data(
            proposed_canonical_prompt=proposal.canonical_prompt,
            proposed_prompt_summary=proposal.prompt_summary,
            proposed_format_instructions=proposal.format_instructions,
            draft_canonical_prompt=proposal.canonical_prompt,
            draft_format_instructions=proposal.format_instructions,
        )
        await message.answer(
            t(
                ui_language,
                "edit_request_preview",
                prompt_summary=proposal.prompt_summary,
                format_instructions=proposal.format_instructions,
                change_summary=proposal.change_summary,
            ),
            reply_markup=_edit_request_preview_keyboard(subscription_id, ui_language),
        )
    except Exception:
        logger.exception("Failed to propose request edit for subscription %s", subscription_id)
        await message.answer(t(ui_language, "edit_request_failed"))


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_CONFIRM_PREFIX))
async def handle_confirm_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_CONFIRM_PREFIX)
    state_data = await state.get_data()
    ui_language = await _ui_language_or_default(callback.from_user.id)

    canonical_prompt = state_data.get("proposed_canonical_prompt")
    prompt_summary = state_data.get("proposed_prompt_summary")
    format_instructions = state_data.get("proposed_format_instructions")
    if not all(
        isinstance(value, str) and value
        for value in (canonical_prompt, prompt_summary, format_instructions)
    ):
        await callback.answer(t(ui_language, "edit_session_expired"))
        await state.clear()
        return

    try:
        api_key = await ensure_api_key(callback.from_user.id, backend)
        await backend.apply_subscription_edit(
            api_key,
            subscription_id,
            canonical_prompt=canonical_prompt,
            prompt_summary=prompt_summary,
            format_instructions=format_instructions,
        )
        if callback.message:
            await callback.message.answer(
                t(ui_language, "edit_request_applied", prompt_summary=prompt_summary)
            )
    except Exception:
        logger.exception("Failed to apply request edit for subscription %s", subscription_id)
        await callback.answer(t(ui_language, "edit_request_apply_failed"))
        return

    await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_REVISE_PREFIX))
async def handle_revise_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.set_state(EditFlow.waiting_for_request)
    if callback.message:
        await callback.message.answer(
            t(await _ui_language_or_default(callback.from_user.id), "edit_request_prompt")
        )


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_CANCEL_PREFIX))
async def handle_cancel_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    if callback.message:
        await callback.message.answer(
            t(await _ui_language_or_default(callback.from_user.id), "edit_request_cancelled")
        )


@router.message(EditFlow.waiting_for_sources)
async def process_sources_edit(message: types.Message, state: FSMContext) -> None:
    ui_language = await _ui_language_or_default(message.from_user.id)
    channels, subreddits, twitter_accounts = parse_source_tokens(message.text or "")
    if not channels and not subreddits and not twitter_accounts:
        await message.answer(t(ui_language, "sources_parse_failed"))
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
        result = await backend.append_subscription_sources(
            api_key,
            subscription_id,
            fixed_telegram_channels=channels,
            fixed_reddit_subreddits=subreddits,
            fixed_twitter_accounts=twitter_accounts,
        )
        if result.added_sources_count == 0:
            await message.answer(t(ui_language, "sources_already_added"))
        else:
            await message.answer(t(ui_language, "sources_added", count=result.added_sources_count))
    except Exception:
        logger.exception("Failed to append sources for subscription %s", subscription_id)
        await message.answer(t(ui_language, "sources_add_failed"))
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(DELETE_PREFIX))
async def handle_delete(callback: CallbackQuery) -> None:
    await callback.answer()
    subscription_id = callback.data[len(DELETE_PREFIX) :]
    ui_language = await _ui_language_or_default(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_confirm_delete"),
                    callback_data=f"{CONFIRM_DELETE_PREFIX}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_cancel"),
                    callback_data=f"{CANCEL_DELETE_PREFIX}{subscription_id}",
                ),
            ]
        ]
    )
    if callback.message:
        await callback.message.edit_text(
            t(ui_language, "confirm_delete_prompt"),
            reply_markup=keyboard,
        )


@router.callback_query(lambda c: c.data and c.data.startswith(CONFIRM_DELETE_PREFIX))
async def handle_confirm_delete(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(CONFIRM_DELETE_PREFIX) :]
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


@router.callback_query(lambda c: c.data and c.data.startswith(CANCEL_DELETE_PREFIX))
async def handle_cancel_delete(callback: CallbackQuery) -> None:
    ui_language = await _ui_language_or_default(callback.from_user.id)
    await callback.answer(t(ui_language, "delete_cancelled"))
    if callback.message:
        await callback.message.delete()


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
                text=t(ui_language, "button_edit_request"),
                callback_data=f"{EDIT_REQUEST_PREFIX}{subscription_id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(ui_language, "button_add_edit_sources"),
                callback_data=f"{ADD_SOURCES_PREFIX}{subscription_id}",
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


def _edit_request_preview_keyboard(
    subscription_id: str,
    ui_language: UILanguage,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_confirm"),
                    callback_data=f"{EDIT_CONFIRM_PREFIX}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_revise"),
                    callback_data=f"{EDIT_REVISE_PREFIX}{subscription_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_cancel"),
                    callback_data=f"{EDIT_CANCEL_PREFIX}{subscription_id}",
                )
            ],
        ]
    )


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"
