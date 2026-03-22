"""Subscription actions: send now, edit, delete — entered from menu detail view."""

import logging

from aiogram import Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.language import UILanguage
from tgbot.menu_utils import (
    E_CANCEL_EDIT,
    E_CONFIRM,
    E_DISABLE_SCHED,
    E_LANGUAGE,
    E_REQUEST,
    E_REVISE,
    E_SCHEDULE,
    E_SET_LANG,
    E_SOURCES,
    M_SUB,
    M_SUBS,
    S_CANCEL_DEL,
    S_CONFIRM_DEL,
    S_DELETE,
    S_EDIT,
    S_SEND_NOW,
    back_button,
    edit_menu,
)
from tgbot.source_parser import parse_source_tokens
from tgbot.storage import get_ui_language
from tgbot.ui_text import interface_language_name, t
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()


class EditFlow(StatesGroup):
    waiting_for_schedule = State()
    waiting_for_request = State()
    waiting_for_sources = State()


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


# ---------- Edit menu ----------


@router.callback_query(lambda c: c.data and c.data.startswith(S_EDIT))
async def handle_edit_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(S_EDIT) :]
    await _show_edit_menu(callback, state, subscription_id)


async def _show_edit_menu(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    subscription_id: str,
) -> None:
    lang = await _ui_lang(_telegram_id(event))
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

    text = t(
        lang,
        "edit_menu_header",
        prompt_summary=sub.prompt_summary,
        canonical_prompt=sub.canonical_prompt or sub.raw_prompt or "",
    )

    buttons: list[list[InlineKeyboardButton]] = []
    if sub.delivery_mode == "digest":
        buttons.append(
            [
                InlineKeyboardButton(
                    text=t(lang, "button_change_schedule"),
                    callback_data=f"{E_SCHEDULE}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_disable_schedule"),
                    callback_data=f"{E_DISABLE_SCHED}{subscription_id}",
                ),
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(lang, "button_change_language"),
                callback_data=f"{E_LANGUAGE}{subscription_id}",
            ),
            InlineKeyboardButton(
                text=t(lang, "button_edit_request"),
                callback_data=f"{E_REQUEST}{subscription_id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(lang, "button_add_edit_sources"),
                callback_data=f"{E_SOURCES}{subscription_id}",
            ),
        ]
    )
    buttons.append([back_button(lang, f"{M_SUB}{subscription_id}")])
    await edit_menu(event, state, text, InlineKeyboardMarkup(inline_keyboard=buttons))


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
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
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

    # Return to edit menu
    await _show_edit_menu(callback, state, subscription_id)


# ---------- Edit schedule ----------


@router.callback_query(lambda c: c.data and c.data.startswith(E_SCHEDULE))
async def handle_edit_schedule(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_SCHEDULE) :]
    await state.set_state(EditFlow.waiting_for_schedule)
    await state.update_data(subscription_id=subscription_id)
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
        ]
    )
    await edit_menu(callback, state, t(lang, "edit_schedule_prompt"), keyboard)


@router.message(EditFlow.waiting_for_schedule)
async def process_schedule_edit(message: types.Message, state: FSMContext) -> None:
    schedule_text = (message.text or "").strip()
    lang = await _ui_lang(message.from_user.id)

    if schedule_text == "📋 Menu":
        return

    if not schedule_text:
        await edit_menu(message, state, t(lang, "describe_schedule"))
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await edit_menu(message, state, t(lang, "edit_session_expired"))
        await state.clear()
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        schedule_cron = await backend.parse_schedule(api_key, schedule_text)
        await backend.update_subscription(api_key, subscription_id, schedule_cron=schedule_cron)
    except Exception:
        logger.exception("Failed to update schedule for subscription %s", subscription_id)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [back_button(lang, f"{S_EDIT}{subscription_id}")],
            ]
        )
        await edit_menu(message, state, t(lang, "schedule_update_failed"), keyboard)
        await state.clear()
        return

    await state.clear()
    await callback_answer_via_edit(message, state, t(lang, "schedule_updated"), subscription_id)


# ---------- Disable schedule ----------


@router.callback_query(lambda c: c.data and c.data.startswith(E_DISABLE_SCHED))
async def handle_disable_schedule(callback: CallbackQuery, state: FSMContext) -> None:
    subscription_id = callback.data[len(E_DISABLE_SCHED) :]
    telegram_id = callback.from_user.id
    lang = await _ui_lang(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(api_key, subscription_id, schedule_cron=None)
        await callback.answer(t(lang, "schedule_disabled"))
    except Exception:
        logger.exception("Failed to disable schedule for subscription %s", subscription_id)
        await callback.answer(t(lang, "schedule_disable_failed"))
        return

    await _show_edit_menu(callback, state, subscription_id)


# ---------- Edit request ----------


@router.callback_query(lambda c: c.data and c.data.startswith(E_REQUEST))
async def handle_edit_request(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_REQUEST) :]
    await state.set_state(EditFlow.waiting_for_request)
    await state.update_data(
        subscription_id=subscription_id,
        draft_canonical_prompt=None,
        draft_format_instructions=None,
        proposed_canonical_prompt=None,
        proposed_prompt_summary=None,
        proposed_format_instructions=None,
    )
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
        ]
    )
    await edit_menu(callback, state, t(lang, "edit_request_prompt"), keyboard)


@router.message(EditFlow.waiting_for_request)
async def process_request_edit(message: types.Message, state: FSMContext) -> None:
    change_request = (message.text or "").strip()
    lang = await _ui_lang(message.from_user.id)

    if change_request == "📋 Menu":
        return

    if not change_request:
        await edit_menu(message, state, t(lang, "edit_request_empty"))
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await edit_menu(message, state, t(lang, "edit_session_expired"))
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
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=t(lang, "button_confirm"),
                        callback_data=f"{E_CONFIRM}{subscription_id}",
                    ),
                    InlineKeyboardButton(
                        text=t(lang, "button_revise"),
                        callback_data=f"{E_REVISE}{subscription_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=t(lang, "button_cancel"),
                        callback_data=f"{E_CANCEL_EDIT}{subscription_id}",
                    )
                ],
            ]
        )
        await edit_menu(
            message,
            state,
            t(
                lang,
                "edit_request_preview",
                prompt_summary=proposal.prompt_summary,
                format_instructions=proposal.format_instructions,
                change_summary=proposal.change_summary,
            ),
            keyboard,
        )
    except Exception:
        logger.exception(
            "Failed to propose request edit for subscription %s",
            subscription_id,
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [back_button(lang, f"{S_EDIT}{subscription_id}")],
            ]
        )
        await edit_menu(message, state, t(lang, "edit_request_failed"), keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(E_CONFIRM))
async def handle_confirm_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_CONFIRM) :]
    state_data = await state.get_data()
    lang = await _ui_lang(callback.from_user.id)

    canonical_prompt = state_data.get("proposed_canonical_prompt")
    prompt_summary = state_data.get("proposed_prompt_summary")
    format_instructions = state_data.get("proposed_format_instructions")
    if not all(
        isinstance(v, str) and v for v in (canonical_prompt, prompt_summary, format_instructions)
    ):
        await callback.answer(t(lang, "edit_session_expired"))
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
    except Exception:
        logger.exception(
            "Failed to apply request edit for subscription %s",
            subscription_id,
        )
        await callback.answer(t(lang, "edit_request_apply_failed"))
        return

    await state.clear()
    # Show success and return to detail
    from tgbot.handlers.menu import show_subscription_detail

    await show_subscription_detail(callback, state, subscription_id)


@router.callback_query(lambda c: c.data and c.data.startswith(E_REVISE))
async def handle_revise_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_REVISE) :]
    await state.set_state(EditFlow.waiting_for_request)
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
        ]
    )
    await edit_menu(callback, state, t(lang, "edit_request_prompt"), keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(E_CANCEL_EDIT))
async def handle_cancel_request_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_CANCEL_EDIT) :]
    await state.clear()
    await _show_edit_menu(callback, state, subscription_id)


# ---------- Add sources ----------


@router.callback_query(lambda c: c.data and c.data.startswith(E_SOURCES))
async def handle_add_sources(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = callback.data[len(E_SOURCES) :]
    await state.set_state(EditFlow.waiting_for_sources)
    await state.update_data(subscription_id=subscription_id)
    lang = await _ui_lang(callback.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
        ]
    )
    await edit_menu(callback, state, t(lang, "edit_sources_prompt"), keyboard)


@router.message(EditFlow.waiting_for_sources)
async def process_sources_edit(message: types.Message, state: FSMContext) -> None:
    if (message.text or "").strip() == "📋 Menu":
        return

    lang = await _ui_lang(message.from_user.id)
    channels, subreddits, twitter_accounts = parse_source_tokens(message.text or "")
    if not channels and not subreddits and not twitter_accounts:
        await edit_menu(message, state, t(lang, "sources_parse_failed"))
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await edit_menu(message, state, t(lang, "edit_session_expired"))
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
            text = t(lang, "sources_already_added")
        else:
            text = t(lang, "sources_added", count=result.added_sources_count)
    except Exception:
        logger.exception("Failed to append sources for subscription %s", subscription_id)
        text = t(lang, "sources_add_failed")

    await state.clear()
    await callback_answer_via_edit(message, state, text, subscription_id)


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
    """Show a brief result then navigate back to the edit menu."""
    lang = await _ui_lang(message.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(lang, f"{S_EDIT}{subscription_id}")],
        ]
    )
    await edit_menu(message, state, text, keyboard)


def _telegram_id(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_lang(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"
