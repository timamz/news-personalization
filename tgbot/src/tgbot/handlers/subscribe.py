"""New subscription flow — conversational agent relay with inline buttons."""

import logging

from aiogram import Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient, ConversationTurnInfo
from tgbot.handlers import start as start_handler
from tgbot.language import UILanguage
from tgbot.menu_utils import (
    M_MAIN,
    M_NEW,
    M_SUBS,
    SUB_CANCEL,
    back_button,
    cancel_button,
    edit_menu,
)
from tgbot.storage import get_language_preference, get_ui_language
from tgbot.ui_text import t
from tgbot.user_registry import ensure_api_key
from tgbot.webhook_server import delivery_webhook_url

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

RECENT_EVENTS_YES = "subscribe:recent_events:yes"
RECENT_EVENTS_NO = "subscribe:recent_events:no"
CONV_CHOICE_PREFIX = "subscribe:conv_choice:"


class SubscribeFlow(StatesGroup):
    chatting = State()
    waiting_for_recent_events_decision = State()


# ---------- Entry point (from menu) ----------


@router.callback_query(lambda c: c.data == M_NEW)
async def handle_new_subscription(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    telegram_id = callback.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await edit_menu(callback, state, t("en", "registration_failed"))
        return

    if not await start_handler.ensure_user_setup(
        callback,
        state,
        api_key=api_key,
        next_action="subscribe",
        reset_state=True,
    ):
        return

    await start_subscribe_flow(callback, state)


async def start_subscribe_flow(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    """Public entry: show the prompt step."""
    await state.clear()
    await state.set_state(SubscribeFlow.chatting)
    lang = await _ui_language_for_event(event)
    await edit_menu(
        event,
        state,
        t(lang, "subscription_prompt"),
        _cancel_keyboard(lang),
    )


# ---------- Cancel ----------


@router.callback_query(lambda c: c.data == SUB_CANCEL)
async def handle_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    state_data = await state.get_data()
    conversation_id = state_data.get("conversation_id")

    if conversation_id:
        telegram_id = callback.from_user.id
        try:
            api_key = await ensure_api_key(telegram_id, backend)
            await backend.cancel_subscription_conversation(api_key, conversation_id)
        except Exception:
            logger.debug("Failed to cancel conversation %s", conversation_id)

    await state.clear()
    from tgbot.handlers.menu import show_main_menu

    await show_main_menu(callback, state)


# ---------- Conversation choice callback ----------


@router.callback_query(lambda c: c.data and c.data.startswith(CONV_CHOICE_PREFIX))
async def handle_conversation_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    current_state = await state.get_state()
    if current_state != SubscribeFlow.chatting.state:
        return

    choice_value = callback.data[len(CONV_CHOICE_PREFIX) :]  # noqa: E203
    await _send_user_message(callback, state, choice_value)


# ---------- Free text message ----------


@router.message(SubscribeFlow.chatting)
async def process_chat_message(message: types.Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if text == "📋 Menu":
        return
    if not text:
        return

    await _send_user_message(message, state, text)


# ---------- Recent events flow (kept for event subscriptions) ----------


@router.callback_query(
    lambda c: c.data in {RECENT_EVENTS_YES, RECENT_EVENTS_NO},
    SubscribeFlow.waiting_for_recent_events_decision,
)
async def handle_recent_events_decision(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    ui_language = await _ui_language_or_default(callback.from_user.id)
    if callback.data == RECENT_EVENTS_NO:
        await state.clear()
        from tgbot.handlers.menu import show_subscription_list

        await show_subscription_list(callback, state)
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("created_subscription_id")
    if not isinstance(subscription_id, str) or not subscription_id:
        await edit_menu(callback, state, t(ui_language, "recent_events_expired"))
        await state.clear()
        return

    telegram_id = callback.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await edit_menu(callback, state, t(ui_language, "registration_failed"))
        await state.clear()
        return

    await edit_menu(callback, state, t(ui_language, "recent_events_loading"))
    try:
        recent_events = await backend.list_recent_events(api_key, subscription_id)
    except Exception:
        logger.exception(
            "Failed to load recent events for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await edit_menu(callback, state, t(ui_language, "recent_events_failed"))
        await state.clear()
        return

    if recent_events is None:
        await edit_menu(callback, state, t(ui_language, "recent_events_empty"))
    else:
        if callback.message:
            await callback.message.answer(f"{recent_events.subject}\n\n{recent_events.body}")
        try:
            await backend.acknowledge_recent_events(
                api_key, subscription_id, recent_events.news_item_ids
            )
        except Exception:
            logger.exception(
                "Failed to acknowledge recent events for telegram_id=%d subscription=%s",
                telegram_id,
                subscription_id,
            )

    await state.clear()
    from tgbot.handlers.menu import show_subscription_list

    await show_subscription_list(callback, state)


# ---------- Core relay logic ----------


async def _send_user_message(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    text: str,
) -> None:
    """Send a user message to the backend conversation agent and handle the response."""
    telegram_id = _telegram_id_from_event(event)
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await edit_menu(event, state, t(ui_language, "registration_failed"))
        await state.clear()
        return

    state_data = await state.get_data()
    conversation_id = state_data.get("conversation_id")

    try:
        if conversation_id:
            turn = await backend.continue_subscription_conversation(api_key, conversation_id, text)
        else:
            # First message — start a new conversation
            user_info = await backend.get_current_user(api_key)
            language_pref = await get_language_preference(telegram_id)
            user_language = language_pref.code if language_pref else None

            turn = await backend.start_subscription_conversation(
                api_key,
                text,
                user_language=user_language,
                user_timezone=user_info.timezone,
            )
            await state.update_data(conversation_id=turn.conversation_id)
    except Exception:
        logger.exception(
            "Conversation API call failed for telegram_id=%d",
            telegram_id,
        )
        await edit_menu(
            event,
            state,
            t(ui_language, "failed_process_request"),
            _cancel_keyboard(ui_language),
        )
        return

    if turn.status == "ready" and turn.finalized_config:
        await _create_subscription_from_config(event, state, turn)
        return

    # Show agent response with optional choice buttons
    keyboard = _build_conversation_keyboard(ui_language, turn)
    await edit_menu(event, state, turn.agent_message, keyboard)


async def _create_subscription_from_config(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    turn: ConversationTurnInfo,
) -> None:
    """Create the subscription using the finalized config from the conversation agent."""
    telegram_id = _telegram_id_from_event(event)
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await edit_menu(event, state, t(ui_language, "registration_failed"))
        await state.clear()
        return

    config = turn.finalized_config
    webhook_url = delivery_webhook_url(telegram_id)

    await edit_menu(event, state, t(ui_language, "processing_request"))

    create_kwargs: dict[str, object | None] = {
        "include_discovered_sources": config.get("include_discovered_sources", True),
        "schedule_cron_override": config.get("schedule_cron"),
        "manual_only": config.get("manual_only", False),
        "delivery_mode": config.get("delivery_mode", "digest"),
        "digest_language": config.get("digest_language"),
        "prompt_summary": config.get("prompt_summary"),
        "short_label": config.get("short_label"),
        "format_instructions": config.get("format_instructions"),
        "event_matching_mode": config.get("event_matching_mode"),
    }

    channels = config.get("fixed_telegram_channels", [])
    subreddits = config.get("fixed_reddit_subreddits", [])
    twitter_accounts = config.get("fixed_twitter_accounts", [])

    if channels:
        create_kwargs["fixed_telegram_channels"] = channels
    if subreddits:
        create_kwargs["fixed_reddit_subreddits"] = subreddits
    if twitter_accounts:
        create_kwargs["fixed_twitter_accounts"] = twitter_accounts

    prompt = config.get("prompt_summary", "")
    try:
        subscription = await backend.create_subscription(
            api_key, prompt, webhook_url, **create_kwargs
        )
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [back_button(ui_language, M_MAIN)],
            ]
        )
        await edit_menu(
            event,
            state,
            t(ui_language, "create_subscription_failed"),
            keyboard,
        )
        await state.clear()
        return

    delivery_mode = config.get("delivery_mode", "digest")
    completion_key = (
        "subscription_created_event" if delivery_mode == "event" else "subscription_created_digest"
    )

    if delivery_mode == "event":
        await state.update_data(created_subscription_id=subscription.id)
        await state.set_state(SubscribeFlow.waiting_for_recent_events_decision)
        text = (
            t(ui_language, completion_key, prompt_summary=subscription.prompt_summary)
            + "\n\n"
            + t(ui_language, "show_recent_events_prompt")
        )
        await edit_menu(event, state, text, _recent_events_choice_keyboard(ui_language))
        return

    await state.clear()
    text = t(ui_language, completion_key, prompt_summary=subscription.prompt_summary)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(ui_language, M_SUBS)],
        ]
    )
    await edit_menu(event, state, text, keyboard)


# ---------- Keyboards ----------


def _cancel_keyboard(lang: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[cancel_button(lang)]])


def _build_conversation_keyboard(
    lang: UILanguage,
    turn: ConversationTurnInfo,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    if turn.choices:
        choice_row: list[InlineKeyboardButton] = []
        for choice in turn.choices:
            choice_row.append(
                InlineKeyboardButton(
                    text=choice.label,
                    callback_data=f"{CONV_CHOICE_PREFIX}{choice.value}",
                )
            )
            # Max 2 buttons per row
            if len(choice_row) == 2:
                rows.append(choice_row)
                choice_row = []
        if choice_row:
            rows.append(choice_row)

    rows.append([cancel_button(lang)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _recent_events_choice_keyboard(lang: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(lang, "button_yes_show_recent"),
                    callback_data=RECENT_EVENTS_YES,
                ),
                InlineKeyboardButton(
                    text=t(lang, "button_no_future_only"),
                    callback_data=RECENT_EVENTS_NO,
                ),
            ],
        ]
    )


# ---------- Helpers ----------


def _telegram_id_from_event(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"


async def _ui_language_for_event(event: types.Message | CallbackQuery) -> UILanguage:
    return await _ui_language_or_default(_telegram_id_from_event(event))
