"""New subscription flow — fully conversational agent relay with live status."""

import contextlib
import logging

from aiogram import Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, MessageEntity

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
    send_new_menu,
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

_SPINNER_EMOJI_ID = "5386367538735104399"


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
    await _reply(event, t(lang, "subscription_prompt"))


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

    # Remove the buttons from the original message
    if callback.message:
        with contextlib.suppress(TelegramBadRequest):
            await callback.message.edit_reply_markup(reply_markup=None)

    status_msg = await _send_status(callback, t(ui_language, "recent_events_loading"))

    preview: dict | None = None
    try:
        async for event_data in backend.list_recent_events_stream(api_key, subscription_id):
            match event_data.get("event"):
                case "status":
                    key = event_data.get("status_key", "status_thinking")
                    skip = {"event", "status_key"}
                    kwargs = {k: v for k, v in event_data.items() if k not in skip}
                    await _edit_status(status_msg, t(ui_language, key, **kwargs))
                case "done":
                    preview = event_data.get("preview")
    except Exception:
        logger.exception(
            "Failed to load recent events for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await _delete_status(status_msg)
        await state.clear()
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[back_button(ui_language, M_SUBS)]],
        )
        await edit_menu(callback, state, t(ui_language, "recent_events_failed"), keyboard)
        return

    await _delete_status(status_msg)

    if preview is None:
        await state.clear()
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[back_button(ui_language, M_SUBS)]],
        )
        await _reply(callback, t(ui_language, "recent_events_empty"), keyboard)
        return

    await _reply(callback, f"{preview['subject']}\n\n{preview['body']}")
    try:
        await backend.acknowledge_recent_events(api_key, subscription_id, preview["news_item_ids"])
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
        await _reply(event, t(ui_language, "registration_failed"))
        await state.clear()
        return

    state_data = await state.get_data()
    conversation_id = state_data.get("conversation_id")

    # Send initial status message with animated emoji
    status_msg = await _send_status(event, t(ui_language, "status_thinking"))

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
                api_key, text, user_language=user_language, user_timezone=user_info.timezone
            )

        turn_data: dict | None = None
        async for event_data in stream:
            match event_data.get("event"):
                case "status":
                    key = event_data.get("status_key", "status_thinking")
                    skip = {"event", "status_key"}
                    kwargs = {k: v for k, v in event_data.items() if k not in skip}
                    await _edit_status(status_msg, t(ui_language, key, **kwargs))
                case "done":
                    turn_data = event_data
                    if not conversation_id:
                        await state.update_data(conversation_id=turn_data["conversation_id"])
                case "error":
                    await _delete_status(status_msg)
                    await _reply(
                        event,
                        t(ui_language, "failed_process_request"),
                        _cancel_keyboard(ui_language),
                    )
                    return
    except Exception:
        logger.exception("Conversation API call failed for telegram_id=%d", telegram_id)
        await _delete_status(status_msg)
        await _reply(
            event,
            t(ui_language, "failed_process_request"),
            _cancel_keyboard(ui_language),
        )
        return

    if turn_data is None:
        await _delete_status(status_msg)
        await _reply(event, t(ui_language, "failed_process_request"), _cancel_keyboard(ui_language))
        return

    turn = ConversationTurnInfo(
        conversation_id=turn_data["conversation_id"],
        agent_message=turn_data["agent_message"],
        status=turn_data["status"],
        finalized_config=turn_data.get("finalized_config"),
    )

    if turn.status == "ready" and turn.finalized_config:
        await _create_subscription_from_config(event, state, turn, status_msg)
        return

    # Edit the status message into the final agent response
    await _edit_to_final(status_msg, turn.agent_message, _cancel_keyboard(ui_language))


async def _create_subscription_from_config(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    turn: ConversationTurnInfo,
    status_msg: types.Message | None = None,
) -> None:
    """Create the subscription using the finalized config from the conversation agent."""
    telegram_id = _telegram_id_from_event(event)
    ui_language = await _ui_language_or_default(telegram_id)

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _delete_status(status_msg)
        await _reply(event, t(ui_language, "registration_failed"))
        await state.clear()
        return

    config = turn.finalized_config
    webhook_url = delivery_webhook_url(telegram_id)

    create_kwargs: dict[str, object | None] = {
        "include_discovered_sources": config.get("include_discovered_sources", True),
        "schedule_cron_override": config.get("schedule_cron"),
        "manual_only": config.get("manual_only", False),
        "delivery_mode": config.get("delivery_mode", "digest"),
        "digest_language": config.get("digest_language"),
        "prompt_summary": config.get("prompt_summary"),
        "short_label": config.get("short_label"),
        "format_instructions": config.get("format_instructions"),
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
    subscription_data: dict | None = None
    try:
        async for event_data in backend.create_subscription_stream(
            api_key, prompt, webhook_url, **create_kwargs
        ):
            match event_data.get("event"):
                case "status":
                    key = event_data.get("status_key", "status_thinking")
                    skip = {"event", "status_key"}
                    kwargs = {k: v for k, v in event_data.items() if k not in skip}
                    await _edit_status(status_msg, t(ui_language, key, **kwargs))
                case "done":
                    subscription_data = event_data["subscription"]
                case "error":
                    raise RuntimeError(event_data.get("detail", "Unknown error"))
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        await _delete_status(status_msg)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [back_button(ui_language, M_MAIN)],
            ]
        )
        await _reply(event, t(ui_language, "create_subscription_failed"), keyboard)
        await state.clear()
        return

    if subscription_data is None:
        await _delete_status(status_msg)
        await _reply(event, t(ui_language, "create_subscription_failed"))
        await state.clear()
        return

    subscription = backend._parse_subscription(subscription_data)
    delivery_mode = config.get("delivery_mode", "digest")
    completion_key = (
        "subscription_created_event" if delivery_mode == "event" else "subscription_created_digest"
    )

    await _delete_status(status_msg)

    if delivery_mode == "event":
        await state.update_data(created_subscription_id=subscription.id)
        await state.set_state(SubscribeFlow.waiting_for_recent_events_decision)
        text = (
            t(ui_language, completion_key, prompt_summary=subscription.prompt_summary)
            + "\n\n"
            + t(ui_language, "show_recent_events_prompt")
        )
        await _reply(event, text, _recent_events_choice_keyboard(ui_language))
        return

    await state.clear()
    text = t(ui_language, completion_key, prompt_summary=subscription.prompt_summary)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [back_button(ui_language, M_SUBS)],
        ]
    )
    await send_new_menu(event, state, text, keyboard)


# ---------- Keyboards ----------


def _cancel_keyboard(lang: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[cancel_button(lang)]])


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


# ---------- Status message helpers ----------


def _build_status_text(text: str) -> tuple[str, list[MessageEntity]]:
    """Build text with custom animated emoji prepended."""
    placeholder = "\u2b50"
    full_text = f"{placeholder} {text}"
    entity = MessageEntity(
        type="custom_emoji",
        offset=0,
        length=len(placeholder),
        custom_emoji_id=_SPINNER_EMOJI_ID,
    )
    return full_text, [entity]


async def _send_status(event: types.Message | CallbackQuery, text: str) -> types.Message | None:
    """Send a status message with animated custom emoji."""
    chat_id, bot = _resolve_chat(event)
    if not chat_id or not bot:
        return None
    full_text, entities = _build_status_text(text)
    try:
        return await bot.send_message(chat_id=chat_id, text=full_text, entities=entities)
    except TelegramBadRequest:
        # Custom emoji not available — fall back to plain text
        return await bot.send_message(chat_id=chat_id, text=f"\u23f3 {text}")


async def _edit_status(msg: types.Message | None, text: str) -> None:
    """Edit the status message, keeping the custom emoji."""
    if msg is None:
        return
    full_text, entities = _build_status_text(text)
    with contextlib.suppress(TelegramBadRequest):
        await msg.edit_text(text=full_text, entities=entities)


async def _edit_to_final(
    msg: types.Message | None,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Edit the status message into the final agent response (no emoji)."""
    if msg is None:
        return
    try:
        await msg.edit_text(text=text, reply_markup=reply_markup)
    except TelegramBadRequest:
        logger.debug("Failed to edit status message to final response")


async def _delete_status(msg: types.Message | None) -> None:
    """Delete the status message."""
    if msg is None:
        return
    with contextlib.suppress(Exception):
        await msg.delete()


# ---------- Helpers ----------


async def _reply(
    event: types.Message | CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Send a new message in the chat (real conversation, no editing)."""
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


def _telegram_id_from_event(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"


async def _ui_language_for_event(event: types.Message | CallbackQuery) -> UILanguage:
    return await _ui_language_or_default(_telegram_id_from_event(event))
