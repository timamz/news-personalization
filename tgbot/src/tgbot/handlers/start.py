"""Bot chat surface: /start + text-message relay to the conversational agent.

All user-facing behavior lives in a single chat with the backend agent. The
tgbot here is a thin transport: it ensures the user has an API key, keeps
track of the current backend conversation_id, streams agent turns, and
forwards the agent's reply to Telegram. No menus, no inline keyboards, no
FSM -- every text the user sends goes to the agent.
"""

import contextlib
import logging

import httpx
from aiogram import Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

from tgbot.client import BackendClient
from tgbot.storage import (
    clear_conversation_id,
    get_conversation_id,
    save_conversation_id,
)
from tgbot.telegram_format import render_html_message
from tgbot.user_registry import ensure_api_key
from tgbot.webhook_server import delivery_webhook_url

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

_WELCOME_TEXT = "What can I help you with?"
_ERROR_TEXT = "Something went wrong. Please try again in a moment."
_TELEGRAM_MESSAGE_LIMIT = 4000


@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    """Register the user if needed, discard any open conversation, greet."""
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to register user for telegram_id=%d", telegram_id)
        await message.answer(_ERROR_TEXT)
        return

    await clear_conversation_id(telegram_id)
    await message.answer(_WELCOME_TEXT)


@router.message()
async def handle_user_message(message: types.Message) -> None:
    """Relay every other text message to the conversational agent."""
    text = (message.text or "").strip()
    if not text:
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(_ERROR_TEXT)
        return

    await _safe_typing(message)

    conversation_id = await get_conversation_id(telegram_id)
    turn = await _run_turn(api_key, conversation_id, text, message)
    if turn is None:
        return

    new_conversation_id, agent_message, status, finalized_config = turn

    if new_conversation_id:
        await save_conversation_id(telegram_id, new_conversation_id)

    if agent_message:
        await _send_long_message(message, agent_message)

    if status == "ready" and finalized_config:
        await _finalize_subscription(message, api_key, finalized_config, text)
        await clear_conversation_id(telegram_id)


async def _run_turn(
    api_key: str,
    conversation_id: str | None,
    text: str,
    message: types.Message,
) -> tuple[str | None, str, str, dict | None] | None:
    """Run one agent turn, falling back to a fresh conversation on 404/409.

    Returns (conversation_id, agent_message, status, finalized_config) on
    success, or None on unrecoverable error (error already sent to the user).
    """
    for attempt in (conversation_id, None) if conversation_id else (None,):
        try:
            return await _stream_turn(api_key, attempt, text, message)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if attempt and status_code in {404, 409}:
                logger.info("Conversation %s stale (%d); starting fresh", attempt, status_code)
                continue
            logger.exception("Backend conversation call failed (status=%d)", status_code)
            await message.answer(_ERROR_TEXT)
            return None
        except Exception:
            logger.exception("Conversation turn failed")
            await message.answer(_ERROR_TEXT)
            return None
    return None


async def _stream_turn(
    api_key: str,
    conversation_id: str | None,
    text: str,
    message: types.Message,
) -> tuple[str | None, str, str, dict | None]:
    if conversation_id:
        stream = backend.continue_subscription_conversation_stream(api_key, conversation_id, text)
    else:
        stream = backend.start_subscription_conversation_stream(api_key, text)

    new_id: str | None = conversation_id
    agent_message = ""
    status = "in_progress"
    finalized_config: dict | None = None

    async for event in stream:
        kind = event.get("event")
        if kind == "status":
            await _safe_typing(message)
        elif kind == "done":
            new_id = event.get("conversation_id") or new_id
            agent_message = event.get("agent_message") or ""
            status = event.get("status", "in_progress")
            finalized_config = event.get("finalized_config")
        elif kind == "error":
            agent_message = event.get("detail") or _ERROR_TEXT
            status = "error"

    return new_id, agent_message, status, finalized_config


async def _finalize_subscription(
    message: types.Message,
    api_key: str,
    config: dict,
    prompt: str,
) -> None:
    """Turn a finalized_config into a created subscription via the backend."""
    webhook_url = delivery_webhook_url(message.from_user.id)
    create_kwargs: dict[str, object] = {
        "include_discovered_sources": config.get("include_discovered_sources", True),
        "schedule_cron_override": config.get("schedule_cron"),
        "manual_only": config.get("manual_only", False),
        "delivery_mode": config.get("delivery_mode", "digest"),
    }
    if config.get("digest_language"):
        create_kwargs["digest_language"] = config["digest_language"]
    if config.get("format_instructions"):
        create_kwargs["format_instructions"] = config["format_instructions"]
    for key in ("fixed_telegram_channels", "fixed_reddit_subreddits", "fixed_twitter_accounts"):
        if config.get(key):
            create_kwargs[key] = config[key]

    try:
        async for event in backend.create_subscription_stream(
            api_key, prompt[:500], webhook_url, **create_kwargs
        ):
            if event.get("event") == "status":
                await _safe_typing(message)
            elif event.get("event") == "error":
                detail = event.get("detail") or _ERROR_TEXT
                await message.answer(detail)
                return
    except Exception:
        logger.exception("Failed to create subscription from finalized config")
        await message.answer(_ERROR_TEXT)


async def _safe_typing(message: types.Message) -> None:
    with contextlib.suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, "typing")


async def _send_long_message(message: types.Message, text: str) -> None:
    """Split on paragraph boundaries if the agent's reply exceeds 4096 chars."""
    rendered = render_html_message(text)
    for chunk in _split(rendered, _TELEGRAM_MESSAGE_LIMIT):
        await message.answer(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks
