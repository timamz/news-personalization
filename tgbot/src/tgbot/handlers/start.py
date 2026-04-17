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

    new_conversation_id, agent_message = turn

    if new_conversation_id:
        await save_conversation_id(telegram_id, new_conversation_id)

    if agent_message:
        await _send_long_message(message, agent_message)


async def _run_turn(
    api_key: str,
    conversation_id: str | None,
    text: str,
    message: types.Message,
) -> tuple[str | None, str] | None:
    """Run one agent turn, retrying with a fresh conversation on 404/409."""
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
) -> tuple[str | None, str]:
    if conversation_id:
        stream = backend.continue_subscription_conversation_stream(api_key, conversation_id, text)
    else:
        stream = backend.start_subscription_conversation_stream(api_key, text)

    new_id: str | None = conversation_id
    agent_message = ""

    async for event in stream:
        kind = event.get("event")
        if kind == "status":
            await _safe_typing(message)
        elif kind == "done":
            new_id = event.get("conversation_id") or new_id
            agent_message = event.get("agent_message") or ""
        elif kind == "error":
            agent_message = event.get("detail") or _ERROR_TEXT

    return new_id, agent_message


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
