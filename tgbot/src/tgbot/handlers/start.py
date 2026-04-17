"""Bot chat surface: /start + text-message relay to the conversational agent.

The backend holds one persistent conversation per user, so the tgbot is a
thin transport: make sure the user has an API key, stream each turn to
the backend, forward the reply. /start is non-destructive -- it just
ensures registration and greets; the conversation thread persists across
/start invocations. The backend has a separate reset endpoint reserved
for an explicit escape hatch if one is ever exposed.
"""

import contextlib
import logging

from aiogram import Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

from tgbot.client import BackendClient
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
    """Ensure the user has an API key and greet. Never touches the thread."""
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to register user for telegram_id=%d", telegram_id)
        await message.answer(_ERROR_TEXT)
        return

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

    try:
        agent_message = await _stream_turn(api_key, text, message)
    except Exception:
        logger.exception("Conversation turn failed")
        await message.answer(_ERROR_TEXT)
        return

    if agent_message:
        await _send_long_message(message, agent_message)


async def _stream_turn(
    api_key: str,
    text: str,
    message: types.Message,
) -> str:
    agent_message = ""
    async for event in backend.send_conversation_message_stream(api_key, text):
        kind = event.get("event")
        if kind == "status":
            await _safe_typing(message)
        elif kind == "done":
            agent_message = event.get("agent_message") or ""
        elif kind == "error":
            agent_message = event.get("detail") or _ERROR_TEXT
    return agent_message


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
