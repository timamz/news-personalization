"""Bot chat surface: /start, /help, and text-message relay to the agent.

The backend holds one persistent conversation per user, so the tgbot is a
thin transport. /start and /help both render a fixed, bot-authored message
(no LLM call) so the user always gets the same, coherent introduction.
After /start we POST to the backend's acknowledge-onboarding endpoint so
the agent treats the user's first real message as a regular turn instead
of re-greeting them.
"""

import contextlib
import logging

from aiogram import Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart

from tgbot.client import BackendClient
from tgbot.telegram_format import render_html_message
from tgbot.text_split import split_for_telegram
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

_ERROR_TEXT = "Something went wrong. Please try again in a moment."
_TELEGRAM_MESSAGE_LIMIT = 4000

_WELCOME_TEXT = (
    "<b>Hi — I'm your personal news assistant.</b>\n"
    "\n"
    "Tell me what you want to follow and I'll deliver it to you: "
    "scheduled digests on your cadence, or real-time alerts when "
    "something breaking happens.\n"
    "\n"
    "<b>What I can do</b>\n"
    '• Build subscriptions to any topic: "AI research", '
    '"Arsenal FC news in English", "EU tech regulation".\n'
    "• Pull from RSS feeds, Telegram channels, Reddit subs, "
    "and X/Twitter accounts — or find sources for you automatically.\n"
    "• Deliver scheduled digests (every morning, every weekday evening, "
    "weekly on Sundays…) or event alerts the moment a relevant item "
    "appears.\n"
    "• Customize tone, length, language, and what to skip.\n"
    "• Edit anything later — add or remove sources, change schedule, "
    "tweak format.\n"
    "\n"
    "<b>How to talk to me</b>\n"
    "Just write in plain language. Examples:\n"
    '• "AI safety research, three bullets every morning."\n'
    '• "Breaking Ukraine news in Russian, skip opinion pieces."\n'
    '• "Arsenal matches only, five bullets with scorelines."\n'
    '• "Show me my subscriptions."\n'
    '• "Add @bbcworld as a source to my news sub."\n'
    '• "Delete the crypto one."\n'
    "\n"
    "Type /help for a reference. What would you like to follow?"
)

_HELP_TEXT = (
    "<b>Personal News Assistant — help</b>\n"
    "\n"
    "I help you stay informed on topics you choose, without doomscrolling "
    "feeds.\n"
    "\n"
    "<b>Subscriptions</b>\n"
    "A subscription is one topic you care about plus how you want it "
    "delivered. Examples:\n"
    '• "Follow EU tech regulation, weekly recap on Sunday mornings."\n'
    '• "Crypto news, skip meme coins, short bullets every evening."\n'
    "\n"
    "<b>Delivery modes</b>\n"
    "• <b>Digest</b> (default): scheduled summary on your cadence — "
    "morning, evening, weekly, twice a day, any pattern you describe "
    "in words.\n"
    "• <b>Event</b>: real-time alert the moment something relevant lands "
    "— good for breaking news.\n"
    "\n"
    "<b>Sources</b>\n"
    "Tell me what to read, or let me find sources for you:\n"
    "• RSS feeds (any URL).\n"
    "• Telegram channels (channel username).\n"
    "• Reddit subs (just the sub name).\n"
    "• X/Twitter accounts (just the handle).\n"
    "\n"
    "<b>Managing subscriptions</b>\n"
    "Everything is conversational — just say what you want:\n"
    '• List: "show me my subscriptions".\n'
    '• Edit: "make the AI one shorter", "switch to Russian".\n'
    '• Add source: "add techcrunch.com to my tech sub".\n'
    '• Remove source: "drop Reuters from my politics sub".\n'
    '• Delete: "remove the Arsenal one".\n'
    '• On-demand: "send me a digest now".\n'
    "\n"
    "<b>Settings</b>\n"
    '• Language: "switch to Russian", "reply to me in English".\n'
    '• Timezone: "I\'m in Berlin", "my timezone is PST".\n'
    "\n"
    "Write anything — I'll figure it out."
)


@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    """Register the user (if needed) and render the fixed onboarding screen."""
    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to register user for telegram_id=%d", telegram_id)
        await message.answer(_ERROR_TEXT)
        return

    try:
        await backend.acknowledge_onboarding(api_key)
    except Exception:
        logger.exception("Failed to acknowledge onboarding for telegram_id=%d", telegram_id)

    await _send_static_html(message, _WELCOME_TEXT)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    """Render the fixed help screen. Does not touch the conversation thread."""
    await _send_static_html(message, _HELP_TEXT)


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
    """Split on paragraph boundaries if the message exceeds the Telegram limit."""
    rendered = render_html_message(text)
    for chunk in split_for_telegram(rendered, _TELEGRAM_MESSAGE_LIMIT):
        await message.answer(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


async def _send_static_html(message: types.Message, html_text: str) -> None:
    """Send a bot-authored HTML message as-is, without URL rewriting or escaping.

    Used for the /start and /help screens whose Telegram HTML is hand-
    crafted and does not need the agent-output linkifier.
    """
    for chunk in split_for_telegram(html_text, _TELEGRAM_MESSAGE_LIMIT):
        await message.answer(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
