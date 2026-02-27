import logging

from aiogram import Router, types
from aiogram.filters import Command, CommandStart

from tgbot.client import BackendClient
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

WELCOME_TEXT = (
    "Welcome to News Bot!\n\n"
    "I deliver personalized news digests right here in Telegram.\n\n"
    "Commands:\n"
    "/subscribe - set up a news subscription\n"
    "/list - view your active subscriptions\n"
    "/help - show this message"
)


@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to register user for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
        return

    await message.answer(WELCOME_TEXT)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    await message.answer(WELCOME_TEXT)
