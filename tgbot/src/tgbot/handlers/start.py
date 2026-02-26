import logging

from aiogram import Router, types
from aiogram.filters import CommandStart

from tgbot.client import BackendClient
from tgbot.storage import get_api_key, save_api_key

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
    api_key = await get_api_key(telegram_id)

    if api_key is None:
        try:
            api_key = await backend.register_user()
            await save_api_key(telegram_id, api_key)
            logger.info("Registered new user for telegram_id=%d", telegram_id)
        except Exception:
            logger.exception("Failed to register user for telegram_id=%d", telegram_id)
            await message.answer("Registration failed. Please try again later.")
            return

    await message.answer(WELCOME_TEXT)


@router.message(types.Message)
async def cmd_help(message: types.Message) -> None:
    if message.text and message.text.strip() == "/help":
        await message.answer(WELCOME_TEXT)
