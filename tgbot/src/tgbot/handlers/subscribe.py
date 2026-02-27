import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from tgbot.client import BackendClient
from tgbot.core.config import get_settings
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()
settings = get_settings()


class SubscribeFlow(StatesGroup):
    waiting_for_prompt = State()


@router.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
        return

    await state.set_state(SubscribeFlow.waiting_for_prompt)
    await message.answer(
        "Describe what news you want and how often.\n\n"
        'Example: "I want AI and tech news every morning as a brief summary"'
    )


@router.message(SubscribeFlow.waiting_for_prompt)
async def process_prompt(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
        await state.clear()
        return
    prompt = message.text

    if not prompt:
        await message.answer("Something went wrong. Please try /subscribe again.")
        await state.clear()
        return

    webhook_url = (
        f"http://{settings.webhook_public_host}:{settings.webhook_port}/deliver/{telegram_id}"
    )

    await message.answer("Processing your request...")

    try:
        sub = await backend.create_subscription(api_key, prompt, webhook_url)
        topics_str = ", ".join(sub.topics)
        await message.answer(
            f"Subscription created!\n\n"
            f"Topics: {topics_str}\n"
            "\n"
            f"You'll receive digests right here in this chat."
        )
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        await message.answer("Failed to create subscription. Please try again.")

    await state.clear()
