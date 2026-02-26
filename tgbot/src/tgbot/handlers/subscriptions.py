import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.storage import get_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

DELETE_PREFIX = "delete_sub:"
SEND_NOW_PREFIX = "send_now:"


@router.message(Command("list"))
async def cmd_list(message: types.Message) -> None:
    telegram_id = message.from_user.id
    api_key = await get_api_key(telegram_id)

    if api_key is None:
        await message.answer("Please /start first to register.")
        return

    try:
        subs = await backend.list_subscriptions(api_key)
    except Exception:
        logger.exception("Failed to list subscriptions for telegram_id=%d", telegram_id)
        await message.answer("Failed to load subscriptions. Please try again.")
        return

    if not subs:
        await message.answer("You have no active subscriptions. Use /subscribe to create one.")
        return

    for sub in subs:
        topics_str = ", ".join(sub.topics)
        text = (
            f"Topics: {topics_str}\n"
            f"Schedule: {sub.schedule_cron}\n"
            f"Format: {sub.format_instructions}"
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Send now",
                        callback_data=f"{SEND_NOW_PREFIX}{sub.id}",
                    ),
                    InlineKeyboardButton(text="Delete", callback_data=f"{DELETE_PREFIX}{sub.id}"),
                ]
            ]
        )
        await message.answer(text, reply_markup=keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(SEND_NOW_PREFIX))
async def handle_send_now(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    api_key = await get_api_key(telegram_id)
    subscription_id = callback.data[len(SEND_NOW_PREFIX) :]

    if api_key is None:
        await callback.answer("Please /start first.")
        return

    try:
        await backend.send_now(api_key, subscription_id)
        await callback.answer("Digest queued.")
    except Exception:
        logger.exception("Failed to queue digest for subscription %s", subscription_id)
        await callback.answer("Failed to queue digest. Try again.")


@router.callback_query(lambda c: c.data and c.data.startswith(DELETE_PREFIX))
async def handle_delete(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    api_key = await get_api_key(telegram_id)
    subscription_id = callback.data[len(DELETE_PREFIX) :]

    if api_key is None:
        await callback.answer("Please /start first.")
        return

    try:
        await backend.delete_subscription(api_key, subscription_id)
        await callback.answer("Subscription deleted.")
        if callback.message:
            await callback.message.edit_text("Subscription deleted.")
    except Exception:
        logger.exception("Failed to delete subscription %s", subscription_id)
        await callback.answer("Failed to delete. Try again.")
