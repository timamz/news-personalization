import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.user_registry import ensure_api_key
from tgbot.webhook_server import delivery_webhook_url

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

DELETE_PREFIX = "delete_sub:"
SEND_NOW_PREFIX = "send_now:"
EDIT_PREFIX = "edit_sub:"
EDIT_SCHEDULE_PREFIX = "edit_sched:"
DISABLE_SCHEDULE_PREFIX = "disable_sched:"
EDIT_FORMAT_PREFIX = "edit_fmt:"
DELIVER_HERE_PREFIX = "deliver_here:"


class EditFlow(StatesGroup):
    waiting_for_schedule = State()
    waiting_for_format = State()


@router.message(Command("list"))
async def cmd_list(message: types.Message) -> None:
    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
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
        mode_label = "Event notifications" if sub.delivery_mode == "event" else "Digest"
        text = f"Topics: {topics_str}\nType: {mode_label}"
        buttons = []
        if sub.delivery_mode == "digest":
            buttons.append(
                InlineKeyboardButton(
                    text="Send now",
                    callback_data=f"{SEND_NOW_PREFIX}{sub.id}",
                )
            )
        buttons.append(
            InlineKeyboardButton(
                text="Edit",
                callback_data=f"{EDIT_PREFIX}{sub.delivery_mode}:{sub.id}",
            )
        )
        buttons.append(
            InlineKeyboardButton(
                text="Delete",
                callback_data=f"{DELETE_PREFIX}{sub.id}",
            )
        )
        keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
        await message.answer(text, reply_markup=keyboard)


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_PREFIX))
async def handle_edit_menu(callback: CallbackQuery) -> None:
    await callback.answer()
    delivery_mode, subscription_id = _parse_mode_and_subscription_id(callback.data, EDIT_PREFIX)
    if callback.message is None:
        return
    await callback.message.answer(
        "What do you want to change?",
        reply_markup=_edit_keyboard(subscription_id, delivery_mode),
    )


@router.callback_query(lambda c: c.data and c.data.startswith(SEND_NOW_PREFIX))
async def handle_send_now(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(SEND_NOW_PREFIX) :]

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer("Registration failed. Try again.")
        return

    try:
        await backend.send_now(api_key, subscription_id)
        await callback.answer("Digest queued.")
    except Exception:
        logger.exception("Failed to queue digest for subscription %s", subscription_id)
        await callback.answer("Failed to queue digest. Try again.")


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_SCHEDULE_PREFIX))
async def handle_edit_schedule(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_SCHEDULE_PREFIX)
    await state.set_state(EditFlow.waiting_for_schedule)
    await state.update_data(subscription_id=subscription_id)
    if callback.message:
        await callback.message.answer(
            "Describe the new schedule in natural language.\n"
            'Example: "every weekday at 9:00"'
        )


@router.message(EditFlow.waiting_for_schedule)
async def process_schedule_edit(message: types.Message, state: FSMContext) -> None:
    schedule_text = (message.text or "").strip()
    if not schedule_text:
        await message.answer("Please describe the schedule.")
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await message.answer("Edit session expired. Open /list and try again.")
        await state.clear()
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        schedule_cron = await backend.parse_schedule(api_key, schedule_text)
        await backend.update_subscription(
            api_key,
            subscription_id,
            schedule_cron=schedule_cron,
        )
        await message.answer("Schedule updated.")
    except Exception:
        logger.exception("Failed to update schedule for subscription %s", subscription_id)
        await message.answer("Failed to update schedule. Please try again.")
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(DISABLE_SCHEDULE_PREFIX))
async def handle_disable_schedule(callback: CallbackQuery) -> None:
    subscription_id = _subscription_id_from_callback(callback.data, DISABLE_SCHEDULE_PREFIX)
    telegram_id = callback.from_user.id

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(api_key, subscription_id, schedule_cron=None)
        await callback.answer("Automatic schedule disabled.")
    except Exception:
        logger.exception("Failed to disable schedule for subscription %s", subscription_id)
        await callback.answer("Failed to update schedule. Try again.")


@router.callback_query(lambda c: c.data and c.data.startswith(EDIT_FORMAT_PREFIX))
async def handle_edit_format(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    subscription_id = _subscription_id_from_callback(callback.data, EDIT_FORMAT_PREFIX)
    await state.set_state(EditFlow.waiting_for_format)
    await state.update_data(subscription_id=subscription_id)
    if callback.message:
        await callback.message.answer("Send new format instructions.")


@router.message(EditFlow.waiting_for_format)
async def process_format_edit(message: types.Message, state: FSMContext) -> None:
    format_instructions = (message.text or "").strip()
    if not format_instructions:
        await message.answer("Please send the new format instructions.")
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("subscription_id")
    if not isinstance(subscription_id, str):
        await message.answer("Edit session expired. Open /list and try again.")
        await state.clear()
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(
            api_key,
            subscription_id,
            format_instructions=format_instructions,
        )
        await message.answer("Format updated.")
    except Exception:
        logger.exception("Failed to update format for subscription %s", subscription_id)
        await message.answer("Failed to update format. Please try again.")
    finally:
        await state.clear()


@router.callback_query(lambda c: c.data and c.data.startswith(DELIVER_HERE_PREFIX))
async def handle_deliver_here(callback: CallbackQuery) -> None:
    subscription_id = _subscription_id_from_callback(callback.data, DELIVER_HERE_PREFIX)
    telegram_id = callback.from_user.id

    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_subscription(
            api_key,
            subscription_id,
            delivery_webhook_url=_webhook_url_for_chat(telegram_id),
        )
        await callback.answer("Delivery updated to this chat.")
    except Exception:
        logger.exception("Failed to update delivery target for subscription %s", subscription_id)
        await callback.answer("Failed to update delivery. Try again.")


@router.callback_query(lambda c: c.data and c.data.startswith(DELETE_PREFIX))
async def handle_delete(callback: CallbackQuery) -> None:
    telegram_id = callback.from_user.id
    subscription_id = callback.data[len(DELETE_PREFIX) :]

    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await callback.answer("Registration failed. Try again.")
        return

    try:
        await backend.delete_subscription(api_key, subscription_id)
        await callback.answer("Subscription deleted.")
        if callback.message:
            await callback.message.edit_text("Subscription deleted.")
    except Exception:
        logger.exception("Failed to delete subscription %s", subscription_id)
        await callback.answer("Failed to delete. Try again.")


def _edit_keyboard(subscription_id: str, delivery_mode: str) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    if delivery_mode == "digest":
        buttons.append(
            [
                InlineKeyboardButton(
                    text="Change schedule",
                    callback_data=f"{EDIT_SCHEDULE_PREFIX}{subscription_id}",
                ),
                InlineKeyboardButton(
                    text="Disable schedule",
                    callback_data=f"{DISABLE_SCHEDULE_PREFIX}{subscription_id}",
                ),
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                text="Change format",
                callback_data=f"{EDIT_FORMAT_PREFIX}{subscription_id}",
            ),
            InlineKeyboardButton(
                text="Deliver here",
                callback_data=f"{DELIVER_HERE_PREFIX}{subscription_id}",
            ),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                text="Delete",
                callback_data=f"{DELETE_PREFIX}{subscription_id}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _parse_mode_and_subscription_id(callback_data: str | None, prefix: str) -> tuple[str, str]:
    if callback_data is None:
        return "digest", ""
    payload = callback_data[len(prefix) :]
    delivery_mode, subscription_id = payload.split(":", maxsplit=1)
    return delivery_mode, subscription_id


def _subscription_id_from_callback(callback_data: str | None, prefix: str) -> str:
    if callback_data is None:
        return ""
    return callback_data[len(prefix) :]


def _webhook_url_for_chat(chat_id: int) -> str:
    return delivery_webhook_url(chat_id)
