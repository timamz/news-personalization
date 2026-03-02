import logging
from collections.abc import Awaitable, Callable

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.core.config import get_settings
from tgbot.source_parser import extract_telegram_channels, parse_telegram_channel_tokens
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()
settings = get_settings()

KNOW_SOURCES_YES = "subscribe:know_sources:yes"
KNOW_SOURCES_NO = "subscribe:know_sources:no"
SCOPE_ONLY_PROVIDED = "subscribe:scope:only_provided"
SCOPE_WITH_DISCOVERY = "subscribe:scope:with_discovery"
SCHEDULE_ENABLE_YES = "subscribe:schedule:yes"
SCHEDULE_ENABLE_NO = "subscribe:schedule:no"


class SubscribeFlow(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_schedule_decision = State()
    waiting_for_schedule_input = State()
    waiting_for_source_knowledge = State()
    waiting_for_channels_input = State()
    waiting_for_scope_choice = State()


@router.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
        return

    await state.clear()
    await state.set_state(SubscribeFlow.waiting_for_prompt)
    await message.answer(
        "Describe what news you want.\n\n"
        'Example: "I want AI and tech news every morning as a brief summary"'
    )


@router.message(SubscribeFlow.waiting_for_prompt)
async def process_prompt(message: types.Message, state: FSMContext) -> None:
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Please describe your subscription request.")
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        parsed = await backend.parse_subscription_prompt(api_key, prompt)
    except Exception:
        logger.exception("Failed to parse subscription prompt for telegram_id=%d", telegram_id)
        await message.answer("Failed to process your request. Please try again.")
        await state.clear()
        return

    await state.update_data(
        prompt=prompt,
        schedule_cron_override=parsed.schedule_cron,
        manual_only=False,
    )

    if parsed.schedule_was_explicit and parsed.schedule_cron:
        await _continue_with_source_flow(message, state, prompt)
        return

    await state.set_state(SubscribeFlow.waiting_for_schedule_decision)
    await message.answer(
        "Do you want this digest to be delivered automatically on a schedule?\n"
        "You can always use the Send now button.",
        reply_markup=_schedule_choice_keyboard(),
    )


@router.callback_query(
    lambda c: c.data in {SCHEDULE_ENABLE_YES, SCHEDULE_ENABLE_NO},
    SubscribeFlow.waiting_for_schedule_decision,
)
async def handle_schedule_decision(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.data == SCHEDULE_ENABLE_YES:
        await state.set_state(SubscribeFlow.waiting_for_schedule_input)
        if callback.message:
            await callback.message.answer(
                "Describe the schedule in natural language.\n"
                'Example: "every weekday at 9:00"'
            )
        return

    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    await state.update_data(schedule_cron_override=None, manual_only=True)
    if not isinstance(prompt, str) or not prompt.strip():
        await _answer(callback, "Subscription setup expired. Please run /subscribe again.")
        await state.clear()
        return
    await _continue_with_source_flow(callback, state, prompt)


@router.message(SubscribeFlow.waiting_for_schedule_input)
async def process_schedule_input(message: types.Message, state: FSMContext) -> None:
    schedule_text = (message.text or "").strip()
    if not schedule_text:
        await message.answer("Please describe the schedule.")
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        schedule_cron = await backend.parse_schedule(api_key, schedule_text)
    except Exception:
        logger.exception("Failed to parse schedule for telegram_id=%d", telegram_id)
        await message.answer("Couldn't parse this schedule. Please try another wording.")
        return

    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    await state.update_data(schedule_cron_override=schedule_cron, manual_only=False)
    if not isinstance(prompt, str) or not prompt.strip():
        await message.answer("Subscription setup expired. Please run /subscribe again.")
        await state.clear()
        return
    await _continue_with_source_flow(message, state, prompt)


@router.callback_query(
    lambda c: c.data in {KNOW_SOURCES_YES, KNOW_SOURCES_NO},
    SubscribeFlow.waiting_for_source_knowledge,
)
async def handle_source_knowledge_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.data == KNOW_SOURCES_NO:
        await _create_subscription_from_state(
            callback,
            state,
            telegram_channels=[],
            include_discovered_sources=True,
        )
        return

    await state.set_state(SubscribeFlow.waiting_for_channels_input)
    if callback.message:
        await callback.message.answer(
            "Send Telegram channels you want to include (for example: @channel_one @channel_two)."
        )


@router.message(SubscribeFlow.waiting_for_channels_input)
async def process_channels_input(message: types.Message, state: FSMContext) -> None:
    channels = parse_telegram_channel_tokens(message.text or "")
    if not channels:
        await message.answer(
            "I couldn't parse channels. Send handles like @channel_one or links like https://t.me/channel."
        )
        return

    await state.update_data(telegram_channels=channels)
    await state.set_state(SubscribeFlow.waiting_for_scope_choice)
    await message.answer(
        "Got it. Should digest be limited only to these channels?\n"
        f"{_format_channels(channels)}",
        reply_markup=_scope_keyboard(),
    )


@router.callback_query(
    lambda c: c.data in {SCOPE_ONLY_PROVIDED, SCOPE_WITH_DISCOVERY},
    SubscribeFlow.waiting_for_scope_choice,
)
async def handle_scope_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    include_discovered_sources = callback.data == SCOPE_WITH_DISCOVERY
    state_data = await state.get_data()
    channels = list(state_data.get("telegram_channels", []))
    await _create_subscription_from_state(
        callback,
        state,
        telegram_channels=channels,
        include_discovered_sources=include_discovered_sources,
    )


async def _continue_with_source_flow(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    prompt: str,
) -> None:
    telegram_channels = extract_telegram_channels(prompt)
    await state.update_data(telegram_channels=telegram_channels)

    if telegram_channels:
        await state.set_state(SubscribeFlow.waiting_for_scope_choice)
        await _answer_with_markup(
            event,
            "I found these Telegram channels in your request:\n"
            f"{_format_channels(telegram_channels)}\n\n"
            "Should digest be limited only to these channels?",
            _scope_keyboard(),
        )
        return

    await state.set_state(SubscribeFlow.waiting_for_source_knowledge)
    await _answer_with_markup(
        event,
        "Do you already have specific Telegram channels for this digest?",
        _source_knowledge_keyboard(),
    )


async def _create_subscription_from_state(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    telegram_channels: list[str],
    include_discovered_sources: bool,
) -> None:
    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        await _answer(event, "Subscription setup expired. Please run /subscribe again.")
        await state.clear()
        return

    schedule_cron_override = state_data.get("schedule_cron_override")
    manual_only_value = state_data.get("manual_only")
    manual_only = bool(manual_only_value) if manual_only_value is not None else None

    telegram_id = _telegram_id_from_event(event)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(event, "Registration failed. Please try again later.")
        await state.clear()
        return

    webhook_url = f"http://{settings.webhook_public_host}:{settings.webhook_port}/deliver/{telegram_id}"
    await _answer(event, "Processing your request...")

    try:
        subscription = await backend.create_subscription(
            api_key,
            prompt,
            webhook_url,
            fixed_telegram_channels=telegram_channels,
            include_discovered_sources=include_discovered_sources,
            schedule_cron_override=schedule_cron_override,
            manual_only=manual_only,
        )
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        await _answer(event, "Failed to create subscription. Please try again.")
        await state.clear()
        return

    topics_str = ", ".join(subscription.topics)
    await _answer(
        event,
        "Subscription created!\n\n"
        f"Topics: {topics_str}\n\n"
        "You'll receive digests right here in this chat.",
    )
    await state.clear()


def _source_knowledge_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Yes, I have channels", callback_data=KNOW_SOURCES_YES),
                InlineKeyboardButton(text="No, find sources for me", callback_data=KNOW_SOURCES_NO),
            ]
        ]
    )


def _scope_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Only these channels",
                    callback_data=SCOPE_ONLY_PROVIDED,
                ),
                InlineKeyboardButton(
                    text="Add more relevant sources",
                    callback_data=SCOPE_WITH_DISCOVERY,
                ),
            ]
        ]
    )


def _schedule_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Yes, set schedule", callback_data=SCHEDULE_ENABLE_YES),
                InlineKeyboardButton(
                    text="No, send only by button",
                    callback_data=SCHEDULE_ENABLE_NO,
                ),
            ]
        ]
    )


def _format_channels(channels: list[str]) -> str:
    return "\n".join(f"- @{channel}" for channel in channels)


def _telegram_id_from_event(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _answer(event: types.Message | CallbackQuery, text: str) -> None:
    sender: Callable[[str], Awaitable[types.Message | bool]]
    if hasattr(event, "message"):
        if event.message is None:
            return
        sender = event.message.answer
    else:
        sender = event.answer
    await sender(text)


async def _answer_with_markup(
    event: types.Message | CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    if hasattr(event, "message"):
        if event.message is None:
            return
        await event.message.answer(text, reply_markup=reply_markup)
        return
    await event.answer(text, reply_markup=reply_markup)
