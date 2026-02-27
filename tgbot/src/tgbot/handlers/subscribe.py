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


class SubscribeFlow(StatesGroup):
    waiting_for_prompt = State()
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
        "Describe what news you want and how often.\n\n"
        'Example: "I want AI and tech news every morning as a brief summary"'
    )


@router.message(SubscribeFlow.waiting_for_prompt)
async def process_prompt(message: types.Message, state: FSMContext) -> None:
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Please describe your subscription request.")
        return

    telegram_channels = extract_telegram_channels(prompt)
    await state.update_data(prompt=prompt, telegram_channels=telegram_channels)

    if telegram_channels:
        await state.set_state(SubscribeFlow.waiting_for_scope_choice)
        await message.answer(
            "I found these Telegram channels in your request:\n"
            f"{_format_channels(telegram_channels)}\n\n"
            "Should digest be limited only to these channels?",
            reply_markup=_scope_keyboard(),
        )
        return

    await state.set_state(SubscribeFlow.waiting_for_source_knowledge)
    await message.answer(
        "Do you already have specific Telegram channels for this digest?",
        reply_markup=_source_knowledge_keyboard(),
    )


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
