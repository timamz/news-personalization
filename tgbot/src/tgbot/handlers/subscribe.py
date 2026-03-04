import logging
from collections.abc import Awaitable, Callable

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.source_parser import extract_telegram_channels, parse_telegram_channel_tokens
from tgbot.user_registry import ensure_api_key
from tgbot.webhook_server import delivery_webhook_url

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

KNOW_SOURCES_YES = "subscribe:know_sources:yes"
KNOW_SOURCES_NO = "subscribe:know_sources:no"
SCOPE_ONLY_PROVIDED = "subscribe:scope:only_provided"
SCOPE_WITH_DISCOVERY = "subscribe:scope:with_discovery"
SCHEDULE_ENABLE_YES = "subscribe:schedule:yes"
SCHEDULE_ENABLE_NO = "subscribe:schedule:no"
RECENT_EVENTS_YES = "subscribe:recent_events:yes"
RECENT_EVENTS_NO = "subscribe:recent_events:no"
BACK = "subscribe:back"


class SubscribeFlow(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_schedule_decision = State()
    waiting_for_schedule_input = State()
    waiting_for_source_knowledge = State()
    waiting_for_channels_input = State()
    waiting_for_scope_choice = State()
    waiting_for_recent_events_decision = State()


@router.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer("Registration failed. Please try again later.")
        return

    await _show_prompt_step(message, state, reset_data=True)


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

    delivery_mode = parsed.delivery_mode
    schedule_cron_override = None if delivery_mode == "event" else parsed.schedule_cron
    await state.update_data(
        prompt=prompt,
        delivery_mode=delivery_mode,
        schedule_cron_override=schedule_cron_override,
        manual_only=False,
    )

    if delivery_mode == "event":
        await _continue_with_source_flow(
            message,
            state,
            prompt,
            delivery_mode,
            source_back_target="prompt",
        )
        return

    if parsed.schedule_was_explicit and parsed.schedule_cron:
        await _continue_with_source_flow(
            message,
            state,
            prompt,
            delivery_mode,
            source_back_target="prompt",
        )
        return

    await _show_schedule_decision_step(message, state)


@router.callback_query(
    lambda c: c.data in {SCHEDULE_ENABLE_YES, SCHEDULE_ENABLE_NO},
    SubscribeFlow.waiting_for_schedule_decision,
)
async def handle_schedule_decision(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.data == SCHEDULE_ENABLE_YES:
        await _show_schedule_input_step(callback, state)
        return

    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    await state.update_data(schedule_cron_override=None, manual_only=True)
    if not isinstance(prompt, str) or not prompt.strip():
        await _answer(callback, "Subscription setup expired. Please run /subscribe again.")
        await state.clear()
        return
    await _continue_with_source_flow(
        callback,
        state,
        prompt,
        "digest",
        source_back_target="schedule_decision",
    )


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
    await _continue_with_source_flow(
        message,
        state,
        prompt,
        "digest",
        source_back_target="schedule_input",
    )


@router.callback_query(
    lambda c: c.data in {KNOW_SOURCES_YES, KNOW_SOURCES_NO},
    SubscribeFlow.waiting_for_source_knowledge,
)
async def handle_source_knowledge_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.data == KNOW_SOURCES_NO:
        await state.update_data(creation_back_target="source_knowledge")
        await _create_subscription_from_state(
            callback,
            state,
            telegram_channels=[],
            include_discovered_sources=True,
        )
        return

    await _show_channels_input_step(callback, state)


@router.message(SubscribeFlow.waiting_for_channels_input)
async def process_channels_input(message: types.Message, state: FSMContext) -> None:
    channels = parse_telegram_channel_tokens(message.text or "")
    if not channels:
        await message.answer(
            "I couldn't parse channels. Send handles like @channel_one or links like https://t.me/channel."
        )
        return

    await state.update_data(
        telegram_channels=channels,
        scope_back_target="channels_input",
        scope_channels_origin="manual",
    )
    await _show_scope_choice_step(message, state)


@router.callback_query(
    lambda c: c.data in {SCOPE_ONLY_PROVIDED, SCOPE_WITH_DISCOVERY},
    SubscribeFlow.waiting_for_scope_choice,
)
async def handle_scope_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    include_discovered_sources = callback.data == SCOPE_WITH_DISCOVERY
    state_data = await state.get_data()
    channels = list(state_data.get("telegram_channels", []))
    await state.update_data(creation_back_target="scope_choice")
    await _create_subscription_from_state(
        callback,
        state,
        telegram_channels=channels,
        include_discovered_sources=include_discovered_sources,
    )


@router.callback_query(
    lambda c: c.data in {RECENT_EVENTS_YES, RECENT_EVENTS_NO},
    SubscribeFlow.waiting_for_recent_events_decision,
)
async def handle_recent_events_decision(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.data == RECENT_EVENTS_NO:
        await _answer(callback, "Okay. I will only send new event notifications from now on.")
        await state.clear()
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("created_subscription_id")
    if not isinstance(subscription_id, str) or not subscription_id:
        await _answer(callback, "This preview expired. Please create a new subscription if needed.")
        await state.clear()
        return

    telegram_id = _telegram_id_from_event(callback)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(callback, "Registration failed. Please try again later.")
        await state.clear()
        return

    await _answer(callback, "Checking what you might have missed in the last 7 days...")
    try:
        recent_events = await backend.list_recent_events(api_key, subscription_id)
    except Exception:
        logger.exception(
            "Failed to load recent events for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await _answer(callback, "Couldn't load recent events right now.")
        await state.clear()
        return

    if not recent_events:
        await _answer(callback, "No matching events were found in the last 7 days.")
        await state.clear()
        return

    await _answer(callback, "Here's what you might have missed in the last 7 days:")
    for recent_event in recent_events:
        await _answer(callback, f"{recent_event.subject}\n\n{recent_event.body}")

    try:
        await backend.acknowledge_recent_events(
            api_key,
            subscription_id,
            [recent_event.news_item_id for recent_event in recent_events],
        )
    except Exception:
        logger.exception(
            "Failed to acknowledge recent events for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )

    await state.clear()


@router.callback_query(lambda c: c.data == BACK)
async def handle_back(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    current_state = await state.get_state()

    if current_state == SubscribeFlow.waiting_for_schedule_decision.state:
        await _show_prompt_step(callback, state, reset_data=True)
        return
    if current_state == SubscribeFlow.waiting_for_schedule_input.state:
        await _show_schedule_decision_step(callback, state)
        return
    if current_state == SubscribeFlow.waiting_for_source_knowledge.state:
        await _show_source_previous_step(callback, state)
        return
    if current_state == SubscribeFlow.waiting_for_channels_input.state:
        await _show_source_knowledge_step(callback, state)
        return
    if current_state == SubscribeFlow.waiting_for_scope_choice.state:
        await _show_scope_previous_step(callback, state)
        return
    if current_state == SubscribeFlow.waiting_for_recent_events_decision.state:
        await _undo_recent_event_step(callback, state)
        return

    await _answer(callback, "Back is not available here.")


async def _continue_with_source_flow(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    prompt: str,
    delivery_mode: str,
    *,
    source_back_target: str,
) -> None:
    telegram_channels = extract_telegram_channels(prompt)
    await state.update_data(
        telegram_channels=telegram_channels,
        source_back_target=source_back_target,
    )

    if telegram_channels:
        await state.update_data(
            scope_back_target=source_back_target,
            scope_channels_origin="prompt",
        )
        await _show_scope_choice_step(
            event,
            state,
            delivery_mode=delivery_mode,
            channels=telegram_channels,
            origin="prompt",
        )
        return

    await _show_source_knowledge_step(event, state, delivery_mode=delivery_mode)


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
    delivery_mode = str(state_data.get("delivery_mode", "digest"))

    telegram_id = _telegram_id_from_event(event)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(event, "Registration failed. Please try again later.")
        await state.clear()
        return

    webhook_url = delivery_webhook_url(telegram_id)
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
            delivery_mode=delivery_mode,
        )
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        await _answer(event, "Failed to create subscription. Please try again.")
        await state.clear()
        return

    topics_str = ", ".join(subscription.topics)
    if delivery_mode == "event":
        completion_text = "You'll receive event notifications right here in this chat."
    else:
        completion_text = "You'll receive digests right here in this chat."
    await _answer(
        event,
        f"Subscription created!\n\nTopics: {topics_str}\n\n{completion_text}",
    )
    if delivery_mode == "event":
        await state.update_data(created_subscription_id=subscription.id)
        await state.set_state(SubscribeFlow.waiting_for_recent_events_decision)
        await _answer_with_markup(
            event,
            "Would you like to see what you might have missed in the last 7 days?",
            _recent_events_choice_keyboard(),
        )
        return

    await state.clear()


async def _show_prompt_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    reset_data: bool,
) -> None:
    if reset_data:
        await state.clear()
    await state.set_state(SubscribeFlow.waiting_for_prompt)
    await _answer(event, _subscription_prompt_text())


async def _show_schedule_decision_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    await state.set_state(SubscribeFlow.waiting_for_schedule_decision)
    await _answer_with_markup(
        event,
        "Do you want this digest to be delivered automatically on a schedule?\n"
        "You can always use the Send now button.",
        _schedule_choice_keyboard(),
    )


async def _show_schedule_input_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    await state.set_state(SubscribeFlow.waiting_for_schedule_input)
    await _answer_with_markup(
        event,
        'Describe the schedule in natural language.\nExample: "every weekday at 9:00"',
        _back_only_keyboard(),
    )


async def _show_source_knowledge_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    delivery_mode: str | None = None,
) -> None:
    if delivery_mode is None:
        state_data = await state.get_data()
        delivery_mode = str(state_data.get("delivery_mode", "digest"))
    await state.set_state(SubscribeFlow.waiting_for_source_knowledge)
    await _answer_with_markup(
        event,
        _source_knowledge_question(delivery_mode),
        _source_knowledge_keyboard(),
    )


async def _show_channels_input_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    await state.set_state(SubscribeFlow.waiting_for_channels_input)
    await _answer_with_markup(
        event,
        "Send Telegram channels you want to include (for example: @channel_one @channel_two).",
        _back_only_keyboard(),
    )


async def _show_scope_choice_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    delivery_mode: str | None = None,
    channels: list[str] | None = None,
    origin: str | None = None,
) -> None:
    if delivery_mode is None or channels is None or origin is None:
        state_data = await state.get_data()
        if delivery_mode is None:
            delivery_mode = str(state_data.get("delivery_mode", "digest"))
        if channels is None:
            channels = list(state_data.get("telegram_channels", []))
        if origin is None:
            origin = str(state_data.get("scope_channels_origin", "manual"))
    await state.set_state(SubscribeFlow.waiting_for_scope_choice)

    if origin == "prompt":
        text = (
            "I found these Telegram channels in your request:\n"
            f"{_format_channels(channels)}\n\n"
            f"{_scope_question(delivery_mode)}"
        )
    else:
        text = f"Got it. {_scope_question(delivery_mode)}\n{_format_channels(channels)}"

    await _answer_with_markup(event, text, _scope_keyboard())


async def _show_source_previous_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    state_data = await state.get_data()
    target = str(state_data.get("source_back_target", "prompt"))

    if target == "schedule_decision":
        await _show_schedule_decision_step(event, state)
        return
    if target == "schedule_input":
        await _show_schedule_input_step(event, state)
        return

    await _show_prompt_step(event, state, reset_data=True)


async def _show_scope_previous_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    state_data = await state.get_data()
    target = str(state_data.get("scope_back_target", "prompt"))

    if target == "channels_input":
        await _show_channels_input_step(event, state)
        return
    if target == "schedule_decision":
        await _show_schedule_decision_step(event, state)
        return
    if target == "schedule_input":
        await _show_schedule_input_step(event, state)
        return

    await _show_prompt_step(event, state, reset_data=True)


async def _undo_recent_event_step(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    state_data = await state.get_data()
    subscription_id = state_data.get("created_subscription_id")
    if not isinstance(subscription_id, str) or not subscription_id:
        await _answer(callback, "This step can no longer be undone.")
        await state.clear()
        return

    telegram_id = _telegram_id_from_event(callback)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.delete_subscription(api_key, subscription_id)
    except Exception:
        logger.exception(
            "Failed to undo created subscription for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await _answer(callback, "Couldn't go back right now. Please try again.")
        return

    await state.update_data(created_subscription_id=None)
    target = str(state_data.get("creation_back_target", "source_knowledge"))
    if target == "scope_choice":
        await _show_scope_choice_step(callback, state)
        return

    await _show_source_knowledge_step(callback, state)


def _source_knowledge_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Yes, I have channels", callback_data=KNOW_SOURCES_YES),
                InlineKeyboardButton(text="No, find sources for me", callback_data=KNOW_SOURCES_NO),
            ],
            [InlineKeyboardButton(text="Back", callback_data=BACK)],
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
            ],
            [InlineKeyboardButton(text="Back", callback_data=BACK)],
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
            ],
            [InlineKeyboardButton(text="Back", callback_data=BACK)],
        ]
    )


def _recent_events_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Yes, show missed events",
                    callback_data=RECENT_EVENTS_YES,
                ),
                InlineKeyboardButton(
                    text="No, only future ones",
                    callback_data=RECENT_EVENTS_NO,
                ),
            ],
            [InlineKeyboardButton(text="Back", callback_data=BACK)],
        ]
    )


def _back_only_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Back", callback_data=BACK)],
        ]
    )


def _subscription_prompt_text() -> str:
    return (
        "Describe what news you want.\n\n"
        "Examples:\n"
        '- "I want AI and tech news every morning as a brief summary"\n'
        '- "Notify me when a new episode of Severance is announced"'
    )


def _scope_question(delivery_mode: str) -> str:
    if delivery_mode == "event":
        return "Should these notifications be limited only to these channels?"
    return "Should this digest be limited only to these channels?"


def _source_knowledge_question(delivery_mode: str) -> str:
    if delivery_mode == "event":
        return "Do you already have specific Telegram channels for these notifications?"
    return "Do you already have specific Telegram channels for this digest?"


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
