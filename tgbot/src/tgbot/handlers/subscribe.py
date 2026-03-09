import logging
from collections.abc import Awaitable, Callable

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.handlers import start as start_handler
from tgbot.language import UILanguage
from tgbot.source_parser import extract_telegram_channels, parse_telegram_channel_tokens
from tgbot.storage import get_language_preference, get_ui_language
from tgbot.ui_text import t
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
SUBSCRIPTION_LANGUAGE_RU = "subscribe:language:ru"
SUBSCRIPTION_LANGUAGE_EN = "subscribe:language:en"
BACK = "subscribe:back"


class SubscribeFlow(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_language_choice = State()
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
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    if not await start_handler.ensure_user_setup(
        message,
        state,
        api_key=api_key,
        next_action="subscribe",
        reset_state=True,
    ):
        return

    await _show_prompt_step(message, state, reset_data=True)


@router.message(SubscribeFlow.waiting_for_prompt)
async def process_prompt(message: types.Message, state: FSMContext) -> None:
    prompt = (message.text or "").strip()
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not prompt:
        await message.answer(t(ui_language, "choose_subscription_request"))
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        parsed = await backend.parse_subscription_prompt(api_key, prompt)
    except Exception:
        logger.exception("Failed to parse subscription prompt for telegram_id=%d", telegram_id)
        await message.answer(t(ui_language, "failed_process_request"))
        await state.clear()
        return

    delivery_mode = parsed.delivery_mode
    schedule_cron_override = None if delivery_mode == "event" else parsed.schedule_cron
    await state.update_data(
        prompt=prompt,
        delivery_mode=delivery_mode,
        schedule_cron_override=schedule_cron_override,
        manual_only=False,
        schedule_was_explicit=parsed.schedule_was_explicit,
    )

    preference = await get_language_preference(telegram_id)
    if preference is None:
        await start_handler.prompt_subscription_language_selection(
            message,
            state,
            ui_language=ui_language,
            next_action="subscribe",
            initial=True,
            reset_state=False,
        )
        return
    if preference.mode == "ask":
        await _show_language_choice_step(message, state)
        return

    await state.update_data(digest_language_override=preference.code)
    await _continue_after_prompt(message, state, back_target="prompt")
    return


@router.callback_query(
    lambda c: c.data in {SUBSCRIPTION_LANGUAGE_RU, SUBSCRIPTION_LANGUAGE_EN},
    SubscribeFlow.waiting_for_language_choice,
)
async def handle_subscription_language_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    digest_language = "ru" if callback.data == SUBSCRIPTION_LANGUAGE_RU else "en"
    await state.update_data(digest_language_override=digest_language)
    await _continue_after_prompt(callback, state, back_target="language_choice")


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
        await _answer(
            callback,
            t(await _ui_language_or_default(callback.from_user.id), "subscription_setup_expired"),
        )
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
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not schedule_text:
        await message.answer(t(ui_language, "describe_schedule"))
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        schedule_cron = await backend.parse_schedule(api_key, schedule_text)
    except Exception:
        logger.exception("Failed to parse schedule for telegram_id=%d", telegram_id)
        await message.answer(t(ui_language, "schedule_parse_failed"))
        return

    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    await state.update_data(schedule_cron_override=schedule_cron, manual_only=False)
    if not isinstance(prompt, str) or not prompt.strip():
        await message.answer(t(ui_language, "subscription_setup_expired"))
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
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not channels:
        await message.answer(t(ui_language, "channels_parse_failed"))
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
    ui_language = await _ui_language_or_default(callback.from_user.id)
    if callback.data == RECENT_EVENTS_NO:
        await _answer(callback, t(ui_language, "recent_events_future_only"))
        await state.clear()
        return

    state_data = await state.get_data()
    subscription_id = state_data.get("created_subscription_id")
    if not isinstance(subscription_id, str) or not subscription_id:
        await _answer(callback, t(ui_language, "recent_events_expired"))
        await state.clear()
        return

    telegram_id = _telegram_id_from_event(callback)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(callback, t(ui_language, "registration_failed"))
        await state.clear()
        return

    await _answer(callback, t(ui_language, "recent_events_loading"))
    try:
        recent_events = await backend.list_recent_events(api_key, subscription_id)
    except Exception:
        logger.exception(
            "Failed to load recent events for telegram_id=%d subscription=%s",
            telegram_id,
            subscription_id,
        )
        await _answer(callback, t(ui_language, "recent_events_failed"))
        await state.clear()
        return

    if recent_events is None:
        await _answer(callback, t(ui_language, "recent_events_empty"))
        await state.clear()
        return

    await _answer(callback, f"{recent_events.subject}\n\n{recent_events.body}")

    try:
        await backend.acknowledge_recent_events(
            api_key,
            subscription_id,
            recent_events.news_item_ids,
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

    if current_state == SubscribeFlow.waiting_for_language_choice.state:
        await _show_prompt_step(callback, state, reset_data=True)
        return
    if current_state == SubscribeFlow.waiting_for_schedule_decision.state:
        state_data = await state.get_data()
        if state_data.get("schedule_back_target") == "language_choice":
            await _show_language_choice_step(callback, state)
        else:
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

    await _answer(
        callback,
        t(await _ui_language_or_default(callback.from_user.id), "back_not_available"),
    )


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
    ui_language = await _ui_language_or_default(_telegram_id_from_event(event))
    if not isinstance(prompt, str) or not prompt.strip():
        await _answer(event, t(ui_language, "subscription_setup_expired"))
        await state.clear()
        return

    schedule_cron_override = state_data.get("schedule_cron_override")
    manual_only_value = state_data.get("manual_only")
    manual_only = bool(manual_only_value) if manual_only_value is not None else None
    delivery_mode = str(state_data.get("delivery_mode", "digest"))
    digest_language = state_data.get("digest_language_override")

    telegram_id = _telegram_id_from_event(event)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(event, t(ui_language, "registration_failed"))
        await state.clear()
        return

    webhook_url = delivery_webhook_url(telegram_id)
    await _answer(event, t(ui_language, "processing_request"))

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
            digest_language=digest_language if isinstance(digest_language, str) else None,
        )
    except Exception:
        logger.exception("Failed to create subscription for telegram_id=%d", telegram_id)
        await _answer(event, t(ui_language, "create_subscription_failed"))
        await state.clear()
        return

    topics_str = ", ".join(subscription.topics)
    completion_key = (
        "subscription_created_event" if delivery_mode == "event" else "subscription_created_digest"
    )
    await _answer(event, t(ui_language, completion_key, topics=topics_str))
    if delivery_mode == "event":
        await state.update_data(created_subscription_id=subscription.id)
        await state.set_state(SubscribeFlow.waiting_for_recent_events_decision)
        await _answer_with_markup(
            event,
            t(ui_language, "show_recent_events_prompt"),
            _recent_events_choice_keyboard(ui_language),
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
    await _answer(event, _subscription_prompt_text(await _ui_language_for_event(event)))


async def _show_schedule_decision_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    back_target: str = "prompt",
) -> None:
    ui_language = await _ui_language_for_event(event)
    await state.update_data(schedule_back_target=back_target)
    await state.set_state(SubscribeFlow.waiting_for_schedule_decision)
    await _answer_with_markup(
        event,
        t(ui_language, "schedule_choice"),
        _schedule_choice_keyboard(ui_language),
    )


async def _show_schedule_input_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    ui_language = await _ui_language_for_event(event)
    await state.set_state(SubscribeFlow.waiting_for_schedule_input)
    await _answer_with_markup(
        event,
        t(ui_language, "schedule_input_prompt"),
        _back_only_keyboard(ui_language),
    )


async def _show_source_knowledge_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    delivery_mode: str | None = None,
) -> None:
    ui_language = await _ui_language_for_event(event)
    if delivery_mode is None:
        state_data = await state.get_data()
        delivery_mode = str(state_data.get("delivery_mode", "digest"))
    await state.set_state(SubscribeFlow.waiting_for_source_knowledge)
    await _answer_with_markup(
        event,
        _source_knowledge_question(ui_language, delivery_mode),
        _source_knowledge_keyboard(ui_language),
    )


async def _show_language_choice_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    ui_language = await _ui_language_for_event(event)
    await state.set_state(SubscribeFlow.waiting_for_language_choice)
    await _answer_with_markup(
        event,
        t(ui_language, "subscription_language_choose"),
        _subscription_language_keyboard(ui_language),
    )


async def _show_channels_input_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    ui_language = await _ui_language_for_event(event)
    await state.set_state(SubscribeFlow.waiting_for_channels_input)
    await _answer_with_markup(
        event,
        t(ui_language, "channels_input_prompt"),
        _back_only_keyboard(ui_language),
    )


async def _show_scope_choice_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    delivery_mode: str | None = None,
    channels: list[str] | None = None,
    origin: str | None = None,
) -> None:
    ui_language = await _ui_language_for_event(event)
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
        text = t(
            ui_language,
            "scope_prompt_found",
            channels=_format_channels(channels),
            question=_scope_question(ui_language, delivery_mode),
        )
    else:
        text = t(
            ui_language,
            "scope_prompt_manual",
            question=_scope_question(ui_language, delivery_mode),
            channels=_format_channels(channels),
        )

    await _answer_with_markup(event, text, _scope_keyboard(ui_language))


async def _show_source_previous_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
) -> None:
    state_data = await state.get_data()
    target = str(state_data.get("source_back_target", "prompt"))

    if target == "language_choice":
        await _show_language_choice_step(event, state)
        return
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

    if target == "language_choice":
        await _show_language_choice_step(event, state)
        return
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
    ui_language = await _ui_language_or_default(callback.from_user.id)
    if not isinstance(subscription_id, str) or not subscription_id:
        await _answer(callback, t(ui_language, "undo_recent_events_expired"))
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
        await _answer(callback, t(ui_language, "undo_recent_events_failed"))
        return

    await state.update_data(created_subscription_id=None)
    target = str(state_data.get("creation_back_target", "source_knowledge"))
    if target == "scope_choice":
        await _show_scope_choice_step(callback, state)
        return

    await _show_source_knowledge_step(callback, state)


def _source_knowledge_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_yes_have_channels"),
                    callback_data=KNOW_SOURCES_YES,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_no_find_sources"),
                    callback_data=KNOW_SOURCES_NO,
                ),
            ],
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _subscription_language_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_russian"),
                    callback_data=SUBSCRIPTION_LANGUAGE_RU,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_english"),
                    callback_data=SUBSCRIPTION_LANGUAGE_EN,
                ),
            ],
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _scope_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_only_channels"),
                    callback_data=SCOPE_ONLY_PROVIDED,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_add_sources"),
                    callback_data=SCOPE_WITH_DISCOVERY,
                ),
            ],
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _schedule_choice_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_yes_set_schedule"),
                    callback_data=SCHEDULE_ENABLE_YES,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_no_button_only"),
                    callback_data=SCHEDULE_ENABLE_NO,
                ),
            ],
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _recent_events_choice_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_yes_show_recent"),
                    callback_data=RECENT_EVENTS_YES,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_no_future_only"),
                    callback_data=RECENT_EVENTS_NO,
                ),
            ],
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _back_only_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t(ui_language, "button_back"), callback_data=BACK)],
        ]
    )


def _subscription_prompt_text(ui_language: UILanguage) -> str:
    return t(ui_language, "subscription_prompt")


def _scope_question(ui_language: UILanguage, delivery_mode: str) -> str:
    if delivery_mode == "event":
        return t(ui_language, "scope_question_event")
    return t(ui_language, "scope_question_digest")


def _source_knowledge_question(ui_language: UILanguage, delivery_mode: str) -> str:
    if delivery_mode == "event":
        return t(ui_language, "source_question_event")
    return t(ui_language, "source_question_digest")


def _format_channels(channels: list[str]) -> str:
    return "\n".join(f"- @{channel}" for channel in channels)


def _telegram_id_from_event(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"


async def _ui_language_for_event(event: types.Message | CallbackQuery) -> UILanguage:
    return await _ui_language_or_default(_telegram_id_from_event(event))


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


async def _continue_after_prompt(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    back_target: str,
) -> None:
    state_data = await state.get_data()
    prompt = state_data.get("prompt")
    delivery_mode = str(state_data.get("delivery_mode", "digest"))
    schedule_cron_override = state_data.get("schedule_cron_override")
    schedule_was_explicit = bool(state_data.get("schedule_was_explicit"))
    if not isinstance(prompt, str) or not prompt.strip():
        await _answer(
            event,
            t(await _ui_language_for_event(event), "subscription_setup_expired"),
        )
        await state.clear()
        return

    if delivery_mode == "event":
        await _continue_with_source_flow(
            event,
            state,
            prompt,
            delivery_mode,
            source_back_target=back_target,
        )
        return

    if schedule_was_explicit and schedule_cron_override:
        await _continue_with_source_flow(
            event,
            state,
            prompt,
            delivery_mode,
            source_back_target=back_target,
        )
        return

    await _show_schedule_decision_step(event, state, back_target=back_target)
