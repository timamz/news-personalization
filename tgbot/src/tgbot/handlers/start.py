import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from aiogram import Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from tgbot.client import BackendClient
from tgbot.language import LanguagePreference, UILanguage
from tgbot.storage import (
    get_language_preference,
    get_ui_language,
    save_language_preference,
    save_ui_language,
)
from tgbot.telegram_format import render_html_message
from tgbot.ui_text import interface_language_name, subscription_preference_summary, t
from tgbot.user_registry import ensure_api_key

logger = logging.getLogger(__name__)

router = Router()
backend = BackendClient()

UI_LANGUAGE_RU = "ui_language:ru"
UI_LANGUAGE_EN = "ui_language:en"
SUBSCRIPTION_LANGUAGE_FIXED_RU = "subscription_language:fixed:ru"
SUBSCRIPTION_LANGUAGE_FIXED_EN = "subscription_language:fixed:en"
SUBSCRIPTION_LANGUAGE_ASK = "subscription_language:ask"
TIMEZONE_CONFIRM = "timezone:confirm"
TIMEZONE_RETRY = "timezone:retry"
TIMEZONE_SELECT_PREFIX = "timezone:select:"

_SETUP_NEXT_ACTION = "setup_next_action"
_SETUP_REQUIRE_SUBSCRIPTION = "setup_require_subscription_after_ui"
_SETUP_PENDING_TIMEZONE = "setup_pending_timezone"
_SETUP_PENDING_TIMEZONE_LABEL = "setup_pending_timezone_label"
_SETUP_PENDING_TIMEZONE_LOCAL_TIME = "setup_pending_timezone_local_time"
_SETUP_TIMEZONE_CHOICES = "setup_timezone_choices"
_SETUP_TIMEZONE_EDIT = "setup_timezone_edit"


class SetupFlow(StatesGroup):
    waiting_for_timezone_city = State()


@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to register user for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    if not await ensure_user_setup(
        message,
        state,
        api_key=api_key,
        next_action="welcome",
        reset_state=True,
    ):
        return

    ui_language = await _ui_language_or_default(telegram_id)
    await message.answer(t(ui_language, "welcome"))


@router.message(Command("language"))
async def cmd_language(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    current_ui_language = await _ui_language_or_default(telegram_id)
    await prompt_ui_language_selection(
        message,
        state,
        current_ui_language=current_ui_language,
        next_action=None,
        require_subscription_language_after_ui=False,
        initial=False,
        reset_state=False,
    )


@router.message(Command("subscription_language"))
async def cmd_subscription_language(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    try:
        await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await message.answer(t("en", "registration_failed"))
        return

    ui_language = await get_ui_language(telegram_id)
    if ui_language is None:
        await prompt_ui_language_selection(
            message,
            state,
            current_ui_language="en",
            next_action=None,
            require_subscription_language_after_ui=True,
            initial=True,
            reset_state=False,
        )
        return

    preference = await get_language_preference(telegram_id)
    await prompt_subscription_language_selection(
        message,
        state,
        ui_language=ui_language,
        next_action=None,
        initial=preference is None,
        reset_state=False,
        current_preference=preference,
    )


@router.message(Command("timezone"))
async def cmd_timezone(message: types.Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        profile = await backend.get_current_user(api_key)
    except Exception:
        logger.exception("Failed to load timezone settings for telegram_id=%d", telegram_id)
        await message.answer(t(ui_language, "registration_failed"))
        return

    await prompt_timezone_selection(
        message,
        state,
        ui_language=ui_language,
        next_action=None,
        initial=profile.timezone is None,
        reset_state=True,
        current_timezone=profile.timezone,
        editing=True,
    )


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    ui_language = await _ui_language_or_default(message.from_user.id)
    await message.answer(t(ui_language, "welcome"))


@router.callback_query(lambda c: c.data in {UI_LANGUAGE_RU, UI_LANGUAGE_EN})
async def handle_ui_language_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    ui_language = _ui_language_from_callback(callback.data)
    if ui_language is None:
        return

    telegram_id = callback.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(callback, t(ui_language, "registration_failed"))
        return

    state_data = await state.get_data()
    next_action = state_data.get(_SETUP_NEXT_ACTION)
    require_subscription = bool(state_data.get(_SETUP_REQUIRE_SUBSCRIPTION))

    await save_ui_language(telegram_id, api_key, ui_language)

    if require_subscription and await get_language_preference(telegram_id) is None:
        await prompt_subscription_language_selection(
            callback,
            state,
            ui_language=ui_language,
            next_action=next_action if isinstance(next_action, str) else None,
            initial=True,
            reset_state=False,
        )
        return

    await _continue_after_setup_step(
        callback,
        state,
        api_key=api_key,
        ui_language=ui_language,
        next_action=next_action if isinstance(next_action, str) else None,
        fallback_message=t(
            ui_language,
            "ui_language_updated",
            language=interface_language_name(ui_language, ui_language),
        ),
    )


@router.callback_query(
    lambda c: (
        c.data
        in {
            SUBSCRIPTION_LANGUAGE_FIXED_RU,
            SUBSCRIPTION_LANGUAGE_FIXED_EN,
            SUBSCRIPTION_LANGUAGE_ASK,
        }
    )
)
async def handle_subscription_language_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    preference = _subscription_preference_from_callback(callback.data)
    if preference is None:
        return

    telegram_id = callback.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
    except Exception:
        logger.exception("Failed to ensure API key for telegram_id=%d", telegram_id)
        await _answer(callback, t(ui_language, "registration_failed"))
        return

    state_data = await state.get_data()
    next_action = state_data.get(_SETUP_NEXT_ACTION)
    previous_preference = await get_language_preference(telegram_id)
    await save_language_preference(telegram_id, api_key, preference)

    update_failed = False
    updated = 0
    failed = 0
    if preference.mode == "fixed" and preference.code is not None:
        try:
            updated, failed = await _apply_fixed_language_to_existing_subscriptions(
                api_key,
                preference.code,
            )
        except Exception:
            logger.exception(
                "Failed to update existing subscription languages for telegram_id=%d",
                telegram_id,
            )
            update_failed = True

    await state.update_data(
        **{
            _SETUP_NEXT_ACTION: None,
            _SETUP_REQUIRE_SUBSCRIPTION: False,
        }
    )

    await _answer(
        callback,
        _subscription_language_confirmation(
            ui_language,
            previous_preference,
            preference,
            updated=updated,
            failed=failed,
            update_failed=update_failed,
        ),
    )
    await _continue_after_setup_step(
        callback,
        state,
        api_key=api_key,
        ui_language=ui_language,
        next_action=next_action if isinstance(next_action, str) else None,
        fallback_message=None,
    )


@router.message(SetupFlow.waiting_for_timezone_city)
async def handle_timezone_city_input(message: types.Message, state: FSMContext) -> None:
    timezone_query = (message.text or "").strip()
    ui_language = await _ui_language_or_default(message.from_user.id)
    if not timezone_query:
        await message.answer(t(ui_language, "timezone_input_empty"))
        return

    telegram_id = message.from_user.id
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        resolution = await backend.resolve_timezone(api_key, timezone_query)
    except Exception:
        logger.exception("Failed to resolve timezone for telegram_id=%d", telegram_id)
        await message.answer(t(ui_language, "timezone_lookup_failed"))
        return

    if resolution.status == "not_found" or not resolution.candidates:
        await state.update_data(
            **{
                _SETUP_PENDING_TIMEZONE: None,
                _SETUP_PENDING_TIMEZONE_LABEL: None,
                _SETUP_PENDING_TIMEZONE_LOCAL_TIME: None,
                _SETUP_TIMEZONE_CHOICES: [],
            }
        )
        await message.answer(t(ui_language, "timezone_not_found"))
        return

    if resolution.status == "ambiguous":
        choices = [
            {
                "label": candidate.label,
                "timezone": candidate.timezone,
                "local_time": candidate.local_time,
            }
            for candidate in resolution.candidates
        ]
        await state.update_data(
            **{
                _SETUP_PENDING_TIMEZONE: None,
                _SETUP_PENDING_TIMEZONE_LABEL: None,
                _SETUP_PENDING_TIMEZONE_LOCAL_TIME: None,
                _SETUP_TIMEZONE_CHOICES: choices,
            }
        )
        await message.answer(
            t(ui_language, "timezone_ambiguous"),
            reply_markup=_timezone_choice_keyboard(choices, ui_language),
        )
        return

    candidate = resolution.candidates[0]
    await state.update_data(
        **{
            _SETUP_PENDING_TIMEZONE: candidate.timezone,
            _SETUP_PENDING_TIMEZONE_LABEL: candidate.label,
            _SETUP_PENDING_TIMEZONE_LOCAL_TIME: candidate.local_time,
            _SETUP_TIMEZONE_CHOICES: [],
        }
    )
    await message.answer(
        t(
            ui_language,
            "timezone_confirm",
            location=candidate.label,
            timezone=candidate.timezone,
            local_time=_format_backend_datetime(candidate.local_time),
        ),
        reply_markup=_timezone_confirm_keyboard(ui_language),
    )


@router.callback_query(lambda c: c.data == TIMEZONE_CONFIRM)
async def handle_timezone_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    state_data = await state.get_data()
    timezone_name = state_data.get(_SETUP_PENDING_TIMEZONE)
    timezone_label = state_data.get(_SETUP_PENDING_TIMEZONE_LABEL)
    local_time = state_data.get(_SETUP_PENDING_TIMEZONE_LOCAL_TIME)
    if not isinstance(timezone_name, str) or not isinstance(timezone_label, str):
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await _answer(callback, t(ui_language, "timezone_retry"))
        return
    await _save_timezone_selection(
        callback,
        state,
        timezone_name=timezone_name,
        timezone_label=timezone_label,
        local_time=local_time if isinstance(local_time, str) else None,
    )


@router.callback_query(lambda c: c.data == TIMEZONE_RETRY)
async def handle_timezone_retry(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    telegram_id = callback.from_user.id
    ui_language = await _ui_language_or_default(telegram_id)
    state_data = await state.get_data()
    next_action = state_data.get(_SETUP_NEXT_ACTION)
    current_timezone = None
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        profile = await backend.get_current_user(api_key)
        current_timezone = profile.timezone
    except Exception:
        logger.exception("Failed to reload timezone state for telegram_id=%d", telegram_id)

    await prompt_timezone_selection(
        callback,
        state,
        ui_language=ui_language,
        next_action=next_action if isinstance(next_action, str) else None,
        initial=current_timezone is None,
        reset_state=False,
        current_timezone=current_timezone,
        editing=bool(state_data.get(_SETUP_TIMEZONE_EDIT)),
    )


@router.callback_query(lambda c: c.data and c.data.startswith(TIMEZONE_SELECT_PREFIX))
async def handle_timezone_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    state_data = await state.get_data()
    choices = state_data.get(_SETUP_TIMEZONE_CHOICES, [])
    if not isinstance(choices, list):
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await _answer(callback, t(ui_language, "timezone_retry"))
        return

    payload = callback.data[len(TIMEZONE_SELECT_PREFIX) :]
    try:
        choice_index = int(payload)
    except ValueError:
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await _answer(callback, t(ui_language, "timezone_retry"))
        return
    if choice_index < 0 or choice_index >= len(choices):
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await _answer(callback, t(ui_language, "timezone_retry"))
        return

    choice = choices[choice_index]
    timezone_name = choice.get("timezone")
    timezone_label = choice.get("label")
    local_time = choice.get("local_time")
    if not isinstance(timezone_name, str) or not isinstance(timezone_label, str):
        ui_language = await _ui_language_or_default(callback.from_user.id)
        await _answer(callback, t(ui_language, "timezone_retry"))
        return

    await _save_timezone_selection(
        callback,
        state,
        timezone_name=timezone_name,
        timezone_label=timezone_label,
        local_time=local_time if isinstance(local_time, str) else None,
    )


async def ensure_user_setup(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    api_key: str,
    next_action: str | None,
    reset_state: bool,
) -> bool:
    telegram_id = _telegram_id_from_event(event)
    ui_language = await get_ui_language(telegram_id)
    if ui_language is None:
        await prompt_ui_language_selection(
            event,
            state,
            current_ui_language="en",
            next_action=next_action,
            require_subscription_language_after_ui=True,
            initial=True,
            reset_state=reset_state,
        )
        return False

    preference = await get_language_preference(telegram_id)
    if preference is None:
        await prompt_subscription_language_selection(
            event,
            state,
            ui_language=ui_language,
            next_action=next_action,
            initial=True,
            reset_state=reset_state,
        )
        return False

    try:
        profile = await backend.get_current_user(api_key)
    except Exception:
        logger.exception("Failed to load user profile for telegram_id=%d", telegram_id)
        await _answer(event, t(ui_language, "registration_failed"))
        return False
    if profile.timezone is None:
        await prompt_timezone_selection(
            event,
            state,
            ui_language=ui_language,
            next_action=next_action,
            initial=True,
            reset_state=reset_state,
            current_timezone=None,
            editing=False,
        )
        return False
    return True


async def prompt_ui_language_selection(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    current_ui_language: UILanguage,
    next_action: str | None,
    require_subscription_language_after_ui: bool,
    initial: bool,
    reset_state: bool,
) -> None:
    if reset_state:
        await state.clear()
    await state.update_data(
        **{
            _SETUP_NEXT_ACTION: next_action,
            _SETUP_REQUIRE_SUBSCRIPTION: require_subscription_language_after_ui,
        }
    )
    text_key = "ui_language_initial" if initial else "ui_language_current"
    text = (
        t(current_ui_language, text_key)
        if initial
        else t(
            current_ui_language,
            text_key,
            current_language=interface_language_name(current_ui_language, current_ui_language),
        )
    )
    await _answer_with_markup(event, text, _ui_language_keyboard(current_ui_language))


async def prompt_subscription_language_selection(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    ui_language: UILanguage,
    next_action: str | None,
    initial: bool,
    reset_state: bool,
    current_preference: LanguagePreference | None = None,
) -> None:
    if reset_state:
        await state.clear()
    await state.update_data(
        **{
            _SETUP_NEXT_ACTION: next_action,
            _SETUP_REQUIRE_SUBSCRIPTION: False,
        }
    )
    text = (
        t(ui_language, "subscription_language_initial")
        if initial
        else t(
            ui_language,
            "subscription_language_current",
            summary=subscription_preference_summary(ui_language, current_preference),
        )
    )
    await _answer_with_markup(event, text, _subscription_language_keyboard(ui_language))


async def prompt_timezone_selection(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    ui_language: UILanguage,
    next_action: str | None,
    initial: bool,
    reset_state: bool,
    current_timezone: str | None,
    editing: bool,
) -> None:
    if reset_state:
        await state.clear()
    await state.set_state(SetupFlow.waiting_for_timezone_city)
    await state.update_data(
        **{
            _SETUP_NEXT_ACTION: next_action,
            _SETUP_REQUIRE_SUBSCRIPTION: False,
            _SETUP_PENDING_TIMEZONE: None,
            _SETUP_PENDING_TIMEZONE_LABEL: None,
            _SETUP_PENDING_TIMEZONE_LOCAL_TIME: None,
            _SETUP_TIMEZONE_CHOICES: [],
            _SETUP_TIMEZONE_EDIT: editing,
        }
    )
    if initial or current_timezone is None:
        text = t(ui_language, "timezone_initial")
    else:
        text = t(
            ui_language,
            "timezone_current",
            timezone=current_timezone,
            local_time=_format_timezone_now(current_timezone),
        )
    await _answer(event, text)


async def _apply_fixed_language_to_existing_subscriptions(
    api_key: str,
    digest_language: str,
) -> tuple[int, int]:
    subscriptions = await backend.list_subscriptions(api_key)
    updated = 0
    failed = 0
    for subscription in subscriptions:
        if subscription.digest_language == digest_language:
            continue
        try:
            await backend.update_subscription(
                api_key,
                subscription.id,
                digest_language=digest_language,
            )
            updated += 1
        except Exception:
            failed += 1
            logger.exception("Failed to update language for subscription %s", subscription.id)
    return updated, failed


async def _continue_after_setup_step(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    api_key: str,
    ui_language: UILanguage,
    next_action: str | None,
    fallback_message: str | None,
) -> None:
    try:
        profile = await backend.get_current_user(api_key)
    except Exception:
        logger.exception("Failed to load user profile during setup continuation")
        await _answer(event, t(ui_language, "registration_failed"))
        return

    if profile.timezone is None:
        await prompt_timezone_selection(
            event,
            state,
            ui_language=ui_language,
            next_action=next_action,
            initial=True,
            reset_state=False,
            current_timezone=None,
            editing=False,
        )
        return

    await _finish_setup(
        event,
        state,
        ui_language=ui_language,
        next_action=next_action,
        fallback_message=fallback_message,
    )


async def _finish_setup(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    ui_language: UILanguage,
    next_action: str | None,
    fallback_message: str | None,
) -> None:
    await state.update_data(
        **{
            _SETUP_NEXT_ACTION: None,
            _SETUP_REQUIRE_SUBSCRIPTION: False,
            _SETUP_PENDING_TIMEZONE: None,
            _SETUP_PENDING_TIMEZONE_LABEL: None,
            _SETUP_PENDING_TIMEZONE_LOCAL_TIME: None,
            _SETUP_TIMEZONE_CHOICES: [],
            _SETUP_TIMEZONE_EDIT: False,
        }
    )

    if next_action == "welcome":
        await state.clear()
        await _answer(event, t(ui_language, "welcome"))
        return
    if next_action == "subscribe":
        from tgbot.handlers import subscribe as subscribe_handler

        await subscribe_handler._show_prompt_step(event, state, reset_data=True)
        return
    if fallback_message is not None:
        await state.clear()
        await _answer(event, fallback_message)
        return
    await state.clear()


async def _save_timezone_selection(
    event: types.Message | CallbackQuery,
    state: FSMContext,
    *,
    timezone_name: str,
    timezone_label: str,
    local_time: str | None,
) -> None:
    telegram_id = _telegram_id_from_event(event)
    ui_language = await _ui_language_or_default(telegram_id)
    try:
        api_key = await ensure_api_key(telegram_id, backend)
        await backend.update_user_timezone(api_key, timezone_name)
    except Exception:
        logger.exception("Failed to save timezone for telegram_id=%d", telegram_id)
        await _answer(event, t(ui_language, "timezone_save_failed"))
        return

    state_data = await state.get_data()
    next_action = state_data.get(_SETUP_NEXT_ACTION)
    editing = bool(state_data.get(_SETUP_TIMEZONE_EDIT))
    formatted_local_time = (
        _format_backend_datetime(local_time)
        if local_time is not None
        else _format_timezone_now(timezone_name)
    )
    await _answer(
        event,
        t(
            ui_language,
            "timezone_updated" if editing else "timezone_saved",
            location=timezone_label,
            timezone=timezone_name,
            local_time=formatted_local_time,
        ),
    )
    await _finish_setup(
        event,
        state,
        ui_language=ui_language,
        next_action=next_action if isinstance(next_action, str) else None,
        fallback_message=None,
    )


def _subscription_language_confirmation(
    ui_language: UILanguage,
    previous_preference: LanguagePreference | None,
    new_preference: LanguagePreference,
    *,
    updated: int,
    failed: int,
    update_failed: bool,
) -> str:
    del previous_preference
    if new_preference.mode == "ask":
        return t(ui_language, "subscription_language_saved_ask")

    language = interface_language_name(ui_language, new_preference.code or "en")
    if update_failed:
        return t(ui_language, "subscription_language_saved_fixed_failed", language=language)
    if failed:
        return t(
            ui_language,
            "subscription_language_saved_fixed_partial",
            language=language,
            updated=updated,
            failed=failed,
        )
    return t(
        ui_language,
        "subscription_language_saved_fixed",
        language=language,
        updated=updated,
    )


def _ui_language_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_english"),
                    callback_data=UI_LANGUAGE_EN,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_russian"),
                    callback_data=UI_LANGUAGE_RU,
                ),
            ]
        ]
    )


def _subscription_language_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_russian"),
                    callback_data=SUBSCRIPTION_LANGUAGE_FIXED_RU,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_english"),
                    callback_data=SUBSCRIPTION_LANGUAGE_FIXED_EN,
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_ask_every_time"),
                    callback_data=SUBSCRIPTION_LANGUAGE_ASK,
                )
            ],
        ]
    )


def _timezone_confirm_keyboard(ui_language: UILanguage) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t(ui_language, "button_confirm"),
                    callback_data=TIMEZONE_CONFIRM,
                ),
                InlineKeyboardButton(
                    text=t(ui_language, "button_try_another_city"),
                    callback_data=TIMEZONE_RETRY,
                ),
            ]
        ]
    )


def _timezone_choice_keyboard(
    choices: list[dict[str, str]],
    ui_language: UILanguage,
) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                text=choice["label"],
                callback_data=f"{TIMEZONE_SELECT_PREFIX}{index}",
            )
        ]
        for index, choice in enumerate(choices)
    ]
    buttons.append(
        [
            InlineKeyboardButton(
                text=t(ui_language, "button_try_another_city"),
                callback_data=TIMEZONE_RETRY,
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _ui_language_from_callback(callback_data: str | None) -> UILanguage | None:
    if callback_data == UI_LANGUAGE_RU:
        return "ru"
    if callback_data == UI_LANGUAGE_EN:
        return "en"
    return None


def _subscription_preference_from_callback(
    callback_data: str | None,
) -> LanguagePreference | None:
    if callback_data == SUBSCRIPTION_LANGUAGE_FIXED_RU:
        return LanguagePreference(mode="fixed", code="ru")
    if callback_data == SUBSCRIPTION_LANGUAGE_FIXED_EN:
        return LanguagePreference(mode="fixed", code="en")
    if callback_data == SUBSCRIPTION_LANGUAGE_ASK:
        return LanguagePreference(mode="ask", code=None)
    return None


async def _ui_language_or_default(telegram_id: int) -> UILanguage:
    return await get_ui_language(telegram_id) or "en"


def _format_timezone_now(timezone_name: str) -> str:
    now = datetime.now(UTC).astimezone(ZoneInfo(timezone_name))
    return now.strftime("%Y-%m-%d %H:%M")


def _format_backend_datetime(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")


def _telegram_id_from_event(event: types.Message | CallbackQuery) -> int:
    return event.from_user.id


async def _answer(event: types.Message | CallbackQuery, text: str) -> None:
    if hasattr(event, "message"):
        if event.message is None:
            return
        await event.message.answer(
            render_html_message(text),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return
    await event.answer(text)


async def _answer_with_markup(
    event: types.Message | CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    if hasattr(event, "message"):
        if event.message is None:
            return
        await event.message.answer(
            render_html_message(text),
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return
    await event.answer(text, reply_markup=reply_markup)
