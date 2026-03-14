from tgbot.language import LanguagePreference, UILanguage, normalize_language_code

_TEXTS: dict[UILanguage, dict[str, str]] = {
    "en": {
        "welcome": (
            "Welcome to News Bot!\n\n"
            "I deliver personalized news digests and event notifications "
            "right here in Telegram.\n\n"
            "Commands:\n"
            "/subscribe - set up a news subscription\n"
            "/list - view your active subscriptions\n"
            "/language - change bot interface language\n"
            "/timezone - change your time zone\n"
            "/subscription_language - change default subscription language behavior\n"
            "/help - show this message"
        ),
        "registration_failed": "Registration failed. Please try again later.",
        "ui_language_initial": "Choose bot interface language.",
        "ui_language_current": (
            "Current bot interface language: {current_language}.\n\n"
            "Choose a new interface language."
        ),
        "ui_language_updated": "Bot interface language updated to {language}.",
        "subscription_language_initial": (
            "Now choose how subscription output language should work.\n\n"
            "You can keep one language for all subscriptions or ask every time."
        ),
        "subscription_language_current": (
            "Current subscription language setting: {summary}.\n\n"
            "Choose a new default for digests and event notifications."
        ),
        "subscription_language_saved_ask": (
            "Saved. I will ask for the subscription language during each new subscription flow.\n"
            "Existing subscriptions keep their current language."
        ),
        "subscription_language_saved_fixed": (
            "Saved. New subscriptions will use {language}.\n"
            "Updated {updated} existing subscriptions."
        ),
        "subscription_language_saved_fixed_partial": (
            "Saved. New subscriptions will use {language}.\n"
            "Updated {updated} existing subscriptions, but {failed} updates failed."
        ),
        "subscription_language_saved_fixed_failed": (
            "Saved. New subscriptions will use {language}.\n"
            "I couldn't update existing subscriptions right now. "
            "Run /subscription_language again later."
        ),
        "timezone_initial": (
            "What city are you in?\n\n"
            "This lets me deliver digests at your local time.\n"
            "Examples: Berlin, Tbilisi, New York, or Берлин.\n"
            "You can also send a time zone like Europe/Berlin."
        ),
        "timezone_current": (
            "Current time zone: {timezone}.\n"
            "Local time there is {local_time}.\n\n"
            "Send a new city or time zone."
        ),
        "timezone_input_empty": "Send a city or time zone.",
        "timezone_lookup_failed": "I couldn't look up that city right now. Please try again.",
        "timezone_not_found": (
            "I couldn't match that city. Try a clearer form like Berlin, Берлин, "
            "or Paris, France."
        ),
        "timezone_ambiguous": (
            "I found multiple matches. Choose the right one or send another city."
        ),
        "timezone_confirm": (
            "I found {location}.\n"
            "Time zone: {timezone}\n"
            "Local time there: {local_time}\n\n"
            "Is this correct?"
        ),
        "timezone_saved": (
            "Saved.\n"
            "{location}\n"
            "{timezone}\n"
            "Local time: {local_time}"
        ),
        "timezone_updated": (
            "Time zone updated.\n"
            "{location}\n"
            "{timezone}\n"
            "Local time: {local_time}"
        ),
        "timezone_save_failed": "I couldn't save the time zone. Please try again.",
        "timezone_retry": "Send another city or time zone.",
        "choose_subscription_request": "Please describe your subscription request.",
        "failed_process_request": "Failed to process your request. Please try again.",
        "subscription_setup_expired": "Subscription setup expired. Please run /subscribe again.",
        "describe_schedule": "Please describe the schedule.",
        "schedule_parse_failed": "Couldn't parse this schedule. Please try another wording.",
        "sources_parse_failed": (
            "I couldn't parse sources. Send Telegram handles like @channel_one, "
            "links like t.me/channel or x.com/OpenAI, "
            "or Reddit names like r/python."
        ),
        "recent_events_future_only": "Okay. I will only send new event notifications from now on.",
        "recent_events_expired": (
            "This preview expired. Please create a new subscription if needed."
        ),
        "recent_events_loading": "Checking what you might have missed in the last 7 days...",
        "recent_events_failed": "Couldn't load recent events right now.",
        "recent_events_empty": "No matching events were found in the last 7 days.",
        "recent_events_header": "Here's what you might have missed in the last 7 days:",
        "back_not_available": "Back is not available here.",
        "processing_request": "Processing your request...",
        "create_subscription_failed": "Failed to create subscription. Please try again.",
        "subscription_created_digest": (
            "Subscription created!\n\nRequest: {prompt_summary}\n\n"
            "You'll receive digests right here in this chat."
        ),
        "subscription_created_event": (
            "Subscription created!\n\nRequest: {prompt_summary}\n\n"
            "You'll receive event notifications right here in this chat."
        ),
        "show_recent_events_prompt": (
            "Would you like to see what you might have missed in the last 7 days?"
        ),
        "subscription_prompt": (
            "Describe what news you want.\n\n"
            "Examples:\n"
            '- "I want AI and tech news every morning as a brief summary"\n'
            '- "Notify me when a new episode of Severance is announced"'
        ),
        "subscription_language_choose": (
            "Choose the language for this subscription's digests and event notifications."
        ),
        "schedule_choice": (
            "Do you want this digest to be delivered automatically on a schedule?\n"
            "You can always use the Send now button."
        ),
        "schedule_input_prompt": (
            'Describe the schedule in natural language.\nExample: "every weekday at 9:00"'
        ),
        "source_question_digest": (
            "Do you already have specific sources for this digest "
            "(Telegram channels, Reddit subreddits, or X accounts)?"
        ),
        "source_question_event": (
            "Do you already have specific sources for these notifications "
            "(Telegram channels, Reddit subreddits, or X accounts)?"
        ),
        "channels_input_prompt": (
            "Send sources you want to include "
            "(for example: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "scope_prompt_found": (
            "I found these sources in your request:\n{channels}\n\n{question}"
        ),
        "scope_prompt_manual": "Got it. {question}\n{channels}",
        "scope_question_digest": "Should this digest be limited only to these sources?",
        "scope_question_event": "Should these notifications be limited only to these sources?",
        "undo_recent_events_failed": "Couldn't go back right now. Please try again.",
        "undo_recent_events_expired": "This step can no longer be undone.",
        "edit_menu_prompt": "What do you want to change?",
        "no_subscriptions": "You have no active subscriptions. Use /subscribe to create one.",
        "failed_load_subscriptions": "Failed to load subscriptions. Please try again.",
        "subscription_card": "Request: {prompt_summary}\nType: {type}\nLanguage: {language}",
        "type_digest": "Digest",
        "type_event": "Event notifications",
        "edit_subscription_language_prompt": "Choose the language for this subscription.",
        "subscription_language_updated": "Language updated to {language}.",
        "subscription_language_update_failed": "Failed to update language. Try again.",
        "digest_queued": "Digest queued.",
        "digest_queue_failed": "Failed to queue digest. Try again.",
        "edit_schedule_prompt": (
            'Describe the new schedule in natural language.\nExample: "every weekday at 9:00"'
        ),
        "edit_session_expired": "Edit session expired. Open /list and try again.",
        "schedule_updated": "Schedule updated.",
        "schedule_update_failed": "Failed to update schedule. Please try again.",
        "schedule_disabled": "Automatic schedule disabled.",
        "schedule_disable_failed": "Failed to update schedule. Try again.",
        "edit_request_prompt": "Describe what should change in this subscription.",
        "edit_request_empty": "Please describe what should change.",
        "edit_request_preview": (
            "Proposed update:\n\n"
            "Request: {prompt_summary}\n"
            "Format: {format_instructions}\n"
            "Change: {change_summary}\n\n"
            "Sources, schedule, and language stay unchanged."
        ),
        "edit_request_failed": "Failed to prepare the update. Please try again.",
        "edit_request_applied": "Subscription updated.\n\nRequest: {prompt_summary}",
        "edit_request_apply_failed": "Failed to save the update. Please try again.",
        "edit_request_cancelled": "Update cancelled.",
        "edit_sources_prompt": (
            "Send sources to add "
            "(for example: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "sources_added": "Added {count} sources.",
        "sources_already_added": "Those sources are already included.",
        "sources_add_failed": "Failed to add sources. Please try again.",
        "confirm_delete_prompt": "Are you sure you want to delete this subscription?",
        "delete_cancelled": "Deletion cancelled.",
        "subscription_deleted": "Subscription deleted.",
        "subscription_delete_failed": "Failed to delete. Try again.",
        "button_english": "English",
        "button_russian": "Russian",
        "button_ask_every_time": "Ask every time",
        "button_send_now": "Send now",
        "button_edit": "Edit",
        "button_delete": "Delete",
        "button_confirm_delete": "Yes, delete",
        "button_change_schedule": "Change schedule",
        "button_disable_schedule": "Disable schedule",
        "button_edit_request": "Edit request",
        "button_change_language": "Change language",
        "button_add_edit_sources": "Add sources",
        "button_confirm": "Confirm",
        "button_revise": "Revise",
        "button_cancel": "Cancel",
        "button_try_another_city": "Another city",
        "button_yes_set_schedule": "Yes, set schedule",
        "button_no_button_only": "No, send only by button",
        "button_yes_have_channels": "Yes, I have sources",
        "button_no_find_sources": "No, find sources for me",
        "button_only_channels": "Only these sources",
        "button_add_sources": "Add more sources",
        "button_yes_show_recent": "Yes, show missed events",
        "button_no_future_only": "No, only future ones",
        "button_back": "Back",
        "summary_not_set": "Not set yet",
        "summary_ask": "Ask for each new subscription",
        "summary_fixed": "{language} for all subscriptions",
    },
    "ru": {
        "welcome": (
            "Добро пожаловать в News Bot!\n\n"
            "Я отправляю персональные дайджесты новостей и уведомления "
            "о событиях прямо в Telegram.\n\n"
            "Команды:\n"
            "/subscribe - создать подписку\n"
            "/list - посмотреть активные подписки\n"
            "/language - изменить язык интерфейса бота\n"
            "/timezone - изменить часовой пояс\n"
            "/subscription_language - изменить язык подписок по умолчанию\n"
            "/help - показать это сообщение"
        ),
        "registration_failed": "Не удалось зарегистрировать пользователя. Попробуйте позже.",
        "ui_language_initial": "Выберите язык интерфейса бота.",
        "ui_language_current": (
            "Текущий язык интерфейса бота: {current_language}.\n\nВыберите новый язык интерфейса."
        ),
        "ui_language_updated": "Язык интерфейса бота изменён на {language}.",
        "subscription_language_initial": (
            "Теперь выберите, как должен работать язык подписок.\n\n"
            "Можно использовать один язык для всех подписок или выбирать его каждый раз."
        ),
        "subscription_language_current": (
            "Текущая настройка языка подписок: {summary}.\n\n"
            "Выберите новый язык по умолчанию для дайджестов и уведомлений о событиях."
        ),
        "subscription_language_saved_ask": (
            "Сохранено. В каждом новом сценарии подписки я буду спрашивать язык отдельно.\n"
            "У существующих подписок текущий язык сохранится."
        ),
        "subscription_language_saved_fixed": (
            "Сохранено. Новые подписки будут использовать язык {language}.\n"
            "Обновлено существующих подписок: {updated}."
        ),
        "subscription_language_saved_fixed_partial": (
            "Сохранено. Новые подписки будут использовать язык {language}.\n"
            "Обновлено существующих подписок: {updated}, "
            "но {failed} обновлений завершились ошибкой."
        ),
        "subscription_language_saved_fixed_failed": (
            "Сохранено. Новые подписки будут использовать язык {language}.\n"
            "Сейчас не удалось обновить существующие подписки. "
            "Повторите /subscription_language позже."
        ),
        "timezone_initial": (
            "В каком вы городе?\n\n"
            "Это нужно, чтобы присылать дайджесты по вашему местному времени.\n"
            "Примеры: Берлин, Тбилиси, Нью-Йорк.\n"
            "Можно также отправить часовой пояс вроде Europe/Berlin."
        ),
        "timezone_current": (
            "Текущий часовой пояс: {timezone}.\n"
            "Локальное время там: {local_time}.\n\n"
            "Отправьте новый город или часовой пояс."
        ),
        "timezone_input_empty": "Отправьте город или часовой пояс.",
        "timezone_lookup_failed": "Сейчас не удалось определить этот город. Попробуйте ещё раз.",
        "timezone_not_found": (
            "Не удалось сопоставить этот город. Попробуйте вариант вроде Берлин "
            "или Paris, France."
        ),
        "timezone_ambiguous": (
            "Нашёл несколько вариантов. Выберите нужный или отправьте другой город."
        ),
        "timezone_confirm": (
            "Я нашёл: {location}\n"
            "Часовой пояс: {timezone}\n"
            "Локальное время: {local_time}\n\n"
            "Это правильно?"
        ),
        "timezone_saved": (
            "Сохранено.\n"
            "{location}\n"
            "{timezone}\n"
            "Локальное время: {local_time}"
        ),
        "timezone_updated": (
            "Часовой пояс обновлён.\n"
            "{location}\n"
            "{timezone}\n"
            "Локальное время: {local_time}"
        ),
        "timezone_save_failed": "Не удалось сохранить часовой пояс. Попробуйте ещё раз.",
        "timezone_retry": "Отправьте другой город или часовой пояс.",
        "choose_subscription_request": "Опишите, какую подписку вы хотите.",
        "failed_process_request": "Не удалось обработать запрос. Попробуйте ещё раз.",
        "subscription_setup_expired": (
            "Сценарий создания подписки истёк. Запустите /subscribe ещё раз."
        ),
        "describe_schedule": "Опишите расписание.",
        "schedule_parse_failed": (
            "Не удалось распознать это расписание. Попробуйте другую формулировку."
        ),
        "sources_parse_failed": (
            "Не удалось распознать источники. Отправьте Telegram-хэндлы вроде @channel_one, "
            "ссылки вроде t.me/channel или x.com/OpenAI, "
            "или сабреддиты вроде r/python."
        ),
        "recent_events_future_only": (
            "Хорошо. Я буду отправлять только новые уведомления о событиях."
        ),
        "recent_events_expired": "Предпросмотр устарел. При необходимости создайте новую подписку.",
        "recent_events_loading": "Проверяю, что вы могли пропустить за последние 7 дней...",
        "recent_events_failed": "Сейчас не удалось загрузить недавние события.",
        "recent_events_empty": "За последние 7 дней подходящих событий не найдено.",
        "recent_events_header": "Вот что вы могли пропустить за последние 7 дней:",
        "back_not_available": "Здесь нельзя вернуться назад.",
        "processing_request": "Обрабатываю ваш запрос...",
        "create_subscription_failed": "Не удалось создать подписку. Попробуйте ещё раз.",
        "subscription_created_digest": (
            "Подписка создана!\n\nЗапрос: {prompt_summary}\n\n"
            "Я буду присылать дайджесты прямо в этот чат."
        ),
        "subscription_created_event": (
            "Подписка создана!\n\nЗапрос: {prompt_summary}\n\n"
            "Я буду присылать уведомления о событиях прямо в этот чат."
        ),
        "show_recent_events_prompt": "Показать, что вы могли пропустить за последние 7 дней?",
        "subscription_prompt": (
            "Опишите, какие новости вам нужны.\n\n"
            "Примеры:\n"
            '- "Хочу новости про AI и технологии каждое утро в виде краткой сводки"\n'
            '- "Сообщай, когда анонсируют новый эпизод Severance"'
        ),
        "subscription_language_choose": (
            "Выберите язык дайджестов и уведомлений для этой подписки."
        ),
        "schedule_choice": (
            "Хотите получать этот дайджест автоматически по расписанию?\n"
            "Кнопка отправки вручную всегда останется доступной."
        ),
        "schedule_input_prompt": (
            'Опишите расписание естественным языком.\nПример: "каждый будний день в 9:00"'
        ),
        "source_question_digest": (
            "У вас уже есть конкретные источники для этого дайджеста "
            "(Telegram-каналы, Reddit-сабреддиты или аккаунты X)?"
        ),
        "source_question_event": (
            "У вас уже есть конкретные источники для этих уведомлений "
            "(Telegram-каналы, Reddit-сабреддиты или аккаунты X)?"
        ),
        "channels_input_prompt": (
            "Отправьте источники, которые хотите включить "
            "(например: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "scope_prompt_found": (
            "Я нашёл в вашем запросе такие источники:\n{channels}\n\n{question}"
        ),
        "scope_prompt_manual": "Понял. {question}\n{channels}",
        "scope_question_digest": "Ограничить этот дайджест только этими источниками?",
        "scope_question_event": "Ограничить эти уведомления только этими источниками?",
        "undo_recent_events_failed": "Сейчас не удалось вернуться назад. Попробуйте ещё раз.",
        "undo_recent_events_expired": "На этот шаг уже нельзя вернуться.",
        "edit_menu_prompt": "Что вы хотите изменить?",
        "no_subscriptions": (
            "У вас нет активных подписок. Используйте /subscribe, чтобы создать новую."
        ),
        "failed_load_subscriptions": "Не удалось загрузить подписки. Попробуйте ещё раз.",
        "subscription_card": "Запрос: {prompt_summary}\nТип: {type}\nЯзык: {language}",
        "type_digest": "Дайджест",
        "type_event": "Уведомления о событиях",
        "edit_subscription_language_prompt": "Выберите язык для этой подписки.",
        "subscription_language_updated": "Язык подписки изменён на {language}.",
        "subscription_language_update_failed": (
            "Не удалось обновить язык подписки. Попробуйте ещё раз."
        ),
        "digest_queued": "Дайджест поставлен в очередь.",
        "digest_queue_failed": "Не удалось поставить дайджест в очередь. Попробуйте ещё раз.",
        "edit_schedule_prompt": (
            'Опишите новое расписание естественным языком.\nПример: "каждый будний день в 9:00"'
        ),
        "edit_session_expired": "Сессия редактирования истекла. Откройте /list и попробуйте снова.",
        "schedule_updated": "Расписание обновлено.",
        "schedule_update_failed": "Не удалось обновить расписание. Попробуйте ещё раз.",
        "schedule_disabled": "Автоматическое расписание отключено.",
        "schedule_disable_failed": "Не удалось обновить расписание. Попробуйте ещё раз.",
        "edit_request_prompt": "Опишите, что нужно изменить в этой подписке.",
        "edit_request_empty": "Пожалуйста, опишите, что нужно изменить.",
        "edit_request_preview": (
            "Предлагаемое обновление:\n\n"
            "Запрос: {prompt_summary}\n"
            "Формат: {format_instructions}\n"
            "Изменение: {change_summary}\n\n"
            "Источники, расписание и язык не изменятся."
        ),
        "edit_request_failed": "Не удалось подготовить обновление. Попробуйте ещё раз.",
        "edit_request_applied": "Подписка обновлена.\n\nЗапрос: {prompt_summary}",
        "edit_request_apply_failed": "Не удалось сохранить обновление. Попробуйте ещё раз.",
        "edit_request_cancelled": "Обновление отменено.",
        "edit_sources_prompt": (
            "Отправьте источники, которые хотите добавить "
            "(например: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "sources_added": "Добавлено источников: {count}.",
        "sources_already_added": "Эти источники уже добавлены.",
        "sources_add_failed": "Не удалось добавить источники. Попробуйте ещё раз.",
        "confirm_delete_prompt": "Вы уверены, что хотите удалить эту подписку?",
        "delete_cancelled": "Удаление отменено.",
        "subscription_deleted": "Подписка удалена.",
        "subscription_delete_failed": "Не удалось удалить подписку. Попробуйте ещё раз.",
        "button_english": "English",
        "button_russian": "Русский",
        "button_ask_every_time": "Спрашивать каждый раз",
        "button_send_now": "Отправить сейчас",
        "button_edit": "Изменить",
        "button_delete": "Удалить",
        "button_confirm_delete": "Да, удалить",
        "button_change_schedule": "Изменить расписание",
        "button_disable_schedule": "Отключить расписание",
        "button_edit_request": "Изменить запрос",
        "button_change_language": "Изменить язык",
        "button_add_edit_sources": "Добавить источники",
        "button_confirm": "Подтвердить",
        "button_revise": "Уточнить",
        "button_cancel": "Отмена",
        "button_try_another_city": "Другой город",
        "button_yes_set_schedule": "Да, настроить расписание",
        "button_no_button_only": "Нет, только по кнопке",
        "button_yes_have_channels": "Да, у меня есть источники",
        "button_no_find_sources": "Нет, пусть бот найдёт источники",
        "button_only_channels": "Только эти источники",
        "button_add_sources": "Добавить ещё",
        "button_yes_show_recent": "Да, показать пропущенное",
        "button_no_future_only": "Нет, только будущие",
        "button_back": "Назад",
        "summary_not_set": "Ещё не настроено",
        "summary_ask": "Спрашивать для каждой новой подписки",
        "summary_fixed": "{language} для всех подписок",
    },
}


def t(ui_language: UILanguage, key: str, **kwargs: object) -> str:
    return _TEXTS[ui_language][key].format(**kwargs)


def interface_language_name(ui_language: UILanguage, target_language: str) -> str:
    normalized = normalize_language_code(target_language) or "en"
    if ui_language == "ru":
        return "Русский" if normalized == "ru" else "Английский"
    return "Russian" if normalized == "ru" else "English"


def subscription_preference_summary(
    ui_language: UILanguage,
    preference: LanguagePreference | None,
) -> str:
    if preference is None:
        return t(ui_language, "summary_not_set")
    if preference.mode == "ask":
        return t(ui_language, "summary_ask")
    return t(
        ui_language,
        "summary_fixed",
        language=interface_language_name(ui_language, preference.code or "en"),
    )
