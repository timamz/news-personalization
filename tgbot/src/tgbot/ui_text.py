from tgbot.language import LanguagePreference, UILanguage, normalize_language_code

_TEXTS: dict[UILanguage, dict[str, str]] = {
    "en": {
        "welcome": (
            "Welcome to News Bot! 👋\n\n"
            "I deliver personalized news digests and event notifications "
            "right here in Telegram.\n\n"
            "Tap 📋 Menu below to get started."
        ),
        "registration_failed": "Registration failed. Please try again later.",
        "ui_language_initial": "🌐 Choose bot interface language.",
        "ui_language_current": (
            "Current bot interface language: {current_language}.\n\n"
            "Choose a new interface language."
        ),
        "ui_language_updated": "Bot interface language updated to {language}.",
        "subscription_language_initial": (
            "🗣 Choose how subscription output language should work.\n\n"
            "You can keep one language for all subscriptions or ask every time."
        ),
        "subscription_language_current": (
            "Current subscription language setting: {summary}.\n\n"
            "Choose a new default for digests and event notifications."
        ),
        "subscription_language_saved_ask": (
            "Saved. I will ask for the subscription language during each new "
            "subscription flow.\n"
            "Existing subscriptions keep their current language."
        ),
        "subscription_language_saved_fixed": (
            "Saved. New subscriptions will use {language}.\n"
            "Updated {updated} existing subscriptions."
        ),
        "subscription_language_saved_fixed_partial": (
            "Saved. New subscriptions will use {language}.\n"
            "Updated {updated} existing subscriptions, "
            "but {failed} updates failed."
        ),
        "subscription_language_saved_fixed_failed": (
            "Saved. New subscriptions will use {language}.\n"
            "I couldn't update existing subscriptions right now. "
            "Try again from ⚙️ Settings."
        ),
        "summary_not_set": "Not configured yet",
        "summary_ask": "Ask every time",
        "summary_fixed": "Always {language}",
        "timezone_initial": (
            "🕐 What city are you in?\n\n"
            "This lets me deliver digests at your local time.\n"
            "Examples: Berlin, Tbilisi, New York, or Берлин.\n"
            "You can also send a time zone like Europe/Berlin."
        ),
        "timezone_current": (
            "🕐 Current time zone: {timezone}.\n"
            "Local time there is {local_time}.\n\n"
            "Send a new city or time zone."
        ),
        "timezone_input_empty": "Send a city or time zone.",
        "timezone_lookup_failed": ("I couldn't look up that city right now. Please try again."),
        "timezone_not_found": (
            "I couldn't match that city. Try a clearer form like Berlin, Берлин, or Paris, France."
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
        "timezone_saved": ("Saved.\n{location}\n{timezone}\nLocal time: {local_time}"),
        "timezone_updated": (
            "Time zone updated.\n{location}\n{timezone}\nLocal time: {local_time}"
        ),
        "timezone_save_failed": ("I couldn't save the time zone. Please try again."),
        "timezone_retry": "Send another city or time zone.",
        "failed_process_request": ("Failed to process your request. Please try again."),
        "recent_events_expired": (
            "This preview expired. Please create a new subscription if needed."
        ),
        "recent_events_loading": ("Checking what you might have missed in the last 7 days..."),
        "recent_events_failed": "Couldn't load recent events right now.",
        "recent_events_empty": ("No matching events were found in the last 7 days."),
        "status_thinking": "Thinking...",
        "status_checking_source": "Checking {source}...",
        "status_analyzing": "Analyzing your request...",
        "status_registering_sources": "Registering sources...",
        "status_discovering_sources": "Discovering sources...",
        "status_searching_known_sources": "Searching known sources...",
        "status_searching_web": "Searching the web...",
        "status_validating_source": "Validating source...",
        "status_looking_for_events": "Looking for events...",
        "create_subscription_failed": ("Failed to create subscription. Please try again."),
        "subscription_created_digest": (
            "✅ Subscription created!\n\n"
            "📝 {prompt_summary}\n\n"
            "You'll receive digests right here in this chat."
        ),
        "subscription_created_event": (
            "✅ Subscription created!\n\n"
            "📝 {prompt_summary}\n\n"
            "You'll receive event notifications right here in this chat."
        ),
        "show_recent_events_prompt": (
            "Would you like to see what you might have missed in the last 7 days?"
        ),
        "subscription_prompt": (
            "✍️ Describe what news you want.\n\n"
            "Examples:\n"
            '- "I want AI and tech news every morning as a brief summary"\n'
            '- "Notify me when a new episode of Severance is announced"'
        ),
        # Menu
        "menu_title": "📋 Menu",
        "subscriptions_title": "📰 My Subscriptions",
        "subscriptions_empty_hint": "You don't have any subscriptions yet.",
        "subscription_detail": (
            "📰 {prompt_summary}\n\n{canonical_prompt}\n\n📋 {type}  •  🌐 {language}"
        ),
        "edit_prompt": (
            "Describe what you want to change — schedule, sources, format, topic, or anything else."
        ),
        "edit_subscription_updated": "✅ Subscription updated.",
        "edit_subscription_failed": "Failed to update subscription. Please try again.",
        "settings_title": "⚙️ Settings\n\nChoose a setting to change:",
        "help_text": (
            "❓ Help\n\n"
            "I deliver personalized news digests and event notifications.\n\n"
            "Use the 📋 Menu button to:\n"
            "📰 View and manage subscriptions\n"
            "➕ Create new subscriptions\n"
            "⚙️ Change settings\n\n"
            "Press 📋 Menu anytime to return."
        ),
        # Subscription management
        "failed_load_subscriptions": ("Failed to load subscriptions. Please try again."),
        "type_digest": "Digest",
        "type_event": "Event notifications",
        "edit_subscription_language_prompt": ("🌐 Choose the language for this subscription."),
        "subscription_language_updated": "Language updated to {language}.",
        "subscription_language_update_failed": ("Failed to update language. Try again."),
        "digest_queued": "📤 Digest queued.",
        "digest_queue_failed": "Failed to queue digest. Try again.",
        "confirm_delete_prompt": ("🗑 Are you sure you want to delete this subscription?"),
        "subscription_deleted": "🗑 Subscription deleted.",
        "subscription_delete_failed": "Failed to delete. Try again.",
        # Buttons
        "button_subscriptions": "📰 Subscriptions",
        "button_new_subscription": "➕ New",
        "button_settings": "⚙️ Settings",
        "button_help": "❓ Help",
        "button_interface_language": "🌐 Interface language",
        "button_sub_language_setting": "🗣 Subscription language",
        "button_timezone_setting": "🕐 Timezone",
        "button_create_one": "➕ Create one",
        "button_english": "🇬🇧 English",
        "button_russian": "🇷🇺 Russian",
        "button_ask_every_time": "🔄 Ask every time",
        "button_send_now": "📤 Send now",
        "button_edit": "✏️ Edit",
        "button_delete": "🗑 Delete",
        "button_confirm_delete": "✅ Yes, delete",
        "button_change_language": "🌐 Language",
        "button_cancel": "❌ Cancel",
        "button_try_another_city": "🏙 Another city",
        "button_yes_show_recent": "✅ Show missed",
        "button_no_future_only": "❌ Future only",
        "button_back": "◀️ Back",
    },
    "ru": {
        "welcome": (
            "Добро пожаловать в News Bot! 👋\n\n"
            "Я отправляю персональные дайджесты новостей и уведомления "
            "о событиях прямо в Telegram.\n\n"
            "Нажмите 📋 Меню внизу, чтобы начать."
        ),
        "registration_failed": ("Не удалось зарегистрировать пользователя. Попробуйте позже."),
        "ui_language_initial": "🌐 Выберите язык интерфейса бота.",
        "ui_language_current": (
            "Текущий язык интерфейса бота: {current_language}.\n\nВыберите новый язык интерфейса."
        ),
        "ui_language_updated": "Язык интерфейса бота изменён на {language}.",
        "subscription_language_initial": (
            "🗣 Выберите, как должен работать язык подписок.\n\n"
            "Можно использовать один язык для всех подписок "
            "или выбирать его каждый раз."
        ),
        "subscription_language_current": (
            "Текущая настройка языка подписок: {summary}.\n\n"
            "Выберите новый язык по умолчанию для дайджестов "
            "и уведомлений о событиях."
        ),
        "subscription_language_saved_ask": (
            "Сохранено. В каждом новом сценарии подписки "
            "я буду спрашивать язык отдельно.\n"
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
            "Попробуйте снова из ⚙️ Настроек."
        ),
        "summary_not_set": "Ещё не настроено",
        "summary_ask": "Спрашивать каждый раз",
        "summary_fixed": "Всегда {language}",
        "timezone_initial": (
            "🕐 В каком вы городе?\n\n"
            "Это нужно, чтобы присылать дайджесты по вашему местному времени.\n"
            "Примеры: Берлин, Тбилиси, Нью-Йорк.\n"
            "Можно также отправить часовой пояс вроде Europe/Berlin."
        ),
        "timezone_current": (
            "🕐 Текущий часовой пояс: {timezone}.\n"
            "Локальное время там: {local_time}.\n\n"
            "Отправьте новый город или часовой пояс."
        ),
        "timezone_input_empty": "Отправьте город или часовой пояс.",
        "timezone_lookup_failed": ("Сейчас не удалось определить этот город. Попробуйте ещё раз."),
        "timezone_not_found": (
            "Не удалось сопоставить этот город. Попробуйте вариант вроде Берлин или Paris, France."
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
        "timezone_saved": ("Сохранено.\n{location}\n{timezone}\nЛокальное время: {local_time}"),
        "timezone_updated": (
            "Часовой пояс обновлён.\n{location}\n{timezone}\nЛокальное время: {local_time}"
        ),
        "timezone_save_failed": ("Не удалось сохранить часовой пояс. Попробуйте ещё раз."),
        "timezone_retry": "Отправьте другой город или часовой пояс.",
        "failed_process_request": ("Не удалось обработать запрос. Попробуйте ещё раз."),
        "recent_events_expired": (
            "Предпросмотр устарел. При необходимости создайте новую подписку."
        ),
        "recent_events_loading": ("Проверяю, что вы могли пропустить за последние 7 дней..."),
        "recent_events_failed": ("Сейчас не удалось загрузить недавние события."),
        "recent_events_empty": ("За последние 7 дней подходящих событий не найдено."),
        "status_thinking": "Думаю...",
        "status_checking_source": "Проверяю {source}...",
        "status_analyzing": "Анализирую запрос...",
        "status_registering_sources": "Регистрирую источники...",
        "status_discovering_sources": "Ищу источники...",
        "status_searching_known_sources": "Ищу среди известных источников...",
        "status_searching_web": "Ищу в интернете...",
        "status_validating_source": "Проверяю источник...",
        "status_looking_for_events": "Ищу события...",
        "create_subscription_failed": ("Не удалось создать подписку. Попробуйте ещё раз."),
        "subscription_created_digest": (
            "✅ Подписка создана!\n\n"
            "📝 {prompt_summary}\n\n"
            "Я буду присылать дайджесты прямо в этот чат."
        ),
        "subscription_created_event": (
            "✅ Подписка создана!\n\n"
            "📝 {prompt_summary}\n\n"
            "Я буду присылать уведомления о событиях прямо в этот чат."
        ),
        "show_recent_events_prompt": ("Показать, что вы могли пропустить за последние 7 дней?"),
        "subscription_prompt": (
            "✍️ Опишите, какие новости вам нужны.\n\n"
            "Примеры:\n"
            '- "Хочу новости про AI и технологии каждое утро '
            'в виде краткой сводки"\n'
            '- "Сообщай, когда анонсируют новый эпизод Severance"'
        ),
        # Menu
        "menu_title": "📋 Меню",
        "subscriptions_title": "📰 Мои подписки",
        "subscriptions_empty_hint": "У вас пока нет подписок.",
        "subscription_detail": (
            "📰 {prompt_summary}\n\n{canonical_prompt}\n\n📋 {type}  •  🌐 {language}"
        ),
        "edit_prompt": (
            "Опишите, что хотите изменить — расписание, источники, формат, тему или что-то другое."
        ),
        "edit_subscription_updated": "✅ Подписка обновлена.",
        "edit_subscription_failed": "Не удалось обновить подписку. Попробуйте ещё раз.",
        "settings_title": "⚙️ Настройки\n\nВыберите, что изменить:",
        "help_text": (
            "❓ Помощь\n\n"
            "Я отправляю персональные дайджесты новостей "
            "и уведомления о событиях.\n\n"
            "Нажмите 📋 Меню, чтобы:\n"
            "📰 Просматривать и управлять подписками\n"
            "➕ Создавать новые подписки\n"
            "⚙️ Менять настройки\n\n"
            "Нажмите 📋 Меню, чтобы вернуться."
        ),
        # Subscription management
        "failed_load_subscriptions": ("Не удалось загрузить подписки. Попробуйте ещё раз."),
        "type_digest": "Дайджест",
        "type_event": "Уведомления о событиях",
        "edit_subscription_language_prompt": ("🌐 Выберите язык для этой подписки."),
        "subscription_language_updated": ("Язык подписки изменён на {language}."),
        "subscription_language_update_failed": (
            "Не удалось обновить язык подписки. Попробуйте ещё раз."
        ),
        "digest_queued": "📤 Дайджест поставлен в очередь.",
        "digest_queue_failed": ("Не удалось поставить дайджест в очередь. Попробуйте ещё раз."),
        "confirm_delete_prompt": ("🗑 Вы уверены, что хотите удалить эту подписку?"),
        "subscription_deleted": "🗑 Подписка удалена.",
        "subscription_delete_failed": ("Не удалось удалить подписку. Попробуйте ещё раз."),
        # Buttons
        "button_subscriptions": "📰 Подписки",
        "button_new_subscription": "➕ Новая",
        "button_settings": "⚙️ Настройки",
        "button_help": "❓ Помощь",
        "button_interface_language": "🌐 Язык интерфейса",
        "button_sub_language_setting": "🗣 Язык подписок",
        "button_timezone_setting": "🕐 Часовой пояс",
        "button_create_one": "➕ Создать",
        "button_english": "🇬🇧 English",
        "button_russian": "🇷🇺 Русский",
        "button_ask_every_time": "🔄 Спрашивать каждый раз",
        "button_send_now": "📤 Отправить",
        "button_edit": "✏️ Изменить",
        "button_delete": "🗑 Удалить",
        "button_confirm_delete": "✅ Да, удалить",
        "button_change_language": "🌐 Язык",
        "button_cancel": "❌ Отмена",
        "button_try_another_city": "🏙 Другой город",
        "button_yes_show_recent": "✅ Показать",
        "button_no_future_only": "❌ Только будущие",
        "button_back": "◀️ Назад",
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
