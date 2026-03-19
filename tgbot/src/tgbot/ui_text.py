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
        "choose_subscription_request": "Please describe your subscription request.",
        "failed_process_request": ("Failed to process your request. Please try again."),
        "subscription_setup_expired": ("Subscription setup expired. Start again from the menu."),
        "describe_schedule": "Please describe the schedule.",
        "schedule_parse_failed": ("Couldn't parse this schedule. Please try another wording."),
        "sources_parse_failed": (
            "I couldn't parse sources. Send Telegram handles like @channel_one, "
            "links like t.me/channel or x.com/OpenAI, "
            "or Reddit names like r/python."
        ),
        "recent_events_future_only": (
            "Okay. I will only send new event notifications from now on."
        ),
        "recent_events_expired": (
            "This preview expired. Please create a new subscription if needed."
        ),
        "recent_events_loading": ("Checking what you might have missed in the last 7 days..."),
        "recent_events_failed": "Couldn't load recent events right now.",
        "recent_events_empty": ("No matching events were found in the last 7 days."),
        "recent_events_header": ("Here's what you might have missed in the last 7 days:"),
        "back_not_available": "Back is not available here.",
        "processing_request": "⏳ Processing your request...",
        "status_thinking": "Thinking...",
        "status_checking_source": "Checking {source}...",
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
        "subscription_language_choose": (
            "🌐 Choose the language for this subscription's digests and event notifications."
        ),
        "schedule_choice": (
            "📅 Do you want this digest to be delivered automatically "
            "on a schedule?\nYou can always use the Send now button."
        ),
        "schedule_input_prompt": (
            '📅 Describe the schedule in natural language.\nExample: "every weekday at 9:00"'
        ),
        "source_question_digest": (
            "📡 Do you already have specific sources for this digest "
            "(Telegram channels, Reddit subreddits, or X accounts)?"
        ),
        "source_question_event": (
            "📡 Do you already have specific sources for these notifications "
            "(Telegram channels, Reddit subreddits, or X accounts)?"
        ),
        "channels_input_prompt": (
            "📡 Send sources you want to include "
            "(for example: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "scope_prompt_found": ("I found these sources in your request:\n{channels}\n\n{question}"),
        "scope_prompt_manual": "Got it. {question}\n{channels}",
        "scope_question_digest": ("Should this digest be limited only to these sources?"),
        "scope_question_event": ("Should these notifications be limited only to these sources?"),
        "undo_recent_events_failed": ("Couldn't go back right now. Please try again."),
        "undo_recent_events_expired": "This step can no longer be undone.",
        # Menu
        "menu_title": "📋 Menu",
        "subscriptions_title": "📰 My Subscriptions",
        "subscriptions_empty_hint": "You don't have any subscriptions yet.",
        "subscription_detail": ("📰 {prompt_summary}\n\n📋 {type}  •  🌐 {language}"),
        "edit_menu_header": ("✏️ {prompt_summary}\n\nWhat do you want to change?"),
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
        "flow_cancelled": "Cancelled.",
        # Subscription management
        "no_subscriptions": "You have no active subscriptions.",
        "failed_load_subscriptions": ("Failed to load subscriptions. Please try again."),
        "subscription_card": ("📰 {prompt_summary}\n\n📋 {type}  •  🌐 {language}"),
        "type_digest": "Digest",
        "type_event": "Event notifications",
        "edit_menu_prompt": "What do you want to change?",
        "edit_subscription_language_prompt": ("🌐 Choose the language for this subscription."),
        "subscription_language_updated": "Language updated to {language}.",
        "subscription_language_update_failed": ("Failed to update language. Try again."),
        "digest_queued": "📤 Digest queued.",
        "digest_queue_failed": "Failed to queue digest. Try again.",
        "edit_schedule_prompt": (
            '📅 Describe the new schedule in natural language.\nExample: "every weekday at 9:00"'
        ),
        "edit_session_expired": ("Edit session expired. Open 📰 Subscriptions from the menu."),
        "schedule_updated": "📅 Schedule updated.",
        "schedule_update_failed": ("Failed to update schedule. Please try again."),
        "schedule_disabled": "🚫 Automatic schedule disabled.",
        "schedule_disable_failed": "Failed to update schedule. Try again.",
        "edit_request_prompt": (
            "📝 Describe what should change.\n\n"
            "Edit is best for small tweaks: format, language details, "
            "or adding a specific filter.\n"
            "For a completely different topic, delete this subscription "
            "and create a new one."
        ),
        "edit_request_empty": "Please describe what should change.",
        "edit_request_preview": (
            "Proposed update:\n\n"
            "📝 {prompt_summary}\n"
            "📋 {format_instructions}\n"
            "✏️ {change_summary}\n\n"
            "Sources, schedule, and language stay unchanged."
        ),
        "edit_request_failed": ("Failed to prepare the update. Please try again."),
        "edit_request_applied": ("✅ Subscription updated.\n\n📝 {prompt_summary}"),
        "edit_request_apply_failed": ("Failed to save the update. Please try again."),
        "edit_request_cancelled": "Update cancelled.",
        "edit_sources_prompt": (
            "📡 Send sources to add (for example: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "sources_added": "📡 Added {count} sources.",
        "sources_already_added": "Those sources are already included.",
        "sources_add_failed": "Failed to add sources. Please try again.",
        "confirm_delete_prompt": ("🗑 Are you sure you want to delete this subscription?"),
        "delete_cancelled": "Deletion cancelled.",
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
        "button_change_schedule": "📅 Schedule",
        "button_disable_schedule": "🚫 Disable schedule",
        "button_edit_request": "📝 Request",
        "button_change_language": "🌐 Language",
        "button_add_edit_sources": "📡 Sources",
        "button_confirm": "✅ Confirm",
        "button_revise": "🔄 Revise",
        "button_cancel": "❌ Cancel",
        "button_try_another_city": "🏙 Another city",
        "button_yes_set_schedule": "📅 Set schedule",
        "button_no_button_only": "❌ Manual only",
        "button_yes_have_channels": "✅ Yes, I have sources",
        "button_no_find_sources": "🔍 Find for me",
        "button_only_channels": "📌 Only these",
        "button_add_sources": "🔍 Discover more",
        "button_yes_show_recent": "✅ Show missed",
        "button_no_future_only": "❌ Future only",
        "button_back": "◀️ Back",
        "summary_not_set": "Not set yet",
        "summary_ask": "Ask for each new subscription",
        "summary_fixed": "{language} for all subscriptions",
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
        "choose_subscription_request": ("Опишите, какую подписку вы хотите."),
        "failed_process_request": ("Не удалось обработать запрос. Попробуйте ещё раз."),
        "subscription_setup_expired": ("Сценарий создания подписки истёк. Начните заново из меню."),
        "describe_schedule": "Опишите расписание.",
        "schedule_parse_failed": (
            "Не удалось распознать это расписание. Попробуйте другую формулировку."
        ),
        "sources_parse_failed": (
            "Не удалось распознать источники. "
            "Отправьте Telegram-хэндлы вроде @channel_one, "
            "ссылки вроде t.me/channel или x.com/OpenAI, "
            "или сабреддиты вроде r/python."
        ),
        "recent_events_future_only": (
            "Хорошо. Я буду отправлять только новые уведомления о событиях."
        ),
        "recent_events_expired": (
            "Предпросмотр устарел. При необходимости создайте новую подписку."
        ),
        "recent_events_loading": ("Проверяю, что вы могли пропустить за последние 7 дней..."),
        "recent_events_failed": ("Сейчас не удалось загрузить недавние события."),
        "recent_events_empty": ("За последние 7 дней подходящих событий не найдено."),
        "recent_events_header": ("Вот что вы могли пропустить за последние 7 дней:"),
        "back_not_available": "Здесь нельзя вернуться назад.",
        "processing_request": "⏳ Обрабатываю ваш запрос...",
        "status_thinking": "Думаю...",
        "status_checking_source": "Проверяю {source}...",
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
        "subscription_language_choose": (
            "🌐 Выберите язык дайджестов и уведомлений для этой подписки."
        ),
        "schedule_choice": (
            "📅 Хотите получать этот дайджест автоматически по расписанию?\n"
            "Кнопка отправки вручную всегда останется доступной."
        ),
        "schedule_input_prompt": (
            '📅 Опишите расписание естественным языком.\nПример: "каждый будний день в 9:00"'
        ),
        "source_question_digest": (
            "📡 У вас уже есть конкретные источники для этого дайджеста "
            "(Telegram-каналы, Reddit-сабреддиты или аккаунты X)?"
        ),
        "source_question_event": (
            "📡 У вас уже есть конкретные источники для этих уведомлений "
            "(Telegram-каналы, Reddit-сабреддиты или аккаунты X)?"
        ),
        "channels_input_prompt": (
            "📡 Отправьте источники, которые хотите включить "
            "(например: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "scope_prompt_found": (
            "Я нашёл в вашем запросе такие источники:\n{channels}\n\n{question}"
        ),
        "scope_prompt_manual": "Понял. {question}\n{channels}",
        "scope_question_digest": ("Ограничить этот дайджест только этими источниками?"),
        "scope_question_event": ("Ограничить эти уведомления только этими источниками?"),
        "undo_recent_events_failed": ("Сейчас не удалось вернуться назад. Попробуйте ещё раз."),
        "undo_recent_events_expired": ("На этот шаг уже нельзя вернуться."),
        # Menu
        "menu_title": "📋 Меню",
        "subscriptions_title": "📰 Мои подписки",
        "subscriptions_empty_hint": "У вас пока нет подписок.",
        "subscription_detail": ("📰 {prompt_summary}\n\n📋 {type}  •  🌐 {language}"),
        "edit_menu_header": ("✏️ {prompt_summary}\n\nЧто вы хотите изменить?"),
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
        "flow_cancelled": "Отменено.",
        # Subscription management
        "no_subscriptions": "У вас нет активных подписок.",
        "failed_load_subscriptions": ("Не удалось загрузить подписки. Попробуйте ещё раз."),
        "subscription_card": ("📰 {prompt_summary}\n\n📋 {type}  •  🌐 {language}"),
        "type_digest": "Дайджест",
        "type_event": "Уведомления о событиях",
        "edit_menu_prompt": "Что вы хотите изменить?",
        "edit_subscription_language_prompt": ("🌐 Выберите язык для этой подписки."),
        "subscription_language_updated": ("Язык подписки изменён на {language}."),
        "subscription_language_update_failed": (
            "Не удалось обновить язык подписки. Попробуйте ещё раз."
        ),
        "digest_queued": "📤 Дайджест поставлен в очередь.",
        "digest_queue_failed": ("Не удалось поставить дайджест в очередь. Попробуйте ещё раз."),
        "edit_schedule_prompt": (
            '📅 Опишите новое расписание естественным языком.\nПример: "каждый будний день в 9:00"'
        ),
        "edit_session_expired": ("Сессия редактирования истекла. Откройте 📰 Подписки из меню."),
        "schedule_updated": "📅 Расписание обновлено.",
        "schedule_update_failed": ("Не удалось обновить расписание. Попробуйте ещё раз."),
        "schedule_disabled": "🚫 Автоматическое расписание отключено.",
        "schedule_disable_failed": ("Не удалось обновить расписание. Попробуйте ещё раз."),
        "edit_request_prompt": (
            "📝 Опишите, что нужно изменить.\n\n"
            "Редактирование подходит для мелких правок: формат, детали языка, "
            "добавление фильтра.\n"
            "Если нужна совсем другая тема — удалите подписку и создайте новую."
        ),
        "edit_request_empty": ("Пожалуйста, опишите, что нужно изменить."),
        "edit_request_preview": (
            "Предлагаемое обновление:\n\n"
            "📝 {prompt_summary}\n"
            "📋 {format_instructions}\n"
            "✏️ {change_summary}\n\n"
            "Источники, расписание и язык не изменятся."
        ),
        "edit_request_failed": ("Не удалось подготовить обновление. Попробуйте ещё раз."),
        "edit_request_applied": ("✅ Подписка обновлена.\n\n📝 {prompt_summary}"),
        "edit_request_apply_failed": ("Не удалось сохранить обновление. Попробуйте ещё раз."),
        "edit_request_cancelled": "Обновление отменено.",
        "edit_sources_prompt": (
            "📡 Отправьте источники, которые хотите добавить "
            "(например: @channel_one r/python t.me/channel x.com/OpenAI)."
        ),
        "sources_added": "📡 Добавлено источников: {count}.",
        "sources_already_added": "Эти источники уже добавлены.",
        "sources_add_failed": ("Не удалось добавить источники. Попробуйте ещё раз."),
        "confirm_delete_prompt": ("🗑 Вы уверены, что хотите удалить эту подписку?"),
        "delete_cancelled": "Удаление отменено.",
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
        "button_change_schedule": "📅 Расписание",
        "button_disable_schedule": "🚫 Выключить расписание",
        "button_edit_request": "📝 Запрос",
        "button_change_language": "🌐 Язык",
        "button_add_edit_sources": "📡 Источники",
        "button_confirm": "✅ Подтвердить",
        "button_revise": "🔄 Уточнить",
        "button_cancel": "❌ Отмена",
        "button_try_another_city": "🏙 Другой город",
        "button_yes_set_schedule": "📅 Настроить расписание",
        "button_no_button_only": "❌ Только вручную",
        "button_yes_have_channels": "✅ Да, есть источники",
        "button_no_find_sources": "🔍 Найти за меня",
        "button_only_channels": "📌 Только эти",
        "button_add_sources": "🔍 Найти ещё",
        "button_yes_show_recent": "✅ Показать",
        "button_no_future_only": "❌ Только будущие",
        "button_back": "◀️ Назад",
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
