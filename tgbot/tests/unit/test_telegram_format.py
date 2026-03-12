from tgbot.telegram_format import render_html_message


def test_render_html_message_keeps_only_url_for_source_line() -> None:
    rendered = render_html_message("Источник: Telegram @fondnauk — https://t.me/fondnauk/123")

    assert rendered == '<a href="https://t.me/fondnauk/123"><i>Источник</i></a>'


def test_render_html_message_autolinks_plain_url() -> None:
    rendered = render_html_message("https://example.com/post")

    assert rendered == '<a href="https://example.com/post"><i>Source</i></a>'


def test_render_html_message_rewrites_inline_url_label() -> None:
    rendered = render_html_message("Лекция. URL: https://t.me/fondnauk/8141")

    assert rendered == 'Лекция. <a href="https://t.me/fondnauk/8141"><i>Источник</i></a>'


def test_render_html_message_prefers_explicit_english_source_label_over_cyrillic_text() -> None:
    rendered = render_html_message(
        "Русское название статьи\nSource: https://example.com/post-1\nhttps://example.com/post-2"
    )

    assert '<a href="https://example.com/post-1"><i>Source</i></a>' in rendered
    assert rendered.endswith('<a href="https://example.com/post-2"><i>Source</i></a>')
