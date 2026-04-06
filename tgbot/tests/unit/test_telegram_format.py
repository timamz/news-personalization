import logging

from tgbot.telegram_format import render_html_message

logging.disable(logging.CRITICAL)

_SRC_LINK = '<a href="https://t.me/fondnauk/123">'
_SRC_LABEL = "<i>\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a</i></a>"


def test_render_html_message_converts_source_line_to_link() -> None:
    rendered = render_html_message(
        "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: Telegram @fondnauk"
        " \u2014 https://t.me/fondnauk/123"
    )

    assert rendered == f"{_SRC_LINK}{_SRC_LABEL}", "render_html_message did not convert source line"


def test_render_html_message_autolinks_plain_url() -> None:
    rendered = render_html_message("https://example.com/post")

    assert rendered == ('<a href="https://example.com/post"><i>Source</i></a>'), (
        "render_html_message did not autolink a standalone URL"
    )


def test_render_html_message_rewrites_inline_url_with_cyrillic() -> None:
    rendered = render_html_message(
        "\u041b\u0435\u043a\u0446\u0438\u044f. URL: https://t.me/fondnauk/8141"
    )

    expected = (
        "\u041b\u0435\u043a\u0446\u0438\u044f."
        ' <a href="https://t.me/fondnauk/8141">'
        "<i>\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a</i></a>"
    )
    assert rendered == expected, "render_html_message did not rewrite inline URL"


def test_render_html_message_labels_both_urls_as_source_links() -> None:
    text = (
        "\u0421\u0442\u0430\u0442\u044c\u044f\n"
        "Source: https://example.com/post-1\n"
        "https://example.com/post-2"
    )
    rendered = render_html_message(text)

    assert '<a href="https://example.com/post-1"><i>Source</i></a>' in rendered, (
        "render_html_message did not use Source label for first URL"
    )
    expected_end = '<a href="https://example.com/post-2"><i>Source</i></a>'
    assert rendered.endswith(expected_end), "render_html_message did not label second URL"
