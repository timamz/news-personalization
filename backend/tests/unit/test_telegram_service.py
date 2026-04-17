import logging
import uuid
from datetime import datetime

import pytest

from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    extract_telegram_channels,
    parse_telegram_posts,
)

logging.disable(logging.CRITICAL)


def _tg_html(
    channel: str,
    post_id: str,
    text: str,
    dt_iso: str,
    data_post: str | None = None,
) -> str:
    data_post_attr = f' data-post="{data_post}"' if data_post else ""
    return f"""
    <div class="tgme_widget_message_wrap">
      <div class="tgme_widget_message"{data_post_attr}>
        <div class="tgme_widget_message_bubble">
          <div class="tgme_widget_message_text">{text}</div>
          <div class="tgme_widget_message_info">
            <a class="tgme_widget_message_date" href="https://t.me/{channel}/{post_id}">
              <time datetime="{dt_iso}"></time>
            </a>
          </div>
        </div>
      </div>
    </div>
    """


def test_extract_telegram_channels_deduplicates_and_ignores_email_addresses() -> None:
    tag = uuid.uuid4().hex[:6]
    prompt = (
        f"Updates from @fondnauk and @FONDNAUK. tag={tag} "
        f"Do not parse admin@fondnauk.ru as a channel."
    )
    assert extract_telegram_channels(prompt) == ["fondnauk"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://t.me/s/fondnauk", "fondnauk"),
        ("https://t.me/fondnauk", "fondnauk"),
        ("https://example.com/s/fondnauk", None),
    ],
    ids=["with_s_prefix", "without_s_prefix", "non_telegram_returns_none"],
)
def test_extract_telegram_channel_from_url(url: str, expected: str | None) -> None:
    assert extract_telegram_channel_from_url(url) == expected


def test_build_telegram_channel_url_normalizes_handle() -> None:
    assert build_telegram_channel_url("@FondNauk") == "https://t.me/s/fondnauk"


def test_parse_telegram_posts_extracts_all_fields_and_expands_br_tags() -> None:
    html = _tg_html(
        "fondnauk",
        "123",
        "\u0421\u0442\u0440\u043e\u043a\u0430 1<br/>\u0421\u0442\u0440\u043e\u043a\u0430 2",
        "2026-02-27T10:00:00+00:00",
    )
    posts = parse_telegram_posts(html)
    assert len(posts) == 1
    assert posts[0].url == "https://t.me/fondnauk/123"
    assert (
        posts[0].body
        == "\u0421\u0442\u0440\u043e\u043a\u0430 1\n\u0421\u0442\u0440\u043e\u043a\u0430 2"
    )
    assert posts[0].published_at == datetime.fromisoformat("2026-02-27T10:00:00+00:00")


def test_parse_telegram_posts_prefers_data_post_attribute_over_date_href() -> None:
    html = _tg_html(
        "fondnauk",
        "8118",
        "body",
        "2026-03-05T12:32:46+00:00",
        data_post="fondnauk/8119",
    )
    posts = parse_telegram_posts(html)
    assert posts[0].url == "https://t.me/fondnauk/8119"
