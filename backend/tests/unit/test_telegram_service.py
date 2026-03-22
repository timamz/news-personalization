import logging
import uuid
from datetime import datetime

from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    extract_telegram_channels,
    parse_telegram_posts,
)

logging.disable(logging.CRITICAL)


def _make_telegram_post_html(
    channel: str, post_id: str, text: str, dt_iso: str, data_post: str | None = None
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


def test_extract_telegram_channels_deduplicates_and_ignores_email() -> None:
    tag = uuid.uuid4().hex[:6]
    prompt = (
        f"Нужны обновления от @fondnauk и @FONDNAUK. tag={tag} "
        f"Не парсить email admin@fondnauk.ru как канал."
    )
    channels = extract_telegram_channels(prompt)
    assert channels == ["fondnauk"], (
        "extract did not deduplicate channels or wrongly included email"
    )


def test_extract_telegram_channel_from_url_with_s_prefix() -> None:
    result = extract_telegram_channel_from_url("https://t.me/s/fondnauk")
    assert result == "fondnauk", "extract did not parse t.me/s/ URL correctly"


def test_extract_telegram_channel_from_url_without_s_prefix() -> None:
    result = extract_telegram_channel_from_url("https://t.me/fondnauk")
    assert result == "fondnauk", "extract did not parse t.me/ URL correctly"


def test_extract_telegram_channel_from_non_telegram_url_returns_none() -> None:
    result = extract_telegram_channel_from_url("https://example.com/s/fondnauk")
    assert result is None, "extract did not return None for non-telegram URL"


def test_parse_telegram_posts_returns_single_post() -> None:
    html = _make_telegram_post_html(
        "fondnauk", "123", "Строка 1<br/>Строка 2", "2026-02-27T10:00:00+00:00"
    )
    posts = parse_telegram_posts(html)
    assert len(posts) == 1, "parse did not return exactly one post"


def test_parse_telegram_posts_extracts_url() -> None:
    html = _make_telegram_post_html(
        "fondnauk", "123", "Строка 1<br/>Строка 2", "2026-02-27T10:00:00+00:00"
    )
    posts = parse_telegram_posts(html)
    assert posts[0].url == "https://t.me/fondnauk/123", "parse did not extract correct post URL"


def test_parse_telegram_posts_extracts_body_with_newlines() -> None:
    html = _make_telegram_post_html(
        "fondnauk", "123", "Строка 1<br/>Строка 2", "2026-02-27T10:00:00+00:00"
    )
    posts = parse_telegram_posts(html)
    assert posts[0].body == "Строка 1\nСтрока 2", "parse did not extract body with correct newlines"


def test_parse_telegram_posts_extracts_published_at() -> None:
    html = _make_telegram_post_html("fondnauk", "123", "Текст", "2026-02-27T10:00:00+00:00")
    posts = parse_telegram_posts(html)
    assert posts[0].published_at == datetime.fromisoformat("2026-02-27T10:00:00+00:00"), (
        "parse did not extract correct published_at datetime"
    )


def test_parse_telegram_posts_prefers_data_post_over_date_href() -> None:
    html = _make_telegram_post_html(
        "fondnauk",
        "8118",
        "Строка 1<br/>Строка 2",
        "2026-03-05T12:32:46+00:00",
        data_post="fondnauk/8119",
    )
    posts = parse_telegram_posts(html)
    assert posts[0].url == "https://t.me/fondnauk/8119", (
        "parse did not prefer data-post attribute over date href"
    )


def test_build_telegram_channel_url_normalizes() -> None:
    result = build_telegram_channel_url("@FondNauk")
    assert result == "https://t.me/s/fondnauk", (
        "build did not produce correct normalized channel URL"
    )
