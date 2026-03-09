from datetime import datetime

from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    extract_telegram_channels,
    parse_telegram_posts,
)


def test_extract_telegram_channels_deduplicates_and_ignores_email() -> None:
    prompt = (
        "Need updates from @fondnauk and @FONDNAUK. "
        "Do not parse email admin@fondnauk.ru as a channel."
    )

    channels = extract_telegram_channels(prompt)

    assert channels == ["fondnauk"]


def test_extract_telegram_channel_from_url() -> None:
    assert extract_telegram_channel_from_url("https://t.me/s/fondnauk") == "fondnauk"
    assert extract_telegram_channel_from_url("https://t.me/fondnauk") == "fondnauk"
    assert extract_telegram_channel_from_url("https://example.com/s/fondnauk") is None


def test_parse_telegram_posts_extracts_text_url_and_datetime() -> None:
    html = """
    <div class="tgme_widget_message_wrap">
      <div class="tgme_widget_message">
        <div class="tgme_widget_message_bubble">
          <div class="tgme_widget_message_text">Line 1<br/>Line 2</div>
          <div class="tgme_widget_message_info">
            <a class="tgme_widget_message_date" href="https://t.me/fondnauk/123">
              <time datetime="2026-02-27T10:00:00+00:00"></time>
            </a>
          </div>
        </div>
      </div>
    </div>
    """

    posts = parse_telegram_posts(html)

    assert len(posts) == 1
    assert posts[0].url == "https://t.me/fondnauk/123"
    assert posts[0].body == "Line 1\nLine 2"
    assert posts[0].published_at == datetime.fromisoformat("2026-02-27T10:00:00+00:00")


def test_parse_telegram_posts_prefers_data_post_over_date_href() -> None:
    html = """
    <div class="tgme_widget_message_wrap">
      <div class="tgme_widget_message" data-post="fondnauk/8119">
        <div class="tgme_widget_message_bubble">
          <div class="tgme_widget_message_text">Line 1<br/>Line 2</div>
          <div class="tgme_widget_message_info">
            <a class="tgme_widget_message_date" href="https://t.me/fondnauk/8118">
              <time datetime="2026-03-05T12:32:46+00:00"></time>
            </a>
          </div>
        </div>
      </div>
    </div>
    """

    posts = parse_telegram_posts(html)

    assert len(posts) == 1
    assert posts[0].url == "https://t.me/fondnauk/8119"


def test_build_telegram_channel_url() -> None:
    assert build_telegram_channel_url("@FondNauk") == "https://t.me/s/fondnauk"
