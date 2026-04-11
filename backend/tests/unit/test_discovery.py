import logging
import uuid

from news_service.agents.discovery import normalize_source_url

logging.disable(logging.CRITICAL)


def test_normalize_source_url_strips_whitespace_for_rss() -> None:
    url = f"  https://{uuid.uuid4().hex[:8]}.com/feed  "
    result = normalize_source_url(url, source_kind="rss")
    assert result == url.strip(), "normalize_source_url did not strip whitespace from RSS URL"


def test_normalize_source_url_returns_none_for_empty_rss() -> None:
    result = normalize_source_url("  ", source_kind="rss")
    assert result is None, "normalize_source_url did not return None for empty RSS URL"


def test_normalize_source_url_returns_none_for_telegram_url_as_rss() -> None:
    result = normalize_source_url("https://t.me/s/somechannel", source_kind="rss")
    assert result is None, "normalize_source_url did not return None for Telegram URL passed as RSS"


def test_normalize_source_url_normalizes_telegram_channel() -> None:
    result = normalize_source_url("https://t.me/s/testchannel", source_kind="telegram_channel")
    assert result is not None, "normalize_source_url returned None for valid Telegram channel URL"
    assert "testchannel" in result, "normalize_source_url did not preserve channel name"


def test_normalize_source_url_normalizes_reddit_subreddit() -> None:
    result = normalize_source_url(
        "https://www.reddit.com/r/machinelearning", source_kind="reddit_subreddit"
    )
    assert result is not None, "normalize_source_url returned None for valid Reddit subreddit URL"
    assert "machinelearning" in result, "normalize_source_url did not preserve subreddit name"
