"""Tests for source_display_name function."""

import logging
import uuid
from types import SimpleNamespace

from news_service.services.source_display import source_display_name

logging.disable(logging.CRITICAL)


def _make_source(url: str, title: str = "") -> SimpleNamespace:
    """Build a stub Source-like object."""
    return SimpleNamespace(url=url, title=title)


class TestSourceDisplayName:
    def test_telegram_url_returns_at_channel_format(self) -> None:
        channel = f"testchannel{uuid.uuid4().hex[:6]}"
        source = _make_source(f"https://t.me/s/{channel}")
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == f"@{channel}", f"did not return @channel for telegram URL, got {result}"

    def test_telegram_url_with_cyrillic_title_returns_channel(self) -> None:
        source = _make_source(
            "https://t.me/s/durov",
            title="Павел Дуров",
        )
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == "@durov", f"did not return @durov for telegram URL, got {result}"

    def test_reddit_url_returns_r_subreddit_format(self) -> None:
        sub = f"testsub{uuid.uuid4().hex[:5]}"
        url = f"https://www.reddit.com/r/{sub}/new/"
        source = _make_source(url)
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == f"r/{sub}", f"did not return r/subreddit for reddit URL, got {result}"

    def test_twitter_x_url_returns_at_account_format(self) -> None:
        account = f"user{uuid.uuid4().hex[:6]}"
        source = _make_source(f"https://x.com/{account}")
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == f"@{account}", f"did not return @account for twitter URL, got {result}"

    def test_twitter_classic_url_returns_at_account(self) -> None:
        source = _make_source("https://twitter.com/elonmusk")
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == "@elonmusk", f"did not return @elonmusk, got {result}"

    def test_generic_rss_url_returns_source_title(self) -> None:
        tag = uuid.uuid4().hex[:4]
        title = f"Новости России {tag}"
        source = _make_source("https://example.com/rss/feed.xml", title=title)
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == title, f"did not return source title for generic RSS, got {result}"

    def test_generic_rss_url_with_no_title_returns_url(self) -> None:
        tag = uuid.uuid4().hex[:6]
        url = f"https://example-{tag}.com/rss"
        source = _make_source(url, title="")
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == url, f"did not return URL when title is empty, got {result}"

    def test_generic_rss_url_with_none_title_returns_url(self) -> None:
        tag = uuid.uuid4().hex[:6]
        url = f"https://feed-{tag}.org/atom.xml"
        source = _make_source(url, title=None)
        result = source_display_name(source)  # type: ignore[arg-type]
        assert result == url, f"did not return URL when title is None, got {result}"
