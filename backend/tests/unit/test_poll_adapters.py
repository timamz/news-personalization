"""Tests for source adapter normalization logic."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.tasks.poll_adapters import (
    RedditAdapter,
    RssAdapter,
    TelegramAdapter,
    TwitterAdapter,
)


def _make_source(url: str, title: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), url=url, title=title)


@pytest.mark.asyncio
async def test_rss_adapter_extracts_title_link_and_summary_with_description_fallback() -> None:
    src = _make_source("https://example.com/feed.xml", "Feed")
    adapter = RssAdapter(src)

    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item>"
        "<title>Headline</title>"
        "<link>https://example.com/article-42</link>"
        "<summary>Summary content</summary>"
        "<pubDate>Mon, 10 Mar 2026 08:00:00 +0000</pubDate>"
        "</item>"
        "<item>"
        "<title>Description fallback</title>"
        "<link>https://example.com/post</link>"
        "<description>Only description</description>"
        "</item>"
        "</channel></rss>"
    )

    with patch(
        "news_service.tasks.poll_adapters._fetch_rss_feed_content",
        new_callable=AsyncMock,
        return_value=xml.encode(),
    ):
        posts = await adapter.fetch_posts()

    assert len(posts) == 2
    assert posts[0].headline == "Headline"
    assert posts[0].url == "https://example.com/article-42"
    assert "Summary content" in posts[0].text_to_embed
    assert "Only description" in posts[1].body, "body did not fall back to <description>"


@pytest.mark.asyncio
async def test_rss_adapter_skips_entries_without_link() -> None:
    src = _make_source("https://example.com/rss")
    adapter = RssAdapter(src)

    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>No link</title></item>"
        "<item><title>Has link</title><link>https://example.com/ok</link></item>"
        "</channel></rss>"
    )

    with patch(
        "news_service.tasks.poll_adapters._fetch_rss_feed_content",
        new_callable=AsyncMock,
        return_value=xml.encode(),
    ):
        posts = await adapter.fetch_posts()

    assert len(posts) == 1 and posts[0].url == "https://example.com/ok"


def test_rss_adapter_source_name_uses_title_or_url_fallback() -> None:
    titled = RssAdapter(_make_source("https://example.com/rss", "Feed Title"))
    untitled = RssAdapter(_make_source("https://example.com/rss", ""))
    assert titled.source_name() == "Feed Title"
    assert untitled.source_name() == "https://example.com/rss"


@pytest.mark.asyncio
async def test_telegram_and_twitter_adapters_take_headline_from_first_body_line() -> None:
    tg_src = _make_source("https://t.me/s/testchannel", "\u0422\u0435\u0441\u0442")
    tw_src = _make_source("https://x.com/testaccount")

    tg_post = MagicMock()
    tg_post.url = "https://t.me/testchannel/42"
    tg_post.body = "First line headline\nSecond line body"
    tg_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    tweet = MagicMock()
    tweet.url = "https://x.com/testaccount/status/12345"
    tweet.body = "Tweet headline\nRest of text"
    tweet.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    with (
        patch(
            "news_service.tasks.poll_adapters.fetch_telegram_posts",
            new_callable=AsyncMock,
            return_value=[tg_post],
        ),
        patch(
            "news_service.tasks.poll_adapters.fetch_twitter_posts",
            new_callable=AsyncMock,
            return_value=[tweet],
        ),
    ):
        tg_posts = await TelegramAdapter(tg_src, "testchannel").fetch_posts()
        tw_posts = await TwitterAdapter(tw_src, "testaccount").fetch_posts()

    assert (
        tg_posts[0].headline == "First line headline" and tg_posts[0].text_to_embed == tg_post.body
    )
    assert tw_posts[0].headline == "Tweet headline" and tw_posts[0].text_to_embed == tweet.body


@pytest.mark.asyncio
async def test_reddit_adapter_combines_title_and_body_into_embed_text() -> None:
    src = _make_source("https://www.reddit.com/r/neuralnet")

    reddit_post = MagicMock()
    reddit_post.url = "https://reddit.com/r/neuralnet/post/abc"
    reddit_post.title = "New ML model"
    reddit_post.body = "Details about the model"
    reddit_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    with patch(
        "news_service.tasks.poll_adapters.fetch_reddit_posts",
        new_callable=AsyncMock,
        return_value=[reddit_post],
    ):
        posts = await RedditAdapter(src, "neuralnet").fetch_posts()

    assert (
        posts[0].headline == "New ML model"
        and "New ML model" in posts[0].text_to_embed
        and "Details about the model" in posts[0].text_to_embed
    ), "reddit adapter did not combine title and body into the embed text"
