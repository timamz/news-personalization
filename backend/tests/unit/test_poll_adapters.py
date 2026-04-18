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
async def test_rss_adapter_uses_fetched_article_body_over_feed_summary() -> None:
    src = _make_source("https://example.com/feed.xml", "Feed")
    adapter = RssAdapter(src)

    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item>"
        "<title>Headline</title>"
        "<link>https://example.com/article-42</link>"
        "<summary>teaser</summary>"
        "<pubDate>Mon, 10 Mar 2026 08:00:00 +0000</pubDate>"
        "</item>"
        "</channel></rss>"
    )
    full_article = f"Full article body {uuid.uuid4().hex[:8]} spans multiple paragraphs."

    with (
        patch(
            "news_service.tasks.poll_adapters._fetch_rss_feed_content",
            new_callable=AsyncMock,
            return_value=xml.encode(),
        ),
        patch(
            "news_service.tasks.poll_adapters.fetch_article_text",
            new_callable=AsyncMock,
            return_value=full_article,
        ),
    ):
        posts = await adapter.fetch_posts()

    assert (
        len(posts) == 1
        and posts[0].body == full_article
        and full_article in posts[0].text_to_embed
        and "teaser" not in posts[0].body
    ), "RSS adapter must prefer the fetched article body over the feed-level summary"


@pytest.mark.asyncio
async def test_rss_adapter_falls_back_to_feed_summary_when_article_fetch_returns_none() -> None:
    src = _make_source("https://example.com/feed.xml", "Feed")
    adapter = RssAdapter(src)

    stub = f"Feed stub {uuid.uuid4().hex[:6]}"
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item>"
        "<title>H</title>"
        "<link>https://example.com/article</link>"
        f"<description>{stub}</description>"
        "</item>"
        "</channel></rss>"
    )

    with (
        patch(
            "news_service.tasks.poll_adapters._fetch_rss_feed_content",
            new_callable=AsyncMock,
            return_value=xml.encode(),
        ),
        patch(
            "news_service.tasks.poll_adapters.fetch_article_text",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        posts = await adapter.fetch_posts()

    assert posts[0].body == stub and stub in posts[0].text_to_embed, (
        "RSS adapter must fall back to the feed summary when the article cannot be fetched"
    )


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

    with (
        patch(
            "news_service.tasks.poll_adapters._fetch_rss_feed_content",
            new_callable=AsyncMock,
            return_value=xml.encode(),
        ),
        patch(
            "news_service.tasks.poll_adapters.fetch_article_text",
            new_callable=AsyncMock,
            return_value=None,
        ),
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
    reddit_post.external_url = None
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


@pytest.mark.asyncio
async def test_reddit_adapter_enriches_link_post_by_fetching_external_url() -> None:
    src = _make_source("https://www.reddit.com/r/technology")

    link_post = MagicMock()
    link_post.url = "https://reddit.com/r/technology/comments/abc/nvidia_h200"
    link_post.title = "Nvidia announces H200 chips"
    link_post.body = ""
    link_post.external_url = "https://nvidia.com/h200"
    link_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)
    article = f"Full H200 article {uuid.uuid4().hex[:8]} with benchmarks and pricing."

    with (
        patch(
            "news_service.tasks.poll_adapters.fetch_reddit_posts",
            new_callable=AsyncMock,
            return_value=[link_post],
        ),
        patch(
            "news_service.tasks.poll_adapters.fetch_article_text",
            new_callable=AsyncMock,
            return_value=article,
        ),
    ):
        posts = await RedditAdapter(src, "technology").fetch_posts()

    assert posts[0].body == article and article in posts[0].text_to_embed, (
        "link post must be enriched with the fetched external article body"
    )


@pytest.mark.asyncio
async def test_reddit_adapter_does_not_fetch_when_self_post_body_exists() -> None:
    src = _make_source("https://www.reddit.com/r/ml")
    self_post = MagicMock()
    self_post.url = "https://reddit.com/r/ml/post/xyz"
    self_post.title = "Self post"
    self_post.body = "I ran some experiments and here's what I found..."
    self_post.external_url = "https://ignored.example/"
    self_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    fetch_mock = AsyncMock(return_value="should-not-be-used")
    with (
        patch(
            "news_service.tasks.poll_adapters.fetch_reddit_posts",
            new_callable=AsyncMock,
            return_value=[self_post],
        ),
        patch("news_service.tasks.poll_adapters.fetch_article_text", new=fetch_mock),
    ):
        posts = await RedditAdapter(src, "ml").fetch_posts()

    assert posts[0].body == self_post.body and fetch_mock.await_count == 0, (
        "self-posts already have body text; adapter must not waste a fetch on them"
    )
