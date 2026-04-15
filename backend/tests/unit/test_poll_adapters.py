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
async def test_rss_adapter_normalizes_entry_with_title_and_summary():
    src = _make_source("https://example.com/feed.xml", "Exemple Flux")
    adapter = RssAdapter(src)

    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item>"
        "<title>Les nouvelles d'aujourd'hui</title>"
        "<link>https://example.com/article-42</link>"
        "<summary>Résumé de l'article</summary>"
        "<pubDate>Mon, 10 Mar 2026 08:00:00 +0000</pubDate>"
        "</item>"
        "</channel></rss>"
    )

    with patch(
        "news_service.tasks.poll_adapters._fetch_rss_feed_content",
        new_callable=AsyncMock,
        return_value=xml.encode(),
    ):
        posts = await adapter.fetch_posts()

    assert len(posts) == 1, "adapter should return exactly one post"
    assert posts[0].headline == "Les nouvelles d'aujourd'hui", "headline should come from <title>"
    assert "Résumé" in posts[0].text_to_embed, "embed text should contain the summary"
    assert posts[0].url == "https://example.com/article-42", "url should come from <link>"


@pytest.mark.asyncio
async def test_rss_adapter_uses_description_when_summary_absent():
    src = _make_source("https://example.com/rss")
    adapter = RssAdapter(src)

    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item>"
        "<title>Título</title>"
        "<link>https://example.com/post</link>"
        "<description>Descripción detallada del artículo</description>"
        "</item>"
        "</channel></rss>"
    )

    with patch(
        "news_service.tasks.poll_adapters._fetch_rss_feed_content",
        new_callable=AsyncMock,
        return_value=xml.encode(),
    ):
        posts = await adapter.fetch_posts()

    assert len(posts) == 1, "adapter should return one post"
    assert "Descripción" in posts[0].body, "body should fall back to <description>"


@pytest.mark.asyncio
async def test_rss_adapter_skips_entries_without_link():
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

    assert len(posts) == 1, "adapter should skip entries without a <link>"
    assert posts[0].url == "https://example.com/ok", "only entry with link should be returned"


@pytest.mark.asyncio
async def test_telegram_adapter_extracts_headline_from_first_body_line():
    src = _make_source("https://t.me/s/testchannel", "ТестКанал")

    tg_post = MagicMock()
    tg_post.url = "https://t.me/testchannel/42"
    tg_post.body = "Первая строка заголовка\nВторая строка тела"
    tg_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    with patch(
        "news_service.tasks.poll_adapters.fetch_telegram_posts",
        new_callable=AsyncMock,
        return_value=[tg_post],
    ):
        adapter = TelegramAdapter(src, "testchannel")
        posts = await adapter.fetch_posts()

    assert len(posts) == 1, "adapter should return one post"
    assert posts[0].headline == "Первая строка заголовка", (
        "headline should be the first line of body"
    )
    assert posts[0].text_to_embed == tg_post.body, "embed text should be the full body"


@pytest.mark.asyncio
async def test_reddit_adapter_joins_title_and_body_for_embed_text():
    src = _make_source("https://www.reddit.com/r/neuralnet")

    reddit_post = MagicMock()
    reddit_post.url = "https://reddit.com/r/neuralnet/post/abc"
    reddit_post.title = "Новая модель ИИ"
    reddit_post.body = "Подробности о модели в статье"
    reddit_post.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    with patch(
        "news_service.tasks.poll_adapters.fetch_reddit_posts",
        new_callable=AsyncMock,
        return_value=[reddit_post],
    ):
        adapter = RedditAdapter(src, "neuralnet")
        posts = await adapter.fetch_posts()

    assert len(posts) == 1, "adapter should return one post"
    assert "Новая модель ИИ" in posts[0].text_to_embed, "embed text should include title"
    assert "Подробности" in posts[0].text_to_embed, "embed text should include body"
    assert posts[0].headline == "Новая модель ИИ", "headline should come from title field"


@pytest.mark.asyncio
async def test_twitter_adapter_extracts_headline_from_first_body_line():
    src = _make_source("https://x.com/testaccount")

    tweet = MagicMock()
    tweet.url = "https://x.com/testaccount/status/12345"
    tweet.body = "Заголовок твита\nОстальной текст"
    tweet.published_at = datetime(2026, 3, 10, tzinfo=UTC)

    with patch(
        "news_service.tasks.poll_adapters.fetch_twitter_posts",
        new_callable=AsyncMock,
        return_value=[tweet],
    ):
        adapter = TwitterAdapter(src, "testaccount")
        posts = await adapter.fetch_posts()

    assert len(posts) == 1, "adapter should return one post"
    assert posts[0].headline == "Заголовок твита", "headline should be the first line of body"
    assert posts[0].text_to_embed == tweet.body, "embed text should be the full body"


def test_rss_adapter_source_name_uses_title_when_available():
    src = _make_source("https://example.com/rss", "Mon Flux RSS")
    adapter = RssAdapter(src)
    assert adapter.source_name() == "Mon Flux RSS", "source name should use title"


def test_rss_adapter_source_name_falls_back_to_url():
    src = _make_source("https://example.com/rss", "")
    adapter = RssAdapter(src)
    assert adapter.source_name() == "https://example.com/rss", (
        "source name should fall back to url when title is empty"
    )
