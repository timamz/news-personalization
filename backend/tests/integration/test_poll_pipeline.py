"""End-to-end integration test for the RSS polling pipeline.

Replaces the mock-heavy unit tests that stub out the session and HTTP
layer. Here the polling task runs against a real aiohttp server on an
ephemeral port and writes real NewsItem rows into the Postgres database
backing the rest of the integration suite. Only the embedding call is
replaced with a deterministic stub because it would otherwise hit the
LiteLLM embedding provider.
"""

import uuid
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.db.session import engine
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.tasks import poll_feeds


@pytest.mark.asyncio(loop_scope="session")
async def test_poll_all_feeds_persists_real_rss_entries_as_news_items(mocker) -> None:
    """A real RSS feed served over HTTP lands as NewsItem rows with correct fields."""
    run_id = uuid.uuid4().hex
    english_headline = f"Neutrino burst {run_id[:6]}"
    cyrillic_headline = f"Нейтринный всплеск {run_id[6:12]} в Дубне"
    third_headline = f"Collider update {run_id[12:18]}"
    english_body = f"English body {uuid.uuid4().hex} describing the detection."
    cyrillic_body = f"Русское тело {uuid.uuid4().hex} описывает детекцию."
    third_body = f"Third body {uuid.uuid4().hex}."

    english_slug = f"en-{uuid.uuid4().hex}"
    cyrillic_slug = f"ru-{uuid.uuid4().hex}"
    third_slug = f"misc-{uuid.uuid4().hex}"

    recent_pubdate = format_datetime(datetime.now(UTC) - timedelta(hours=3))

    articles = {
        english_slug: (english_headline, english_body),
        cyrillic_slug: (cyrillic_headline, cyrillic_body),
        third_slug: (third_headline, third_body),
    }

    def _feed_xml(base_url: str) -> str:
        items_xml = "".join(
            f"<item>"
            f"<title>{title}</title>"
            f"<link>{base_url}/article/{slug}</link>"
            f"<description>RSS stub for {slug}</description>"
            f"<pubDate>{recent_pubdate}</pubDate>"
            f"</item>"
            for slug, (title, _body) in articles.items()
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel>'
            f"<title>Integration feed {run_id}</title>"
            f"<link>{base_url}</link>"
            f"<description>Integration fixture</description>"
            f"{items_xml}"
            "</channel></rss>"
        )

    state: dict[str, str] = {}

    async def feed_handler(_request: web.Request) -> web.Response:
        return web.Response(
            body=_feed_xml(state["base_url"]).encode("utf-8"),
            content_type="application/rss+xml",
            charset="utf-8",
        )

    async def article_handler(request: web.Request) -> web.Response:
        slug = request.match_info["slug"]
        if slug not in articles:
            return web.Response(status=404, text="not found")
        title, body = articles[slug]
        html = (
            f"<html><head><title>{title}</title></head>"
            f"<body><article><h1>{title}</h1><p>{body}</p></article></body></html>"
        )
        return web.Response(body=html.encode("utf-8"), content_type="text/html", charset="utf-8")

    app = web.Application()
    app.router.add_get("/feed.xml", feed_handler)
    app.router.add_get("/article/{slug}", article_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    try:
        await site.start()
        bound_port = site._server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{bound_port}"
        state["base_url"] = base_url
        feed_url = f"{base_url}/feed.xml"

        deterministic_embedding = [0.042] * 1536

        async def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
            return [deterministic_embedding for _ in texts]

        mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(side_effect=_fake_embed_texts))

        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as setup_session:
            source = Source(
                id=uuid.uuid4(),
                url=feed_url,
                title=f"Integration source {run_id}",
                source_description=f"integration fixture {run_id}",
                is_active=True,
                subscriber_count=0,
            )
            setup_session.add(source)
            await setup_session.commit()
            source_id = source.id

        result = await poll_feeds._poll_all_feeds()

        async with session_factory() as verify_session:
            rows_result = await verify_session.execute(
                select(NewsItem).where(NewsItem.source_id == source_id)
            )
            news_items = list(rows_result.scalars().all())
            by_url = {item.url: item for item in news_items}

            expected_english_url = f"{base_url}/article/{english_slug}"
            expected_cyrillic_url = f"{base_url}/article/{cyrillic_slug}"
            expected_third_url = f"{base_url}/article/{third_slug}"

            persisted_source = (
                await verify_session.execute(select(Source).where(Source.id == source_id))
            ).scalar_one()

        assert result["feeds_polled"] >= 1, (
            "poll_all_feeds did not register the inserted source in its feeds_polled tally"
        )
        assert result["new_items"] == len(articles), (
            "poll_all_feeds reported the wrong number of freshly inserted items "
            f"(expected {len(articles)}, got {result['new_items']})"
        )
        assert len(news_items) == len(articles), (
            "database did not receive one NewsItem per RSS entry "
            f"(expected {len(articles)}, found {len(news_items)})"
        )
        assert set(by_url) == {
            expected_english_url,
            expected_cyrillic_url,
            expected_third_url,
        }, "persisted NewsItem URLs do not match the URLs advertised by the served feed"

        cyrillic_item = by_url[expected_cyrillic_url]
        assert cyrillic_item.headline == cyrillic_headline, (
            "Cyrillic headline did not round-trip through polling into the NewsItem row"
        )
        assert cyrillic_body in cyrillic_item.body, (
            "Cyrillic article body was not extracted from the served HTML into NewsItem.body"
        )
        assert persisted_source.last_polled_at is not None, (
            "polling did not stamp last_polled_at on the Source row after a successful run"
        )
    finally:
        await runner.cleanup()
