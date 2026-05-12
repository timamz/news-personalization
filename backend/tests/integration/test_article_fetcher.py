"""Integration test for the real article-body fetcher against a local HTTP server.

Digest quality depends entirely on ``fetch_article_text`` returning clean
main-content body for arbitrary web pages. Every polled RSS entry is
enriched with the full linked article at ingest time via this function,
and the extracted text is what the Digest Writer composes digests from.
A silent regression in the BeautifulSoup cleanup (adding/removing a
stripped tag, or breaking the HTTP path entirely) would degrade every
digest shipped by the system.

This test serves three realistic HTML fixtures -- a clean article, a
noisy page wrapped in nav/header/footer junk, and a page with no article
content -- off an ``aiohttp`` server bound to an ephemeral port, then
calls the production fetcher against each. Only the fast ``httpx`` path
is exercised; ``use_browser_fallback`` is left at its default ``False``
so Selenium is never launched (a real Firefox install is not available
in the test environment).
"""

import uuid

import pytest
from aiohttp import web

from news_service.services.article_fetch import fetch_article_text


@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_article_text_returns_clean_body_for_a_plain_article() -> None:
    run_id = uuid.uuid4().hex[:8]
    cyrillic_paragraph = (
        f"Научная группа {run_id} обнаружила необычный сигнал "
        "в данных с детектора нейтрино, установленного под озером Байкал."
    )
    second_paragraph = (
        f"Повторное измерение подтвердило устойчивость эффекта в течение "
        f"семи суток наблюдений (ref {run_id})."
    )
    page_title = f"Байкальский сигнал {run_id}"
    clean_html = (
        '<!DOCTYPE html><html lang="ru"><head>'
        f"<title>{page_title}</title>"
        '<meta charset="utf-8">'
        "</head><body>"
        "<article>"
        f"<h1>{page_title}</h1>"
        f"<p>{cyrillic_paragraph}</p>"
        f"<p>{second_paragraph}</p>"
        "</article>"
        "</body></html>"
    )

    async def clean_handler(_request: web.Request) -> web.Response:
        return web.Response(
            body=clean_html.encode("utf-8"),
            content_type="text/html",
            charset="utf-8",
        )

    app = web.Application()
    app.router.add_get("/clean", clean_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    try:
        bound_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        url = f"http://127.0.0.1:{bound_port}/clean"

        extracted = await fetch_article_text(
            url,
            timeout_seconds=5.0,
            max_chars=10_000,
        )

        assert extracted is not None, (
            "fetcher returned None for a well-formed article page that should parse cleanly"
        )
        assert cyrillic_paragraph in extracted, (
            "fetcher dropped the Cyrillic body paragraph from the extracted article text; "
            "BS4 extraction may be broken for non-ASCII content"
        )
        assert second_paragraph in extracted, (
            "fetcher dropped the second article paragraph; "
            "get_text may be truncating or skipping siblings inside <article>"
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_article_text_strips_navigation_header_and_footer_chrome() -> None:
    run_id = uuid.uuid4().hex[:8]
    nav_marker = f"NAV_JUNK_{run_id}"
    header_marker = f"HEADER_JUNK_{run_id}"
    footer_marker = f"FOOTER_JUNK_{run_id}"
    script_marker = f"SCRIPT_JUNK_{run_id}"
    style_marker = f"STYLE_JUNK_{run_id}"
    article_body = (
        f"Главная суть статьи {run_id}: независимое расследование выявило "
        "нарушения в логистической цепочке трёх крупных поставщиков."
    )
    noisy_html = (
        "<!DOCTYPE html><html><head>"
        f"<title>Noisy page {run_id}</title>"
        f"<style>.hidden {{ color: red; }} /* {style_marker} */</style>"
        f"<script>var tracker = '{script_marker}';</script>"
        "</head><body>"
        f'<header><div class="logo">{header_marker}</div></header>'
        f'<nav><ul><li><a href="/x">{nav_marker}</a></li></ul></nav>'
        "<article>"
        f"<h1>Заголовок {run_id}</h1>"
        f"<p>{article_body}</p>"
        "</article>"
        f"<footer><small>{footer_marker}</small></footer>"
        "</body></html>"
    )

    async def noisy_handler(_request: web.Request) -> web.Response:
        return web.Response(
            body=noisy_html.encode("utf-8"),
            content_type="text/html",
            charset="utf-8",
        )

    app = web.Application()
    app.router.add_get("/noisy", noisy_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    try:
        bound_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        url = f"http://127.0.0.1:{bound_port}/noisy"

        extracted = await fetch_article_text(
            url,
            timeout_seconds=5.0,
            max_chars=10_000,
        )

        assert extracted is not None, (
            "fetcher returned None for a noisy-but-non-empty page; "
            "extraction must not fail just because chrome wraps the article"
        )
        assert article_body in extracted, (
            "fetcher removed the article body while stripping chrome; "
            "the <article> contents should always survive extraction"
        )
        assert nav_marker not in extracted, (
            "fetcher leaked <nav> contents into the article body; "
            "navigation chrome must be stripped before get_text"
        )
        assert header_marker not in extracted, (
            "fetcher leaked <header> contents into the article body; "
            "site headers must be stripped before get_text"
        )
        assert footer_marker not in extracted, (
            "fetcher leaked <footer> contents into the article body; "
            "site footers must be stripped before get_text"
        )
        assert script_marker not in extracted, (
            "fetcher leaked <script> contents into the article body; "
            "inline JavaScript must never appear in extracted text"
        )
        assert style_marker not in extracted, (
            "fetcher leaked <style> contents into the article body; "
            "inline CSS must never appear in extracted text"
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_article_text_returns_none_for_a_page_with_no_readable_content() -> None:
    run_id = uuid.uuid4().hex[:8]
    empty_html = (
        "<!DOCTYPE html><html><head>"
        f"<title></title>"
        f"<script>var beacon = '{run_id}';</script>"
        "<style>body {} </style>"
        "</head><body>"
        "<nav></nav>"
        "<header></header>"
        "<footer></footer>"
        "</body></html>"
    )

    async def empty_handler(_request: web.Request) -> web.Response:
        return web.Response(
            body=empty_html.encode("utf-8"),
            content_type="text/html",
            charset="utf-8",
        )

    app = web.Application()
    app.router.add_get("/empty", empty_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    try:
        bound_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        url = f"http://127.0.0.1:{bound_port}/empty"

        extracted = await fetch_article_text(
            url,
            timeout_seconds=5.0,
            max_chars=10_000,
        )

        assert not extracted, (
            "fetcher returned non-empty text for a page containing only chrome and scripts; "
            f"nothing readable should survive stripping -- got: {extracted!r}"
        )
    finally:
        await runner.cleanup()
