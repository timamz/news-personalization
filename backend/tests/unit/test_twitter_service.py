import logging
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from news_service.services import twitter
from news_service.services.twitter import (
    TwitterPost,
    TwitterRateLimitError,
    build_twitter_account_url,
    extract_twitter_account_from_url,
    extract_twitter_accounts,
    normalize_twitter_account,
    parse_twitter_posts,
)

logging.disable(logging.CRITICAL)


def _make_twitter_timeline_html(screen_name: str, tweet_id: str, text: str, created_at: str) -> str:
    return f"""
    <html><body>
    <script id="__NEXT_DATA__" type="application/json">
    {{"props":{{"pageProps":{{"timeline":{{"entries":[
        {{"type":"tweet","content":{{"tweet":{{
            "id_str":"{tweet_id}",
            "full_text":"{text}",
            "created_at":"{created_at}",
            "user":{{"screen_name":"{screen_name}"}}
        }}}}}}
    ]}}}}}}}}
    </script>
    </body></html>
    """


def test_extract_twitter_accounts_deduplicates_urls_and_mentions() -> None:
    tag = uuid.uuid4().hex[:6]
    prompt = (
        f"Track https://x.com/OpenAI and @NASA on Twitter. tag={tag} "
        f"Ignore duplicate https://mobile.x.com/openai."
    )
    accounts = extract_twitter_accounts(prompt)
    assert accounts == ["openai", "nasa"], "extract did not deduplicate twitter accounts correctly"


def test_normalize_twitter_account_from_handle() -> None:
    result = normalize_twitter_account("@OpenAI")
    assert result == "openai", "normalize did not lowercase twitter handle"


def test_normalize_twitter_account_from_short_url() -> None:
    result = normalize_twitter_account("x.com/NASA")
    assert result == "nasa", "normalize did not extract account from short URL"


def test_normalize_twitter_account_from_full_url() -> None:
    result = normalize_twitter_account("https://twitter.com/NASA/status/123")
    assert result == "nasa", "normalize did not extract account from full twitter.com URL"


def test_extract_twitter_account_from_x_url() -> None:
    result = extract_twitter_account_from_url("https://x.com/OpenAI")
    assert result == "openai", "extract did not parse x.com URL"


def test_extract_twitter_account_from_mobile_twitter_url() -> None:
    result = extract_twitter_account_from_url("https://mobile.twitter.com/NASA/status/42")
    assert result == "nasa", "extract did not parse mobile.twitter.com URL"


def test_extract_twitter_account_from_reserved_path_returns_none() -> None:
    result = extract_twitter_account_from_url("https://x.com/home")
    assert result is None, "extract did not return None for reserved path"


def test_parse_twitter_posts_returns_single_post() -> None:
    html = _make_twitter_timeline_html(
        "NASA",
        "2032045283488473242",
        "ПРЯМОЙ ЭФИР: Идёт стыковка.",
        "Thu Mar 12 10:45:23 +0000 2026",
    )
    posts = parse_twitter_posts(html, "nasa", limit=20)
    assert len(posts) == 1, "parse did not return exactly one post"


def test_parse_twitter_posts_extracts_url() -> None:
    html = _make_twitter_timeline_html(
        "NASA",
        "2032045283488473242",
        "ПРЯМОЙ ЭФИР: Идёт стыковка.",
        "Thu Mar 12 10:45:23 +0000 2026",
    )
    posts = parse_twitter_posts(html, "nasa", limit=20)
    assert posts[0].url == "https://x.com/nasa/status/2032045283488473242", (
        "parse did not extract correct tweet URL"
    )


def test_parse_twitter_posts_extracts_title() -> None:
    html = _make_twitter_timeline_html(
        "NASA",
        "2032045283488473242",
        "ПРЯМОЙ ЭФИР: Идёт стыковка.",
        "Thu Mar 12 10:45:23 +0000 2026",
    )
    posts = parse_twitter_posts(html, "nasa", limit=20)
    assert posts[0].title == "ПРЯМОЙ ЭФИР: Идёт стыковка.", (
        "parse did not extract correct tweet title"
    )


def test_parse_twitter_posts_extracts_published_at() -> None:
    html = _make_twitter_timeline_html(
        "NASA",
        "2032045283488473242",
        "ПРЯМОЙ ЭФИР: Идёт стыковка.",
        "Thu Mar 12 10:45:23 +0000 2026",
    )
    posts = parse_twitter_posts(html, "nasa", limit=20)
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC), (
        "parse did not extract correct published_at timestamp"
    )


def test_build_twitter_account_url_normalizes() -> None:
    result = build_twitter_account_url("@OpenAI")
    assert result == "https://x.com/openai", "build did not produce correct normalized twitter URL"


def test_extract_retry_after_seconds_uses_rate_limit_reset_header(mocker) -> None:
    mocker.patch.object(twitter.time, "time", return_value=100.0)
    response = httpx.Response(429, headers={"x-rate-limit-reset": "112"})
    delay = twitter._extract_retry_after_seconds(response)
    assert delay == pytest.approx(12.0), (
        "extract_retry_after_seconds did not compute correct delay from header"
    )


@pytest.mark.asyncio
async def test_fetch_twitter_posts_retries_after_rate_limit(mocker, monkeypatch) -> None:
    monkeypatch.setattr(twitter.settings, "twitter_fetch_attempts", 2)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_retry_backoff_seconds", 2.0)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_max_rate_limit_wait_seconds", 5.0)

    post = TwitterPost(
        url=f"https://x.com/openai/status/{uuid.uuid4().int}",
        title="Привет мир",
        body="Привет мир",
        published_at=datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC),
    )
    mocker.patch.object(
        twitter,
        "_request_twitter_timeline_html",
        new=AsyncMock(side_effect=[TwitterRateLimitError(120.0), "<html></html>"]),
    )
    mocker.patch.object(twitter, "parse_twitter_posts", return_value=[post])
    mocker.patch.object(twitter.asyncio, "sleep", new=AsyncMock())

    posts = await twitter.fetch_twitter_posts("OpenAI", timeout_seconds=1.0)

    assert posts == [post], "fetch_twitter_posts did not return posts after rate limit retry"


@pytest.mark.asyncio
async def test_fetch_twitter_posts_retries_exactly_once_after_rate_limit(
    mocker, monkeypatch
) -> None:
    monkeypatch.setattr(twitter.settings, "twitter_fetch_attempts", 2)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_retry_backoff_seconds", 2.0)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_max_rate_limit_wait_seconds", 5.0)

    post = TwitterPost(
        url=f"https://x.com/openai/status/{uuid.uuid4().int}",
        title="Привет",
        body="Привет",
        published_at=datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC),
    )
    request_html = mocker.patch.object(
        twitter,
        "_request_twitter_timeline_html",
        new=AsyncMock(side_effect=[TwitterRateLimitError(120.0), "<html></html>"]),
    )
    mocker.patch.object(twitter, "parse_twitter_posts", return_value=[post])
    mocker.patch.object(twitter.asyncio, "sleep", new=AsyncMock())

    await twitter.fetch_twitter_posts("OpenAI", timeout_seconds=1.0)

    assert request_html.await_count == 2, (
        "fetch_twitter_posts did not retry exactly once after rate limit"
    )
