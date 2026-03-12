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


def test_extract_twitter_accounts_deduplicates_urls_and_contextual_mentions() -> None:
    prompt = (
        "Track https://x.com/OpenAI and @NASA on Twitter. "
        "Ignore duplicate https://mobile.x.com/openai."
    )

    accounts = extract_twitter_accounts(prompt)

    assert accounts == ["openai", "nasa"]


def test_normalize_twitter_account_accepts_handles_and_urls() -> None:
    assert normalize_twitter_account("@OpenAI") == "openai"
    assert normalize_twitter_account("x.com/NASA") == "nasa"
    assert normalize_twitter_account("https://twitter.com/NASA/status/123") == "nasa"


def test_extract_twitter_account_from_url() -> None:
    assert extract_twitter_account_from_url("https://x.com/OpenAI") == "openai"
    assert extract_twitter_account_from_url("https://mobile.twitter.com/NASA/status/42") == "nasa"
    assert extract_twitter_account_from_url("https://x.com/home") is None


def test_parse_twitter_posts_extracts_post_fields() -> None:
    payload = """
    <html><body>
    <script id="__NEXT_DATA__" type="application/json">
    {"props":{"pageProps":{"timeline":{"entries":[
        {"type":"tweet","content":{"tweet":{
            "id_str":"2032045283488473242",
            "full_text":"LIVE: Docking operations are underway.",
            "created_at":"Thu Mar 12 10:45:23 +0000 2026",
            "user":{"screen_name":"NASA"}
        }}}
    ]}}}}
    </script>
    </body></html>
    """

    posts = parse_twitter_posts(payload, "nasa", limit=20)

    assert len(posts) == 1
    assert posts[0].url == "https://x.com/nasa/status/2032045283488473242"
    assert posts[0].title == "LIVE: Docking operations are underway."
    assert posts[0].body == "LIVE: Docking operations are underway."
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC)


def test_build_twitter_account_url() -> None:
    assert build_twitter_account_url("@OpenAI") == "https://x.com/openai"


def test_extract_retry_after_seconds_uses_rate_limit_reset_header(mocker) -> None:
    mocker.patch.object(twitter.time, "time", return_value=100.0)
    response = httpx.Response(429, headers={"x-rate-limit-reset": "112"})

    delay = twitter._extract_retry_after_seconds(response)

    assert delay == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_fetch_twitter_posts_retries_after_rate_limit(mocker, monkeypatch) -> None:
    monkeypatch.setattr(twitter.settings, "twitter_fetch_attempts", 2)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_retry_backoff_seconds", 2.0)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_max_rate_limit_wait_seconds", 5.0)

    post = TwitterPost(
        url="https://x.com/openai/status/1",
        title="hello",
        body="hello",
        published_at=datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC),
    )
    request_html = mocker.patch.object(
        twitter,
        "_request_twitter_timeline_html",
        new=AsyncMock(side_effect=[TwitterRateLimitError(120.0), "<html></html>"]),
    )
    mocker.patch.object(twitter, "parse_twitter_posts", return_value=[post])
    sleep = mocker.patch.object(twitter.asyncio, "sleep", new=AsyncMock())

    posts = await twitter.fetch_twitter_posts("OpenAI", timeout_seconds=1.0)

    assert posts == [post]
    assert request_html.await_count == 2
    assert sleep.await_count == 1
    assert 0.0 < sleep.await_args.args[0] <= 5.0
