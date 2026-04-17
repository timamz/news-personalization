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


def _make_html(screen_name: str, tweet_id: str, text: str, created_at: str) -> str:
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
    assert extract_twitter_accounts(prompt) == ["openai", "nasa"]


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        ("@OpenAI", "openai"),
        ("x.com/NASA", "nasa"),
        ("https://twitter.com/NASA/status/123", "nasa"),
    ],
)
def test_normalize_twitter_account_extracts_and_lowercases(input_val: str, expected: str) -> None:
    assert normalize_twitter_account(input_val) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x.com/OpenAI", "openai"),
        ("https://mobile.twitter.com/NASA/status/42", "nasa"),
        ("https://x.com/home", None),
    ],
)
def test_extract_twitter_account_from_url(url: str, expected: str | None) -> None:
    assert extract_twitter_account_from_url(url) == expected


def test_build_twitter_account_url_normalizes_handle() -> None:
    assert build_twitter_account_url("@OpenAI") == "https://x.com/openai"


def test_parse_twitter_posts_extracts_all_fields_from_next_data() -> None:
    text = "\u041f\u0440\u044f\u043c\u043e\u0439 \u044d\u0444\u0438\u0440"
    html = _make_html("NASA", "2032045283488473242", text, "Thu Mar 12 10:45:23 +0000 2026")
    posts = parse_twitter_posts(html, "nasa", limit=20)
    assert len(posts) == 1
    assert posts[0].url == "https://x.com/nasa/status/2032045283488473242"
    assert posts[0].title == text
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC)


def test_extract_retry_after_seconds_uses_rate_limit_reset_header(mocker) -> None:
    mocker.patch.object(twitter.time, "time", return_value=100.0)
    response = httpx.Response(429, headers={"x-rate-limit-reset": "112"})
    assert twitter._extract_retry_after_seconds(response) == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_fetch_twitter_posts_retries_once_after_rate_limit(mocker, monkeypatch) -> None:
    monkeypatch.setattr(twitter.settings, "twitter_fetch_attempts", 2)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_retry_backoff_seconds", 2.0)
    monkeypatch.setattr(twitter.settings, "twitter_fetch_max_rate_limit_wait_seconds", 5.0)

    post = TwitterPost(
        url=f"https://x.com/openai/status/{uuid.uuid4().int}",
        title="hi",
        body="hi",
        published_at=datetime(2026, 3, 12, 10, 45, 23, tzinfo=UTC),
    )
    request_html = mocker.patch.object(
        twitter,
        "_request_twitter_timeline_html",
        new=AsyncMock(side_effect=[TwitterRateLimitError(120.0), "<html></html>"]),
    )
    mocker.patch.object(twitter, "parse_twitter_posts", return_value=[post])
    mocker.patch.object(twitter.asyncio, "sleep", new=AsyncMock())

    posts = await twitter.fetch_twitter_posts("OpenAI", timeout_seconds=1.0)

    assert posts == [post] and request_html.await_count == 2, (
        "fetch_twitter_posts did not retry exactly once after a rate-limit error"
    )
