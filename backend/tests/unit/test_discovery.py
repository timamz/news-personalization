import uuid

import pytest

from news_service.agents.discovery import normalize_source_url


@pytest.mark.parametrize(
    ("url", "kind", "expected_contains"),
    [
        (f"  https://{uuid.uuid4().hex[:8]}.com/feed  ", "rss", "feed"),
        ("https://t.me/s/testchannel", "telegram_channel", "testchannel"),
        ("https://www.reddit.com/r/machinelearning", "reddit_subreddit", "machinelearning"),
    ],
    ids=["rss_strips_whitespace", "telegram_normalizes", "reddit_normalizes"],
)
def test_normalize_source_url_preserves_content_for_valid_urls(
    url: str, kind: str, expected_contains: str
) -> None:
    result = normalize_source_url(url, source_kind=kind)
    assert result is not None and expected_contains in result


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("  ", "rss"),
        ("https://t.me/s/somechannel", "rss"),
    ],
    ids=["empty_rss", "telegram_passed_as_rss"],
)
def test_normalize_source_url_returns_none_for_invalid_or_mismatched_input(
    url: str, kind: str
) -> None:
    assert normalize_source_url(url, source_kind=kind) is None
