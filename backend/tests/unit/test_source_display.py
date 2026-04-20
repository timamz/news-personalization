"""Tests for source_display_name function."""

import logging
import uuid
from types import SimpleNamespace

import pytest

from news_service.services.source_display import source_display_name

logging.disable(logging.CRITICAL)


def _make_source(url: str, title: str = "") -> SimpleNamespace:
    """Build a stub Source-like object."""
    return SimpleNamespace(url=url, title=title)


@pytest.mark.parametrize(
    ("url", "title", "expected_factory"),
    [
        (
            "https://t.me/s/{channel}",
            "",
            "@{channel}",
        ),
        (
            "https://t.me/s/durov",
            "Павел Дуров",
            "@durov",
        ),
        (
            "https://www.reddit.com/r/{sub}/new/",
            "",
            "r/{sub}",
        ),
        (
            "https://example.com/rss/feed.xml",
            "{title}",
            "{title}",
        ),
        (
            "https://example-{tag}.com/rss",
            "",
            "https://example-{tag}.com/rss",
        ),
        (
            "https://feed-{tag}.org/atom.xml",
            None,
            "https://feed-{tag}.org/atom.xml",
        ),
    ],
    ids=[
        "telegram_url_returns_at_channel",
        "telegram_url_with_cyrillic_title_returns_channel",
        "reddit_url_returns_r_subreddit",
        "generic_rss_with_title_returns_title",
        "generic_rss_no_title_returns_url",
        "generic_rss_none_title_returns_url",
    ],
)
def test_source_display_name_returns_expected_format(
    url: str, title: str | None, expected_factory: str
) -> None:
    tag = uuid.uuid4().hex[:6]
    channel = f"testchannel{tag}"
    sub = f"testsub{tag}"
    title_val = f"Новости России {tag}"

    replacements = {
        "{channel}": channel,
        "{sub}": sub,
        "{title}": title_val,
        "{tag}": tag,
    }

    resolved_url = url
    resolved_title = title
    resolved_expected = expected_factory
    for placeholder, value in replacements.items():
        resolved_url = resolved_url.replace(placeholder, value)
        if resolved_title is not None:
            resolved_title = resolved_title.replace(placeholder, value)
        resolved_expected = resolved_expected.replace(placeholder, value)

    source = _make_source(resolved_url, title=resolved_title)
    result = source_display_name(source)  # type: ignore[arg-type]

    assert result == resolved_expected, (
        f"source_display_name returned {result!r}, expected {resolved_expected!r}"
    )
