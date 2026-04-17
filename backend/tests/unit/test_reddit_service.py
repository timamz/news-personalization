import logging
import uuid
from datetime import UTC, datetime

import pytest

from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddit_from_url,
    extract_reddit_subreddits,
    normalize_reddit_subreddit,
    parse_reddit_posts,
)

logging.disable(logging.CRITICAL)


def _reddit_payload(title: str, selftext: str, permalink: str, created_utc: float) -> dict:
    return {
        "data": {
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "title": title,
                        "selftext": selftext,
                        "permalink": permalink,
                        "created_utc": created_utc,
                    },
                }
            ]
        }
    }


def test_extract_reddit_subreddits_deduplicates_across_url_and_mention_forms() -> None:
    tag = uuid.uuid4().hex[:6]
    prompt = (
        f"Track r/Badminton and https://www.reddit.com/r/badminton/new/. "
        f"Also include /r/tennis. Tag={tag}"
    )
    subreddits = extract_reddit_subreddits(prompt)
    assert set(subreddits) == {"badminton", "tennis"} and len(subreddits) == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("r/Badminton", "badminton"),
        ("https://www.reddit.com/r/tennis/comments/abc/title/", "tennis"),
    ],
)
def test_normalize_reddit_subreddit_handles_prefix_and_url_forms(value: str, expected: str) -> None:
    assert normalize_reddit_subreddit(value) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.reddit.com/r/badminton/new/", "badminton"),
        ("https://old.reddit.com/r/tennis/comments/abc/", "tennis"),
        ("https://example.com/r/badminton/", None),
    ],
    ids=["www", "old", "non_reddit_returns_none"],
)
def test_extract_reddit_subreddit_from_url(url: str, expected: str | None) -> None:
    assert extract_reddit_subreddit_from_url(url) == expected


def test_build_reddit_subreddit_url_normalizes_prefix() -> None:
    assert build_reddit_subreddit_url("/r/Badminton") == "https://www.reddit.com/r/badminton/new/"


def test_parse_reddit_posts_extracts_all_fields_from_json() -> None:
    title = "Swiss Open \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435"
    body = "\u0427\u0442\u043e \u0432\u044b \u0434\u0443\u043c\u0430\u0435\u0442\u0435?"
    permalink = "/r/badminton/comments/abc123/swiss_open/"
    payload = _reddit_payload(title, body, permalink, 1773312452.0)

    posts = parse_reddit_posts(payload)

    assert len(posts) == 1
    assert posts[0].title == title
    assert posts[0].body == body
    assert posts[0].url == f"https://www.reddit.com{permalink}"
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 47, 32, tzinfo=UTC)
