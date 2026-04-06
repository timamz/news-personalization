import logging
import uuid
from datetime import UTC, datetime

from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddit_from_url,
    extract_reddit_subreddits,
    normalize_reddit_subreddit,
    parse_reddit_posts,
)

logging.disable(logging.CRITICAL)


def _make_prompt_with_duplicate_subreddit() -> str:
    tag = uuid.uuid4().hex[:6]
    return (
        f"Track r/Badminton and https://www.reddit.com/r/badminton/new/. "
        f"Also include /r/tennis. Tag={tag}"
    )


def _make_reddit_post_payload(
    title: str, selftext: str, permalink: str, created_utc: float
) -> dict:
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


def test_extract_reddit_subreddits_deduplicates_and_includes_all() -> None:
    prompt = _make_prompt_with_duplicate_subreddit()
    subreddits = extract_reddit_subreddits(prompt)

    assert "badminton" in subreddits, "extract did not include badminton subreddit"
    assert "tennis" in subreddits, "extract did not include tennis subreddit"
    assert len(subreddits) == 2, "extract did not deduplicate subreddit mentions"


def test_normalize_reddit_subreddit_from_r_prefix() -> None:
    result = normalize_reddit_subreddit("r/Badminton")
    assert result == "badminton", "normalize did not lowercase r/ prefix subreddit"


def test_normalize_reddit_subreddit_from_full_url() -> None:
    result = normalize_reddit_subreddit("https://www.reddit.com/r/tennis/comments/abc/title/")
    assert result == "tennis", "normalize did not extract subreddit from full URL"


def test_extract_reddit_subreddit_from_www_url() -> None:
    result = extract_reddit_subreddit_from_url("https://www.reddit.com/r/badminton/new/")
    assert result == "badminton", "extract did not parse www.reddit.com URL"


def test_extract_reddit_subreddit_from_old_url() -> None:
    result = extract_reddit_subreddit_from_url("https://old.reddit.com/r/tennis/comments/abc/")
    assert result == "tennis", "extract did not parse old.reddit.com URL"


def test_extract_reddit_subreddit_from_non_reddit_url_returns_none() -> None:
    result = extract_reddit_subreddit_from_url("https://example.com/r/badminton/")
    assert result is None, "extract did not return None for non-reddit URL"


def test_parse_reddit_posts_returns_single_post() -> None:
    payload = _make_reddit_post_payload(
        title="Обсуждение Swiss Open",
        selftext="Что вы думаете о финале?",
        permalink="/r/badminton/comments/abc123/swiss_open_discussion/",
        created_utc=1773312452.0,
    )
    posts = parse_reddit_posts(payload)
    assert len(posts) == 1, "parse did not return exactly one post"


def test_parse_reddit_posts_extracts_url() -> None:
    payload = _make_reddit_post_payload(
        title="Обсуждение Swiss Open",
        selftext="Что вы думаете о финале?",
        permalink="/r/badminton/comments/abc123/swiss_open_discussion/",
        created_utc=1773312452.0,
    )
    posts = parse_reddit_posts(payload)
    assert posts[0].url == (
        "https://www.reddit.com/r/badminton/comments/abc123/swiss_open_discussion/"
    ), "parse did not extract correct post URL"


def test_parse_reddit_posts_extracts_title() -> None:
    payload = _make_reddit_post_payload(
        title="Обсуждение Swiss Open",
        selftext="Что вы думаете о финале?",
        permalink="/r/badminton/comments/abc123/swiss_open_discussion/",
        created_utc=1773312452.0,
    )
    posts = parse_reddit_posts(payload)
    assert posts[0].title == "Обсуждение Swiss Open", "parse did not extract correct title"


def test_parse_reddit_posts_extracts_body() -> None:
    payload = _make_reddit_post_payload(
        title="Обсуждение Swiss Open",
        selftext="Что вы думаете о финале?",
        permalink="/r/badminton/comments/abc123/swiss_open_discussion/",
        created_utc=1773312452.0,
    )
    posts = parse_reddit_posts(payload)
    assert posts[0].body == "Что вы думаете о финале?", "parse did not extract correct body"


def test_parse_reddit_posts_extracts_published_at() -> None:
    payload = _make_reddit_post_payload(
        title="Обсуждение Swiss Open",
        selftext="Что вы думаете о финале?",
        permalink="/r/badminton/comments/abc123/swiss_open_discussion/",
        created_utc=1773312452.0,
    )
    posts = parse_reddit_posts(payload)
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 47, 32, tzinfo=UTC), (
        "parse did not extract correct published_at timestamp"
    )


def test_build_reddit_subreddit_url_normalizes() -> None:
    result = build_reddit_subreddit_url("/r/Badminton")
    assert result == "https://www.reddit.com/r/badminton/new/", (
        "build did not produce correct subreddit URL"
    )
