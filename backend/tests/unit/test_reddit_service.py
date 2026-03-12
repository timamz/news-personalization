from datetime import UTC, datetime

from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddit_from_url,
    extract_reddit_subreddits,
    normalize_reddit_subreddit,
    parse_reddit_posts,
)


def test_extract_reddit_subreddits_deduplicates_mentions_and_urls() -> None:
    prompt = (
        "Track r/Badminton and https://www.reddit.com/r/badminton/new/. Also include /r/tennis."
    )

    subreddits = extract_reddit_subreddits(prompt)

    assert subreddits == ["badminton", "tennis"]


def test_normalize_reddit_subreddit_accepts_names_and_urls() -> None:
    assert normalize_reddit_subreddit("r/Badminton") == "badminton"
    assert normalize_reddit_subreddit("https://www.reddit.com/r/tennis/comments/abc/title/") == (
        "tennis"
    )


def test_extract_reddit_subreddit_from_url() -> None:
    assert extract_reddit_subreddit_from_url("https://www.reddit.com/r/badminton/new/") == (
        "badminton"
    )
    assert extract_reddit_subreddit_from_url("https://old.reddit.com/r/tennis/comments/abc/") == (
        "tennis"
    )
    assert extract_reddit_subreddit_from_url("https://example.com/r/badminton/") is None


def test_parse_reddit_posts_extracts_post_fields() -> None:
    payload = {
        "data": {
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "title": "Swiss Open discussion",
                        "selftext": "What did you think about the final?",
                        "permalink": "/r/badminton/comments/abc123/swiss_open_discussion/",
                        "created_utc": 1773312452.0,
                    },
                }
            ]
        }
    }

    posts = parse_reddit_posts(payload)

    assert len(posts) == 1
    assert posts[0].url == (
        "https://www.reddit.com/r/badminton/comments/abc123/swiss_open_discussion/"
    )
    assert posts[0].title == "Swiss Open discussion"
    assert posts[0].body == "What did you think about the final?"
    assert posts[0].published_at == datetime(2026, 3, 12, 10, 47, 32, tzinfo=UTC)


def test_build_reddit_subreddit_url() -> None:
    assert build_reddit_subreddit_url("/r/Badminton") == "https://www.reddit.com/r/badminton/new/"
