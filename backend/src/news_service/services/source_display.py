"""Display name extraction for sources."""

from news_service.models.source import Source
from news_service.services.reddit import extract_reddit_subreddit_from_url
from news_service.services.telegram import extract_telegram_channel_from_url
from news_service.services.twitter import extract_twitter_account_from_url


def source_display_name(source: Source) -> str:
    """Return a short user-friendly display name for a source."""
    channel = extract_telegram_channel_from_url(source.url)
    if channel is not None:
        return f"@{channel}"

    subreddit = extract_reddit_subreddit_from_url(source.url)
    if subreddit is not None:
        return f"r/{subreddit}"

    account = extract_twitter_account_from_url(source.url)
    if account is not None:
        return f"@{account}"

    return source.title or source.url
