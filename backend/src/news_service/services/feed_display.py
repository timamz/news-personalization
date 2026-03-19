"""Display name extraction for feeds."""

from news_service.models.rss_feed import RssFeed
from news_service.services.reddit import extract_reddit_subreddit_from_url
from news_service.services.telegram import extract_telegram_channel_from_url
from news_service.services.twitter import extract_twitter_account_from_url


def feed_display_name(feed: RssFeed) -> str:
    """Return a short user-friendly display name for a feed."""
    channel = extract_telegram_channel_from_url(feed.url)
    if channel is not None:
        return f"@{channel}"

    subreddit = extract_reddit_subreddit_from_url(feed.url)
    if subreddit is not None:
        return f"r/{subreddit}"

    account = extract_twitter_account_from_url(feed.url)
    if account is not None:
        return f"@{account}"

    return feed.title or feed.url
