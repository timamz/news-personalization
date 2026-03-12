from tgbot.source_parser import (
    extract_reddit_subreddits,
    extract_telegram_channels,
    extract_twitter_accounts,
    parse_source_tokens,
    parse_telegram_channel_tokens,
    parse_twitter_account_tokens,
)


def test_extract_telegram_channels_from_mentions_and_urls() -> None:
    text = "Следи за @Gonzo_ML и https://t.me/s/fondnauk, дубли @gonzo_ml"
    result = extract_telegram_channels(text)
    assert result == ["gonzo_ml", "fondnauk"]


def test_parse_telegram_channel_tokens_supports_mixed_input() -> None:
    text = "@gonzo_ml, https://t.me/fondnauk invalid-token"
    result = parse_telegram_channel_tokens(text)
    assert result == ["gonzo_ml", "fondnauk"]


def test_extract_reddit_subreddits_from_mentions_and_urls() -> None:
    text = "Следи за r/Python и https://www.reddit.com/r/programming/new/, дубли /r/python"
    result = extract_reddit_subreddits(text)
    assert result == ["python", "programming"]


def test_parse_source_tokens_supports_telegram_and_reddit() -> None:
    text = "@gonzo_ml r/python https://t.me/fondnauk https://x.com/OpenAI"
    channels, subreddits, twitter_accounts = parse_source_tokens(text)
    assert channels == ["gonzo_ml", "fondnauk"]
    assert subreddits == ["python"]
    assert twitter_accounts == ["openai"]


def test_extract_twitter_accounts_from_urls() -> None:
    text = "Следи за https://x.com/OpenAI и https://twitter.com/NASA"
    result = extract_twitter_accounts(text)
    assert result == ["openai", "nasa"]


def test_parse_twitter_account_tokens_supports_mixed_input() -> None:
    text = "https://x.com/OpenAI NASA invalid-token-too-long-123456"
    result = parse_twitter_account_tokens(text)
    assert result == ["openai"]
