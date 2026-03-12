from tgbot.source_parser import (
    extract_reddit_subreddits,
    extract_telegram_channels,
    parse_source_tokens,
    parse_telegram_channel_tokens,
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
    text = "@gonzo_ml r/python https://t.me/fondnauk"
    channels, subreddits = parse_source_tokens(text)
    assert channels == ["gonzo_ml", "fondnauk"]
    assert subreddits == ["python"]
