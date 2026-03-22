import logging

from tgbot.source_parser import (
    extract_reddit_subreddits,
    extract_telegram_channels,
    extract_twitter_accounts,
    parse_source_tokens,
    parse_telegram_channel_tokens,
    parse_twitter_account_tokens,
)

logging.disable(logging.CRITICAL)


def test_extract_telegram_channels_finds_mention_in_cyrillic_text() -> None:
    text = (
        "\u0421\u043b\u0435\u0434\u0438 \u0437\u0430 @Gonzo_ML"
        " \u0438 \u0434\u0440\u0443\u0433\u0438\u043c\u0438"
    )
    result = extract_telegram_channels(text)

    assert "gonzo_ml" in result, "extract_telegram_channels did not find @Gonzo_ML mention"


def test_extract_telegram_channels_finds_url_with_s_prefix() -> None:
    text = (
        "\u041a\u0430\u043d\u0430\u043b https://t.me/s/fondnauk"
        " \u0438\u043d\u0442\u0435\u0440\u0435\u0441\u043d\u044b\u0439"
    )
    result = extract_telegram_channels(text)

    assert "fondnauk" in result, "extract_telegram_channels did not find t.me/s/ URL"


def test_extract_telegram_channels_deduplicates_case_insensitive() -> None:
    text = "@Gonzo_ML \u0438 \u0434\u0443\u0431\u043b\u044c @gonzo_ml"
    result = extract_telegram_channels(text)

    assert result.count("gonzo_ml") == 1, "extract_telegram_channels did not deduplicate"


def test_parse_telegram_channel_tokens_finds_at_mention() -> None:
    text = "@gonzo_ml, t.me/fondnauk invalid-token"
    result = parse_telegram_channel_tokens(text)

    assert "gonzo_ml" in result, "parse_telegram_channel_tokens did not find @mention"


def test_parse_telegram_channel_tokens_finds_bare_url() -> None:
    text = "@gonzo_ml, t.me/fondnauk invalid-token"
    result = parse_telegram_channel_tokens(text)

    assert "fondnauk" in result, "parse_telegram_channel_tokens did not find bare t.me URL"


def test_parse_telegram_channel_tokens_finds_full_url() -> None:
    text = "https://t.me/s/sciencefocus some-invalid"
    result = parse_telegram_channel_tokens(text)

    assert "sciencefocus" in result, "parse_telegram_channel_tokens did not find https URL"


def test_extract_reddit_subreddits_finds_mention() -> None:
    text = "\u0421\u043b\u0435\u0434\u0438 \u0437\u0430 r/Python \u0438 \u0434\u0440."
    result = extract_reddit_subreddits(text)

    assert "python" in result, "extract_reddit_subreddits did not find r/Python mention"


def test_extract_reddit_subreddits_finds_url() -> None:
    text = "https://www.reddit.com/r/programming/new/ \u043d\u043e\u0432\u043e\u0441\u0442\u0438"
    result = extract_reddit_subreddits(text)

    assert "programming" in result, "extract_reddit_subreddits did not find reddit URL"


def test_extract_reddit_subreddits_deduplicates_case_insensitive() -> None:
    text = "r/Python \u0438 /r/python"
    result = extract_reddit_subreddits(text)

    assert result.count("python") == 1, "extract_reddit_subreddits did not deduplicate"


def test_parse_source_tokens_finds_telegram_channel() -> None:
    text = "@gonzo_ml r/python https://t.me/fondnauk https://x.com/OpenAI"
    channels, _, _ = parse_source_tokens(text)

    assert "gonzo_ml" in channels, "parse_source_tokens did not find telegram channel mention"


def test_parse_source_tokens_finds_fondnauk_from_url() -> None:
    text = "@gonzo_ml r/python https://t.me/fondnauk https://x.com/OpenAI"
    channels, _, _ = parse_source_tokens(text)

    assert "fondnauk" in channels, "parse_source_tokens did not find fondnauk from URL"


def test_parse_source_tokens_finds_reddit_subreddit() -> None:
    text = "@gonzo_ml r/python https://t.me/fondnauk https://x.com/OpenAI"
    _, subreddits, _ = parse_source_tokens(text)

    assert "python" in subreddits, "parse_source_tokens did not find reddit subreddit"


def test_parse_source_tokens_finds_twitter_account() -> None:
    text = "@gonzo_ml r/python https://t.me/fondnauk https://x.com/OpenAI"
    _, _, twitter = parse_source_tokens(text)

    assert "openai" in twitter, "parse_source_tokens did not find twitter account"


def test_extract_twitter_accounts_finds_x_com_url() -> None:
    text = "\u0421\u043b\u0435\u0434\u0438 \u0437\u0430 https://x.com/OpenAI"
    result = extract_twitter_accounts(text)

    assert "openai" in result, "extract_twitter_accounts did not find x.com URL"


def test_extract_twitter_accounts_finds_twitter_com_url() -> None:
    text = "\u041f\u043e\u0434\u043f\u0438\u0441\u043a\u0430 \u043d\u0430 https://twitter.com/NASA"
    result = extract_twitter_accounts(text)

    assert "nasa" in result, "extract_twitter_accounts did not find twitter.com URL"


def test_parse_twitter_account_tokens_finds_x_com_bare_url() -> None:
    text = "x.com/OpenAI https://twitter.com/NASA"
    result = parse_twitter_account_tokens(text)

    assert "openai" in result, "parse_twitter_account_tokens did not find bare x.com URL"


def test_parse_twitter_account_tokens_finds_full_twitter_url() -> None:
    text = "x.com/OpenAI https://twitter.com/NASA"
    result = parse_twitter_account_tokens(text)

    assert "nasa" in result, "parse_twitter_account_tokens did not find full twitter URL"


def test_parse_twitter_account_tokens_ignores_long_token() -> None:
    text = "invalid-token-too-long-123456"
    result = parse_twitter_account_tokens(text)

    assert result == [], "parse_twitter_account_tokens did not ignore an overly long token"
