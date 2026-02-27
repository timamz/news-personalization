from tgbot.source_parser import extract_telegram_channels, parse_telegram_channel_tokens


def test_extract_telegram_channels_from_mentions_and_urls() -> None:
    text = "Следи за @Gonzo_ML и https://t.me/s/fondnauk, дубли @gonzo_ml"
    result = extract_telegram_channels(text)
    assert result == ["gonzo_ml", "fondnauk"]


def test_parse_telegram_channel_tokens_supports_mixed_input() -> None:
    text = "@gonzo_ml, https://t.me/fondnauk invalid-token"
    result = parse_telegram_channel_tokens(text)
    assert result == ["gonzo_ml", "fondnauk"]
