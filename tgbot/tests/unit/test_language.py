"""Tests for normalize_language_code function."""

import logging

from tgbot.language import normalize_language_code

logging.disable(logging.CRITICAL)


class TestNormalizeLanguageCode:
    def test_returns_en_for_en(self) -> None:
        result = normalize_language_code("en")
        assert result == "en", f"did not return en for input en, got {result!r}"

    def test_returns_ru_for_ru(self) -> None:
        result = normalize_language_code("ru")
        assert result == "ru", f"did not return ru for input ru, got {result!r}"

    def test_returns_en_for_en_us(self) -> None:
        result = normalize_language_code("en-US")
        assert result == "en", f"did not return en for input en-US, got {result!r}"

    def test_returns_ru_for_ru_ru(self) -> None:
        result = normalize_language_code("ru-RU")
        assert result == "ru", f"did not return ru for input ru-RU, got {result!r}"

    def test_returns_none_for_none(self) -> None:
        result = normalize_language_code(None)
        assert result is None, f"did not return None for None input, got {result!r}"

    def test_returns_none_for_unsupported_language(self) -> None:
        result = normalize_language_code("de")
        assert result is None, f"did not return None for unsupported language de, got {result!r}"

    def test_handles_whitespace_padded_input(self) -> None:
        result = normalize_language_code("  ru  ")
        assert result == "ru", f"did not handle whitespace-padded input, got {result!r}"

    def test_handles_uppercase_input(self) -> None:
        result = normalize_language_code("EN")
        assert result == "en", f"did not handle uppercase input EN, got {result!r}"
