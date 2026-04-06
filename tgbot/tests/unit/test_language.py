"""Tests for normalize_language_code function."""

import logging

import pytest

from tgbot.language import normalize_language_code

logging.disable(logging.CRITICAL)


class TestNormalizeLanguageCode:
    @pytest.mark.parametrize(
        ("raw_input", "expected"),
        [
            ("en", "en"),
            ("ru", "ru"),
            ("en-US", "en"),
            ("ru-RU", "ru"),
            (None, None),
            ("de", None),
            ("  ru  ", "ru"),
            ("EN", "en"),
        ],
        ids=[
            "en_returns_en",
            "ru_returns_ru",
            "en_us_returns_en",
            "ru_ru_returns_ru",
            "none_returns_none",
            "unsupported_returns_none",
            "whitespace_padded_returns_stripped",
            "uppercase_returns_lowercase",
        ],
    )
    def test_normalize_language_code_maps_input_to_expected_output(
        self, raw_input: str | None, expected: str | None
    ) -> None:
        result = normalize_language_code(raw_input)
        assert result == expected, (
            f"normalize_language_code({raw_input!r}) did not return {expected!r}, got {result!r}"
        )
