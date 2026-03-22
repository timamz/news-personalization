"""Tests for dedup utilities in event_notifications."""

import logging
import uuid

from news_service.services.event_notifications import (
    normalize_event_text,
    text_similarity,
    token_overlap,
)

logging.disable(logging.CRITICAL)


class TestTokenOverlap:
    def test_identical_texts_return_one(self) -> None:
        tag = uuid.uuid4().hex[:8]
        text = f"производство технологий компания {tag}"
        result = token_overlap(text, text)
        assert result == 1.0, f"identical texts did not yield overlap 1.0, got {result}"

    def test_completely_different_texts_return_zero(self) -> None:
        left = f"abcdef{uuid.uuid4().hex[:8]}"
        right = f"zyxwvut{uuid.uuid4().hex[:8]}"
        result = token_overlap(left, right)
        assert result == 0.0, f"completely different texts did not yield 0.0, got {result}"

    def test_short_tokens_are_ignored(self) -> None:
        left = "the and for but"
        right = "the and for but"
        result = token_overlap(left, right)
        assert result == 0.0, f"short tokens (<4 chars) were not ignored, got {result}"

    def test_partial_overlap_returns_correct_value(self) -> None:
        left = "производство технологий компания"
        right = "производство технологий обновление"
        result = token_overlap(left, right)
        assert 0.0 < result < 1.0, f"partial overlap not between 0 and 1, got {result}"

    def test_empty_input_returns_zero(self) -> None:
        result = token_overlap("", "")
        assert result == 0.0, f"empty input did not yield overlap 0.0, got {result}"


class TestTextSimilarity:
    def test_identical_strings_return_one(self) -> None:
        tag = uuid.uuid4().hex[:6]
        text = f"Россия политика новости {tag}"
        result = text_similarity(text, text)
        assert result == 1.0, f"identical strings did not yield similarity 1.0, got {result}"

    def test_completely_different_strings_return_zero(self) -> None:
        left = "aaaa"
        right = "zzzz"
        result = text_similarity(left, right)
        assert result == 0.0, f"different strings did not yield similarity 0.0, got {result}"

    def test_partial_match_returns_value_between_zero_and_one(self) -> None:
        tag = uuid.uuid4().hex[:4]
        left = f"производство технологий {tag}"
        right = f"производство компаний {tag}"
        result = text_similarity(left, right)
        assert 0.0 < result < 1.0, f"partial match not between 0 and 1, got {result}"


class TestNormalizeEventText:
    def test_joins_multiple_parts(self) -> None:
        tag = uuid.uuid4().hex[:6]
        result = normalize_event_text(f"Новость {tag}", "подробности")
        assert tag in result, f"joined parts do not contain tag {tag}: {result!r}"

    def test_handles_none_parts(self) -> None:
        tag = uuid.uuid4().hex[:6]
        result = normalize_event_text(None, f"текст {tag}", None)
        assert tag in result, f"None parts caused loss of content, got {result!r}"

    def test_normalizes_russian_yo_to_ye(self) -> None:
        result = normalize_event_text("ёлка пёс")
        assert "ё" not in result, f"yo character was not normalized to ye: {result!r}"

    def test_normalizes_russian_yo_replacement_value(self) -> None:
        result = normalize_event_text("ёлка")
        assert "елка" in result, f"yo was not replaced with ye properly: {result!r}"

    def test_extracts_word_tokens_only(self) -> None:
        result = normalize_event_text("hello!!! world??? @#$% test123")
        assert result == "hello world test123", f"non-word tokens not stripped: {result!r}"

    def test_all_none_parts_return_empty_string(self) -> None:
        result = normalize_event_text(None, None)
        assert result == "", f"all-None parts did not return empty string: {result!r}"
