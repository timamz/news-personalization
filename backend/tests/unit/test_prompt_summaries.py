"""Tests for build_prompt_summary function."""

import logging
import random
import uuid

from news_service.services.prompt_summaries import build_prompt_summary

logging.disable(logging.CRITICAL)


class TestBuildPromptSummary:
    def test_short_prompt_returns_unchanged(self) -> None:
        tag = uuid.uuid4().hex[:6]
        prompt = f"Новости технологий {tag}"
        result = build_prompt_summary(prompt, max_length=200)
        assert result == prompt, f"short prompt was unexpectedly modified: {result!r}"

    def test_prompt_with_excessive_whitespace_is_normalized(self) -> None:
        tag = uuid.uuid4().hex[:6]
        prompt = f"  Новости   технологий   {tag}  "
        result = build_prompt_summary(prompt, max_length=200)
        expected = f"Новости технологий {tag}"
        assert result == expected, f"whitespace was not normalized: {result!r}"

    def test_long_prompt_is_truncated_with_ellipsis(self) -> None:
        max_len = random.randint(20, 50)
        long_prompt = "Технологические " * 20
        result = build_prompt_summary(long_prompt.strip(), max_length=max_len)
        assert result.endswith("\u2026"), f"truncated prompt does not end with ellipsis: {result!r}"

    def test_truncated_prompt_does_not_exceed_max_length(self) -> None:
        max_len = random.randint(15, 60)
        long_prompt = "Архитектура " * 30
        result = build_prompt_summary(long_prompt.strip(), max_length=max_len)
        assert len(result) <= max_len, f"result length {len(result)} exceeds max_length {max_len}"

    def test_exact_length_prompt_returns_unchanged(self) -> None:
        max_len = random.randint(10, 50)
        prompt = "А" * max_len
        result = build_prompt_summary(prompt, max_length=max_len)
        assert result == prompt, f"exact-length prompt was unexpectedly modified: {result!r}"

    def test_non_ascii_russian_prompt_truncation_is_correct(self) -> None:
        max_len = 25
        prompt = "Последние новости о российской экономике и финансах"
        result = build_prompt_summary(prompt, max_length=max_len)
        assert len(result) <= max_len, (
            f"Russian prompt truncation produced length {len(result)}, exceeds {max_len}"
        )
