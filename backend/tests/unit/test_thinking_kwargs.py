"""Unit tests for the DeepSeek thinking-mode kwargs translator."""

import logging

import pytest

from news_service.core.llm import thinking_kwargs

logging.disable(logging.CRITICAL)


def test_returns_enabled_extra_body_for_deepseek_when_reasoning_true() -> None:
    result = thinking_kwargs("deepseek/deepseek-v4-flash", reasoning=True)
    assert result == {"extra_body": {"thinking": {"type": "enabled"}}}, (
        f"deepseek + reasoning=True did not produce enabled thinking flag: {result!r}"
    )


def test_returns_disabled_extra_body_for_deepseek_when_reasoning_false() -> None:
    result = thinking_kwargs("deepseek/deepseek-v4-flash", reasoning=False)
    assert result == {"extra_body": {"thinking": {"type": "disabled"}}}, (
        f"deepseek + reasoning=False did not produce disabled thinking flag: {result!r}"
    )


def test_returns_empty_for_deepseek_when_reasoning_none() -> None:
    result = thinking_kwargs("deepseek/deepseek-v4-flash", reasoning=None)
    assert result == {}, f"reasoning=None should defer to provider default, got: {result!r}"


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.4-nano",
        "openai/text-embedding-3-small",
        "anthropic/claude-sonnet-4-7",
        "gemini/gemini-2.5-pro",
    ],
)
def test_returns_empty_for_non_deepseek_models_even_when_reasoning_set(model: str) -> None:
    result_true = thinking_kwargs(model, reasoning=True)
    result_false = thinking_kwargs(model, reasoning=False)
    assert result_true == {} and result_false == {}, (
        f"non-DeepSeek model {model!r} should not receive thinking extra_body, "
        f"got True={result_true!r} False={result_false!r}"
    )
