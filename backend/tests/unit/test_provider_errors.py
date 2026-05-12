"""Unit tests for the provider-error classifier."""

import logging

import litellm
import pytest

from news_service.core.provider_errors import (
    ProviderLimitError,
    classify_litellm_error,
    classify_search_status,
)

logging.disable(logging.CRITICAL)


def test_classifier_recognizes_balance_keyword_in_rate_limit_error() -> None:
    exc = litellm.RateLimitError(
        message="недостаточно средств на балансе (request id: abc)",
        model="openai/text-embedding-3-small",
        llm_provider="openai",
    )
    result = classify_litellm_error(exc, provider="openai/text-embedding-3-small")
    assert result is not None and result.kind == "balance", (
        f"balance keyword was not classified as balance: kind={result.kind if result else None}"
    )


def test_classifier_marks_plain_rate_limit_as_rate_limit() -> None:
    exc = litellm.RateLimitError(
        message="Rate limit reached for requests",
        model="openai/gpt-5.4-nano",
        llm_provider="openai",
    )
    result = classify_litellm_error(exc, provider="openai/gpt-5.4-nano")
    assert result is not None and result.kind == "rate_limit", (
        f"plain rate limit was misclassified: kind={result.kind if result else None}"
    )


def test_classifier_marks_authentication_error_as_auth() -> None:
    exc = litellm.AuthenticationError(
        message="Invalid API key",
        model="openai/gpt-5.4-nano",
        llm_provider="openai",
    )
    result = classify_litellm_error(exc, provider="openai/gpt-5.4-nano")
    assert result is not None and result.kind == "auth", (
        f"AuthenticationError was misclassified: kind={result.kind if result else None}"
    )


def test_classifier_marks_402_api_error_as_balance() -> None:
    exc = litellm.APIError(
        status_code=402,
        message="Payment Required",
        llm_provider="openai",
        model="openai/gpt-5.4-nano",
    )
    result = classify_litellm_error(exc, provider="openai/gpt-5.4-nano")
    assert result is not None and result.kind == "balance", (
        f"402 was misclassified: kind={result.kind if result else None}"
    )


def test_classifier_ignores_unrelated_api_error() -> None:
    exc = litellm.APIError(
        status_code=500,
        message="Internal server error",
        llm_provider="openai",
        model="openai/gpt-5.4-nano",
    )
    result = classify_litellm_error(exc, provider="openai/gpt-5.4-nano")
    assert result is None, "500 should not be treated as a usage limit"


def test_classifier_detects_balance_keyword_in_generic_exception() -> None:
    exc = ValueError("Provider returned: insufficient_user_quota for request")
    result = classify_litellm_error(exc, provider="openai/text-embedding-3-small")
    assert result is not None and result.kind == "balance", (
        f"balance keyword in generic exception was not classified: "
        f"kind={result.kind if result else None}"
    )


def test_search_classifier_marks_402_as_balance() -> None:
    result = classify_search_status(status_code=402, body_snippet="Payment Required")
    assert result is not None and result.kind == "balance", (
        f"402 was misclassified for search: kind={result.kind if result else None}"
    )


def test_search_classifier_marks_403_without_balance_keyword_as_auth() -> None:
    result = classify_search_status(
        status_code=403, body_snippet="Service account does not have search-api.executor role"
    )
    assert result is not None and result.kind == "auth", (
        f"403 with no balance hint should be auth, got kind={result.kind if result else None}"
    )


def test_search_classifier_marks_429_as_rate_limit() -> None:
    result = classify_search_status(status_code=429, body_snippet="Too Many Requests")
    assert result is not None and result.kind == "rate_limit", (
        f"429 was misclassified for search: kind={result.kind if result else None}"
    )


def test_search_classifier_ignores_5xx() -> None:
    result = classify_search_status(status_code=503, body_snippet="upstream timeout")
    assert result is None, "5xx is not a usage limit and must not be classified"


def test_provider_limit_error_carries_provider_and_kind() -> None:
    err = ProviderLimitError(provider="openai/text-embedding-3-small", kind="balance", message="x")
    assert err.provider == "openai/text-embedding-3-small" and err.kind == "balance", (
        "ProviderLimitError lost provider/kind metadata"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "insufficient_user_quota",
        "Insufficient balance to complete request",
        "недостаточно средств на балансе",
        "Quota exceeded for this model",
    ],
)
def test_classifier_recognizes_multiple_balance_phrases(phrase: str) -> None:
    exc = litellm.APIError(
        status_code=403,
        message=phrase,
        llm_provider="openai",
        model="openai/text-embedding-3-small",
    )
    result = classify_litellm_error(exc, provider="openai/text-embedding-3-small")
    assert result is not None and result.kind == "balance", (
        f"balance phrase {phrase!r} was not detected: kind={result.kind if result else None}"
    )
