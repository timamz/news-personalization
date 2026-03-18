"""Tests for the LLM retry decorator."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from news_service.core.llm_retry import with_llm_retry


def _make_api_timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return APITimeoutError(request=request)


def _make_api_connection_error() -> APIConnectionError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return APIConnectionError(request=request)


def _make_rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.example.com"))
    return RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_internal_server_error() -> InternalServerError:
    response = httpx.Response(500, request=httpx.Request("POST", "https://api.example.com"))
    return InternalServerError(
        message="Internal server error",
        response=response,
        body={"error": {"message": "Internal server error"}},
    )


def _make_auth_error() -> AuthenticationError:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.example.com"))
    return AuthenticationError(
        message="Invalid API key",
        response=response,
        body={"error": {"message": "Invalid API key"}},
    )


def _make_bad_request_error() -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.example.com"))
    return BadRequestError(
        message="Bad request",
        response=response,
        body={"error": {"message": "Bad request"}},
    )


@pytest.mark.asyncio
async def test_success_on_first_call():
    call_count = 0

    @with_llm_retry(max_attempts=3)
    async def succeed():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await succeed()
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        _make_api_timeout_error,
        _make_api_connection_error,
        _make_rate_limit_error,
        _make_internal_server_error,
    ],
    ids=["timeout", "connection", "rate_limit", "server_error"],
)
async def test_retries_on_transient_error_then_succeeds(error_factory):
    call_count = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise error_factory()
        return "ok"

    with patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await fail_then_succeed()

    assert result == "ok"
    assert call_count == 3
    assert mock_sleep.await_count == 2


@pytest.mark.asyncio
async def test_raises_after_all_retries_exhausted():
    @with_llm_retry(max_attempts=2, base_delay_seconds=0.01)
    async def always_fail():
        raise _make_api_timeout_error()

    with (
        patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(APITimeoutError),
    ):
        await always_fail()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [_make_auth_error, _make_bad_request_error],
    ids=["auth", "bad_request"],
)
async def test_permanent_errors_not_retried(error_factory):
    call_count = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_permanently():
        nonlocal call_count
        call_count += 1
        raise error_factory()

    with pytest.raises(type(error_factory())):
        await fail_permanently()

    assert call_count == 1


@pytest.mark.asyncio
async def test_exponential_backoff_delays():
    @with_llm_retry(max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=30.0)
    async def always_fail():
        raise _make_api_timeout_error()

    with (
        patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("news_service.core.llm_retry.random.uniform", return_value=0.0),
        pytest.raises(APITimeoutError),
    ):
        await always_fail()

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(delays) == 3
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[2] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_max_delay_capped():
    @with_llm_retry(max_attempts=4, base_delay_seconds=10.0, max_delay_seconds=15.0)
    async def always_fail():
        raise _make_api_timeout_error()

    with (
        patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("news_service.core.llm_retry.random.uniform", return_value=0.0),
        pytest.raises(APITimeoutError),
    ):
        await always_fail()

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert all(d <= 15.0 for d in delays)


@pytest.mark.asyncio
async def test_preserves_function_metadata():
    @with_llm_retry()
    async def my_function():
        """My docstring."""
        return 42

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."
