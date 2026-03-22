import logging
import random
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

logging.disable(logging.CRITICAL)


def _make_api_timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", f"https://api.example.com/v1/{random.randint(1, 9999)}")
    return APITimeoutError(request=request)


def _make_api_connection_error() -> APIConnectionError:
    request = httpx.Request("POST", f"https://api.example.com/v1/{random.randint(1, 9999)}")
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
async def test_decorated_function_returns_result_on_first_successful_call() -> None:
    expected = f"результат-{random.randint(100, 999)}"

    @with_llm_retry(max_attempts=3)
    async def succeed():
        return expected

    result = await succeed()

    assert result == expected, "decorated function did not return expected result"


@pytest.mark.asyncio
async def test_decorated_function_calls_underlying_function_exactly_once_on_success() -> None:
    call_count = 0

    @with_llm_retry(max_attempts=3)
    async def succeed():
        nonlocal call_count
        call_count += 1
        return "ок"

    await succeed()

    assert call_count == 1, "function was not called exactly once on success"


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
async def test_retries_on_transient_error_then_returns_result(error_factory) -> None:
    call_count = 0
    expected = f"восстановление-{random.randint(100, 999)}"

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise error_factory()
        return expected

    with patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await fail_then_succeed()

    assert result == expected, "function did not return expected result after transient errors"


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
async def test_retries_on_transient_error_calls_function_expected_number_of_times(
    error_factory,
) -> None:
    call_count = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise error_factory()
        return "ок"

    with patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock):
        await fail_then_succeed()

    assert call_count == 3, "function was not called expected number of times during retries"


@pytest.mark.asyncio
async def test_raises_after_all_retries_exhausted() -> None:
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
async def test_permanent_error_is_not_retried(error_factory) -> None:
    call_count = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_permanently():
        nonlocal call_count
        call_count += 1
        raise error_factory()

    with pytest.raises(type(error_factory())):
        await fail_permanently()

    assert call_count == 1, "permanent error was retried when it should not have been"


@pytest.mark.asyncio
async def test_exponential_backoff_first_delay_equals_base_delay() -> None:
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
    assert delays[0] == pytest.approx(1.0), "first backoff delay did not equal base delay"


@pytest.mark.asyncio
async def test_exponential_backoff_second_delay_doubles_base_delay() -> None:
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
    assert delays[1] == pytest.approx(2.0), "second backoff delay was not double the base delay"


@pytest.mark.asyncio
async def test_exponential_backoff_third_delay_quadruples_base_delay() -> None:
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
    assert delays[2] == pytest.approx(4.0), "third backoff delay was not quadruple the base delay"


@pytest.mark.asyncio
async def test_exponential_backoff_produces_three_sleep_calls_for_four_attempts() -> None:
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
    assert len(delays) == 3, "backoff did not produce expected number of sleep calls"


@pytest.mark.asyncio
async def test_max_delay_caps_backoff_at_configured_maximum() -> None:
    max_delay = 15.0

    @with_llm_retry(max_attempts=4, base_delay_seconds=10.0, max_delay_seconds=max_delay)
    async def always_fail():
        raise _make_api_timeout_error()

    with (
        patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("news_service.core.llm_retry.random.uniform", return_value=0.0),
        pytest.raises(APITimeoutError),
    ):
        await always_fail()

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert all(d <= max_delay for d in delays), "backoff delay exceeded configured maximum"
