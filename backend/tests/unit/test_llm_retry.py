import logging
import random
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from news_service.core.llm_retry import with_llm_retry

logging.disable(logging.CRITICAL)


def _timeout() -> litellm.Timeout:
    return litellm.Timeout(
        message=f"timed out {random.randint(1, 9999)}",
        model="test-model",
        llm_provider="openai",
    )


def _auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message="Invalid API key",
        model="test-model",
        llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_success_returns_result_without_retrying() -> None:
    calls = 0

    @with_llm_retry(max_attempts=3)
    async def succeed() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await succeed()
    assert result == "ok" and calls == 1, (
        "success path did not short-circuit retries on the first call"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: litellm.Timeout(message="t", model="m", llm_provider="openai"),
        lambda: litellm.APIConnectionError(message="c", model="m", llm_provider="openai"),
        lambda: litellm.RateLimitError(message="r", model="m", llm_provider="openai"),
        lambda: litellm.InternalServerError(message="s", model="m", llm_provider="openai"),
    ],
    ids=["timeout", "connection", "rate_limit", "server_error"],
)
async def test_transient_errors_are_retried(error_factory) -> None:
    calls = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise error_factory()
        return "recovered"

    with patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock):
        result = await flaky()

    assert result == "recovered" and calls == 3, "transient error path did not retry until success"


@pytest.mark.asyncio
async def test_exhausted_attempts_raise_last_exception() -> None:
    @with_llm_retry(max_attempts=2, base_delay_seconds=0.01)
    async def always_fail() -> None:
        raise _timeout()

    with (
        patch("news_service.core.llm_retry.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(litellm.Timeout),
    ):
        await always_fail()


@pytest.mark.asyncio
async def test_permanent_error_is_not_retried() -> None:
    calls = 0

    @with_llm_retry(max_attempts=3, base_delay_seconds=0.01)
    async def fail_permanently() -> None:
        nonlocal calls
        calls += 1
        raise _auth_error()

    with pytest.raises(litellm.AuthenticationError):
        await fail_permanently()

    assert calls == 1, "permanent error should not be retried"
