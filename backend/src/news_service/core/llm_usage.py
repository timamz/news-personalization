"""Per-call attribution + cost ledger for every LiteLLM dispatch.

The backend has two paths to LiteLLM:

  * Direct ``litellm.acompletion`` / ``litellm.aembedding`` calls from
    ``core.llm`` (judges, batch assessor, embeddings).
  * Google ADK's ``LiteLlm`` wrapper used by every agentic agent
    (Conversational, Discovery, Finder, Writer, Reflector, Verifier).

Both paths ultimately invoke ``litellm.acompletion`` at the bottom of
the stack, which means ``litellm.success_callback`` / ``failure_callback``
fire exactly once per call in either path. Registering a single callable
in this module is therefore sufficient to account for every LLM call
the service makes.

Attribution comes from four ``ContextVar`` stacks set by the caller:

  * ``current_agent``       -- which agent made the call
  * ``current_run_id``      -- benchmark / correlation id, from the
                               ``X-Run-Id`` request header
  * ``current_user_id``     -- authenticated user, set by the request
                               middleware
  * ``current_subscription`` -- the subscription a Celery task was
                               dispatched for, set at task entry

ContextVars propagate across ``await`` in the same asyncio task, so the
callback sees the values that were set by whoever initiated the call.

USD cost is computed from the static pricing table in
``settings.llm_model_pricing_usd_per_1m``. The config validator refuses
to start the service if any configured model is missing a price entry,
so a zero-cost row here indicates zero tokens, not a missing rate card.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from decimal import Decimal
from typing import Any

import litellm

from news_service.core.config import get_settings
from news_service.db.session import async_session_factory
from news_service.models.llm_usage import LLMUsage

logger = logging.getLogger(__name__)

_TOKEN_UNIT = Decimal(1_000_000)

current_agent: ContextVar[str | None] = ContextVar("current_agent", default=None)
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
current_user_id: ContextVar[uuid.UUID | None] = ContextVar("current_user_id", default=None)
current_subscription_id: ContextVar[uuid.UUID | None] = ContextVar(
    "current_subscription_id", default=None
)


@contextmanager
def agent_tag(name: str):
    """Scope LLM calls inside the block to a named agent."""
    token = current_agent.set(name)
    try:
        yield
    finally:
        current_agent.reset(token)


@contextmanager
def run_id_tag(run_id: str | None):
    """Scope LLM calls inside the block to a benchmark / correlation run id."""
    token = current_run_id.set(run_id)
    try:
        yield
    finally:
        current_run_id.reset(token)


@contextmanager
def user_tag(user_id: uuid.UUID | None):
    """Scope LLM calls inside the block to an authenticated user."""
    token = current_user_id.set(user_id)
    try:
        yield
    finally:
        current_user_id.reset(token)


@contextmanager
def subscription_tag(subscription_id: uuid.UUID | None):
    """Scope LLM calls inside the block to a subscription."""
    token = current_subscription_id.set(subscription_id)
    try:
        yield
    finally:
        current_subscription_id.reset(token)


def _resolve_pricing(model: str) -> dict[str, float] | None:
    """Find a pricing entry for ``model`` tolerating provider-prefix variants.

    LiteLLM / ADK sometimes strip the ``openai/`` provider prefix by the
    time a call reaches the success callback, and sometimes leave it on.
    Configured pricing keys are typically ``openai/gpt-5.4-nano`` but the
    observed ``kwargs['model']`` may be ``gpt-5.4-nano``. We try the
    model verbatim, then try it with and without the leading provider
    segment so either form matches a single config entry.
    """
    pricing_map = get_settings().llm_model_pricing_usd_per_1m
    if model in pricing_map:
        return pricing_map[model]
    if "/" in model:
        stripped = model.split("/", 1)[1]
        if stripped in pricing_map:
            return pricing_map[stripped]
    for configured_key in pricing_map:
        if "/" in configured_key and configured_key.split("/", 1)[1] == model:
            return pricing_map[configured_key]
    return None


def compute_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> Decimal | None:
    """Return the USD cost for a call given its provider-reported token counts.

    Returns ``None`` when the model has no pricing entry -- should never
    happen after the config validator, but we stay defensive here so an
    unknown model only skips cost accounting instead of breaking the
    call ledger entirely.
    """
    pricing = _resolve_pricing(model)
    if pricing is None:
        return None
    input_price = Decimal(str(pricing["input"]))
    output_price = Decimal(str(pricing["output"]))
    pt = Decimal(prompt_tokens or 0)
    ct = Decimal(completion_tokens or 0)
    return (input_price * pt + output_price * ct) / _TOKEN_UNIT


def _as_int(value: Any) -> int | None:
    """Coerce a token-count field to int, returning None on missing / garbage."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage(response_obj: Any) -> dict[str, int | None]:
    """Pull prompt / completion / cached / reasoning token counts from a response.

    LiteLLM normalises responses to OpenAI's shape. Depending on provider
    and version, ``usage`` is either a Pydantic-style object (attribute
    access) or a dict. We accept either; missing fields become ``None``.
    """
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }

    def pick(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    prompt_details = pick("prompt_tokens_details")
    completion_details = pick("completion_tokens_details")

    def pick_detail(details: Any, name: str) -> Any:
        if details is None:
            return None
        if isinstance(details, dict):
            return details.get(name)
        return getattr(details, name, None)

    return {
        "prompt_tokens": _as_int(pick("prompt_tokens")),
        "completion_tokens": _as_int(pick("completion_tokens")),
        "total_tokens": _as_int(pick("total_tokens")),
        "cached_tokens": _as_int(pick_detail(prompt_details, "cached_tokens")),
        "reasoning_tokens": _as_int(pick_detail(completion_details, "reasoning_tokens")),
    }


def _infer_call_type(kwargs: dict[str, Any], response_obj: Any = None) -> str:
    """Distinguish chat completion from embedding using both kwargs and response.

    LiteLLM's callback kwargs can carry ``messages`` even for embedding
    calls (set to None/empty), so a bare "is 'messages' a key?" test is
    unreliable. Prefer the response shape when available: a
    ``ModelResponse.choices`` attribute means chat; ``EmbeddingResponse.data``
    with a first-element ``embedding`` field means embedding. Fall back
    to kwargs only if response inspection fails.
    """
    if response_obj is not None:
        if getattr(response_obj, "choices", None):
            return "chat"
        data = getattr(response_obj, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
            return "embedding"
    messages = kwargs.get("messages")
    if isinstance(messages, list) and messages:
        return "chat"
    if "input" in kwargs and kwargs.get("input") is not None:
        return "embedding"
    return "other"


def _latency_ms(start_time: Any, end_time: Any) -> int | None:
    """Best-effort latency in milliseconds. LiteLLM passes datetime or float."""
    try:
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return int((end_time - start_time).total_seconds() * 1000)
        return int((float(end_time) - float(start_time)) * 1000)
    except (TypeError, ValueError):
        return None


async def _persist(row: LLMUsage) -> None:
    """Write one usage row on a dedicated short-lived session.

    Errors are swallowed with a log line: a dropped ledger row must
    never break the user-visible call path. Pipeline correctness is
    strictly more important than perfect accounting fidelity.
    """
    try:
        async with async_session_factory() as session:
            session.add(row)
            await session.commit()
    except Exception:
        logger.exception(
            "Failed to persist llm_usage row (agent=%s, model=%s)",
            row.agent,
            row.model,
        )


async def record_web_search(
    *,
    latency_ms: int | None,
    error: str | None,
) -> None:
    """Emit one ``call_type='web_search'`` row per Yandex search dispatch.

    Web searches are not LLM calls but they cost real money and the
    discovery pipeline fires a lot of them -- the unit-economics report
    needs to count and price them alongside LLM spend. The row reuses
    ``LLMUsage`` (the table's docstring already lists web searches as in
    scope) with tokens left NULL and ``cost_usd`` taken from
    ``settings.yandex_search_price_usd_per_call``. Attribution comes from
    the same ContextVars that tag LLM rows, so a search fired inside a
    finder's ReAct loop attributes to ``agent='finder'`` and the
    enclosing subscription.
    """
    settings = get_settings()
    try:
        cost = Decimal(str(settings.yandex_search_price_usd_per_call))
    except Exception:
        cost = Decimal("0")
    row = LLMUsage(
        agent=current_agent.get(),
        run_id=current_run_id.get(),
        user_id=current_user_id.get(),
        subscription_id=current_subscription_id.get(),
        provider="yandex",
        model="yandex_search_v2",
        call_type="web_search",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        reasoning_tokens=None,
        cost_usd=cost,
        latency_ms=latency_ms,
        error=error,
    )
    await _persist(row)


async def _record_usage_event(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    *,
    error: str | None = None,
) -> None:
    """Build and persist one LLMUsage row for a completed or failed call."""
    model = str(kwargs.get("model") or "")
    call_type = _infer_call_type(kwargs, response_obj)

    tokens = (
        _extract_usage(response_obj)
        if response_obj is not None
        else {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }
    )

    cost = compute_cost_usd(
        model=model,
        prompt_tokens=tokens["prompt_tokens"],
        completion_tokens=tokens["completion_tokens"],
    )

    row = LLMUsage(
        agent=current_agent.get(),
        run_id=current_run_id.get(),
        user_id=current_user_id.get(),
        subscription_id=current_subscription_id.get(),
        provider="litellm",
        model=model,
        call_type=call_type,
        prompt_tokens=tokens["prompt_tokens"],
        completion_tokens=tokens["completion_tokens"],
        total_tokens=tokens["total_tokens"],
        cached_tokens=tokens["cached_tokens"],
        reasoning_tokens=tokens["reasoning_tokens"],
        cost_usd=cost,
        latency_ms=_latency_ms(start_time, end_time),
        error=error,
    )
    await _persist(row)


async def _on_success(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    try:
        await _record_usage_event(kwargs, response_obj, start_time, end_time, error=None)
    except Exception:
        logger.exception("llm_usage success callback crashed")


async def _on_failure(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    err: str | None = None
    try:
        exc = kwargs.get("exception") if isinstance(kwargs, dict) else None
        if exc is None and isinstance(response_obj, BaseException):
            exc = response_obj
        if exc is not None:
            err = f"{type(exc).__name__}: {exc}"[:2000]
        await _record_usage_event(kwargs, None, start_time, end_time, error=err or "unknown")
    except Exception:
        logger.exception("llm_usage failure callback crashed")


_INSTALLED = False


def install_usage_callback() -> None:
    """Register the usage callback with LiteLLM. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    if asyncio.iscoroutinefunction(_on_success):
        success_list: list[Any] = list(getattr(litellm, "success_callback", []) or [])
        failure_list: list[Any] = list(getattr(litellm, "failure_callback", []) or [])
        if _on_success not in success_list:
            success_list.append(_on_success)
        if _on_failure not in failure_list:
            failure_list.append(_on_failure)
        litellm.success_callback = success_list
        litellm.failure_callback = failure_list
    _INSTALLED = True
    logger.info("LLM usage callback installed")
