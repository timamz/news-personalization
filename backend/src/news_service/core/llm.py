"""Provider-agnostic LLM interface via LiteLLM.

Wraps litellm for chat completions and embeddings, enabling transparent
switching between OpenAI, Gemini, Anthropic, self-hosted models, and
any other provider supported by LiteLLM.
"""

import logging
from typing import Any

import litellm

from news_service.core.config import get_settings
from news_service.core.llm_errors import StructuredOutputParseError
from news_service.core.llm_retry import with_llm_retry
from news_service.core.provider_errors import ProviderLimitError, classify_litellm_error

__all__ = [
    "StructuredOutputParseError",
    "chat_completion",
    "embed_text",
    "embed_texts",
    "thinking_kwargs",
]

logger = logging.getLogger(__name__)

settings = get_settings()

EMBEDDING_MAX_CHARS = 4000
EMBEDDING_BATCH_SIZE = 6

_PARSE_ERROR_CONTENT_SNIPPET_MAX = 500


def thinking_kwargs(model: str, reasoning: bool | None) -> dict[str, Any]:
    """Return LiteLLM kwargs that toggle provider thinking/reasoning mode.

    DeepSeek v4 exposes a ``thinking.type`` switch (``enabled`` / ``disabled``)
    via OpenAI SDK ``extra_body``. The default for ``deepseek-v4-flash`` is
    ``enabled``, which spends extra completion tokens on chain-of-thought
    before producing the answer. We expose this to call sites as a simple
    ``reasoning: bool | None`` flag so each agent can pick the mode that
    matches its volume / quality tradeoff.

    Returns an empty dict for non-DeepSeek models, since sending
    ``extra_body={"thinking": ...}`` to OpenAI / Anthropic / etc. is either
    silently ignored or rejected depending on the provider. Passing
    ``reasoning=None`` likewise returns ``{}`` and leaves the provider's
    own default in place.
    """
    if reasoning is None:
        return {}
    if not model.startswith("deepseek/"):
        return {}
    return {"extra_body": {"thinking": {"type": "enabled" if reasoning else "disabled"}}}


async def chat_completion(
    *,
    messages: list[dict[str, Any]],
    response_format: type | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    tools: list[dict] | None = None,
    reasoning: bool | None = None,
) -> Any:
    """Run a chat completion via LiteLLM with structured output support.

    Returns the raw litellm response object. Callers should access
    ``response.choices[0].message`` for the result. When ``response_format``
    is a Pydantic ``BaseModel`` subclass, ``message.parsed`` is populated
    with a validated instance (LiteLLM only returns the JSON string in
    ``content`` and does not set ``parsed`` itself).

    ``reasoning`` toggles DeepSeek thinking mode (see ``thinking_kwargs``).
    Pass ``True`` to opt into chain-of-thought, ``False`` to force a direct
    answer, or leave ``None`` to use the provider's default.
    """
    resolved_model = model or settings.litellm_model
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": settings.llm_timeout_seconds,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    kwargs.update(thinking_kwargs(resolved_model, reasoning))
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        await _raise_provider_limit_if_match(exc, provider=kwargs["model"])
        raise

    if response_format is not None and _is_pydantic_model(response_format):
        any_parsed = False
        last_failed_content: str | None = None
        for choice in response.choices:
            message = choice.message
            content = getattr(message, "content", None)
            if not content:
                continue
            try:
                parsed = response_format.model_validate_json(content)
            except Exception:
                logger.warning(
                    "Failed to parse structured LLM output as %s; content snippet=%r",
                    response_format,
                    content[:_PARSE_ERROR_CONTENT_SNIPPET_MAX],
                )
                last_failed_content = content
                continue
            any_parsed = True
            try:
                message.parsed = parsed
            except (AttributeError, ValueError):
                object.__setattr__(message, "parsed", parsed)

        if not any_parsed:
            snippet = (last_failed_content or "")[:_PARSE_ERROR_CONTENT_SNIPPET_MAX]
            raise StructuredOutputParseError(
                f"LLM returned content that failed to parse as "
                f"{response_format.__name__} for all {len(response.choices)} "
                f"choice(s); content snippet={snippet!r}"
            )

    return response


def _is_pydantic_model(candidate: Any) -> bool:
    from pydantic import BaseModel

    return isinstance(candidate, type) and issubclass(candidate, BaseModel)


def _normalize_embedding_text(content: str) -> str:
    normalized = " ".join(content.split())
    return normalized[:EMBEDDING_MAX_CHARS]


@with_llm_retry()
async def embed_text(content: str) -> list[float]:
    """Embed a single text string via LiteLLM."""
    kwargs: dict[str, Any] = {
        "model": settings.litellm_embedding_model,
        "input": [_normalize_embedding_text(content)],
        "dimensions": settings.embedding_dimensions,
        "timeout": settings.llm_timeout_seconds,
    }
    try:
        response = await litellm.aembedding(**kwargs)
    except Exception as exc:
        await _raise_provider_limit_if_match(exc, provider=settings.litellm_embedding_model)
        raise
    return response.data[0]["embedding"]


async def embed_texts(contents: list[str]) -> list[list[float]]:
    """Embed multiple texts in batches with per-item fallback on error."""
    if not contents:
        return []

    normalized = [_normalize_embedding_text(c) for c in contents]
    embeddings: list[list[float]] = []

    for i in range(0, len(normalized), EMBEDDING_BATCH_SIZE):
        batch = normalized[i : i + EMBEDDING_BATCH_SIZE]
        try:
            kwargs: dict[str, Any] = {
                "model": settings.litellm_embedding_model,
                "input": batch,
                "dimensions": settings.embedding_dimensions,
                "timeout": settings.llm_timeout_seconds,
            }
            response = await litellm.aembedding(**kwargs)
            embeddings.extend(item["embedding"] for item in response.data)
        except ProviderLimitError:
            raise
        except Exception as exc:
            await _raise_provider_limit_if_match(exc, provider=settings.litellm_embedding_model)
            logger.exception(
                "Batch embedding failed; retrying per-item for batch size=%d",
                len(batch),
            )
            for text in batch:
                embeddings.append(await embed_text(text))

    return embeddings


async def _raise_provider_limit_if_match(exc: BaseException, *, provider: str) -> None:
    """Convert a provider usage failure into ProviderLimitError and notify admin.

    Inlined here rather than at the Celery task boundary so every LLM
    call site (conversational agent, agents, embeddings) participates
    in admin alerting and produces the same typed exception that the
    task-level retry handler keys off of.
    """
    limit_err = classify_litellm_error(exc, provider=provider)
    if limit_err is None:
        return
    from news_service.services.admin_alerts import notify_provider_limit

    try:
        await notify_provider_limit(limit_err)
    except Exception:
        logger.exception("Admin alert delivery failed (non-fatal)")
    raise limit_err from exc
