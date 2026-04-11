"""Provider-agnostic LLM interface via LiteLLM.

Wraps litellm for chat completions and embeddings, enabling transparent
switching between OpenAI, Gemini, Anthropic, self-hosted models, and
any other provider supported by LiteLLM.
"""

import logging
from typing import Any

import httpx
import litellm
from openai import AsyncOpenAI

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

settings = get_settings()

EMBEDDING_MAX_CHARS = 4000
EMBEDDING_BATCH_SIZE = 6

_http_client = httpx.AsyncClient(proxy=settings.proxy_url) if settings.proxy_url else None
_openai_client: AsyncOpenAI | None = None
if settings.proxy_url:
    _openai_client = AsyncOpenAI(http_client=_http_client)


async def chat_completion(
    *,
    messages: list[dict[str, Any]],
    response_format: type | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    tools: list[dict] | None = None,
) -> Any:
    """Run a chat completion via LiteLLM with structured output support.

    Returns the raw litellm response object. Callers should access
    ``response.choices[0].message`` for the result.
    """
    kwargs: dict[str, Any] = {
        "model": model or settings.litellm_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": settings.llm_timeout_seconds,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if _openai_client is not None:
        kwargs["client"] = _openai_client
    return await litellm.acompletion(**kwargs)


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
    }
    if _openai_client is not None:
        kwargs["client"] = _openai_client
    response = await litellm.aembedding(**kwargs)
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
            }
            if _openai_client is not None:
                kwargs["client"] = _openai_client
            response = await litellm.aembedding(**kwargs)
            embeddings.extend(item["embedding"] for item in response.data)
        except Exception:
            logger.exception(
                "Batch embedding failed; retrying per-item for batch size=%d",
                len(batch),
            )
            for text in batch:
                embeddings.append(await embed_text(text))

    return embeddings
