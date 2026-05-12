"""Cross-provider failure classification for LLM and search providers.

Background tasks need a stable way to recognize the small set of upstream
failures that warrant (a) notifying the operator and (b) retrying the
whole task hours later instead of dropping the work. Provider SDKs raise
their own exception hierarchies (litellm.RateLimitError, httpx errors,
plain ValueError from JSON envelopes, ...), so we centralize the
"is this a usage / balance / auth problem?" decision here.

Raise sites: ``core.llm`` for LiteLLM completions and embeddings,
``services.search`` for the Yandex Search API. Handle sites: the
Celery task wrappers in ``tasks/`` that decide to ``self.retry`` on
ProviderLimitError instead of failing immediately.
"""

from __future__ import annotations

import logging
from typing import Literal

import litellm

logger = logging.getLogger(__name__)

ProviderKind = Literal["balance", "rate_limit", "auth", "timeout", "connection", "unknown"]

_BALANCE_KEYWORDS: tuple[str, ...] = (
    "insufficient_user_quota",
    "insufficient_quota",
    "insufficient balance",
    "insufficient funds",
    "недостаточно средств",
    "quota exceeded",
    "billing",
    "payment required",
)


class ProviderLimitError(Exception):
    """Raised when an upstream LLM or search provider rejects a call for a usage reason.

    ``provider`` is a short identifier (e.g. the LiteLLM model string or
    ``"yandex_search"``) that the operator alert surfaces verbatim.
    ``kind`` distinguishes balance / rate_limit / auth so the message
    can be precise. ``original`` is the raw exception (kept for logs).
    """

    def __init__(
        self,
        *,
        provider: str,
        kind: ProviderKind,
        message: str,
        original: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.kind = kind
        self.message = message
        self.original = original
        super().__init__(f"[{provider}] {kind}: {message}")


def classify_litellm_error(exc: BaseException, *, provider: str) -> ProviderLimitError | None:
    """Map a litellm exception to ProviderLimitError if it represents a usage limit.

    Returns ``None`` for transient errors that the LLM retry decorator
    already handles (timeouts, generic connection blips), so the caller
    can re-raise them unchanged.
    """
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, litellm.AuthenticationError):
        return ProviderLimitError(provider=provider, kind="auth", message=message, original=exc)
    if isinstance(exc, litellm.RateLimitError):
        if _matches_any(lowered, _BALANCE_KEYWORDS):
            return ProviderLimitError(
                provider=provider, kind="balance", message=message, original=exc
            )
        return ProviderLimitError(
            provider=provider, kind="rate_limit", message=message, original=exc
        )
    if isinstance(exc, litellm.BadRequestError):
        if _matches_any(lowered, _BALANCE_KEYWORDS):
            return ProviderLimitError(
                provider=provider, kind="balance", message=message, original=exc
            )
        return None
    if isinstance(exc, litellm.APIError):
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            if _matches_any(lowered, _BALANCE_KEYWORDS):
                return ProviderLimitError(
                    provider=provider, kind="balance", message=message, original=exc
                )
            return ProviderLimitError(provider=provider, kind="auth", message=message, original=exc)
        if status == 402:
            return ProviderLimitError(
                provider=provider, kind="balance", message=message, original=exc
            )
        if status == 429:
            kind: ProviderKind = (
                "balance" if _matches_any(lowered, _BALANCE_KEYWORDS) else "rate_limit"
            )
            return ProviderLimitError(provider=provider, kind=kind, message=message, original=exc)
        if _matches_any(lowered, _BALANCE_KEYWORDS):
            return ProviderLimitError(
                provider=provider, kind="balance", message=message, original=exc
            )
        return None
    if _matches_any(lowered, _BALANCE_KEYWORDS):
        return ProviderLimitError(provider=provider, kind="balance", message=message, original=exc)
    return None


def classify_search_status(
    *,
    status_code: int,
    body_snippet: str,
    provider: str = "yandex_search",
) -> ProviderLimitError | None:
    """Map a Yandex Search HTTP status to a ProviderLimitError when applicable.

    401/403 with no balance keyword is auth (revoked key, wrong role).
    402 or 401/403 carrying a balance keyword is balance. 429 is
    rate_limit. Other statuses (5xx, 4xx) are not considered usage
    limits and return ``None`` so existing transient handling stays.
    """
    lowered = body_snippet.lower()
    if status_code == 402:
        return ProviderLimitError(
            provider=provider, kind="balance", message=body_snippet or "402 Payment Required"
        )
    if status_code in (401, 403):
        kind: ProviderKind = "balance" if _matches_any(lowered, _BALANCE_KEYWORDS) else "auth"
        return ProviderLimitError(
            provider=provider, kind=kind, message=body_snippet or f"HTTP {status_code}"
        )
    if status_code == 429:
        kind = "balance" if _matches_any(lowered, _BALANCE_KEYWORDS) else "rate_limit"
        return ProviderLimitError(
            provider=provider, kind=kind, message=body_snippet or "429 Too Many Requests"
        )
    return None


def _matches_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)
