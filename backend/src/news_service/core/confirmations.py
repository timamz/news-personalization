"""Server-side nonce store for destructive / expensive tool confirmations.

Threat being closed: a prompt-only confirmation rule trusts the LLM to
pass ``confirm=True`` only after seeing user agreement. A misaligned or
mis-prompted model can defeat it. Moving confirmation to a Redis-backed
nonce makes the gate hard:

1. On first call, the tool mints a cryptographically random nonce, stores
   ``{user_id, tool_name, args, description}`` under it, and pushes a
   ``requires_confirmation`` event onto the conversation stream.
2. The frontend renders the event as an inline yes/no button. The nonce
   never reaches the LLM; it travels frontend -> button payload -> the
   ``/conversations/confirm`` endpoint.
3. The endpoint re-invokes the same tool with ``confirmation_token``
   set to the nonce. The tool atomically consumes the nonce (Redis
   ``GETDEL``); if it is missing, expired, or belongs to another user,
   the tool refuses.

The LLM cannot fabricate a valid nonce because it has never seen one.
The nonce is one-shot (consumed on use) and has a TTL so abandoned
pending confirmations eventually disappear.

Cross-tenant safety: ``user_id`` is checked on lookup. A nonce minted
for user A cannot be redeemed by user B even if the nonce string leaks.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from typing import Any

from news_service.core.redis import get_redis_client

logger = logging.getLogger(__name__)

_KEY_PREFIX = "confirm:pending:"
_TTL_SECONDS = 600


@dataclass(frozen=True)
class PendingConfirmation:
    """One pending tool execution waiting for a yes/no decision."""

    nonce: str
    user_id: str
    tool_name: str
    args: dict[str, Any]
    description: str


def _key(nonce: str) -> str:
    return f"{_KEY_PREFIX}{nonce}"


async def create(
    *,
    user_id: str,
    tool_name: str,
    args: dict[str, Any],
    description: str,
) -> str:
    """Mint a fresh nonce and store the pending action under it.

    Returns the nonce string. The pending record self-expires after
    ``_TTL_SECONDS`` (10 minutes), generous enough that a user who
    walked away from their phone can still confirm on return.
    """
    nonce = secrets.token_urlsafe(16)
    payload = {
        "user_id": user_id,
        "tool_name": tool_name,
        "args": args,
        "description": description,
    }
    await get_redis_client().set(_key(nonce), json.dumps(payload), ex=_TTL_SECONDS)
    return nonce


def _decode(nonce: str, raw: object, user_id: str) -> PendingConfirmation | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(raw)  # type: ignore[arg-type]
    except (json.JSONDecodeError, TypeError):
        return None
    if payload.get("user_id") != user_id:
        return None
    return PendingConfirmation(
        nonce=nonce,
        user_id=payload["user_id"],
        tool_name=payload["tool_name"],
        args=payload.get("args") or {},
        description=payload.get("description") or "",
    )


async def peek(nonce: str, user_id: str) -> PendingConfirmation | None:
    """Look up a pending confirmation without consuming it."""
    if not nonce:
        return None
    raw = await get_redis_client().get(_key(nonce))
    return _decode(nonce, raw, user_id)


async def consume(nonce: str, user_id: str) -> PendingConfirmation | None:
    """Atomically look up and delete a pending confirmation.

    Single-shot: a successful consume removes the record, so a second
    redeem returns ``None``. This prevents replay attacks where a
    leaked nonce gets used twice.
    """
    if not nonce:
        return None
    raw = await get_redis_client().getdel(_key(nonce))
    return _decode(nonce, raw, user_id)


async def cancel(nonce: str, user_id: str) -> bool:
    """Delete a pending confirmation. Returns True if it existed."""
    pending = await consume(nonce, user_id)
    return pending is not None
