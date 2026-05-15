"""Server-side share-token store for handing a subscription to another user.

The conversational agent has two tools, ``share_subscription`` and
``import_shared_subscription``, that together implement a frontend-
agnostic copy hand-off:

1. User A asks to share subscription X. The agent calls
   ``share_subscription``; this module mints a short opaque token,
   stores ``{owner_user_id, subscription_id, created_at}`` under
   ``share:sub:{token}`` in Redis with a TTL, and returns the token.
   The agent surfaces the token verbatim so the frontend can copy it
   into a deep link or a chat message.
2. User B pastes the token to their own agent, which calls
   ``import_shared_subscription`` -- this module atomically consumes
   (``GETDEL``) the record. The agent then copies the source
   subscription into a fresh row owned by User B.

The backend has no notion of Telegram usernames or any other frontend
identity, so we never address the recipient by name -- the token IS
the bearer credential. One-shot consume + a 7-day TTL keeps the blast
radius small if a token leaks.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from news_service.core.redis import get_redis_client

logger = logging.getLogger(__name__)

_KEY_PREFIX = "share:sub:"


@dataclass(frozen=True)
class PendingShare:
    """One outstanding share-token record waiting to be redeemed."""

    token: str
    owner_user_id: str
    subscription_id: str
    created_at: str


def _key(token: str) -> str:
    return f"{_KEY_PREFIX}{token}"


async def create(
    *,
    owner_user_id: str,
    subscription_id: str,
    ttl_seconds: int,
) -> str:
    """Mint a fresh share token and store the pending hand-off under it.

    Returns the token string. The pending record self-expires after
    ``ttl_seconds`` (typically 7 days). The token uses ``token_urlsafe(12)``
    -- 16 url-safe characters, ~96 bits of entropy: enough to make
    guessing infeasible while keeping the string short enough to paste
    by hand.
    """
    token = secrets.token_urlsafe(12)
    payload = {
        "owner_user_id": owner_user_id,
        "subscription_id": subscription_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await get_redis_client().set(_key(token), json.dumps(payload), ex=ttl_seconds)
    return token


def _decode(token: str, raw: object) -> PendingShare | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(raw)  # type: ignore[arg-type]
    except (json.JSONDecodeError, TypeError):
        return None
    owner = payload.get("owner_user_id")
    sub_id = payload.get("subscription_id")
    if not owner or not sub_id:
        return None
    return PendingShare(
        token=token,
        owner_user_id=str(owner),
        subscription_id=str(sub_id),
        created_at=str(payload.get("created_at") or ""),
    )


async def consume(token: str) -> PendingShare | None:
    """Atomically look up and delete a pending share.

    Single-shot: a successful consume removes the record, so a second
    redeem returns ``None``. This stops a leaked token from being
    imported twice. Unlike the confirmation nonces, share tokens are
    deliberately NOT user-scoped on lookup: anyone holding the token
    may redeem it, which is the whole point of the share. The caller
    is responsible for enforcing the "no self-import" rule against
    the owner_user_id returned here.
    """
    if not token:
        return None
    raw = await get_redis_client().getdel(_key(token))
    return _decode(token, raw)
