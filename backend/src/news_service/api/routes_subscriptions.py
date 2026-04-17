"""REST endpoints that exist beyond the conversational agent's surface.

Subscription CRUD, source management, immediate-send, and listing are
owned entirely by the conversational agent (see
``agents/conversational.py`` tools). This module keeps only the
endpoints the agent cannot drive directly:

- ``POST /subscriptions/{id}/recent-events/stream`` -- poll sources and
  render a short preview of what this event-mode subscription would
  have fired over the lookback window. The tgbot uses it during event
  subscription onboarding.
- ``POST /subscriptions/{id}/recent-events/acknowledge`` -- mark
  recent-event preview items as already-shown so they are not re-sent
  once live notifications kick in.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.api.dependencies import get_current_user
from news_service.core.concurrency import preview_semaphore
from news_service.db.session import get_session
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.schemas.subscription import RecentEventAcknowledgeRequest
from news_service.services.event_notifications import (
    build_recent_events_preview_for_subscription,
)
from news_service.services.source_display import source_display_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


async def _recent_events_preview_streaming(
    subscription: Subscription,
    session: AsyncSession,
) -> AsyncGenerator[dict[str, Any], None]:
    """Poll subscription sources, then build a recent events preview. Yields NDJSON."""
    from news_service.tasks.poll_feeds import _poll_single_source

    source_result = await session.execute(
        select(Source)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    sources = list(source_result.scalars().all())

    if not sources:
        yield {"event": "done", "preview": None}
        return

    async with preview_semaphore:
        session.info["event_source_ids"] = set()
        session.info["event_item_ids"] = []

        for src in sources:
            display = source_display_name(src)
            yield {"event": "status", "status_key": "status_checking_source", "source": display}
            try:
                await _poll_single_source(session, src)
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception(
                    "Failed to poll source %s during recent events preview",
                    src.url,
                    extra={"source_id": str(src.id)},
                )

        yield {"event": "status", "status_key": "status_looking_for_events"}
        try:
            preview = await build_recent_events_preview_for_subscription(
                session,
                subscription,
                lookback_days=7,
            )
        except Exception:
            logger.exception(
                "Failed to build recent events preview for subscription %s",
                subscription.id,
            )
            preview = None

    if preview is None:
        yield {"event": "done", "preview": None}
    else:
        yield {
            "event": "done",
            "preview": {
                "news_item_ids": [str(item_id) for item_id in preview.news_item_ids],
                "subject": preview.subject,
                "body": preview.body,
            },
        }


@router.post("/{subscription_id}/recent-events/stream")
async def recent_events_preview_stream(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Poll sources and build a recent events preview with streaming status updates."""
    result = await session.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == user.id,
        )
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")
    if not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription is inactive",
        )
    if subscription.delivery_mode != "event":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recent events preview is available only for event subscriptions",
        )

    async def generate() -> AsyncGenerator[str, None]:
        async for event in _recent_events_preview_streaming(subscription, session):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/{subscription_id}/recent-events/acknowledge", status_code=status.HTTP_204_NO_CONTENT)
async def acknowledge_recent_events(
    subscription_id: str,
    payload: RecentEventAcknowledgeRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    result = await session.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == user.id,
        )
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")
    if not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription is inactive",
        )
    if subscription.delivery_mode != "event":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recent events preview is available only for event subscriptions",
        )
    requested_item_ids = list(dict.fromkeys(payload.news_item_ids))
    try:
        await session.execute(
            insert(SentItem)
            .values(
                [
                    {
                        "subscription_id": subscription.id,
                        "news_item_id": item_id,
                    }
                    for item_id in requested_item_ids
                ]
            )
            .on_conflict_do_nothing(
                index_elements=["subscription_id", "news_item_id"],
            )
        )
        await session.commit()
    except Exception:
        await session.rollback()
