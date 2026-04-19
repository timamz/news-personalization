"""Debug endpoints -- only for development and testing.

These are still exposed in deployed environments, so they require an API
key like the rest of the user surface, and ``trigger-digest`` will only
fire deliveries for subscriptions owned by the caller. Without these
checks any authenticated user could enqueue webhook deliveries for any
subscription by id, and any unauthenticated client could DoS the poll
loop.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.api.dependencies import get_current_user
from news_service.db.session import get_session
from news_service.models.subscription import Subscription
from news_service.models.user import User
from news_service.tasks.deliver_digest import deliver_digest
from news_service.tasks.poll_feeds import poll_all_feeds

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/trigger-poll")
async def trigger_poll(user: User = Depends(get_current_user)) -> dict:
    del user
    result = poll_all_feeds.delay()
    return {"task_id": result.id, "status": "queued"}


@router.post("/trigger-digest/{subscription_id}")
async def trigger_digest(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        sub_uuid = uuid.UUID(subscription_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid subscription_id",
        ) from exc

    owned = await session.execute(
        select(Subscription.id).where(
            Subscription.id == sub_uuid,
            Subscription.user_id == user.id,
        )
    )
    if owned.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="subscription not found",
        )

    result = deliver_digest.delay(subscription_id)
    return {"task_id": result.id, "status": "queued"}
