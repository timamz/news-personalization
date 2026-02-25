"""Debug endpoints — only for development and testing."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.db.session import get_session
from news_service.tasks.deliver_digest import deliver_digest
from news_service.tasks.poll_feeds import poll_all_feeds

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/trigger-poll")
async def trigger_poll(
    session: AsyncSession = Depends(get_session),  # noqa: ARG001
) -> dict:
    result = poll_all_feeds.delay()
    return {"task_id": result.id, "status": "queued"}


@router.post("/trigger-digest/{subscription_id}")
async def trigger_digest(
    subscription_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: ARG001
) -> dict:
    result = deliver_digest.delay(subscription_id)
    return {"task_id": result.id, "status": "queued"}
