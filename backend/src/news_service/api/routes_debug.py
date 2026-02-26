"""Debug endpoints — only for development and testing."""

from fastapi import APIRouter

from news_service.tasks.deliver_digest import deliver_digest
from news_service.tasks.poll_feeds import poll_all_feeds

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/trigger-poll")
async def trigger_poll() -> dict:
    result = poll_all_feeds.delay()
    return {"task_id": result.id, "status": "queued"}


@router.post("/trigger-digest/{subscription_id}")
async def trigger_digest(subscription_id: str) -> dict:
    result = deliver_digest.delay(subscription_id)
    return {"task_id": result.id, "status": "queued"}
