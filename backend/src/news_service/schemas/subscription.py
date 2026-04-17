import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

type DeliveryMode = Literal["digest", "event"]


class SubscriptionResponse(BaseModel):
    """Read-model for a subscription row.

    Exposed via the recent-events preview response and for agents /
    test fixtures that serialize subscriptions. All mutations go
    through the conversational agent.
    """

    id: uuid.UUID
    user_spec: str
    delivery_mode: DeliveryMode
    schedule_cron: str | None
    digest_language: str
    delivery_webhook_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class RecentEventsPreviewResponse(BaseModel):
    news_item_ids: list[uuid.UUID]
    subject: str
    body: str


class RecentEventAcknowledgeRequest(BaseModel):
    news_item_ids: list[uuid.UUID] = Field(
        ...,
        min_length=1,
        description="Recent event preview items that were actually shown to the user",
    )
