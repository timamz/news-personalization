import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TimezoneCandidateResponse(BaseModel):
    label: str
    timezone: str
    local_time: datetime


class TimezoneResolveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


class TimezoneResolveResponse(BaseModel):
    status: Literal["resolved", "ambiguous", "not_found"]
    candidates: list[TimezoneCandidateResponse]


class UserUpdate(BaseModel):
    timezone: str | None = Field(default=None, min_length=1, max_length=255)
    delivery_webhook_url: str | None = Field(default=None, min_length=1, max_length=2048)


class UserResponse(BaseModel):
    id: uuid.UUID
    api_key: str
    timezone: str | None
    delivery_webhook_url: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
