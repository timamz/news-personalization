import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User email address")


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    api_key: str
    created_at: datetime

    model_config = {"from_attributes": True}
