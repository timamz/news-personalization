import uuid
from datetime import datetime

from pydantic import BaseModel


class UserResponse(BaseModel):
    id: uuid.UUID
    api_key: str
    created_at: datetime

    model_config = {"from_attributes": True}
