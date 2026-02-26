import secrets

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.db.session import get_session
from news_service.models.user import User
from news_service.schemas.user import UserResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    session: AsyncSession = Depends(get_session),
) -> User:
    user = User(api_key=secrets.token_urlsafe(32))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
