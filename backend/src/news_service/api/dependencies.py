from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.db.session import get_session
from news_service.models.user import User


async def get_current_user(
    x_api_key: str = Header(..., description="API key for authentication"),
    session: AsyncSession = Depends(get_session),
) -> User:
    result = await session.execute(select(User).where(User.api_key == x_api_key))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return user
