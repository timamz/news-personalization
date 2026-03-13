import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.api.dependencies import get_current_user
from news_service.db.session import get_session
from news_service.models.user import User
from news_service.schemas.user import (
    TimezoneCandidateResponse,
    TimezoneResolveRequest,
    TimezoneResolveResponse,
    UserResponse,
    UserUpdate,
)
from news_service.services.timezones import normalize_timezone_name, resolve_timezone

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


@router.get("/me", response_model=UserResponse)
async def get_user_profile(
    user: User = Depends(get_current_user),
) -> User:
    return user


@router.patch("/me", response_model=UserResponse)
async def update_user_profile(
    payload: UserUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> User:
    try:
        user.timezone = normalize_timezone_name(payload.timezone)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await session.commit()
    await session.refresh(user)
    return user


@router.post("/resolve-timezone", response_model=TimezoneResolveResponse)
async def resolve_user_timezone(
    payload: TimezoneResolveRequest,
    user: User = Depends(get_current_user),
) -> TimezoneResolveResponse:
    del user
    resolution = resolve_timezone(payload.query)
    return TimezoneResolveResponse(
        status=resolution.status,
        candidates=[
            TimezoneCandidateResponse(
                label=candidate.label,
                timezone=candidate.timezone,
                local_time=candidate.local_time(),
            )
            for candidate in resolution.candidates
        ],
    )
