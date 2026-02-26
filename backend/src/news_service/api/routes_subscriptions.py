import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.parser import parse_subscription
from news_service.api.dependencies import get_current_user
from news_service.db.session import get_session
from news_service.models.subscription import Subscription
from news_service.models.user import User
from news_service.schemas.subscription import SubscriptionCreate, SubscriptionResponse
from news_service.services.coverage import ensure_topic_coverage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.post("", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    payload: SubscriptionCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Subscription:
    config = await parse_subscription(payload.prompt)

    subscription = Subscription(
        user_id=user.id,
        raw_prompt=payload.prompt,
        topics=config.topics,
        schedule_cron=config.schedule_cron,
        format_instructions=config.format_instructions,
        delivery_webhook_url=payload.delivery_webhook_url,
    )
    session.add(subscription)
    await session.flush()

    await ensure_topic_coverage(session, config.topics)

    await session.commit()
    await session.refresh(subscription)

    logger.info(
        "Created subscription %s for user %s",
        subscription.id,
        user.id,
        extra={"subscription_id": str(subscription.id), "user_id": str(user.id)},
    )
    return subscription


@router.get("", response_model=list[SubscriptionResponse])
async def list_subscriptions(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Subscription]:
    result = await session.execute(
        select(Subscription).where(
            Subscription.user_id == user.id,
            Subscription.is_active.is_(True),
        )
    )
    return list(result.scalars().all())


@router.delete("/{subscription_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_subscription(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    result = await session.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == user.id,
        )
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")

    subscription.is_active = False
    await session.commit()
