import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.parser import parse_schedule_preference, parse_subscription
from news_service.api.dependencies import get_current_user
from news_service.db.session import get_session
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.schemas.subscription import (
    ScheduleParseRequest,
    ScheduleParseResponse,
    SubscriptionCreate,
    SubscriptionParseRequest,
    SubscriptionParseResponse,
    SubscriptionResponse,
    SubscriptionUpdate,
)
from news_service.services.coverage import ensure_telegram_channel_coverage, ensure_topic_coverage
from news_service.services.telegram import extract_telegram_channels, normalize_telegram_channel
from news_service.tasks.deliver_digest import deliver_digest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.post("/parse", response_model=SubscriptionParseResponse)
async def parse_subscription_prompt(
    payload: SubscriptionParseRequest,
    user: User = Depends(get_current_user),
) -> SubscriptionParseResponse:
    del user
    config = await parse_subscription(payload.prompt)
    return SubscriptionParseResponse(
        topics=config.topics,
        delivery_mode=config.delivery_mode,
        schedule_cron=config.schedule_cron,
        schedule_was_explicit=config.schedule_was_explicit,
        format_instructions=config.format_instructions,
        digest_language=config.digest_language,
    )


@router.post("/parse-schedule", response_model=ScheduleParseResponse)
async def parse_schedule(
    payload: ScheduleParseRequest,
    user: User = Depends(get_current_user),
) -> ScheduleParseResponse:
    del user
    schedule_cron = await parse_schedule_preference(payload.schedule_text)
    return ScheduleParseResponse(schedule_cron=schedule_cron)


@router.post("", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    payload: SubscriptionCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Subscription:
    prompt_channels = extract_telegram_channels(payload.prompt)
    try:
        explicit_channels = [
            normalize_telegram_channel(channel)
            for channel in payload.fixed_telegram_channels
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    # For backward compatibility with older clients that only send a prompt.
    telegram_channels = explicit_channels or prompt_channels
    config = await parse_subscription(payload.prompt)
    include_discovered_sources = (
        payload.include_discovered_sources
        if payload.include_discovered_sources is not None
        else not bool(telegram_channels)
    )
    delivery_mode = payload.delivery_mode or config.delivery_mode
    schedule_cron = (
        payload.schedule_cron_override
        if payload.schedule_cron_override is not None
        else config.schedule_cron
    )
    if delivery_mode == "event" or payload.manual_only:
        schedule_cron = None

    subscription = Subscription(
        user_id=user.id,
        raw_prompt=payload.prompt,
        topics=config.topics,
        delivery_mode=delivery_mode,
        schedule_cron=schedule_cron,
        format_instructions=config.format_instructions,
        digest_language=config.digest_language,
        delivery_webhook_url=payload.delivery_webhook_url,
    )
    session.add(subscription)
    await session.flush()

    selected_sources: dict[uuid.UUID, RssFeed] = {}
    if telegram_channels:
        telegram_sources = await ensure_telegram_channel_coverage(
            session,
            telegram_channels,
            config.topics,
        )
        for source in telegram_sources:
            selected_sources[source.id] = source

    if include_discovered_sources:
        discovered_sources = await ensure_topic_coverage(session, config.topics)
        for source in discovered_sources:
            selected_sources[source.id] = source

    if not selected_sources:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No sources were resolved for this subscription",
        )

    for feed_id in selected_sources:
        session.add(SubscriptionSource(subscription_id=subscription.id, feed_id=feed_id))

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


@router.patch("/{subscription_id}", response_model=SubscriptionResponse)
async def update_subscription(
    subscription_id: str,
    payload: SubscriptionUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Subscription:
    result = await session.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == user.id,
        )
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")
    if not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription is inactive",
        )

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No editable fields were provided",
        )

    if "schedule_cron" in updates:
        if subscription.delivery_mode != "digest" and updates["schedule_cron"] is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Automatic schedule is available only for digest subscriptions",
            )
        subscription.schedule_cron = updates["schedule_cron"]
    if "format_instructions" in updates:
        subscription.format_instructions = updates["format_instructions"]
    if "delivery_webhook_url" in updates:
        subscription.delivery_webhook_url = updates["delivery_webhook_url"]

    await session.commit()
    await session.refresh(subscription)

    logger.info(
        "Updated subscription %s for user %s",
        subscription.id,
        user.id,
        extra={"subscription_id": str(subscription.id), "user_id": str(user.id)},
    )
    return subscription


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

    links_result = await session.execute(
        select(SubscriptionSource).where(SubscriptionSource.subscription_id == subscription.id)
    )
    source_links = list(links_result.scalars().all())
    if source_links:
        feed_ids = [link.feed_id for link in source_links]
        feeds_result = await session.execute(select(RssFeed).where(RssFeed.id.in_(feed_ids)))
        for feed in feeds_result.scalars().all():
            feed.subscriber_count = max(feed.subscriber_count - 1, 0)
            if feed.subscriber_count == 0:
                feed.is_active = False

        for link in source_links:
            await session.delete(link)

    subscription.is_active = False
    await session.commit()


@router.post("/{subscription_id}/send-now", status_code=status.HTTP_202_ACCEPTED)
async def send_subscription_now(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    result = await session.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == user.id,
        )
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")
    if not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription is inactive",
        )
    if subscription.delivery_mode != "digest":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Send now is available only for digest subscriptions",
        )

    task = deliver_digest.delay(str(subscription.id), True)
    logger.info(
        "Queued immediate digest for subscription %s",
        subscription.id,
        extra={"subscription_id": str(subscription.id), "user_id": str(user.id)},
    )
    return {"task_id": task.id, "status": "queued"}
