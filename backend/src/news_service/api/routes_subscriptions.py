import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.parser import parse_schedule_preference, parse_subscription
from news_service.agents.subscription_edit import propose_subscription_edit
from news_service.api.dependencies import get_current_user
from news_service.db.session import get_session
from news_service.db.vector_store import embed_text
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.schemas.subscription import (
    RecentEventAcknowledgeRequest,
    RecentEventsPreviewResponse,
    ScheduleParseRequest,
    ScheduleParseResponse,
    SubscriptionCreate,
    SubscriptionEditApplyRequest,
    SubscriptionEditProposalRequest,
    SubscriptionEditProposalResponse,
    SubscriptionParseRequest,
    SubscriptionParseResponse,
    SubscriptionResponse,
    SubscriptionSourcesAppendRequest,
    SubscriptionSourcesAppendResponse,
    SubscriptionUpdate,
)
from news_service.services.coverage import (
    ensure_prompt_coverage,
    ensure_source_coverage,
)
from news_service.services.event_notifications import (
    build_recent_events_preview_for_subscription,
)
from news_service.services.prompt_summaries import build_prompt_summary
from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddits,
    normalize_reddit_subreddit,
)
from news_service.services.scheduler import parse_cron_to_celery
from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channels,
    normalize_telegram_channel,
)
from news_service.services.twitter import (
    build_twitter_account_url,
    extract_twitter_accounts,
    normalize_twitter_account,
)
from news_service.tasks.deliver_digest import deliver_digest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


async def _subscription_prompt_embedding(raw_prompt: str) -> list[float]:
    return await embed_text(raw_prompt)


def _canonical_prompt(subscription: Subscription) -> str:
    prompt = subscription.canonical_prompt.strip()
    if prompt:
        return prompt
    return subscription.raw_prompt


def _normalize_fixed_sources(
    telegram_channels: list[str],
    reddit_subreddits: list[str],
    twitter_accounts: list[str],
) -> tuple[list[str], list[str], list[str]]:
    try:
        return (
            _dedupe_strings([normalize_telegram_channel(channel) for channel in telegram_channels]),
            _dedupe_strings(
                [normalize_reddit_subreddit(subreddit) for subreddit in reddit_subreddits]
            ),
            _dedupe_strings([normalize_twitter_account(account) for account in twitter_accounts]),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


async def _existing_subscription_source_urls(
    session: AsyncSession,
    subscription_id: uuid.UUID,
) -> set[str]:
    result = await session.execute(
        select(RssFeed.url)
        .join(SubscriptionSource, SubscriptionSource.feed_id == RssFeed.id)
        .where(SubscriptionSource.subscription_id == subscription_id)
    )
    return {url for url in result.scalars().all()}


def _normalized_digest_language(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower().split("-", maxsplit=1)[0]
    if len(normalized) < 2 or len(normalized) > 16:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid digest language code",
        )
    return normalized


def _validated_schedule_or_422(schedule_cron: str | None) -> str | None:
    if schedule_cron is None:
        return None

    normalized = " ".join(schedule_cron.split())
    try:
        parse_cron_to_celery(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid cron expression: {normalized}",
        ) from exc
    return normalized


def _ensure_user_timezone_for_schedule(user: User, schedule_cron: str | None) -> None:
    if schedule_cron is None or user.timezone is not None:
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Set your timezone before enabling automatic schedules",
    )


def _ensure_prompt_summary(subscription: Subscription) -> str:
    if subscription.prompt_summary.strip():
        return subscription.prompt_summary
    subscription.prompt_summary = build_prompt_summary(_canonical_prompt(subscription))
    return subscription.prompt_summary


@router.post("/parse", response_model=SubscriptionParseResponse)
async def parse_subscription_prompt(
    payload: SubscriptionParseRequest,
    user: User = Depends(get_current_user),
) -> SubscriptionParseResponse:
    del user
    config = await parse_subscription(payload.prompt)
    schedule_cron = _validated_schedule_or_422(config.schedule_cron)
    return SubscriptionParseResponse(
        prompt_summary=config.prompt_summary,
        delivery_mode=config.delivery_mode,
        schedule_cron=schedule_cron,
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
    validated_schedule = _validated_schedule_or_422(schedule_cron)
    if validated_schedule is None:
        raise RuntimeError("Schedule parser returned an empty cron expression")
    return ScheduleParseResponse(schedule_cron=validated_schedule)


@router.post("", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    payload: SubscriptionCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Subscription:
    prompt_channels = extract_telegram_channels(payload.prompt)
    prompt_subreddits = extract_reddit_subreddits(payload.prompt)
    prompt_twitter_accounts = extract_twitter_accounts(payload.prompt)
    explicit_channels, explicit_subreddits, explicit_twitter_accounts = _normalize_fixed_sources(
        payload.fixed_telegram_channels,
        payload.fixed_reddit_subreddits,
        payload.fixed_twitter_accounts,
    )
    # For backward compatibility with older clients that only send a prompt.
    telegram_channels = explicit_channels or prompt_channels
    reddit_subreddits = explicit_subreddits or prompt_subreddits
    twitter_accounts = explicit_twitter_accounts or prompt_twitter_accounts
    config = await parse_subscription(payload.prompt)
    include_discovered_sources = (
        payload.include_discovered_sources
        if payload.include_discovered_sources is not None
        else not bool(telegram_channels or reddit_subreddits or twitter_accounts)
    )
    delivery_mode = payload.delivery_mode or config.delivery_mode
    event_matching_mode = config.event_matching_mode if delivery_mode == "event" else "basic"
    schedule_cron = (
        payload.schedule_cron_override
        if payload.schedule_cron_override is not None
        else config.schedule_cron
    )
    if delivery_mode == "event" or payload.manual_only:
        schedule_cron = None
    schedule_cron = _validated_schedule_or_422(schedule_cron)
    _ensure_user_timezone_for_schedule(user, schedule_cron)
    digest_language = (
        _normalized_digest_language(payload.digest_language_override) or config.digest_language
    )
    raw_prompt_embedding = await _subscription_prompt_embedding(payload.prompt)
    canonical_prompt_embedding = list(raw_prompt_embedding)
    prompt_summary = config.prompt_summary or build_prompt_summary(payload.prompt)
    short_label = config.short_label or prompt_summary[:30]

    subscription = Subscription(
        user_id=user.id,
        raw_prompt=payload.prompt,
        raw_prompt_embedding=raw_prompt_embedding,
        canonical_prompt=payload.prompt,
        canonical_prompt_embedding=canonical_prompt_embedding,
        prompt_summary=prompt_summary,
        short_label=short_label,
        delivery_mode=delivery_mode,
        event_matching_mode=event_matching_mode,
        event_constraints=[],
        schedule_cron=schedule_cron,
        format_instructions=config.format_instructions,
        digest_language=digest_language,
        delivery_webhook_url=payload.delivery_webhook_url,
    )
    session.add(subscription)
    await session.flush()

    selected_sources: dict[uuid.UUID, RssFeed] = {}
    for identifiers, kind in [
        (telegram_channels, "telegram_channel"),
        (reddit_subreddits, "reddit_subreddit"),
        (twitter_accounts, "twitter_account"),
    ]:
        if identifiers:
            for source in await ensure_source_coverage(session, identifiers, kind):
                selected_sources[source.id] = source

    if include_discovered_sources:
        discovered_sources = await ensure_prompt_coverage(
            session,
            payload.prompt,
            canonical_prompt_embedding,
        )
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


@router.post(
    "/{subscription_id}/sources",
    response_model=SubscriptionSourcesAppendResponse,
)
async def append_subscription_sources(
    subscription_id: str,
    payload: SubscriptionSourcesAppendRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SubscriptionSourcesAppendResponse:
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

    telegram_channels, reddit_subreddits, twitter_accounts = _normalize_fixed_sources(
        payload.fixed_telegram_channels,
        payload.fixed_reddit_subreddits,
        payload.fixed_twitter_accounts,
    )
    existing_urls = await _existing_subscription_source_urls(session, subscription.id)

    added_telegram_channels = [
        channel
        for channel in telegram_channels
        if build_telegram_channel_url(channel) not in existing_urls
    ]
    added_reddit_subreddits = [
        subreddit
        for subreddit in reddit_subreddits
        if build_reddit_subreddit_url(subreddit) not in existing_urls
    ]
    added_twitter_accounts = [
        account
        for account in twitter_accounts
        if build_twitter_account_url(account) not in existing_urls
    ]

    selected_sources: dict[uuid.UUID, RssFeed] = {}
    for identifiers, kind in [
        (added_telegram_channels, "telegram_channel"),
        (added_reddit_subreddits, "reddit_subreddit"),
        (added_twitter_accounts, "twitter_account"),
    ]:
        if identifiers:
            for source in await ensure_source_coverage(session, identifiers, kind):
                selected_sources[source.id] = source

    for feed_id in selected_sources:
        session.add(SubscriptionSource(subscription_id=subscription.id, feed_id=feed_id))

    await session.commit()

    added_sources_count = (
        len(added_telegram_channels) + len(added_reddit_subreddits) + len(added_twitter_accounts)
    )
    return SubscriptionSourcesAppendResponse(
        added_telegram_channels=added_telegram_channels,
        added_reddit_subreddits=added_reddit_subreddits,
        added_twitter_accounts=added_twitter_accounts,
        added_sources_count=added_sources_count,
    )


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
    subscriptions = list(result.scalars().all())
    updated = False
    for subscription in subscriptions:
        previous = subscription.prompt_summary
        _ensure_prompt_summary(subscription)
        updated = updated or previous != subscription.prompt_summary
    if updated:
        await session.commit()
    return subscriptions


@router.post(
    "/{subscription_id}/edit/propose",
    response_model=SubscriptionEditProposalResponse,
)
async def propose_subscription_edit_for_subscription(
    subscription_id: str,
    payload: SubscriptionEditProposalRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SubscriptionEditProposalResponse:
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

    canonical_prompt = payload.draft_canonical_prompt or _canonical_prompt(subscription)
    format_instructions = payload.draft_format_instructions or subscription.format_instructions
    return await propose_subscription_edit(
        canonical_prompt=canonical_prompt,
        format_instructions=format_instructions,
        change_request=payload.change_request,
    )


@router.post(
    "/{subscription_id}/edit/apply",
    response_model=SubscriptionResponse,
)
async def apply_subscription_edit(
    subscription_id: str,
    payload: SubscriptionEditApplyRequest,
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

    subscription.canonical_prompt = payload.canonical_prompt.strip()
    subscription.canonical_prompt_embedding = await _subscription_prompt_embedding(
        subscription.canonical_prompt
    )
    subscription.prompt_summary = payload.prompt_summary.strip()
    subscription.format_instructions = payload.format_instructions.strip()

    await session.commit()
    await session.refresh(subscription)
    return subscription


@router.get(
    "/{subscription_id}/recent-events",
    response_model=RecentEventsPreviewResponse | None,
)
async def list_recent_events(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> RecentEventsPreviewResponse | None:
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
    if subscription.delivery_mode != "event":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recent events preview is available only for event subscriptions",
        )

    preview = await build_recent_events_preview_for_subscription(
        session,
        subscription,
        lookback_days=7,
    )
    if preview is None:
        return None
    return RecentEventsPreviewResponse(
        news_item_ids=preview.news_item_ids,
        subject=preview.subject,
        body=preview.body,
    )


@router.post("/{subscription_id}/recent-events/acknowledge", status_code=status.HTTP_204_NO_CONTENT)
async def acknowledge_recent_events(
    subscription_id: str,
    payload: RecentEventAcknowledgeRequest,
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
    if not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription is inactive",
        )
    if subscription.delivery_mode != "event":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recent events preview is available only for event subscriptions",
        )
    requested_item_ids = list(dict.fromkeys(payload.news_item_ids))

    link_result = await session.execute(
        select(SubscriptionSource.feed_id).where(
            SubscriptionSource.subscription_id == subscription.id
        )
    )
    allowed_feed_ids = set(link_result.scalars().all())
    if not allowed_feed_ids:
        return

    items_result = await session.execute(
        select(NewsItem).where(
            NewsItem.id.in_(requested_item_ids),
            NewsItem.feed_id.in_(allowed_feed_ids),
            NewsItem.event_title.is_not(None),
        )
    )
    items = list(items_result.scalars().all())
    valid_item_ids = {item.id for item in items}
    if valid_item_ids != set(requested_item_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="One or more recent event items are invalid for this subscription",
        )

    await session.execute(
        insert(SentItem)
        .values(
            [
                {
                    "subscription_id": subscription.id,
                    "news_item_id": item_id,
                }
                for item_id in requested_item_ids
            ]
        )
        .on_conflict_do_nothing(
            index_elements=["subscription_id", "news_item_id"],
        )
    )

    await session.commit()


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
        validated_schedule = _validated_schedule_or_422(updates["schedule_cron"])
        _ensure_user_timezone_for_schedule(user, validated_schedule)
        subscription.schedule_cron = validated_schedule
    if "format_instructions" in updates:
        subscription.format_instructions = updates["format_instructions"]
    if "delivery_webhook_url" in updates:
        subscription.delivery_webhook_url = updates["delivery_webhook_url"]
    if "digest_language" in updates:
        if updates["digest_language"] is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="digest_language cannot be null",
            )
        subscription.digest_language = _normalized_digest_language(updates["digest_language"])

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


@router.post("/backfill-labels", status_code=status.HTTP_200_OK)
async def backfill_short_labels(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    """Generate short_label for all active subscriptions that don't have one."""
    from news_service.agents.short_label import generate_short_label

    result = await session.execute(
        select(Subscription).where(
            Subscription.user_id == user.id,
            Subscription.is_active.is_(True),
            Subscription.short_label == "",
        )
    )
    subscriptions = list(result.scalars().all())
    updated = 0
    for subscription in subscriptions:
        try:
            label = await generate_short_label(subscription.prompt_summary)
            subscription.short_label = label[:30]
            updated += 1
        except Exception:
            logger.exception(
                "Failed to generate short label for subscription %s",
                subscription.id,
            )
    await session.commit()
    return {"updated": updated}
