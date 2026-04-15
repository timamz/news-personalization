import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.schedule_parser import parse_schedule_preference
from news_service.api.dependencies import get_current_user
from news_service.core.concurrency import discovery_semaphore, preview_semaphore
from news_service.db.session import get_session
from news_service.db.vector_store import embed_text
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.models.user_spec import UserSpecSections, render_user_spec
from news_service.schemas.conversation import FinalizedSubscriptionConfig
from news_service.schemas.subscription import (
    RecentEventAcknowledgeRequest,
    ScheduleParseRequest,
    ScheduleParseResponse,
    SubscriptionCreate,
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
from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddits,
    normalize_reddit_subreddit,
)
from news_service.services.scheduler import parse_cron_to_celery
from news_service.services.source_display import source_display_name
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
        select(Source.url)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
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


def _validate_create_payload(
    payload: SubscriptionCreate,
    user: User,
) -> dict[str, Any]:
    """Validate subscription creation payload and return resolved parameters.

    Raises HTTPException on validation failure so errors are returned as proper
    HTTP status codes *before* the streaming response starts.
    """
    prompt_channels = extract_telegram_channels(payload.prompt)
    prompt_subreddits = extract_reddit_subreddits(payload.prompt)
    prompt_twitter_accounts = extract_twitter_accounts(payload.prompt)
    explicit_channels, explicit_subreddits, explicit_twitter_accounts = _normalize_fixed_sources(
        payload.fixed_telegram_channels,
        payload.fixed_reddit_subreddits,
        payload.fixed_twitter_accounts,
    )
    telegram_channels = explicit_channels or prompt_channels
    reddit_subreddits = explicit_subreddits or prompt_subreddits
    twitter_accounts = explicit_twitter_accounts or prompt_twitter_accounts
    include_discovered_sources = (
        payload.include_discovered_sources
        if payload.include_discovered_sources is not None
        else not bool(telegram_channels or reddit_subreddits or twitter_accounts)
    )
    delivery_mode = payload.delivery_mode or "digest"
    schedule_cron = payload.schedule_cron_override
    if delivery_mode == "event" or payload.manual_only:
        schedule_cron = None
    schedule_cron = _validated_schedule_or_422(schedule_cron)
    _ensure_user_timezone_for_schedule(user, schedule_cron)
    digest_language = _normalized_digest_language(payload.digest_language_override) or "en"
    return {
        "telegram_channels": telegram_channels,
        "reddit_subreddits": reddit_subreddits,
        "twitter_accounts": twitter_accounts,
        "include_discovered_sources": include_discovered_sources,
        "delivery_mode": delivery_mode,
        "schedule_cron": schedule_cron,
        "digest_language": digest_language,
    }


async def _create_subscription_streaming(
    payload: SubscriptionCreate,
    user: User,
    session: AsyncSession,
    validated: dict[str, Any],
) -> AsyncGenerator[dict[str, Any], None]:
    """Run subscription creation and yield NDJSON status events."""
    telegram_channels: list[str] = validated["telegram_channels"]
    reddit_subreddits: list[str] = validated["reddit_subreddits"]
    twitter_accounts: list[str] = validated["twitter_accounts"]
    include_discovered_sources: bool = validated["include_discovered_sources"]
    delivery_mode: str = validated["delivery_mode"]
    schedule_cron: str | None = validated["schedule_cron"]
    digest_language: str = validated["digest_language"]

    # --- embedding ---
    yield {"event": "status", "status_key": "status_analyzing"}
    topic_embedding = await _subscription_prompt_embedding(payload.prompt)

    subscription = Subscription(
        user_id=user.id,
        raw_prompt=payload.prompt,
        topic_embedding=topic_embedding,
        user_spec=render_user_spec(UserSpecSections(topic=payload.prompt)),
        delivery_mode=delivery_mode,
        schedule_cron=schedule_cron,
        format_instructions=payload.format_instructions or "brief summary",
        digest_language=digest_language,
        delivery_webhook_url=payload.delivery_webhook_url,
    )
    session.add(subscription)
    await session.flush()

    # --- fixed sources ---
    selected_sources: dict[uuid.UUID, Source] = {}
    user_specified_source_ids: set[uuid.UUID] = set()
    has_fixed = bool(telegram_channels or reddit_subreddits or twitter_accounts)
    if has_fixed:
        yield {"event": "status", "status_key": "status_registering_sources"}
    for identifiers, kind in [
        (telegram_channels, "telegram_channel"),
        (reddit_subreddits, "reddit_subreddit"),
        (twitter_accounts, "twitter_account"),
    ]:
        if identifiers:
            for source in await ensure_source_coverage(session, identifiers, kind):
                selected_sources[source.id] = source
                user_specified_source_ids.add(source.id)

    # --- source discovery (the slow part, concurrency-limited) ---
    if include_discovered_sources:
        yield {"event": "status", "status_key": "status_discovering_sources"}

        async with discovery_semaphore:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            discovery_task = asyncio.create_task(
                ensure_prompt_coverage(session, payload.prompt, topic_embedding, status_queue=queue)
            )

            # Drain status events from the queue while discovery runs
            while not discovery_task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield event
                except TimeoutError:
                    continue

            # Drain remaining events
            while not queue.empty():
                yield queue.get_nowait()

            # Get result (re-raises on failure)
            try:
                discovered_sources = await discovery_task
            except Exception:
                logger.exception("Source discovery failed during streaming creation")
                yield {"event": "error", "detail": "Source discovery failed"}
                return

        for source in discovered_sources:
            selected_sources[source.id] = source

    # --- finalize ---
    if not selected_sources:
        yield {"event": "error", "detail": "No sources were resolved for this subscription"}
        return

    for source_id in selected_sources:
        session.add(
            SubscriptionSource(
                subscription_id=subscription.id,
                source_id=source_id,
                is_user_specified=source_id in user_specified_source_ids,
            )
        )

    await session.commit()
    await session.refresh(subscription)

    logger.info(
        "Created subscription %s for user %s (streaming)",
        subscription.id,
        user.id,
        extra={"subscription_id": str(subscription.id), "user_id": str(user.id)},
    )
    yield {
        "event": "done",
        "subscription": SubscriptionResponse.model_validate(subscription).model_dump(mode="json"),
    }


@router.post("/stream")
async def create_subscription_stream(
    payload: SubscriptionCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Create a subscription with streaming status updates (NDJSON)."""
    validated = _validate_create_payload(payload, user)

    async def generate() -> AsyncGenerator[str, None]:
        async for event in _create_subscription_streaming(payload, user, session, validated):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


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

    selected_sources: dict[uuid.UUID, Source] = {}
    for identifiers, kind in [
        (added_telegram_channels, "telegram_channel"),
        (added_reddit_subreddits, "reddit_subreddit"),
        (added_twitter_accounts, "twitter_account"),
    ]:
        if identifiers:
            for source in await ensure_source_coverage(session, identifiers, kind):
                selected_sources[source.id] = source

    for source_id in selected_sources:
        session.add(
            SubscriptionSource(
                subscription_id=subscription.id,
                source_id=source_id,
                is_user_specified=True,
            )
        )

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
    return list(result.scalars().all())


@router.post("/{subscription_id}/edit/apply-config", response_model=SubscriptionResponse)
async def apply_subscription_config(
    subscription_id: str,
    payload: FinalizedSubscriptionConfig,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Subscription:
    """Apply a complete FinalizedSubscriptionConfig to an existing subscription.

    Updates all subscription fields and reconciles linked sources (adds new, removes old).
    """
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

    # Update format and language
    subscription.format_instructions = payload.format_instructions.strip()
    subscription.digest_language = payload.digest_language

    # Update schedule
    if payload.delivery_mode == "event" or payload.manual_only:
        subscription.schedule_cron = None
    else:
        subscription.schedule_cron = _validated_schedule_or_422(payload.schedule_cron)

    subscription.delivery_mode = payload.delivery_mode

    # Reconcile sources
    telegram_channels, reddit_subreddits, twitter_accounts = _normalize_fixed_sources(
        payload.fixed_telegram_channels,
        payload.fixed_reddit_subreddits,
        payload.fixed_twitter_accounts,
    )
    desired_urls: set[str] = set()
    for channel in telegram_channels:
        desired_urls.add(build_telegram_channel_url(channel))
    for subreddit in reddit_subreddits:
        desired_urls.add(build_reddit_subreddit_url(subreddit))
    for account in twitter_accounts:
        desired_urls.add(build_twitter_account_url(account))

    current_urls = await _existing_subscription_source_urls(session, subscription.id)

    # Remove sources that are no longer desired
    urls_to_remove = current_urls - desired_urls
    if urls_to_remove:
        source_ids_result = await session.execute(
            select(Source.id).where(Source.url.in_(urls_to_remove))
        )
        source_ids_to_remove = [row[0] for row in source_ids_result.all()]
        if source_ids_to_remove:
            links_result = await session.execute(
                select(SubscriptionSource).where(
                    SubscriptionSource.subscription_id == subscription.id,
                    SubscriptionSource.source_id.in_(source_ids_to_remove),
                )
            )
            for link in links_result.scalars().all():
                await session.delete(link)

    # Add new sources
    new_telegram = [
        ch for ch in telegram_channels if build_telegram_channel_url(ch) not in current_urls
    ]
    new_reddit = [
        sub for sub in reddit_subreddits if build_reddit_subreddit_url(sub) not in current_urls
    ]
    new_twitter = [
        acc for acc in twitter_accounts if build_twitter_account_url(acc) not in current_urls
    ]

    new_sources: dict[uuid.UUID, Source] = {}
    for identifiers, kind in [
        (new_telegram, "telegram_channel"),
        (new_reddit, "reddit_subreddit"),
        (new_twitter, "twitter_account"),
    ]:
        if identifiers:
            for source in await ensure_source_coverage(session, identifiers, kind):
                new_sources[source.id] = source

    for source_id in new_sources:
        session.add(
            SubscriptionSource(
                subscription_id=subscription.id,
                source_id=source_id,
                is_user_specified=True,
            )
        )

    await session.commit()
    await session.refresh(subscription)

    logger.info(
        "Applied config edit to subscription %s for user %s",
        subscription.id,
        user.id,
        extra={"subscription_id": str(subscription.id), "user_id": str(user.id)},
    )
    return subscription


async def _recent_events_preview_streaming(
    subscription: Subscription,
    session: AsyncSession,
) -> AsyncGenerator[dict[str, Any], None]:
    """Poll subscription sources, then build a recent events preview. Yields NDJSON."""
    from news_service.tasks.poll_feeds import _poll_single_source

    # Load sources for this subscription
    source_result = await session.execute(
        select(Source)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    sources = list(source_result.scalars().all())

    if not sources:
        yield {"event": "done", "preview": None}
        return

    async with preview_semaphore:
        # Pre-initialize session.info keys that _poll_single_source reads/writes
        session.info["event_source_ids"] = set()
        session.info["event_item_ids"] = []

        for src in sources:
            display = source_display_name(src)
            yield {"event": "status", "status_key": "status_checking_source", "source": display}
            try:
                await _poll_single_source(session, src)
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception(
                    "Failed to poll source %s during recent events preview",
                    src.url,
                    extra={"source_id": str(src.id)},
                )

        yield {"event": "status", "status_key": "status_looking_for_events"}
        try:
            preview = await build_recent_events_preview_for_subscription(
                session,
                subscription,
                lookback_days=7,
            )
        except Exception:
            logger.exception(
                "Failed to build recent events preview for subscription %s",
                subscription.id,
            )
            preview = None

    if preview is None:
        yield {"event": "done", "preview": None}
    else:
        yield {
            "event": "done",
            "preview": {
                "news_item_ids": [str(item_id) for item_id in preview.news_item_ids],
                "subject": preview.subject,
                "body": preview.body,
            },
        }


@router.post("/{subscription_id}/recent-events/stream")
async def recent_events_preview_stream(
    subscription_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Poll sources and build a recent events preview with streaming status updates."""
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

    async def generate() -> AsyncGenerator[str, None]:
        async for event in _recent_events_preview_streaming(subscription, session):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


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
    try:
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
    except Exception:
        await session.rollback()


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
        linked_source_ids = [link.source_id for link in source_links]
        sources_result = await session.execute(
            select(Source).where(Source.id.in_(linked_source_ids))
        )
        for src in sources_result.scalars().all():
            src.subscriber_count = max(src.subscriber_count - 1, 0)
            if src.subscriber_count == 0:
                src.is_active = False

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
