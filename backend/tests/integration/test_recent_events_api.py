import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.agents.event import RecentEventsPreviewDecision
from news_service.db.session import async_session_factory
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from tests.integration.helpers import create_subscription_via_stream, parse_ndjson_done_event

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_subscription(
    api_client: AsyncClient,
    mocker,
    *,
    delivery_mode: str = "event",
) -> tuple[str, str, uuid.UUID]:
    source_ref: dict[str, uuid.UUID] = {}

    async def fake_ensure_prompt_coverage(session, raw_prompt, prompt_embedding):  # noqa: ANN001
        assert raw_prompt == "Notify me about upcoming events"
        assert prompt_embedding == [2.0] * 1536
        src = Source(
            url="https://example.com/events.xml",
            title="Events Feed",
            source_description=f"Events Feed ({raw_prompt})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(src)
        await session.flush()
        source_ref["id"] = src.id
        return [src]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    sub = await create_subscription_via_stream(
        api_client,
        api_key,
        {
            "prompt": "Notify me about upcoming events",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": delivery_mode,
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    return api_key, sub["id"], source_ref["id"]


async def test_recent_events_returns_last_week_previews(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id, source_id = await _create_subscription(api_client, mocker)
    now = datetime.now(UTC)

    async with async_session_factory() as session:
        session.add_all(
            [
                NewsItem(
                    source_id=source_id,
                    headline="New concert announced",
                    body="A new concert was announced for next month.",
                    url="https://example.com/events/recent",
                    source="Events Feed",
                    published_at=now - timedelta(days=2),
                    fetched_at=now - timedelta(days=2),
                ),
                NewsItem(
                    source_id=source_id,
                    headline="Old concert announced",
                    body="This was announced more than a week ago.",
                    url="https://example.com/events/old",
                    source="Events Feed",
                    published_at=now - timedelta(days=9),
                    fetched_at=now - timedelta(days=9),
                ),
            ]
        )
        await session.commit()

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/stream",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    done_event = parse_ndjson_done_event(response.text)
    assert done_event is not None
    preview = done_event["preview"]
    assert preview is not None
    assert preview["news_item_ids"]
    assert preview["subject"] == "Recent events you may have missed"
    assert "Title: New concert announced" in preview["body"]
    assert "A new concert was announced for next month." in preview["body"]


async def test_recent_events_deduplicates_same_headline_posts(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id, source_id = await _create_subscription(api_client, mocker)
    now = datetime.now(UTC)

    async with async_session_factory() as session:
        session.add_all(
            [
                NewsItem(
                    source_id=source_id,
                    headline="Intellectual club announced",
                    body="A lecture by Stanislav Drobyshevsky and a book presentation.",
                    url="https://example.com/events/club-1",
                    source="Events Feed",
                    published_at=now - timedelta(days=1),
                    fetched_at=now - timedelta(days=1),
                ),
                NewsItem(
                    source_id=source_id,
                    headline="Intellectual club announced",
                    body="A lecture by Stanislav Drobyshevsky and a book presentation.",
                    url="https://example.com/events/club-2",
                    source="Events Feed",
                    published_at=now - timedelta(days=2),
                    fetched_at=now - timedelta(days=2),
                ),
            ]
        )
        await session.commit()

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/stream",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    done_event = parse_ndjson_done_event(response.text)
    assert done_event is not None
    preview = done_event["preview"]
    assert preview is not None
    assert len(preview["news_item_ids"]) == 1


async def test_recent_events_filters_strict_subscription_with_preview_renderer(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id, source_id = await _create_subscription(
        api_client,
        mocker,
    )
    now = datetime.now(UTC)

    from unittest.mock import AsyncMock

    async def _preview_renderer(  # noqa: ANN001
        *,
        raw_prompt,
        target_language,
        lookback_days,
        candidate_events,
        recent_notifications,
    ):
        del raw_prompt, target_language, lookback_days, recent_notifications
        selected_item_id = ""
        selected_entry = ""
        for entry in candidate_events:
            if "Stanislav Drobyshevsky" not in entry:
                continue
            selected_entry = entry
            for line in entry.splitlines():
                if line.startswith("ID: "):
                    selected_item_id = line.removeprefix("ID: ").strip()
                    break
        return RecentEventsPreviewDecision(
            selected_item_ids=[selected_item_id],
            subject="Recent events you may have missed",
            body=selected_entry,
        )

    preview_renderer = mocker.patch(
        "news_service.services.event_notifications.render_recent_events_preview",
        new=AsyncMock(side_effect=_preview_renderer),
    )

    async with async_session_factory() as session:
        session.add_all(
            [
                NewsItem(
                    source_id=source_id,
                    headline="Lecture by another person",
                    body="A different speaker was announced for next week.",
                    url="https://example.com/events/other",
                    source="Events Feed",
                    published_at=now - timedelta(days=1),
                    fetched_at=now - timedelta(days=1),
                ),
                NewsItem(
                    source_id=source_id,
                    headline="Stanislav Drobyshevsky lecture announced",
                    body="A lecture by Stanislav Drobyshevsky was announced for next week.",
                    url="https://example.com/events/drobyshevsky",
                    source="Events Feed",
                    published_at=now - timedelta(days=2),
                    fetched_at=now - timedelta(days=2),
                ),
            ]
        )
        await session.commit()

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/stream",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    done_event = parse_ndjson_done_event(response.text)
    assert done_event is not None
    preview = done_event["preview"]
    assert preview is not None
    assert len(preview["news_item_ids"]) == 1
    assert preview["subject"] == "Recent events you may have missed"
    assert "Stanislav Drobyshevsky lecture announced" in preview["body"]
    preview_renderer.assert_awaited_once()


async def test_recent_events_rejects_digest_subscription(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id, _source_id = await _create_subscription(
        api_client,
        mocker,
        delivery_mode="digest",
    )

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/stream",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 409
    assert (
        response.json()["detail"]
        == "Recent events preview is available only for event subscriptions"
    )


async def test_acknowledge_recent_events_marks_items_as_sent(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id, source_id = await _create_subscription(api_client, mocker)
    now = datetime.now(UTC)
    first_item_id = uuid.uuid4()
    second_item_id = uuid.uuid4()

    async with async_session_factory() as session:
        session.add_all(
            [
                NewsItem(
                    id=first_item_id,
                    source_id=source_id,
                    headline="Concert one announced",
                    body="The first concert was announced for next month.",
                    url="https://example.com/events/ack-1",
                    source="Events Feed",
                    published_at=now - timedelta(days=2),
                    fetched_at=now - timedelta(days=2),
                ),
                NewsItem(
                    id=second_item_id,
                    source_id=source_id,
                    headline="Concert two announced",
                    body="The second concert was announced for next month.",
                    url="https://example.com/events/ack-2",
                    source="Events Feed",
                    published_at=now - timedelta(days=1),
                    fetched_at=now - timedelta(days=1),
                ),
            ]
        )
        await session.commit()

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/acknowledge",
        headers={"X-API-Key": api_key},
        json={"news_item_ids": [str(first_item_id), str(second_item_id)]},
    )

    assert response.status_code == 204

    retry_response = await api_client.post(
        f"/subscriptions/{subscription_id}/recent-events/acknowledge",
        headers={"X-API-Key": api_key},
        json={"news_item_ids": [str(first_item_id), str(second_item_id)]},
    )

    assert retry_response.status_code == 204

    async with async_session_factory() as session:
        sent_result = await session.execute(
            select(SentItem).where(SentItem.news_item_id.in_([first_item_id, second_item_id]))
        )
        sent_rows = list(sent_result.scalars().all())
        assert len(sent_rows) == 2
