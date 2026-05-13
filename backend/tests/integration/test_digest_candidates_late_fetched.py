"""Integration test: late-fetched, old-published items survive the digest cutoff.

Background. The digest pipeline excludes already-sent items via
SentItem.news_item_id and gates remaining items by a publish-time cutoff
derived from the last sent_at (or news_item_max_age_days lookback for the
first digest). Before this fix the gate filtered solely on
``coalesce(published_at, fetched_at) >= cutoff``, which silently dropped any
item that was fetched after the previous digest but whose published_at
predates the cutoff -- the typical "late-discovered RSS article" case.

This test reproduces that case end-to-end against the real Postgres bench
DB used by the rest of the integration suite. It seeds three items:

  A. published_at = 36h before now, fetched_at = 30 min ago, never sent
     (the DeepMind-style case the previous digest missed).
  B. published_at = 12h before now, fetched_at = 12h before now, never sent
     (a normal recent item; should always have been a candidate).
  C. published_at = 6h before now, fetched_at = 6h before now, ALREADY sent
     (the candidate query must still skip it).

The cutoff (last_sent_at) is set to 24h before now, so item A's
published_at is older than the cutoff. The expectation: A and B come
back, C does not.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from news_service.agents.digest.candidates import fetch_candidate_items
from news_service.db.session import engine
from news_service.models.news_item import NewsItem
from news_service.models.source import Source


@pytest.mark.asyncio(loop_scope="session")
async def test_late_fetched_item_published_before_last_digest_is_a_candidate() -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    last_sent_at = now - timedelta(hours=24)
    embedding = [0.1] * 1536

    async with factory() as session:
        source = Source(
            url=f"https://test.invalid/{uuid.uuid4().hex}.xml",
            title="late-fetch test source",
            source_description="",
            is_active=True,
        )
        session.add(source)
        await session.flush()

        item_late_fetched = NewsItem(
            source_id=source.id,
            headline=f"old-pub-late-fetch {uuid.uuid4().hex}",
            body="An older article we only just discovered.",
            url=f"https://test.invalid/article/{uuid.uuid4().hex}",
            source="rss",
            published_at=now - timedelta(hours=36),
            fetched_at=now - timedelta(minutes=30),
            embedding=embedding,
        )
        item_recent = NewsItem(
            source_id=source.id,
            headline=f"recent {uuid.uuid4().hex}",
            body="A normal recent article.",
            url=f"https://test.invalid/article/{uuid.uuid4().hex}",
            source="rss",
            published_at=now - timedelta(hours=12),
            fetched_at=now - timedelta(hours=12),
            embedding=embedding,
        )
        item_already_sent = NewsItem(
            source_id=source.id,
            headline=f"already-sent {uuid.uuid4().hex}",
            body="Already delivered in a prior digest.",
            url=f"https://test.invalid/article/{uuid.uuid4().hex}",
            source="rss",
            published_at=now - timedelta(hours=6),
            fetched_at=now - timedelta(hours=6),
            embedding=embedding,
        )
        session.add_all([item_late_fetched, item_recent, item_already_sent])
        await session.commit()

        candidates = await fetch_candidate_items(
            session,
            query_embedding=embedding,
            exclude_ids={item_already_sent.id},
            allowed_source_ids={source.id},
            published_after=last_sent_at,
            fetched_after=last_sent_at,
        )

    returned_ids = {c.id for c in candidates}
    assert (
        item_late_fetched.id in returned_ids
        and item_recent.id in returned_ids
        and (item_already_sent.id not in returned_ids)
    ), (
        "candidate query did not return the late-fetched old-published item or "
        "leaked an already-sent item; "
        f"returned={returned_ids} late_fetched={item_late_fetched.id} "
        f"recent={item_recent.id} sent={item_already_sent.id}"
    )
