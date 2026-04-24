"""
Pre-flight: verify the e2e-month v2 staleness trigger wiring.

The v2 test's engineered reflector trigger depends on three fragile
pieces all lining up:

1. ``REFLECTOR_SOURCE_STALENESS_DAYS=10`` env override takes effect
   when the digest pipeline reads ``settings.reflector_source_staleness_days``.
2. Polling (``_poll_all_feeds``) actually inserts NewsItem rows with
   ``published_at`` values drawn from the ScenarioItem ``fake_ts``, so
   ``max(NewsItem.published_at)`` computed by ``_load_source_contexts``
   resolves to the right timestamp.
3. With the stale source's last item at simulated day 15 and
   ``CLOCK.now()`` at day 26, the resulting
   ``days_since_last_published`` is ``>= 10`` and
   ``_compute_reflect_reasons`` returns a ``source_staleness`` reason.

This test inlines the exact v2 configuration (``DIGEST_USER_SPEC``,
``DIGEST_SOURCE_UNIVERSE``, body banks, 10-day threshold, day-15
cutoff, day-26 clock), runs one poll cycle + one digest delivery, and
asserts the reason appears. It is deterministic: no LLM flakiness
because we assert only the trigger reason, not the reflector's
removal decision.

Opt-in: costs a handful of cents for one real digest pipeline
invocation. Gate with ``RUN_E2E_MONTH_PREFLIGHT=1``.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.adapters import FakeAdapter
from tests.integration._e2e_month_corpus import (
    DIGEST_RETRIEVAL_QUERY,
    DIGEST_USER_SPEC,
    build_timeline,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E_MONTH_PREFLIGHT") != "1",
    reason=(
        "Pre-flight for e2e-month staleness trigger. Opt-in because it "
        "hits the real LLM provider for one digest pipeline invocation."
    ),
)


_SIM_START = datetime(2026, 5, 1, tzinfo=UTC)
_STALE_DAY = 15
_CLOCK_DAY = 26
_STALENESS_THRESHOLD_DAYS = 10
_WEBHOOK_URL = "https://bench.invalid/webhook/e2e-month-preflight"

_DIGEST_SOURCE_UNIVERSE: list[str] = [
    "https://www.euractiv.com/section/energy/feed/",
    "https://www.politico.eu/section/energy/feed/",
    "https://euobserver.com/feeds/energy.rss",
]


@pytest.mark.asyncio
async def test_e2e_month_staleness_reason_fires_on_day_26(world) -> None:
    """``source_staleness`` appears in reflector reasons when stale source crosses threshold."""
    from news_service.agents.digest import pipeline as pipeline_mod

    prior_threshold = pipeline_mod.settings.reflector_source_staleness_days
    pipeline_mod.settings.reflector_source_staleness_days = _STALENESS_THRESHOLD_DAYS

    try:
        CLOCK.reset_to(_SIM_START)

        items = build_timeline(
            source_urls=_DIGEST_SOURCE_UNIVERSE,
            topic="digest",
            start=_SIM_START,
            days=30,
            items_per_source_per_day=3,
        )
        by_source: dict[str, list] = {}
        for it in items:
            by_source.setdefault(it.source_url, []).append(it)

        stale_source_url = _DIGEST_SOURCE_UNIVERSE[0]
        stale_cutoff = _SIM_START + timedelta(days=_STALE_DAY)
        by_source[stale_source_url] = [
            i for i in by_source[stale_source_url] if i.fake_ts <= stale_cutoff
        ]

        for url in _DIGEST_SOURCE_UNIVERSE:
            world.adapters[url] = FakeAdapter(
                source_url=url,
                items=sorted(by_source.get(url, []), key=lambda x: x.fake_ts),
            )

        from news_service.db.session import async_session_factory
        from news_service.db.vector_store import embed_text
        from news_service.models.source import Source
        from news_service.models.subscription import Subscription
        from news_service.models.subscription_source import SubscriptionSource
        from news_service.models.user import User

        user_id = uuid.uuid4()
        sub_id = uuid.uuid4()
        topic_embedding = await embed_text(DIGEST_RETRIEVAL_QUERY)

        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-preflight-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    delivery_webhook_url=_WEBHOOK_URL,
                    has_onboarded=True,
                )
            )
            s.add(
                Subscription(
                    id=sub_id,
                    user_id=user_id,
                    user_spec=DIGEST_USER_SPEC,
                    topic_embedding=topic_embedding,
                    delivery_mode="digest",
                    digest_language="en",
                    schedule_cron="0 8 * * *",
                    delivery_webhook_url=_WEBHOOK_URL,
                    is_active=True,
                )
            )
            for idx, url in enumerate(_DIGEST_SOURCE_UNIVERSE):
                source_id = uuid.uuid4()
                s.add(
                    Source(
                        id=source_id,
                        url=url,
                        title=f"preflight-src-{idx}",
                        source_description=f"Pre-flight digest source #{idx}.",
                    )
                )
                s.add(
                    SubscriptionSource(
                        subscription_id=sub_id,
                        source_id=source_id,
                        is_user_specified=False,
                    )
                )
            await s.commit()

        from news_service.tasks.poll_feeds import _poll_all_feeds

        CLOCK.advance_to(_SIM_START + timedelta(days=_CLOCK_DAY))
        await _poll_all_feeds()
        await world.celery.drain()

        captured_reasons: list[list[str]] = []

        original_fn = pipeline_mod._compute_reflect_reasons

        def _spy(**kwargs):
            reasons = original_fn(**kwargs)
            captured_reasons.append(list(reasons))
            return reasons

        pipeline_mod._compute_reflect_reasons = _spy  # type: ignore[assignment]

        try:
            from news_service.tasks.deliver_digest import _deliver_digest

            await _deliver_digest(sub_id)
            await world.celery.drain()
        finally:
            pipeline_mod._compute_reflect_reasons = original_fn  # type: ignore[assignment]

        assert captured_reasons, (
            "expected _compute_reflect_reasons to have been called at least once "
            "during _deliver_digest; it was not invoked"
        )
        all_reasons = [r for batch in captured_reasons for r in batch]
        stale_reasons = [r for r in all_reasons if "has not published" in r]
        assert stale_reasons, (
            "expected at least one 'has not published' staleness reason once the "
            f"stale source crossed the {_STALENESS_THRESHOLD_DAYS}-day threshold. "
            f"All reasons captured: {all_reasons!r}"
        )
        assert any(stale_source_url in r for r in stale_reasons), (
            f"expected stale source {stale_source_url} to appear in a staleness "
            f"reason; got {stale_reasons!r}"
        )
    finally:
        pipeline_mod.settings.reflector_source_staleness_days = prior_threshold
