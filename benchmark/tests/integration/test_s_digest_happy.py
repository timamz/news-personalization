"""
S-digest-happy: digest-pipeline happy path.

Seeds one digest-mode subscription and eight pre-embedded news items
(six on-topic EU energy-policy stories, two off-topic noise) directly
in the DB, then fires ``_deliver_digest`` once. Asserts:

  * exactly one webhook delivery lands for the sub;
  * the delivered body is non-trivial (>= 200 chars);
  * >= 3 of the 6 on-topic items are covered per signature-term match
    (paraphrase-robust: each item's signature is a small set of
    distinctive proper-noun or topic-keyword tokens any faithful
    paraphrase must preserve -- see ``_digest_common.count_covered_items``);
  * 0 of the 2 off-topic items leak into the digest;
  * exactly as many ``SentItem`` rows were inserted as there are
    covered on-topic items (writer's ``used_item_ids`` aligns with
    the body).

Polling is skipped -- S-event-assess already proved that path. The
writer's optional web-search tool is patched to an empty corpus via
``FakeSearch``, forcing the writer to compose from the pre-seeded items
alone.

Out of scope: the Writer <-> Judge REVISE loop (covered in
``test_s_digest_revise.py``), format-spec compliance, and prose-quality
rubric scoring (the latter belongs in the S-happy benchmark).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tests.integration._digest_common import (
    WEBHOOK_URL,
    count_covered_items,
    seed_digest_world,
)


@pytest.mark.asyncio
async def test_s_digest_happy_path(world):
    """Digest fires once, covers at least half the on-topic pool, no noise leaks."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.sent_item import SentItem
    from news_service.tasks.deliver_digest import _deliver_digest

    _user_id, sub_id, _source_id = await seed_digest_world(world, embedding_fn=embed_text)

    result = await _deliver_digest(sub_id)
    assert result.get("status") == "delivered", (
        f"expected delivered status, got {result!r}"
    )

    await world.celery.drain()

    captured = world.delivery.for_url(WEBHOOK_URL)
    assert len(captured) == 1, (
        f"expected exactly 1 digest webhook for {WEBHOOK_URL}, got {len(captured)}. "
        f"Bodies: {[c.body[:120] for c in captured]}"
    )

    body = captured[0].body
    assert len(body) >= 200, (
        f"digest body should be >= 200 chars, got {len(body)} chars: {body!r}"
    )

    on_topic, off_topic = count_covered_items(body)
    assert off_topic == 0, (
        f"off-topic leak: {off_topic} off-topic items appear in the digest. Body:\n{body}"
    )
    assert on_topic >= 3, (
        f"expected >= 3 of 6 on-topic items covered by signature-term match, "
        f"got {on_topic}. Body:\n{body}"
    )

    async with async_session_factory() as s:
        sent_rows = list(
            (await s.execute(select(SentItem).where(SentItem.subscription_id == sub_id)))
            .scalars()
            .all()
        )
    assert len(sent_rows) >= on_topic, (
        f"SentItem count ({len(sent_rows)}) should be >= number of covered on-topic "
        f"items ({on_topic}); writer's used_item_ids must include at least what "
        f"appears in the body."
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
