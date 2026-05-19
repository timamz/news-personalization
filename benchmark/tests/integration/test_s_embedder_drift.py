"""
S-embedder-drift: two-layer verification of the embedder's semantic signal.

Layer 1 -- direct embedder sanity. Embeds four clearly on-topic EU energy
and climate policy paragraphs plus four clearly off-topic paragraphs
(celebrity gossip, reality TV, sports commentary, fashion trends), and
asserts that the on-topic cluster sits well above the off-topic cluster
under cosine similarity against a fixed topic vector. This catches a
model swap or a broken embedding call: when the backend is configured
with a functioning sentence embedder, on-topic cosine should comfortably
exceed 0.35 while off-topic stays at or below 0.32, with a mean cluster
gap of at least 0.12.

Layer 2 -- distribution-shift detection via the reflector's per-source
item cosine metrics. Seeds a digest subscription with a single source
whose recent items form a bimodal mix of old-on-topic and new-off-topic
bodies. For the seeded items we compute the three-number summary
(``item_cosine_p50``, ``item_cosine_p90``, ``item_cosine_std``) against
the subscription's topic vector using exactly the same formula the
``update_subscription_source_stats`` Celery task uses to populate the
``ReflectorSourceContext`` fields. The bimodal mix is expected to land
with a central p50 (0.20..0.50, between the two modes), elevated std
(>= 0.10, reflecting the bimodal spread), and a p90 that still detects
the on-topic tail (>= 0.35). This documents an intentional coverage
gap in the current backend: the reflector's triggers key off
``source_description_embedding`` drift, not per-item cosine drift, so
even though the item-cosine metrics clearly show the distribution
shift, no reflector run is kicked off by it. The test captures that
the SIGNAL is present so a future trigger can be wired against it.

Thresholds are calibrated for ``text-embedding-3-small``. If the
configured embedding model is swapped the thresholds may need to be
revisited; the assertion failure messages print every individual cosine
to make that re-tuning straightforward.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from tests.integration._reflector_common import (
    SourceSpec,
    seed_reflector_world,
)

_TOPIC_TEXT = "EU energy and climate policy"


_ON_TOPIC_PARAGRAPHS: list[str] = [
    (
        "The Council of the European Union adopted conclusions on energy "
        "security and gas storage. Ministers endorsed tighter filling "
        "targets for underground storage ahead of winter. The text was "
        "approved by qualified majority."
    ),
    (
        "The European Commission proposed a new directive raising the "
        "binding renewable electricity target for the EU-27 to 45% by "
        "2030. The proposal cites falling costs for solar and offshore "
        "wind generation."
    ),
    (
        "EUR-Lex published the final text of a directive on accelerated "
        "permitting for onshore wind and grid reinforcement. Member "
        "States have eighteen months to transpose. The directive enters "
        "force twenty days after publication."
    ),
    (
        "ENTSO-E released its winter adequacy outlook for the European "
        "electricity system, flagging capacity margins in Central "
        "Europe. The publication coincides with ACER's review of "
        "balancing-market network codes."
    ),
]


_OFF_TOPIC_PARAGRAPHS: list[str] = [
    (
        "A Hollywood actress and her musician fiance announced their "
        "summer wedding on social media. Tabloids speculated about the "
        "designer rumored to be making the dress, while celebrity "
        "friends flooded the comments with congratulations."
    ),
    (
        "The reality dating show's season finale pulled in a record "
        "streaming audience. Viewers crashed the app during the final "
        "rose ceremony, and the runner-up couple's dramatic exit "
        "dominated social-media reactions for hours."
    ),
    (
        "The quarterback threw three touchdown passes in the second "
        "half to clinch a divisional playoff berth. Post-game interviews "
        "focused on the rookie receiver and the head coach's aggressive "
        "fourth-down play-calling."
    ),
    (
        "At Milan Fashion Week, designers brought back exaggerated "
        "shoulders and metallic fabrics. Front-row celebrities were "
        "photographed in ruffled gowns, and street-style bloggers hailed "
        "the return of 1980s silhouettes."
    ),
]


_DRIFT_ON_TOPIC_ITEMS: list[dict[str, str]] = [
    {
        "headline": "Council of the EU adopts emergency gas storage directive for 2027 winter",
        "body": (
            "The Council of the European Union on Tuesday adopted a directive "
            "requiring Member States to fill underground gas storage to 90% of "
            "capacity by 1 November each year starting in 2027. The text was "
            "approved by qualified majority. Commission officials said the "
            "measure guards against supply shocks like those seen in 2022."
        ),
    },
    {
        "headline": "European Commission proposes 40% renewable electricity target by 2030",
        "body": (
            "The European Commission on Wednesday proposed a binding 40% "
            "renewable electricity target for the EU-27 by 2030, up from 32%. "
            "Commissioner Kadri Simson said the higher ceiling reflects falling "
            "costs for solar and wind generation."
        ),
    },
    {
        "headline": "ACER publishes final network code on electricity balancing markets",
        "body": (
            "The EU Agency for the Cooperation of Energy Regulators published "
            "the final text of the amended network code on electricity "
            "balancing. National regulators have six months to transpose."
        ),
    },
    {
        "headline": "ENTSO-E warns of winter capacity shortfall in Central Europe",
        "body": (
            "ENTSO-E's winter outlook warns that Central European grids may "
            "face capacity shortfalls in December and January. Germany, Austria "
            "and the Czech Republic are most exposed."
        ),
    },
    {
        "headline": "EUR-Lex publishes directive on accelerated offshore wind permitting",
        "body": (
            "A new directive accelerating permitting for offshore wind "
            "installations was published on EUR-Lex and will enter force 20 "
            "days after publication. Transposition deadline is 18 months."
        ),
    },
    {
        "headline": "ENVI committee tightens methane-leak limits for imported LNG",
        "body": (
            "ENVI voted to tighten methane-intensity limits for imported LNG. "
            "The amendment extends the Methane Regulation's import standard to "
            "the full upstream value chain."
        ),
    },
]


_DRIFT_OFF_TOPIC_ITEMS: list[dict[str, str]] = [
    {
        "headline": "Hollywood couple announces summer wedding",
        "body": (
            "A-list actress and her musician boyfriend confirmed on social "
            "media that they will marry this summer at a private ceremony in "
            "Malibu. Celebrity friends quickly flooded the comments with "
            "congratulations, and tabloids speculated about the guest list "
            "and the designer rumored to be making the dress."
        ),
    },
    {
        "headline": "Reality TV villa finale draws record viewership",
        "body": (
            "The season finale of the hit reality dating show pulled in a "
            "record live audience last night. Viewers crashed the streaming "
            "app in the final rose ceremony, and social media lit up with "
            "memes about the runner-up couple's dramatic exit."
        ),
    },
    {
        "headline": "Milan Fashion Week: bold shoulders return",
        "body": (
            "Runways at Milan Fashion Week signaled the return of exaggerated "
            "shoulders and metallic fabrics. Front-row celebrities were "
            "photographed in ruffled gowns, and street-style bloggers hailed "
            "the comeback of 1980s silhouettes."
        ),
    },
    {
        "headline": "Pop star drops surprise album at midnight",
        "body": (
            "A chart-topping pop singer released a surprise twelve-track "
            "album overnight, triggering a frenzy across streaming platforms. "
            "Fans dissected cryptic lyrics for hints about an alleged "
            "celebrity feud, and the lead single is already trending globally."
        ),
    },
    {
        "headline": "Blockbuster sequel breaks opening-weekend record",
        "body": (
            "The latest entry in a long-running superhero franchise broke "
            "opening-weekend records at the domestic box office. Studio "
            "executives credited the returning cast and the viral marketing "
            "campaign built around the lead actor's red-carpet appearances."
        ),
    },
    {
        "headline": "Royal couple seen at charity gala in London",
        "body": (
            "A senior royal couple attended a charity gala in central London "
            "last night, posing for photos with Hollywood guests and sports "
            "stars. Tabloid coverage focused on the duchess's gown and a "
            "rumored feud with a former palace aide."
        ),
    },
]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile on an already-sorted list.

    Mirrors ``update_subscription_source_stats._percentile`` so the Layer 2
    assertions are measured against the exact same formula the backend
    uses to populate the ``item_cosine_*`` columns.
    """
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


@pytest.mark.asyncio
async def test_s_embedder_clusters_on_topic_vs_off_topic(world):
    """Embedder must separate EU-policy bodies from celebrity/TV/sports/fashion."""
    from news_service.db.vector_store import embed_text
    from news_service.services.relevance import cosine_similarity

    topic_vec = await embed_text(_TOPIC_TEXT)

    on_topic_sims: list[float] = []
    for paragraph in _ON_TOPIC_PARAGRAPHS:
        p_vec = await embed_text(paragraph)
        on_topic_sims.append(cosine_similarity(p_vec, topic_vec))

    off_topic_sims: list[float] = []
    for paragraph in _OFF_TOPIC_PARAGRAPHS:
        p_vec = await embed_text(paragraph)
        off_topic_sims.append(cosine_similarity(p_vec, topic_vec))

    min_on_topic = min(on_topic_sims)
    assert min_on_topic >= 0.35, (
        f"on-topic paragraph cosine fell below 0.35: min={min_on_topic:.4f}. "
        f"All on-topic sims={on_topic_sims!r}, off-topic sims={off_topic_sims!r}. "
        f"This usually means the configured embedding model was swapped to one "
        f"with weaker semantic separation; re-tune the threshold or revert."
    )

    max_off_topic = max(off_topic_sims)
    assert max_off_topic <= 0.32, (
        f"off-topic paragraph cosine rose above 0.32: max={max_off_topic:.4f}. "
        f"All off-topic sims={off_topic_sims!r}, on-topic sims={on_topic_sims!r}. "
        f"This usually means the embedder is producing near-uniform vectors or "
        f"the off-topic paragraphs accidentally mention policy language."
    )

    gap = statistics.mean(on_topic_sims) - statistics.mean(off_topic_sims)
    assert gap >= 0.12, (
        f"on-topic vs off-topic mean cluster gap collapsed to {gap:.4f} "
        f"(expected >= 0.12). mean(on_topic)={statistics.mean(on_topic_sims):.4f}, "
        f"mean(off_topic)={statistics.mean(off_topic_sims):.4f}. "
        f"All on-topic sims={on_topic_sims!r}, off-topic sims={off_topic_sims!r}. "
        f"A shrinking gap usually indicates a weaker embedding model."
    )


@pytest.mark.asyncio
async def test_s_embedder_item_cosine_distribution_shifts_over_time(world):
    """Reflector item-cosine metrics must surface a bimodal on-/off-topic mix."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.news_item import NewsItem
    from news_service.models.subscription import Subscription
    from news_service.services.relevance import cosine_similarity

    from news_benchmark.clock import CLOCK

    drift_spec = SourceSpec(
        label="source_drifting_over_time",
        url="https://eu-policy-wire-drift.invalid/feed.xml",
        description_text=(
            "Independent policy wire covering EU energy and climate "
            "regulation: Council of the European Union decisions, "
            "European Commission proposals, ENTSO-E and ACER network "
            "codes, EUR-Lex directives on renewables and methane."
        ),
        is_user_specified=False,
        items=[],
        items_age_hours=6,
    )

    _user_id, sub_id, source_id_by_label = await seed_reflector_world(
        world,
        embedding_fn=embed_text,
        sources=[drift_spec],
    )
    source_id = source_id_by_label["source_drifting_over_time"]

    now = CLOCK.now()

    async with async_session_factory() as s:
        for idx, item in enumerate(_DRIFT_ON_TOPIC_ITEMS):
            age_days = 20 + idx * 2
            emb = await embed_text(item["headline"] + "\n" + item["body"][:400])
            s.add(
                NewsItem(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    headline=item["headline"],
                    body=item["body"],
                    url=f"{drift_spec.url.rstrip('/')}/old-{idx:02d}",
                    source=drift_spec.label,
                    published_at=now - timedelta(days=age_days),
                    fetched_at=now - timedelta(days=age_days),
                    embedding=emb,
                )
            )
        for idx, item in enumerate(_DRIFT_OFF_TOPIC_ITEMS):
            age_days = idx
            emb = await embed_text(item["headline"] + "\n" + item["body"][:400])
            s.add(
                NewsItem(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    headline=item["headline"],
                    body=item["body"],
                    url=f"{drift_spec.url.rstrip('/')}/new-{idx:02d}",
                    source=drift_spec.label,
                    published_at=now - timedelta(days=age_days),
                    fetched_at=now - timedelta(days=age_days),
                    embedding=emb,
                )
            )
        await s.commit()

    async with async_session_factory() as s:
        sub = (
            await s.execute(select(Subscription).where(Subscription.id == sub_id))
        ).scalar_one()
        topic_embedding = list(sub.topic_embedding)
        item_rows = (
            await s.execute(
                select(NewsItem.embedding).where(
                    NewsItem.source_id == source_id,
                    NewsItem.embedding.is_not(None),
                )
            )
        ).all()

    cosines = [
        cosine_similarity(list(emb), topic_embedding)
        for (emb,) in item_rows
        if emb is not None
    ]
    assert len(cosines) == len(_DRIFT_ON_TOPIC_ITEMS) + len(_DRIFT_OFF_TOPIC_ITEMS), (
        f"expected {len(_DRIFT_ON_TOPIC_ITEMS) + len(_DRIFT_OFF_TOPIC_ITEMS)} "
        f"item embeddings for the drifting source, got {len(cosines)}. "
        f"Something in the seeding path dropped items silently."
    )

    ordered = sorted(cosines)
    p50 = _percentile(ordered, 0.5)
    p90 = _percentile(ordered, 0.9)
    mean = sum(cosines) / len(cosines)
    variance = sum((v - mean) ** 2 for v in cosines) / len(cosines)
    std = variance**0.5

    assert 0.20 <= p50 <= 0.50, (
        f"item_cosine_p50={p50:.4f} fell outside the bimodal band [0.20, 0.50]. "
        f"p90={p90:.4f}, std={std:.4f}, all cosines sorted={ordered!r}. "
        f"A p50 above 0.50 means the off-topic half is not landing off-topic; "
        f"below 0.20 means the on-topic half stopped clustering on-topic."
    )

    assert std >= 0.10, (
        f"item_cosine_std={std:.4f} is below the bimodal-spread floor of 0.10. "
        f"p50={p50:.4f}, p90={p90:.4f}, all cosines sorted={ordered!r}. "
        f"A collapsed std means the embedder no longer separates on-topic "
        f"from off-topic items -- the drift signal is invisible."
    )

    assert p90 >= 0.35, (
        f"item_cosine_p90={p90:.4f} dropped below 0.35: the on-topic tail "
        f"of the distribution is no longer detectable. "
        f"p50={p50:.4f}, std={std:.4f}, all cosines sorted={ordered!r}."
    )
