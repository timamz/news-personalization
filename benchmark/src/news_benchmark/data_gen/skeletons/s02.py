"""
s02_digest_reflector_drift — EU energy analyst, drifting source.

Same persona and user_spec as s01. Extends s01's timeline with ~32 new
items from ONE of the "good" sources that gradually drifts off-topic:
the first week it publishes on-topic EU energy items (carried over from
s01), then its output pivots to unrelated Bloomberg content
(crypto / equities / luxury) through weeks 2-4. This forces the
Reflector's drift trigger: the source's aggregate cosine similarity to
the subscription's topic embedding falls below
`reflector_drift_similarity_threshold` (0.30), and the Reflector is
expected to call `remove_source` on it.

Reuses every labeled item from s01 unchanged (body cache is shared via
content-addressed hashing), so the incremental generation cost is
bounded to ~32 new bodies.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from news_benchmark.data_gen.skeletons import s01 as s01_skeleton
from news_benchmark.data_gen.skeletons._bulk import bulk
from news_benchmark.scenarios.base import AssertionSpec, Scenario

DRIFTING_SOURCE = "https://www.bloomberg.com/feeds/podcasts/etf_iq.xml"
SUB = s01_skeleton.SUB
START = s01_skeleton.START


def _drift_items():
    drift_rows: list[tuple[str, str, str]] = [
        (DRIFTING_SOURCE, "Bitcoin ETF inflows top $1.8B for the week", "drift-crypto"),
        (DRIFTING_SOURCE, "Solana futures volumes hit new monthly record", "drift-crypto"),
        (
            DRIFTING_SOURCE,
            "Coinbase Q1 earnings preview: trading volume modeled up 22 percent",
            "drift-crypto",
        ),
        (DRIFTING_SOURCE, "Ethereum staking yields slip under 3 percent", "drift-crypto"),
        (
            DRIFTING_SOURCE,
            "BlackRock spot crypto ETF gains traction with pension allocators",
            "drift-crypto",
        ),
        (
            DRIFTING_SOURCE,
            "Memecoin rally continues: top five gainers on Solana this week",
            "drift-crypto",
        ),
        (
            DRIFTING_SOURCE,
            "Binance BNB burn reaches 17 percent cumulative supply reduction",
            "drift-crypto",
        ),
        (DRIFTING_SOURCE, "Circle files confidential S-1 with SEC for IPO", "drift-crypto"),
        (DRIFTING_SOURCE, "Luxury watch market secondary prices rebound in March", "drift-luxury"),
        (DRIFTING_SOURCE, "Rolex increases retail prices by 4 percent globally", "drift-luxury"),
        (DRIFTING_SOURCE, "Patek Philippe wait-list policy update leaks online", "drift-luxury"),
        (DRIFTING_SOURCE, "LVMH reports steady handbag sales in quarterly update", "drift-luxury"),
        (DRIFTING_SOURCE, "Richemont names new watchmaking division chief", "drift-luxury"),
        (DRIFTING_SOURCE, "Christie's auction of historical timepieces tops $42M", "drift-luxury"),
        (
            DRIFTING_SOURCE,
            "Hermes Birkin bag price index sees 6 percent annual increase",
            "drift-luxury",
        ),
        (DRIFTING_SOURCE, "Private jet charter bookings dip in Q1 industry survey", "drift-luxury"),
        (DRIFTING_SOURCE, "S&P 500 closes at new record above 5,900", "drift-equities"),
        (DRIFTING_SOURCE, "Nvidia market cap passes $3.4 trillion mark", "drift-equities"),
        (DRIFTING_SOURCE, "Tesla Q2 delivery estimates revised up by analysts", "drift-equities"),
        (
            DRIFTING_SOURCE,
            "Apple reports iPhone 17 sales slightly below consensus",
            "drift-equities",
        ),
        (DRIFTING_SOURCE, "Microsoft Azure growth accelerates on AI demand", "drift-equities"),
        (
            DRIFTING_SOURCE,
            "Amazon logistics network expansion set for Q3 announcement",
            "drift-equities",
        ),
        (
            DRIFTING_SOURCE,
            "Netflix subscriber additions beat Street expectations",
            "drift-equities",
        ),
        (DRIFTING_SOURCE, "Meta Reality Labs posts deeper operating loss in Q1", "drift-equities"),
        (
            DRIFTING_SOURCE,
            "Disney parks segment shows resilience in Q1 commentary",
            "drift-equities",
        ),
        (DRIFTING_SOURCE, "Alphabet repurchase authorization raised by $70B", "drift-equities"),
        (
            DRIFTING_SOURCE,
            "Berkshire Hathaway increases stake in Japanese trading houses",
            "drift-equities",
        ),
        (
            DRIFTING_SOURCE,
            "Oracle cloud revenue accelerates past Amazon growth rate",
            "drift-equities",
        ),
        (
            DRIFTING_SOURCE,
            "IBM mainframe refresh cycle drives systems revenue upside",
            "drift-equities",
        ),
        (DRIFTING_SOURCE, "Snowflake CEO shakeup rattles investor base", "drift-equities"),
        (
            DRIFTING_SOURCE,
            "Robinhood monthly active users plateau despite product launches",
            "drift-equities",
        ),
        (
            DRIFTING_SOURCE,
            "Palantir commercial book-to-bill ratio tops 1.3x in Q1",
            "drift-equities",
        ),
    ]
    return bulk(
        SUB,
        START + timedelta(days=8, hours=10),
        spread_days=21,
        difficulty="near_miss_negative",
        positive=False,
        rows=drift_rows,
        style_cycle=("newsroom", "wire"),
    )


def build() -> Scenario:
    base = s01_skeleton.build()
    drift = _drift_items()
    timeline = list(base.timeline) + drift

    assertions = [
        AssertionSpec(
            kind="subscription_exists_matching",
            payload={
                "goal_id": SUB,
                "expected_user_spec_keywords": ["EU", "energy", "climate", "regulation"],
                "expected_schedule_cron": "0 9 * * 1",
                "expected_delivery_mode": "digest",
                "expected_digest_language": "en",
            },
        ),
        AssertionSpec(kind="failed_tasks_zero"),
        AssertionSpec(
            kind="digest_webhooks_delivered",
            payload={"goal_id": SUB, "min_count": 3, "max_count": 5},
        ),
        AssertionSpec(
            kind="sources_within_bounds",
            payload={"goal_id": SUB, "min": 2, "max": 10},
        ),
        AssertionSpec(
            kind="source_removed_by_reflector",
            payload={"goal_id": SUB, "expected_url": DRIFTING_SOURCE},
        ),
    ]

    return replace(
        base,
        scenario_id="s02",
        timeline=timeline,
        assertions=assertions,
    )
