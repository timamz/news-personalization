"""
s04_verifier_catches_miss — rare-earth events, one buried miss.

Same persona and subscription as s03. Extends s03's timeline with a
small set of items deliberately crafted so the Batch Event Assessor
is expected to *miss* them on first pass (headline reads innocuous,
body carries a real supply-chain event one sentence in). Adds two
new search-corpus anchors that surface those misses when the Event
Verifier runs its weekly web_search pass.

Ground truth labels these items as `should_notify=True` — either the
Assessor catches them (good) or the Verifier's catch-up delivery
closes the gap (also good). A scenario run fails only if neither
path delivers.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from news_benchmark.data_gen.skeletons import s03 as s03_skeleton
from news_benchmark.scenarios.base import (
    AssertionSpec,
    Scenario,
    SearchCorpusAnchor,
    TimelineEntry,
)

SUB = s03_skeleton.SUB
START = s03_skeleton.START


def _buried_misses() -> list[TimelineEntry]:
    REU = "https://www.reuters.com/pf/api/v3/feed/metals"
    FST = "https://www.fastmarkets.com/rss/rare-earths"
    ARG = "https://www.argusmedia.com/rss/rare-earths"
    items: list[TimelineEntry] = []
    # Each miss: bland headline (assessor likely to skip) but body contains a
    # real event. Verifier queries should surface them via web search.
    miss_rows = [
        (
            7.6,
            REU,
            "MP Materials quarterly operations summary published",
            "hard_positive",
            "newsroom",
        ),
        (
            14.3,
            FST,
            "Annual industry survey: selected data points released",
            "hard_positive",
            "wire",
        ),
        (
            24.6,
            ARG,
            "Trade press roundup: rare-earth corner notes",
            "hard_positive",
            "newsroom",
        ),
    ]
    for day, src, headline, diff, style in miss_rows:
        ts = (START + timedelta(days=int(day), hours=9)).isoformat()
        items.append(
            TimelineEntry(
                fake_ts=ts,
                source_url=src,
                headline=headline,
                difficulty=diff,
                should_notify_per_sub={SUB: True},
                should_contribute_to_digest_per_sub={SUB: False},
                body_style_hint=style,
                body_adversarial=False,
                body_language="en",
            )
        )
    return items


def _verifier_anchors() -> list[SearchCorpusAnchor]:
    return [
        SearchCorpusAnchor(
            query_prefix="rare-earth supply chain event week",
            curated_results=[
                {
                    "title": "MP Materials reports temporary halt at processing unit",
                    "url": "https://www.reuters.com/pf/api/v3/feed/metals/mp-temporary-halt",
                    "snippet": (
                        "Quarterly operations summary discloses a three-day halt at the "
                        "Nevada processing unit pending regulatory review."
                    ),
                },
                {
                    "title": "Industry report reveals offtake deal renegotiation",
                    "url": "https://www.fastmarkets.com/rss/rare-earths/offtake-renegotiation",
                    "snippet": (
                        "Annual industry survey reveals two Australian producers "
                        "quietly renegotiated long-term offtake contracts with Asian buyers."
                    ),
                },
            ],
            fluff_count=3,
            fluff_topic_hint="generic commodity market commentary",
        ),
        SearchCorpusAnchor(
            query_prefix="rare-earth corner notes trade press",
            curated_results=[
                {
                    "title": "Argus: corner notes flag bottleneck at Chinese refinery",
                    "url": "https://www.argusmedia.com/rss/rare-earths/bottleneck",
                    "snippet": (
                        "Weekly corner notes paragraph mentions an unexpected bottleneck at "
                        "a Sichuan refinery that is not yet formally reported."
                    ),
                },
            ],
            fluff_count=3,
            fluff_topic_hint="general trade-press columns unrelated to rare earths",
        ),
    ]


def build() -> Scenario:
    base = s03_skeleton.build()
    timeline = list(base.timeline) + _buried_misses()
    corpus = list(base.search_corpus) + _verifier_anchors()

    assertions = [
        AssertionSpec(
            kind="subscription_exists_matching",
            payload={
                "goal_id": SUB,
                "expected_user_spec_keywords": ["rare earth", "supply", "event"],
                "expected_delivery_mode": "event",
            },
        ),
        AssertionSpec(kind="failed_tasks_zero"),
        AssertionSpec(
            kind="sources_within_bounds",
            payload={"goal_id": SUB, "min": 3, "max": 8},
        ),
        AssertionSpec(
            kind="verifier_catchup_delivered",
            payload={"goal_id": SUB, "min_catchups": 1},
        ),
    ]

    return replace(
        base,
        scenario_id="s04",
        timeline=timeline,
        search_corpus=corpus,
        assertions=assertions,
    )
