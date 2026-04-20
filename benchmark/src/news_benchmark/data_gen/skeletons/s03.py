"""
s03_event_breaking_news — commodities trader tracking rare-earth supply shocks.

Persona: Marcus, a commodities desk analyst in Frankfurt. Tracking supply
disruptions for rare earths (neodymium, dysprosium, lanthanum) and
specifically wants real-time event notifications — not digests — whenever a
named supply-chain event affects prices or flows. Explicitly excludes
routine price updates and long-form analyses.

This scenario exercises the event pipeline:
  - Conversational onboarding with delivery_mode="event"
  - Polling cycle picks up items from scripted timeline
  - Batch Event Assessor classifies each item
  - Event Judge critiques; REVISE loop activates for borderline cases
  - Delivery fires for pass-gated items, dropped-after-judge otherwise

Classification metrics (P/R/F1) are the primary signal here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from news_benchmark.scenarios.base import (
    AssertionSpec,
    ConversationTurn,
    Persona,
    Scenario,
    SearchCorpusAnchor,
    SourceEntry,
    SubscriptionGoal,
    TimelineEntry,
)

SUB = "rare_earth_events"
START = datetime(2026, 3, 1, 0, 0, 0).replace(microsecond=0)


def _ts(day: float, hour: int = 8, minute: int = 0) -> str:
    return (START + timedelta(days=int(day), hours=hour, minutes=minute)).isoformat()


def _e(
    day: float,
    source: str,
    headline: str,
    difficulty: str,
    notify: bool,
    style: str = "newsroom",
    adversarial: bool = False,
    hour: int = 8,
) -> TimelineEntry:
    return TimelineEntry(
        fake_ts=_ts(day, hour=hour),
        source_url=source,
        headline=headline,
        difficulty=difficulty,
        should_notify_per_sub={SUB: notify},
        should_contribute_to_digest_per_sub={SUB: False},
        body_style_hint=style,
        body_adversarial=adversarial,
    )


def _sources() -> list[SourceEntry]:
    good = [
        ("https://www.reuters.com/pf/api/v3/feed/metals", "rss", "Reuters metals wire"),
        ("https://www.bloomberg.com/feeds/metals.xml", "rss", "Bloomberg metals"),
        ("https://www.argusmedia.com/rss/rare-earths", "rss", "Argus rare earths"),
        ("https://www.fastmarkets.com/rss/rare-earths", "rss", "Fastmarkets rare earths"),
        ("https://t.me/s/rare_earth_watch", "telegram", "Rare-earth watch channel"),
    ]
    noise = [
        ("https://techcrunch.com/feed/", "rss", "Tech blog"),
        ("https://www.reddit.com/r/GlobalTalk/.rss", "reddit", "Global talk subreddit"),
        ("https://t.me/s/random_finance", "telegram", "Random finance channel"),
    ]
    return [
        SourceEntry(url=u, source_type=t, description=d, should_be_picked_by_finder=True)
        for u, t, d in good
    ] + [
        SourceEntry(url=u, source_type=t, description=d, should_be_picked_by_finder=False)
        for u, t, d in noise
    ]


def _timeline() -> list[TimelineEntry]:
    REU = "https://www.reuters.com/pf/api/v3/feed/metals"
    BLM = "https://www.bloomberg.com/feeds/metals.xml"
    ARG = "https://www.argusmedia.com/rss/rare-earths"
    FST = "https://www.fastmarkets.com/rss/rare-earths"
    TGR = "https://t.me/s/rare_earth_watch"

    TCR = "https://techcrunch.com/feed/"
    RDG = "https://www.reddit.com/r/GlobalTalk/.rss"
    TGF = "https://t.me/s/random_finance"

    items: list[TimelineEntry] = []

    items += [
        _e(
            0.5,
            REU,
            "China halts dysprosium exports citing national security review",
            "easy_positive",
            True,
        ),
        _e(
            2.3,
            ARG,
            "Lynas processing plant in Malaysia hit by fire, operations suspended",
            "easy_positive",
            True,
        ),
        _e(
            4.9,
            BLM,
            "Myanmar bans rare-earth mining in Kachin state",
            "easy_positive",
            True,
            style="wire",
        ),
        _e(
            8.7,
            FST,
            "Neodymium spot price jumps 14 percent on Chinese export curbs",
            "easy_positive",
            True,
        ),
        _e(
            11.2,
            REU,
            "Strike shuts down Mountain Pass rare-earth mine in California",
            "easy_positive",
            True,
        ),
        _e(
            15.4,
            TGR,
            "BREAKING: Vietnamese cabinet approves emergency rare-earth export licence revocation",
            "easy_positive",
            True,
            style="telegram",
        ),
        _e(
            21.8,
            ARG,
            "Russia sanctions target lanthanum-containing metallurgical exports",
            "easy_positive",
            True,
        ),
    ]

    items += [
        _e(
            1.4,
            BLM,
            "Pentagon signs $320M offtake agreement tied to Australian processor",
            "hard_positive",
            True,
        ),
        _e(
            3.1,
            REU,
            "Japan establishes strategic stockpile to cover nine-month dysprosium demand",
            "hard_positive",
            True,
        ),
        _e(
            6.0,
            ARG,
            "MP Materials halts shipment from Chinese refining partner",
            "hard_positive",
            True,
        ),
        _e(
            9.3,
            FST,
            "European Commission invokes Critical Raw Materials Act safeguard clause",
            "hard_positive",
            True,
        ),
        _e(
            12.8,
            REU,
            "Korea-Mongolia MoU secures 20% of Seoul's projected rare-earth demand",
            "hard_positive",
            True,
        ),
        _e(
            17.5,
            BLM,
            "Seaborne shipment delayed at Durban after dispute over export manifest",
            "hard_positive",
            True,
        ),
        _e(
            19.9,
            TGR,
            "Hint: rumors swirl around production cut at Sichuan refinery",
            "hard_positive",
            True,
            style="telegram",
        ),
        _e(
            23.4,
            FST,
            "Cerium smelter in Inner Mongolia temporarily closes after environmental inspection",
            "hard_positive",
            True,
        ),
        _e(
            26.1,
            ARG,
            "Brazilian Araxá project faces force majeure on three-month export shipment",
            "hard_positive",
            True,
        ),
        _e(
            28.6,
            REU,
            "Greenland grants exploration licence for monazite deposit near Narsaq",
            "hard_positive",
            True,
        ),
    ]

    items += [
        _e(
            0.8,
            TCR,
            "OpenAI releases new reasoning API pricing tier",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            1.2,
            RDG,
            "What's the deal with French pension reform lately?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            2.0,
            TGF,
            "Weekly fund flows recap: equities +2B, bonds -800M",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            3.5,
            TCR,
            "Framework laptop ships new AMD option",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            4.4,
            RDG,
            "Recommend me books on 20th century diplomatic history",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            5.1,
            TGF,
            "Bitcoin dominance at 61 percent, altseason delayed",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            6.8,
            TCR,
            "Y Combinator announces W26 demo day schedule",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            7.5,
            RDG,
            "Why are trains in Germany so bad now?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            8.9,
            TGF,
            "Forex: USD/JPY tests 150 level again",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            10.1,
            TCR,
            "AWS updates Bedrock console with agent templates",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            10.7,
            RDG,
            "Just got my first apartment in Lisbon, tips?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            11.9,
            TGF,
            "Oil: Brent steady at $84 amid OPEC communiqué",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            13.6,
            TCR,
            "React Server Components stabilise in Next.js 15.4",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            14.2,
            RDG,
            "Best way to learn Japanese in six months?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            14.9,
            TGF,
            "Gold: spot testing $2100 after Fed comments",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            16.5,
            TCR,
            "Apple Vision Pro gets developer-preview enterprise tools",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            17.9,
            RDG,
            "Anyone here attended the Porto marathon?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            18.4,
            TGF,
            "Copper: LME three-month at $9250, steady",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            20.3,
            TCR,
            "TypeScript 5.6 RC lands with control-flow improvements",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            21.1,
            RDG,
            "Best budget airlines in Southeast Asia?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            22.0,
            TGF,
            "Agri: wheat futures down 3% on Russian production upside",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            24.8,
            TCR,
            "Rust Foundation announces new security bounty program",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            25.5,
            RDG,
            "How do you budget in a dual-income household?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            27.2,
            TGF,
            "Silver: spot at $24, testing resistance",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            29.0,
            TCR,
            "Google rolls out Gemini Nano to Pixel 10 devices",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            1.9,
            RDG,
            "What flowers grow best in Frankfurt climate?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            5.8,
            TCR,
            "GitHub Copilot rolls out workspace-level context",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(8.3, RDG, "Music recommendations for studying?", "easy_negative", False, style="reddit"),
        _e(
            12.3,
            TCR,
            "Kubernetes 1.31 release candidate ships",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(17.1, RDG, "Good bakeries in Berlin Mitte?", "easy_negative", False, style="reddit"),
        _e(
            23.9,
            TCR,
            "Cloudflare launches new bot-mitigation API",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            26.4,
            RDG,
            "Why is Formula 1 so popular in the US now?",
            "easy_negative",
            False,
            style="reddit",
        ),
    ]

    items += [
        _e(
            2.7,
            FST,
            "Neodymium spot closes at 87 USD/kg, flat on the week",
            "near_miss_negative",
            False,
        ),
        _e(3.8, REU, "MP Materials publishes Q1 production guidance", "near_miss_negative", False),
        _e(
            5.6,
            ARG,
            "Analysis: long-term rare-earth demand hinges on EV penetration",
            "near_miss_negative",
            False,
        ),
        _e(
            7.2,
            BLM,
            "Mining equipment maker Sandvik raises FY guidance",
            "near_miss_negative",
            False,
        ),
        _e(9.8, FST, "Cobalt prices stabilise after recent drawdown", "near_miss_negative", False),
        _e(
            11.6,
            REU,
            "China releases 2025 rare-earth production quota unchanged YoY",
            "near_miss_negative",
            False,
        ),
        _e(
            13.1,
            ARG,
            "Tantalum exports from DRC resume at pre-conflict levels",
            "near_miss_negative",
            False,
        ),
        _e(
            14.7,
            BLM,
            "Nickel market oversupplied through 2027 says Wood Mackenzie",
            "near_miss_negative",
            False,
        ),
        _e(
            16.0,
            REU,
            "Lithium spodumene prices slip on Chinese converter inventories",
            "near_miss_negative",
            False,
        ),
        _e(18.1, FST, "Scandium export permits revived in Kazakhstan", "near_miss_negative", False),
        _e(
            19.4,
            TGR,
            "Chart: 10-year neodymium price history vs EV sales",
            "near_miss_negative",
            False,
            style="telegram",
        ),
        _e(20.9, ARG, "Graphite suppliers warn of Q3 tightness", "near_miss_negative", False),
        _e(
            22.6,
            REU,
            "Rio Tinto announces dividend in line with consensus",
            "near_miss_negative",
            False,
        ),
        _e(
            24.1,
            BLM,
            "Platinum prices rebound on auto-catalyst demand",
            "near_miss_negative",
            False,
        ),
        _e(
            25.7, FST, "Gallium supply picture for 2027 looks balanced", "near_miss_negative", False
        ),
        _e(
            26.9,
            ARG,
            "Indium prices at two-year high on semiconductor demand",
            "near_miss_negative",
            False,
        ),
        _e(
            27.6,
            REU,
            "Uranium enrichment capacity constrained through 2028",
            "near_miss_negative",
            False,
        ),
        _e(
            28.3,
            BLM,
            "Molybdenum market shrugs off Chinese production outage",
            "near_miss_negative",
            False,
        ),
        _e(
            29.5,
            FST,
            "Tin inventories at LME remain at multi-decade lows",
            "near_miss_negative",
            False,
        ),
        _e(
            4.2,
            REU,
            "Commentary: why rare-earth prices will normalise by year end",
            "near_miss_negative",
            False,
        ),
        _e(
            6.4,
            BLM,
            "Long read: how Mountain Pass came back from bankruptcy",
            "near_miss_negative",
            False,
        ),
        _e(8.0, ARG, "Explainer: what rare-earth magnets actually do", "near_miss_negative", False),
        _e(10.3, FST, "Podcast: our monthly metals roundtable", "near_miss_negative", False),
        _e(12.5, REU, "Rare-earth ETF flows hit six-month low", "near_miss_negative", False),
        _e(
            15.8,
            BLM,
            "Opinion: time to diversify critical-mineral supply chains",
            "near_miss_negative",
            False,
        ),
        _e(
            2.5,
            FST,
            "Rare-earth weekly wrap: praseodymium flat, terbium steady",
            "near_miss_negative",
            False,
        ),
        _e(
            4.6,
            REU,
            "MP Materials guides Q2 rare-earth output toward prior-quarter figure",
            "near_miss_negative",
            False,
        ),
        _e(
            6.9,
            ARG,
            "Rare-earth shipments from Malaysia resume routine schedule",
            "near_miss_negative",
            False,
        ),
        _e(
            9.2,
            TGR,
            "Rare-earth chart of the day: 30-day moving average",
            "near_miss_negative",
            False,
            style="telegram",
        ),
        _e(
            11.0,
            BLM,
            "Rare-earth ETFs see small net outflows on the week",
            "near_miss_negative",
            False,
        ),
        _e(
            14.5,
            FST,
            "Rare-earth feature: inside a Sichuan processing plant tour",
            "near_miss_negative",
            False,
        ),
        _e(
            16.7,
            REU,
            "Rare-earth conference in Perth wraps with industry-panel recap",
            "near_miss_negative",
            False,
        ),
        _e(
            18.7,
            ARG,
            "Rare-earth inventory data from Shanghai Metals Market released",
            "near_miss_negative",
            False,
        ),
        _e(
            21.4,
            BLM,
            "Rare-earth mining equipment firm reports steady earnings",
            "near_miss_negative",
            False,
        ),
        _e(
            24.5,
            FST,
            "Rare-earth recycling startup raises Series B funding",
            "near_miss_negative",
            False,
        ),
    ]

    items += [
        _e(
            7.3,
            TGR,
            "URGENT: MASSIVE rare-earth export ban, details inside",
            "adversarial",
            False,
            style="telegram",
            adversarial=True,
        ),
        _e(
            13.9,
            REU,
            "Shocking truth about dysprosium shortages nobody is telling you",
            "adversarial",
            False,
            style="techcrunch",
            adversarial=True,
        ),
        _e(
            20.5,
            FST,
            "Opinion: the rare-earth crisis is overblown",
            "adversarial",
            False,
            style="opinion",
        ),
        _e(
            23.0,
            ARG,
            "Commentary: time to rethink critical minerals policy",
            "adversarial",
            False,
            style="opinion",
        ),
    ]

    items += [
        _e(
            0.6,
            BLM,
            "China halts dysprosium exports under security review",
            "duplicate",
            True,
            style="wire",
        ),
        _e(0.9, ARG, "Beijing confirms dysprosium export halt", "duplicate", True),
        _e(15.5, REU, "Vietnam cabinet revokes rare-earth export licences", "duplicate", True),
    ]

    return items


def _search_corpus() -> list[SearchCorpusAnchor]:
    return [
        SearchCorpusAnchor(
            query_prefix="rare earth supply chain news RSS",
            curated_results=[
                {
                    "title": "Argus Media — Rare Earths",
                    "url": "https://www.argusmedia.com/rss/rare-earths",
                    "snippet": "Specialist rare-earths pricing and supply-chain coverage.",
                },
                {
                    "title": "Fastmarkets — Rare Earths",
                    "url": "https://www.fastmarkets.com/rss/rare-earths",
                    "snippet": "Daily rare-earth spot prices and disruption news.",
                },
                {
                    "title": "Reuters metals wire",
                    "url": "https://www.reuters.com/pf/api/v3/feed/metals",
                    "snippet": "Global metals and mining news from Reuters.",
                },
            ],
            fluff_count=4,
            fluff_topic_hint="generic commodities news and base metals",
        ),
        SearchCorpusAnchor(
            query_prefix="dysprosium neodymium export restrictions",
            curated_results=[
                {
                    "title": "Argus: dysprosium export review timeline",
                    "url": "https://www.argusmedia.com/rss/rare-earths",
                    "snippet": "Timeline of Chinese export-licence decisions on heavy rare earths.",
                },
            ],
            fluff_count=3,
            fluff_topic_hint="EV battery materials, lithium policy",
        ),
    ]


def _scripted_turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(
            fake_day=0,
            message=(
                "Hi — I work at a commodities desk. I need real-time alerts when a "
                "named supply-chain event hits the rare-earth market: export bans, "
                "plant closures, sanctions, force majeure. NOT routine price moves, "
                "NOT general analysis pieces, NOT opinion. One alert per event."
            ),
        ),
    ]


def _assertions() -> list[AssertionSpec]:
    return [
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
    ]


def build() -> Scenario:
    return Scenario(
        scenario_id="s03",
        persona=Persona(
            language="en",
            timezone="Europe/Berlin",
            tech_literacy="high",
            verbosity="low",
            can_suggest_urls=False,
        ),
        goals=[
            SubscriptionGoal(
                goal_id=SUB,
                description=(
                    "Real-time event notifications about rare-earth supply-chain "
                    "disruptions: export bans/quotas, sanctions, mine/plant "
                    "closures, force-majeure declarations, stockpile and offtake "
                    "announcements. "
                    "EXCLUDE: routine spot price updates, general analysis, "
                    "explainers/backgrounders, opinion pieces. "
                    "English only."
                ),
                expected_user_spec_keywords=["rare earth", "supply", "event", "export"],
                expected_delivery_mode="event",
                expected_webhook_url=f"https://bench.invalid/sub/{SUB}/event",
                expected_sources_min=3,
                expected_sources_max=8,
            )
        ],
        simulated_days=30,
        start_date_iso=START.isoformat(),
        source_universe=_sources(),
        timeline=_timeline(),
        search_corpus=_search_corpus(),
        scripted_turns=_scripted_turns(),
        assertions=_assertions(),
    )
