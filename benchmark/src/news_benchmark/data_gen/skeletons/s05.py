"""
s05_multi_sub_power_user — three subscriptions, one user.

Anna, a Brussels policy analyst, has three subscriptions:
  - `eu_energy_digest` (digest, weekly)      — same as s01
  - `rare_earth_events` (event mode)          — borrowed from s03 via multi-hat persona
  - `eu_ai_regulation_digest` (digest, daily) — new, fresh topic

Items inherit from s01's and s03's timelines but are re-scoped: each
item carries labels only for the subscription it is relevant to. A
third pool of ~40 new items covers the EU AI regulation topic so that
subscription has its own labeled pool.

This scenario exercises:
  - Three consecutive `create_subscription` tool calls in one conversation
  - Multi-sub scheduler concurrency (weekly digest + daily digest + event)
  - `update_subscription` mid-run edit to the AI-regulation digest
  - `add_source` appending a hand-supplied URL to one sub
"""

from __future__ import annotations

from datetime import timedelta

from news_benchmark.data_gen.skeletons import s01 as s01_skeleton
from news_benchmark.data_gen.skeletons import s03 as s03_skeleton
from news_benchmark.data_gen.skeletons._bulk import bulk
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

SUB_ENERGY = "eu_energy_digest"
SUB_RARE = "rare_earth_events"
SUB_AIREG = "eu_ai_regulation_digest"

START = s01_skeleton.START


def _rescope_to_single_sub(items, keep_sub: str) -> list[TimelineEntry]:
    """Strip foreign-sub labels from imported items; keep only `keep_sub`."""
    out: list[TimelineEntry] = []
    for t in items:
        notify = t.should_notify_per_sub.get(keep_sub)
        contrib = t.should_contribute_to_digest_per_sub.get(keep_sub)
        out.append(
            TimelineEntry(
                fake_ts=t.fake_ts,
                source_url=t.source_url,
                headline=t.headline,
                difficulty=t.difficulty,
                should_notify_per_sub=({keep_sub: notify} if notify is not None else {}),
                should_contribute_to_digest_per_sub=(
                    {keep_sub: contrib} if contrib is not None else {}
                ),
                body_style_hint=t.body_style_hint,
                body_adversarial=t.body_adversarial,
                body_language=t.body_language,
            )
        )
    return out


def _ai_regulation_items() -> list[TimelineEntry]:
    """Third-sub pool: EU AI regulation digest. Daily schedule, narrow topic."""
    EUR = "https://www.euractiv.com/section/digital/feed/"
    POL = "https://www.politico.eu/section/technology/feed/"
    MLB = "https://mlblog.example/feed/"
    ARX = "https://arxiv.org/rss/cs.CL"

    items: list[TimelineEntry] = []

    pos_rows = [
        (EUR, "EU AI Act implementation guidance published", "easy_positive"),
        (
            POL,
            "European Commission opens consultation on general-purpose AI code of practice",
            "easy_positive",
        ),
        (EUR, "AI Office issues first transparency enforcement notice", "easy_positive"),
        (POL, "Council compromise text on AI Liability Directive leaks", "easy_positive"),
        (
            EUR,
            "European Parliament adopts resolution on foundation-model oversight",
            "easy_positive",
        ),
        (
            POL,
            "AI Office pilots risk-classification framework for deployed systems",
            "easy_positive",
        ),
        (EUR, "Trilogue resumes on AI provisions in Media Freedom Act", "hard_positive"),
        (
            POL,
            "EU justice ministers debate liability rules for agentic AI systems",
            "hard_positive",
        ),
        (
            EUR,
            "Commission clarifies scope of high-risk AI categories in guidance note",
            "hard_positive",
        ),
        (
            POL,
            "AI Office publishes joint guidance with national data-protection authorities",
            "hard_positive",
        ),
        (
            EUR,
            "Member states present implementation roadmaps for AI-Act transparency rules",
            "hard_positive",
        ),
        (
            POL,
            "Informal council compromise narrows foundation-model reporting thresholds",
            "hard_positive",
        ),
        (
            EUR,
            "European Data Protection Board opens guidelines consultation on AI training data",
            "hard_positive",
        ),
        (
            POL,
            "ECJ fast-tracks preliminary reference on AI-decision right to explanation",
            "hard_positive",
        ),
        (
            EUR,
            "Commission proposes AI-safety clauses in public-procurement review",
            "hard_positive",
        ),
        (POL, "Trilogue advances AI Liability Directive Tier 2 provisions", "hard_positive"),
    ]
    for i, (src, headline, diff) in enumerate(pos_rows):
        day = 1 + i * 1.7
        ts = (START + timedelta(days=day, hours=9)).isoformat()
        items.append(
            TimelineEntry(
                fake_ts=ts,
                source_url=src,
                headline=headline,
                difficulty=diff,
                should_notify_per_sub={SUB_AIREG: True},
                should_contribute_to_digest_per_sub={SUB_AIREG: True},
                body_style_hint="newsroom",
                body_adversarial=False,
                body_language="en",
            )
        )

    items += bulk(
        SUB_AIREG,
        START + timedelta(days=0, hours=8),
        spread_days=29,
        difficulty="easy_negative",
        positive=False,
        rows=[
            (MLB, "Paper review: a new architecture for long-context reasoning", "ml-paper"),
            (MLB, "Hackathon roundup: best projects from the Stanford AI weekend", "ml-event"),
            (ARX, "Empirical study of speculative decoding overhead", "arxiv"),
            (ARX, "Benchmark for code-generation on competitive programming", "arxiv"),
            (MLB, "Industry roundup: Q1 AI fundraises totalled $18B", "ml-industry"),
            (ARX, "Paper: sparse attention variants revisited", "arxiv"),
            (
                MLB,
                "Interview: model interpretability researcher discusses saliency tools",
                "ml-industry",
            ),
            (ARX, "Paper: adversarial prompt robustness across frontier models", "arxiv"),
            (MLB, "Opinion: scaling laws are reaching their asymptote", "ml-opinion"),
            (ARX, "Paper: encoder-only architectures for multilingual retrieval", "arxiv"),
            (MLB, "New eval: reasoning benchmarks dropped across three tracks", "ml-industry"),
            (ARX, "Paper: parameter-efficient fine-tuning on quantised weights", "arxiv"),
            (MLB, "Conference recap: highlights from NeurIPS European workshop", "ml-industry"),
            (ARX, "Paper: continuous-time diffusion applied to control tasks", "arxiv"),
            (MLB, "Industry: AI-infra unicorn cuts 18% of workforce", "ml-industry"),
            (ARX, "Paper: optimal data mixtures for instruction-tuned models", "arxiv"),
            (MLB, "Opinion: why open-weights will win the enterprise market", "ml-opinion"),
            (
                ARX,
                "Paper: test-time compute scaling laws for retrieval-augmented generation",
                "arxiv",
            ),
            (MLB, "Industry: European AI startup ecosystem map updated", "ml-industry"),
            (ARX, "Paper: human-feedback efficiency under preference noise", "arxiv"),
            (MLB, "Startup news: funding round for AI tutoring platform", "ml-industry"),
            (MLB, "Interview: RL researcher discusses credit assignment challenges", "ml-industry"),
            (MLB, "Blog: why latency matters more than accuracy for voice agents", "ml-industry"),
            (ARX, "Paper: mechanistic interpretability of induction heads revisited", "arxiv"),
            (ARX, "Paper: scaling laws for data curation", "arxiv"),
            (MLB, "Roundup: top arXiv submissions from the week", "ml-industry"),
            (ARX, "Paper: emergent world models in 2B-parameter code models", "arxiv"),
            (MLB, "Conference: accepted papers list for ICML 2026 released", "ml-industry"),
            (ARX, "Paper: on the role of layer normalization in transformer stability", "arxiv"),
            (MLB, "Opinion: the hardware lottery is not what we think", "ml-opinion"),
            (ARX, "Paper: speculative sampling with shallow drafts", "arxiv"),
            (MLB, "News: lab announces open-weights reasoning model family", "ml-industry"),
            (ARX, "Paper: evaluation reproducibility in prompt-based benchmarks", "arxiv"),
            (
                MLB,
                "Interview: MLOps lead at neobank on production latency trade-offs",
                "ml-industry",
            ),
            (ARX, "Paper: low-rank adaptation with sign-based updates", "arxiv"),
            (MLB, "Opinion: AGI timelines debate resurfaces after new benchmark", "ml-opinion"),
            (ARX, "Paper: continual learning under distribution shift", "arxiv"),
            (MLB, "News: new open dataset for tool-use traces released", "ml-industry"),
            (ARX, "Paper: calibration as a proxy for reliability in RAG", "arxiv"),
            (MLB, "Startup news: developer-tools company raises Series C", "ml-industry"),
            (ARX, "Paper: pretraining compute allocation under fixed token budgets", "arxiv"),
            (MLB, "Industry: hardware startup announces new inference accelerator", "ml-industry"),
            (ARX, "Paper: activation steering for refusal suppression study", "arxiv"),
            (MLB, "Roundup: the month in retrieval-augmented-generation research", "ml-industry"),
        ],
        style_cycle=("techcrunch", "newsroom", "wire"),
    )

    items += bulk(
        SUB_AIREG,
        START + timedelta(days=2, hours=11),
        spread_days=28,
        difficulty="near_miss_negative",
        positive=False,
        rows=[
            (
                EUR,
                "EU pharmaceutical regulation update affects clinical trial reporting",
                "eu-nonai",
            ),
            (POL, "EU competition regulator clears tech merger with remedies", "eu-nonai"),
            (
                EUR,
                "Commission proposes revisions to cybersecurity certification schemes",
                "eu-nonai",
            ),
            (POL, "Council signs off on data-retention directive update", "eu-nonai"),
            (
                EUR,
                "Brussels proposes update to copyright exception for text-and-data mining",
                "eu-ai-adjacent",
            ),
            (
                POL,
                "UK announces separate AI regulation roadmap, diverging from EU framework",
                "non-eu-ai",
            ),
            (
                EUR,
                "US executive order on AI released, analysts compare with EU AI Act",
                "non-eu-ai",
            ),
            (POL, "ISO publishes new AI risk-management technical standard", "non-eu-ai"),
            (EUR, "OECD issues non-binding guidance on generative AI governance", "non-eu-ai"),
            (POL, "China announces updates to its deep-synthesis regulations", "non-eu-ai"),
            (
                POL,
                "India releases AI advisory for platform-intermediary classifications",
                "non-eu-ai",
            ),
            (EUR, "Singapore model governance framework updated for generative AI", "non-eu-ai"),
            (
                POL,
                "Australia proposes mandatory-guardrails AI bill for high-risk contexts",
                "non-eu-ai",
            ),
            (EUR, "Japan METI guidelines balance AI promotion with risk management", "non-eu-ai"),
            (POL, "Canada tables AIDA implementing regulations consultation", "non-eu-ai"),
            (EUR, "NIST releases risk management framework supplement for LLMs", "non-eu-ai"),
            (POL, "G7 Hiroshima AI process publishes code-of-conduct progress report", "non-eu-ai"),
            (EUR, "South Korea drafts framework act on AI development", "non-eu-ai"),
            (POL, "Brazil Senate debates AI regulation bill in plenary session", "non-eu-ai"),
            (EUR, "Council of Europe AI treaty opens for signature", "non-eu-ai"),
            (EUR, "EU Media Freedom Act enters force across member states", "eu-nonai"),
            (POL, "EU platform workers directive consultation closes", "eu-nonai"),
            (EUR, "Commission proposes cyber solidarity regulation amendments", "eu-nonai"),
            (POL, "EU data-act enforcement begins with first national inspections", "eu-nonai"),
            (
                EUR,
                "European Supervisory Authorities flag Markets in Crypto-Assets gaps",
                "eu-nonai",
            ),
            (POL, "Council of the EU clears product liability directive revision", "eu-nonai"),
            (EUR, "EU cloud certification scheme public consultation closes", "eu-nonai"),
            (POL, "European Parliament adopts report on chip-shortage preparedness", "eu-nonai"),
            (EUR, "EU eID wallet technical specification updated by expert group", "eu-nonai"),
            (
                POL,
                "Commission opens infringement proceedings over DSA implementation delays",
                "eu-nonai",
            ),
            (EUR, "Digital Markets Act quarterly compliance summary published", "eu-nonai"),
            (POL, "EU chips act funding tranche allocated to Saxony project", "eu-nonai"),
            (EUR, "EU Cyber Resilience Act secondary legislation consultation opens", "eu-nonai"),
            (POL, "Informal council debates digital euro privacy safeguards", "eu-nonai"),
            (EUR, "EU Data Governance Act regulator network publishes annual report", "eu-nonai"),
            (
                POL,
                "Commissioner outlines post-2027 digital transition budget priorities",
                "eu-nonai",
            ),
            (
                EUR,
                "Network and Information Security directive national transposition update",
                "eu-nonai",
            ),
            (
                POL,
                "European Data Protection Supervisor releases AI-systems audit template",
                "eu-ai-adjacent",
            ),
            (
                EUR,
                "Horizon Europe pillar-II AI research call closes with record applications",
                "eu-ai-adjacent",
            ),
            (
                POL,
                "EU agencies finalise joint guidance on automated decision-making",
                "eu-ai-adjacent",
            ),
            (EUR, "European AI alliance annual assembly set for June 3-4", "eu-ai-adjacent"),
            (
                POL,
                "Digital Europe Programme AI testing facility tender decisions published",
                "eu-ai-adjacent",
            ),
            (
                EUR,
                "European digital innovation hub network status report released",
                "eu-ai-adjacent",
            ),
            (POL, "Joint Research Centre publishes AI watch benchmark update", "eu-ai-adjacent"),
            (
                EUR,
                "EU Parliament research service paper on deepfake provenance published",
                "eu-ai-adjacent",
            ),
            (
                POL,
                "European standardisation bodies coordinate on AI harmonised standards",
                "eu-ai-adjacent",
            ),
            (
                EUR,
                "CEPS conference recap: implementation challenges of the AI Act",
                "eu-ai-adjacent",
            ),
        ],
        style_cycle=("newsroom", "wire", "techcrunch"),
    )

    return items


def _subscription_goals() -> list[SubscriptionGoal]:
    return [
        SubscriptionGoal(
            goal_id=SUB_ENERGY,
            description=(
                "Weekly Monday 09:00 Europe/Brussels digest about EU (27-member-state) "
                "energy policy and EU climate regulation. English only. No opinion pieces."
            ),
            expected_user_spec_keywords=["EU", "energy", "climate"],
            expected_delivery_mode="digest",
            expected_schedule_cron="0 9 * * 1",
            expected_digest_language="en",
            expected_webhook_url=f"https://bench.invalid/sub/{SUB_ENERGY}/digest",
            expected_sources_min=3,
            expected_sources_max=10,
        ),
        SubscriptionGoal(
            goal_id=SUB_RARE,
            description=(
                "Real-time event notifications about rare-earth supply-chain "
                "disruptions (export bans, sanctions, plant closures, force-majeure)."
            ),
            expected_user_spec_keywords=["rare earth", "supply", "event"],
            expected_delivery_mode="event",
            expected_webhook_url=f"https://bench.invalid/sub/{SUB_RARE}/event",
            expected_sources_min=3,
            expected_sources_max=8,
        ),
        SubscriptionGoal(
            goal_id=SUB_AIREG,
            description=(
                "Daily 08:00 digest about EU AI regulation, AI Act implementation, "
                "and European AI governance. English only."
            ),
            expected_user_spec_keywords=["EU", "AI", "regulation"],
            expected_delivery_mode="digest",
            expected_schedule_cron="0 8 * * *",
            expected_digest_language="en",
            expected_webhook_url=f"https://bench.invalid/sub/{SUB_AIREG}/digest",
            expected_sources_min=2,
            expected_sources_max=8,
        ),
    ]


def _sources() -> list[SourceEntry]:
    seen: set[str] = set()
    combined: list[SourceEntry] = []
    for src in s01_skeleton._sources() + s03_skeleton._sources():
        if src.url in seen:
            continue
        seen.add(src.url)
        combined.append(src)
    ai_sources = [
        SourceEntry(
            url="https://www.euractiv.com/section/digital/feed/",
            source_type="rss",
            description="Euractiv digital / AI policy wire",
            should_be_picked_by_finder=True,
        ),
        SourceEntry(
            url="https://www.politico.eu/section/technology/feed/",
            source_type="rss",
            description="Politico Europe tech policy feed",
            should_be_picked_by_finder=True,
        ),
        SourceEntry(
            url="https://mlblog.example/feed/",
            source_type="rss",
            description="General ML industry blog",
            should_be_picked_by_finder=False,
        ),
        SourceEntry(
            url="https://arxiv.org/rss/cs.CL",
            source_type="rss",
            description="arXiv cs.CL daily listing",
            should_be_picked_by_finder=False,
        ),
    ]
    for src in ai_sources:
        if src.url not in seen:
            combined.append(src)
            seen.add(src.url)
    return combined


def _search_corpus() -> list[SearchCorpusAnchor]:
    return s01_skeleton._search_corpus() + [
        SearchCorpusAnchor(
            query_prefix="EU AI regulation RSS",
            curated_results=[
                {
                    "title": "Euractiv digital — EU AI regulation wire",
                    "url": "https://www.euractiv.com/section/digital/feed/",
                    "snippet": "Daily digital-policy coverage including EU AI Act implementation.",
                },
                {
                    "title": "Politico Europe — Technology",
                    "url": "https://www.politico.eu/section/technology/feed/",
                    "snippet": "Brussels-focused technology and AI-regulation reporting.",
                },
            ],
            fluff_count=3,
            fluff_topic_hint="US AI policy and global AI industry news",
        ),
    ]


def _scripted_turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(
            fake_day=0,
            message=(
                "Hi — I'm a Brussels policy analyst. I want three separate "
                "subscriptions: (1) a weekly Monday morning digest about EU "
                "energy policy and climate regulation, English only, no "
                "opinions; (2) real-time event alerts for rare-earth supply "
                "disruptions — export bans, plant closures, sanctions; and "
                "(3) a daily morning digest about EU AI regulation. Can you "
                "set all three up?"
            ),
            comment="Three-sub onboarding — triggers Conversational -> Discovery three times.",
        ),
        ConversationTurn(
            fake_day=8,
            message=(
                "Quick change to the AI regulation digest — can you make it "
                "twice a week (Monday and Thursday) instead of daily? Same "
                "content otherwise."
            ),
            comment="Triggers update_subscription on SUB_AIREG.",
        ),
        ConversationTurn(
            fake_day=15,
            message=(
                "Also please add this URL to the rare-earth subscription: "
                "https://www.argusmedia.com/rss/rare-earths"
            ),
            comment="Triggers add_source on SUB_RARE.",
        ),
    ]


def _assertions() -> list[AssertionSpec]:
    return [
        AssertionSpec(
            kind="subscription_exists_matching",
            payload={
                "goal_id": SUB_ENERGY,
                "expected_user_spec_keywords": ["EU", "energy", "climate"],
                "expected_schedule_cron": "0 9 * * 1",
                "expected_delivery_mode": "digest",
                "expected_digest_language": "en",
            },
        ),
        AssertionSpec(
            kind="subscription_exists_matching",
            payload={
                "goal_id": SUB_RARE,
                "expected_user_spec_keywords": ["rare earth", "supply", "event"],
                "expected_delivery_mode": "event",
            },
        ),
        AssertionSpec(
            kind="subscription_exists_matching",
            payload={
                "goal_id": SUB_AIREG,
                "expected_user_spec_keywords": ["EU", "AI", "regulation"],
                "expected_delivery_mode": "digest",
                "expected_digest_language": "en",
            },
        ),
        AssertionSpec(kind="failed_tasks_zero"),
        AssertionSpec(
            kind="digest_webhooks_delivered",
            payload={"goal_id": SUB_ENERGY, "min_count": 3, "max_count": 5},
        ),
    ]


def build() -> Scenario:
    s01 = s01_skeleton.build()
    s03 = s03_skeleton.build()

    timeline = (
        _rescope_to_single_sub(s01.timeline, SUB_ENERGY)
        + _rescope_to_single_sub(s03.timeline, SUB_RARE)
        + _ai_regulation_items()
    )

    return Scenario(
        scenario_id="s05",
        persona=Persona(
            language="en",
            timezone="Europe/Brussels",
            tech_literacy="medium",
            verbosity="medium",
            can_suggest_urls=True,
        ),
        goals=_subscription_goals(),
        simulated_days=30,
        start_date_iso=START.isoformat(),
        source_universe=_sources(),
        timeline=timeline,
        search_corpus=_search_corpus(),
        scripted_turns=_scripted_turns(),
        assertions=_assertions(),
    )
