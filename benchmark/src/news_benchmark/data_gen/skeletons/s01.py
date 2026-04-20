"""
s01_onboarding_digest — EU energy policy analyst.

Persona: Anna, a policy analyst at a Brussels NGO. She wants a weekly
Monday-morning digest about EU energy policy and climate regulation,
explicitly excluding opinion pieces. English-only. Medium tech-literacy:
she has heard of RSS, but cannot write a cron expression. She never
volunteers URLs.

This scenario exercises the happy path end-to-end:
  - Conversational onboarding -> create_subscription tool
  - Discovery spawning parallel Finders
  - Source Finder using fake web search + inline validation
  - Scheduler firing the Monday 9am digest cron 5 times in 30 simulated days
  - Digest Writer + Judge producing each weekly digest
  - Fake webhook delivery capturing the payloads

Timeline size: ~85 labeled items. Positive rate ~25%. Stupid-baseline
target F1 band: 0.50-0.70 (mix of lexically obvious and semantically
harder items forces non-trivial discrimination).
"""

from __future__ import annotations

from datetime import datetime, timedelta

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

SUB = "eu_energy_digest"
START = datetime(2026, 3, 1, 0, 0, 0).replace(microsecond=0)


def _ts(day: float, hour: int = 9, minute: int = 0) -> str:
    return (START + timedelta(days=int(day), hours=hour, minutes=minute)).isoformat()


def _e(
    day: float,
    source: str,
    headline: str,
    difficulty: str,
    notify: bool,
    contribute: bool | None = None,
    style: str = "newsroom",
    adversarial: bool = False,
    language: str = "en",
    hour: int = 9,
) -> TimelineEntry:
    if contribute is None:
        contribute = notify
    return TimelineEntry(
        fake_ts=_ts(day, hour=hour),
        source_url=source,
        headline=headline,
        difficulty=difficulty,
        should_notify_per_sub={SUB: notify},
        should_contribute_to_digest_per_sub={SUB: contribute},
        body_style_hint=style,
        body_adversarial=adversarial,
        body_language=language,
    )


def _sources() -> list[SourceEntry]:
    good = [
        ("https://www.euractiv.com/section/energy/feed/", "rss", "Euractiv energy policy wire"),
        ("https://www.politico.eu/section/energy/feed/", "rss", "Politico Europe energy"),
        ("https://www.reuters.com/pf/api/v3/feed/eu-energy", "rss", "Reuters EU energy wire"),
        ("https://www.bloomberg.com/feeds/podcasts/etf_iq.xml", "rss", "Bloomberg EU energy feed"),
        ("https://euobserver.com/feeds/energy.rss", "rss", "EU Observer energy desk"),
        (
            "https://www.euronews.com/rss?level=vertical&name=climate",
            "rss",
            "Euronews climate desk",
        ),
        ("https://reneweconomy.com.au/feed/", "rss", "Renew Economy EU coverage"),
        ("https://www.entsog.eu/rss", "rss", "ENTSOG infrastructure updates"),
    ]
    noise = [
        ("https://www.techcrunch.com/feed/", "rss", "Tech industry blog"),
        ("https://www.politico.com/rss/politics.xml", "rss", "US politics"),
        ("https://www.tmz.com/rss.xml", "rss", "Celebrity news"),
        ("https://www.reddit.com/r/gardening/.rss", "reddit", "Gardening subreddit"),
        ("https://t.me/s/crypto_whispers", "telegram", "Crypto channel"),
        ("https://www.espn.com/espn/rss/news", "rss", "Sports news"),
        ("https://www.reddit.com/r/AskHistorians/.rss", "reddit", "History Q&A subreddit"),
    ]
    out = [
        SourceEntry(url=u, source_type=t, description=d, should_be_picked_by_finder=True)
        for (u, t, d) in good
    ]
    out += [
        SourceEntry(url=u, source_type=t, description=d, should_be_picked_by_finder=False)
        for (u, t, d) in noise
    ]
    return out


def _timeline() -> list[TimelineEntry]:
    EU1 = "https://www.euractiv.com/section/energy/feed/"
    EU2 = "https://www.politico.eu/section/energy/feed/"
    REU = "https://www.reuters.com/pf/api/v3/feed/eu-energy"
    BLM = "https://www.bloomberg.com/feeds/podcasts/etf_iq.xml"
    EUO = "https://euobserver.com/feeds/energy.rss"
    EUN = "https://www.euronews.com/rss?level=vertical&name=climate"
    REN = "https://reneweconomy.com.au/feed/"
    ENT = "https://www.entsog.eu/rss"

    TCR = "https://www.techcrunch.com/feed/"
    USP = "https://www.politico.com/rss/politics.xml"
    TMZ = "https://www.tmz.com/rss.xml"
    GRD = "https://www.reddit.com/r/gardening/.rss"
    CRY = "https://t.me/s/crypto_whispers"
    ESP = "https://www.espn.com/espn/rss/news"
    HIS = "https://www.reddit.com/r/AskHistorians/.rss"

    items: list[TimelineEntry] = []

    items += [
        _e(1.2, EU1, "EU adopts binding 2030 energy efficiency regulation", "easy_positive", True),
        _e(
            3.5,
            EU2,
            "European Parliament passes Renewable Energy Directive amendment",
            "easy_positive",
            True,
        ),
        _e(
            7.3,
            REU,
            "EU climate commissioner sets new methane emissions rules for gas importers",
            "easy_positive",
            True,
        ),
        _e(
            11.1,
            BLM,
            "EU energy ministers approve capacity mechanism reforms",
            "easy_positive",
            True,
        ),
        _e(
            15.0,
            EUO,
            "Brussels finalises REPowerEU implementation guidance for 2026",
            "easy_positive",
            True,
        ),
        _e(
            20.8,
            EUN,
            "EU Commission proposes revised state-aid rules for decarbonisation projects",
            "easy_positive",
            True,
        ),
        _e(
            25.4,
            ENT,
            "EU gas network operators publish joint decarbonisation roadmap",
            "easy_positive",
            True,
        ),
    ]

    items += [
        _e(
            2.2,
            EU1,
            "Member states clash over grid-fee harmonisation proposal",
            "hard_positive",
            True,
        ),
        _e(
            4.6,
            REN,
            "Hydrogen backbone plan advances as Germany and Netherlands sign MoU",
            "hard_positive",
            True,
        ),
        _e(
            6.7,
            REU,
            "Capacity auctions restructured to favour battery storage in Iberian corridor",
            "hard_positive",
            True,
        ),
        _e(
            9.9,
            EU2,
            "Commissioner signals tighter rules on cross-border renewable PPAs",
            "hard_positive",
            True,
        ),
        _e(
            12.5,
            EUN,
            "Carbon border adjustment mechanism review finds limited leakage",
            "hard_positive",
            True,
        ),
        _e(
            14.2,
            BLM,
            "Council compromise text on electricity market reform leaks",
            "hard_positive",
            True,
        ),
        _e(
            17.6,
            ENT,
            "Grid operators warn of congestion bottlenecks in southern corridor",
            "hard_positive",
            True,
        ),
        _e(
            19.3,
            EUO,
            "New permitting rules could accelerate offshore wind buildout",
            "hard_positive",
            True,
        ),
        _e(
            22.1,
            REN,
            "Just Transition Fund disbursement schedule updated for coal regions",
            "hard_positive",
            True,
        ),
        _e(
            24.7,
            EU1,
            "Energy-intensive industries cluster seeks emergency price cap extension",
            "hard_positive",
            True,
        ),
        _e(
            27.5,
            REU,
            "Heat pump deployment targets tightened under revised EPBD",
            "hard_positive",
            True,
        ),
        _e(
            29.2,
            EU2,
            "ACER proposes new methodology for calculating network tariffs",
            "hard_positive",
            True,
        ),
    ]

    items += [
        _e(
            0.8,
            TCR,
            "Startup raises $50M to build AI-powered warehouse robots",
            "easy_negative",
            False,
        ),
        _e(1.5, TMZ, "Reality TV star spotted at LAX with new boyfriend", "easy_negative", False),
        _e(
            2.0,
            ESP,
            "NFL quarterback signs three-year extension with Chiefs",
            "easy_negative",
            False,
        ),
        _e(
            2.8,
            GRD,
            "Ask: my tomato plants wilting despite regular watering",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            3.2,
            CRY,
            "BREAKING: Whale wallet moves 4000 BTC to exchange",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            4.0,
            HIS,
            "Why did the Bronze Age collapse happen so suddenly?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            4.5,
            TCR,
            "OpenAI releases new developer API tier with caching",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(5.0, TMZ, "Oscar winner spotted buying coffee in Soho", "easy_negative", False),
        _e(5.5, ESP, "LeBron passes 41000 career points in Lakers win", "easy_negative", False),
        _e(
            6.0,
            GRD,
            "Best mulch for heavy clay soil in zone 7?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            6.3,
            CRY,
            "Altcoin rally continues as Solana hits new ATH",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            7.8,
            TCR,
            "Framework laptop ships new modular GPU option",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(8.2, ESP, "Tennis: Djokovic advances to Indian Wells final", "easy_negative", False),
        _e(
            8.9,
            HIS,
            "How did medieval cities handle sanitation?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            9.3,
            GRD,
            "Show-off: my first daffodils of the year",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(10.2, TMZ, "Pop star cancels tour dates citing laryngitis", "easy_negative", False),
        _e(
            10.9,
            TCR,
            "AWS re:Invent 2026 keynote dates announced",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            11.7,
            CRY,
            "Degen plays: memecoin of the week recap",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(12.8, ESP, "Formula 1: Verstappen takes pole in Melbourne", "easy_negative", False),
        _e(
            13.4,
            GRD,
            "Tips for starting seeds indoors under grow lights",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(13.9, TMZ, "Royal family photo controversy resurfaces", "easy_negative", False),
        _e(
            14.8,
            HIS,
            "Were vikings really as brutal as depicted on TV?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            15.5,
            TCR,
            "New React 20 release candidate drops this week",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(
            16.2,
            CRY,
            "Ethereum L2 fees drop after Dencun anniversary",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            17.0,
            ESP,
            "Champions League: Real Madrid dispatches Leipzig 3-1",
            "easy_negative",
            False,
        ),
        _e(18.5, TMZ, "Reality show drama as couple separates on air", "easy_negative", False),
        _e(
            19.8,
            GRD,
            "What to do with leftover compost in March?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            21.5,
            TCR,
            "Y Combinator unveils new AI-safety track for W26",
            "easy_negative",
            False,
            style="techcrunch",
        ),
        _e(23.2, ESP, "NBA trade deadline recap: winners and losers", "easy_negative", False),
        _e(
            24.0,
            HIS,
            "Why did Byzantium survive while the West fell?",
            "easy_negative",
            False,
            style="reddit",
        ),
        _e(
            26.1,
            CRY,
            "Japan ETF flows hit new weekly record",
            "easy_negative",
            False,
            style="telegram",
        ),
        _e(
            28.4,
            GRD,
            "Raised beds versus in-ground for a new gardener",
            "easy_negative",
            False,
            style="reddit",
        ),
    ]

    items += [
        _e(
            0.5,
            USP,
            "Biden administration proposes new US LNG export rules",
            "near_miss_negative",
            False,
        ),
        _e(
            1.8,
            USP,
            "US Senate debates clean electricity tax credit extension",
            "near_miss_negative",
            False,
        ),
        _e(
            2.9,
            USP,
            "FERC approves interconnection queue reform at federal level",
            "near_miss_negative",
            False,
        ),
        _e(
            3.8,
            USP,
            "California updates state-level cap-and-trade design",
            "near_miss_negative",
            False,
        ),
        _e(
            4.9,
            USP,
            "Texas grid operator warns of summer reliability risk",
            "near_miss_negative",
            False,
        ),
        _e(
            5.7,
            USP,
            "New York SEQRA changes may slow offshore wind permitting",
            "near_miss_negative",
            False,
        ),
        _e(
            7.0,
            REU,
            "Russia-Ukraine gas transit deal expires with no replacement agreed",
            "near_miss_negative",
            False,
        ),
        _e(
            8.4,
            REU,
            "OPEC+ considers further production cuts amid oil glut",
            "near_miss_negative",
            False,
        ),
        _e(9.0, REU, "China's coal-plant approvals reach decade high", "near_miss_negative", False),
        _e(
            10.5,
            REU,
            "Australia announces new critical-minerals export strategy",
            "near_miss_negative",
            False,
        ),
        _e(11.3, REN, "Indian solar auction attracts record-low bids", "near_miss_negative", False),
        _e(
            12.1,
            BLM,
            "African Development Bank fund for grid projects expands to $3B",
            "near_miss_negative",
            False,
        ),
        _e(
            13.6,
            EUN,
            "UN IPCC publishes Working Group III synthesis draft",
            "near_miss_negative",
            False,
        ),
        _e(
            14.9,
            EUN,
            "Paper in Nature Climate finds slower Amazon carbon uptake",
            "near_miss_negative",
            False,
        ),
        _e(
            15.9,
            EUN,
            "Glaciologists report record-low Arctic sea ice for March",
            "near_miss_negative",
            False,
        ),
        _e(
            16.8,
            REU,
            "UK government consults on contracts-for-difference round 7",
            "near_miss_negative",
            False,
        ),
        _e(
            17.3,
            REU,
            "Norway's sovereign wealth fund divests from thermal coal holdings",
            "near_miss_negative",
            False,
        ),
        _e(
            18.9,
            REN,
            "Chile proposes green hydrogen export pricing framework",
            "near_miss_negative",
            False,
        ),
        _e(20.0, REU, "South Korea unveils new nuclear buildout plan", "near_miss_negative", False),
        _e(
            21.1,
            BLM,
            "Japanese trading houses bet heavy on LNG long-term offtake",
            "near_miss_negative",
            False,
        ),
        _e(
            22.7,
            USP,
            "US DOE launches $2B grid-resilience grant program",
            "near_miss_negative",
            False,
        ),
        _e(
            23.8, USP, "EPA finalises power plant greenhouse gas rules", "near_miss_negative", False
        ),
        _e(
            25.1,
            EUN,
            "Climate scientists call for 1.5C trajectory update",
            "near_miss_negative",
            False,
        ),
        _e(
            26.4,
            REU,
            "Saudi Aramco expands petrochemicals investment into Asia",
            "near_miss_negative",
            False,
        ),
        _e(
            27.2,
            BLM,
            "Global battery-metal prices rebound after 18-month slump",
            "near_miss_negative",
            False,
        ),
        _e(
            28.1,
            USP,
            "US House Republicans block IRA-related spending bill",
            "near_miss_negative",
            False,
        ),
        _e(
            28.9,
            REN,
            "Indonesian geothermal projects win World Bank backing",
            "near_miss_negative",
            False,
        ),
        _e(
            2.4,
            REU,
            "UK passes its own Climate and Energy Security Act outside EU framework",
            "near_miss_negative",
            False,
        ),
        _e(
            3.9,
            REU,
            "Norway updates its offshore wind regulation for North Sea operations",
            "near_miss_negative",
            False,
        ),
        _e(
            4.3,
            EUN,
            "Brussels court delivers ruling on financial-services regulation",
            "near_miss_negative",
            False,
        ),
        _e(
            5.2,
            EU1,
            "EU agricultural policy reform advances in Council",
            "near_miss_negative",
            False,
        ),
        _e(
            6.5,
            EU2,
            "European Commission opens infringement case over data-retention directive",
            "near_miss_negative",
            False,
        ),
        _e(
            8.0,
            EUN,
            "Switzerland signs bilateral electricity agreement with EU without joining market",
            "near_miss_negative",
            False,
        ),
        _e(
            9.7,
            EU1,
            "EU foreign policy chief meets Chinese counterpart on climate diplomacy",
            "near_miss_negative",
            False,
        ),
        _e(
            11.9,
            EU2,
            "EU digital services regulation takes effect for large platforms",
            "near_miss_negative",
            False,
        ),
        _e(
            13.0,
            EUO,
            "European Court of Auditors releases report on cohesion fund spending",
            "near_miss_negative",
            False,
        ),
        _e(
            15.7,
            REU,
            "Turkey announces energy-strategy update referencing EU neighbour policy",
            "near_miss_negative",
            False,
        ),
        _e(
            17.9,
            EUN,
            "Climate activists stage protest outside European Parliament on transport policy",
            "near_miss_negative",
            False,
        ),
        _e(
            20.4,
            EU2,
            "EU customs regulation update affects cross-border e-commerce traders",
            "near_miss_negative",
            False,
        ),
        _e(
            22.3,
            EU1,
            "European Investment Bank reports record lending for SME programs",
            "near_miss_negative",
            False,
        ),
        _e(
            24.3,
            EUO,
            "Brussels proposes new framework for pharmaceutical regulation",
            "near_miss_negative",
            False,
        ),
        _e(
            26.7,
            EUN,
            "EU competition commissioner clears telecom merger with remedies",
            "near_miss_negative",
            False,
        ),
    ]

    items += [
        _e(
            6.2,
            EU2,
            "The EU's energy policy is a disaster and here is why",
            "adversarial",
            False,
            style="opinion",
            adversarial=False,
        ),
        _e(
            13.2,
            EUO,
            "Opinion: Brussels' climate agenda sacrifices working families",
            "adversarial",
            False,
            style="opinion",
            adversarial=False,
        ),
        _e(
            16.5,
            EU1,
            "Shocking secret about REPowerEU that Brussels wont tell you",
            "adversarial",
            False,
            style="techcrunch",
            adversarial=True,
        ),
        _e(
            23.0,
            EUN,
            "Brussels cancels all energy regulation, sources say",
            "adversarial",
            False,
            style="newsroom",
            adversarial=True,
        ),
    ]

    items += [
        _e(
            7.4,
            EU1,
            "EU climate commissioner sets new methane emissions rules for gas importers (updated)",
            "duplicate",
            True,
        ),
        _e(7.9, EUN, "New methane rules for EU gas importers take effect", "duplicate", True),
        _e(
            21.0,
            EU2,
            "European Parliament passes Renewable Energy Directive amendment (wire)",
            "duplicate",
            True,
            style="wire",
        ),
    ]

    bulk_easy_neg: list[tuple[str, str, str]] = [
        (TCR, "Anthropic releases Claude 5 with 10M context window", "tech"),
        (TCR, "Vercel acquires design-systems startup for $180M", "tech"),
        (TCR, "Databricks lands $10B secondary at $62B valuation", "tech"),
        (TCR, "Nvidia H300 launch expected Q3 per supply chain leak", "tech"),
        (TCR, "GitLab lays off 7% of workforce as it pivots to AI", "tech"),
        (TCR, "Microsoft Copilot gets agent-marketplace tab", "tech"),
        (TCR, "Figma raises prices after IPO", "tech"),
        (TCR, "Stripe reports record-breaking quarterly volume", "tech"),
        (TCR, "Y Combinator reveals Fall 26 batch demo-day schedule", "tech"),
        (TCR, "Hugging Face hits 2 million open-weight checkpoints", "tech"),
        (TMZ, "Celebrity couple announces surprise engagement on Instagram", "celeb"),
        (TMZ, "Reality TV star launches skincare line", "celeb"),
        (TMZ, "Oscar nominee spotted at Coachella with mystery partner", "celeb"),
        (TMZ, "Pop star tour cancellation refund policy details released", "celeb"),
        (TMZ, "Late-night host announces exit after 15-year run", "celeb"),
        (TMZ, "Streaming show renewed for sixth season", "celeb"),
        (TMZ, "Awards committee finalises hosts for the next ceremony", "celeb"),
        (TMZ, "Pop star denies tour cancellation rumours", "celeb"),
        (TMZ, "Reality contestant sparks debate with wedding speech", "celeb"),
        (ESP, "Champions League: Bayern edges Arsenal in extra time", "sports"),
        (ESP, "NFL: Chiefs trade star receiver to Dolphins", "sports"),
        (ESP, "NBA: Wembanyama posts career-high 54 points", "sports"),
        (ESP, "Olympic qualifier wraps: team USA tops medal tally", "sports"),
        (ESP, "Cricket World Cup: India defends title in nail-biter final", "sports"),
        (ESP, "Formula E adds Seoul street circuit to 2027 schedule", "sports"),
        (ESP, "Rugby: All Blacks announce squad changes for autumn tour", "sports"),
        (ESP, "Boxing: heavyweight title unification set for December", "sports"),
        (ESP, "Tennis: next-gen final showcases 21-year-old stars", "sports"),
        (GRD, "Raised beds: best drainage layer materials debated", "garden"),
        (GRD, "Companion planting guide for tomatoes this season", "garden"),
        (GRD, "What fertilizer for blueberry bushes in zone 6?", "garden"),
        (GRD, "Show-off: finally pruned that hedge after three years", "garden"),
        (GRD, "Pest control without pesticides — share tips", "garden"),
        (GRD, "Seed swap in Ghent next weekend — anyone coming?", "garden"),
        (GRD, "First azalea bloom of the year photo thread", "garden"),
        (GRD, "How do you protect seedlings from late frost?", "garden"),
        (CRY, "Bitcoin ETF weekly flows: $1.2B net inflow", "crypto"),
        (CRY, "Altcoin of the week: Jupiter Exchange pumps 40%", "crypto"),
        (CRY, "Whale watch: 3200 BTC moved to Coinbase", "crypto"),
        (CRY, "Ethereum staking yield drops below 3%", "crypto"),
        (CRY, "Degen plays: new memecoin launch on Solana", "crypto"),
        (CRY, "Regulatory: SEC delays Solana ETF decision", "crypto"),
        (CRY, "Stablecoin supply crosses $250B milestone", "crypto"),
        (CRY, "Layer-2 roundup: Arbitrum beats Optimism in TVL again", "crypto"),
        (HIS, "Why did the Ottomans fail to take Vienna in 1683?", "history"),
        (HIS, "How accurate is the Vikings TV show?", "history"),
        (HIS, "What was daily life like for a medieval peasant?", "history"),
        (HIS, "Did Rome really salt Carthage's fields?", "history"),
        (HIS, "Who invented the umbrella, and why so late?", "history"),
        (HIS, "How did Genghis Khan organise his army?", "history"),
        (HIS, "What caused the fall of Knossos?", "history"),
        (HIS, "Was the Library of Alexandria really destroyed by fire?", "history"),
        (HIS, "How did samurai actually live in peacetime?", "history"),
        (HIS, "When did forks become standard cutlery in Europe?", "history"),
    ]
    items += bulk(
        SUB,
        START + timedelta(hours=10),
        spread_days=29,
        difficulty="easy_negative",
        positive=False,
        rows=bulk_easy_neg,
        style_cycle=("newsroom", "techcrunch", "reddit", "telegram"),
    )

    bulk_near_miss: list[tuple[str, str, str]] = [
        (EU1, "EU pharmaceutical regulation revision enters trilogue", "eu-nonenergy"),
        (EU2, "EU Commission unveils AI Liability Directive amendments", "eu-nonenergy"),
        (EUO, "European Council adopts new sanctions package over Belarus", "eu-nonenergy"),
        (EUN, "EU foreign policy chief visits Addis Ababa for trade talks", "eu-nonenergy"),
        (EU1, "European Central Bank keeps rates unchanged at April meeting", "eu-nonenergy"),
        (EU2, "EU migration pact implementation status report published", "eu-nonenergy"),
        (EUO, "EU fisheries council agrees 2027 quota for North Sea", "eu-nonenergy"),
        (EUN, "Brussels proposes minimum corporate tax floor revisions", "eu-nonenergy"),
        (EU1, "EU ombudsman launches inquiry into lobbying disclosure rules", "eu-nonenergy"),
        (EU2, "European Parliament adopts resolution on media freedom", "eu-nonenergy"),
        (EUO, "Commission proposes changes to Erasmus programme funding", "eu-nonenergy"),
        (EUN, "EU justice ministers discuss cross-border digital evidence", "eu-nonenergy"),
        (EU1, "ECJ ruling clarifies GDPR cross-border enforcement mechanism", "eu-nonenergy"),
        (EU2, "EU telecom ministers debate 5G rollout targets", "eu-nonenergy"),
        (EUO, "European Investment Bank backs rail-infrastructure upgrade", "eu-nonenergy"),
        (EUN, "EU competition authority clears airline merger with remedies", "eu-nonenergy"),
        (REU, "UK energy secretary announces North Sea licensing round", "non-eu-energy"),
        (REU, "US Senate votes down clean-energy tax credit extension", "non-eu-energy"),
        (REU, "Norway expands offshore wind tender to Sørlige Nordsjø", "non-eu-energy"),
        (REU, "Switzerland extends nuclear operating permits", "non-eu-energy"),
        (REU, "Turkey signs LNG supply deal with Azerbaijan", "non-eu-energy"),
        (REU, "Japan restarts Kashiwazaki-Kariwa reactor unit", "non-eu-energy"),
        (REU, "Australia approves new offshore LNG export project", "non-eu-energy"),
        (REU, "Canada Energy Regulator approves pipeline expansion", "non-eu-energy"),
        (REU, "Saudi Aramco raises capex guidance for 2027", "non-eu-energy"),
        (REU, "South Korea announces nuclear export roadmap", "non-eu-energy"),
        (REU, "Indonesia scales back coal plant pipeline", "non-eu-energy"),
        (REU, "Argentina opens Vaca Muerta to new operators", "non-eu-energy"),
        (REU, "Russia extends oil export duty reduction", "non-eu-energy"),
        (REU, "China's NEA publishes revised solar capacity targets", "non-eu-energy"),
        (BLM, "Global LNG prices slide on weaker Asian demand", "non-eu-energy"),
        (BLM, "US Henry Hub gas futures close at 14-month low", "non-eu-energy"),
        (BLM, "OPEC+ technical committee convenes ahead of June meet", "non-eu-energy"),
        (BLM, "Singapore bunker-fuel prices reach multi-year peak", "non-eu-energy"),
        (BLM, "IEA revises global oil-demand outlook upward", "non-eu-energy"),
        (EUN, "IPCC publishes regional assessment for South Asia", "climate-science"),
        (EUN, "Nature Climate paper links permafrost thaw to methane surge", "climate-science"),
        (EUN, "Arctic sea-ice volume hits March low per NSIDC", "climate-science"),
        (EUN, "Amazon rainforest carbon sink weaker than modeled", "climate-science"),
        (EUN, "Scientific report: coral bleaching accelerates on GBR", "climate-science"),
        (EUN, "Researchers map record-low Antarctic sea-ice extent", "climate-science"),
        (EUN, "Cloud-feedback paper rekindles climate-sensitivity debate", "climate-science"),
        (EUN, "Study identifies tipping-point threshold for West Antarctic ice", "climate-science"),
        (EUN, "Ocean heat content sets new annual record, study finds", "climate-science"),
        (EUN, "New attribution study ties 2025 heat wave to climate change", "climate-science"),
        (EU1, "Opinion: the problem with EU industrial policy is timing", "editorial"),
        (EU2, "Commentary: why the EU must rethink its trade strategy", "editorial"),
        (EUO, "Op-ed: Brussels' digital rules squeeze startups", "editorial"),
        (EUN, "Editorial: the case for a looser fiscal framework in the EU", "editorial"),
        (REN, "Long-read: five years of REPowerEU — what worked, what didn't", "analysis"),
        (REN, "Analysis: why auction design matters more than headline targets", "analysis"),
        (REN, "Backgrounder: how the CBAM calculation actually works", "analysis"),
        (REN, "Explainer: electricity market reform jargon decoded", "analysis"),
        (REN, "Analysis: the offshore wind supply-chain bottleneck", "analysis"),
        (REN, "Podcast recap: monthly EU energy-policy roundtable", "analysis"),
        (REN, "Methodology note: calculating national heat-pump uptake", "analysis"),
        (REN, "Research digest: papers on electricity market design", "analysis"),
        (REU, "UK passes Climate and Energy Security Act standalone of EU", "non-eu-energy"),
        (REU, "Washington DC climate summit concludes without binding deal", "non-eu-energy"),
        (REU, "Moscow announces new energy-efficiency targets for industry", "non-eu-energy"),
        (REU, "Ankara unveils domestic solar manufacturing scheme", "non-eu-energy"),
        (REU, "Pretoria grants incentives for coal-to-hydrogen pilot", "non-eu-energy"),
    ]
    items += bulk(
        SUB,
        START + timedelta(hours=13),
        spread_days=29,
        difficulty="near_miss_negative",
        positive=False,
        rows=bulk_near_miss,
        style_cycle=("newsroom", "wire", "techcrunch"),
    )

    return items


def _search_corpus() -> list[SearchCorpusAnchor]:
    return [
        SearchCorpusAnchor(
            query_prefix="EU energy policy RSS",
            curated_results=[
                {
                    "title": "Euractiv Energy — EU energy policy wire",
                    "url": "https://www.euractiv.com/section/energy/feed/",
                    "snippet": "Daily wire covering EU energy policy, regulation, and grids.",
                },
                {
                    "title": "Politico Europe — Energy",
                    "url": "https://www.politico.eu/section/energy/feed/",
                    "snippet": "Brussels-focused energy and climate reporting.",
                },
                {
                    "title": "EU Observer — Energy desk",
                    "url": "https://euobserver.com/feeds/energy.rss",
                    "snippet": "Independent Brussels reporting on EU energy regulation.",
                },
            ],
            fluff_count=4,
            fluff_topic_hint="general energy news; not EU-specific",
        ),
        SearchCorpusAnchor(
            query_prefix="best RSS feeds EU climate regulation",
            curated_results=[
                {
                    "title": "Euronews — Climate vertical",
                    "url": "https://www.euronews.com/rss?level=vertical&name=climate",
                    "snippet": "EU-focused climate policy reporting in English.",
                },
                {
                    "title": "Reuters — EU energy wire",
                    "url": "https://www.reuters.com/pf/api/v3/feed/eu-energy",
                    "snippet": "Reuters dedicated EU energy reporting feed.",
                },
            ],
            fluff_count=4,
            fluff_topic_hint="general climate science blogs; not EU policy",
        ),
        SearchCorpusAnchor(
            query_prefix="REPowerEU implementation status",
            curated_results=[
                {
                    "title": "EU Observer: REPowerEU 2026 implementation",
                    "url": "https://euobserver.com/feeds/energy.rss",
                    "snippet": "Roll-up of REPowerEU status reports.",
                },
            ],
            fluff_count=3,
            fluff_topic_hint="US energy policy",
        ),
        SearchCorpusAnchor(
            query_prefix="EU methane emissions rules gas importers",
            curated_results=[
                {
                    "title": "Euractiv: new methane rules for EU importers",
                    "url": "https://www.euractiv.com/section/energy/feed/",
                    "snippet": "Summary of the new methane rules proposed by DG ENER.",
                },
            ],
            fluff_count=3,
            fluff_topic_hint="US methane policy and oil-and-gas industry news",
        ),
        SearchCorpusAnchor(
            query_prefix="hydrogen backbone plan Germany Netherlands",
            curated_results=[
                {
                    "title": "Renew Economy: Germany–NL hydrogen MoU",
                    "url": "https://reneweconomy.com.au/feed/",
                    "snippet": "Coverage of the joint hydrogen-backbone MoU.",
                },
            ],
            fluff_count=3,
            fluff_topic_hint="hydrogen industry generally; not EU policy",
        ),
    ]


def _scripted_turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(
            fake_day=0,
            message=(
                "Hi — I'm a policy analyst at a Brussels NGO. I want a weekly "
                "newsletter, in English, covering EU energy policy and climate "
                "regulation. I don't want opinion pieces, just reporting. "
                "Can you set that up to arrive on Monday mornings?"
            ),
            comment="Initial onboarding — triggers Conversational -> Discovery.",
        ),
    ]


def _assertions() -> list[AssertionSpec]:
    return [
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
            payload={"goal_id": SUB, "min_count": 4, "max_count": 5},
        ),
        AssertionSpec(
            kind="sources_within_bounds",
            payload={"goal_id": SUB, "min": 3, "max": 10},
        ),
        AssertionSpec(
            kind="sources_are_from_good_pool",
            payload={"goal_id": SUB, "max_noise_sources": 1},
        ),
    ]


def build() -> Scenario:
    return Scenario(
        scenario_id="s01",
        persona=Persona(
            language="en",
            timezone="Europe/Brussels",
            tech_literacy="medium",
            verbosity="medium",
            can_suggest_urls=False,
        ),
        goals=[
            SubscriptionGoal(
                goal_id=SUB,
                description=(
                    "Weekly Monday 09:00 Europe/Brussels digest about EU "
                    "(27-member-state) energy policy and EU climate regulation "
                    "SPECIFICALLY: electricity markets, grid, renewables, gas, "
                    "hydrogen, decarbonisation directives, methane rules, carbon "
                    "border adjustment, state-aid for clean energy. "
                    "EXCLUDE: non-EU geographies (UK, US, Norway, Switzerland, "
                    "Russia, Asia), non-energy EU regulation (digital, agricultural, "
                    "financial, customs, transport, pharma), opinion pieces and "
                    "editorials, climate-science papers without policy relevance. "
                    "English only."
                ),
                expected_user_spec_keywords=["EU", "energy", "climate", "regulation", "policy"],
                expected_delivery_mode="digest",
                expected_schedule_cron="0 9 * * 1",
                expected_digest_language="en",
                expected_webhook_url=f"https://bench.invalid/sub/{SUB}/digest",
                expected_sources_min=3,
                expected_sources_max=10,
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
