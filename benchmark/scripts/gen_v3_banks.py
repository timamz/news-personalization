"""Generate bulk headline banks for v3 scenario expansion.

Calls an LLM in small batches to produce unique headlines per (scenario,
tier, theme, source-URL) tuple. Writes the output as JSON banks the
skeleton files load via ``_load_v3_bank``. Run once; commit the banks
alongside the skeletons.

Usage:
    uv run python scripts/gen_v3_banks.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_env() -> None:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / ".env", here.parent.parent / "backend" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_env()

import litellm  # noqa: E402

MODEL = os.environ.get("BENCHMARK_DATAGEN_MODEL", "openai/gpt-5.4-nano")

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"


# Theme specs — each produces a bank of N unique headlines.
#
# scenario: which skeleton this gets added to (s01, s03, s05-ai)
# tier: difficulty tier (easy_negative, near_miss_negative)
# source_urls: rotate among these
# topic: short hint for the LLM prompt
# instructions: the gen instructions the LLM uses
# count: how many to generate
# style_cycle: passed to bulk() at skeleton side
# examples: 3-6 existing seed examples for tone guidance
THEMES: list[dict[str, Any]] = [
    # ---------------- s01 expansion (eu_energy_digest) ----------------
    {
        "scenario": "s01",
        "tier": "near_miss_negative",
        "theme": "eu-nonenergy",
        "source_urls": [
            "https://www.euractiv.com/section/energy/feed/",
            "https://www.politico.eu/section/energy/feed/",
            "https://euobserver.com/feeds/energy.rss",
            "https://www.euronews.com/rss?level=vertical&name=climate",
        ],
        "topic": (
            "Non-energy EU-institutional regulation — e.g. pharma, digital,"
            " agriculture, fisheries, migration, trade, customs, ECB, ECJ,"
            " competition, merger clearance, GDPR, telecom 5G, erasmus,"
            " justice, sanctions on third countries."
        ),
        "instructions": (
            "Write news-wire headlines about non-energy EU regulation. Each"
            " MUST mention an EU body (Commission/Parliament/Council/ECB/ECJ/"
            "ECA/EIB) or Brussels-based process (trilogue, Member State"
            " council, negotiation) to share vocabulary with EU-energy positives."
            " But topic must be NOT about energy or climate: pharma/digital/"
            "agriculture/fisheries/migration/trade/customs/telecom/justice/banking."
            " No duplicates. Distinctive entity names (real EU dossiers)."
        ),
        "count": 400,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "EU pharmaceutical regulation revision enters trilogue",
            "EU Commission unveils AI Liability Directive amendments",
            "European Council adopts new sanctions package over Belarus",
            "European Parliament adopts resolution on media freedom",
            "ECJ ruling clarifies GDPR cross-border enforcement mechanism",
        ],
    },
    {
        "scenario": "s01",
        "tier": "near_miss_negative",
        "theme": "non-eu-energy",
        "source_urls": [
            "https://www.reuters.com/pf/api/v3/feed/eu-energy",
            "https://www.bloomberg.com/feeds/podcasts/etf_iq.xml",
            "https://reneweconomy.com.au/feed/",
        ],
        "topic": (
            "Energy policy outside the EU-27 — UK, US, Norway, Switzerland,"
            " Turkey, Russia, China, Japan, Korea, Indonesia, Australia,"
            " Canada, India, Brazil, Argentina, Saudi Arabia, South Africa,"
            " Mexico, Chile, Vietnam, Thailand, Nigeria, Angola."
        ),
        "instructions": (
            "Write news-wire headlines about ENERGY policy and decisions in"
            " non-EU countries. Share vocabulary with EU-energy positives"
            " (renewable, grid, LNG, nuclear, CCS, hydrogen, offshore wind,"
            " capacity auction, tariff) but geography MUST be outside the"
            " EU-27. Use specific country/agency names so it's clear."
        ),
        "count": 350,
        "style_cycle": ("newsroom", "wire", "techcrunch"),
        "examples": [
            "UK energy secretary announces North Sea licensing round",
            "US Senate votes down clean-energy tax credit extension",
            "Norway expands offshore wind tender to Sorlige Nordsjo",
            "Japan restarts Kashiwazaki-Kariwa reactor unit",
            "Indonesia scales back coal plant pipeline",
            "Saudi Aramco raises capex guidance for 2027",
        ],
    },
    {
        "scenario": "s01",
        "tier": "near_miss_negative",
        "theme": "climate-science",
        "source_urls": [
            "https://www.euronews.com/rss?level=vertical&name=climate",
            "https://reneweconomy.com.au/feed/",
        ],
        "topic": (
            "Peer-reviewed climate science findings, data releases, attribution"
            " studies — IPCC, Nature Climate, Copernicus, NSIDC, NOAA, ECMWF,"
            " regional climate assessments, satellite data releases, model"
            " intercomparison, tipping-point research."
        ),
        "instructions": (
            "Write news-wire headlines about CLIMATE SCIENCE research — studies,"
            " papers, data releases, observations. Share 'climate' vocabulary"
            " with positives but topic is SCIENCE, not policy/regulation. No"
            " mention of EU institutions or regulatory action."
        ),
        "count": 300,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "IPCC publishes regional assessment for South Asia",
            "Nature Climate paper links permafrost thaw to methane surge",
            "Arctic sea-ice volume hits March low per NSIDC",
            "Amazon rainforest carbon sink weaker than modeled",
            "Ocean heat content sets new annual record, study finds",
        ],
    },
    {
        "scenario": "s01",
        "tier": "near_miss_negative",
        "theme": "eu-energy-editorial",
        "source_urls": [
            "https://www.euractiv.com/section/energy/feed/",
            "https://www.politico.eu/section/energy/feed/",
            "https://euobserver.com/feeds/energy.rss",
            "https://reneweconomy.com.au/feed/",
        ],
        "topic": (
            "Opinion pieces, op-eds, editorials, commentary, analysis, explainers,"
            " backgrounders, podcast recaps about EU energy policy."
        ),
        "instructions": (
            "Write headlines for OPINION/EDITORIAL/ANALYSIS pieces about EU"
            " energy and climate policy. These ARE on-topic by domain but the"
            " user excluded opinion pieces. Start each with a marker like"
            " Opinion:/Op-ed:/Editorial:/Commentary:/Long-read:/Analysis:/"
            " Explainer:/Backgrounder:/Podcast recap:/Methodology note:."
        ),
        "count": 300,
        "style_cycle": ("opinion", "newsroom"),
        "examples": [
            "Opinion: the problem with EU industrial policy is timing",
            "Commentary: why the EU must rethink its trade strategy",
            "Editorial: the case for a looser fiscal framework in the EU",
            "Long-read: five years of REPowerEU what worked, what didn't",
            "Explainer: electricity market reform jargon decoded",
        ],
    },
    {
        "scenario": "s01",
        "tier": "near_miss_negative",
        "theme": "eu-energy-markets-commercial",
        "source_urls": [
            "https://www.bloomberg.com/feeds/podcasts/etf_iq.xml",
            "https://reneweconomy.com.au/feed/",
            "https://www.entsog.eu/rss",
        ],
        "topic": (
            "Corporate earnings, M&A, utility dividends, stock moves, ETF flows,"
            " bond issuance, CEO changes, financial guidance in the EU energy"
            " sector. Commercial news, not regulatory/policy action."
        ),
        "instructions": (
            "Write headlines about EU ENERGY SECTOR COMPANIES' commercial news:"
            " earnings, M&A, dividends, bond issuance, CEO changes, stock"
            " analyst ratings, ETF flows. Vocabulary should overlap with"
            " positives (energy, grid, renewable, utility) but the event is"
            " corporate/financial, not regulatory."
        ),
        "count": 300,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "Iberdrola lifts full-year guidance on renewables margin",
            "RWE dividend increase proposed ahead of AGM",
            "Enel sells Argentine assets, books EUR 400M gain",
            "Engie CFO departs for consulting role",
            "Orsted downgraded to hold by JP Morgan analysts",
        ],
    },
    {
        "scenario": "s01",
        "tier": "easy_negative",
        "theme": "off-topic-tech-sports-celeb",
        "source_urls": [
            "https://www.techcrunch.com/feed/",
            "https://www.politico.com/rss/politics.xml",
            "https://www.tmz.com/rss.xml",
            "https://www.reddit.com/r/gardening/.rss",
            "https://t.me/s/crypto_whispers",
            "https://www.espn.com/espn/rss/news",
            "https://www.reddit.com/r/AskHistorians/.rss",
        ],
        "topic": (
            "Plainly unrelated news: US politics, sports, celebrity gossip,"
            " crypto, gardening Q&A, history Q&A, consumer tech launches,"
            " streaming entertainment."
        ),
        "instructions": (
            "Write headlines that are VERY CLEARLY unrelated to EU energy/"
            "climate policy. Topics: tech launches (NOT energy tech), US"
            " politics, sports, celebrity gossip, crypto, gardening, history"
            " questions, streaming/gaming. Make them sound like real RSS"
            " titles from tech/sports/celeb sites."
        ),
        "count": 450,
        "style_cycle": ("newsroom", "techcrunch", "reddit", "telegram"),
        "examples": [
            "Startup raises $50M to build AI-powered warehouse robots",
            "Reality TV star spotted at LAX with new boyfriend",
            "NFL quarterback signs three-year extension with Chiefs",
            "Ask: my tomato plants wilting despite regular watering",
            "BREAKING: Whale wallet moves 4000 BTC to exchange",
        ],
    },
    # ---------------- s03 expansion (rare_earth_events) ----------------
    {
        "scenario": "s03",
        "tier": "near_miss_negative",
        "theme": "rare-earth-routine",
        "source_urls": [
            "https://www.reuters.com/pf/api/v3/feed/metals",
            "https://www.bloomberg.com/feeds/metals.xml",
            "https://www.argusmedia.com/rss/rare-earths",
            "https://www.fastmarkets.com/rss/rare-earths",
        ],
        "topic": (
            "Routine rare-earth coverage: weekly price wraps, inventory updates,"
            " analyst notes, company guidance, conference recaps, podcasts,"
            " ETF flows, CEO profiles, quarterly earnings, capex calls."
        ),
        "instructions": (
            "Write headlines that sound like ROUTINE rare-earth coverage"
            " (neodymium, dysprosium, lanthanum, praseodymium, terbium, cerium,"
            " yttrium). Topics: price wraps, inventory, earnings, guidance,"
            " conferences, analyst calls, ETF flows, podcasts, recaps,"
            " backgrounders. Share vocabulary with supply-shock positives but"
            " describe routine activity (no halts, disruptions, bans, strikes)."
        ),
        "count": 350,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "Rare-earth weekly wrap: praseodymium flat, terbium steady",
            "MP Materials guides Q2 rare-earth output toward prior-quarter figure",
            "Rare-earth conference in Perth wraps with industry-panel recap",
            "Rare-earth recycling startup raises Series B funding",
            "Analyst notes: Lynas valuation attractive on stable dysprosium outlook",
        ],
    },
    {
        "scenario": "s03",
        "tier": "near_miss_negative",
        "theme": "other-commodities",
        "source_urls": [
            "https://www.reuters.com/pf/api/v3/feed/metals",
            "https://www.bloomberg.com/feeds/metals.xml",
            "https://www.argusmedia.com/rss/rare-earths",
            "https://www.fastmarkets.com/rss/rare-earths",
        ],
        "topic": (
            "Other critical-minerals/metals commodities: lithium, cobalt,"
            " nickel, copper, tin, graphite, platinum, palladium, tungsten,"
            " vanadium, gallium, indium, molybdenum, uranium, cadmium,"
            " selenium, bismuth, antimony, germanium, beryllium, zirconium,"
            " iron ore, steel."
        ),
        "instructions": (
            "Write headlines about NON-rare-earth critical minerals and metals"
            " commodities. Include SUPPLY-SHOCK events (halts, bans, strikes,"
            " force majeure) on these OTHER commodities — which share"
            " vocabulary with rare-earth positives but are scope-wrong. User"
            " only wants rare-earth events, not lithium/cobalt/copper events."
        ),
        "count": 350,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "Chile declares force majeure at Escondida copper mine",
            "Indonesia halts nickel ore exports for 30 days",
            "Lithium producer SQM downgrades output guidance",
            "Zambia declares state of emergency at copper processing hub",
            "Tungsten prices jump on Vietnam licensing row",
        ],
    },
    {
        "scenario": "s03",
        "tier": "easy_negative",
        "theme": "off-topic-noise",
        "source_urls": [
            "https://techcrunch.com/feed/",
            "https://www.reddit.com/r/GlobalTalk/.rss",
            "https://t.me/s/random_finance",
        ],
        "topic": (
            "General noise far from rare-earth commodities: tech launches,"
            " global politics discussions, consumer finance, random finance"
            " rumours, gaming, sports, celebrity."
        ),
        "instructions": (
            "Write headlines that are CLEARLY unrelated to rare-earth metals"
            " or any commodity supply-chain. Random tech, finance gossip,"
            " subreddit discussions, consumer topics."
        ),
        "count": 300,
        "style_cycle": ("newsroom", "techcrunch", "reddit", "telegram"),
        "examples": [
            "Startup launches AI tutor for piano lessons",
            "Global Talk thread: which airline has the best economy class?",
            "Random finance: Fed governor hints at dovish turn",
            "Gaming update: new season of popular battle royale goes live",
        ],
    },
    # ---------------- s05 expansion (eu_ai_regulation_digest) ----------------
    {
        "scenario": "s05-ai",
        "tier": "near_miss_negative",
        "theme": "non-ai-eu-regulation",
        "source_urls": [
            "https://www.euractiv.com/section/digital/feed/",
            "https://www.politico.eu/section/technology/feed/",
            "https://www.reuters.com/technology/rss",
            "https://digital-strategy.ec.europa.eu/feed",
        ],
        "topic": (
            "Non-AI EU digital/platform/telecom regulation: DSA/DMA enforcement,"
            " data-governance, GDPR enforcement actions, telecom 5G, platform"
            " liability, content moderation, data-center infrastructure,"
            " cybersecurity directives. All digital-adjacent but NOT AI-specific."
        ),
        "instructions": (
            "Write headlines about EU digital/tech regulation that is NOT about"
            " AI/machine-learning specifically. DSA/DMA/GDPR enforcement,"
            " telecom, cybersecurity, data governance. Share vocabulary"
            " (Commission, Parliament, directive, enforcement, tech) but"
            " topic scope is NOT AI."
        ),
        "count": 300,
        "style_cycle": ("newsroom", "wire"),
        "examples": [
            "DSA enforcement action: Commission fines VLOP for ad-transparency lapse",
            "DMA gatekeeper list updated with additional platforms",
            "GDPR cross-border ruling clarifies data-subject rights",
            "EU cybersecurity agency publishes incident-reporting guidance",
            "EU telecom council debates spectrum allocation for 6G trials",
        ],
    },
    {
        "scenario": "s05-ai",
        "tier": "near_miss_negative",
        "theme": "non-eu-ai-news",
        "source_urls": [
            "https://www.reuters.com/technology/rss",
            "https://techcrunch.com/feed/",
            "https://www.theverge.com/rss/index.xml",
        ],
        "topic": (
            "AI news outside EU jurisdiction: US/UK/Chinese AI policy, company"
            " product launches, model releases, research labs, benchmark"
            " results, startup funding rounds."
        ),
        "instructions": (
            "Write headlines about AI/ML news OUTSIDE of EU regulatory scope:"
            " US executive orders, UK AI Safety Institute reports, China"
            " AI strategy, company model launches (GPT/Claude/Gemini/Llama),"
            " academic research, startup funding. Share AI vocabulary but"
            " NOT EU-regulatory."
        ),
        "count": 300,
        "style_cycle": ("newsroom", "techcrunch"),
        "examples": [
            "OpenAI releases GPT-6 model family with extended context",
            "Anthropic publishes updated responsible scaling policy",
            "UK AI Safety Institute benchmarks new frontier model",
            "US executive order on AI procurement signed",
            "Chinese AI startup raises $1B Series C",
        ],
    },
    {
        "scenario": "s05-ai",
        "tier": "easy_negative",
        "theme": "off-topic-general",
        "source_urls": [
            "https://techcrunch.com/feed/",
            "https://www.reuters.com/technology/rss",
            "https://www.theverge.com/rss/index.xml",
        ],
        "topic": (
            "Tech news that is not AI-regulation at all: hardware, consumer"
            " apps, gaming, crypto, Web3, streaming, chip industry mergers,"
            " phone launches."
        ),
        "instructions": (
            "Write headlines about tech news CLEARLY unrelated to AI or EU"
            " regulation: consumer hardware, gaming, crypto, streaming, phone"
            " launches, chip mergers, venture news on non-AI topics."
        ),
        "count": 250,
        "style_cycle": ("newsroom", "techcrunch"),
        "examples": [
            "Meta unveils third-generation VR headset at developer conference",
            "New console launches with exclusive lineup for holidays",
            "Crypto exchange FTX successor files for IPO",
            "Streaming service announces price tier restructuring",
        ],
    },
]


async def generate_theme(theme: dict[str, Any]) -> list[str]:
    """Return ``count`` unique headlines for this theme via LLM."""
    batch_size = 40
    target = int(theme["count"])
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    consecutive_fails = 0
    while len(out) < target and attempts < 40 and consecutive_fails < 5:
        remaining = target - len(out)
        n = min(batch_size, remaining + 8)  # overshoot a bit for dedup
        prompt = (
            f"Generate {n} UNIQUE, REALISTIC news headlines for this theme.\n\n"
            f"TOPIC: {theme['topic']}\n\n"
            f"INSTRUCTIONS: {theme['instructions']}\n\n"
            f"STYLE: Real news-wire tone. No clickbait, no emoji. Each headline"
            f" 60-110 characters. Varied entity names, dates, and phrasings.\n\n"
            f"Here are {len(theme['examples'])} seed examples to match tone"
            f" (but DO NOT duplicate them):\n"
            + "\n".join(f"- {ex}" for ex in theme["examples"])
            + f"\n\nRESPOND WITH JSON: an object with a single field 'headlines'"
            f" containing an array of {n} strings. No other text."
        )
        try:
            resp = await asyncio.wait_for(
                litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.85,
                    max_tokens=3500,
                    response_format={"type": "json_object"},
                    timeout=60,
                ),
                timeout=90,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            candidates = data.get("headlines") or data.get("items") or []
            consecutive_fails = 0
        except TimeoutError:
            print(f"  [batch timeout attempt={attempts}]", flush=True)
            attempts += 1
            consecutive_fails += 1
            continue
        except Exception as exc:
            print(f"  [batch failed attempt={attempts}] {exc}", flush=True)
            attempts += 1
            consecutive_fails += 1
            continue
        attempts += 1
        for h in candidates:
            if not isinstance(h, str):
                continue
            normalized = h.strip().strip('"').strip("'")
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            out.append(normalized)
            if len(out) >= target:
                break
        print(
            f"  [batch {attempts}] +{len(candidates)} candidates, kept {len(out)}/{target}",
            flush=True,
        )
    return out[:target]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Comma-separated scenario ids to run")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    random.seed(42)
    total_written = 0
    for theme in THEMES:
        sid = theme["scenario"]
        if only is not None and sid not in only:
            continue
        theme_name = theme["theme"]
        tier = theme["tier"]
        bank_dir = OUT_DIR / sid / "_v3_banks"
        bank_path = bank_dir / f"{tier}__{theme_name}.json"
        if bank_path.exists():
            existing = json.loads(bank_path.read_text())
            if len(existing.get("rows", [])) >= int(theme["count"]) * 0.9:
                print(
                    f"\n==> {sid} / {tier} / {theme_name} SKIPPED "
                    f"(already have {len(existing.get('rows', []))} rows)",
                    flush=True,
                )
                continue
        print(f"\n==> {sid} / {tier} / {theme_name} (target {theme['count']})", flush=True)
        headlines = await generate_theme(theme)
        rows = []
        urls = theme["source_urls"]
        for i, h in enumerate(headlines):
            rows.append([urls[i % len(urls)], h, theme_name])

        bank_dir = OUT_DIR / sid / "_v3_banks"
        bank_dir.mkdir(parents=True, exist_ok=True)
        bank_path = bank_dir / f"{tier}__{theme_name}.json"
        bank_path.write_text(
            json.dumps(
                {
                    "scenario": sid,
                    "tier": tier,
                    "theme": theme_name,
                    "style_cycle": list(theme["style_cycle"]),
                    "rows": rows,
                },
                indent=2,
            )
        )
        total_written += len(rows)
        print(f"    wrote {len(rows)} rows -> {bank_path}", flush=True)
    print(f"\n[DONE] wrote {total_written} total headlines", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
