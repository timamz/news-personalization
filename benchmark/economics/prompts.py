"""
Ten subscription-creation prompts used by the economics baseline.

Five digest-mode and five event-mode. Every prompt pre-answers every piece
of information the Conversational Agent might try to clarify (topic,
include/exclude rules, delivery schedule with timezone, response
language, location where relevant) so the agent has zero excuse to emit
a question instead of calling ``create_subscription`` on the first
turn. A CAPS-LOCK trailer spells that out explicitly.
"""

from __future__ import annotations


_NO_QUESTIONS_TRAIL = (
    "\n\nASSUME: my timezone is Europe/Berlin, my language is English, "
    "my city is Berlin. You have every piece of information you need. "
    "DO NOT ASK ANY CLARIFYING QUESTIONS. DO NOT ASK FOR TIMEZONE, "
    "LANGUAGE, LOCATION, DELIVERY PREFERENCES, OR ANYTHING ELSE. "
    "CALL create_subscription ON THIS VERY TURN."
)


PROMPTS: list[dict[str, str]] = [
    {
        "id": "digest_ai_industry",
        "mode": "digest",
        "text": (
            "Create a weekly Monday 09:00 Europe/Berlin digest in English about "
            "the AI and large-language-model industry. Include: product releases "
            "from OpenAI, Anthropic, Google DeepMind, Meta AI, and xAI; funding "
            "rounds above $100M for AI startups; and notable research papers. "
            "Exclude: tutorial content, generic ML explainers, and social-media "
            "drama."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "digest_premier_league",
        "mode": "digest",
        "text": (
            "Create a weekly Sunday 20:00 Europe/London digest in English on "
            "English Premier League football. Include: match results from the "
            "past week, injury updates, and transfer rumours involving top-six "
            "clubs. Exclude: lower-division football, women's football, and "
            "international breaks."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "digest_climate_policy",
        "mode": "digest",
        "text": (
            "Create a daily 08:00 Europe/Brussels digest in English on climate "
            "policy. Include: national policy announcements, UN / COP negotiation "
            "news, major court rulings on climate cases, and carbon-market moves. "
            "Exclude: opinion pieces, pure climate-science papers with no policy "
            "angle, and local weather stories."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "digest_crypto_markets",
        "mode": "digest",
        "text": (
            "Create a weekly Friday 18:00 UTC digest in English on crypto markets. "
            "Include: BTC and ETH price analysis, major protocol upgrades, hacks "
            "or exploits above $10M, SEC and EU MiCA regulatory moves, and "
            "stablecoin news. Exclude: memecoin pump-and-dump coverage and "
            "airdrop hunt content."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "digest_space_exploration",
        "mode": "digest",
        "text": (
            "Create a weekly Saturday 10:00 UTC digest in English on space "
            "exploration. Include: rocket launches with payload details, NASA "
            "and ESA mission updates, SpaceX and Rocket Lab operations, and "
            "notable satellite-constellation news. Exclude: UFO content and "
            "astronomy-hobbyist stories."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "event_apple_launches",
        "mode": "event",
        "text": (
            "Create an EVENT subscription (delivery_mode='event', no schedule, "
            "instant alerts in English) that notifies me every time Apple "
            "officially announces a new product - Mac, iPhone, iPad, Apple Watch, "
            "Vision Pro, AirPods, or a new major iOS / macOS version - or holds "
            "a keynote or WWDC event. Exclude: rumours, supply-chain leaks, and "
            "analyst speculation. Only confirmed Apple announcements."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "event_cyber_breaches",
        "mode": "event",
        "text": (
            "Create an EVENT subscription (delivery_mode='event', no schedule, "
            "instant alerts in English) that notifies me when a major data "
            "breach or ransomware incident is publicly disclosed affecting "
            "either a Fortune 500 company or more than 1 million users. "
            "Exclude: small incidents, vulnerability disclosures without "
            "confirmed exploitation, and generic CVE announcements."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "event_central_bank_rates",
        "mode": "event",
        "text": (
            "Create an EVENT subscription (delivery_mode='event', no schedule, "
            "instant alerts in English) for central-bank interest-rate decisions "
            "from the US Federal Reserve (FOMC), European Central Bank, Bank of "
            "England, and Bank of Japan. Notify me the moment a rate decision "
            "is officially announced with the actual number. Exclude: speeches, "
            "minutes, and economist forecasts."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "event_spacex_launches",
        "mode": "event",
        "text": (
            "Create an EVENT subscription (delivery_mode='event', no schedule, "
            "instant alerts in English) that alerts me for every SpaceX launch. "
            "I want one notification when a launch is scheduled (roughly T-24h) "
            "and one when the launch actually happens with the outcome. Covers "
            "Falcon 9, Falcon Heavy, and Starship. Exclude scrub-and-reschedule "
            "noise - one scheduled + one outcome notification per mission."
        ) + _NO_QUESTIONS_TRAIL,
    },
    {
        "id": "event_tech_ipo_filings",
        "mode": "event",
        "text": (
            "Create an EVENT subscription (delivery_mode='event', no schedule, "
            "instant alerts in English) that notifies me whenever a US-based "
            "technology company publicly files an S-1 for IPO. Include company "
            "name, proposed ticker, and target exchange. Exclude: SPAC mergers, "
            "direct listings without an S-1, and non-US filings."
        ) + _NO_QUESTIONS_TRAIL,
    },
]
