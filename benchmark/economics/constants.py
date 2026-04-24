"""Measured unit-economics constants pinned for steady-state simulation.

Numbers come from real devbox production data, not theoretical
estimates. Sources:

- ``reports_and_documents/unit_economics.md`` section 8.6 (top-6
  DB ingest rate on 2026-04-24) feeds ``MAX_ITEMS_PER_SOURCE_PER_DAY``.
- ``reports_and_documents/unit_economics.md`` section 8.1 (attached-
  source counts in the f8fce5be onboarding benchmark, 5 digest + 5
  event prompts) feeds the ``AVG_SOURCES_*`` constants.

Re-measure and update whenever a new benchmark or DB snapshot lands;
the markdown report is the canonical write-up. This module is the
machine-readable copy so simulation scripts can import it without
re-parsing the report.
"""

AVG_ITEMS_PER_SOURCE_PER_DAY: int = 3
"""Population-wide mean daily post volume per source.

Derived from section 8.6 of the unit-economics report: 732 items/day
across 231 active sources on 2026-04-24. 81% of sources returned
zero new items so the arithmetic mean is pulled down by the dormant
tail; this constant captures the *typical* rate a simulation should
assume when the worst-case upper bound (``MAX_ITEMS_PER_SOURCE_PER_DAY``)
would overstate the load.

Use this for steady-state cost projection; use the MAX constant for
capacity-planning worst case.
"""

MAX_ITEMS_PER_SOURCE_PER_DAY: int = 149
"""Upper bound for daily post volume per source.

Mean of the 24h ingest counts of the six busiest sources in the
production DB on 2026-04-24:

- r/OpenAI                                    216
- r/spacex                                    169
- vedomosti.ru/rss/rubric/politics            160
- tass.com/rss/v2.xml                         149
- apple.com/newsroom/rss-feed.rss             108
- animenewsnetwork.com/rss.xml                 92
- mean                                        149

The population-wide mean across 231 active sources is far lower
(~3 items/source/day because 81% of sources are dormant), so this
constant models the realistic worst case a busy subscription can
hit, not the typical source.
"""

AVG_SOURCES_PER_DIGEST_SUB: float = 2.8
"""Average sources attached to a digest subscription after Discovery.

Measured across the five digest prompts of the f8fce5be onboarding
benchmark. Per-prompt values: [5, 1, 4, 0, 4].
"""

AVG_SOURCES_PER_EVENT_SUB: float = 3.2
"""Average sources attached to an event subscription after Discovery.

Measured across the five event prompts of the f8fce5be onboarding
benchmark. Per-prompt values: [0, 7, 1, 3, 5].
"""

AVG_SOURCES_PER_SUB: float = 3.0
"""Average sources attached per subscription across delivery modes.

Measured across all ten prompts of the f8fce5be onboarding benchmark
(30 sources / 10 subs). Use this when the simulation is agnostic to
delivery mode.
"""
