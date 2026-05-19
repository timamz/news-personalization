"""Measured unit-economics constants pinned for steady-state simulation.

Numbers come from real devbox production data, not theoretical
estimates. Sources:

- ``reports_and_documents/unit_economics.md`` section 8.6 (top-6
  DB ingest rate on 2026-04-24) feeds ``MAX_ITEMS_PER_SOURCE_PER_DAY``.

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
