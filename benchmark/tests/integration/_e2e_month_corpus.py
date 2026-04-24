"""
Tiny synthetic corpus used by the one-month e2e cost benchmark.

Kept outside any ``test_*.py`` module so pytest does not collect it.
The content is deliberately thin -- enough for the pipeline to have
something topical to ingest, rank, and deliver on, not a
quality-measuring corpus. Quality evaluation belongs to the v3
benchmark fabric that lived in commit 535dab2; this file only has to
produce enough typed items to exercise the cost path.

Two topic banks match the two subs driven by the e2e test:

- ``DIGEST_BANK`` (EU energy and climate policy), parallel to the
  existing ``_digest_common.SEED_ITEMS`` but larger so a month of
  AVG_ITEMS_PER_SOURCE_PER_DAY emissions has non-repetitive content.
- ``EVENT_BANK`` (lithium supply chain), parallel to
  ``test_s_event_assess.py``'s inline items.

``build_timeline`` rotates one item per emission slot and spreads
emissions evenly across each simulated day per source.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from news_benchmark.fakes.adapters import ScenarioItem

DIGEST_USER_SPEC = (
    "# EU energy and climate policy daily digest\n"
    "\n"
    "Give me a daily round-up of EU energy and climate policy: Council "
    "decisions, Commission proposals, ENTSO-E / ACER publications, "
    "EUR-Lex directives, Parliament committee votes (ENVI, ITRE). "
    "Focus on policy, not market prices.\n"
    "\n"
    "Skip: EV sales news, sports, celebrity coverage, generic tech."
)

DIGEST_RETRIEVAL_QUERY = (
    "EU energy and climate policy Council Commission ENTSO-E ACER "
    "EUR-Lex ENVI ITRE methane renewable wind LNG gas storage"
)


EVENT_USER_SPEC = (
    "# Lithium supply-chain alerts\n"
    "\n"
    "Notify me the instant news breaks on lithium mining, lithium "
    "refining, battery-grade lithium carbonate or hydroxide pricing, "
    "or regulatory actions affecting the lithium supply chain.\n"
    "\n"
    "Do NOT notify on: EV sales, Tesla stock, downstream battery-cell "
    "or pack news, other metals (copper, nickel, cobalt) unless "
    "lithium-linked."
)

EVENT_RETRIEVAL_QUERY = (
    "lithium supply chain carbonate hydroxide mining refinery "
    "battery-grade pricing royalty export permit"
)


DIGEST_BANK: list[tuple[str, str]] = [
    (
        "Council of the EU adopts gas storage directive for 2027 winter",
        "The Council of the European Union formally adopted a directive "
        "requiring Member States to fill underground gas storage to 90% "
        "of capacity by 1 November each year starting in 2027. The text "
        "makes the storage obligation permanent and replaces the 2022 "
        "temporary regulation due to expire at end of 2026.",
    ),
    (
        "European Commission proposes 40% renewable electricity target by 2030",
        "The European Commission unveiled a proposal to set a binding 40% "
        "renewable electricity target for the EU-27 by 2030, up from 32%. "
        "Commissioner Simson said the higher ceiling reflects falling "
        "costs for solar and wind generation. The proposal goes to "
        "Council and Parliament for the ordinary legislative procedure.",
    ),
    (
        "ACER publishes final network code on electricity balancing",
        "ACER published the final text of the amended network code on "
        "electricity balancing. The code harmonises cross-border "
        "balancing platforms and tightens imbalance-settlement timelines. "
        "National regulators have six months to transpose the changes.",
    ),
    (
        "ENTSO-E winter outlook flags Central European capacity risk",
        "ENTSO-E's winter outlook warned Central European grids may face "
        "capacity shortfalls during cold snaps in December and January. "
        "Germany, Austria and the Czech Republic are most exposed. "
        "National operators are urged to finalise contingency reserves.",
    ),
    (
        "EUR-Lex publishes directive on accelerated offshore wind permitting",
        "A directive accelerating permitting for offshore wind "
        "installations was published on EUR-Lex. The text caps permitting "
        "timelines in pre-designated renewable acceleration areas at 24 "
        "months. The Commission estimates up to 32 GW of additional "
        "offshore capacity by 2030.",
    ),
    (
        "ENVI committee tightens methane limits for imported LNG",
        "The European Parliament ENVI committee voted to tighten "
        "methane-intensity limits for imported LNG. The amendment "
        "extends the Methane Regulation's import standard to cover the "
        "full upstream value chain. Trilogue with the Council expected "
        "in June.",
    ),
    (
        "ITRE committee backs binding 2040 EU emissions reduction target",
        "The ITRE committee of the European Parliament endorsed a 90% "
        "binding emissions reduction target for 2040 against 1990 "
        "levels. The vote clears the way for plenary debate and "
        "subsequent trilogue with the Council on the long-term climate "
        "architecture.",
    ),
    (
        "Commission approves EUR 2.3 billion hydrogen production IPCEI",
        "The European Commission approved, under state-aid rules, a EUR "
        "2.3 billion Important Project of Common European Interest "
        "covering 15 renewable-hydrogen production sites across six "
        "Member States. Projects must commission by 2029 to retain "
        "funding eligibility.",
    ),
    (
        "Council greenlights revision to the Energy Efficiency Directive",
        "Council of the EU endorsed the revised Energy Efficiency "
        "Directive, raising the 2030 energy-savings target to 11.7% "
        "against the 2020 reference scenario. Public-sector buildings "
        "face a 1.9% annual renovation obligation starting in 2027.",
    ),
    (
        "EUR-Lex posts regulation on grid-scale battery storage rules",
        "A regulation clarifying grid-scale battery storage's treatment "
        "as a generation-and-consumption hybrid asset was published on "
        "EUR-Lex. Transmission tariffs may no longer double-charge "
        "storage on both injection and withdrawal.",
    ),
    (
        "ACER report finds cross-border transmission still under-built",
        "ACER's 2026 market-monitoring report concludes cross-border "
        "transmission capacity in the EU remains 30% below the level "
        "implied by the TEN-E planning targets. The agency recommends "
        "an expedited Article 14 identification process for the next "
        "projects of common interest.",
    ),
    (
        "Commission opens probe into Member State capacity-mechanism reforms",
        "The European Commission opened formal proceedings into three "
        "Member States' proposed national capacity mechanisms, citing "
        "concerns that the designs favour domestic gas generation over "
        "cross-border participation and demand response.",
    ),
]


EVENT_BANK: list[tuple[str, str]] = [
    (
        "Chile ministry unveils lithium pricing floor for 2026 contracts",
        "Chile's economy ministry published draft regulations setting a "
        "minimum floor price for battery-grade lithium carbonate sold "
        "under long-term contracts from January 2026. Industry "
        "consultations close in May. SQM and Albemarle are reviewing "
        "the text.",
    ),
    (
        "Albemarle pauses expansion of Kemerton lithium hydroxide refinery",
        "Albemarle said it will pause the third-train expansion of its "
        "Kemerton lithium hydroxide refinery in Western Australia, "
        "citing weaker-than-expected battery-grade demand through 2026. "
        "Trains 1 and 2 remain on stream.",
    ),
    (
        "Ganfeng announces 40 kt lithium carbonate capacity in Argentina",
        "Ganfeng Lithium completed commissioning of a 40,000 tpa lithium "
        "carbonate module at its Cauchari-Olaroz joint venture in "
        "Argentina. The addition brings total Argentine capacity on the "
        "project to 65 kt LCE per year.",
    ),
    (
        "Pilbara Minerals signs long-term spodumene supply deal with POSCO",
        "Pilbara Minerals signed a five-year spodumene concentrate "
        "supply deal with POSCO Holdings' South Korean refining arm. "
        "Pricing references a formula tied to battery-grade carbonate "
        "benchmarks rather than flat-rate contracts.",
    ),
    (
        "Zimbabwe confirms ban on raw lithium ore exports",
        "Zimbabwe's mines ministry confirmed the ban on exports of raw "
        "lithium ore takes effect 1 July, forcing producers to domestic "
        "processing. The measure will increase local value capture but "
        "may delay short-term ore shipments.",
    ),
    (
        "Tianqi reports first-quarter battery-grade carbonate margin pressure",
        "Tianqi Lithium said battery-grade lithium carbonate unit cash "
        "margins compressed 18% quarter-on-quarter as spot prices "
        "recovered more slowly than realisation from long-term "
        "contracts. Full-year guidance retained.",
    ),
    (
        "Bolivian state lithium producer signs direct-extraction partnership",
        "YLB, Bolivia's state lithium producer, signed a memorandum "
        "with a Russian technology consortium for direct-lithium-"
        "extraction pilots at Salar de Uyuni. Commercial-scale volumes "
        "would follow a successful 2027 field demonstration.",
    ),
    (
        "Liontown Resources achieves first production at Kathleen Valley",
        "Liontown Resources announced first spodumene production at its "
        "Kathleen Valley mine in Western Australia. Ramp to nameplate "
        "of 500 kt spodumene concentrate per year is targeted over the "
        "next four quarters.",
    ),
    (
        "South Korea cuts battery-grade lithium import tariff to zero",
        "South Korea's finance ministry cut its battery-grade lithium "
        "carbonate and hydroxide import tariff to zero for the 2026 "
        "calendar year. The measure is aimed at supporting Korean "
        "cathode-material manufacturers facing imported-chemistry "
        "competition.",
    ),
    (
        "Chilean regulator approves SQM Salar de Atacama royalty overhaul",
        "Chile's environmental regulator signed off on the revised "
        "royalty schedule for SQM's Salar de Atacama operations, "
        "cementing the hybrid public-private governance shape agreed "
        "with Codelco in 2024.",
    ),
]


def build_timeline(
    *,
    source_urls: list[str],
    topic: str,
    start: datetime,
    days: int,
    items_per_source_per_day: int,
) -> list[ScenarioItem]:
    """Spread ``items_per_source_per_day`` items across ``days`` per source.

    Items are taken from the bank matching ``topic`` (``"digest"`` or
    ``"event"``), rotated so every item appears in each cohort before
    repeats start. Emission timestamps are evenly spaced inside each
    day so the pipeline observes a steady stream rather than bursts.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    if topic == "digest":
        bank = DIGEST_BANK
    elif topic == "event":
        bank = EVENT_BANK
    else:
        raise ValueError(f"unknown topic: {topic!r}")

    items: list[ScenarioItem] = []
    counter = 0
    for day in range(days):
        for slot in range(items_per_source_per_day):
            hour_offset = int((slot + 0.5) * (24 / items_per_source_per_day))
            ts = start + timedelta(days=day, hours=hour_offset)
            for source_url in source_urls:
                headline, body = bank[counter % len(bank)]
                items.append(
                    ScenarioItem(
                        fake_ts=ts.astimezone(UTC),
                        source_url=source_url,
                        headline=headline,
                        body=body,
                    )
                )
                counter += 1
    return items
