"""
S-judge-digest-scoring: direct scoring behavior of the Digest Quality Judge.

This test calls the real ``judge_digest`` (``backend/src/news_service/agents/
digest/judge.py``) against hand-crafted drafts to verify the grader works
score-axis by score-axis. No pipeline, no writer, no stubs: just the
structured-output LLM call behind the judge, with the bench environment
(DATABASE_URL, LLM config) initialized by the ``world`` fixture even though
no DB tables are touched here.

Coverage, one test per failure mode:

  * clean draft -> PASS with high scores on all three axes;
  * off-topic draft -> low relevance, REVISE, feedback mentions relevance/topic;
  * format-violating draft -> low format score, REVISE, feedback mentions format;
  * bloated draft -> low conciseness, REVISE, feedback mentions filler/redundancy;
  * all three broken simultaneously -> low on all three axes, REVISE.

Out of scope: the digest pipeline wiring, the Writer, the Writer <-> Judge
REVISE loop (covered in ``test_s_digest_revise.py``), and the prose-quality
rubric benchmark.
"""

from __future__ import annotations

import pytest


CLEAN_ON_TOPIC_USER_SPEC = """\
Topic: EU energy policy, focused on gas supply security, renewables
build-out, and electricity market reform. Include only stories that
have direct EU-level policy relevance (Commission, Council, Parliament,
ACER, ENTSO-E).

Presentation: 5 items, each a short numbered paragraph of 2-3 sentences
in plain English. No markdown headers, no bullet lists, no emoji.

Exclusions: celebrity news, sports, consumer product launches.
"""


CLEAN_ON_TOPIC_DIGEST = """\
1. The European Commission proposed a reinforced gas-storage coordination
mechanism for the 2026-2027 winter season, raising the mandatory fill
target for member states to 92% by 1 November. The draft also tightens
reporting deadlines for operators.

2. ACER published updated guidance on capacity-allocation mechanisms at
interconnectors, pushing for earlier auction windows and stricter
congestion-income reporting. National regulators have six months to
align their rules.

3. Germany finalized a revised renewables auction calendar that brings
forward 8 GW of offshore wind tenders originally scheduled for 2027.
The move follows grid-connection concessions negotiated with TenneT
and 50Hertz in March.

4. The Council of the EU reached a general approach on the electricity
market reform package, confirming wider use of two-way contracts for
difference for new low-carbon generation. Final trilogues with the
Parliament are expected before summer recess.

5. ENTSO-E flagged that projected winter peak demand in Central Europe
could exceed available firm capacity by up to 3 GW under a cold-snap
scenario, citing delayed French nuclear returns and lower Norwegian
hydro stocks. It urged member states to accelerate demand-response
tenders.
"""


OFF_TOPIC_DIGEST = """\
1. A reality TV star announced her engagement on Instagram this week,
sparking a flurry of comments from fellow cast members. The ring is
said to weigh in at over five carats.

2. A veteran pop singer teased a surprise album drop at a late-night
talk show appearance, hinting at collaborations with three younger
chart-toppers. Fans immediately began speculating about the track list.

3. A tabloid-favorite actor was photographed leaving a Los Angeles
restaurant with a rumored new partner, fueling fresh dating speculation.
Representatives declined to comment.

4. A streaming dating show renewed for a fourth season announced its
new host, a former contestant turned podcaster. Filming reportedly
begins next month in Mallorca.

5. A viral TikTok feud between two influencers escalated into a pair
of diss tracks released on the same day. Both songs briefly charted
on regional Spotify lists before being overtaken by mainstream hits.
"""


STRICT_FORMAT_USER_SPEC = """\
Topic: EU energy policy -- gas supply security, renewables build-out,
electricity market reform.

Presentation rules (strict): exactly 5 items, each formatted as a
short numbered paragraph of 2-3 sentences. No bullet points. No
markdown headers. No asterisks. No hyphen-prefixed lists. Plain
numbered paragraphs only.
"""


WRONG_FORMAT_DIGEST = """\
# EU Energy Policy Digest

## Top stories this week

- The European Commission proposed a reinforced gas-storage coordination
  mechanism for winter 2026-2027, with a 92% fill target by November.
- ACER published updated guidance on interconnector capacity allocation,
  mandating earlier auctions and stricter congestion-income reports.
- Germany pulled forward 8 GW of offshore wind tenders from 2027 into
  the 2026 auction calendar.
- The Council of the EU reached a general approach on the electricity
  market reform package, endorsing two-way CfDs for new low-carbon
  generation.
- ENTSO-E warned of a possible 3 GW firm-capacity shortfall in Central
  Europe under a cold-snap scenario this winter.

## Regulatory watch

- The Commission opened a consultation on cross-border hydrogen network
  planning, with responses due by 30 June.
- Spain extended its windfall-tax regime on electricity utilities into
  2027, drawing pushback from industry associations.
- France confirmed a second tranche of nuclear-plant lifetime extensions,
  covering six reactors through the 2030s.
"""


BLOATED_DIGEST = """\
1. The European Commission, in a move that was announced earlier this
week and which many observers have been watching closely, proposed a
new and reinforced gas-storage coordination mechanism for the upcoming
winter season of 2026-2027. The mechanism, which is a coordination
mechanism, raises the fill target to 92% by the 1st of November. As a
result of this announcement, it can be said that the Commission has
proposed a mechanism for gas storage that is reinforced, and the target,
which is 92%, applies by the 1st of November, which is in November.
This is, broadly speaking, a development that is generally considered
to be a development in the area of gas storage coordination.

2. ACER, the EU energy regulator, published some new guidance, which
is guidance, on the topic of capacity allocation at interconnectors.
The guidance, which is new, pushes for earlier auction windows, meaning
that the auction windows should be earlier than they currently are. In
addition, and on top of that, it also calls for stricter reporting of
congestion income, which essentially means reporting that is stricter
than before. National regulators, which are the regulators at the
national level, have six months to align, more or less, with the new
guidance that has been published by ACER.

3. Germany, which is a member state of the European Union, finalized a
revised auction calendar for renewables, specifically renewables in the
area of offshore wind. The calendar, which has been revised, brings
forward a total of 8 GW of offshore wind tenders. These tenders, which
are offshore wind tenders, were originally scheduled for the year 2027
but are now being brought forward. It should also be noted and mentioned
that the move follows grid-connection concessions that were negotiated
with the two grid operators, TenneT and 50Hertz, in the month of March.

4. The Council of the European Union, which represents the member
states, reached what is called a general approach on the electricity
market reform package. The general approach, which is a general
approach, confirms the wider use of two-way contracts for difference,
which are sometimes abbreviated as CfDs, for new generation that is
low-carbon in nature. As things stand right now, at this moment in
time, the final trilogue negotiations with the European Parliament are
broadly expected, in general, to take place before the summer recess,
which is the recess that happens in the summer.

5. ENTSO-E, the European network of transmission system operators for
electricity, flagged in a recent publication that the projected peak
demand for the winter season in Central Europe could, potentially, in
certain scenarios, exceed the available firm capacity by an amount of
up to approximately 3 GW under a cold-snap scenario. This, needless to
say, is a scenario that is a cold-snap scenario. ENTSO-E, which is the
network, cited reasons, and the reasons include delayed French nuclear
returns and lower Norwegian hydro stocks, and it urged, as a general
matter, the member states to accelerate, in the sense of speeding up,
demand-response tenders.
"""


ALL_BROKEN_DIGEST = """\
# Celebrity News Roundup

## This week's hottest stories

- A reality TV star, who is well known for being a reality TV star,
  announced on the social media platform Instagram, which is a social
  media platform, that she is, in fact, now engaged to be married to
  her long-time partner, who is her partner of a long time. The ring,
  which is the engagement ring, weighs in, it is reported, at over
  five carats, which is a measurement of the weight of the ring.
- A veteran pop singer, meaning a singer who has been in the pop genre
  for many years, teased, in a teasing manner, a surprise album drop
  during a late-night talk show appearance on a late-night talk show.
  The singer, who is the singer, hinted at collaborations, which are
  partnerships on songs, with three younger chart-topping artists,
  who are artists that top the charts and are younger in age.
- A tabloid-favorite actor, an actor favored by tabloids, was
  photographed by photographers as he was leaving a restaurant in
  the city of Los Angeles, which is in California. He was accompanied
  by a rumored new partner, meaning a partner who is rumored to be new,
  which fueled speculation, in the speculative sense, about his dating
  life, which is the life of dating.
- A streaming dating show, which is a show about dating that streams,
  was renewed for its fourth season, which is the season that comes
  after the third. The announcement, which was an announcement, also
  named a new host, and the new host is a former contestant on the
  show who has since become a podcaster, which is a person who hosts
  a podcast.
- A viral feud on the platform TikTok, which is the platform known as
  TikTok, between two influencers, who are people who influence others,
  escalated, in an escalating fashion, into a pair of diss tracks,
  which are songs that insult, that were released on the same day.
  Both songs, being both of the songs, briefly, for a short time,
  charted on regional Spotify lists, which are lists on Spotify that
  are regional, before being overtaken by other hits, which are songs
  that are mainstream in character.

## Also trending

- Another TikTok personality, who is a personality on TikTok, posted a
  video that received, in a reception sense, a large number of views,
  which are views on the video, and this, broadly speaking, was
  generally considered to be a notable event in the view-counting
  world.
- A podcast host, meaning the host of a podcast, announced a new
  podcast about, as a matter of fact, podcasting, which is the act
  of producing and releasing podcasts, and this, needless to say,
  struck some observers as a somewhat recursive development.
"""


@pytest.mark.asyncio
async def test_s_judge_digest_clean_draft_passes(world):
    """Competent on-topic digest with correct format and no padding scores high on all axes."""
    from news_service.agents.digest.judge import judge_digest

    scores = await judge_digest(
        digest_text=CLEAN_ON_TOPIC_DIGEST,
        user_spec=CLEAN_ON_TOPIC_USER_SPEC,
        candidates_summary="6 EU energy policy items available for inclusion",
    )

    assert scores.relevance >= 4, (
        f"clean on-topic draft should score relevance >= 4, got {scores.relevance}. "
        f"Full scores: {scores!r}"
    )
    assert scores.format_score >= 4, (
        f"clean numbered-paragraph draft should score format_score >= 4, got "
        f"{scores.format_score}. Full scores: {scores!r}"
    )
    assert scores.conciseness >= 4, (
        f"clean non-bloated draft should score conciseness >= 4, got "
        f"{scores.conciseness}. Full scores: {scores!r}"
    )
    assert scores.verdict == "PASS", (
        f"clean draft should yield verdict PASS, got {scores.verdict!r}. "
        f"Full scores: {scores!r}"
    )


@pytest.mark.asyncio
async def test_s_judge_digest_off_topic_fails_relevance(world):
    """Celebrity-gossip draft against an EU-energy user_spec tanks the relevance axis."""
    from news_service.agents.digest.judge import judge_digest

    scores = await judge_digest(
        digest_text=OFF_TOPIC_DIGEST,
        user_spec=CLEAN_ON_TOPIC_USER_SPEC,
        candidates_summary="6 EU energy policy items available for inclusion",
    )

    assert scores.relevance <= 2, (
        f"off-topic celebrity draft should score relevance <= 2 against an "
        f"EU-energy user_spec, got {scores.relevance}. Full scores: {scores!r}"
    )
    assert scores.verdict == "REVISE", (
        f"off-topic draft should yield verdict REVISE, got {scores.verdict!r}. "
        f"Full scores: {scores!r}"
    )
    feedback_lower = scores.feedback.lower()
    assert any(token in feedback_lower for token in ("relevance", "topic", "user")), (
        f"off-topic feedback should mention relevance/topic/user interests, got "
        f"feedback {scores.feedback!r}. Full scores: {scores!r}"
    )


@pytest.mark.asyncio
async def test_s_judge_digest_wrong_format_fails_format_score(world):
    """On-topic content but bullets + markdown headers violate the strict numbered-paragraph spec."""
    from news_service.agents.digest.judge import judge_digest

    scores = await judge_digest(
        digest_text=WRONG_FORMAT_DIGEST,
        user_spec=STRICT_FORMAT_USER_SPEC,
        candidates_summary="8 EU energy policy items available for inclusion",
    )

    assert scores.format_score <= 2, (
        f"bulleted markdown-header draft should score format_score <= 2 against a "
        f"strict numbered-paragraph spec, got {scores.format_score}. "
        f"Full scores: {scores!r}"
    )
    assert scores.verdict == "REVISE", (
        f"format-violating draft should yield verdict REVISE, got "
        f"{scores.verdict!r}. Full scores: {scores!r}"
    )
    feedback_lower = scores.feedback.lower()
    assert any(
        token in feedback_lower for token in ("format", "bullet", "paragraph", "list")
    ), (
        f"format-failure feedback should mention format/bullet/paragraph/list, got "
        f"feedback {scores.feedback!r}. Full scores: {scores!r}"
    )


@pytest.mark.asyncio
async def test_s_judge_digest_bloated_fails_conciseness(world):
    """Grossly padded paragraphs with repetition and filler tank the conciseness axis."""
    from news_service.agents.digest.judge import judge_digest

    scores = await judge_digest(
        digest_text=BLOATED_DIGEST,
        user_spec=CLEAN_ON_TOPIC_USER_SPEC,
        candidates_summary="6 EU energy policy items available for inclusion",
    )

    assert scores.conciseness <= 2, (
        f"grossly bloated draft should score conciseness <= 2, got "
        f"{scores.conciseness}. Full scores: {scores!r}"
    )
    assert scores.verdict == "REVISE", (
        f"bloated draft should yield verdict REVISE, got {scores.verdict!r}. "
        f"Full scores: {scores!r}"
    )
    feedback_lower = scores.feedback.lower()
    assert any(
        token in feedback_lower
        for token in ("concise", "filler", "redundant", "repetitive", "padding")
    ), (
        f"conciseness-failure feedback should mention concise/filler/redundant/"
        f"repetitive/padding, got feedback {scores.feedback!r}. Full scores: {scores!r}"
    )


@pytest.mark.asyncio
async def test_s_judge_digest_all_three_broken_fails_all(world):
    """Off-topic + wrong format + bloated draft scores low on every axis."""
    from news_service.agents.digest.judge import judge_digest

    scores = await judge_digest(
        digest_text=ALL_BROKEN_DIGEST,
        user_spec=STRICT_FORMAT_USER_SPEC,
        candidates_summary="6 EU energy policy items available for inclusion",
    )

    assert scores.relevance <= 3, (
        f"off-topic-plus-broken draft should score relevance <= 3 against an "
        f"EU-energy spec, got {scores.relevance}. Full scores: {scores!r}"
    )
    assert scores.format_score <= 3, (
        f"bulleted markdown-header draft should score format_score <= 3 against a "
        f"strict numbered-paragraph spec, got {scores.format_score}. "
        f"Full scores: {scores!r}"
    )
    assert scores.conciseness <= 3, (
        f"padded-and-repetitive draft should score conciseness <= 3, got "
        f"{scores.conciseness}. Full scores: {scores!r}"
    )
    assert scores.verdict == "REVISE", (
        f"triple-broken draft should yield verdict REVISE, got {scores.verdict!r}. "
        f"Full scores: {scores!r}"
    )
