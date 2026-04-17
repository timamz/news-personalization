"""Tests for user_spec parsing, validation, rendering, and observation append."""

import random
import string

from news_service.models.user_spec import (
    MAX_OBSERVATIONS_LENGTH,
    MAX_USER_SPEC_LENGTH,
    UserSpecSections,
    append_observations,
    extract_topic,
    parse_user_spec,
    render_user_spec,
    validate_user_spec,
)


def _rs(length: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def test_parse_extracts_all_three_sections_including_cyrillic() -> None:
    topic = "\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0418\u0418"
    pref = "\u041a\u0440\u0430\u0442\u043a\u043e"
    obs = "Source X went offline on 2026-04-10"
    spec = f"## Topic\n{topic}\n\n## Preferences\n{pref}\n\n## Observations\n{obs}"
    sections = parse_user_spec(spec)
    assert sections.topic == topic and sections.preferences == pref and sections.observations == obs


def test_parse_falls_back_to_raw_text_when_no_headers() -> None:
    raw = f"plain text topic {_rs(6)}"
    assert parse_user_spec(raw).topic == raw


def test_render_and_parse_roundtrip_preserves_content() -> None:
    original = UserSpecSections(
        topic=f"Space exploration {_rs(8)}",
        preferences="include images",
        observations=f"Source A healthy {_rs(8)}",
    )
    parsed = parse_user_spec(render_user_spec(original))
    assert (
        parsed.topic == original.topic
        and parsed.preferences == original.preferences
        and parsed.observations == original.observations
    ), "roundtrip did not preserve user_spec content"


def test_extract_topic_returns_structured_or_raw_content() -> None:
    spec = "## Topic\nDeep learning hardware\n\n## Preferences\nconcise"
    assert extract_topic(spec) == "Deep learning hardware"
    raw = f"autonomous vehicles {_rs(6)}"
    assert extract_topic(raw) == raw


def test_validate_user_spec_caps_length_and_normalises_valid_input() -> None:
    long = "x" * (MAX_USER_SPEC_LENGTH + 500)
    assert len(validate_user_spec(long)) <= MAX_USER_SPEC_LENGTH

    tag = _rs(10)
    spec = f"## Topic\nCybersecurity {tag}\n\n## Preferences\nbullet points"
    result = validate_user_spec(spec)
    assert f"Cybersecurity {tag}" in result and "## Topic" in result


def test_append_observations_adds_caps_and_creates_section_when_absent() -> None:
    tag = _rs(8)
    with_existing = append_observations(
        f"## Topic\nFintech {tag}\n\n## Observations\nOld note", f"New note {tag}"
    )
    parsed_existing = parse_user_spec(with_existing)
    assert "Old note" in parsed_existing.observations
    assert f"New note {tag}" in parsed_existing.observations

    long_existing = "## Topic\nTest\n\n## Observations\n" + "A" * MAX_OBSERVATIONS_LENGTH
    capped = parse_user_spec(append_observations(long_existing, "B" * 500))
    assert len(capped.observations) <= MAX_OBSERVATIONS_LENGTH
    assert capped.observations.endswith("B" * 500)

    created = parse_user_spec(append_observations(f"## Topic\nBiotech {tag}", f"First {tag}"))
    assert f"First {tag}" in created.observations
