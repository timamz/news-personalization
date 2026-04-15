"""Tests for the user_spec parsing, validation, and rendering module."""

import random
import string

import pytest
from pydantic import ValidationError

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


def _random_string(length: int) -> str:
    """Generate a random ASCII string of the given length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class TestParseUserSpec:
    def test_extracts_topic_from_standard_format(self) -> None:
        spec = "## Topic\nArtificial intelligence breakthroughs\n\n## Sources\nsome feeds"
        sections = parse_user_spec(spec)
        assert sections.topic == "Artificial intelligence breakthroughs", (
            "topic was not extracted from standard ## Topic header"
        )

    def test_treats_text_as_topic_when_no_headers(self) -> None:
        raw = "just a plain text topic about quantum computing"
        sections = parse_user_spec(raw)
        assert sections.topic == raw, (
            "entire text was not used as topic when no headers are present"
        )

    def test_handles_all_six_sections(self) -> None:
        spec = (
            "## Topic\nBlockchain regulation\n\n"
            "## Sources\nCoinDesk, Reuters\n\n"
            "## Schedule\nevery morning at 8\n\n"
            "## Preferences\nshort bullets only\n\n"
            "## Feedback\ntoo verbose last time\n\n"
            "## Observations\nSource X went offline on 2026-04-10"
        )
        sections = parse_user_spec(spec)
        assert sections.topic == "Blockchain regulation", "topic section was not parsed correctly"
        assert sections.sources == "CoinDesk, Reuters", "sources section was not parsed correctly"
        assert sections.schedule == "every morning at 8", (
            "schedule section was not parsed correctly"
        )
        assert sections.preferences == "short bullets only", (
            "preferences section was not parsed correctly"
        )
        assert sections.feedback == "too verbose last time", (
            "feedback section was not parsed correctly"
        )
        assert sections.observations == "Source X went offline on 2026-04-10", (
            "observations section was not parsed correctly"
        )

    def test_handles_non_ascii_cyrillic_content(self) -> None:
        cyrillic_topic = (
            "\u041d\u043e\u0432\u043e\u0441\u0442\u0438 "
            "\u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0433\u043e "
            "\u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442\u0430"
        )
        cyrillic_pref = (
            "\u041a\u0440\u0430\u0442\u043a\u043e\u0435 "
            "\u0438\u0437\u043b\u043e\u0436\u0435\u043d\u0438\u0435"
        )
        spec = f"## Topic\n{cyrillic_topic}\n\n## Preferences\n{cyrillic_pref}"
        sections = parse_user_spec(spec)
        assert sections.topic == cyrillic_topic, "cyrillic topic was not parsed correctly"
        assert sections.preferences == cyrillic_pref, (
            "cyrillic preferences were not parsed correctly"
        )

    def test_ignores_unknown_sections(self) -> None:
        spec = "## Topic\nML papers\n\n## UnknownHeader\nshould be ignored\n\n## Sources\nArXiv"
        sections = parse_user_spec(spec)
        assert sections.topic == "ML papers", (
            "topic was not parsed when unknown sections are present"
        )
        assert sections.sources == "ArXiv", "sources section was lost due to unknown section"

    def test_case_insensitive_header_matching(self) -> None:
        spec = "## topic\nLower-case header topic\n\n## PREFERENCES\nall caps preference"
        sections = parse_user_spec(spec)
        assert sections.topic == "Lower-case header topic", (
            "case-insensitive topic header was not matched"
        )
        assert sections.preferences == "all caps preference", (
            "case-insensitive PREFERENCES header was not matched"
        )


class TestRenderUserSpec:
    def test_omits_empty_sections(self) -> None:
        sections = UserSpecSections(
            topic="Climate change policy",
            sources="",
            schedule="",
            preferences="detailed analysis",
            feedback="",
            observations="",
        )
        rendered = render_user_spec(sections)
        assert "## Topic" in rendered, "rendered spec did not contain Topic header"
        assert "## Preferences" in rendered, "rendered spec did not contain Preferences header"
        assert "## Sources" not in rendered, "rendered spec contained empty Sources section"
        assert "## Schedule" not in rendered, "rendered spec contained empty Schedule section"
        assert "## Feedback" not in rendered, "rendered spec contained empty Feedback section"
        assert "## Observations" not in rendered, (
            "rendered spec contained empty Observations section"
        )

    def test_includes_all_non_empty_sections(self) -> None:
        tag = _random_string(12)
        sections = UserSpecSections(
            topic=f"Robotics {tag}",
            sources=f"IEEE {tag}",
            schedule="weekly",
            preferences="concise",
            feedback="good",
            observations="all green",
        )
        rendered = render_user_spec(sections)
        for header in ("Topic", "Sources", "Schedule", "Preferences", "Feedback", "Observations"):
            assert f"## {header}" in rendered, f"rendered spec did not contain {header} header"


class TestRenderAndParseRoundtrip:
    def test_roundtrip_preserves_content(self) -> None:
        tag = _random_string(16)
        original = UserSpecSections(
            topic=f"Space exploration {tag}",
            sources=f"NASA, ESA {tag}",
            schedule="daily at 7am",
            preferences="include images",
            feedback="too many duplicates",
            observations=f"Source A healthy {tag}",
        )
        rendered = render_user_spec(original)
        parsed = parse_user_spec(rendered)
        assert parsed.topic == original.topic, "roundtrip did not preserve topic"
        assert parsed.sources == original.sources, "roundtrip did not preserve sources"
        assert parsed.schedule == original.schedule, "roundtrip did not preserve schedule"
        assert parsed.preferences == original.preferences, "roundtrip did not preserve preferences"
        assert parsed.feedback == original.feedback, "roundtrip did not preserve feedback"
        assert parsed.observations == original.observations, (
            "roundtrip did not preserve observations"
        )


class TestExtractTopic:
    def test_returns_topic_content(self) -> None:
        spec = "## Topic\nDeep learning hardware\n\n## Sources\nAnandTech"
        result = extract_topic(spec)
        assert result == "Deep learning hardware", (
            "extract_topic did not return topic from structured spec"
        )

    def test_falls_back_to_raw_text(self) -> None:
        raw = "autonomous vehicles safety standards"
        result = extract_topic(raw)
        assert result == raw, "extract_topic did not fall back to raw text when no headers present"


class TestValidateUserSpec:
    def test_caps_length(self) -> None:
        long_topic = "x" * (MAX_USER_SPEC_LENGTH + 500)
        result = validate_user_spec(long_topic)
        assert len(result) <= MAX_USER_SPEC_LENGTH, (
            "validate_user_spec did not cap the total length"
        )

    def test_raises_for_empty_topic(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            validate_user_spec("## Topic\n\n## Sources\nSome source")

    def test_normalises_valid_spec(self) -> None:
        tag = _random_string(10)
        spec = f"## Topic\nCybersecurity {tag}\n\n## Preferences\nbullet points"
        result = validate_user_spec(spec)
        assert f"Cybersecurity {tag}" in result, "validate_user_spec did not preserve topic content"
        assert "## Topic" in result, "validate_user_spec did not produce Topic header"


class TestAppendObservations:
    def test_adds_to_existing_observations(self) -> None:
        tag = _random_string(8)
        spec = f"## Topic\nFintech {tag}\n\n## Observations\nOld note"
        result = append_observations(spec, f"New note {tag}")
        sections = parse_user_spec(result)
        assert "Old note" in sections.observations, "existing observations were lost after append"
        assert f"New note {tag}" in sections.observations, "new observations were not appended"

    def test_caps_at_max_length(self) -> None:
        spec = "## Topic\nTest capping\n\n## Observations\n" + "A" * MAX_OBSERVATIONS_LENGTH
        new_obs = "B" * 500
        result = append_observations(spec, new_obs)
        sections = parse_user_spec(result)
        assert len(sections.observations) <= MAX_OBSERVATIONS_LENGTH, (
            "observations section exceeded MAX_OBSERVATIONS_LENGTH after append"
        )
        assert sections.observations.endswith("B" * 500), (
            "most recent observations were not kept at the end"
        )

    def test_creates_section_when_absent(self) -> None:
        tag = _random_string(8)
        spec = f"## Topic\nBiotech {tag}"
        result = append_observations(spec, f"First observation {tag}")
        sections = parse_user_spec(result)
        assert f"First observation {tag}" in sections.observations, (
            "observations section was not created when absent"
        )

    def test_preserves_other_sections(self) -> None:
        tag = _random_string(8)
        spec = (
            f"## Topic\nQuantum computing {tag}\n\n"
            f"## Sources\nNature {tag}\n\n"
            f"## Preferences\nDetailed {tag}"
        )
        result = append_observations(spec, "new reflector note")
        sections = parse_user_spec(result)
        assert sections.topic == f"Quantum computing {tag}", (
            "topic was altered by append_observations"
        )
        assert sections.sources == f"Nature {tag}", "sources were altered by append_observations"
        assert sections.preferences == f"Detailed {tag}", (
            "preferences were altered by append_observations"
        )
        assert "new reflector note" in sections.observations, "observations were not appended"
