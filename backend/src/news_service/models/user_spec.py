"""Structured user_spec parsing, validation, and rendering.

The user_spec is a markdown document with three sections:

- ``## Topic`` -- required. The subject matter the user cares about.
  Consumed by retrieval, discovery, and the Writer. Embedded into
  ``topic_embedding`` for cosine candidate ranking.
- ``## Preferences`` -- optional. Freeform guidance for the Writer
  (format, length, exclusions, tone). Anything LLM-facing that shapes
  how content is presented.
- ``## Observations`` -- optional. Notes appended by the Pipeline
  Reflector about past deliveries. Read by the Writer and Reflector.

Dispatch concerns (schedule, language, sources) live in dedicated
Subscription columns / join tables, not in the markdown.
"""

import re

from pydantic import BaseModel, Field

MAX_USER_SPEC_LENGTH = 10_000
MAX_OBSERVATIONS_LENGTH = 2_000

KNOWN_SECTIONS = ("Topic", "Preferences", "Observations")

_SECTION_HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

_SECTION_NAME_MAP: dict[str, str] = {name.lower(): name for name in KNOWN_SECTIONS}


class UserSpecSections(BaseModel):
    """Structured representation of the user_spec markdown document.

    Three fields, one required, two optional.

    Example::

        sections = UserSpecSections(topic="AI news", preferences="Short bullets")
        assert sections.observations == ""
    """

    topic: str = Field(..., min_length=1, max_length=500)
    preferences: str = Field(default="", max_length=2000)
    observations: str = Field(default="", max_length=MAX_OBSERVATIONS_LENGTH)


def _extract_raw_sections(text: str) -> dict[str, str]:
    """Extract raw section data from markdown text as a dict.

    Splits on ``## Header`` lines and maps content to known section field names.
    If no ``##`` headers are found, treats the entire text as the topic.
    Unknown sections are silently ignored.
    """
    headers = list(_SECTION_HEADER_RE.finditer(text))

    if not headers:
        return {"topic": text.strip()}

    raw_sections: dict[str, str] = {}

    preamble = text[: headers[0].start()].strip()
    if preamble:
        raw_sections["topic"] = preamble

    for idx, match in enumerate(headers):
        section_name = match.group(1).strip().lower()
        content_start = match.end()
        content_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        content = text[content_start:content_end].strip()

        canonical = _SECTION_NAME_MAP.get(section_name)
        if canonical is not None:
            field_name = canonical.lower()
            if field_name not in raw_sections:
                raw_sections[field_name] = content
            else:
                raw_sections[field_name] += "\n\n" + content

    return raw_sections


def parse_user_spec(text: str) -> UserSpecSections:
    """Parse a markdown user_spec into structured sections.

    Splits on ``## Header`` lines, maps content to known sections.
    If no ``##`` headers are found, treats the entire text as the Topic.
    Unknown sections are silently ignored.

    Raises ``ValueError`` via Pydantic if the topic is empty after parsing
    or if any section exceeds its field length limit.
    """
    return UserSpecSections(**_extract_raw_sections(text))


def render_user_spec(sections: UserSpecSections) -> str:
    """Render structured sections back to markdown, omitting empty sections."""
    parts = [f"## Topic\n{sections.topic}"]
    for name in ("Preferences", "Observations"):
        value = getattr(sections, name.lower())
        if value.strip():
            parts.append(f"## {name}\n{value}")
    return "\n\n".join(parts)


def extract_topic(text: str) -> str:
    """Extract the topic from a user_spec string."""
    raw = _extract_raw_sections(text)
    topic = raw.get("topic", "")
    return topic if topic else text.strip()[:500]


def _truncate_to_field_limits(raw: dict[str, str]) -> dict[str, str]:
    """Truncate raw section values to fit within UserSpecSections field limits."""
    field_limits: dict[str, int] = {}
    for field_name, field_info in UserSpecSections.model_fields.items():
        for meta in field_info.metadata:
            if hasattr(meta, "max_length"):
                field_limits[field_name] = meta.max_length
                break
    truncated = dict(raw)
    for field_name, limit in field_limits.items():
        if field_name in truncated and len(truncated[field_name]) > limit:
            truncated[field_name] = truncated[field_name][:limit]
    return truncated


def validate_user_spec(text: str) -> str:
    """Validate and normalise a user_spec string.

    Parses to sections, validates constraints via Pydantic, and re-renders.
    Caps total input at ``MAX_USER_SPEC_LENGTH`` before parsing.
    Truncates individual sections to their field limits.
    Raises ``ValueError`` (from Pydantic) if the topic is empty.
    """
    if len(text) > MAX_USER_SPEC_LENGTH:
        text = text[:MAX_USER_SPEC_LENGTH]
    raw = _extract_raw_sections(text)
    raw = _truncate_to_field_limits(raw)
    sections = UserSpecSections(**raw)
    return render_user_spec(sections)


def append_observations(current_spec: str, new_observations: str) -> str:
    """Structured append of reflector observations.

    Caps the observations section at ``MAX_OBSERVATIONS_LENGTH``,
    keeping the most recent content when truncation is needed.
    """
    sections = parse_user_spec(current_spec)
    if sections.observations:
        combined = sections.observations.rstrip() + "\n\n" + new_observations
    else:
        combined = new_observations
    if len(combined) > MAX_OBSERVATIONS_LENGTH:
        combined = combined[-MAX_OBSERVATIONS_LENGTH:]
    sections.observations = combined
    return render_user_spec(sections)
