"""Pipeline guardrails — content sanitization and output validation.

Content sanitization: wraps external content in boundary tags and detects
injection patterns before including in LLM prompts.

Output validation: verifies LLM outputs meet structural and semantic
constraints before acting on them.
"""

import logging
import re

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"IMPORTANT\s*:\s*override", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
]

_MAX_NOTIFICATION_BODY_LENGTH = 4000
_MAX_DIGEST_TEXT_LENGTH = 100_000


def wrap_untrusted_content(text: str) -> str:
    """Wrap external content in boundary tags for LLM consumption."""
    return f"<untrusted-content>\n{text}\n</untrusted-content>"


def scan_for_injection(text: str) -> list[str]:
    """Scan text for known prompt injection patterns.

    Returns list of matched pattern descriptions. Empty list means clean.
    """
    matches = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_article_content(headline: str, body: str) -> tuple[str, str, list[str]]:
    """Sanitize article content: wrap in boundaries, detect injections.

    Returns (wrapped_headline, wrapped_body, injection_flags).
    """
    flags = scan_for_injection(headline) + scan_for_injection(body)
    if flags:
        logger.warning(
            "Potential injection detected in article content: %s",
            flags[:3],
        )
    return wrap_untrusted_content(headline), wrap_untrusted_content(body), flags


def validate_used_item_ids(
    claimed_ids: list[str],
    candidate_ids: set[str],
) -> list[str]:
    """Verify all used_item_ids exist in the candidate set.

    Returns only valid IDs, logging warnings for phantoms.
    """
    valid = []
    for item_id in claimed_ids:
        if item_id in candidate_ids:
            valid.append(item_id)
        else:
            logger.warning("Phantom used_item_id from LLM: %s", item_id)
    return valid


def validate_cron(cron_str: str) -> bool:
    """Validate a cron expression using croniter.

    Returns True if valid, False otherwise.
    """
    try:
        from croniter import croniter

        croniter(cron_str)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def validate_notification_body(
    body: str,
    is_relevant: bool,
    original_url: str | None = None,
) -> str | None:
    """Validate event notification body.

    Returns the body if valid, None if invalid (with logging).
    """
    if is_relevant and not body.strip():
        logger.warning("Relevant event has empty notification body")
        return None

    if len(body) > _MAX_NOTIFICATION_BODY_LENGTH:
        logger.warning(
            "Notification body exceeds %d chars, truncating", _MAX_NOTIFICATION_BODY_LENGTH
        )
        body = body[:_MAX_NOTIFICATION_BODY_LENGTH] + "..."

    return body


def validate_digest_text(
    text: str,
    max_length: int = _MAX_DIGEST_TEXT_LENGTH,
) -> str:
    """Validate and cap digest text length."""
    if len(text) > max_length:
        logger.warning("Digest text exceeds %d chars, truncating", max_length)
        return text[:max_length] + "\n\n..."
    return text
