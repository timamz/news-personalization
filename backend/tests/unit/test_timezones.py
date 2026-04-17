import logging

import pytest

from news_service.services.timezones import normalize_timezone_name, resolve_timezone

logging.disable(logging.CRITICAL)


@pytest.mark.parametrize(
    ("query", "expected_tz"),
    [
        ("Berlin", "Europe/Berlin"),
        ("moskow", "Europe/Moscow"),
        ("\u0411\u0435\u0440\u043b\u0438\u043d", "Europe/Berlin"),
        ("\u041f\u0438\u0442\u0435\u0440", "Europe/Moscow"),
        ("america/new_york", "America/New_York"),
    ],
    ids=["exact_en", "misspelling", "cyrillic_city", "cyrillic_alias", "iana_form"],
)
def test_resolve_timezone_resolves_city_names_aliases_and_iana_forms(
    query: str, expected_tz: str
) -> None:
    resolution = resolve_timezone(query)
    assert resolution.candidates[0].timezone == expected_tz, (
        f"resolve_timezone({query!r}) returned {resolution.candidates[0].timezone!r}"
    )


def test_resolve_timezone_returns_multiple_candidates_for_ambiguous_city() -> None:
    resolution = resolve_timezone("Springfield")
    assert resolution.status == "ambiguous" and len(resolution.candidates) >= 2


def test_normalize_timezone_name_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        normalize_timezone_name("Mars/Olympus")
