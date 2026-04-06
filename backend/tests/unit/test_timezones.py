import logging

import pytest

from news_service.services.timezones import normalize_timezone_name, resolve_timezone

logging.disable(logging.CRITICAL)


def test_resolve_timezone_exact_city_returns_resolved_with_correct_timezone_and_label() -> None:
    resolution = resolve_timezone("Berlin")

    assert resolution.status == "resolved", (
        "resolve_timezone did not return resolved status for exact city name"
    )
    assert resolution.candidates[0].timezone == "Europe/Berlin", (
        "resolve_timezone did not return Europe/Berlin for Berlin"
    )
    assert resolution.candidates[0].label == "Berlin, Germany", (
        "resolve_timezone did not return correct label for Berlin"
    )


def test_resolve_timezone_accepts_common_misspelling() -> None:
    resolution = resolve_timezone("moskow")
    assert resolution.candidates[0].timezone == "Europe/Moscow", (
        "resolve_timezone did not handle misspelling moskow"
    )


def test_resolve_timezone_accepts_russian_city_name() -> None:
    resolution = resolve_timezone("Берлин")
    assert resolution.candidates[0].timezone == "Europe/Berlin", (
        "resolve_timezone did not handle Russian city name Берлин"
    )


def test_resolve_timezone_accepts_russian_city_alias() -> None:
    resolution = resolve_timezone("Питер")
    assert resolution.candidates[0].timezone == "Europe/Moscow", (
        "resolve_timezone did not handle Russian alias Питер"
    )


def test_resolve_timezone_returns_ambiguous_status_for_springfield() -> None:
    resolution = resolve_timezone("Springfield")
    assert resolution.status == "ambiguous", (
        "resolve_timezone did not return ambiguous status for Springfield"
    )


def test_resolve_timezone_returns_multiple_candidates_for_springfield() -> None:
    resolution = resolve_timezone("Springfield")
    assert len(resolution.candidates) == 3, (
        "resolve_timezone did not return 3 candidates for Springfield"
    )


def test_resolve_timezone_accepts_iana_timezone_name() -> None:
    resolution = resolve_timezone("america/new_york")
    assert resolution.candidates[0].timezone == "America/New_York", (
        "resolve_timezone did not normalize IANA timezone name"
    )


def test_normalize_timezone_name_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        normalize_timezone_name("Mars/Olympus")
