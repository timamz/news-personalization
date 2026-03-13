from news_service.services.timezones import normalize_timezone_name, resolve_timezone


def test_resolve_timezone_exact_city():
    resolution = resolve_timezone("Berlin")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "Europe/Berlin"
    assert resolution.candidates[0].label == "Berlin, Germany"


def test_resolve_timezone_accepts_common_misspelling():
    resolution = resolve_timezone("moskow")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "Europe/Moscow"


def test_resolve_timezone_accepts_russian_city_name():
    resolution = resolve_timezone("Берлин")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "Europe/Berlin"


def test_resolve_timezone_accepts_russian_city_alias():
    resolution = resolve_timezone("Питер")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "Europe/Moscow"


def test_resolve_timezone_returns_ambiguous_candidates():
    resolution = resolve_timezone("Springfield")

    assert resolution.status == "ambiguous"
    assert len(resolution.candidates) == 3


def test_resolve_timezone_accepts_iana_timezone_name():
    resolution = resolve_timezone("america/new_york")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "America/New_York"


def test_normalize_timezone_name_rejects_invalid_value():
    try:
        normalize_timezone_name("Mars/Olympus")
    except ValueError as exc:
        assert str(exc) == "Unknown timezone: Mars/Olympus"
    else:
        raise AssertionError("Expected ValueError for invalid timezone")
