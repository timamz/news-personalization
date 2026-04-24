"""Tests for the nightly subscription-source stats task."""

from news_service.tasks.update_subscription_source_stats import (
    _distribution_stats,
    _percentile,
)


def test_percentile_interpolates_between_adjacent_ranks() -> None:
    ordered = [0.0, 0.5, 1.0]
    low = _percentile(ordered, 0.25)
    high = _percentile(ordered, 0.75)

    assert abs(low - 0.25) < 1e-9 and abs(high - 0.75) < 1e-9, (
        "percentile did not linearly interpolate between adjacent ranks"
    )


def test_distribution_stats_returns_p50_p90_and_population_std() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p50, p90, std = _distribution_stats(values)

    mean = sum(values) / len(values)
    expected_std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    assert (
        p50 is not None
        and p90 is not None
        and std is not None
        and abs(p50 - 0.5) < 1e-9
        and abs(p90 - 0.82) < 1e-9
        and abs(std - expected_std) < 1e-9
    ), "distribution stats did not produce correct p50, p90, and population std"


def test_distribution_stats_handles_empty_and_singleton_inputs() -> None:
    p50_empty, p90_empty, std_empty = _distribution_stats([])
    p50_one, p90_one, std_one = _distribution_stats([0.42])

    assert (
        p50_empty is None
        and p90_empty is None
        and std_empty is None
        and p50_one == 0.42
        and p90_one == 0.42
        and std_one == 0.0
    ), "distribution stats did not degrade gracefully for empty or single-value inputs"
