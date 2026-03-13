import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from geonamescache import GeonamesCache

_ALIAS_MATCH_LIMIT = 8
_AMBIGUOUS_MATCH_LIMIT = 3
_FUZZY_CUTOFF = 0.78
_EXACT_POPULATION_MULTIPLIER = 5
_FUZZY_SCORE_MARGIN = 0.08
_COMMON_CITY_ALIASES = {
    "la": "los angeles",
    "nyc": "new york city",
    "spb": "saint petersburg",
    "st petersburg": "saint petersburg",
    "спб": "saint petersburg",
    "питер": "saint petersburg",
    "мск": "moscow",
}


@dataclass(frozen=True)
class TimezoneCandidate:
    city_name: str
    country_name: str
    country_code: str
    timezone: str
    population: int

    @property
    def label(self) -> str:
        return f"{self.city_name}, {self.country_name}"

    def local_time(self, now: datetime | None = None) -> datetime:
        current_time = now or datetime.now(UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        return current_time.astimezone(ZoneInfo(self.timezone))


@dataclass(frozen=True)
class TimezoneResolution:
    status: str
    candidates: tuple[TimezoneCandidate, ...]


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: TimezoneCandidate
    score: float


@lru_cache(maxsize=1)
def _timezone_aliases() -> dict[str, str]:
    return {_normalize_text(name): name for name in available_timezones()}


@lru_cache(maxsize=1)
def _city_index() -> tuple[dict[str, tuple[TimezoneCandidate, ...]], tuple[str, ...]]:
    cache = GeonamesCache()
    country_names = {
        code.upper(): data["name"]
        for code, data in cache.get_countries().items()
        if data.get("name")
    }
    entries_by_key: dict[tuple[str, str, str], dict[str, object]] = {}

    for raw_city in cache.get_cities().values():
        timezone = raw_city.get("timezone")
        city_name = raw_city.get("name")
        country_code = str(raw_city.get("countrycode") or "").upper()
        country_name = country_names.get(country_code)
        if not timezone or not city_name or not country_name:
            continue

        key = (city_name, country_code, timezone)
        entry = entries_by_key.setdefault(
            key,
            {
                "candidate": TimezoneCandidate(
                    city_name=city_name,
                    country_name=country_name,
                    country_code=country_code,
                    timezone=timezone,
                    population=int(raw_city.get("population") or 0),
                ),
                "aliases": set(),
            },
        )
        aliases: set[str] = entry["aliases"]  # type: ignore[assignment]
        aliases.add(city_name)
        aliases.add(f"{city_name}, {country_name}")
        aliases.add(f"{city_name} {country_name}")
        aliases.add(f"{city_name}, {country_code}")
        aliases.add(f"{city_name} {country_code}")
        for alternate_name in raw_city.get("alternatenames", []):
            if alternate_name:
                aliases.add(str(alternate_name))

    alias_map: dict[str, list[TimezoneCandidate]] = {}
    for entry in entries_by_key.values():
        candidate = entry["candidate"]
        for alias in entry["aliases"]:
            normalized_alias = _normalize_text(alias)
            if len(normalized_alias) < 3:
                continue
            alias_map.setdefault(normalized_alias, []).append(candidate)

    deduped_alias_map = {
        alias: tuple(_dedupe_candidates(candidates))
        for alias, candidates in alias_map.items()
    }
    return deduped_alias_map, tuple(deduped_alias_map.keys())


def normalize_timezone_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Timezone cannot be empty")

    timezone_name = _timezone_aliases().get(_normalize_text(normalized))
    if timezone_name is None:
        raise ValueError(f"Unknown timezone: {value}")

    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {value}") from exc
    return timezone_name


def resolve_timezone(query: str) -> TimezoneResolution:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return TimezoneResolution(status="not_found", candidates=())

    normalized_query = _COMMON_CITY_ALIASES.get(normalized_query, normalized_query)
    direct_timezone = _timezone_aliases().get(normalized_query)
    if direct_timezone is not None:
        return TimezoneResolution(
            status="resolved",
            candidates=(
                TimezoneCandidate(
                    city_name=direct_timezone.split("/")[-1].replace("_", " "),
                    country_name="Time zone",
                    country_code="TZ",
                    timezone=direct_timezone,
                    population=0,
                ),
            ),
        )

    alias_map, alias_keys = _city_index()
    exact_candidates = alias_map.get(normalized_query)
    if exact_candidates:
        ordered_candidates = _sort_candidates(exact_candidates)
        if len(ordered_candidates) == 1 or _has_clear_population_leader(ordered_candidates):
            return TimezoneResolution(status="resolved", candidates=(ordered_candidates[0],))
        return TimezoneResolution(
            status="ambiguous",
            candidates=tuple(ordered_candidates[:_AMBIGUOUS_MATCH_LIMIT]),
        )

    close_matches = difflib.get_close_matches(
        normalized_query,
        alias_keys,
        n=_ALIAS_MATCH_LIMIT,
        cutoff=_FUZZY_CUTOFF,
    )
    if not close_matches:
        return TimezoneResolution(status="not_found", candidates=())

    ranked_candidates = _rank_fuzzy_candidates(normalized_query, close_matches, alias_map)
    if not ranked_candidates:
        return TimezoneResolution(status="not_found", candidates=())

    top_candidate = ranked_candidates[0]
    if len(ranked_candidates) == 1 or _has_clear_fuzzy_leader(ranked_candidates):
        return TimezoneResolution(status="resolved", candidates=(top_candidate.candidate,))

    return TimezoneResolution(
        status="ambiguous",
        candidates=tuple(
            ranked.candidate for ranked in ranked_candidates[:_AMBIGUOUS_MATCH_LIMIT]
        ),
    )


def _rank_fuzzy_candidates(
    normalized_query: str,
    close_matches: list[str],
    alias_map: dict[str, tuple[TimezoneCandidate, ...]],
) -> list[_RankedCandidate]:
    scores: dict[tuple[str, str], float] = {}
    candidates_by_key: dict[tuple[str, str], TimezoneCandidate] = {}

    for alias in close_matches:
        score = difflib.SequenceMatcher(a=normalized_query, b=alias).ratio()
        for candidate in alias_map[alias]:
            key = (candidate.timezone, candidate.label)
            existing = scores.get(key, 0.0)
            if score > existing:
                scores[key] = score
                candidates_by_key[key] = candidate

    ranked = [
        _RankedCandidate(candidate=candidates_by_key[key], score=score)
        for key, score in scores.items()
    ]
    ranked.sort(
        key=lambda ranked_candidate: (
            ranked_candidate.score,
            ranked_candidate.candidate.population,
            ranked_candidate.candidate.label,
        ),
        reverse=True,
    )
    return ranked


def _sort_candidates(
    candidates: tuple[TimezoneCandidate, ...] | list[TimezoneCandidate],
) -> list[TimezoneCandidate]:
    return sorted(
        _dedupe_candidates(candidates),
        key=lambda candidate: (candidate.population, candidate.label),
        reverse=True,
    )


def _dedupe_candidates(
    candidates: tuple[TimezoneCandidate, ...] | list[TimezoneCandidate],
) -> list[TimezoneCandidate]:
    deduped: dict[tuple[str, str], TimezoneCandidate] = {}
    for candidate in candidates:
        key = (candidate.timezone, candidate.label)
        current = deduped.get(key)
        if current is None or candidate.population > current.population:
            deduped[key] = candidate
    return list(deduped.values())


def _has_clear_population_leader(candidates: list[TimezoneCandidate]) -> bool:
    if len(candidates) < 2:
        return True
    leader, runner_up = candidates[0], candidates[1]
    return leader.population >= max(1, runner_up.population) * _EXACT_POPULATION_MULTIPLIER


def _has_clear_fuzzy_leader(candidates: list[_RankedCandidate]) -> bool:
    if len(candidates) < 2:
        return True

    leader, runner_up = candidates[0], candidates[1]
    if leader.score >= 0.92 and leader.score - runner_up.score >= _FUZZY_SCORE_MARGIN:
        return True
    return leader.score >= runner_up.score and _has_clear_population_leader(
        [leader.candidate, runner_up.candidate]
    )


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = without_marks.lower().replace("&", " and ")
    collapsed = "".join(
        char if char.isalnum() or char in "/_+" else " "
        for char in lowered
    )
    return re.sub(r"\s+", " ", collapsed).strip()
