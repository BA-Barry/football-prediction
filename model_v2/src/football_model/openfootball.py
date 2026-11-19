"""Date-level openfootball records, deliberately separate from timed predictions.

Source clock text has no verified timezone. Fetch time is not historical first
availability. No Fixture, MatchResult, OddsSnapshot or fabricated UTC timestamp
is produced by this adapter.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Any

from .domain import aware


@dataclass
class DateLevelMatch:
    source_match_key: str
    source_url: str
    competition: str
    season: str
    round_label: str
    match_date: date
    source_time_text: str | None
    home_key: str
    away_key: str
    home_label: str
    away_label: str
    full_time: tuple[int, int] | None
    half_time: tuple[int, int] | None
    observed_at: datetime
    score_format: str = "object"
    kickoff_utc: None = None
    historical_first_published_at: None = None
    timezone_status: str = "unverified"

    def as_record(self):
        value = asdict(self)
        value["match_date"] = self.match_date.isoformat()
        value["observed_at"] = self.observed_at.isoformat()
        return value


def _score(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} requires exactly two reported goals or null")
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError(f"{name} goals must be non-negative integers, not booleans")
    return tuple(value)


def score_format(row: Any) -> str:
    """Describe the original JSON shape, including invalid/rejected rows."""
    if not isinstance(row, dict):
        return "invalid_match_row"
    if "score" not in row:
        return "absent"
    value = row["score"]
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "unsupported"


def parse_match(row: dict, *, competition: str, season: str,
                source_url: str, observed_at: datetime) -> DateLevelMatch:
    aware(observed_at)
    if not isinstance(row, dict):
        raise ValueError("A match row must be an object")
    for field in ("date", "team1", "team2"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Missing string field: {field}")
    if row["team1"] == row["team2"]:
        raise ValueError("Home and away teams must differ")
    match_date = date.fromisoformat(row["date"])
    if row["date"] != match_date.isoformat():
        raise ValueError("Match dates must use YYYY-MM-DD")
    clock = row.get("time")
    if clock is not None:
        if not isinstance(clock, str):
            raise ValueError("Source time must be clock text or null")
        parsed = time.fromisoformat(clock)
        if parsed.tzinfo is not None:
            raise ValueError("Unexpected timezone-bearing time; review source semantics")
    scores = row.get("score")
    original_format = score_format(row)
    if scores is None:
        full, half = None, None
    elif isinstance(scores, dict):
        # Only the explicitly reported FT/HT are read. ET/penalty scores never
        # stand in for missing FT, nor do we assert a regulation-90 meaning.
        full, half = _score(scores.get("ft"), "ft"), _score(scores.get("ht"), "ht")
    elif isinstance(scores, (list, tuple)):
        # Current upstream files mix {ft, ht} with a bare reported FT pair.
        full, half = _score(scores, "score"), None
    else:
        raise ValueError("Reported scores must be an object, two-goal array or null")
    if full is not None and half is not None and any(h > f for h, f in zip(half, full)):
        raise ValueError("Half-time score exceeds reported full-time score")
    round_label = row.get("round", "")
    if not isinstance(round_label, str):
        raise ValueError("Round label must be text")
    # A readable source-local key, not a global team ID. Date is excluded so a
    # changed date with the same round/opponents is reported as a duplicate.
    key = f"openfootball:{competition}:{season}:{round_label}:{row['team1']}|{row['team2']}"
    country = competition.split(".")[0]
    return DateLevelMatch(
        key, source_url, competition, season, round_label, match_date, clock,
        f"openfootball:{country}:{row['team1']}", f"openfootball:{country}:{row['team2']}",
        row["team1"], row["team2"], full, half, observed_at,
        score_format=original_format,
    )


def inspect_payload(payload: dict, *, competition: str, season: str,
                    source_url: str, observed_at: datetime) -> dict:
    aware(observed_at)
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ValueError("Expected a JSON object containing a matches array")
    matches, issues = [], []
    for index, row in enumerate(payload["matches"]):
        try:
            matches.append(parse_match(row, competition=competition, season=season,
                                       source_url=source_url, observed_at=observed_at))
        except (ValueError, TypeError) as error:
            issues.append({"row_index": index, "issue": str(error)})
    keys = Counter(m.source_match_key for m in matches)
    duplicates = [key for key, count in keys.items() if count > 1]
    for key in duplicates:
        issues.append({"source_match_key": key, "issue": "Duplicate or conflicting source-local match key; no row silently overwritten"})
    dates = [m.match_date for m in matches]
    teams = {t for m in matches for t in (m.home_key, m.away_key)}
    return {
        "competition": competition, "season": season, "source_url": source_url,
        "reported_name": payload.get("name"), "observed_at": observed_at.isoformat(),
        "input_rows": len(payload["matches"]), "parsed_rows": len(matches),
        "rejected_rows": len(payload["matches"]) - len(matches),
        "raw_score_formats": dict(Counter(score_format(row) for row in payload["matches"])),
        "full_time_rows": sum(m.full_time is not None for m in matches),
        "missing_full_time_rows": sum(m.full_time is None for m in matches),
        "missing_full_time_rounds": dict(Counter(m.round_label for m in matches if m.full_time is None)),
        "half_time_rows": sum(m.half_time is not None for m in matches),
        "missing_half_time_rows": sum(m.half_time is None for m in matches),
        "clock_text_rows": sum(m.source_time_text is not None for m in matches),
        "verified_utc_kickoff_rows": 0, "historical_first_availability_rows": 0,
        "odds_rows": 0, "lineup_rows": 0, "injury_rows": 0,
        "teams": len(teams), "duplicate_keys": duplicates,
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "issues": issues, "records": [m.as_record() for m in matches],
        "notes": [
            "Source reports FT; regulation-90 interpretation must be verified before other competitions are added",
            "No historical T-24h/official-lineup information set reconstructed",
            "Source-local identity keys require explicit alias mapping before cross-source or cross-competition use",
            "CC0 is the repository's declaration; per-record original provenance is not fully established",
        ],
    }
