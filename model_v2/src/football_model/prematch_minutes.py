"""Transparent historical-minute extrapolations, isolated from player effects.

The input unit is one *complete* team on-pitch exposure record in two fixed
45-minute half windows. It is not an available squad, an injury report, a
starting-XI prediction, or untrimmed match time including stoppage. Missing a
player in a complete historical exposure record means that player did not play
in that historical window; it says nothing about current availability.

Calendar dates are deliberately conservative: a record dated on the prediction
day is never used. Earlier dates are assumed usable the next day for historical
research, because the existing sources do not establish first publication time.
This is not a claim to reconstruct a verified historical information set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import fsum, isfinite
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping


MINUTES_BASIS = "fixed_45_each_half"
TEAM_REFERENCE_MINUTES = 990.0
METHODS = ("last_match", "recent_mean")


def _calendar_date(value: date, name: str) -> date:
    # datetime is a date subclass but must not silently lose time or timezone.
    if type(value) is not date:
        raise ValueError(f"{name} must be a calendar date, not a timestamp")
    return value


def _team_id(value: int | str) -> int | str:
    if (type(value) is int and value > 0
            or isinstance(value, str) and value.strip()):
        return value
    raise ValueError("team_id must be a positive source ID or nonempty string")


@dataclass(frozen=True)
class TeamMinutesObservation:
    """Complete retrospective team exposure; never the target match's features.

    ``source`` includes the provider/clock scenario where these differ. Identity
    is not merged across providers. The caller must supply a complete team path;
    partial player samples cannot become valid merely by having a sum <= 990.
    Bench-only players may be retained at zero; recording their identities does
    not establish that they belong to a future squad.
    """

    match_id: str
    team_id: int | str
    match_date: date
    minutes: Mapping[str, float]
    source: str
    basis: str = MINUTES_BASIS

    def __post_init__(self):
        if not isinstance(self.match_id, str) or not self.match_id.strip():
            raise ValueError("A source match_id is required")
        _team_id(self.team_id)
        _calendar_date(self.match_date, "match_date")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("An explicit source/provider/scenario is required")
        if self.basis != MINUTES_BASIS:
            raise ValueError("Only complete fixed_45_each_half exposure is supported")
        if not isinstance(self.minutes, Mapping) or not self.minutes:
            raise ValueError("A nonempty complete team minutes mapping is required")
        copied = {}
        for player_id, value in self.minutes.items():
            if not isinstance(player_id, str) or not player_id.strip():
                raise ValueError("Player identities must be nonempty source strings")
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or not 0 <= value <= 90):
                raise ValueError("Fixed-window player minutes must be finite in [0, 90]")
            copied[player_id] = float(value)
        total = fsum(copied.values())
        if not 0 < total <= TEAM_REFERENCE_MINUTES + 1e-6:
            raise ValueError("Complete team exposure must be positive and no more than 990 minutes")
        # A caller changing its original dict must not rewrite recorded history.
        object.__setattr__(self, "minutes", MappingProxyType(copied))


def estimate_minutes(
    history: Iterable[TeamMinutesObservation],
    *,
    team_id: int | str,
    prediction_date: date,
    source: str,
    method: str = "last_match",
    lookback: int = 5,
) -> dict:
    """Return an unadjusted minute candidate from strictly earlier team dates.

    ``last_match`` copies the latest complete historical team exposure. For
    ``recent_mean`` the denominator is the actual number of selected *team
    matches*, up to ``lookback``; no minimum-history padding is invented. The
    player universe is the union of those historical records, never a target
    lineup/roster. All records are drawn from exactly one source and one team.

    No transfers, injuries, suspension status, current squad, positions, or player
    effects enter. Missing identities remain omissions, not fitted zero effects.
    A forecast can total less than 990 after historical dismissals/temporary
    absences. It is neither rescaled nor automatically converted to ``Lineup``.
    A no-history result is unknown (None totals), not a zero-minute prediction.
    """
    _team_id(team_id)
    _calendar_date(prediction_date, "prediction_date")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("An explicit source/provider/scenario is required")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if type(lookback) is not int or lookback < 1:
        raise ValueError("lookback must be a positive integer")

    eligible = []
    excluded = {"other_source_or_team": 0, "same_or_later_date": 0}
    seen_ids, seen_dates = set(), set()
    for record in history:
        if not isinstance(record, TeamMinutesObservation):
            raise ValueError("history requires TeamMinutesObservation records")
        if record.source != source or record.team_id != team_id:
            excluded["other_source_or_team"] += 1
            continue
        if record.match_date >= prediction_date:
            excluded["same_or_later_date"] += 1
            continue
        if record.match_id in seen_ids:
            raise ValueError("Duplicate historical source/team/match identity")
        if record.match_date in seen_dates:
            raise ValueError("Same-date historical team matches have no reliable last-match order")
        seen_ids.add(record.match_id)
        seen_dates.add(record.match_date)
        eligible.append(record)
    eligible.sort(key=lambda record: record.match_date)
    requested = 1 if method == "last_match" else lookback
    selected = eligible[-requested:]
    predicted = {}
    if selected:
        players = sorted({player for record in selected for player in record.minutes})
        predicted = {
            player: fsum(record.minutes.get(player, 0.0) for record in selected) / len(selected)
            for player in players
        }
    total = fsum(predicted.values()) if selected else None
    return {
        "status": "estimated" if selected else "no_eligible_history",
        "team_id": team_id,
        "source": source,
        "prediction_date": prediction_date.isoformat(),
        "method": method,
        "minutes_basis": MINUTES_BASIS,
        "history_matches_requested": requested,
        "history_matches_available": len(eligible),
        "history_matches_used": len(selected),
        "history_match_ids": [record.match_id for record in selected],
        "history_dates": [record.match_date.isoformat() for record in selected],
        "history_team_totals": [fsum(record.minutes.values()) for record in selected],
        "excluded_records": excluded,
        "predicted_minutes": predicted,
        "total_predicted_minutes": total,
        "full_team_reference_minutes": TEAM_REFERENCE_MINUTES,
        "unallocated_relative_to_990": None if total is None else TEAM_REFERENCE_MINUTES - total,
        "current_roster_known": False,
        "historical_publication_verified": False,
        "availability_semantics": (
            "strict match_date < prediction_date; next-day availability assumed for research; "
            "source first publication time unknown, not a verified historical as-of reconstruction"
        ),
        "interpretation": (
            "history-only minute extrapolation; current roster, transfers, injuries and availability "
            "unknown; newcomers can be omitted and departed/absent players retained; no target "
            "roster filtering, no zero player-effect prior, no 990-minute repair or Lineup conversion"
        ),
        "coverage_assumption": (
            "each supplied observation covers the complete team on-pitch path in two fixed "
            "45-minute half windows; an absent ID is zero only for that historical exposure"
        ),
    }
