"""Prediction-time records; all times are timezone-aware and represent availability."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value


@dataclass(frozen=True)
class Fixture:
    match_id: str
    home: str
    away: str
    competition: str
    kickoff: datetime
    neutral: bool = False

    def __post_init__(self):
        aware(self.kickoff)
        if not isinstance(self.neutral, bool):
            raise ValueError("neutral must be a boolean, not a CSV string")
        if self.home == self.away:
            raise ValueError("Teams must differ")


@dataclass(frozen=True)
class MatchResult:
    fixture: Fixture
    home_goals: int
    away_goals: int
    available_at: datetime

    def __post_init__(self):
        aware(self.available_at)
        if self.available_at < self.fixture.kickoff:
            raise ValueError("Result cannot be available before kickoff")
        for value in (self.home_goals, self.away_goals):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Completed goals must be non-negative integers")


@dataclass(frozen=True)
class PlayerEffect:
    player_id: str
    attack_log_rate: float
    defense_log_rate: float
    fitted_at: datetime
    minutes_seen: float

    def __post_init__(self):
        aware(self.fitted_at)
        if not all(math.isfinite(x) for x in (
            self.attack_log_rate, self.defense_log_rate, self.minutes_seen
        )) or self.minutes_seen < 0:
            raise ValueError("Player effects and exposure must be finite")


@dataclass(frozen=True)
class Lineup:
    """Expected regulation minutes, including substitutes, for one coherent scenario.

    Forecast minutes are expectations known at observed_at, not actual future minutes.
    Reference minutes are training-window weighted averages, not an all-season XI.
    """

    minutes: Mapping[str, float]
    positions: Mapping[str, str]
    observed_at: datetime
    source: str
    team: str

    def __post_init__(self):
        aware(self.observed_at)
        if set(self.minutes) != set(self.positions):
            raise ValueError("Every included player needs a position")
        if any(not math.isfinite(m) or m < 0 or m > 90 for m in self.minutes.values()):
            raise ValueError("Expected minutes must be between 0 and 90")
        if not math.isclose(sum(self.minutes.values()), 990, abs_tol=1e-6):
            raise ValueError("An 11-player regulation scenario must allocate 990 minutes")
        goalkeeper_minutes = sum(m for p, m in self.minutes.items() if self.positions[p] == "GK")
        if not math.isclose(goalkeeper_minutes, 90, abs_tol=1e-6):
            raise ValueError("Goalkeepers must jointly cover 90 minutes")
        if not self.source:
            raise ValueError("Lineup source is required")
        if not self.team:
            raise ValueError("Lineup team identity is required")


@dataclass(frozen=True)
class LineupScenario:
    label: str
    weight: float
    home: Lineup
    away: Lineup

    def __post_init__(self):
        if not math.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise ValueError("Scenario weights must be positive and no greater than 1")


@dataclass(frozen=True)
class GoalRates:
    home: float
    away: float
    estimated_at: datetime
    source: str
    fixture_id: str
    basis: str = "conditional_reference"

    def __post_init__(self):
        aware(self.estimated_at)
        if any(not math.isfinite(x) or x <= 0 for x in (self.home, self.away)):
            raise ValueError("Goal rates must be finite and positive")
        if not self.fixture_id:
            raise ValueError("Goal rates require a fixture identity")
        if self.basis not in {"conditional_reference", "team_marginal"}:
            raise ValueError("Unknown goal-rate reference basis")


@dataclass(frozen=True)
class OddsSnapshot:
    home: float
    draw: float
    away: float
    observed_at: datetime
    bookmaker: str
    fixture_id: str

    def __post_init__(self):
        aware(self.observed_at)
        if any(not math.isfinite(x) or x <= 1 for x in (self.home, self.draw, self.away)):
            raise ValueError("Decimal odds must be finite and greater than 1")
        if not self.fixture_id:
            raise ValueError("Odds require a fixture identity")


@dataclass(frozen=True)
class ProjectedStint:
    """A pre-match assumed constant-XI interval, not an actual future substitution."""

    duration_minutes: float
    home: Lineup
    away: Lineup

    def __post_init__(self):
        if not math.isfinite(self.duration_minutes) or not 0 < self.duration_minutes <= 90:
            raise ValueError("Projected stint duration must be in (0, 90]")
        for lineup in (self.home, self.away):
            if any(m not in (0, 90) for m in lineup.minutes.values()):
                raise ValueError("A constant-XI stint must use 0/90 on-pitch indicators")


@dataclass(frozen=True)
class StintLineupScenario:
    label: str
    weight: float
    stints: tuple[ProjectedStint, ...]

    def __post_init__(self):
        if not math.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise ValueError("Scenario weights must be positive and no greater than 1")
        if not self.stints or not math.isclose(sum(s.duration_minutes for s in self.stints), 90):
            raise ValueError("Ordered projected stints must cover 90 regulation minutes")
