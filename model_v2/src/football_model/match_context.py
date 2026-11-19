"""Prediction snapshots and sourced competition context, without invented effects.

Recorded evidence is explanatory metadata until historical fitting establishes
incremental predictive value. It never silently applies a 'motivation penalty'.
Official starters identify the first XI, not future substitutions or effort.
"""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from .domain import Fixture, LineupScenario, StintLineupScenario, aware


@dataclass(frozen=True)
class ContextEvidence:
    feature: str
    value: str | float | int | bool | None
    source: str
    observed_at: datetime
    published_at: datetime | None = None

    def __post_init__(self):
        aware(self.observed_at)
        if not self.feature or not self.source:
            raise ValueError("Context evidence requires a feature name and source")
        if self.published_at is not None:
            aware(self.published_at)
            if self.published_at > self.observed_at:
                raise ValueError("Evidence cannot be observed before publication")
        if self.value is not None and not isinstance(self.value, (str, int, float, bool)):
            raise ValueError("Context values must be scalar facts, descriptions or explicit unknowns")
        if isinstance(self.value, (float, int)) and not math.isfinite(self.value):
            raise ValueError("Numeric context evidence must be finite")


@dataclass(frozen=True)
class TeamCompetitionContext:
    team: str
    evidence: tuple[ContextEvidence, ...] = ()
    missing_information: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.team:
            raise ValueError("Competition context requires team identity")


@dataclass(frozen=True)
class LineupAnnouncement:
    team: str
    starters: tuple[str, ...]
    source: str
    published_at: datetime
    observed_at: datetime
    fixture_id: str

    def __post_init__(self):
        aware(self.published_at)
        aware(self.observed_at)
        if not self.team or not self.source or not self.fixture_id or len(self.starters) != 11 or len(set(self.starters)) != 11:
            raise ValueError("An official lineup needs a team, source and 11 distinct starters")
        if any(not isinstance(p, str) or not p for p in self.starters):
            raise ValueError("Official starters require stable player IDs")
        if self.published_at > self.observed_at:
            raise ValueError("Lineup cannot be observed before publication")


@dataclass(frozen=True)
class MatchContext:
    fixture_id: str
    version: str
    home: TeamCompetitionContext
    away: TeamCompetitionContext
    announcements: tuple[LineupAnnouncement, ...] = ()
    scenario_weight_source: str = "Unspecified; scenario weights are not claimed to be calibrated"
    scenario_weights_as_of: datetime | None = None

    def __post_init__(self):
        if self.version not in {"day_before", "lineup_confirmed"}:
            raise ValueError("Prediction version must be day_before or lineup_confirmed")
        if not self.fixture_id or not self.scenario_weight_source:
            raise ValueError("Context needs fixture identity and scenario-weight provenance")
        if self.scenario_weights_as_of is not None:
            aware(self.scenario_weights_as_of)

    def for_prediction(
        self, fixture: Fixture, as_of: datetime,
        scenarios: Sequence[LineupScenario | StintLineupScenario],
    ) -> dict:
        aware(as_of)
        if as_of >= fixture.kickoff:
            raise ValueError("Context must be used before kickoff")
        if self.fixture_id != fixture.match_id or (self.home.team, self.away.team) != (fixture.home, fixture.away):
            raise ValueError("Competition context and fixture identities disagree")
        if self.scenario_weights_as_of is not None and self.scenario_weights_as_of > as_of:
            raise ValueError("Scenario weights were unavailable at prediction time")
        teams = []
        for context in (self.home, self.away):
            rows = []
            for item in context.evidence:
                if item.observed_at > as_of:
                    raise ValueError("Future competition-context evidence")
                rows.append({"feature": item.feature, "value": item.value, "source": item.source,
                             "published_at": None if item.published_at is None else item.published_at.isoformat(),
                             "observed_at": item.observed_at.isoformat()})
            teams.append({"team": context.team, "evidence": rows,
                          "missing_information": list(context.missing_information)})
        announcements = {}
        for item in self.announcements:
            if item.fixture_id != fixture.match_id:
                raise ValueError("Official lineup announcement belongs to another fixture")
            if item.team not in {fixture.home, fixture.away} or item.team in announcements:
                raise ValueError("Official lineups must identify distinct fixture teams")
            if item.observed_at > as_of:
                raise ValueError("Official lineup was unavailable at prediction time")
            announcements[item.team] = item
        if self.version == "lineup_confirmed":
            if set(announcements) != {fixture.home, fixture.away}:
                raise ValueError("Confirmed version needs both official lineups")
            for scenario in scenarios:
                if not isinstance(scenario, StintLineupScenario):
                    raise ValueError("Official-XI conditioning requires projected stints, not only average minutes")
                for lineup in (scenario.stints[0].home, scenario.stints[0].away):
                    starters = {p for p, m in lineup.minutes.items() if m == 90}
                    if lineup.team not in announcements or starters != set(announcements[lineup.team].starters):
                        raise ValueError("Projected initial XI disagrees with official starters")
        return {
            "version": self.version,
            "hours_to_kickoff": (fixture.kickoff - as_of).total_seconds() / 3600,
            "team_contexts": teams,
            "official_lineups": [
                {"team": a.team, "fixture_id": a.fixture_id, "starters": list(a.starters), "source": a.source,
                 "published_at": a.published_at.isoformat(), "observed_at": a.observed_at.isoformat()}
                for a in announcements.values()
            ],
            "scenario_weight_source": self.scenario_weight_source,
            "scenario_weights_as_of": None if self.scenario_weights_as_of is None else self.scenario_weights_as_of.isoformat(),
            "weight_provenance_note": (
                "A supplied timestamp still requires trustworthy upstream records; "
                "missing timestamp means weight availability cannot be checked."
            ),
            "competition_context_applied_to_rates": False,
            "context_note": (
                "Sourced context is recorded but no untrained motivation/fatigue coefficient is applied. "
                "Observed rotation may already enter the supplied lineup scenarios; do not deduct it twice. "
                "Official starters do not determine future substitutions, tactics or match outcomes."
            ),
        }


def compare_versions(early: dict, confirmed: dict) -> dict:
    """Describe two stored snapshots; differences are not causal news attribution."""
    if early["match_id"] != confirmed["match_id"]:
        raise ValueError("Snapshots must describe the same fixture")
    if early.get("fixture") != confirmed.get("fixture") or not early.get("fixture"):
        raise ValueError("Snapshot teams, kickoff and competition target must agree")
    if ((early.get("prediction_context") or {}).get("version"),
        (confirmed.get("prediction_context") or {}).get("version")) != ("day_before", "lineup_confirmed"):
        raise ValueError("Compare a day-before snapshot with a confirmed-lineup snapshot")
    first, second = datetime.fromisoformat(early["as_of"]), datetime.fromisoformat(confirmed["as_of"])
    aware(first)
    aware(second)
    if first >= second:
        raise ValueError("Confirmed snapshot must follow the early snapshot")

    def delta(a, b):
        return {name: b[name] - a[name] for name in ("home", "draw", "away")}

    market_early, market_late = early["market_de_vig_one_x_two"], confirmed["market_de_vig_one_x_two"]
    return {
        "match_id": early["match_id"],
        "early_as_of": early["as_of"], "confirmed_as_of": confirmed["as_of"],
        "structural_one_x_two_change": delta(early["structural_summary"]["one_x_two"],
                                                confirmed["structural_summary"]["one_x_two"]),
        "market_one_x_two_change": None if market_early is None or market_late is None else delta(market_early, market_late),
        "final_one_x_two_change": delta(early["summary"]["one_x_two"], confirmed["summary"]["one_x_two"]),
        "market_weights": {"early": early["market_weight"], "confirmed": confirmed["market_weight"]},
        "note": "Descriptive snapshot differences, not isolated causal effects of motivation, lineups or price changes",
    }
