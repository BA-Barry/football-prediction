"""Relative-lineup corrections in log goal-rate units, not raw on/off or xG/90."""

from datetime import datetime
from math import exp
from typing import Mapping

from .domain import GoalRates, Lineup, PlayerEffect, ProjectedStint, aware


def lineup_delta(
    projected: Lineup,
    reference: Lineup,
    effects: Mapping[str, PlayerEffect],
    as_of: datetime,
) -> tuple[float, float]:
    aware(as_of)
    if projected.team != reference.team:
        raise ValueError("Projected and reference lineups belong to different teams")
    if projected.observed_at > as_of or reference.observed_at > as_of:
        raise ValueError("Lineup information was unavailable at prediction time")
    attack = defense = 0.0
    for player in set(projected.minutes) | set(reference.minutes):
        change = (projected.minutes.get(player, 0) - reference.minutes.get(player, 0)) / 90
        if abs(change) < 1e-12:
            continue
        if player not in effects:
            raise ValueError(f"Missing fitted effect or explicit position prior: {player}")
        effect = effects[player]
        if effect.player_id != player:
            raise ValueError("Player-effect key and player identity disagree")
        if effect.fitted_at > as_of:
            raise ValueError(f"Player rating uses future data: {player}")
        attack += change * effect.attack_log_rate
        defense += change * effect.defense_log_rate
    return attack, defense


def adjust_goal_rates(
    rates: GoalRates,
    home: Lineup,
    away: Lineup,
    home_reference: Lineup,
    away_reference: Lineup,
    effects: Mapping[str, PlayerEffect],
    as_of: datetime,
) -> tuple[GoalRates, dict[str, float]]:
    """Expected-minute plug-in approximation, not exact substitution integration.

    Team marginal rates must first be converted by a fitted conditional-reference
    model. Merely centering minutes does not correct Jensen's inequality drift.
    """
    if rates.estimated_at > as_of:
        raise ValueError("Team estimate was unavailable at prediction time")
    if rates.basis != "conditional_reference":
        raise ValueError("Marginal team rates need fitted conditional-reference calibration before lineup adjustment")
    ah, dh = lineup_delta(home, home_reference, effects, as_of)
    aa, da = lineup_delta(away, away_reference, effects, as_of)
    adjusted = GoalRates(
        rates.home * exp(ah - da),
        rates.away * exp(aa - dh),
        as_of,
        f"{rates.source}; expected-minute lineup deviation",
        rates.fixture_id,
    )
    return adjusted, {"home_attack": ah, "home_defense": dh,
                      "away_attack": aa, "away_defense": da}


def integrate_projected_stints(
    rates: GoalRates,
    stints: tuple[ProjectedStint, ...],
    home_reference: Lineup,
    away_reference: Lineup,
    effects: Mapping[str, PlayerEffect],
    as_of: datetime,
) -> tuple[GoalRates, list[dict]]:
    """Integrate exp(delta) over one deterministic, pre-match projected path.

    Exact only conditional on these fixed on-pitch paths and the supplied rate
    model, not for score-dependent future substitutions or changing game states.
    A separately fitted half-time process is still needed for HT/FT markets.
    """
    if not stints or abs(sum(s.duration_minutes for s in stints) - 90) > 1e-6:
        raise ValueError("Projected stints must cover 90 regulation minutes")
    home_total = away_total = 0.0
    details = []
    for stint in stints:
        adjusted, delta = adjust_goal_rates(rates, stint.home, stint.away,
                                           home_reference, away_reference, effects, as_of)
        exposure = stint.duration_minutes / 90
        home_total += exposure * adjusted.home
        away_total += exposure * adjusted.away
        details.append({"duration_minutes": stint.duration_minutes,
                        "full_90_home_rate": adjusted.home, "full_90_away_rate": adjusted.away,
                        "delta_log_rate": delta})
    return GoalRates(home_total, away_total, as_of, rates.source + "; integrated projected stints",
                     rates.fixture_id), details
