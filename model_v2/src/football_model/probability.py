"""One score distribution for all full-time markets; no separate hand-written picks."""

from datetime import datetime
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import poisson

from .domain import (Fixture, GoalRates, Lineup, LineupScenario, OddsSnapshot,
                     PlayerEffect, StintLineupScenario, aware)
from .lineup import adjust_goal_rates, integrate_projected_stints
from .match_context import MatchContext
from .vendor.penaltyblog_grid import FootballProbabilityGrid


def score_grid(home: float, away: float, rho: float = 0, max_goal: int | None = None):
    """Dixon-Coles grid with inclusive endpoints and explicit omitted-tail control.

    The upstream external-lambda factory is not used: its cross-cell factors in
    the reviewed revision disagree with its own fitted-model implementation.
    Standard tau below preserves Poisson marginals before negligible truncation.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (home, away)):
        raise ValueError("Goal rates must be finite and positive")
    if not math.isfinite(rho):
        raise ValueError("rho must be finite")
    if max_goal is None:
        max_goal = max(15, int(poisson.ppf(1 - 1e-12, max(home, away))))
    if isinstance(max_goal, bool) or not isinstance(max_goal, (int, np.integer)) or max_goal < 1:
        raise ValueError("max_goal must include the 1-1 cell")
    if poisson.sf(max_goal, home) + poisson.sf(max_goal, away) > 1e-8:
        raise ValueError("Score truncation would omit material probability")
    tau = (1 - home * away * rho, 1 + home * rho, 1 + away * rho, 1 - rho)
    if min(tau) < 0:
        raise ValueError("rho makes a low-score probability negative for these goal rates")
    goals = np.arange(max_goal + 1)
    matrix = np.outer(poisson.pmf(goals, home), poisson.pmf(goals, away))
    matrix[0, 0] *= tau[0]
    matrix[0, 1] *= tau[1]
    matrix[1, 0] *= tau[2]
    matrix[1, 1] *= tau[3]
    return FootballProbabilityGrid(matrix, home, away, normalize=True)


def implied(snapshot: OddsSnapshot) -> np.ndarray:
    raw = 1 / np.array([snapshot.home, snapshot.draw, snapshot.away], dtype=float)
    return raw / raw.sum()


def mix_grids(grids: Sequence[FootballProbabilityGrid], weights: Sequence[float]):
    if not grids or len(grids) != len(weights):
        raise ValueError("One positive weight is required per grid")
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Scenario probabilities must be positive and sum to one")
    weights = weights / weights.sum()
    if any(not np.isfinite(g.grid).all() or np.any(g.grid < 0) or
           not np.isclose(g.grid.sum(), 1) for g in grids):
        raise ValueError("Each score grid must be finite, nonnegative and normalized")
    height = max(g.grid.shape[0] for g in grids)
    width = max(g.grid.shape[1] for g in grids)
    matrix = np.zeros((height, width))
    for weight, grid in zip(weights, grids):
        h, a = grid.grid.shape
        matrix[:h, :a] += weight * grid.grid
    return FootballProbabilityGrid(
        matrix,
        float(sum(w * g.home_goal_expectation for w, g in zip(weights, grids))),
        float(sum(w * g.away_goal_expectation for w, g in zip(weights, grids))),
    )


def blend_market(grid, snapshot: OddsSnapshot, weight: float, as_of: datetime):
    """Reweight H/D/A regions, preserving the conditional score distribution.

    Weight must be chosen on earlier validation data. Never assume that reusing
    market information plus lineup news automatically creates an independent edge.
    """
    if not 0 <= weight <= 1 or snapshot.observed_at > as_of:
        raise ValueError("Invalid market weight or future odds")
    q = implied(snapshot)
    current = np.asarray(grid.home_draw_away)
    target = (1 - weight) * current + weight * q
    h, a = np.indices(grid.grid.shape)
    matrix = grid.grid.copy()
    for index, mask in enumerate((h > a, h == a, h < a)):
        if current[index] <= 0:
            raise ValueError("Cannot reweight an outcome with zero structural support")
        matrix[mask] *= target[index] / current[index]
    # Market reweighting can change the mean, so recompute it from the final grid.
    return FootballProbabilityGrid(matrix, float((matrix * h).sum()), float((matrix * a).sum()))


def jincai_handicap(grid, side: str, handicap: int) -> dict[str, float]:
    """Three-way handicap on the chosen side, not an Asian-handicap push market."""
    if side not in {"home", "away"} or not isinstance(handicap, int) or isinstance(handicap, bool):
        raise ValueError("Use home/away and an integer three-way handicap")
    h, a = np.indices(grid.grid.shape)
    adjusted = (h - a if side == "home" else a - h) + handicap
    return {"win": float(grid.grid[adjusted > 0].sum()),
            "draw": float(grid.grid[adjusted == 0].sum()),
            "lose": float(grid.grid[adjusted < 0].sum())}


def minimum_decimal_odds(p_win: float, target_ev: float = 0, p_push: float = 0):
    """E[net return] = p_win*(odds-1) - p_lose; push refunds stake."""
    if not (0 <= p_win <= 1 and 0 <= p_push <= 1 and p_win + p_push <= 1 + 1e-12):
        raise ValueError("Invalid settlement probabilities")
    if not math.isfinite(target_ev) or target_ev < 0:
        raise ValueError("target_ev must be finite and non-negative")
    if p_win == 0:
        return None
    return (1 - p_push + target_ev) / p_win


def summarize(grid) -> dict:
    h, a = np.indices(grid.grid.shape)
    totals = grid.total_goals_distribution()
    jc = jincai_handicap(grid, "away", -2)
    return {
        "one_x_two": dict(zip(("home", "draw", "away"), map(float, grid.home_draw_away))),
        "expected_goals": {"home": float((grid.grid * h).sum()), "away": float((grid.grid * a).sum())},
        "total_goals": {str(k): float(totals[k]) for k in range(4)} | {"4+": float(totals[4:].sum())},
        "over_2_5": float(grid.grid[h + a >= 3].sum()),
        "both_score": float(grid.btts_yes),
        "away_minus_two_jincai": jc,
        "away_minus_two_asian": {"win": jc["win"], "push": jc["draw"], "lose": jc["lose"]},
        "fair_odds_unbuffered": {
            "away_win": minimum_decimal_odds(grid.away_win),
            "away_minus_two_jincai_win": minimum_decimal_odds(jc["win"]),
            "away_minus_two_asian": minimum_decimal_odds(jc["win"], p_push=jc["draw"]),
        },
    }


def forecast_scenarios(
    fixture: Fixture,
    as_of: datetime,
    rates: GoalRates,
    home_reference: Lineup,
    away_reference: Lineup,
    effects: Mapping[str, PlayerEffect],
    scenarios: Sequence[LineupScenario | StintLineupScenario],
    rho: float = 0,
    market: OddsSnapshot | None = None,
    market_weight: float = 0,
    context: MatchContext | None = None,
) -> dict:
    aware(as_of)
    if as_of >= fixture.kickoff:
        raise ValueError("This interface predicts pre-match regulation scores")
    if rates.fixture_id != fixture.match_id or (market is not None and market.fixture_id != fixture.match_id):
        raise ValueError("Rates/odds and fixture identities disagree")
    if home_reference.team != fixture.home or away_reference.team != fixture.away:
        raise ValueError("Reference lineups and fixture team identities disagree")
    if market is None and market_weight != 0:
        raise ValueError("A market weight requires an observed odds snapshot")
    context_details = None if context is None else context.for_prediction(fixture, as_of, scenarios)
    grids, details = [], []
    for scenario in scenarios:
        if isinstance(scenario, StintLineupScenario):
            adjusted, delta = integrate_projected_stints(rates, scenario.stints, home_reference,
                                                       away_reference, effects, as_of)
            method = "integrated_fixed_projected_stints"
        else:
            adjusted, delta = adjust_goal_rates(rates, scenario.home, scenario.away,
                                               home_reference, away_reference, effects, as_of)
            method = "expected_minute_plug_in_approximation"
        grid = score_grid(adjusted.home, adjusted.away, rho)
        grids.append(grid)
        details.append({"label": scenario.label, "weight": scenario.weight,
                        "lineup_rate_method": method,
                        "lineup_adjustment_details": delta, **summarize(grid)})
    mixed = mix_grids(grids, [s.weight for s in scenarios])
    structural = summarize(mixed)
    if market is not None:
        mixed = blend_market(mixed, market, market_weight, as_of)
    matrix = np.array([g.home_draw_away for g in grids])
    weights = np.asarray([s.weight for s in scenarios])
    weights = weights / weights.sum()
    mean = weights @ matrix
    within = weights @ (matrix * (1 - matrix))
    between = weights @ np.square(matrix - mean)
    return {
        "match_id": fixture.match_id,
        "fixture": {"home": fixture.home, "away": fixture.away, "competition": fixture.competition,
                    "kickoff": fixture.kickoff.isoformat(), "neutral": fixture.neutral},
        "as_of": as_of.isoformat(),
        "market_weight": market_weight,
        "prediction_context": context_details,
        "structural_summary": structural,
        "market_de_vig_one_x_two": None if market is None else dict(
            zip(("home", "draw", "away"), map(float, implied(market)))),
        "summary": summarize(mixed),
        "scenarios": details,
        "pre_market_scenario_range_not_confidence_interval": {
            name: {"min": float(matrix[:, i].min()), "max": float(matrix[:, i].max())}
            for i, name in enumerate(("home", "draw", "away"))
        },
        "half_full_time": None,
        "half_full_time_note": "Requires a separately fitted joint half-time process",
        "statistical_parameter_intervals": None,
        "pre_market_scenario_variance_decomposition": {
            name: {"within_scenario_event_variance": float(within[i]),
                   "between_scenario_probability_variance": float(between[i]),
                   "total_event_variance": float(mean[i] * (1 - mean[i]))}
            for i, name in enumerate(("home", "draw", "away"))
        },
        "variance_note": (
            "Conditional on supplied scenarios and weights, not a confidence interval. "
            "Identical scenarios can have zero between-scenario variance despite substantial omitted uncertainty."
        ),
        "reference_calibration_note": (
            "Lineup input rates must be conditional-reference rates. TeamGoalModel outputs marginal rates; "
            "a real-data conditional-reference calibration or joint fit is not implemented. "
            "Reference-window linkage and data provenance remain adapter responsibilities."
        ),
    }
