from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from scipy.stats import poisson

from football_model.demo import synthetic_inputs
from football_model.domain import Lineup, OddsSnapshot, ProjectedStint, StintLineupScenario
from football_model.lineup import adjust_goal_rates, integrate_projected_stints
from football_model.probability import (
    blend_market, forecast_scenarios, implied, jincai_handicap, minimum_decimal_odds,
    mix_grids, score_grid, summarize,
)


def test_asymmetric_dixon_coles_cells_conserve_marginals():
    g = score_grid(2, 1, rho=-0.1)
    assert g.grid[1, 0] == pytest.approx(poisson.pmf(1, 2) * poisson.pmf(0, 1) * 0.9, abs=1e-10)
    assert g.grid[0, 1] == pytest.approx(poisson.pmf(0, 2) * poisson.pmf(1, 1) * 0.8, abs=1e-10)
    assert np.dot(np.arange(len(g.grid)), g.grid.sum(axis=1)) == pytest.approx(2, abs=1e-8)
    assert np.dot(np.arange(len(g.grid)), g.grid.sum(axis=0)) == pytest.approx(1, abs=1e-8)
    assert sum(g.home_draw_away) == pytest.approx(1)


def test_jincai_and_asian_two_goal_difference():
    g = score_grid(0.83, 2.72)
    jc = jincai_handicap(g, "away", -2)
    assert jc["win"] == pytest.approx(0.35173, abs=0.0001)
    assert jc["draw"] == pytest.approx(0.21252, abs=0.0001)
    assert minimum_decimal_odds(jc["win"]) > minimum_decimal_odds(jc["win"], p_push=jc["draw"])
    assert sum(summarize(g)["total_goals"].values()) == pytest.approx(1)


def test_striker_replacement_and_goalkeeper_direction():
    fixture, cutoff, rates, home, away, effects, scenarios = synthetic_inputs()
    regular, _ = adjust_goal_rates(rates, home, away, home, away, effects, cutoff)
    out, _ = adjust_goal_rates(rates, scenarios[1].home, away, home, away, effects, cutoff)
    backup, _ = adjust_goal_rates(rates, scenarios[2].home, scenarios[2].away, home, away, effects, cutoff)
    assert regular.home == rates.home
    assert out.home < regular.home
    assert backup.home > out.home
    assert out.away == regular.away


def test_scenario_mixture_not_average_lambda():
    a, b = score_grid(0.5, 1.3), score_grid(3.0, 1.3)
    mixture = mix_grids([a, b], [0.5, 0.5])
    assert mixture.home_win == pytest.approx((a.home_win + b.home_win) / 2)
    assert abs(mixture.home_win - score_grid(1.75, 1.3).home_win) > 0.01


def test_market_reweighting_remains_one_distribution():
    _, cutoff, *_ = synthetic_inputs()
    odds = OddsSnapshot(2, 3.5, 4, cutoff, "synthetic", "SYNTHETIC-NOT-A-REAL-MATCH")
    grid = blend_market(score_grid(1.8, 1.0), odds, 1, cutoff)
    assert grid.home_draw_away == pytest.approx(implied(odds))
    assert sum(summarize(grid)["total_goals"].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="future"):
        blend_market(grid, replace(odds, observed_at=cutoff + timedelta(seconds=1)), 0.5, cutoff)


def test_dynamic_odds_threshold_changes_with_scenario():
    args = synthetic_inputs()
    result = forecast_scenarios(*args, rho=-0.05)
    assert result["half_full_time"] is None
    assert result["statistical_parameter_intervals"] is None
    assert result["scenarios"][0]["fair_odds_unbuffered"] != result["scenarios"][1]["fair_odds_unbuffered"]


def test_unknown_player_and_future_lineup_not_silently_neutral():
    _, cutoff, rates, home, away, effects, scenarios = synthetic_inputs()
    with pytest.raises(ValueError, match="Missing"):
        adjust_goal_rates(rates, scenarios[1].home, away, home, away, {}, cutoff)
    with pytest.raises(ValueError, match="unavailable"):
        adjust_goal_rates(rates, replace(home, observed_at=cutoff + timedelta(days=1)), away,
                          home, away, effects, cutoff)


def test_impossible_lineup_and_invalid_rho():
    _, _, _, home, *_ = synthetic_inputs()
    with pytest.raises(ValueError, match="990"):
        Lineup({p: m / 2 for p, m in home.minutes.items()}, home.positions, home.observed_at, "bad", home.team)
    with pytest.raises(ValueError, match="negative"):
        score_grid(2, 1, rho=0.9)
    with pytest.raises(ValueError, match="omit"):
        score_grid(10, 10, max_goal=5)


def test_stint_integral_differs_from_average_minutes_exponent():
    import math
    fixture, cutoff, rates, home, away, effects, scenarios = synthetic_inputs()
    striker, sub = "H_FW_10", "H_FW_sub"
    # Reference XI uses the substitute throughout. First half uses a log(4) striker.
    reference = scenarios[1].home
    full_minutes = dict(reference.minutes)
    full_minutes[striker], full_minutes[sub] = 90, 0
    starting = replace(reference, minutes=full_minutes)
    away_minutes = dict(away.minutes)
    away_minutes["A_FW_10"], away_minutes["A_FW_sub"] = 90, 0
    away_fixed = replace(away, minutes=away_minutes)
    effects[striker] = replace(effects[striker], attack_log_rate=math.log(4))
    effects[sub] = replace(effects[sub], attack_log_rate=0)
    rates = replace(rates, home=1)
    stints = (ProjectedStint(45, starting, away_fixed), ProjectedStint(45, reference, away_fixed))
    integrated, _ = integrate_projected_stints(rates, stints, reference, away_fixed, effects, cutoff)
    average = replace(reference, minutes={**reference.minutes, striker: 45, sub: 45})
    approximate, _ = adjust_goal_rates(rates, average, away_fixed, reference, away_fixed, effects, cutoff)
    assert integrated.home == pytest.approx(2.5)
    assert approximate.home == pytest.approx(2)
    result = forecast_scenarios(fixture, cutoff, rates, reference, away_fixed, effects,
                               [StintLineupScenario("projected_path", 1, stints)])
    assert result["summary"]["expected_goals"]["home"] == pytest.approx(2.5, abs=1e-8)


def test_match_team_and_player_identity_and_reference_basis():
    fixture, cutoff, rates, home, away, effects, scenarios = synthetic_inputs()
    with pytest.raises(ValueError, match="identities"):
        forecast_scenarios(fixture, cutoff, replace(rates, fixture_id="other"), home, away, effects, scenarios)
    with pytest.raises(ValueError, match="different teams"):
        adjust_goal_rates(rates, replace(home, team="other"), away, home, away, effects, cutoff)
    with pytest.raises(ValueError, match="calibration"):
        adjust_goal_rates(replace(rates, basis="team_marginal"), home, away, home, away, effects, cutoff)
    changed = dict(effects)
    changed["H_FW_10"] = replace(changed["H_FW_10"], player_id="wrong")
    with pytest.raises(ValueError, match="identity"):
        adjust_goal_rates(rates, scenarios[1].home, away, home, away, changed, cutoff)


def test_dual_tracks_not_overwritten_and_range_is_pre_market():
    args = synthetic_inputs()
    odds = OddsSnapshot(2, 3.5, 4, args[1], "synthetic", args[0].match_id)
    result = forecast_scenarios(*args, market=odds, market_weight=1)
    assert list(result["summary"]["one_x_two"].values()) == pytest.approx(implied(odds))
    assert result["structural_summary"]["one_x_two"] != result["summary"]["one_x_two"]
    assert "pre_market_scenario_range_not_confidence_interval" in result
