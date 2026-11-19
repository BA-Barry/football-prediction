"""Paired-cohort tests, independent of real model outcomes."""

from copy import deepcopy

import pytest

from football_model.second_season_player_experiment import SCENARIOS, common_scenario_comparison, quality_summary


def prediction(mid, loss=2.0, mean=2.5, stints=2):
    return {"match_id": mid, "match_date": "2018-02-03", "actual_home_target": 1,
        "actual_away_target": 1, "expected_home_target": mean / 2, "expected_away_target": mean / 2,
        "n_stints": stints, "point_process_loss": loss,
        "loss_kind": "point_process_nll_up_to_model_independent_constants"}


def scores():
    return {s: {"team": {"predictions": [prediction("a"), prediction("b")]},
                "players": {"predictions": [prediction("a", 1.9, 2.4 + .1 * i), prediction("b", 1.8)]}}
            for i, s in enumerate(SCENARIOS)}


def test_cross_clock_comparison_uses_intersection_not_different_supported_cohorts():
    values = scores()
    values["early"]["players"]["predictions"] = values["early"]["players"]["predictions"][:1]
    result = common_scenario_comparison(values)
    assert result["common_match_ids"] == ["a"]
    assert result["mean_conditional_intensity_range"] == pytest.approx(.2)
    assert all(v["mean_player_minus_team_loss"] == pytest.approx(-.1) for v in result["scenarios"].values())


def test_clock_spread_is_not_required_to_have_equal_interval_partition():
    values = scores()
    for kind in ("team", "players"):
        values["late"][kind]["predictions"][0]["n_stints"] = 3
    assert common_scenario_comparison(values)["common_match_ids"] == ["a", "b"]


def test_empty_support_stays_unknown_not_fabricated_zero_loss():
    values = scores()
    values["middle"]["players"]["predictions"] = []
    result = common_scenario_comparison(values)
    assert result["common_match_ids"] == []
    assert result["scenarios"] == {}


def test_clock_comparison_rejects_changed_goal_target():
    values = deepcopy(scores())
    # Within-clock pairing stays valid, but one clock has lost a goal.
    for kind in ("team", "players"):
        values["late"][kind]["predictions"][0]["actual_home_target"] = 0
    with pytest.raises(ValueError, match="same fixed-window"):
        common_scenario_comparison(values)


def test_quality_summary_separates_target_exclusions_from_lineup_conflicts():
    match = {"stints": [{"home_goals": 1, "away_goals": 0, "duration_seconds": 5400}],
        "diagnostics": {"full_source_goals": [2, 0], "actor_conflicts": [
            {"in_target_window": True, "is_goal": True},
            {"in_target_window": False, "is_goal": True},
            {"in_target_window": True, "is_goal": False}]}}
    paths = {s: [deepcopy(match)] for s in SCENARIOS}
    diagnostics = [
        {"common_supported": True, "diagnostics": {"full_source_goals": [2, 0], "window_goals": [1, 0]}},
        {"common_supported": False, "diagnostics": {"full_source_goals": [0, 1], "window_goals": [0, 1]}}]
    result = quality_summary(paths, diagnostics)
    assert result["source_primary_goals"] == 3
    assert result["source_window_goals"] == 2
    assert result["source_stoppage_goals_outside_target"] == 1
    assert result["common_parse_matches"] == 1
    assert len(result["excluded_matches"]) == 1
    for s in SCENARIOS:
        assert result["scenario_summary"][s]["window_goals"] == 1
        assert result["scenario_summary"][s]["retained_full_source_goals"] == 2
        assert result["scenario_summary"][s]["actor_conflicts_in_target_window"] == 2
        assert result["scenario_summary"][s]["goal_actor_conflicts_in_target_window"] == 1
        assert result["scenario_summary"][s]["matches_with_target_actor_conflict"] == 1
