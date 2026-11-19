"""Same-fixed-target comparison and bookkeeping, independent of real scores."""

from copy import deepcopy

import pytest

from football_model.window_clock_experiment import (
    append_score, empty_score, fixed_target_training_comparison, summarize_paths,
)


def prediction(mid, loss=2.0):
    return {"match_id": mid, "match_date": "2016-02-02", "actual_home_target": 1,
        "actual_away_target": 0, "expected_home_target": 1.2, "expected_away_target": .8,
        "n_stints": 2, "point_process_loss": loss,
        "loss_kind": "point_process_nll_up_to_model_independent_constants"}


def tracks(player_loss):
    return {c: {"team": {"predictions": [prediction("a"), prediction("b")]},
                "players": {"predictions": [prediction("a", player_loss), prediction("b", player_loss)]}}
            for c in ("exact", "floor")}


def test_cross_scoring_is_same_fixed_target_not_different_raw_targets():
    result = fixed_target_training_comparison(tracks(1.9), tracks(2.1))
    assert result["common_match_ids"] == ["a", "b"]
    for row in result["by_clock"].values():
        assert row["full_training"]["mean_player_minus_team_loss"] == pytest.approx(-.1)
        assert row["fixed_training"]["mean_player_minus_team_loss"] == pytest.approx(.1)
        assert row["fixed_training_minus_full_training_player_increment"] == pytest.approx(.2)
        assert row["per_kind_same_fixed_target"]["players"]["mean_fixed_training_minus_full_training_loss"] == pytest.approx(.2)


def test_fixed_target_comparison_uses_every_clock_kind_and_training_support():
    first, second = tracks(1.9), tracks(2.1)
    second["floor"]["players"]["predictions"] = [prediction("a", 2.1)]
    result = fixed_target_training_comparison(first, second)
    assert result["common_match_ids"] == ["a"]
    assert all(row["full_training"]["common_match_ids"] == ["a"] for row in result["by_clock"].values())


def test_fixed_target_empty_unknowns_do_not_create_zero_loss():
    first, second = tracks(1.9), tracks(2.1)
    first["exact"]["players"]["predictions"] = []
    result = fixed_target_training_comparison(first, second)
    assert result["common_match_ids"] == []
    assert result["by_clock"]["exact"]["fixed_training_minus_full_training_player_increment"] is None


def test_fixed_target_training_comparison_rejects_different_targets():
    first, second = tracks(1.9), tracks(2.1)
    for kind in ("team", "players"):
        second["exact"][kind]["predictions"][0]["actual_home_target"] = 0
    with pytest.raises(ValueError, match="same response"):
        fixed_target_training_comparison(first, second)


def test_fixed_target_training_comparison_rejects_different_paths():
    first, second = tracks(1.9), tracks(2.1)
    for kind in ("team", "players"):
        second["floor"][kind]["predictions"][0]["n_stints"] = 3
    with pytest.raises(ValueError, match="same response"):
        fixed_target_training_comparison(first, second)


def test_append_score_keeps_unknown_skips_and_requested_denominator():
    total = empty_score()
    score = {"predictions": [prediction("a")], "skipped": [{"match_id": "b", "reason": "unknown"}],
             "requested_matches": 2}
    original = deepcopy(score)
    append_score(total, score)
    append_score(total, {"predictions": [], "skipped": [{"match_id": "c"}], "requested_matches": 1})
    assert score == original
    assert total["requested_matches"] == 3
    assert len(total["predictions"]) == 1
    assert len(total["skipped"]) == 2


def test_summary_separates_exposure_and_targets_without_dropping_conflicts():
    match = {"stints": [{"duration_seconds": 5400, "home_players": ["p1"], "away_players": ["p2"]}],
        "diagnostics": {"event_goals": [1, 2], "actor_conflicts": [{"is_goal": False}, {"is_goal": True}]}}
    result = summarize_paths({"fixed45": {"exact": [match]}})["fixed45"]["exact"]
    assert result["goals"] == 3
    assert result["exposure_seconds"] == 5400
    assert result["actor_conflicts"] == 2
    assert result["goal_actor_conflicts"] == 1
    assert result["positive_exposure_players"] == 2
