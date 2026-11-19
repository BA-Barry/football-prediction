"""Synthetic same-target and cross-window pairing tests; no model fitting."""

from copy import deepcopy

import pytest

from football_model.window_clock_comparison import CLOCKS, compare_tracks


def prediction(mid, loss=2.0, goals=(2, 1), mean=2.5, stints=3, day="2016-02-02"):
    return {"match_id": mid, "match_date": day,
            "actual_home_target": goals[0], "actual_away_target": goals[1],
            "expected_home_target": mean / 2, "expected_away_target": mean / 2,
            "n_stints": stints, "point_process_loss": loss,
            "loss_kind": "point_process_nll_up_to_model_independent_constants"}


def track(*rows):
    return {"predictions": list(rows), "skipped": [], "requested_matches": len(rows)}


def scores(clocks=CLOCKS):
    return {window: {clock: {
        "team": track(prediction("a", goals=goals), prediction("b", goals=goals)),
        "players": track(prediction("a", 1.9 if window == "full" else 2.1, goals, 2.4 + .1 * i),
                         prediction("b", 1.8 if window == "full" else 2.2, goals))}
        for i, clock in enumerate(clocks)}
        for window, goals in (("full", (2, 1)), ("fixed45", (1, 1)))}


def test_same_cohort_intersection_not_different_supported_averages():
    values = scores()
    values["fixed45"]["ceil"]["players"]["predictions"] = values["fixed45"]["ceil"]["players"]["predictions"][:1]
    result = compare_tracks(values)
    assert result["windows"]["full"]["all_clocks_common"]["common_match_ids"] == ["a", "b"]
    assert result["windows"]["fixed45"]["all_clocks_common"]["common_match_ids"] == ["a"]
    assert result["all_windows_clocks_common"]["common_match_ids"] == ["a"]
    for row in result["all_windows_clocks_common"]["by_clock"].values():
        assert row["full_player_minus_team_loss"] == pytest.approx(-.1)
        assert row["fixed_player_minus_team_loss"] == pytest.approx(.1)
        assert row["fixed_minus_full_increment_difference"] == pytest.approx(.2)


def test_empty_support_is_null_not_zero():
    values = scores()
    values["fixed45"]["ceil"]["players"]["predictions"] = []
    result = compare_tracks(values)
    all_common = result["all_windows_clocks_common"]
    assert all_common["common_match_ids"] == []
    for row in all_common["by_clock"].values():
        assert row["full_player_minus_team_loss"] is None
        assert row["fixed_player_minus_team_loss"] is None
        assert row["fixed_minus_full_increment_difference"] is None
        assert row["paired_target_specific_increment_differences"] == []
    spread = result["windows"]["fixed45"]["all_clocks_common"]["player_integrated_intensity_spread"]
    assert spread["rows"] == []
    assert spread["mean_range_width"] is None


def test_clock_partition_may_differ_but_each_model_pair_must_share_path():
    values = scores()
    for kind in ("team", "players"):
        values["fixed45"]["ceil"][kind]["predictions"][0]["n_stints"] = 5
    assert compare_tracks(values)["all_windows_clocks_common"]["common_match_ids"] == ["a", "b"]
    values["fixed45"]["ceil"]["team"]["predictions"][0]["n_stints"] = 6
    with pytest.raises(ValueError, match="same response and actual state path"):
        compare_tracks(values)


def test_goal_target_mismatch_across_clocks_rejected_even_outside_global_intersection():
    values = scores()
    values["fixed45"]["ceil"]["players"]["predictions"] = []
    for kind in ("team", "players"):
        values["full"]["floor"][kind]["predictions"][0]["actual_home_target"] = 3
    with pytest.raises(ValueError, match="same match date, goal target"):
        compare_tracks(values)


def test_cross_window_different_targets_valid_and_described_not_raw_loss_subtraction():
    result = compare_tracks(scores(("exact",)))
    assert "different full/fixed goal targets" in result["all_windows_clocks_common"]["interpretation"]
    row = result["all_windows_clocks_common"]["by_clock"]["exact"]
    assert row["full"]["team_on_common"]["observed_total_target"] == 6
    assert row["fixed45"]["team_on_common"]["observed_total_target"] == 4
    assert row["fixed_minus_full_increment_difference"] == pytest.approx(.3)
    assert "raw_loss_difference" not in row


@pytest.mark.parametrize("field,value,error", [
    ("actual_home_target", 3, "cannot exceed"),
    ("actual_away_target", 2, "cannot exceed"),
    ("match_date", "2016-02-03", "same match date"),
    ("loss_kind", "xg_quasi", "compatible loss definitions"),
])
def test_cross_window_identity_and_target_bound(field, value, error):
    values = scores(("exact",))
    for kind in ("team", "players"):
        values["fixed45"]["exact"][kind]["predictions"][0][field] = value
    with pytest.raises(ValueError, match=error):
        compare_tracks(values)


def test_integrated_intensity_spread_stays_within_target_and_all_clock_cohort():
    result = compare_tracks(scores())
    spread = result["windows"]["fixed45"]["all_clocks_common"]["player_integrated_intensity_spread"]
    assert spread["matches"] == 2
    assert spread["mean_range_width"] == pytest.approx(.15)
    assert spread["max_range_width"] == pytest.approx(.3)
    assert "not 1X2 error" in spread["interpretation"]


def test_duplicate_prediction_ids_reject_instead_of_dictionary_overwrite():
    values = scores(("exact",))
    values["full"]["exact"]["team"]["predictions"].append(deepcopy(values["full"]["exact"]["team"]["predictions"][0]))
    with pytest.raises(ValueError, match="distinct nonempty match IDs"):
        compare_tracks(values)


def test_inputs_not_modified():
    values = scores()
    original = deepcopy(values)
    compare_tracks(values)
    assert values == original


def test_missing_clock_rejected_not_silently_removed_from_comparison():
    values = scores()
    del values["fixed45"]["ceil"]
    with pytest.raises(ValueError, match="same clock set"):
        compare_tracks(values)


def test_nonfinite_losses_rejected():
    values = scores(("exact",))
    values["full"]["exact"]["team"]["predictions"][0]["point_process_loss"] = float("nan")
    with pytest.raises(ValueError, match="finite numeric"):
        compare_tracks(values)
