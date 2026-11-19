"""Synthetic comparison tests, without model fitting or raw-data access."""

from copy import deepcopy

import pytest

from football_model.regularization_comparison import compare_regularization, select_regularization


def prediction(mid, loss=2.0, goals=(2, 1), stints=3, day="2016-02-02"):
    return {"match_id": mid, "match_date": day, "n_stints": stints,
            "actual_home_target": goals[0], "actual_away_target": goals[1],
            "expected_home_target": 1.5, "expected_away_target": .5,
            "point_process_loss": loss,
            "loss_kind": "point_process_nll_up_to_model_independent_constants"}


def track(*rows):
    return {"predictions": list(rows), "skipped": [], "requested_matches": len(rows)}


def validation():
    return {"early": {"10": track(prediction("a", 1), prediction("b", 3)),
                      "100": track(prediction("a", 2), prediction("b", 2))},
            "late": {10.0: track(prediction("a", 4), prediction("b", 2)),
                     100: track(prediction("a", 1), prediction("b", 3))}}


def comparison():
    return {"early": {"team": track(prediction("a", 2), prediction("b", 3)),
                      "fixed100": track(prediction("a", 1.8), prediction("b", 2.7)),
                      "selected": track(prediction("a", 1.5), prediction("b", 2.5))},
            "late": {"team": track(prediction("a", 2.2), prediction("b", 3.2)),
                     "fixed100": track(prediction("a", 2), prediction("b", 3)),
                     "selected": track(prediction("a", 1.8), prediction("b", 2.8))}}


def test_select_normalizes_keys_and_averages_same_matches_then_equal_clocks():
    result = select_regularization(validation())
    assert result["selected_player_ridge"] == 100.0
    assert result["common_match_ids"] == ["a", "b"]
    assert result["candidate_metrics"]["10"]["mean_clock_match_loss"] == pytest.approx(2.5)
    assert result["candidate_metrics"]["100"]["mean_clock_match_loss"] == pytest.approx(2)
    assert result["candidate_metrics"]["10"]["per_clock"]["early"]["matches"] == 2
    assert "not independent" in result["averaging_note"]


def test_selection_uses_intersection_not_each_track_supported_mean():
    values = validation()
    values["late"][100]["predictions"] = values["late"][100]["predictions"][:1]
    values["early"]["10"]["predictions"].append(prediction("not_shared", -1000))
    result = select_regularization(values)
    assert result["common_match_ids"] == ["a"]
    assert result["candidate_metrics"]["10"]["mean_clock_match_loss"] == pytest.approx(2.5)
    assert result["candidate_metrics"]["100"]["mean_clock_match_loss"] == pytest.approx(1.5)


def test_selection_ties_within_tolerance_choose_larger_ridge_not_clock_winner():
    values = {"first": {10: track(prediction("a", 1)), 100: track(prediction("a", 3))},
              "second": {10: track(prediction("a", 3)), 100: track(prediction("a", 1 + 1e-12))}}
    assert select_regularization(values)["selected_player_ridge"] == 100
    values["second"][100]["predictions"][0]["point_process_loss"] = 1 + 1e-10
    assert select_regularization(values)["selected_player_ridge"] == 10


@pytest.mark.parametrize("bad", [True, False, 0, -1, float("nan"), float("inf"), "not_numeric", "", None])
def test_invalid_candidate_keys_rejected(bad):
    with pytest.raises(ValueError, match="positive finite"):
        select_regularization({"exact": {bad: track(prediction("a"))}})


def test_candidate_sets_must_match_and_duplicate_numeric_labels_rejected():
    values = validation()
    del values["early"]["10"]
    with pytest.raises(ValueError, match="same numeric ridge candidate set"):
        select_regularization(values)
    with pytest.raises(ValueError, match="distinct after numeric normalization"):
        select_regularization({"exact": {"10": track(prediction("a")), 10: track(prediction("a"))}})


@pytest.mark.parametrize("values", [{}, {"exact": {}}, {"": {10: track()}}, {4: {10: track()}}])
def test_empty_or_invalid_structure_rejected(values):
    with pytest.raises(ValueError):
        select_regularization(values)


def test_no_shared_validation_is_not_a_zero_loss_candidate():
    values = validation()
    values["late"][100]["predictions"] = []
    with pytest.raises(ValueError, match="nonempty common validation cohort"):
        select_regularization(values)


@pytest.mark.parametrize("field,value", [("match_date", "2016-02-03"),
                                         ("actual_home_target", 3),
                                         ("actual_away_target", 2),
                                         ("loss_kind", "xg_quasi")])
def test_selection_identity_checked_even_outside_common_intersection(field, value):
    values = validation()
    values["late"][100]["predictions"] = values["late"][100]["predictions"][:1]
    values["early"]["10"]["predictions"][1][field] = value
    with pytest.raises(ValueError, match="same match date, target and loss kind"):
        select_regularization(values)


def test_partition_can_differ_between_clocks_not_within_clock_candidates():
    values = validation()
    for scores in values["late"].values():
        scores["predictions"][0]["n_stints"] = 5
    assert select_regularization(values)["selected_player_ridge"] == 100
    values["late"][100]["predictions"][0]["n_stints"] = 6
    with pytest.raises(ValueError, match="same actual state path"):
        select_regularization(values)


def test_all_clock_comparison_means_and_paired_deltas():
    result = compare_regularization(comparison())
    assert result["common_match_ids"] == ["a", "b"]
    assert result["per_clock"]["early"]["mean_match_loss"] == pytest.approx(
        {"team": 2.5, "fixed100": 2.25, "selected": 2})
    assert result["per_clock"]["early"]["selected_minus_fixed100"] == pytest.approx(-.25)
    assert result["per_clock"]["early"]["fixed100_minus_team"] == pytest.approx(-.25)
    assert result["per_clock"]["early"]["selected_minus_team"] == pytest.approx(-.5)
    averaged = result["clock_averaged"]
    assert averaged["matches"] == 2
    assert averaged["clock_count"] == 2
    assert averaged["mean_match_loss"]["selected"] == pytest.approx(2.15)
    assert averaged["selected_minus_fixed100"] == pytest.approx(-.225)
    assert averaged["fixed100_minus_team"] == pytest.approx(-.225)
    assert averaged["selected_minus_team"] == pytest.approx(-.45)


def test_comparison_intersection_across_all_models_and_clocks():
    values = comparison()
    values["late"]["selected"]["predictions"] = values["late"]["selected"]["predictions"][:1]
    values["early"]["team"]["predictions"].append(prediction("team_only", -1000))
    result = compare_regularization(values)
    assert result["common_match_ids"] == ["a"]
    assert result["per_clock"]["early"]["team_on_common"]["matches"] == 1
    assert result["per_clock"]["early"]["mean_match_loss"]["team"] == 2
    assert result["per_clock"]["early"]["selected_minus_fixed100"] == pytest.approx(-.3)
    assert result["clock_averaged"]["selected_minus_fixed100"] == pytest.approx(-.25)


def test_empty_comparison_is_explicit_none_not_success_or_zero():
    values = comparison()
    values["late"]["selected"]["predictions"] = []
    result = compare_regularization(values)
    assert result["common_match_ids"] == []
    for row in result["per_clock"].values():
        assert row["team_on_common"] is None
        assert row["selected_on_common"] is None
        assert row["mean_match_loss"] == {"team": None, "fixed100": None, "selected": None}
        assert row["selected_minus_fixed100"] is None
        assert row["paired_matches"] == []
    assert result["clock_averaged"]["selected_minus_team"] is None
    assert result["clock_averaged"]["matches"] == 0


def test_comparison_same_setting_can_be_selected_without_fabricated_difference():
    values = comparison()
    for tracks in values.values():
        tracks["selected"] = deepcopy(tracks["fixed100"])
    result = compare_regularization(values)
    assert result["clock_averaged"]["selected_minus_fixed100"] == 0
    assert result["clock_averaged"]["selected_minus_team"] == pytest.approx(-.225)


@pytest.mark.parametrize("field,value", [("match_date", "2016-02-03"),
                                         ("actual_home_target", 3),
                                         ("loss_kind", "xg_quasi")])
def test_comparison_cross_clock_identity_checked_before_empty_support(field, value):
    values = comparison()
    values["late"]["selected"]["predictions"] = []
    values["early"]["fixed100"]["predictions"][1][field] = value
    with pytest.raises(ValueError, match="same match date, target and loss kind"):
        compare_regularization(values)


def test_comparison_path_may_differ_between_clocks_not_models_of_one_clock():
    values = comparison()
    for scores in values["late"].values():
        scores["predictions"][0]["n_stints"] = 5
    compare_regularization(values)
    values["late"]["selected"]["predictions"][0]["n_stints"] = 6
    with pytest.raises(ValueError, match="same actual state path"):
        compare_regularization(values)


def test_comparison_requires_exactly_declared_track_set():
    values = comparison()
    del values["late"]["fixed100"]
    with pytest.raises(ValueError, match="team, fixed100 and selected"):
        compare_regularization(values)


@pytest.mark.parametrize("function,values", [(select_regularization, validation),
                                            (compare_regularization, comparison)])
def test_duplicate_ids_are_not_silently_overwritten(function, values):
    scores = values()
    first_track = next(iter(scores["early"].values()))
    first_track["predictions"].append(deepcopy(first_track["predictions"][0]))
    with pytest.raises(ValueError, match="distinct nonempty match IDs"):
        function(scores)


@pytest.mark.parametrize("field,value,error", [
    ("point_process_loss", float("nan"), "finite numeric"),
    ("point_process_loss", float("inf"), "finite numeric"),
    ("actual_home_target", True, "finite numeric"),
    ("expected_away_target", -1, "must not be negative"),
    ("actual_home_target", -1, "must not be negative"),
    ("n_stints", False, "positive integer"),
    ("n_stints", 1.5, "positive integer"),
    ("match_date", "not_a_date", "ISO calendar"),
    ("loss_kind", "", "nonempty string"),
])
def test_invalid_prediction_values_rejected(field, value, error):
    values = validation()
    values["early"]["10"]["predictions"][0][field] = value
    with pytest.raises(ValueError, match=error):
        select_regularization(values)


def test_aggregation_does_not_mix_loss_kinds_across_disjoint_matches():
    values = {"exact": {10: track(prediction("a"), prediction("b"))}}
    values["exact"][10]["predictions"][1]["loss_kind"] = "xg_quasi"
    with pytest.raises(ValueError, match="one loss definition"):
        select_regularization(values)


@pytest.mark.parametrize("function,values", [(select_regularization, validation),
                                            (compare_regularization, comparison)])
def test_inputs_not_mutated(function, values):
    scores = values()
    original = deepcopy(scores)
    function(scores)
    assert scores == original
