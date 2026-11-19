"""Synthetic same-cohort scaling descriptions; no model fitting required."""

from copy import deepcopy

import pytest

from football_model.scaling_comparison import CONTRAST_PAIRS, TRACKS, compare_scaling


def prediction(mid, loss=2.0, goals=(2, 1), stints=3, day="2016-02-02"):
    return {"match_id": mid, "match_date": day, "n_stints": stints,
            "actual_home_target": goals[0], "actual_away_target": goals[1],
            "expected_home_target": 1.5, "expected_away_target": .5,
            "point_process_loss": loss,
            "loss_kind": "point_process_nll_up_to_model_independent_constants"}


def track(*rows):
    return {"predictions": list(rows), "skipped": [], "requested_matches": len(rows)}


def scores():
    losses = {"early": {"team": (2., 4.), "fixed100": (1.8, 3.8),
                        "fixed_old": (1.9, 3.9), "fixed_new": (1.7, 3.7),
                        "scaled_old": (1.6, 3.6), "scaled_new": (1.5, 3.5)},
              "late": {"team": (3., 5.), "fixed100": (2.8, 4.8),
                       "fixed_old": (2.9, 4.9), "fixed_new": (2.7, 4.7),
                       "scaled_old": (2.5, 4.5), "scaled_new": (2.4, 4.4)}}
    return {clock: {name: track(prediction("a", first), prediction("b", second))
                    for name, (first, second) in tracks.items()}
            for clock, tracks in losses.items()}


def assert_decomposition(row):
    values = row["decomposition"]
    assert values["total"] == pytest.approx(row["scaled_new_minus_fixed_old"])
    assert values["policy_at_new"] == pytest.approx(row["scaled_new_minus_fixed_new"])
    assert values["base_at_fixed"] == pytest.approx(row["fixed_new_minus_fixed_old"])
    assert values["policy_at_old"] == pytest.approx(row["scaled_old_minus_fixed_old"])
    assert values["base_at_scaled"] == pytest.approx(row["scaled_new_minus_scaled_old"])
    assert values["policy_at_new_plus_base_at_fixed"] == pytest.approx(values["total"])
    assert values["policy_at_old_plus_base_at_scaled"] == pytest.approx(values["total"])
    assert values["total_minus_new_path_sum"] == pytest.approx(0, abs=1e-14)
    assert values["total_minus_old_path_sum"] == pytest.approx(0, abs=1e-14)
    assert values["interaction_policy_by_base"] == pytest.approx(values["policy_at_new"] - values["policy_at_old"])
    assert values["interaction_policy_by_base"] == pytest.approx(values["base_at_scaled"] - values["base_at_fixed"])


def test_all_tracks_and_all_clocks_share_one_comparison_cohort():
    result = compare_scaling(scores())
    assert result["common_match_ids"] == ["a", "b"]
    early = result["per_clock"]["early"]
    assert early["mean_match_loss"] == pytest.approx(
        {"team": 3, "fixed100": 2.8, "fixed_old": 2.9, "fixed_new": 2.7,
         "scaled_old": 2.6, "scaled_new": 2.5})
    assert early["scaled_new_minus_fixed_old"] == pytest.approx(-.4)
    assert early["scaled_new_minus_fixed_new"] == pytest.approx(-.2)
    assert early["scaled_old_minus_fixed_old"] == pytest.approx(-.3)
    assert early["fixed_new_minus_fixed_old"] == pytest.approx(-.2)
    assert early["scaled_new_minus_scaled_old"] == pytest.approx(-.1)
    assert early["fixed_old_minus_team"] == pytest.approx(-.1)
    assert early["scaled_new_minus_team"] == pytest.approx(-.5)
    assert early["scaled_new_minus_fixed100"] == pytest.approx(-.3)
    for name in TRACKS:
        assert early[f"{name}_on_common"]["matches"] == 2
        assert result["coverage"]["early"][name]["supported_matches"] == 2
    assert "not independent" in result["averaging_note"]


def test_clock_average_is_equal_clock_mean_of_same_match_losses_and_contrasts():
    result = compare_scaling(scores())
    averaged = result["clock_averaged"]
    assert averaged["matches"] == 2
    assert averaged["clock_count"] == 2
    assert averaged["mean_match_loss"]["team"] == pytest.approx(3.5)
    assert averaged["mean_match_loss"]["scaled_new"] == pytest.approx(2.95)
    assert averaged["scaled_new_minus_fixed_old"] == pytest.approx(-.45)
    assert averaged["scaled_new_minus_fixed_new"] == pytest.approx(-.25)
    assert averaged["scaled_old_minus_fixed_old"] == pytest.approx(-.35)
    assert averaged["scaled_new_minus_fixed100"] == pytest.approx(-.35)
    assert len(averaged["paired_matches"]) == 2
    assert averaged["paired_matches"][0]["scaled_new_loss"] == pytest.approx(1.95)
    assert averaged["paired_matches"][0]["scaled_new_minus_fixed_old"] == pytest.approx(-.45)


def test_arithmetic_decomposition_holds_per_match_per_clock_and_after_averaging():
    result = compare_scaling(scores())
    for clock_result in result["per_clock"].values():
        assert_decomposition(clock_result)
        for match_result in clock_result["paired_matches"]:
            assert_decomposition(match_result)
    assert_decomposition(result["clock_averaged"])
    for row in result["clock_averaged"]["paired_matches"]:
        assert_decomposition(row)
    assert "not causal" in result["decomposition_note"]


def test_global_intersection_not_per_track_or_per_clock_means():
    values = scores()
    values["late"]["scaled_old"]["predictions"] = values["late"]["scaled_old"]["predictions"][:1]
    values["early"]["team"]["predictions"].append(prediction("extra", -1000))
    result = compare_scaling(values)
    assert result["common_match_ids"] == ["a"]
    for clock in result["per_clock"].values():
        for name in TRACKS:
            assert clock[f"{name}_on_common"]["matches"] == 1
        assert len(clock["paired_matches"]) == 1
    assert result["per_clock"]["early"]["mean_match_loss"]["team"] == 2
    assert result["clock_averaged"]["mean_match_loss"]["team"] == 2.5
    assert result["coverage"]["early"]["team"] == {"supported_matches": 3, "outside_common_matches": 2}
    assert result["clock_averaged"]["contrast_distributions"]["scaled_new_minus_team"][
        "sample_standard_deviation"] is None


def test_empty_common_is_none_not_zero_and_has_no_paired_observations():
    values = scores()
    values["late"]["scaled_new"]["predictions"] = []
    result = compare_scaling(values)
    assert result["common_match_ids"] == []
    for row in [*result["per_clock"].values(), result["clock_averaged"]]:
        assert row["paired_matches"] == []
        assert all(row["mean_match_loss"][track] is None for track in TRACKS)
        assert all(row[name] is None for name in CONTRAST_PAIRS)
        assert all(value is None for value in row["decomposition"].values())
        assert all(value is None for value in row["contrast_distributions"].values())
    assert result["clock_averaged"]["matches"] == 0
    assert result["per_clock"]["early"]["team_on_common"] is None
    assert result["coverage"]["early"]["team"]["supported_matches"] == 2


@pytest.mark.parametrize("field,value", [("match_date", "2016-02-03"),
                                         ("actual_home_target", 3),
                                         ("actual_away_target", 2),
                                         ("loss_kind", "xg_quasi")])
def test_identity_disagreement_rejected_even_outside_the_common_intersection(field, value):
    values = scores()
    values["late"]["scaled_new"]["predictions"] = []
    values["early"]["fixed100"]["predictions"][1][field] = value
    with pytest.raises(ValueError, match="same match date, target and loss kind"):
        compare_scaling(values)


def test_partitions_can_differ_across_clocks_but_not_within_one_clock():
    values = scores()
    for value in values["late"].values():
        value["predictions"][0]["n_stints"] = 5
    assert compare_scaling(values)["common_match_ids"] == ["a", "b"]
    values["late"]["fixed_old"]["predictions"][0]["n_stints"] = 6
    with pytest.raises(ValueError, match="same actual state path"):
        compare_scaling(values)


def test_duplicate_match_ids_are_not_silently_overwritten():
    values = scores()
    values["early"]["fixed_new"]["predictions"].append(
        deepcopy(values["early"]["fixed_new"]["predictions"][0]))
    with pytest.raises(ValueError, match="distinct nonempty match IDs"):
        compare_scaling(values)


@pytest.mark.parametrize("bad", [{}, {"": {}}, {4: {}}])
def test_empty_or_invalid_clock_mapping_rejected(bad):
    with pytest.raises(ValueError):
        compare_scaling(bad)


@pytest.mark.parametrize("alter", ["delete", "extra", "nondict"])
def test_every_clock_requires_exactly_the_six_declared_tracks(alter):
    values = scores()
    if alter == "delete":
        del values["late"]["fixed100"]
    elif alter == "extra":
        values["late"]["new_track"] = track()
    else:
        values["late"] = list(TRACKS)
    with pytest.raises(ValueError, match="requires exactly team, fixed100"):
        compare_scaling(values)


def test_distributions_count_matches_once_after_averaging_clocks():
    values = scores()
    for clock_values in values.values():
        for name in TRACKS:
            clock_values[name]["predictions"].append(prediction("c", 1.0))
    values["early"]["scaled_new"]["predictions"][1]["point_process_loss"] = 4.9
    values["late"]["scaled_new"]["predictions"][1]["point_process_loss"] = 5.9
    result = compare_scaling(values)
    description = result["clock_averaged"]["contrast_distributions"]["scaled_new_minus_fixed_old"]
    assert description["matches"] == 3
    assert description["wins"] == description["ties"] == description["losses"] == 1
    assert description["min"] == pytest.approx(-.45)
    assert description["max"] == pytest.approx(1)
    assert description["median"] == 0
    assert description["mean"] == pytest.approx(.55 / 3)
    assert description["sample_standard_deviation"] == pytest.approx(.7421819188484758)


def test_identical_track_aliases_give_exact_zero_without_mutating_inputs():
    values = scores()
    for tracks in values.values():
        tracks["scaled_new"] = tracks["fixed_new"]
    before = deepcopy(values)
    result = compare_scaling(values)
    assert result["clock_averaged"]["scaled_new_minus_fixed_new"] == 0
    assert values == before


def test_single_clock_and_single_match_need_no_fabricated_sample_spread():
    values = {"exact": {name: track(prediction("a", 2.0)) for name in TRACKS}}
    result = compare_scaling(values)
    description = result["clock_averaged"]["contrast_distributions"]["scaled_new_minus_team"]
    assert result["clock_averaged"]["clock_count"] == 1
    assert description["sample_standard_deviation"] is None
    assert description["matches"] == description["ties"] == 1
    assert description["wins"] == description["losses"] == 0
    assert result["clock_averaged"]["decomposition"]["interaction_policy_by_base"] == 0


@pytest.mark.parametrize("field,value,error", [
    ("point_process_loss", float("nan"), "finite numeric"),
    ("point_process_loss", float("inf"), "finite numeric"),
    ("actual_home_target", True, "finite numeric"),
    ("expected_away_target", -1, "must not be negative"),
    ("n_stints", False, "positive integer"),
    ("match_date", "not_a_date", "ISO calendar"),
    ("loss_kind", "", "nonempty string"),
])
def test_prediction_checks_reused_without_bypassing_invalid_values(field, value, error):
    values = scores()
    values["early"]["scaled_new"]["predictions"][0][field] = value
    with pytest.raises(ValueError, match=error):
        compare_scaling(values)


def test_disjoint_match_losses_cannot_mix_definitions():
    values = scores()
    for tracks in values.values():
        for value in tracks.values():
            value["predictions"][1]["loss_kind"] = "xg_quasi"
    with pytest.raises(ValueError, match="one loss definition"):
        compare_scaling(values)
