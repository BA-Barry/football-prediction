"""Synthetic implementation checks, not football prediction performance claims."""

from copy import deepcopy
from datetime import date, datetime, timedelta
import json

import numpy as np
import pytest

from football_model.joint_stint_model import JointStintIntensityModel, UnknownStintIdentityError


START = date(2025, 1, 1)
CUTOFF = START + timedelta(days=200)


def match(index=0, *, home_target=1.0, away_target=1.0, duration=2700.0,
          home_players=None, away_players=None):
    return {"match_id": f"match-{index}", "match_date": (START + timedelta(days=index)).isoformat(),
            "home_team_id": 1, "away_team_id": 2,
            "stints": [{"stint_id": f"match-{index}:stint:1", "period": 1,
                        "start_second": 0.0, "end_second": duration, "duration_seconds": duration,
                        "home_players": home_players or ["h1", "h-fixed"],
                        "away_players": away_players or ["a1", "a-fixed"],
                        "home_score_difference": 0, "home_man_difference": 0,
                        "home_goals": home_target, "away_goals": away_target,
                        "home_xg": home_target, "away_xg": away_target}]}


def rotation(n=100):
    result = []
    for index in range(n):
        strong = index % 2 == 0
        duration = 1800.0 if index % 3 == 0 else 3600.0
        result.append(match(index, home_players=["strong" if strong else "weak", "h-fixed"],
                            home_target=duration / 5400 * 1.5 * np.exp(.3 if strong else -.3),
                            away_target=duration / 5400 * 1.1, duration=duration))
    return result


def test_joint_xg_recovers_player_direction_instead_of_fitting_a_marginal_offset():
    model = JointStintIntensityModel(response_kind="xg", player_ridge=.01,
                                     team_ridge=.1, context_ridge=.1).fit(rotation(), CUTOFF)
    coefficients = dict(zip(model.feature_names, model.coefficients))
    assert coefficients["player_attack:strong"] - coefficients["player_attack:weak"] == pytest.approx(.6, abs=.02)
    assert abs(coefficients["player_defence:strong"] - coefficients["player_defence:weak"]) < .02
    assert model.n_matches == model.n_stints == 100
    assert model.parameters()["fixed_offset"].startswith("log(actual duration_seconds/5400)")
    assert "not a goal distribution" in model.parameters()["limits"]


def test_reference_fractions_use_only_earlier_exposure_and_future_rows_are_not_read():
    short = match(0, duration=1800, home_players=["short"])
    long = match(1, duration=3600, home_players=["long"])
    future = {"match_id": "future", "match_date": CUTOFF.isoformat(), "stints": "deliberately not parseable"}
    model = JointStintIntensityModel().fit([short, long, future], CUTOFF)
    assert model.reference_fractions[1] == pytest.approx({"short": 1 / 3, "long": 2 / 3})
    assert model.reference_match_ids == ["match-0", "match-1"]
    assert model.training_max_date == START + timedelta(days=1)
    assert model.players == {"short", "long", "a1", "a-fixed"}


def test_actual_duration_is_used_once_and_point_process_loss_has_no_log_duration_term():
    model = JointStintIntensityModel(include_players=False).fit([match(i) for i in range(10)], CUTOFF)
    full = match(200, home_target=1, away_target=0, duration=5400)
    half = match(201, home_target=1, away_target=0, duration=2700)
    p, q = model.predict_match(full), model.predict_match(half)
    assert p["expected_home_target"] == pytest.approx(2 * q["expected_home_target"])
    row = q["stints"][0]
    expected_loss = row["home_expected_target"] + row["away_expected_target"] - np.log(row["home_intensity_per90"])
    assert q["point_process_loss"] == pytest.approx(expected_loss)
    assert q["point_process_loss"] != pytest.approx(expected_loss - np.log(.5))
    assert "not pre-match" in q["usage"]


def test_unknown_identity_is_a_clear_whole_match_rejection_not_zero_coding():
    model = JointStintIntensityModel().fit([match(0), match(1)], CUTOFF)
    unknown_team = match(200)
    unknown_team["away_team_id"] = 3
    with pytest.raises(UnknownStintIdentityError, match="Unknown teams"):
        model.predict_match(unknown_team)
    with pytest.raises(UnknownStintIdentityError, match="Unknown players"):
        model.predict_match(match(200, home_players=["new-player"]))
    # A team-only model has no player coefficients to invent. The comparison
    # runner must still intersect its coverage with the player model's cohort.
    team = JointStintIntensityModel(include_players=False).fit([match(0)], CUTOFF)
    assert np.isfinite(team.predict_match(match(200, home_players=["new-player"]))["point_process_loss"])


def test_stable_global_player_identity_supports_transfer_without_a_new_effect():
    history = [match(0, home_players=["transfer"], away_players=["a1"]), match(1)]
    model = JointStintIntensityModel().fit(history, CUTOFF)
    future = match(200, home_players=["h1"], away_players=["transfer"])
    assert np.isfinite(model.predict_match(future)["point_process_loss"])


def test_json_parameter_roundtrip_preserves_reference_and_predictions(tmp_path):
    model = JointStintIntensityModel(response_kind="xg").fit(rotation(30), CUTOFF)
    target = match(200, home_players=["strong", "h-fixed"], home_target=.7)
    payload = json.loads(json.dumps(model.parameters()))
    restored = JointStintIntensityModel.from_parameters(payload)
    assert restored.predict_match(target) == model.predict_match(target)
    destination = tmp_path / "joint.json"
    model.save(destination)
    assert JointStintIntensityModel.load(destination).parameters() == model.parameters()
    payload["coefficients"][0] = float("inf")
    with pytest.raises(ValueError, match="Invalid saved"):
        JointStintIntensityModel.from_parameters(payload)


def test_duplicated_matches_and_overlapping_stints_are_rejected():
    with pytest.raises(ValueError, match="Duplicate match"):
        JointStintIntensityModel().fit([match(0), match(0)], CUTOFF)
    bad = match(0)
    bad["stints"].append(deepcopy(bad["stints"][0]))
    bad["stints"][1]["stint_id"] = "overlap"
    with pytest.raises(ValueError, match="overlap"):
        JointStintIntensityModel().fit([bad], CUTOFF)


@pytest.mark.parametrize("field,value,message", [
    ("duration_seconds", 0, "positive"), ("end_second", 2000, "duration"),
    ("home_goals", .5, "Fractional"), ("home_goals", -1, "nonnegative"),
    ("home_score_difference", float("nan"), "finite"),
    ("home_players", ["h1", "h1"], "distinct"),
    ("away_players", ["h1"], "both teams"),
])
def test_invalid_stint_fields_are_not_silently_repaired(field, value, message):
    bad = match(0)
    bad["stints"][0][field] = value
    with pytest.raises(ValueError, match=message):
        JointStintIntensityModel().fit([bad], CUTOFF)


def test_prediction_cannot_use_a_training_date_and_cutoff_cannot_be_fake_datetime():
    model = JointStintIntensityModel().fit([match(0)], CUTOFF)
    with pytest.raises(ValueError, match="at or after"):
        model.predict_match(match(1))
    with pytest.raises(ValueError, match="calendar date"):
        JointStintIntensityModel().fit([match(0)], datetime(2025, 2, 1))
    with pytest.raises(ValueError, match="No earlier"):
        JointStintIntensityModel().fit([match(200)], CUTOFF)


def test_penalty_blocks_are_explicit_summed_likelihood_and_intercept_unpenalized():
    model = JointStintIntensityModel(team_ridge=2, player_ridge=10, context_ridge=3).fit([match(0)], CUTOFF)
    penalties = dict(zip(model.feature_names, model.penalties))
    assert penalties["intercept"] == 0
    assert penalties["team_attack:1"] == penalties["team_defence:2"] == 2
    assert penalties["player_attack:h1"] == penalties["player_defence:a1"] == 10
    assert penalties["home_advantage"] == penalties["own_score_difference"] == penalties["own_man_difference"] == 3
    assert model.optimizer_diagnostics["success"]


def test_all_zero_training_response_has_no_finite_unpenalized_intercept():
    with pytest.raises(ValueError, match="finite unpenalized intercept"):
        JointStintIntensityModel().fit([match(0, home_target=0, away_target=0)], CUTOFF)


@pytest.mark.parametrize("change", ["matrix_coefficients", "unknown_feature", "boolean_team"])
def test_saved_model_shapes_and_identity_types_are_checked(change):
    model = JointStintIntensityModel().fit([match(0)], CUTOFF)
    payload = model.parameters()
    if change == "matrix_coefficients":
        payload["coefficients"] = [[value] for value in payload["coefficients"]]
    elif change == "unknown_feature":
        payload["feature_names"][0] = "unmodelled-feature"
    else:
        payload["teams"][0] = True
    with pytest.raises(ValueError, match="Invalid saved"):
        JointStintIntensityModel.from_parameters(payload)
