"""Synthetic API, likelihood and serialization checks, not prediction skill."""

from copy import deepcopy
from datetime import date, datetime, timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from football_model.joint_stint_model import JointStintIntensityModel
from football_model.starting_xi_model import (
    FullMatchStartingXIModel, UnknownStartingXIIdentityError, _fixture, _penalized_poisson_objective,
)


START = date(2023, 9, 1)
CUTOFF = date(2023, 12, 1)


def players(prefix):
    return [f"impect:player:{prefix}{number}" for number in range(11)]


def row(index=0, *, day=None, home_starters=None, away_starters=None, goals=None):
    return {"match_id": f"impect:match:{index}", "match_date": (day or START + timedelta(days=index)).isoformat(),
            "home_team_id": 1, "away_team_id": 2,
            "home_starters": home_starters or players("h"), "away_starters": away_starters or players("a"),
            "full_time": goals if goals is not None else [index % 3, (index + 1) % 2]}


def history(count=20):
    rows = []
    for index in range(count):
        starters = players("h")
        if index % 2:
            starters[-1] = "impect:player:rotation"
        rows.append(row(index, home_starters=starters))
    return rows


def test_fixed_default_sum_loss_penalties_have_no_artificial_duration_offset():
    model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    assert model.player_ridge == 100
    assert model.home_ridge == model.team_ridge == 1
    assert model.max_iterations == 2000
    penalties = dict(zip(model.feature_names, model.penalties))
    assert penalties["intercept"] == 0
    assert penalties["home_advantage"] == 1
    assert penalties["team_attack:1"] == 1
    assert penalties["player_attack:impect:player:rotation"] == 100
    parameters = model.parameters()
    assert parameters["offset"].startswith("none")
    assert "summed" in parameters["loss"]
    assert "not an optimized" in parameters["penalty_provenance"]
    assert not any("score" in key or "man_" in key for key in model.feature_names)


def test_analytic_gradient_matches_numerical_full_match_sum_loss():
    design = sparse.csr_matrix([[1, 1, .2], [1, 0, -.4], [1, 1, .5], [1, 0, .7]])
    target, coefficients, penalties = np.asarray([2., 0., 1., 3.]), np.asarray([.1, .2, -.3]), np.asarray([0., 1., 100.])
    loss, gradient = _penalized_poisson_objective(coefficients, design, target, penalties)
    eta = design @ coefficients
    assert loss == pytest.approx(np.sum(np.exp(eta) - target * eta) + .5 * np.sum(penalties * coefficients**2))
    numerical = []
    for coordinate in range(3):
        shift = np.zeros(3)
        shift[coordinate] = 1e-6
        numerical.append((_penalized_poisson_objective(coefficients + shift, design, target, penalties)[0]
                          - _penalized_poisson_objective(coefficients - shift, design, target, penalties)[0]) / 2e-6)
    assert gradient == pytest.approx(numerical, rel=1e-7, abs=1e-7)


def test_reference_is_start_frequency_not_minutes_and_only_history_identities():
    rows = history(4)
    rows[0]["actual_minutes"] = {"impect:player:bench-only": 90}
    rows[1]["stints"] = [{"home_players": ["impect:player:never-started"]}]
    rows.append({"match_date": CUTOFF.isoformat(), "home_starters": "invalid target XI"})
    original = deepcopy(rows)
    model = FullMatchStartingXIModel().fit(rows, CUTOFF)
    assert rows == original
    reference = model.reference_frequencies[1]
    assert reference["impect:player:h10"] == .5
    assert reference["impect:player:rotation"] == .5
    assert sum(reference.values()) == 11
    assert model.team_match_counts == {1: 4, 2: 4}
    assert "impect:player:bench-only" not in model.players
    assert "impect:player:never-started" not in model.players
    assert model.reference_match_ids == [f"impect:match:{index}" for index in range(4)]


def test_cutoff_day_future_rows_do_not_change_fit_or_validate_later_fields():
    past = history()
    model = FullMatchStartingXIModel().fit(past, CUTOFF)
    future = [{"match_date": CUTOFF.isoformat(), "home_team_id": True, "full_time": [999, 999]},
              {"match_date": (CUTOFF + timedelta(days=1)).isoformat(), "match_id": "not valid", "full_time": "bad"}]
    with_future = FullMatchStartingXIModel().fit(past + future, CUTOFF)
    assert with_future.parameters() == model.parameters()


def test_prediction_reads_fixture_only_and_explicit_api_has_no_goal_or_path_argument():
    model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    target = row(100, day=CUTOFF)
    expected = model.predict_match(target)
    for key in ("full_time", "stints", "actual_minutes", "bench", "diagnostics", "home_man_difference", "duration_seconds"):
        target[key] = object()  # Deliberately invalid data must never be read.
    assert model.predict_match(target) == expected
    kwargs = {key: target[key] for key in ("match_id", "match_date", "home_team_id", "away_team_id", "home_starters", "away_starters")}
    assert model.predict(**kwargs) == expected
    assert set(expected) == {"match_id", "match_date", "home_mean", "away_mean"}
    with pytest.raises(TypeError, match="full_time"):
        model.predict(**kwargs, full_time=[0, 0])


def test_unknown_starter_skips_player_model_but_not_team_model():
    target = row(100, day=CUTOFF)
    target["home_starters"][-1] = "impect:player:new"
    players_model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    team_model = FullMatchStartingXIModel(include_players=False).fit(history(), CUTOFF)
    with pytest.raises(UnknownStartingXIIdentityError, match="Unknown starting players.*new"):
        players_model.predict_match(target)
    assert team_model.predict_match(target)["home_mean"] > 0
    assert not any(name.startswith("player_") for name in team_model.feature_names)


@pytest.mark.parametrize("include_players", [True, False])
def test_unknown_team_is_not_a_zero_strength_prior(include_players):
    model = FullMatchStartingXIModel(include_players=include_players).fit(history(), CUTOFF)
    target = row(100, day=CUTOFF)
    target["away_team_id"] = 3
    with pytest.raises(UnknownStartingXIIdentityError, match="Unknown teams.*3"):
        model.predict_match(target)


def test_transferred_previously_seen_player_keeps_one_global_identity():
    model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    target = row(100, day=CUTOFF)
    target["home_starters"][-1] = "impect:player:rotation"
    target["away_starters"][-1] = "impect:player:h10"
    assert model.predict_match(target)["away_mean"] > 0


@pytest.mark.parametrize("field,value,message", [
    ("home_team_id", True, "positive integer"), ("away_team_id", 0, "positive integer"),
    ("away_team_id", 1, "distinct"), ("home_team_id", "1", "positive integer"),
    ("home_starters", players("h")[:-1], "exactly 11"),
    ("home_starters", players("h") + ["impect:player:extra"], "exactly 11"),
    ("home_starters", ["impect:player:duplicate"] * 11, "distinct"),
    ("home_starters", ["foreign:player:h"] + players("h")[1:], "namespaced"),
    ("home_starters", ["impect:player:"] + players("h")[1:], "namespaced"),
    ("away_starters", players("h"), "both teams"),
    ("match_id", "", "match_id"), ("match_id", "other:match:1", "namespaced"),
    ("match_date", "2023-09-01T10:00:00Z", "calendar date"),
])
def test_invalid_fixture_identity_or_xi_is_rejected(field, value, message):
    bad = row()
    bad[field] = value
    with pytest.raises(ValueError, match=message):
        FullMatchStartingXIModel().fit([bad], CUTOFF)


@pytest.mark.parametrize("target", [[1.2, 0], [-1, 0], [True, 0], [1], "1-0", None])
def test_invalid_complete_match_goals_are_not_coerced(target):
    bad = row()
    bad["full_time"] = target
    with pytest.raises(ValueError, match="full_time"):
        FullMatchStartingXIModel().fit([bad], CUTOFF)


def test_unfitted_empty_zero_goal_duplicate_and_datetime_errors():
    model = FullMatchStartingXIModel()
    with pytest.raises(ValueError, match="Fit earlier"):
        model.predict_match(row())
    with pytest.raises(ValueError, match="Fit before"):
        model.parameters()
    with pytest.raises(ValueError, match="No strictly earlier"):
        model.fit([], CUTOFF)
    with pytest.raises(ValueError, match="Positive total goals"):
        model.fit([row(goals=[0, 0])], CUTOFF)
    with pytest.raises(ValueError, match="Duplicate"):
        model.fit([row(), row()], CUTOFF)
    with pytest.raises(ValueError, match="calendar date"):
        model.fit(history(), datetime(2023, 12, 1))
    model.fit(history(), CUTOFF)
    with pytest.raises(ValueError, match="fitted cutoff"):
        model.predict_match(row(100, day=CUTOFF - timedelta(days=1)))


@pytest.mark.parametrize("keyword,value", [("include_players", 1), ("player_ridge", 0),
    ("team_ridge", -1), ("home_ridge", True), ("player_ridge", float("nan")),
    ("max_iterations", 0), ("max_iterations", True)])
def test_invalid_hyperparameters_are_not_silently_fixed(keyword, value):
    with pytest.raises(ValueError):
        FullMatchStartingXIModel(**{keyword: value})


@pytest.mark.parametrize("include_players", [True, False])
def test_parameter_file_roundtrip_preserves_mean_and_all_training_references(tmp_path, include_players):
    model = FullMatchStartingXIModel(include_players=include_players).fit(history(), CUTOFF)
    destination = tmp_path / "starting-xi.json"
    model.save(destination)
    restored = FullMatchStartingXIModel.load(destination)
    assert restored.parameters() == model.parameters()
    assert restored.predict_match(row(100, day=CUTOFF)) == model.predict_match(row(100, day=CUTOFF))
    assert json.loads(destination.read_text(encoding="utf-8"))["reference_frequencies"]["1"]["impect:player:rotation"] == .5


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p.update(model="JointStintIntensityModel"), "Not full-match"),
    (lambda p: p.update(training_max_date=p["before_date"]), "cutoff"),
    (lambda p: p["coefficients"].__setitem__(0, float("inf")), "coefficients"),
    (lambda p: p["coefficients"].append(0), "dimensions"),
    (lambda p: p["feature_names"].append("own_score_difference"), "dimensions"),
    (lambda p: p["penalties"].__setitem__(0, 1), "penalties"),
    (lambda p: p["reference_frequencies"]["1"].__setitem__("impect:player:h0", .3), "reference frequencies"),
    (lambda p: p["team_match_counts"].__setitem__("1", 999), "team counts"),
    (lambda p: p.update(n_matches=999), "match count"),
    (lambda p: p["optimizer"].update(success=False), "optimizer"),
])
def test_bad_saved_parameters_are_not_loaded(mutation, message):
    payload = FullMatchStartingXIModel().fit(history(), CUTOFF).parameters()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        FullMatchStartingXIModel.from_parameters(payload)


def test_new_full_match_model_is_separate_from_old_stint_estimator():
    model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    assert not isinstance(model, JointStintIntensityModel)
    with pytest.raises(ValueError, match="Not joint"):
        JointStintIntensityModel.from_parameters(model.parameters())
    assert not hasattr(model, "reference_fractions")
    assert not any("duration" in name or "score_difference" in name or "man_difference" in name
                   for name in model.feature_names)


def test_zero_player_deviations_match_team_only_fit_without_stint_offsets():
    rows = [row(index) for index in range(20)]  # Every player always starts.
    team = FullMatchStartingXIModel(include_players=False).fit(rows, CUTOFF)
    players_model = FullMatchStartingXIModel().fit(rows, CUTOFF)
    assert team.predict_match(row(100, day=CUTOFF)) == pytest.approx(players_model.predict_match(row(100, day=CUTOFF)))
    for name, coefficient in zip(players_model.feature_names, players_model.coefficients):
        if name.startswith("player_"):
            assert coefficient == pytest.approx(0., abs=1e-12)


def test_nonfinite_optimizer_result_is_a_failure_not_a_saved_success(monkeypatch):
    import football_model.starting_xi_model as module
    def failed(function, initial, **kwargs):
        return SimpleNamespace(success=False, fun=0., x=initial, message="synthetic failure")
    monkeypatch.setattr(module, "minimize", failed)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        FullMatchStartingXIModel().fit(history(), CUTOFF)


def test_failed_refit_cannot_pair_old_coefficients_with_new_reference_population():
    model = FullMatchStartingXIModel().fit(history(), CUTOFF)
    with pytest.raises(ValueError, match="Positive total goals"):
        model.fit([row(1, goals=[0, 0])], CUTOFF)
    with pytest.raises(ValueError, match="Fit earlier"):
        model.predict_match(row(100, day=CUTOFF))
    with pytest.raises(ValueError, match="Fit before"):
        model.parameters()


def test_feature_signs_and_reference_frequency_deltas_match_attack_defence_formula():
    model = FullMatchStartingXIModel().fit(history(4), CUTOFF)
    fixture = _fixture(row(100, day=CUTOFF))
    home, away = model._features(fixture)
    assert home["player_attack:impect:player:h10"] == .5
    assert home["player_attack:impect:player:rotation"] == -.5
    assert away["player_defence:impect:player:h10"] == -.5
    assert away["player_defence:impect:player:rotation"] == .5
    assert home["team_defence:2"] == away["team_defence:1"] == -1
