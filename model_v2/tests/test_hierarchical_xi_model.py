"""Tiny synthetic fixed-prior/Laplace checks, not real-season training."""

from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pytest
from scipy import linalg

from football_model.hierarchical_xi_model import (
    HierarchicalStartingXIModel, ROLES, UnknownHierarchicalTeamError, _role_fixture,
)
from football_model.starting_xi_model import FullMatchStartingXIModel, _penalized_poisson_objective


START = date(2023, 9, 1)
CUTOFF = date(2023, 12, 1)


def players(prefix):
    return [f"impect:player:{prefix}{number}" for number in range(11)]


def fixture(index=0, day=None):
    home, away = players("h"), players("a")
    roles = ["GK"] + ["DEF"] * 4 + ["MID"] * 3 + ["FWD"] * 3
    return {"match_id": f"impect:match:{index}", "match_date": (day or START + timedelta(days=index)).isoformat(),
            "home_team_id": 1, "away_team_id": 2, "home_starters": home, "away_starters": away,
            "home_roles": dict(zip(home, roles)), "away_roles": dict(zip(away, roles)),
            "full_time": [index % 3, (index + 1) % 2]}


def history(n=12):
    result = []
    for index in range(n):
        row = fixture(index)
        if index % 2:
            previous = row["home_starters"][-1]
            row["home_starters"][-1] = "impect:player:rotation"
            row["home_roles"]["impect:player:rotation"] = row["home_roles"].pop(previous)
            row["home_roles"]["impect:player:h4"] = "MID"
        result.append(row)
    return result


def replace_player(row, side, old, new):
    row[f"{side}_starters"] = [new if player == old else player for player in row[f"{side}_starters"]]
    row[f"{side}_roles"][new] = row[f"{side}_roles"].pop(old)


def test_fixed_variance_components_have_correct_sum_loss_precision():
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    penalties = dict(zip(model.feature_names, model.penalties))
    assert model.component_variance == pytest.approx(.005)
    assert penalties["role_attack:GK"] == pytest.approx(200)
    assert penalties["player_defence:impect:player:h10"] == pytest.approx(200)
    assert penalties["team_attack:1"] == penalties["home_advantage"] == 1
    assert penalties["intercept"] == 0
    assert "summed" in model.parameters()["loss"]
    assert "not learned" in model.parameters()["prior_structure"]


def test_role_reference_is_per_match_counts_and_player_reference_is_presence():
    rows = history(4)
    original = deepcopy(rows)
    model = HierarchicalStartingXIModel().fit(rows, CUTOFF)
    assert rows == original
    assert model.reference_frequencies[1]["impect:player:h10"] == .5
    assert model.reference_frequencies[1]["impect:player:rotation"] == .5
    assert model.reference_role_counts[1] == {"GK": 1, "DEF": 3.5, "MID": 3.5, "FWD": 3}
    assert not hasattr(model, "permanent_player_roles")
    home, away = model._features(_role_fixture(fixture(100, CUTOFF)))
    assert home["role_attack:DEF"] == .5
    assert home["role_attack:MID"] == -.5
    assert away["role_defence:DEF"] == -.5
    assert "role_attack:GK" not in home
    assert home["player_attack:impect:player:h10"] == .5


def test_likelihood_gradient_and_precision_are_direct_full_match_not_exposure():
    model = HierarchicalStartingXIModel().fit(history(6), CUTOFF)
    normalized = [_role_fixture(row) for row in history(6)]
    design = model._design(normalized)
    target = np.asarray([goal for row in history(6) for goal in row["full_time"]], dtype=float)
    point = np.linspace(-.05, .05, len(model.feature_names))
    loss, gradient = _penalized_poisson_objective(point, design, target, model.penalties)
    eta = design @ point
    assert loss == pytest.approx(np.sum(np.exp(eta) - target * eta) + .5 * np.dot(model.penalties * point, point))
    for coordinate in (0, 1, 6, len(point) - 1):
        shift = np.zeros_like(point)
        shift[coordinate] = 1e-6
        numerical = (_penalized_poisson_objective(point + shift, design, target, model.penalties)[0]
                     - _penalized_poisson_objective(point - shift, design, target, model.penalties)[0]) / 2e-6
        assert gradient[coordinate] == pytest.approx(numerical, abs=1e-7)
    expected_precision = (design.T @ design.multiply(np.exp(design @ model.coefficients)[:, None])).toarray() + np.diag(model.penalties)
    assert model.precision_cholesky @ model.precision_cholesky.T == pytest.approx(expected_precision, abs=1e-10)


def test_known_covariance_keeps_cross_terms_and_matches_full_precision_solve():
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    target = fixture(100, CUTOFF)
    prediction = model.predict_distribution(target)
    design = model._design([_role_fixture(target)]).toarray()
    precision = model.precision_cholesky @ model.precision_cholesky.T
    expected = design @ linalg.solve(precision, design.T, assume_a="pos")
    assert prediction["known_parameter_log_cov"] == pytest.approx(expected)
    assert abs(expected[0, 1]) > 1e-8
    assert prediction["log_cov"] == pytest.approx(expected)
    assert prediction["unknown_player_ids"] == []
    assert prediction["unknown_player_log_cov"] == pytest.approx(np.zeros((2, 2)))


@pytest.mark.parametrize("home_unknown,away_unknown", [(1, 0), (0, 1), (1, 1), (2, 1)])
def test_unknown_residual_prior_variance_counts_both_teams_for_both_means(home_unknown, away_unknown):
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    target = fixture(100, CUTOFF)
    for side, count in (("home", home_unknown), ("away", away_unknown)):
        for number in range(count):
            replace_player(target, side, target[f"{side}_starters"][-1 - number], f"impect:player:new-{side}-{number}")
    prediction = model.predict_distribution(target)
    expected = np.eye(2) * (home_unknown + away_unknown) * .005
    assert prediction["unknown_player_log_cov"] == pytest.approx(expected)
    assert np.asarray(prediction["log_cov"]) - np.asarray(prediction["known_parameter_log_cov"]) == pytest.approx(expected)
    assert len(prediction["unknown_player_ids"]) == home_unknown + away_unknown
    assert all(player not in model.players for player in prediction["unknown_player_ids"])
    assert "integrated" in prediction["unknown_player_policy"]


def test_prior_mean_zero_is_not_zero_effect_imputation():
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    target = fixture(100, CUTOFF)
    replace_player(target, "home", "impect:player:h10", "impect:player:unseen")
    prediction = model.predict_distribution(target)
    assert prediction["unknown_player_log_cov"][0][0] > 0
    assert prediction["unknown_player_log_cov"][1][1] > 0
    log_mean, log_cov = np.asarray(prediction["log_mean"]), np.asarray(prediction["log_cov"])
    integrated_means = np.exp(log_mean + .5 * np.diag(log_cov))
    assert np.all(integrated_means > prediction["map_means"])


def test_team_only_matches_old_team_map_and_adds_matched_laplace_not_player_prior():
    rows = history()
    old = FullMatchStartingXIModel(include_players=False).fit(rows, CUTOFF)
    model = HierarchicalStartingXIModel(include_players=False).fit(rows, CUTOFF)
    assert model.feature_names == old.feature_names
    assert model.coefficients == pytest.approx(old.coefficients, abs=1e-12)
    target = fixture(100, CUTOFF)
    replace_player(target, "away", "impect:player:a10", "impect:player:unknown")
    prediction = model.predict_distribution(target)
    assert prediction["map_means"] == pytest.approx(list(old.predict_match(target)[key] for key in ("home_mean", "away_mean")))
    assert prediction["unknown_player_log_cov"] == pytest.approx(np.zeros((2, 2)))
    assert np.asarray(prediction["known_parameter_log_cov"]).trace() > 0
    assert not any(name.startswith(("role_", "player_")) for name in model.feature_names)


def test_prediction_target_paths_and_minutes_are_not_read_and_future_rows_not_validated():
    rows = history()
    model = HierarchicalStartingXIModel().fit(rows, CUTOFF)
    future = {"match_date": CUTOFF.isoformat(), "home_roles": "invalid future roles", "home_starters": "not a list"}
    with_future = HierarchicalStartingXIModel().fit(rows + [future], CUTOFF)
    assert with_future.parameters() == model.parameters()
    assert np.array_equal(with_future.precision_cholesky, model.precision_cholesky)
    target = fixture(100, CUTOFF)
    expected = model.predict_distribution(target)
    for key in ("full_time", "actual_minutes", "stints", "bench", "duration_seconds", "home_man_difference"):
        target[key] = object()
    assert model.predict_distribution(target) == expected
    assert "impect:player:future" not in model.players


def test_unknown_team_and_earlier_prediction_fail_explicitly():
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    target = fixture(100, CUTOFF)
    target["away_team_id"] = 3
    with pytest.raises(UnknownHierarchicalTeamError, match="Unknown teams"):
        model.predict_distribution(target)
    with pytest.raises(ValueError, match="fitted cutoff"):
        model.predict_distribution(fixture(100, CUTOFF - timedelta(days=1)))


@pytest.mark.parametrize("bad", ["NOT-A-ROLE", None, 3])
def test_unknown_role_labels_not_invented_or_mapped_by_model(bad):
    target = fixture()
    target["home_roles"]["impect:player:h10"] = bad
    with pytest.raises(ValueError, match="GK/DEF/MID/FWD"):
        HierarchicalStartingXIModel().fit([target], CUTOFF)


def test_role_mapping_has_exact_roster_and_one_keeper():
    target = fixture()
    target["home_roles"]["impect:player:bench"] = "MID"
    with pytest.raises(ValueError, match="exactly its 11"):
        HierarchicalStartingXIModel().fit([target], CUTOFF)
    target = fixture()
    target["home_roles"]["impect:player:h1"] = "GK"
    with pytest.raises(ValueError, match="one initial goalkeeper"):
        HierarchicalStartingXIModel().fit([target], CUTOFF)


@pytest.mark.parametrize("prior_sd", [.05, .10, .20])
def test_fixed_prior_sensitivity_values_are_supported_without_search(prior_sd):
    model = HierarchicalStartingXIModel(prior_sd=prior_sd).fit(history(4), CUTOFF)
    assert model.component_variance == pytest.approx(prior_sd**2 / 2)
    assert model.penalties[model.feature_index["role_attack:DEF"]] == pytest.approx(2 / prior_sd**2)


@pytest.mark.parametrize("include_players", [False, True])
def test_save_load_same_stem_factor_and_metadata_roundtrip(tmp_path, include_players):
    model = HierarchicalStartingXIModel(include_players=include_players).fit(history(), CUTOFF)
    path = tmp_path / "laplace.json"
    model.save(path)
    assert path.with_suffix(".npz").exists()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert "precision_cholesky" not in metadata
    restored = HierarchicalStartingXIModel.load(path)
    assert restored.parameters() == model.parameters()
    assert np.array_equal(restored.precision_cholesky, model.precision_cholesky)
    assert restored.predict_distribution(fixture(100, CUTOFF)) == model.predict_distribution(fixture(100, CUTOFF))


@pytest.mark.parametrize("what", ["factor_shape", "factor_upper", "factor_diagonal", "role_refs", "component", "penalty"])
def test_invalid_saved_metadata_and_factors_rejected(what):
    model = HierarchicalStartingXIModel().fit(history(4), CUTOFF)
    metadata, factor = model.parameters(), model.precision_cholesky.copy()
    if what == "factor_shape":
        factor = factor[:-1]
    elif what == "factor_upper":
        factor[0, 1] = 1
    elif what == "factor_diagonal":
        factor[0, 0] = 0
    elif what == "role_refs":
        metadata["reference_role_counts"]["1"]["GK"] = .5
    elif what == "component":
        metadata["component_prior_variance"] = .01
    elif what == "penalty":
        metadata["penalties"][-1] = 100
    with pytest.raises(ValueError):
        HierarchicalStartingXIModel.from_parameters(metadata, factor)


def test_failed_refit_and_cholesky_failure_do_not_keep_stale_model(monkeypatch):
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    def fail(*args, **kwargs):
        raise linalg.LinAlgError("synthetic factor failure, no jitter")
    monkeypatch.setattr(linalg, "cholesky", fail)
    with pytest.raises(linalg.LinAlgError, match="no jitter"):
        model.fit(history(), CUTOFF)
    with pytest.raises(ValueError, match="Fit earlier"):
        model.predict_distribution(fixture(100, CUTOFF))
    with pytest.raises(ValueError, match="Fit before"):
        model.parameters()


@pytest.mark.parametrize("key,value", [("prior_sd", 0), ("prior_sd", float("nan")), ("prior_sd", True),
    ("team_ridge", -1), ("home_ridge", 0), ("include_players", 1), ("max_iterations", False)])
def test_invalid_settings_rejected(key, value):
    with pytest.raises(ValueError):
        HierarchicalStartingXIModel(**{key: value})


def test_constant_keeper_count_does_not_claim_group_learning():
    model = HierarchicalStartingXIModel().fit(history(), CUTOFF)
    column = model.feature_index["role_attack:GK"]
    assert model.coefficients[column] == pytest.approx(0, abs=1e-12)
    precision = model.precision_cholesky @ model.precision_cholesky.T
    assert precision[column, column] == pytest.approx(200)
    assert all(abs(precision[column, other]) < 1e-10 for other in range(len(precision)) if other != column)
    assert "constant GK" in model.parameters()["limits"][3]
