from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import penaltyblog as pb
import pytest

from football_model.football_data import HistoricalMatch
from football_model.penaltyblog_adapter import NativeFitError, NativeGoalModel


def history():
    rng = np.random.default_rng(261022)
    teams = ("A", "B", "C", "D")
    rows = []
    start = date(2024, 8, 1)
    for i in range(96):
        home = teams[i % 4]
        away = teams[(i % 4 + 1 + (i // 4) % 3) % 4]
        h = int(rng.poisson(1.9 if home == "A" else 1.5))
        a = int(rng.poisson(0.9 if away == "D" else 1.2))
        rows.append(HistoricalMatch(
            str(i), "E0", "2024/25", start + timedelta(days=i), "15:00", home, away,
            home, away, (h, a), None, {}, "synthetic", "synthetic"))
    return rows


@pytest.fixture(scope="module")
def fitted():
    rows = history()
    cutoff = date(2024, 11, 1)
    return {name: NativeGoalModel(name).fit(rows, cutoff) for name in ("poisson", "dixon_coles")}


@pytest.mark.parametrize("name", ["poisson", "dixon_coles"])
def test_native_fit_predict_and_json_parameters(fitted, name):
    model = fitted[name]
    grid = model.predict("A", "D", "E0", date(2024, 11, 2))
    assert type(grid) is pb.models.FootballProbabilityGrid
    assert grid.grid.shape == (30, 30)
    assert np.isfinite(grid.grid).all()
    assert (grid.grid >= 0).all()
    assert grid.grid.sum() == pytest.approx(1)
    assert sum(grid.home_draw_away) == pytest.approx(1)
    assert grid.home_goal_expectation != pytest.approx(grid.away_goal_expectation)
    params = model.parameters()
    json.dumps(params, allow_nan=False)
    assert params["basis"] == "team_marginal"
    assert params["optimizer"]["success"] is True
    assert params["training_max_date"] < params["before_date"]
    if name == "dixon_coles":
        assert "rho" in params["native_parameters"]


def test_time_weights_and_unweighted_option(fitted):
    model = fitted["poisson"]
    assert model.weights[0] == pytest.approx(math.exp(-math.log(2) * 92 / 180))
    assert model.weights[-1] == pytest.approx(math.exp(-math.log(2) / 180))
    unweighted = NativeGoalModel(half_life_days=None).fit(history(), date(2024, 11, 1))
    assert np.all(unweighted.weights == 1)


def test_same_day_future_and_administrative_results_do_not_enter_fit():
    rows = history()
    cutoff = date(2024, 10, 1)
    rows[0] = replace(rows[0], goal_fit_exclusion="administratively_awarded_result")
    first = NativeGoalModel().fit(rows, cutoff)
    changed = [replace(r, full_time=(20, 0)) if r.match_date >= cutoff else r for r in rows]
    second = NativeGoalModel().fit(changed, cutoff)
    assert first.n_matches == 60
    assert first.training_max_date < cutoff
    assert "0" not in first.reference_match_ids
    a = first.predict("A", "D", "E0", cutoff).grid
    b = second.predict("A", "D", "E0", cutoff).grid
    assert np.allclose(a, b)


def test_reject_duplicate_cross_league_and_fractional_goals():
    rows = history()
    cutoff = date(2024, 11, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        NativeGoalModel().fit(rows + rows[:1], cutoff)
    with pytest.raises(ValueError, match="one league"):
        NativeGoalModel().fit([replace(r, competition="D1") if i == 0 else r
                               for i, r in enumerate(rows)], cutoff)
    for bad in ((1.2, 0), (True, 0), (-1, 0)):
        with pytest.raises(ValueError, match="non-negative integers"):
            NativeGoalModel().fit([replace(rows[0], full_time=bad), *rows[1:]], cutoff)


def test_prediction_identity_chronology_and_unknown_team(fitted):
    model = fitted["poisson"]
    with pytest.raises(ValueError, match="Unseen"):
        model.predict("new", "A", "E0", model.before_date)
    with pytest.raises(ValueError, match="different league"):
        model.predict("A", "B", "D1", model.before_date)
    with pytest.raises(ValueError, match="precede"):
        model.predict("A", "B", "E0", model.before_date - timedelta(days=1))
    with pytest.raises(ValueError, match="Teams must differ"):
        model.predict("A", "A", "E0", model.before_date)
    for bad in (datetime(2024, 11, 1), datetime(2024, 11, 1, tzinfo=timezone.utc)):
        with pytest.raises(ValueError, match="calendar date"):
            NativeGoalModel().fit(history(), bad)
        with pytest.raises(ValueError, match="precede"):
            model.predict("A", "B", "E0", bad)


def test_native_save_load_self_generated_only(tmp_path, fitted):
    model = fitted["dixon_coles"]
    path = tmp_path / "self_generated_native.pkl"
    model.save(path)
    restored = pb.models.DixonColesGoalModel.load(str(path))
    assert restored.fitted
    expected = model.native.predict("A", "D", max_goals=30).grid
    assert np.allclose(restored.predict("A", "D", max_goals=30).grid, expected)


def test_failed_optimizer_is_identifiable(monkeypatch):
    def failed(self, **kwargs):
        raise ValueError("Iteration limit reached")
    monkeypatch.setattr(pb.models.PoissonGoalsModel, "fit", failed)
    with pytest.raises(NativeFitError, match="Iteration limit"):
        NativeGoalModel().fit(history(), date(2024, 11, 1))


@pytest.mark.parametrize("matrix", [
    np.asarray([[float("nan"), 0.0], [0.0, 1.0]]),
    np.asarray([[-0.01, 0.01], [0.5, 0.5]]),
    np.asarray([[0.1, 0.1], [0.1, 0.1]]),
])
def test_invalid_native_probability_not_clipped_or_silently_normalized(fitted, monkeypatch, matrix):
    model = fitted["poisson"]
    raw = SimpleNamespace(grid=matrix, home_goal_expectation=1.2, away_goal_expectation=1.0)
    monkeypatch.setattr(model.native, "predict", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="probabilities|material mass"):
        model.predict("A", "D", "E0", model.before_date)


def test_unfitted_invalid_configuration_and_public_signatures(tmp_path):
    for args in (("unknown", 180), ("poisson", 0), ("poisson", float("nan")), ("poisson", True)):
        with pytest.raises(ValueError):
            NativeGoalModel(*args)
    model = NativeGoalModel()
    with pytest.raises(ValueError, match="Fit earlier"):
        model.predict("A", "B", "E0", date(2024, 11, 1))
    with pytest.raises(ValueError, match="not been fitted"):
        model.parameters()
    with pytest.raises(ValueError, match="not been fitted"):
        model.save(tmp_path / "not_created.pkl")
    with pytest.raises(ValueError, match="No eligible"):
        model.fit(history(), date(2024, 8, 1))
    assert list(inspect.signature(model.fit).parameters) == ["records", "before_date"]
    assert list(inspect.signature(model.predict).parameters) == ["home", "away", "competition", "match_date"]
