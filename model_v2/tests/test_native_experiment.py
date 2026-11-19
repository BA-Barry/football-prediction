from dataclasses import replace
from datetime import date, timedelta
import inspect

import numpy as np
import pytest

from football_model.football_data import HistoricalMatch
from football_model import native_experiment as experiment
from football_model.penaltyblog_adapter import NativeFitError
from football_model.probability import score_grid


def history():
    start = date(2024, 7, 1)
    rng = np.random.default_rng(1022)
    teams = ("A", "B", "C", "D")
    rows = []
    for i in range(90):
        home = teams[i % 4]
        away = teams[(i % 4 + 1 + (i // 4) % 3) % 4]
        rows.append(HistoricalMatch(
            str(i), "E0", "2024/25", start + timedelta(days=i), "15:00", home, away,
            home, away, (int(rng.poisson(1.6)), int(rng.poisson(1.1))), None,
            {"avg_preclosing": (2.0, 3.5, 4.0)}, "synthetic", "synthetic"))
    return rows


class FastModel:
    """Deterministic ordinary test double; actual native smoke is tested below."""
    def fit(self, rows, cutoff):
        assert all(r.match_date < cutoff for r in rows)
        self.training_max_date = max(r.match_date for r in rows)
        self.n_matches = len(rows)
        self.mean = 0.1 + sum(r.full_time[0] for r in rows) / len(rows)
        return self

    def predict(self, home, away, competition, match_date):
        return score_grid(self.mean, 1.1)

    def parameters(self):
        return {"n_matches": self.n_matches}


def config(family="poisson", half=180):
    return {"id": f"{family}_{half}", "family": family, "half_life_days": half}


def test_future_and_same_block_scores_do_not_change_earlier_predictions(monkeypatch):
    monkeypatch.setattr(experiment, "make_model", lambda _: FastModel())
    rows = history()
    first = experiment.run_config(rows, config(), date(2024, 8, 1), date(2024, 9, 26))
    changed = [replace(r, full_time=(20, 0)) if r.match_date >= date(2024, 8, 1) else r
               for r in rows]
    second = experiment.run_config(changed, config(), date(2024, 8, 1), date(2024, 9, 26))
    early = [p for p in first["predictions"] if p["fit_before_date"] == "2024-08-01"]
    altered = [p for p in second["predictions"] if p["fit_before_date"] == "2024-08-01"]
    assert len(early) == 28
    assert [p["probabilities"] for p in early] == [p["probabilities"] for p in altered]
    assert all(p["training_max_date"] < p["fit_before_date"] <= p["date"] for p in early)


def test_unknown_team_and_administrative_results_excluded_for_all_candidates(monkeypatch):
    monkeypatch.setattr(experiment, "make_model", lambda _: FastModel())
    rows = history()
    rows[31] = replace(rows[31], home="unseen")
    rows[32] = replace(rows[32], goal_fit_exclusion="administratively_awarded_result")
    start, end = date(2024, 8, 1), date(2024, 8, 29)
    expected = experiment.expected_target_ids(rows, start, end)
    completed = []
    for candidate in experiment.CANDIDATES:
        trial = experiment.run_config(rows, candidate, start, end)
        ids = {p["match_id"] for p in trial["predictions"]}
        assert ids == expected
        assert "31" not in ids and "32" not in ids
        assert len(trial["skipped"]) == 2
        assert trial["metrics"]["model"]["n"] == len(expected)
        pair = trial["metrics"]["markets"]["avg_preclosing"]
        assert pair["market"]["n"] == pair["model_on_identical_matches"]["n"] == len(expected)
        completed.append(ids)
    assert all(ids == completed[0] for ids in completed)


def trial(identifier, loss, ids=("a", "b"), failures=None):
    return {
        "start": experiment.VALID_START.isoformat(),
        "end_exclusive": experiment.VALID_END.isoformat(),
        "config": {"id": identifier, "family": "poisson", "half_life_days": 180},
        "predictions": [{"match_id": key, "date": "2024-10-01"} for key in ids],
        "failures": failures or [],
        "metrics": {"model": {"n": len(ids), "log_loss": loss}},
    }


def test_failed_or_incomplete_candidates_cannot_win_selection():
    runs = [trial("failed", 0.0, failures=[{"stage": "predict"}]),
            trial("incomplete", 0.1, ids=("a",)),
            trial("valid_b", 1.0), trial("valid_a", 0.9)]
    selection = experiment.select_config(runs, {"a", "b"})
    assert selection["config"]["id"] == "valid_a"
    assert selection["eligible_candidates"] == 2
    assert selection["expected_matches"] == 2
    with pytest.raises(ValueError, match="No native candidate"):
        experiment.select_config(runs[:2], {"a", "b"})


def test_selection_refuses_test_period_or_later_prediction_dates():
    later = trial("later", 0.1)
    later["start"] = experiment.TEST_START.isoformat()
    later["end_exclusive"] = experiment.TEST_END.isoformat()
    with pytest.raises(ValueError):
        experiment.select_config([later], {"a", "b"})
    later = trial("later_date", 0.1)
    later["predictions"][0]["date"] = experiment.TEST_START.isoformat()
    with pytest.raises(ValueError):
        experiment.select_config([later], {"a", "b"})
    assert list(inspect.signature(experiment.select_config).parameters) == ["validation_runs", "expected_ids"]


def test_native_failures_are_recorded_without_silent_fallback(monkeypatch):
    calls = []

    def fail(_):
        calls.append("native")
        raise NativeFitError("deliberate optimizer failure")

    monkeypatch.setattr(experiment, "make_model", fail)
    result = experiment.run_config(history(), config(), date(2024, 8, 1), date(2024, 8, 29))
    assert calls == ["native"]
    assert result["predictions"] == []
    assert result["fits"] == []
    assert result["failures"][0]["stage"] == "fit"
    assert "deliberate optimizer failure" in result["failures"][0]["error"]


def test_prediction_failures_are_reported_not_replaced(monkeypatch):
    class InvalidPrediction(FastModel):
        def predict(self, *args):
            raise ValueError("negative native probability")

    monkeypatch.setattr(experiment, "make_model", lambda _: InvalidPrediction())
    result = experiment.run_config(history(), config(), date(2024, 8, 1), date(2024, 8, 29))
    assert not result["predictions"]
    assert len(result["failures"]) == 28
    assert all(f["stage"] == "predict" for f in result["failures"])


def test_duplicate_matches_are_rejected_before_fit():
    rows = history()
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.run_config(rows + rows[:1], config(), date(2024, 8, 1), date(2024, 8, 29))


def test_actual_complete_native_package_smoke():
    rows = history()
    result = experiment.run_config(rows, config(), date(2024, 8, 1), date(2024, 8, 8), interval_days=28)
    assert len(result["predictions"]) == 7
    assert not result["failures"]
    assert result["fits"][0]["penaltyblog_version"]
    assert result["fits"][0]["optimizer"]["success"]
    assert all(sum(p["probabilities"]) == pytest.approx(1) for p in result["predictions"])
