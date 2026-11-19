"""Core parameter search changes actual forecasts, without later-score selection."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import inspect
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from football_model import core_tuning as core
from football_model.date_model import DateTeamGoalModel
from football_model.football_data import HistoricalMatch, LEAGUES
from football_model.native_experiment import expected_target_ids
from football_model.probability import score_grid, summarize


def history():
    start = date(2024, 7, 1)
    rng = np.random.default_rng(10623)
    teams = ("A", "B", "C", "D")
    return [HistoricalMatch(
        str(i), "E0", "2024/25", start + timedelta(days=i), None,
        teams[i % 4], teams[(i % 4 + 1 + (i // 4) % 3) % 4],
        teams[i % 4], teams[(i % 4 + 1 + (i // 4) % 3) % 4],
        (int(rng.poisson(1.7)), int(rng.poisson(1.0))), None,
        {"avg_preclosing": (2.0, 3.5, 4.0)}, "synthetic", "synthetic")
        for i in range(90)]


class FastModel:
    def __init__(self, regularization, half_life_days):
        self.regularization = regularization
        self.half_life_days = half_life_days

    def fit(self, rows, cutoff):
        assert all(r.match_date < cutoff and not r.goal_fit_exclusion for r in rows)
        self.training_max_date = max(r.match_date for r in rows)
        self.n_matches = len(rows)
        self.mean = .1 + sum(r.full_time[0] for r in rows) / len(rows)
        self.reference_match_ids = [r.match_id for r in rows]
        self.cutoff = cutoff
        return self

    def predict(self, home, away, competition, match_date):
        return self.mean, 1.1

    def parameters(self):
        return {"n_matches": self.n_matches, "before_date": self.cutoff.isoformat(),
                "training_max_date": self.training_max_date.isoformat(),
                "reference_match_ids": list(self.reference_match_ids),
                "regularization": self.regularization, "half_life_days": self.half_life_days}


def test_grid_is_exactly_nine_settings_and_original_is_included():
    configs = core.candidate_configs()
    assert len(configs) == len({c["id"] for c in configs}) == 9
    assert {(c["half_life_days"], c["regularization"]) for c in configs} == {
        (h, alpha) for h in (90, 180, 365) for alpha in (.001, .01, .1)}
    assert all(c["family"] == "original_regularized" for c in configs)
    assert configs[4]["id"] == "regularized_poisson_h180_a0p01"
    assert json.loads(json.dumps(configs)) == configs
    configs[0]["regularization"] = 999
    assert core.candidate_configs()[0]["regularization"] == .001


def test_run_receives_exact_parameters_and_retains_readable_evidence(monkeypatch):
    monkeypatch.setattr(core, "DateTeamGoalModel", FastModel)
    config = core.candidate_configs()[0]
    result = core.run_candidate(history(), config, date(2024, 8, 1), date(2024, 8, 29))
    assert not result["failures"]
    assert len(result["predictions"]) == 28 and len(result["fits"]) == 1
    fit = result["fits"][0]
    assert fit["regularization"] == .001 and fit["half_life_days"] == 90
    assert fit["reference_match_ids"] == [str(i) for i in range(31)]
    row = result["predictions"][0]
    assert row["training_matches"] == 31
    assert row["training_max_date"] < row["fit_before_date"] <= row["date"]
    assert row["market_prices"] == {"avg_preclosing": (2.0, 3.5, 4.0)}
    assert row["source_url"] == "synthetic"
    assert row["forecast"]["away_minus_two_jincai"]
    assert row["rates"] == [.1 + sum(r.full_time[0] for r in history()[:31])/31, 1.1]
    assert summarize(score_grid(*row["rates"], rho=0)) == row["forecast"]
    json.dumps(result, allow_nan=False)


def test_future_and_same_block_scores_cannot_change_first_block_forecasts(monkeypatch):
    monkeypatch.setattr(core, "DateTeamGoalModel", FastModel)
    rows = history()
    config = core.candidate_configs()[4]
    first = core.run_candidate(rows, config, date(2024, 8, 1), date(2024, 9, 26))
    changed = [replace(r, full_time=(20, 0)) if r.match_date >= date(2024, 8, 1) else r
               for r in rows]
    second = core.run_candidate(changed, config, date(2024, 8, 1), date(2024, 9, 26))
    early = [p["probabilities"] for p in first["predictions"] if p["fit_before_date"] == "2024-08-01"]
    altered = [p["probabilities"] for p in second["predictions"] if p["fit_before_date"] == "2024-08-01"]
    assert early == altered and len(early) == 28
    assert first["fits"][1]["before_date"] == "2024-08-29"
    assert first["fits"][1]["n_matches"] == 59
    assert first["predictions"][-1]["probabilities"] != second["predictions"][-1]["probabilities"]


def test_unknown_team_and_special_result_follow_existing_cohort_policy(monkeypatch):
    monkeypatch.setattr(core, "DateTeamGoalModel", FastModel)
    rows = history()
    rows[31] = replace(rows[31], home="unseen", home_label="unseen")
    rows[32] = replace(rows[32], goal_fit_exclusion="administratively_awarded_result")
    rows[10] = replace(rows[10], goal_fit_exclusion="resumed_partially_played_match")
    start, end = date(2024, 8, 1), date(2024, 8, 29)
    expected = expected_target_ids(rows, start, end)
    result = core.run_candidate(rows, core.candidate_configs()[4], start, end)
    assert {p["match_id"] for p in result["predictions"]} == expected
    assert result["skipped"] == [
        {"match_id": "32", "reason": "administratively_awarded_result"},
        {"match_id": "31", "reason": "unseen_team_no_estimated_prior"}]
    assert "10" not in result["fits"][0]["reference_match_ids"]
    assert result["fits"][0]["n_matches"] == 30


def test_no_history_is_explicit_unknown_skip_not_default_strength(monkeypatch):
    class NeverFit:
        def __init__(self, *args):
            raise AssertionError("No target with unknown teams should be fitted")
    monkeypatch.setattr(core, "DateTeamGoalModel", NeverFit)
    rows = [r for r in history() if r.match_date >= date(2024, 8, 1)]
    result = core.run_candidate(rows, core.candidate_configs()[4], date(2024, 8, 1), date(2024, 8, 29))
    assert not result["predictions"] and not result["fits"]
    assert len(result["skipped"]) == 28
    assert result["metrics"]["model"] is None


def test_empty_blocks_advance_fixed_calendar_anchor(monkeypatch):
    monkeypatch.setattr(core, "DateTeamGoalModel", FastModel)
    rows = [r for r in history() if r.match_date < date(2024, 8, 1)
            or r.match_date >= date(2024, 9, 1)]
    result = core.run_candidate(rows, core.candidate_configs()[4], date(2024, 8, 1), date(2024, 9, 26))
    assert len(result["fits"]) == 1
    assert result["fits"][0]["before_date"] == "2024-08-29"


@pytest.mark.parametrize("failure_type", [ValueError, RuntimeError, FloatingPointError, OverflowError])
def test_fit_failure_is_recorded_without_fallback(monkeypatch, failure_type):
    calls = []
    class FailModel(FastModel):
        def fit(self, rows, cutoff):
            calls.append(cutoff)
            raise failure_type("deliberate numerical fit failure")
    monkeypatch.setattr(core, "DateTeamGoalModel", FailModel)
    result = core.run_candidate(history(), core.candidate_configs()[4], date(2024, 8, 1), date(2024, 8, 29))
    assert calls == [date(2024, 8, 1)]
    assert not result["predictions"] and not result["fits"]
    assert len(result["failures"]) == 1 and result["failures"][0]["stage"] == "fit"
    assert len(result["failures"][0]["matches"]) == 28
    assert result["failures"][0]["error_type"] == failure_type.__name__


def test_convergence_warning_records_failed_fit_and_restores_warning_filters(monkeypatch):
    class WarningModel(FastModel):
        def fit(self, rows, cutoff):
            warnings.warn("optimizer did not converge", ConvergenceWarning)
            return super().fit(rows, cutoff)
    monkeypatch.setattr(core, "DateTeamGoalModel", WarningModel)
    filters = list(warnings.filters)
    result = core.run_candidate(history(), core.candidate_configs()[4], date(2024, 8, 1), date(2024, 8, 29))
    assert not result["fits"] and not result["predictions"]
    assert result["failures"][0]["error_type"] == "ConvergenceWarning"
    assert "optimizer did not converge" in result["failures"][0]["error"]
    assert warnings.filters == filters


@pytest.mark.parametrize("failure_type", [ValueError, FloatingPointError, OverflowError])
def test_prediction_failure_is_not_filled(monkeypatch, failure_type):
    class FailedPrediction(FastModel):
        def predict(self, *args):
            raise failure_type("deliberate bad forecast")
    monkeypatch.setattr(core, "DateTeamGoalModel", FailedPrediction)
    result = core.run_candidate(history(), core.candidate_configs()[4], date(2024, 8, 1), date(2024, 8, 29))
    assert len(result["fits"]) == 1 and not result["predictions"]
    assert len(result["failures"]) == 28
    assert all(f["stage"] == "predict" and f["error_type"] == failure_type.__name__ for f in result["failures"])


@pytest.mark.parametrize("start,end", [
    (datetime(2024, 8, 1), date(2025, 7, 1)),
    (date(2024, 8, 1), datetime(2025, 7, 1)),
    (date(2024, 8, 2), date(2025, 7, 1)),
    (date(2024, 8, 1), date(2024, 8, 1)),
    (date(2024, 8, 1), date(2024, 7, 1)),
    (date(2024, 8, 1), date(2025, 8, 2)),
])
def test_invalid_boundaries_are_rejected(start, end):
    with pytest.raises(ValueError, match="calendar dates"):
        core.run_candidate(history(), core.candidate_configs()[4], start, end)


@pytest.mark.parametrize("field,value", [
    ("id", "invented"), ("family", "native_poisson"),
    ("half_life_days", True), ("half_life_days", None), ("half_life_days", 91),
    ("regularization", False), ("regularization", float("nan")), ("regularization", -.01),
])
def test_unregistered_or_invalid_config_is_rejected(field, value):
    config = core.candidate_configs()[4]
    config[field] = value
    with pytest.raises(ValueError):
        core.run_candidate(history(), config, date(2024, 8, 1), date(2024, 8, 29))


@pytest.mark.parametrize("field,value", [
    ("match_date", datetime(2024, 7, 1)), ("home", ""), ("home", "B"),
    ("source_url", ""), ("full_time", (-1, 0)), ("full_time", (True, 0)),
    ("full_time", (1.5, 0)), ("markets", {"invented": (2., 3., 4.)}),
    ("markets", {"avg_preclosing": (1., 3., 4.)}),
])
def test_invalid_records_are_rejected(field, value):
    rows = history()
    # The first synthetic fixture is A versus B.
    rows[0] = replace(rows[0], **{field: value})
    with pytest.raises(ValueError):
        core.run_candidate(rows, core.candidate_configs()[4], date(2024, 8, 1), date(2024, 8, 29))


def test_empty_mixed_duplicate_and_unknown_league_inputs_fail():
    config = core.candidate_configs()[4]
    rows = history()
    cases = [[], rows + rows[:1], rows + [replace(rows[0], match_id="other", competition="D1")],
             [replace(r, competition="unknown") for r in rows]]
    for case in cases:
        with pytest.raises(ValueError):
            core.run_candidate(case, config, date(2024, 8, 1), date(2024, 8, 29))


def validation_runs():
    forecast = summarize(score_grid(1.5, 1.1))
    records = [{"match_id": f"id-{comp}", "competition": comp, "date": "2024-10-01",
                "home": "A", "away": "B", "fit_before_date": "2024-09-26",
                "training_max_date": "2024-09-25", "training_matches": 100,
                "actual_score": [1, 0], "actual_outcome": 0,
                "probabilities": [.5, .3, .2], "forecast": deepcopy(forecast),
                "market_probabilities": {}, "market_prices": {}, "source_url": "synthetic"}
               for comp in LEAGUES]
    return [{"config": config, "start": "2024-08-01", "end_exclusive": "2025-07-01",
             "update_interval_days": 28, "predictions": deepcopy(records), "failures": [],
             "metrics": {"model": {"n": 5, "log_loss": 0.0}}}
            for config in core.candidate_configs()]


def expected_ids():
    return {f"id-{comp}" for comp in LEAGUES}


def test_selection_recomputes_pooled_score_and_does_not_trust_saved_metric():
    runs = validation_runs()
    runs[7]["predictions"][0]["probabilities"] = [.8, .1, .1]
    before = deepcopy(runs)
    result = core.select_policy(runs, expected_ids())
    assert result["config"] == core.candidate_configs()[7]
    assert result["validation_scores"]["log_loss"] == pytest.approx(-(np.log(.8) + 4*np.log(.5))/5)
    assert result["validation_scores"]["n"] == 5
    assert result["expected_matches"] == 5 and result["eligible_candidates"] == 9
    assert len(result["candidate_scores"]) == 9
    assert runs == before
    assert list(inspect.signature(core.select_policy).parameters) == ["validation_runs", "expected_ids"]
    json.dumps(result, allow_nan=False)


def test_selection_tie_uses_id_and_is_invariant_to_run_or_match_order():
    runs = validation_runs()
    chosen = min(c["id"] for c in core.candidate_configs())
    assert core.select_policy(runs, expected_ids())["config"]["id"] == chosen
    for run in runs:
        run["predictions"].reverse()
    assert core.select_policy(list(reversed(runs)), expected_ids())["config"]["id"] == chosen


def test_selection_compares_fresh_tuple_and_archived_list_quotes_without_false_conflict():
    runs = validation_runs()
    for run in runs:
        for row in run["predictions"]:
            row["market_prices"] = {"avg_preclosing": (2., 4., 4.)}
            row["market_probabilities"] = {"avg_preclosing": [.5, .25, .25]}
    runs[0] = json.loads(json.dumps(runs[0]))
    assert core.select_policy(runs, expected_ids())["expected_matches"] == 5


@pytest.mark.parametrize("mutation", [
    "missing_candidate", "duplicate_candidate", "failure", "missing_match", "extra_match",
    "duplicate_match", "one_league", "late_period", "late_prediction", "bad_anchor", "bad_interval",
    "inconsistent_actual", "inconsistent_market", "negative_probability", "future_training",
])
def test_selection_refuses_incomplete_conflicted_or_late_data(mutation):
    runs = validation_runs()
    if mutation == "missing_candidate":
        runs.pop()
    elif mutation == "duplicate_candidate":
        runs[0] = deepcopy(runs[1])
    elif mutation == "failure":
        runs[0]["failures"] = [{"stage": "fit", "error": "not converged"}]
    elif mutation == "missing_match":
        runs[0]["predictions"].pop()
    elif mutation == "extra_match":
        runs[0]["predictions"].append({**runs[0]["predictions"][0], "match_id": "extra"})
    elif mutation == "duplicate_match":
        runs[0]["predictions"][0] = deepcopy(runs[0]["predictions"][1])
    elif mutation == "one_league":
        for row in runs[0]["predictions"]:
            row["competition"] = "E0"
    elif mutation == "late_period":
        runs[0]["start"] = "2025-08-01"
        runs[0]["end_exclusive"] = "2026-07-01"
    elif mutation == "late_prediction":
        runs[0]["predictions"][0]["date"] = "2025-08-01"
    elif mutation == "bad_anchor":
        runs[0]["predictions"][0]["fit_before_date"] = "2024-09-27"
    elif mutation == "bad_interval":
        runs[0]["update_interval_days"] = 7
    elif mutation == "inconsistent_actual":
        runs[0]["predictions"][0]["actual_score"] = [2, 0]
    elif mutation == "inconsistent_market":
        runs[0]["predictions"][0]["market_prices"] = {"avg_preclosing": [2., 3., 4.]}
    elif mutation == "negative_probability":
        runs[0]["predictions"][0]["probabilities"] = [1.1, -.1, 0]
    elif mutation == "future_training":
        runs[0]["predictions"][0]["training_max_date"] = "2024-09-26"
    with pytest.raises(ValueError):
        core.select_policy(runs, expected_ids())


@pytest.mark.parametrize("ids", [set(), {""}, {False}, {"one-other-id"}])
def test_selection_requires_valid_expected_ids(ids):
    with pytest.raises(ValueError):
        core.select_policy(validation_runs(), ids)


def test_real_poisson_fit_changes_rates_and_matches_direct_existing_model():
    rows = history()
    config = core.candidate_configs()[0]
    result = core.run_candidate(rows, config, date(2024, 8, 1), date(2024, 8, 8))
    assert len(result["predictions"]) == 7 and not result["failures"]
    earlier = [r for r in rows if r.match_date < date(2024, 8, 1)]
    fitted = DateTeamGoalModel(.001, 90).fit(earlier, date(2024, 8, 1))
    assert result["fits"][0] == fitted.parameters()
    first = rows[31]
    direct = summarize(score_grid(*fitted.predict(first.home, first.away, "E0", first.match_date), rho=0))
    assert result["predictions"][0]["forecast"] == direct
    assert result["predictions"][0]["rates"] == list(fitted.predict(
        first.home, first.away, "E0", first.match_date))
    regularized = DateTeamGoalModel(.1, 90).fit(earlier, date(2024, 8, 1))
    assert regularized.predict(first.home, first.away, "E0", first.match_date) != fitted.predict(
        first.home, first.away, "E0", first.match_date)
    assert result["fits"][0]["basis"] == "team_marginal"
    assert result["fits"][0]["rho"] == 0
