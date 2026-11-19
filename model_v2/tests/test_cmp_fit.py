"""Earlier-only, one-parameter fixed-mean CMP fitting and failure traces."""

from copy import deepcopy
from datetime import date, timedelta
import inspect
import json
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from football_model import cmp_fit as fitting
from football_model.football_data import LEAGUES
from football_model.mean_cmp import batch_logpmf
from test_goal_rate_experiment import prediction


def rows(scores=None, rates=(1.2, 1.6)):
    if scores is None:
        scores = [(0, 0), (1, 0), (0, 2), (3, 1), (2, 1), (1, 2)]
    result = []
    for i, score in enumerate(scores):
        day = (date(2024, 8, 1) + timedelta(days=i % 300)).isoformat()
        row = prediction(f"earlier-{i}", day, tuple(map(int, score)), list(LEAGUES)[i % 5], 28, rates)
        row["rates"] = list(rates)
        result.append(row)
    return result


def analytic_poisson_objective(records):
    return math.fsum(math.fsum(rate - goal*math.log(rate) + math.lgamma(goal+1)
                               for goal, rate in zip(row["actual_score"], row["rates"]))
                     for row in records) / len(records)


def test_real_fit_uses_one_nu_equal_match_joint_objective_and_preserves_all_means():
    records = rows()
    before = deepcopy(records)
    result = fitting.fit_cmp(records)
    assert records == before
    assert result["success"] and result["means_changed"] is False
    assert result["n_matches"] == 6 and result["n_marginals"] == 12
    assert result["before_date"] == "2025-08-01"
    assert result["bounds"] == [.25, 4.]
    assert .25 <= result["nu"] <= 4.
    assert result["new_team_fits"] == 0 and result["market_blending"] is False
    assert result["objective_poisson"] == pytest.approx(analytic_poisson_objective(records), abs=2e-12)
    means = np.asarray([row["rates"] for row in records])
    goals = np.asarray([row["actual_score"] for row in records])
    expected = -batch_logpmf(means, goals, result["nu"]).sum(axis=1).mean()
    assert result["objective_opt"] == pytest.approx(expected, abs=2e-12)
    assert result["objective_opt"] <= result["objective_poisson"]
    assert result["training_match_ids"] == [row["match_id"] for row in records]
    assert result["training_min_date"] == "2024-08-01" and result["training_max_date"] == "2024-08-06"
    assert result["shared_parameter"] == "one_nu_for_all_leagues_and_both_sides"
    assert list(inspect.signature(fitting.fit_cmp).parameters) == ["rows", "before_date"]
    json.dumps(result, allow_nan=False)


def test_numerical_history_includes_poisson_exact_endpoints_and_actual_optimizer_evaluations():
    result = fitting.fit_cmp(rows())
    history = result["history"]
    assert [entry["stage"] for entry in history[:3]] == ["poisson_reference", "lower_bound", "upper_bound"]
    assert [entry["nu"] for entry in history[:3]] == [1., .25, 4.]
    assert all(entry["numerical_status"] == "finite" and math.isfinite(entry["nll"]) for entry in history)
    assert result["nfev"] == len(history) == result["optimizer"]["nfev"] + 3
    assert [entry["evaluation"] for entry in history] == list(range(1, len(history)+1))
    best = min(history, key=lambda entry: (entry["nll"], entry["nu"]))
    assert result["nu"] == best["nu"] and result["objective_opt"] == best["nll"]


def test_known_poisson_simulation_recovers_near_one_not_necessarily_exactly_one():
    rng = np.random.default_rng(102306)
    scores = np.column_stack((rng.poisson(1.2, 600), rng.poisson(1.6, 600)))
    result = fitting.fit_cmp(rows(scores))
    assert .8 < result["nu"] < 1.2
    assert not result["at_boundary"]
    assert result["objective_opt"] <= result["objective_poisson"]


def test_underdispersed_binomial_simulation_fits_nu_above_one():
    rng = np.random.default_rng(102307)
    scores = np.column_stack((rng.binomial(4, .3, 450), rng.binomial(5, .32, 450)))
    result = fitting.fit_cmp(rows(scores))
    assert result["nu"] > 1.05
    assert result["objective_opt"] < result["objective_poisson"]


def test_real_upper_boundary_is_explicit_not_an_interior_value_clipped_afterward():
    records = rows([(1, 1)] * 20, rates=(1., 1.))
    result = fitting.fit_cmp(records)
    assert result["nu"] == 4.
    assert result["at_boundary"] is True and result["boundary"] == "upper"
    assert result["optimizer"]["x"] < 4.
    assert any(entry["stage"] == "upper_bound" and entry["nu"] == result["nu"]
               and entry["nll"] == result["objective_opt"] for entry in result["history"])


def test_reordering_rows_changes_neither_nu_nor_equal_match_objective():
    records = rows()
    original = fitting.fit_cmp(records)
    reordered = fitting.fit_cmp(list(reversed(records)))
    assert original == reordered


@pytest.mark.parametrize("before", ["2025-07-01", "2025-08-02", "2026-08-01", date(2025, 8, 1), None])
def test_only_fixed_earlier_selection_cutoff_is_accepted(before):
    with pytest.raises(ValueError, match="2025-08-01"):
        fitting.fit_cmp(rows(), before_date=before)


@pytest.mark.parametrize("day", ["2024-07-31", "2025-07-01", "2025-08-01", "2026-01-01"])
def test_later_or_out_of_earlier_season_rows_cannot_enter_dispersion_training(day):
    records = rows()
    row = prediction("outside", day, (1, 0), rates=(1.2, 1.6))
    row["rates"] = [1.2, 1.6]
    records.append(row)
    with pytest.raises(ValueError, match="2024/25"):
        fitting.fit_cmp(records)


@pytest.mark.parametrize("change", ["same_day_training", "future_training", "future_cutoff",
                                   "unknown_competition", "same_teams", "missing_source", "missing_rates",
                                   "rates_changed_without_grid", "forecast_changed", "outcome_conflict"])
def test_invalid_unknown_or_non_past_prediction_information_is_rejected(change):
    records = rows()
    row = records[0]
    if change == "same_day_training":
        row["training_max_date"] = row["fit_before_date"]
    elif change == "future_training":
        row["training_max_date"] = "2025-08-01"
    elif change == "future_cutoff":
        row["fit_before_date"] = "2024-08-02"
    elif change == "unknown_competition":
        row["competition"] = "UNKNOWN"
    elif change == "same_teams":
        row["away"] = row["home"]
    elif change == "missing_source":
        row["source_url"] = ""
    elif change == "missing_rates":
        row.pop("rates")
    elif change == "rates_changed_without_grid":
        row["rates"][0] *= 2
    elif change == "forecast_changed":
        row["forecast"]["total_goals"]["4+"] += .1
    elif change == "outcome_conflict":
        row["actual_outcome"] = 2
    with pytest.raises(ValueError):
        fitting.fit_cmp(records)


@pytest.mark.parametrize("rates", [[True, 1.], ["1", 1.], [0, 1.], [-1., 1.],
                                   [float("nan"), 1.], [1., float("inf")], [1.], None])
def test_raw_mean_validation_does_not_coerce_invalid_values_or_fill_unknowns(rates):
    records = rows()
    records[0]["rates"] = rates
    with pytest.raises(ValueError):
        fitting.fit_cmp(records)


def test_empty_or_duplicate_matches_fail_without_dropping_any_records():
    for records in ([], rows() + rows()[:1]):
        with pytest.raises(ValueError, match="nonempty|Duplicate"):
            fitting.fit_cmp(records)


@pytest.mark.parametrize("invalid", ["wrong_shape", "nan", "infinity", "positive"])
def test_numerically_invalid_batch_output_is_preserved_as_a_failed_trace(monkeypatch, invalid):
    def broken(means, goals, nu):
        if invalid == "wrong_shape":
            return np.full(len(means), -1.)
        values = np.full(means.shape, -1.)
        values[0, 0] = {"nan": float("nan"), "infinity": float("inf"), "positive": .01}[invalid]
        return values
    monkeypatch.setattr(fitting, "batch_logpmf", broken)
    with pytest.raises(fitting.CMPFitError) as caught:
        fitting.fit_cmp(rows())
    error = caught.value
    assert len(error.history) == 1
    assert error.history[0]["stage"] == "poisson_reference"
    assert error.history[0]["numerical_status"] == "error"
    assert error.history[0]["nll"] is None
    assert error.details["success"] is False
    json.dumps(error.details, allow_nan=False)


def test_true_solver_failure_after_poisson_keeps_completed_reference_and_does_not_fallback(monkeypatch):
    actual = fitting.batch_logpmf
    calls = []
    def fail_boundary(means, goals, nu):
        calls.append(nu)
        if nu != 1.:
            raise ValueError("deliberate mean inversion failure")
        return actual(means, goals, nu)
    monkeypatch.setattr(fitting, "batch_logpmf", fail_boundary)
    with pytest.raises(fitting.CMPFitError) as caught:
        fitting.fit_cmp(rows())
    assert calls == [1., .25]
    assert [entry["numerical_status"] for entry in caught.value.history] == ["finite", "error"]
    assert caught.value.history[1]["error"] == "deliberate mean inversion failure"


def test_runtime_warning_from_numerics_is_an_explicit_failure_and_warning_filters_restore(monkeypatch):
    def warn(means, goals, nu):
        warnings.warn("arithmetic failed", RuntimeWarning)
        return np.full(means.shape, -1.)
    monkeypatch.setattr(fitting, "batch_logpmf", warn)
    filters = list(warnings.filters)
    with pytest.raises(fitting.CMPFitError) as caught:
        fitting.fit_cmp(rows())
    assert caught.value.history[0]["error_type"] == "RuntimeWarning"
    assert warnings.filters == filters


def test_unsuccessful_optimizer_is_not_replaced_by_best_prior_trace_or_poisson(monkeypatch):
    actual = fitting.minimize_scalar
    def failed(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.success = False
        result.status = 1
        result.message = "deliberate iteration exhaustion"
        return result
    monkeypatch.setattr(fitting, "minimize_scalar", failed)
    with pytest.raises(fitting.CMPFitError) as caught:
        fitting.fit_cmp(rows())
    assert len(caught.value.history) > 3
    assert caught.value.optimizer["success"] is False
    assert "iteration exhaustion" in caught.value.optimizer["message"]
    assert caught.value.details["success"] is False


def test_optimizer_exception_is_recorded_without_second_solver_or_later_data(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("deliberate bounded search error")
    monkeypatch.setattr(fitting, "minimize_scalar", fail)
    with pytest.raises(fitting.CMPFitError) as caught:
        fitting.fit_cmp(rows())
    assert len(calls) == 1 and calls[0]["bounds"] == (.25, 4.) and calls[0]["method"] == "bounded"
    assert len(caught.value.history) == 3
    assert all(entry["numerical_status"] == "finite" for entry in caught.value.history)


def test_one_legal_league_sample_is_supported_without_fabricating_five_league_coverage():
    records = rows()[:1]
    result = fitting.fit_cmp(records)
    assert result["n_matches"] == 1 and result["n_marginals"] == 2
    assert result["training_match_ids"] == [records[0]["match_id"]]


@pytest.mark.parametrize("mean", [.8, 1.8])
def test_identical_repeated_matches_do_not_change_fit_weight_or_double_objective(mean):
    base = rows([(0, 1), (1, 0), (2, 1)], rates=(mean, 1.2))
    duplicate = deepcopy(base)
    for row in duplicate:
        row["match_id"] += "-independent-repetition"
    first = fitting.fit_cmp(base)
    second = fitting.fit_cmp(base + duplicate)
    assert second["n_matches"] == 2 * first["n_matches"]
    assert second["objective_poisson"] == pytest.approx(first["objective_poisson"], abs=2e-12)
    assert second["objective_opt"] == pytest.approx(first["objective_opt"], abs=2e-12)
    assert second["nu"] == pytest.approx(first["nu"], abs=1e-8)
