"""Past-only one-shared-common-shock fitting, nulls and failure traces."""

from copy import deepcopy
from datetime import date
import inspect
import json
import math
import warnings

import numpy as np
import pytest

from football_model import bivariate_poisson_fit as fitting
from football_model.bivariate_poisson import logpmf
from football_model.cmp_forward_experiment import TRAINING_START, WINDOWS, window_rows
from test_cmp_fit import analytic_poisson_objective, rows
from test_cmp_forward_experiment import archive_rows


def fit(records=None, *, training_start="2024-08-01", before_date="2025-07-01"):
    return fitting.fit_window(rows() if records is None else records,
                              training_start=training_start, before_date=before_date)


def test_explicit_keyword_only_window_api():
    signature = inspect.signature(fitting.fit_window)
    assert list(signature.parameters) == ["rows", "training_start", "before_date"]
    assert all(signature.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
               for key in ("training_start", "before_date"))
    with pytest.raises(TypeError):
        fitting.fit_window(rows(), "2024-08-01", "2025-07-01")


def test_equal_match_joint_objective_and_all_original_raw_means_are_retained():
    records = rows()
    unchanged = deepcopy(records)
    result = fit(records)
    assert records == unchanged
    assert result["success"] and result["bounds"] == [0., .95]
    assert 0. <= result["k"] <= .95
    assert result["n_matches"] == len(records) == 6
    assert result["new_team_fits"] == 0 and result["market_blending"] is False
    assert result["means_changed"] is False and result["cmp_stacked"] is False
    assert result["poisson_marginals_preserved"] is True
    assert result["negative_dependence_supported"] is False
    assert result["shared_parameter"] == "one_common_shock_fraction_k_for_all_leagues_and_both_sides"
    assert result["common_component_definition"] == "c=k*min(mu_home,mu_away)"
    assert result["pearson_correlation_definition"] == "k*min(mu_home,mu_away)/sqrt(mu_home*mu_away)"
    assert result["objective_definition"] == "equal_match_mean_negative_joint_exact_score_logpmf"
    assert result["objective_null"] == result["objective_poisson"]
    assert result["objective_null"] == pytest.approx(analytic_poisson_objective(records), abs=2e-12)
    expected = -math.fsum(logpmf(*row["actual_score"], *row["rates"], result["k"])
                          for row in records) / len(records)
    assert result["objective_opt"] == pytest.approx(expected, abs=2e-12)
    assert result["objective_opt"] <= result["objective_null"]
    assert result["training_match_ids"] == [row["match_id"] for row in records]
    assert result["training_min_date"] == "2024-08-01"
    assert result["training_max_date"] == "2024-08-06"
    assert result["training_start"] == "2024-08-01"
    assert result["before_date"] == result["training_end_exclusive"] == "2025-07-01"
    json.dumps(result, allow_nan=False)


def test_mesh_null_exact_endpoints_and_actual_numerical_trace_are_reported():
    result = fit()
    history = result["history"]
    assert [(e["stage"], e["k"]) for e in history[:2]] == [("independence_null", 0.), ("upper_bound", .95)]
    assert [e["k"] for e in history[2:20]] == list(fitting.MESH[1:-1])
    assert [e["evaluation"] for e in history] == list(range(1, len(history) + 1))
    assert all(e["numerical_status"] == "finite" and math.isfinite(e["nll"]) for e in history)
    best = min(history, key=lambda entry: (entry["nll"], entry["k"]))
    assert result["k"] == best["k"] and result["objective_opt"] == best["nll"]
    optimizer = result["optimizer"]
    assert optimizer["method"] == "predetermined_mesh_plus_scipy_bounded_brackets"
    assert optimizer["mesh"] == list(fitting.MESH)
    assert optimizer["mesh_nll"][0] == result["objective_null"]
    assert optimizer["global_optimum_certified"] is False
    assert [r["bracket"] for r in optimizer["runs"][:3]] == ["full_range", "lower_adjacent", "upper_adjacent"]
    assert optimizer["xatol"] == fitting.XATOL
    assert optimizer["maxiter"] == fitting.MAXITER
    assert result["nfev"] == len(history) == len(fitting.MESH) + optimizer["nfev"]


def test_reordering_and_iterator_input_change_neither_parameter_nor_trace():
    assert fit() == fit(reversed(rows()))


def test_known_positively_correlated_simulation_recovers_shared_fraction():
    rng = np.random.default_rng(20261023)
    n, truth = 600, .55
    common = truth * 1.2
    w = rng.poisson(common, n)
    scores = np.column_stack((rng.poisson(1.2-common, n) + w,
                             rng.poisson(1.6-common, n) + w))
    result = fit(rows(scores))
    assert .35 < result["k"] < .75
    assert result["boundary"] is None and result["at_boundary"] is False
    assert result["objective_opt"] < result["objective_null"] - .03


def test_independent_poisson_simulation_can_choose_boundary_or_small_positive_k():
    rng = np.random.default_rng(20261024)
    scores = np.column_stack((rng.poisson(1.2, 600), rng.poisson(1.6, 600)))
    result = fit(rows(scores))
    assert 0. <= result["k"] < .2
    assert result["objective_opt"] <= result["objective_null"]


def test_negative_association_selects_exact_independence_null_not_negative_imputation():
    result = fit(rows([(0, 2), (2, 0)] * 12, rates=(1., 1.)))
    assert result["k"] == 0.
    assert result["at_boundary"] is True and result["boundary"] == "lower"
    assert result["objective_opt"] == result["objective_null"]
    assert result["negative_dependence_supported"] is False


def test_upper_boundary_is_explicit_not_interior_fit_clipped_post_hoc():
    result = fit(rows([(1, 1)] * 10, rates=(1., 1.)))
    assert result["k"] == .95
    assert result["at_boundary"] is True and result["boundary"] == "upper"
    assert any(e["k"] == .95 and e["stage"] == "upper_bound" and e["nll"] == result["objective_opt"]
               for e in result["history"])


@pytest.mark.parametrize("fold_index", [0, 2, 4])
def test_real_fit_and_trace_unchanged_when_later_target_scores_are_perturbed(fold_index):
    records = archive_rows()
    cutoff, end = WINDOWS[fold_index]
    original_training, original_target = window_rows(records, cutoff, end)
    first = fitting.fit_window(original_training, training_start=TRAINING_START, before_date=cutoff)
    perturbed = deepcopy(records)
    for row in perturbed:
        if row["date"] >= cutoff:
            home, away = row["actual_score"]
            row["actual_score"] = [home + 7, away + 1]
            row["actual_outcome"] = 0 if home + 7 > away + 1 else 2 if home + 7 < away + 1 else 1
    new_training, new_target = window_rows(perturbed, cutoff, end)
    assert original_training == new_training and original_target != new_target
    second = fitting.fit_window(new_training, training_start=TRAINING_START, before_date=cutoff)
    assert first == second
    assert first["training_max_date"] < cutoff


@pytest.mark.parametrize("name", ["training_start", "before_date"])
@pytest.mark.parametrize("bad", [None, True, 20250801, date(2025, 8, 1), "20250801", "2025-W31-5",
                                 "2025-8-01", "2025-02-30", "", "2025-08-01T00:00:00"])
def test_window_dates_are_canonical_iso_strings(name, bad):
    options = {"training_start": "2024-08-01", "before_date": "2025-07-01", name: bad}
    with pytest.raises(ValueError, match=name):
        fitting.fit_window(rows(), **options)


@pytest.mark.parametrize("start,end", [("2025-07-01", "2025-07-01"), ("2025-07-02", "2025-07-01")])
def test_empty_or_reversed_window_not_silently_corrected(start, end):
    with pytest.raises(ValueError, match="strictly earlier"):
        fit(training_start=start, before_date=end)


@pytest.mark.parametrize("day", ["2024-07-31", "2025-07-01", "2025-08-01", "2026-01-01"])
def test_out_of_window_rows_reject_whole_fit_instead_of_dropping_future_rows(day):
    records = rows()
    records[0]["date"] = day
    # Keep the original source prediction valid to isolate the fit-window rule.
    if day == "2024-07-31":
        records[0]["fit_before_date"] = "2024-07-31"
        records[0]["training_max_date"] = "2024-07-30"
    with pytest.raises(ValueError, match="training_start <= date < before_date"):
        fit(records)


@pytest.mark.parametrize("change", ["same_day_training", "future_training", "future_cutoff",
                                   "missing_source", "unknown_competition", "same_teams", "missing_rates",
                                   "rates_changed_without_grid", "forecast_changed", "probabilities_changed",
                                   "outcome_conflict", "missing_score", "negative_goal", "bool_goal"])
def test_original_forecast_chronology_identity_and_raw_mean_coherence_remain_required(change):
    records = rows()
    row = records[0]
    if change == "same_day_training":
        row["training_max_date"] = row["fit_before_date"]
    elif change == "future_training":
        row["training_max_date"] = "2025-08-01"
    elif change == "future_cutoff":
        row["fit_before_date"] = "2024-08-02"
    elif change == "missing_source":
        row["source_url"] = ""
    elif change == "unknown_competition":
        row["competition"] = "UNKNOWN"
    elif change == "same_teams":
        row["away"] = row["home"]
    elif change == "missing_rates":
        row.pop("rates")
    elif change == "rates_changed_without_grid":
        row["rates"][0] *= 2
    elif change == "forecast_changed":
        row["forecast"]["total_goals"]["4+"] += .1
    elif change == "probabilities_changed":
        row["probabilities"] = [.4, .2, .4]
    elif change == "outcome_conflict":
        row["actual_outcome"] = 0
    elif change == "missing_score":
        row.pop("actual_score")
    elif change == "negative_goal":
        row["actual_score"] = [-1, 0]
    elif change == "bool_goal":
        row["actual_score"] = [False, False]
    with pytest.raises(ValueError):
        fit(records)


@pytest.mark.parametrize("raw", [[True, 1.], ["1", 1.], [0, 1.], [-1., 1.], [float("nan"), 1.],
                                 [1., float("inf")], [1.], None])
def test_invalid_raw_means_not_imputed_or_coerced(raw):
    records = rows()
    records[0]["rates"] = raw
    with pytest.raises(ValueError):
        fit(records)


def test_empty_or_duplicate_cohort_is_error_not_partial_success():
    with pytest.raises(ValueError, match="nonempty"):
        fit([])
    with pytest.raises(ValueError, match="Duplicate"):
        fit(rows() + rows()[:1])


@pytest.mark.parametrize("invalid", ["shape", "nan", "infinite", "positive"])
def test_invalid_batch_is_failed_trace_not_null_substitution(monkeypatch, invalid):
    def broken(home, away, mu_home, mu_away, k):
        if invalid == "shape":
            return np.full((len(home), 2), -1.)
        values = np.full(len(home), -1.)
        values[0] = {"nan": float("nan"), "infinite": float("inf"), "positive": .01}[invalid]
        return values
    monkeypatch.setattr(fitting, "batch_logpmf", broken)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    error = caught.value
    assert len(error.history) == 1
    assert error.history[0]["stage"] == "independence_null"
    assert error.history[0]["nll"] is None and error.history[0]["numerical_status"] == "error"
    assert error.details["success"] is False
    json.dumps(error.details, allow_nan=False)


def test_boundary_numerical_error_retains_completed_null_and_does_not_fallback(monkeypatch):
    actual = fitting.batch_logpmf
    seen = []
    def broken(*args):
        seen.append(args[-1])
        if args[-1] == .95:
            raise ValueError("deliberate endpoint evaluation failure")
        return actual(*args)
    monkeypatch.setattr(fitting, "batch_logpmf", broken)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    assert seen == [0., .95]
    assert [e["numerical_status"] for e in caught.value.history] == ["finite", "error"]
    assert caught.value.history[-1]["error"] == "deliberate endpoint evaluation failure"


def test_runtime_warning_is_failure_and_warning_filters_restore(monkeypatch):
    def broken(*args):
        warnings.warn("deliberate arithmetic warning", RuntimeWarning)
        return np.full(len(args[0]), -1.)
    filters = list(warnings.filters)
    monkeypatch.setattr(fitting, "batch_logpmf", broken)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    assert caught.value.history[0]["error_type"] == "RuntimeWarning"
    assert warnings.filters == filters


def test_unsuccessful_optimizer_not_replaced_by_mesh_best_or_null(monkeypatch):
    actual = fitting.minimize_scalar
    def failed(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.success, result.status, result.message = False, 1, "deliberate iteration exhaustion"
        return result
    monkeypatch.setattr(fitting, "minimize_scalar", failed)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    assert len(caught.value.history) > 20
    assert caught.value.optimizer["success"] is False
    assert caught.value.optimizer["failed_bracket"] == "full_range"
    assert caught.value.optimizer["runs"][0]["message"] == "deliberate iteration exhaustion"
    json.dumps(caught.value.details, allow_nan=False)


def test_optimizer_exception_retains_mesh_and_stops_without_second_search(monkeypatch):
    calls = []
    def broken(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("deliberate optimizer exception")
    monkeypatch.setattr(fitting, "minimize_scalar", broken)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    assert len(calls) == 1 and calls[0]["bounds"] == (0., .95)
    assert calls[0]["method"] == "bounded"
    assert len(caught.value.history) == 20
    assert caught.value.optimizer["completed_runs"] == []
    assert caught.value.optimizer["failed_bracket"] == "full_range"


@pytest.mark.parametrize("field,value", [("x", float("nan")), ("fun", float("inf")), ("x", 1.1),
                                        ("status", None)])
def test_nonfinite_out_of_bounds_or_malformed_solver_result_has_serializable_failed_trace(monkeypatch, field, value):
    actual = fitting.minimize_scalar
    def broken(*args, **kwargs):
        result = actual(*args, **kwargs)
        result[field] = value
        return result
    monkeypatch.setattr(fitting, "minimize_scalar", broken)
    with pytest.raises(fitting.BivariatePoissonFitError) as caught:
        fit()
    assert caught.value.optimizer["success"] is False
    assert caught.value.optimizer["failed_bracket"] == "full_range"
    json.dumps(caught.value.details, allow_nan=False)


def test_predetermined_mesh_local_minimum_brackets_are_refined_without_concavity_claim(monkeypatch):
    def multimodal(home, away, mu_home, mu_away, k):
        # Two separated valleys. This tests search bookkeeping, not the model's
        # probability formula, which is tested independently against scalar PMF.
        objective = 2. + (k-.22)**2 * (k-.73)**2 + .0001*k
        return np.full(len(home), -objective)
    monkeypatch.setattr(fitting, "batch_logpmf", multimodal)
    result = fit()
    labels = [r["bracket"] for r in result["optimizer"]["runs"]]
    assert "mesh_minimum_4" in labels and "mesh_minimum_15" in labels
    assert abs(result["k"] - .22) < .01
    assert result["optimizer"]["global_optimum_certified"] is False


def test_single_legal_match_is_reported_as_one_match_not_five_leagues():
    result = fit(rows()[:1])
    assert result["n_matches"] == 1
    assert result["training_match_ids"] == [rows()[0]["match_id"]]


def test_repeated_independent_matches_keep_equal_match_objective_not_total_sum():
    records = rows()
    extra = deepcopy(records)
    for row in extra:
        row["match_id"] += "-separate-match"
    first, second = fit(records), fit(records + extra)
    assert second["n_matches"] == 2 * first["n_matches"]
    assert first["objective_null"] == pytest.approx(second["objective_null"], abs=2e-12)
    assert first["objective_opt"] == pytest.approx(second["objective_opt"], abs=2e-12)
    assert first["k"] == pytest.approx(second["k"], abs=1e-8)
