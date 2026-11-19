"""Explicit earlier-window CMP fits preserve legacy fixed-season behavior."""

from copy import deepcopy
from datetime import date
import inspect
import json
import math

import numpy as np
import pytest

from football_model import cmp_fit as fitting
from football_model.mean_cmp import batch_logpmf
from test_cmp_fit import rows as earlier_rows
from test_goal_rate_experiment import prediction


def window_rows():
    days = ("2025-08-01", "2025-08-08", "2025-09-05", "2025-09-30")
    scores = ((0, 0), (1, 2), (2, 1), (3, 1))
    rates = ((.8, 1.1), (1.4, 1.2), (1.9, 1.7), (2.2, 1.3))
    records = []
    for i, (day, score, raw) in enumerate(zip(days, scores, rates)):
        row = prediction(f"window-{i}", day, score, ("E0", "D1", "I1", "SP1")[i], rates=raw)
        row["rates"] = list(raw)
        records.append(row)
    return records


def fit(records=None, *, training_start="2025-08-01", before_date="2025-10-01"):
    return fitting.fit_cmp_window(window_rows() if records is None else records,
                                  training_start=training_start, before_date=before_date)


def test_new_window_api_has_explicit_keyword_only_bounds():
    signature = inspect.signature(fitting.fit_cmp_window)
    assert list(signature.parameters) == ["rows", "training_start", "before_date"]
    assert all(signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
               for name in ("training_start", "before_date"))
    with pytest.raises(TypeError):
        fitting.fit_cmp_window(window_rows(), "2025-08-01", "2025-10-01")


def test_window_fits_one_shared_parameter_with_original_equal_match_objective():
    records = window_rows()
    before = deepcopy(records)
    result = fit(records)
    assert records == before
    assert result["training_start"] == "2025-08-01"
    assert result["before_date"] == result["training_end_exclusive"] == "2025-10-01"
    assert result["training_min_date"] == "2025-08-01"
    assert result["training_max_date"] == "2025-09-30"
    assert result["training_match_ids"] == [row["match_id"] for row in records]
    assert result["n_matches"] == 4 and result["n_marginals"] == 8
    assert result["bounds"] == [.25, 4.]
    assert result["success"] and result["means_changed"] is False
    assert result["new_team_fits"] == 0 and result["market_blending"] is False
    assert result["shared_parameter"] == "one_nu_for_all_leagues_and_both_sides"
    means = np.asarray([row["rates"] for row in records])
    goals = np.asarray([row["actual_score"] for row in records])
    expected = -batch_logpmf(means, goals, result["nu"]).sum(axis=1).mean()
    assert result["objective_opt"] == pytest.approx(expected, abs=2e-12)
    poisson = math.fsum(math.fsum(rate - goal * math.log(rate) + math.lgamma(goal + 1)
                                  for goal, rate in zip(row["actual_score"], row["rates"]))
                       for row in records) / len(records)
    assert result["objective_poisson"] == pytest.approx(poisson, abs=2e-12)
    assert result["objective_opt"] <= result["objective_poisson"]
    json.dumps(result, allow_nan=False)


def test_legacy_and_same_explicit_window_have_identical_numerical_fit_and_trace():
    records = earlier_rows()
    legacy = fitting.fit_cmp(records)
    window = fitting.fit_cmp_window(records, training_start="2024-08-01", before_date="2025-07-01")
    assert {key: value for key, value in legacy.items() if key != "before_date"} == {
        key: value for key, value in window.items()
        if key not in ("before_date", "training_start", "training_end_exclusive")}
    assert legacy["before_date"] == "2025-08-01"
    assert window["before_date"] == "2025-07-01"


def test_legacy_does_not_gain_july_rows_when_window_api_is_added():
    row = prediction("july-only", "2025-07-15", (1, 2), rates=(1.2, 1.6))
    row["rates"] = [1.2, 1.6]
    window = fitting.fit_cmp_window([row], training_start="2025-07-01", before_date="2025-08-01")
    assert window["n_matches"] == 1
    with pytest.raises(ValueError, match="2024/25"):
        fitting.fit_cmp([row])
    with pytest.raises(ValueError, match="2025-08-01"):
        fitting.fit_cmp(earlier_rows(), before_date="2025-10-01")


@pytest.mark.parametrize("name", ["training_start", "before_date"])
@pytest.mark.parametrize("bad", [None, True, 20250801, date(2025, 8, 1), "20250801",
                                 "2025-W31-5", "2025-8-01", "2025-02-30", "", "2025-08-01T00:00:00"])
def test_window_bounds_require_canonical_iso_date_strings(name, bad):
    values = {"training_start": "2025-08-01", "before_date": "2025-10-01", name: bad}
    with pytest.raises(ValueError, match=name):
        fitting.fit_cmp_window(window_rows(), **values)


@pytest.mark.parametrize("start,end", [("2025-10-01", "2025-10-01"), ("2025-10-02", "2025-10-01")])
def test_empty_or_reversed_declared_window_is_not_silently_changed(start, end):
    with pytest.raises(ValueError, match="strictly earlier"):
        fit(training_start=start, before_date=end)


@pytest.mark.parametrize("day", ["2025-07-31", "2025-10-01", "2025-10-02", "2026-02-01"])
def test_one_out_of_window_row_rejects_the_whole_fit_not_a_filtered_cohort(day):
    records = window_rows()
    row = prediction("outside", day, (0, 1), rates=(1.2, 1.6))
    row["rates"] = [1.2, 1.6]
    records.append(row)
    with pytest.raises(ValueError, match="training_start <= date < before_date"):
        fit(records)


def test_inclusive_start_exclusive_end_and_sparse_cohort_are_reported_literally():
    records = [window_rows()[0], window_rows()[-1]]
    result = fit(records)
    assert result["n_matches"] == 2
    assert result["training_start"] == result["training_min_date"] == "2025-08-01"
    assert result["training_end_exclusive"] == "2025-10-01"
    assert result["training_max_date"] == "2025-09-30"


@pytest.mark.parametrize("change", ["same_day_training", "future_training", "future_cutoff",
                                   "unknown_competition", "same_teams", "missing_source", "missing_rates",
                                   "rates_changed_without_grid", "forecast_changed", "probability_changed",
                                   "outcome_conflict", "missing_actual_score", "negative_goal"])
def test_window_retains_original_forecast_identity_chronology_and_mean_validation(change):
    records = window_rows()
    row = records[0]
    if change == "same_day_training":
        row["training_max_date"] = row["fit_before_date"]
    elif change == "future_training":
        row["training_max_date"] = "2025-10-01"
    elif change == "future_cutoff":
        row["fit_before_date"] = "2025-08-02"
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
    elif change == "probability_changed":
        row["probabilities"] = [.4, .2, .4]
    elif change == "outcome_conflict":
        row["actual_outcome"] = 0
    elif change == "missing_actual_score":
        row.pop("actual_score")
    elif change == "negative_goal":
        row["actual_score"] = [-1, 0]
    with pytest.raises(ValueError):
        fit(records)


@pytest.mark.parametrize("raw", [[True, 1.], ["1", 1.], [0, 1.], [-1., 1.],
                                 [float("nan"), 1.], [1., float("inf")], [1.], None])
def test_window_does_not_impute_or_coerce_invalid_raw_rates(raw):
    records = window_rows()
    records[0]["rates"] = raw
    with pytest.raises(ValueError):
        fit(records)


def test_empty_and_duplicate_cohorts_are_errors_not_partial_successes():
    with pytest.raises(ValueError, match="nonempty"):
        fit([])
    with pytest.raises(ValueError, match="Duplicate"):
        fit(window_rows() + window_rows()[:1])


def test_window_reordering_or_generator_input_does_not_change_result():
    result = fit()
    assert result == fit(reversed(window_rows()))


def test_window_uses_same_poisson_endpoint_and_optimizer_history_without_new_parameters():
    result = fit()
    history = result["history"]
    assert [entry["nu"] for entry in history[:3]] == [1., .25, 4.]
    assert [entry["stage"] for entry in history[:3]] == ["poisson_reference", "lower_bound", "upper_bound"]
    assert result["nfev"] == len(history) == result["optimizer"]["nfev"] + 3
    assert all(entry["numerical_status"] == "finite" for entry in history)
    assert result["optimizer"]["method"] == "scipy.optimize.minimize_scalar_bounded"
    assert result["optimizer"]["xatol"] == fitting.XATOL
    assert result["optimizer"]["maxiter"] == fitting.MAXITER


def test_window_numerical_failure_preserves_attempts_and_does_not_fallback(monkeypatch):
    actual = fitting.batch_logpmf
    seen = []
    def fail_on_endpoint(means, goals, nu):
        seen.append((means.copy(), goals.copy(), nu))
        if nu == .25:
            raise ValueError("deliberate window mean inversion failure")
        return actual(means, goals, nu)
    monkeypatch.setattr(fitting, "batch_logpmf", fail_on_endpoint)
    with pytest.raises(fitting.CMPFitError) as caught:
        fit()
    assert [entry[2] for entry in seen] == [1., .25]
    assert np.array_equal(seen[0][0], [row["rates"] for row in window_rows()])
    assert np.array_equal(seen[0][1], [row["actual_score"] for row in window_rows()])
    assert [entry["numerical_status"] for entry in caught.value.history] == ["finite", "error"]
    assert caught.value.history[-1]["error"] == "deliberate window mean inversion failure"
    assert caught.value.details["success"] is False
    json.dumps(caught.value.details, allow_nan=False)
