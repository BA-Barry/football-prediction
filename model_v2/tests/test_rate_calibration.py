"""Closed-form rate scaling, exact-score objective and temporal application."""

from copy import deepcopy
from datetime import date, datetime, timezone
import json
import math

import numpy as np
import pytest

import football_model.rate_calibration as calibration


CUTOFF = date(2025, 8, 1)


def training_rows():
    return [
        {"match_id": "later", "date": "2025-05-01", "actual_score": [3, 1], "rates": [1.4, 1.6]},
        {"match_id": "earlier", "date": "2024-08-10", "actual_score": [1, 2], "rates": [1.1, .8]},
        {"match_id": "middle", "date": "2025-02-02", "actual_score": [0, 3], "rates": [2.1, 1.3]},
    ]


def analytic_joint_nll(rows, scales=(1., 1.)):
    return sum(rate * scale - goal * math.log(rate * scale) + math.lgamma(goal + 1)
               for row in rows for goal, rate, scale in zip(row["actual_score"], row["rates"], scales))


def fitted(method="shared_scale"):
    return calibration.fit_rate_calibrator(training_rows(), CUTOFF, method)


@pytest.mark.parametrize("method", calibration.METHODS)
def test_exact_closed_form_formula_sum_mean_objective_and_metadata(method):
    rows = training_rows()
    before = deepcopy(rows)
    fit = calibration.fit_rate_calibrator(rows, CUTOFF, method)
    goals = np.asarray([r["actual_score"] for r in rows]).sum(axis=0)
    rates = np.asarray([r["rates"] for r in rows]).sum(axis=0)
    scales = np.repeat(goals.sum() / rates.sum(), 2) if method == "shared_scale" else goals / rates
    assert fit["method"] == method
    assert fit["parameters"]["scales"] == pytest.approx(scales.tolist())
    assert fit["training_goal_sums"] == goals.tolist()
    assert fit["training_rate_sums"] == pytest.approx(rates.tolist())
    assert fit["training_match_ids"] == ["earlier", "middle", "later"]
    assert fit["training_min_date"] == "2024-08-10"
    assert fit["training_max_date"] == "2025-05-01"
    assert fit["before_date"] == "2025-08-01" and fit["n_matches"] == 3
    assert fit["model_type"] == "independent_poisson_rate_scale_calibrator"
    assert fit["basis"] == "team_marginal" and fit["rho"] == 0
    assert fit["estimation"] == {
        "algorithm": "closed_form", "objective": "equal_match_weight_independent_poisson_joint_score_nll",
        "optimizer": None, "regularization": None, "market_inputs": False}
    for field, expected in (("raw_training_analytic_joint_score_nll", analytic_joint_nll(rows)),
                            ("calibrated_training_analytic_joint_score_nll", analytic_joint_nll(rows, scales))):
        assert fit[field]["sum"] == pytest.approx(expected)
        assert fit[field]["mean"] == pytest.approx(expected / 3)
    assert fit["calibrated_training_analytic_joint_score_nll"]["sum"] <= fit["raw_training_analytic_joint_score_nll"]["sum"]
    residuals = goals - rates * scales
    assert fit["score_equation_residual_sums"] == pytest.approx(
        {"home": residuals[0], "away": residuals[1], "shared": residuals.sum()}, abs=1e-12)
    assert rows == before
    assert json.loads(json.dumps(fit, allow_nan=False)) == fit


def test_known_shared_and_side_scales_are_recovered_without_grid_search():
    shared_rows = [
        {"match_id": "a", "date": "2024-08-10", "actual_score": [4, 2], "rates": [2., 1.]},
        {"match_id": "b", "date": "2024-08-11", "actual_score": [2, 6], "rates": [1., 3.]},
    ]
    assert calibration.fit_rate_calibrator(shared_rows, CUTOFF, "shared_scale")["parameters"]["scales"] == [2., 2.]
    side_rows = [
        {"match_id": "a", "date": "2024-08-10", "actual_score": [3, 3], "rates": [2., 4.]},
        {"match_id": "b", "date": "2024-08-11", "actual_score": [6, 6], "rates": [4., 8.]},
    ]
    fit = calibration.fit_rate_calibrator(side_rows, CUTOFF, "side_scale")
    assert fit["parameters"]["scales"] == [1.5, .75]
    assert fit["score_equation_residual_sums"] == {"home": 0., "away": 0., "shared": 0.}


def test_shared_score_equation_does_not_require_each_side_residual_to_vanish():
    rows = [{"match_id": "a", "date": "2025-05-01", "actual_score": [6, 3], "rates": [2., 4.]}]
    fit = calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")
    assert fit["parameters"]["scales"] == [1.5, 1.5]
    assert fit["score_equation_residual_sums"] == {"home": 3., "away": -3., "shared": 0.}
    np.testing.assert_array_equal(calibration.apply_rate_calibrator([2., 4.], fit, CUTOFF), [3., 6.])


@pytest.mark.parametrize("method", calibration.METHODS)
def test_closed_form_is_the_nll_minimum_against_predefined_perturbations(method):
    rows, fit = training_rows(), fitted(method)
    scales = np.asarray(fit["parameters"]["scales"])
    minimum = analytic_joint_nll(rows, scales)
    for multiplier in (.5, .9, 1.1, 2.):
        assert analytic_joint_nll(rows, scales * multiplier) > minimum
    if method == "side_scale":
        for side in (0, 1):
            changed = scales.copy()
            changed[side] *= 1.1
            assert analytic_joint_nll(rows, changed) > minimum


def test_rows_are_equally_weighted_not_means_of_individual_observed_expected_ratios():
    rows = [
        {"match_id": "small", "date": "2024-08-10", "actual_score": [4, 1], "rates": [.1, .1]},
        {"match_id": "large", "date": "2024-08-11", "actual_score": [4, 1], "rates": [3.9, 1.9]},
    ]
    fit = calibration.fit_rate_calibrator(rows, CUTOFF, "side_scale")
    assert fit["parameters"]["scales"] == [2., 1.]
    assert fit["parameters"]["scales"][0] != pytest.approx(np.mean([40., 4 / 3.9]))


def test_irrelevant_market_and_future_label_fields_are_not_model_inputs():
    rows = training_rows()
    expected = calibration.fit_rate_calibrator(rows, CUTOFF, "side_scale")
    for row in rows:
        row.update(market_probabilities=[float("nan"), 1, 0], future_outcome=999, latest_lineup="unknown")
    assert calibration.fit_rate_calibrator(rows, CUTOFF, "side_scale") == expected


def test_appending_fit_day_or_future_results_is_an_error_not_silent_filtering():
    rows = training_rows()
    for day in ("2025-08-01", "2026-05-01"):
        future = {"match_id": "future", "date": day, "actual_score": [9, 8], "rates": [1., 1.]}
        with pytest.raises(ValueError, match="strictly before"):
            calibration.fit_rate_calibrator(rows + [future], CUTOFF, "shared_scale")
    # An already fitted transformation has no later-label argument to refit it.
    fit = fitted()
    unchanged = deepcopy(fit)
    calibration.apply_rate_calibrator([1., 2.], fit, "2026-05-01")
    assert fit == unchanged


def test_training_order_and_date_vs_iso_containers_are_equivalent():
    rows = training_rows()
    expected = calibration.fit_rate_calibrator(rows, CUTOFF, "side_scale")
    for row in rows:
        row["date"] = date.fromisoformat(row["date"])
        row["rates"] = tuple(row["rates"])
        row["actual_score"] = tuple(row["actual_score"])
    assert calibration.fit_rate_calibrator(reversed(rows), CUTOFF, "side_scale") == expected


def test_unknown_method_and_empty_or_duplicate_training_are_refused():
    for method in ("temperature", "", None):
        with pytest.raises(ValueError, match="method"):
            calibration.fit_rate_calibrator(training_rows(), CUTOFF, method)
    for rows in (None, [], [training_rows()[0], training_rows()[0]], [None]):
        with pytest.raises(ValueError):
            calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")


def test_dates_are_real_calendar_dates_not_compact_iso_timestamps_or_datetime():
    for cutoff in ("2025-08-01", datetime(2025, 8, 1), datetime(2025, 8, 1, tzinfo=timezone.utc)):
        with pytest.raises(ValueError, match="before_date"):
            calibration.fit_rate_calibrator(training_rows(), cutoff, "shared_scale")
    for day in ("2024-02-30", "20240810", "2024-08-10T00:00:00", datetime(2024, 8, 10), None):
        rows = training_rows()
        rows[0]["date"] = day
        with pytest.raises(ValueError, match="date"):
            calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")


def test_training_rates_and_scores_are_not_coerced_clipped_or_filled():
    bad_values = [
        ("rates", [0, 1]), ("rates", [-1, 1]), ("rates", [float("nan"), 1]),
        ("rates", [float("inf"), 1]), ("rates", [True, 1.]), ("rates", ["1", 1]),
        ("rates", [[1, 2]]), ("rates", [1]), ("rates", None),
        ("actual_score", [True, 1]), ("actual_score", [1., 1]), ("actual_score", [-1, 1]),
        ("actual_score", [1, float("nan")]), ("actual_score", [1]),
        ("actual_score", None), ("match_id", " "), ("match_id", None),
    ]
    for field, value in bad_values:
        rows = training_rows()
        rows[0][field] = value
        with pytest.raises(ValueError):
            calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")
    for field in ("date", "rates", "actual_score", "match_id"):
        rows = training_rows()
        del rows[0][field]
        with pytest.raises(ValueError):
            calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")


def test_zero_goal_boundary_mle_is_refused_without_epsilon_or_identity_fallback():
    zero = [{"match_id": "a", "date": "2024-08-10", "actual_score": [0, 0], "rates": [1., 1.]}]
    for method in calibration.METHODS:
        with pytest.raises(ValueError, match="no strictly positive finite"):
            calibration.fit_rate_calibrator(zero, CUTOFF, method)
    zero[0]["actual_score"] = [0, 3]
    with pytest.raises(ValueError, match="side"):
        calibration.fit_rate_calibrator(zero, CUTOFF, "side_scale")
    assert calibration.fit_rate_calibrator(zero, CUTOFF, "shared_scale")["parameters"]["scales"] == [1.5, 1.5]


@pytest.mark.parametrize("method", calibration.METHODS)
def test_apply_preserves_vector_or_matrix_shape_and_does_not_mutate_inputs(method):
    fit = fitted(method)
    before_fit = deepcopy(fit)
    rates = np.asarray([[1.2, .7], [2.3, 1.4]])
    before_rates = rates.copy()
    expected = rates * np.asarray(fit["parameters"]["scales"])
    result = calibration.apply_rate_calibrator(rates, fit, CUTOFF)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, expected)
    single = calibration.apply_rate_calibrator(rates[0].tolist(), fit, "2025-08-01")
    assert single.shape == (2,)
    np.testing.assert_array_equal(single, expected[0])
    np.testing.assert_array_equal(rates, before_rates)
    assert fit == before_fit
    assert result is not rates


def test_apply_refuses_all_training_period_dates_and_timestamp_imprecision():
    fit = fitted()
    for day in ("2024-08-10", "2025-05-01", "2025-07-31"):
        with pytest.raises(ValueError, match="on/after"):
            calibration.apply_rate_calibrator([1., 2.], fit, day)
    for day in ("2025-08-01T00:00:00", "20250801", datetime(2025, 8, 1), None):
        with pytest.raises(ValueError, match="date"):
            calibration.apply_rate_calibrator([1., 2.], fit, day)


def test_saved_model_identity_configuration_and_fitting_metadata_must_agree():
    mutations = [
        ("model_type", "multiclass_calibrator"), ("basis", "conditional_reference"), ("rho", .1),
        ("rho", False), ("method", "unknown"), ("before_date", "2025-05-01"),
        ("training_min_date", "2025-06-01"), ("training_max_date", "2025-08-01"),
        ("n_matches", True), ("n_matches", 4), ("training_match_ids", ["a", "a", "b"]),
        ("estimation", {"algorithm": "minimize"}),
    ]
    for field, value in mutations:
        fit = fitted()
        fit[field] = value
        with pytest.raises(ValueError):
            calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)
    for field in ("before_date", "parameters", "training_goal_sums", "estimation"):
        fit = fitted()
        del fit[field]
        with pytest.raises(ValueError):
            calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)


def test_saved_scales_cannot_be_changed_without_their_closed_form_aggregates():
    for scales in ([1., 1.], [float("nan"), 1.], [0, 1.], [True, 1.], [[1., 1.]], [1., 1.001]):
        fit = fitted()
        fit["parameters"]["scales"] = scales
        with pytest.raises(ValueError):
            calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)
    fit = fitted()
    fit["score_equation_residual_sums"]["shared"] += .1
    with pytest.raises(ValueError, match="residual"):
        calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)
    fit = fitted()
    fit["raw_training_analytic_joint_score_nll"]["mean"] += .1
    with pytest.raises(ValueError, match="NLL"):
        calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)
    fit = fitted()
    fit["estimation"]["market_inputs"] = 0  # Equality to False must not mask a malformed declaration.
    with pytest.raises(ValueError, match="market"):
        calibration.apply_rate_calibrator([1., 2.], fit, CUTOFF)


def test_apply_refuses_bad_rate_shapes_and_overflow_or_underflow_without_clipping():
    fit = fitted()
    for rates in ([], [[1, 2, 3]], [[[1, 2]]], [True, 2.], ["1", 2.], [0, 1], [float("inf"), 1.]):
        with pytest.raises(ValueError):
            calibration.apply_rate_calibrator(rates, fit, CUTOFF)
    rows = [{"match_id": "a", "date": "2025-05-01", "actual_score": [4, 4], "rates": [1., 1.]}]
    scale_up = calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")
    with pytest.raises(ValueError, match="overflowed"):
        calibration.apply_rate_calibrator([1e308, 1.], scale_up, CUTOFF)
    rows[0].update(actual_score=[1, 0], rates=[1e308, 1.])
    scale_down = calibration.fit_rate_calibrator(rows, CUTOFF, "shared_scale")
    with pytest.raises(ValueError, match="underflowed"):
        calibration.apply_rate_calibrator([np.nextafter(0., 1.), 1.], scale_down, CUTOFF)


def test_large_score_raw_loss_is_analytic_not_a_clipped_grid_cell_probability():
    rows = [{"match_id": "a", "date": "2025-05-01", "actual_score": [250, 170], "rates": [1.2, .7]}]
    fit = calibration.fit_rate_calibrator(rows, CUTOFF, "side_scale")
    assert fit["raw_training_analytic_joint_score_nll"]["sum"] == pytest.approx(analytic_joint_nll(rows))
    assert fit["raw_training_analytic_joint_score_nll"]["sum"] > 1000
    assert fit["parameters"]["scales"] == pytest.approx([250 / 1.2, 170 / .7])
