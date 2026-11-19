"""Earlier-only existing scale estimators, without distribution-layer stacking."""

from copy import deepcopy
import csv
from datetime import date
import inspect
import math

import numpy as np
import pytest

from football_model import rate_calibration_experiment as legacy
from football_model import rate_calibration_forward as forward
from football_model.cmp_forward_experiment import TRAINING_START, WINDOWS, window_rows
from football_model.goal_rate_diagnostics import row_statistics
from football_model.probability import score_grid, summarize
from test_cmp_forward_experiment import archive_rows
from test_goal_rate_experiment import prediction


START = "2025-08-01"
CUTOFF = "2025-11-01"


def source_rows():
    result = []
    for index, (day, score, rates) in enumerate((
            ("2025-08-01", (0, 1), (1.2, 1.1)),
            ("2025-08-17", (1, 2), (1.8, 1.4)),
            ("2025-09-07", (3, 1), (2.1, 1.6)),
            ("2025-10-31", (2, 0), (1.5, 1.3)))):
        row = prediction(f"learn-{index}", day, score, rates=rates)
        row["rates"] = list(rates)
        result.append(row)
    return result


def target_rows():
    result = []
    for index, (day, score, rates) in enumerate((
            ("2025-11-01", (1, 1), (1.3, 1.1)),
            ("2025-11-08", (0, 3), (1.8, 1.4)),
            ("2026-01-31", (4, 0), (2.1, 1.6)))):
        row = prediction(f"target-{index}", day, score, rates=rates)
        row["rates"] = list(rates)
        row["market_prices"] = {"avg_preclosing": [2., 4., 4.]}
        row["market_probabilities"] = {"avg_preclosing": [.5, .25, .25]}
        result.append(row)
    return result


def fit(rows=None, **changes):
    return forward.fit_window(source_rows() if rows is None else rows,
        **{"training_start": START, "before_date": CUTOFF, **changes})


def research(rows=None, parameters=None, **changes):
    return forward.research_tracks(target_rows() if rows is None else rows,
        fit() if parameters is None else parameters,
        **{"before_date": CUTOFF, **changes})


def test_explicit_fit_and_application_cutoffs_are_required_keyword_arguments():
    for function, names in ((forward.fit_window, ("training_start", "before_date")),
                            (forward.research_tracks, ("before_date",))):
        signature = inspect.signature(function)
        for name in names:
            assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            assert signature.parameters[name].default == inspect.Parameter.empty


def test_same_existing_closed_form_estimators_equal_match_weights_and_no_source_mutation():
    rows = source_rows()
    original = deepcopy(rows)
    fits = fit(rows)
    assert rows == original
    assert tuple(fits) == forward.METHODS == ("shared_scale", "side_scale")
    goals = np.asarray([r["actual_score"] for r in rows]).sum(axis=0)
    rates = np.asarray([r["rates"] for r in rows]).sum(axis=0)
    for method, model in fits.items():
        expected = goals / rates if method == "side_scale" else np.repeat(goals.sum()/rates.sum(), 2)
        np.testing.assert_allclose(model["parameters"]["scales"], expected, rtol=1e-14, atol=0)
        assert model["training_match_ids"] == [r["match_id"] for r in rows]
        assert model["n_matches"] == 4
        assert model["before_date"] == model["training_end_exclusive"] == CUTOFF
        assert model["training_start"] == START
        assert model["training_min_date"] == START
        assert model["training_max_date"] == "2025-10-31"
        assert model["estimation"]["algorithm"] == "closed_form"
        assert model["estimation"]["optimizer"] is None
        assert model["estimation"]["market_inputs"] is False
        scales = model["parameters"]["scales"]
        nll = np.mean([sum(rate*c - goal*math.log(rate*c) + math.lgamma(goal+1)
            for goal, rate, c in zip(row["actual_score"], row["rates"], scales)) for row in rows])
        assert model["calibrated_training_analytic_joint_score_nll"]["mean"] == pytest.approx(nll)
        assert nll <= model["raw_training_analytic_joint_score_nll"]["mean"]
    assert fit(reversed(rows)) == fits
    assert fit(row for row in rows) == fits


@pytest.mark.parametrize("fold_index", [0, 2, 4])
def test_three_actual_closed_form_fits_ignore_all_valid_future_score_and_market_changes(fold_index):
    rows = archive_rows()
    cutoff, end = WINDOWS[fold_index]
    training, target = window_rows(rows, cutoff, end)
    first = forward.fit_window(training, training_start=TRAINING_START, before_date=cutoff)
    perturbed = deepcopy(rows)
    for row in perturbed:
        if row["date"] >= cutoff:
            h, a = row["actual_score"]
            row["actual_score"] = [h+9, a+1]
            row["actual_outcome"] = 0 if h+9 > a+1 else 2 if h+9 < a+1 else 1
            row["market_prices"] = {"avg_preclosing": [3., 4., 5.]}
            inverse = 1 / np.asarray([3., 4., 5.])
            row["market_probabilities"] = {"avg_preclosing": (inverse/inverse.sum()).tolist()}
    other_training, other_target = window_rows(perturbed, cutoff, end)
    assert other_training == training and other_target != target
    assert forward.fit_window(other_training, training_start=TRAINING_START, before_date=cutoff) == first
    for model in first.values():
        assert model["training_max_date"] < cutoff
        assert not set(model["training_match_ids"]) & {r["match_id"] for r in other_target}
    original_tracks, _, _ = forward.research_tracks(target, first, before_date=cutoff)
    changed_tracks, _, _ = forward.research_tracks(other_target, first, before_date=cutoff)
    for arm in forward.ARMS:
        for original, changed in zip(original_tracks[arm], changed_tracks[arm]):
            assert changed["rates"] == original["rates"]
            assert changed["forecast"] == original["forecast"]
            assert changed["probabilities"] == original["probabilities"]
            assert changed["exact_score_nll"] != original["exact_score_nll"]


def test_each_forward_probability_and_full_support_nll_comes_from_the_recorded_adjusted_means():
    rows, fits = target_rows(), fit()
    unchanged = deepcopy((rows, fits))
    tracks, stats, exported = research(rows, fits)
    assert (rows, fits) == unchanged
    identity_fields = ("match_id", "date", "competition", "home", "away", "actual_score",
                       "actual_outcome", "source_url", "fit_before_date", "training_max_date",
                       "market_prices", "market_probabilities")
    for arm in forward.ARMS:
        scales = [1., 1.] if arm == "original" else fits[arm]["parameters"]["scales"]
        for i, row in enumerate(tracks[arm]):
            assert {k: row[k] for k in identity_fields} == {k: rows[i][k] for k in identity_fields}
            assert row["base_rates"] == rows[i]["rates"]
            np.testing.assert_array_equal(row["rates"], np.asarray(row["base_rates"])*scales)
            assert row["calibration_before_date"] == CUTOFF
            assert row["calibration_training_start"] == START
            assert row["calibration_method"] == arm
            assert row["calibration_scales"] == scales
            assert row["forecast"] == summarize(score_grid(*row["rates"], rho=0))
            assert row["probabilities"] == list(row["forecast"]["one_x_two"].values())
            assert stats[arm][i] == row_statistics(rows[i], row["rates"])
            assert row["exact_score_nll"] == stats[arm][i]["joint_score_nll"]
            assert exported[i]["calibration_before_date"] == CUTOFF
            assert exported[i]["rates"][arm] == row["rates"]
            assert exported[i]["research_forecasts"][arm] == row["forecast"]
            assert exported[i]["production_api_changed"] is False
    for method in forward.METHODS:
        assert any(a["forecast"]["one_x_two"] != b["forecast"]["one_x_two"]
            for a, b in zip(tracks[method], tracks["original"]))
        assert any(a["forecast"]["total_goals"] != b["forecast"]["total_goals"]
            for a, b in zip(tracks[method], tracks["original"]))
        assert any(a["forecast"]["away_minus_two_jincai"] != b["forecast"]["away_minus_two_jincai"]
            for a, b in zip(tracks[method], tracks["original"]))
    tracks["original"][0]["market_prices"]["avg_preclosing"][0] = 9.
    tracks["original"][0]["forecast"]["total_goals"]["4+"] = 0.
    assert (rows, fits) == unchanged  # Returned nested objects cannot mutate input source data.


def test_target_order_is_sorted_and_never_changes_statistics_or_coverage():
    tracks, statistics, exported = research()
    other, other_stats, other_exported = research(reversed(target_rows()))
    assert (other, other_stats, other_exported) == (tracks, statistics, exported)


def test_legacy_csv_can_export_mixed_dynamic_cutoffs_without_changing_fixed_api_default(tmp_path):
    tracks, stats, _ = research()
    later_cutoff = "2026-02-01"
    earlier_fits = fit(source_rows(), before_date=later_cutoff)
    future = prediction("later-target", "2026-02-01", (2, 3), rates=(1.2, 1.4))
    future["rates"] = [1.2, 1.4]
    other, other_stats, _ = forward.research_tracks([future], earlier_fits, before_date=later_cutoff)
    for arm in forward.ARMS:
        tracks[arm].extend(other[arm])
        stats[arm].extend(other_stats[arm])
    path = tmp_path / "dynamic.csv"
    legacy.write_csv(path, tracks, stats)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert [r["calibration_before_date"] for r in saved] == [CUTOFF]*3 + [later_cutoff]
    for row in saved:
        assert row["base_fit_before_date"]
    assert inspect.signature(legacy.research_tracks).parameters["before_date"].default == legacy.TEST_START


@pytest.mark.parametrize("which", ["training_start", "before_date"])
@pytest.mark.parametrize("bad", [None, True, 20250801, date(2025, 8, 1), "20250801",
                                 "2025-W31-5", "2025-8-01", "2025-02-30", "", "2025-08-01T00:00:00"])
def test_window_bounds_require_literal_iso_dates(which, bad):
    with pytest.raises(ValueError, match=which):
        fit(**{which: bad})


@pytest.mark.parametrize("start,end", [(CUTOFF, CUTOFF), ("2025-11-02", CUTOFF)])
def test_nonpositive_window_is_an_error(start, end):
    with pytest.raises(ValueError, match="strictly earlier"):
        fit(training_start=start, before_date=end)


@pytest.mark.parametrize("phase", ["training", "target"])
@pytest.mark.parametrize("field", ["date", "fit_before_date", "training_max_date"])
def test_noncanonical_source_dates_are_not_accepted_even_if_parseable(phase, field):
    rows = source_rows() if phase == "training" else target_rows()
    rows[0][field] = rows[0][field].replace("-", "")
    with pytest.raises(ValueError, match=field):
        fit(rows) if phase == "training" else research(rows)


@pytest.mark.parametrize("day", ["2025-07-31", CUTOFF, "2025-11-02"])
def test_training_outside_window_is_refused_without_dropping_rows(day):
    rows = source_rows()
    outside = prediction("outside", day, rates=(1., 1.))
    outside["rates"] = [1., 1.]
    with pytest.raises(ValueError, match="training_start <= date < before_date"):
        fit(rows + [outside])


def test_target_with_even_one_pre_cutoff_row_is_refused():
    with pytest.raises(ValueError, match="on/after"):
        research(target_rows() + [source_rows()[-1]])


@pytest.mark.parametrize("phase", ["training", "target"])
@pytest.mark.parametrize("problem", ["none", "empty", "not_mapping", "duplicate", "rates_bool", "rates_string",
    "rates_nan", "rates_zero", "rates_missing", "mean_grid_mismatch", "grid_tamper", "wrong_outcome",
    "goal_bool", "same_day_source_training", "future_source_cutoff", "unknown_competition", "missing_source"])
def test_bad_inputs_cannot_be_silently_filtered_imputed_or_repaired(phase, problem):
    rows = source_rows() if phase == "training" else target_rows()
    if problem == "none":
        rows = None
    elif problem == "empty":
        rows = []
    elif problem == "not_mapping":
        rows.append(3)
    elif problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "rates_bool":
        rows[0]["rates"][0] = True
    elif problem == "rates_string":
        rows[0]["rates"][0] = "1.2"
    elif problem == "rates_nan":
        rows[0]["rates"][0] = float("nan")
    elif problem == "rates_zero":
        rows[0]["rates"][0] = 0.
    elif problem == "rates_missing":
        rows[0].pop("rates")
    elif problem == "mean_grid_mismatch":
        rows[0]["rates"][0] += .7
    elif problem == "grid_tamper":
        rows[0]["forecast"]["total_goals"]["4+"] += .1
    elif problem == "wrong_outcome":
        rows[0]["actual_outcome"] = 0
    elif problem == "goal_bool":
        rows[0]["actual_score"][0] = True
    elif problem == "same_day_source_training":
        rows[0]["training_max_date"] = rows[0]["fit_before_date"]
    elif problem == "future_source_cutoff":
        rows[0]["fit_before_date"] = "2027-01-01"
    elif problem == "unknown_competition":
        rows[0]["competition"] = "OTHER"
    elif problem == "missing_source":
        rows[0]["source_url"] = ""
    with pytest.raises(ValueError):
        # Explicit calls distinguish a supplied None from the fixture default.
        if phase == "training":
            forward.fit_window(rows, training_start=START, before_date=CUTOFF)
        else:
            forward.research_tracks(rows, fit(), before_date=CUTOFF)


@pytest.mark.parametrize("field", forward.RESEARCH_FIELDS)
@pytest.mark.parametrize("phase", ["training", "target"])
def test_even_null_research_layers_are_not_stacked(field, phase):
    rows = source_rows() if phase == "training" else target_rows()
    rows[0][field] = 0
    with pytest.raises(ValueError, match="not stacked"):
        fit(rows) if phase == "training" else research(rows)


@pytest.mark.parametrize("problem", ["missing_method", "swapped_slots", "wrong_cutoff", "wrong_end", "wrong_start",
                                   "different_training_ids", "invalid_scales", "wrong_model", "market_estimation"])
def test_application_refuses_incompatible_parameters_without_relearning(problem):
    parameters = fit()
    if problem == "missing_method":
        parameters.pop("shared_scale")
    elif problem == "swapped_slots":
        parameters["shared_scale"], parameters["side_scale"] = parameters["side_scale"], parameters["shared_scale"]
    elif problem == "wrong_cutoff":
        parameters["side_scale"]["before_date"] = "2025-11-02"
    elif problem == "wrong_end":
        parameters["side_scale"]["training_end_exclusive"] = "2025-11-02"
    elif problem == "wrong_start":
        parameters["side_scale"]["training_start"] = "2025-08-02"
    elif problem == "different_training_ids":
        parameters["side_scale"]["training_match_ids"][0] = "another-history"
    elif problem == "invalid_scales":
        parameters["side_scale"]["parameters"]["scales"][0] = True
    elif problem == "wrong_model":
        parameters["side_scale"]["model_type"] = "different_model"
    elif problem == "market_estimation":
        parameters["side_scale"]["estimation"]["market_inputs"] = True
    with pytest.raises(ValueError):
        research(parameters=parameters)


@pytest.mark.parametrize("phase", ["training", "target"])
def test_errors_leave_original_source_inputs_untouched(phase):
    rows = source_rows() if phase == "training" else target_rows()
    rows[-1]["source_url"] = ""
    original = deepcopy(rows)
    with pytest.raises(ValueError):
        fit(rows) if phase == "training" else research(rows)
    assert rows == original


@pytest.mark.parametrize("goals", [(0, 0), (0, 2), (2, 0)])
def test_zero_goal_boundary_has_no_positive_scale_mle_and_no_fallback(goals):
    rows = source_rows()
    for row in rows:
        row["actual_score"] = list(goals)
        row["actual_outcome"] = 0 if goals[0] > goals[1] else 2 if goals[0] < goals[1] else 1
    with pytest.raises(ValueError, match="no strictly positive finite"):
        fit(rows)
