"""Ordinary numerical checks for the standalone coherent-rate audit."""

import ast
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1]/"src"/"football_model"/"independent_rate_calibration_audit.py"
_SPEC = importlib.util.spec_from_file_location("standalone_rate_scaling_audit_test", _PATH)
standalone = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(standalone)


def test_closed_form_shared_and_side_factors_have_known_answers():
    rows = [{"actual_score": [3, 0]}, {"actual_score": [1, 5]}]
    goals, totals, scales = standalone.closed_form_parameters(rows, np.array([[1., 2.], [2., 2.]]))
    assert goals == [4, 5]
    assert totals == [3, 4]
    assert scales["shared_scale"] == pytest.approx([9/7, 9/7])
    assert scales["side_scale"] == pytest.approx([4/3, 5/4])


@pytest.mark.parametrize("scales", [[1., 1.], [.9, 1.1], [.001, 7.]])
def test_offset_difference_is_independent_of_factorial_terms(scales):
    score, rates = [21, 0], [2., 1.5]
    delta = standalone.offset_nll_change(score, rates, scales)
    direct = [(r*c-y*math.log(r*c)+math.log(math.factorial(y)))-
              (r-y*math.log(r)+math.log(math.factorial(y))) for y, r, c in zip(score, rates, scales)]
    assert delta["home_poisson_nll"] == pytest.approx(direct[0], abs=1e-12)
    assert delta["away_poisson_nll"] == pytest.approx(direct[1], abs=1e-12)
    assert delta["joint_score_nll"] == pytest.approx(sum(direct), abs=1e-12)


def test_recurrent_forecast_is_one_coherent_distribution_with_expected_thresholds():
    forecast = standalone.poisson_forecast((1., 2.))
    assert sum(forecast["one_x_two"].values()) == pytest.approx(1)
    assert forecast["total_goals"]["0"] == pytest.approx(math.exp(-3))
    assert forecast["total_goals"]["4+"] == pytest.approx(1-13*math.exp(-3))
    assert forecast["both_score"] == pytest.approx((1-math.exp(-1))*(1-math.exp(-2)))
    assert forecast["over_2_5"] == pytest.approx(1-8.5*math.exp(-3))
    assert forecast["expected_goals"]["home"] == pytest.approx(1)
    assert forecast["expected_goals"]["away"] == pytest.approx(2)
    jc = forecast["away_minus_two_jincai"]
    assert sum(jc.values()) == pytest.approx(1)
    assert forecast["away_minus_two_asian"]["push"] == jc["draw"]
    assert forecast["fair_odds_unbuffered"]["away_minus_two_jincai_win"] == pytest.approx(1/jc["win"])
    assert forecast["fair_odds_unbuffered"]["away_minus_two_asian"] == pytest.approx((1-jc["draw"])/jc["win"])


def test_three_domains_use_score_and_three_way_handicap_not_asian_push():
    rows = [{"actual_score": [0, 3]}, {"actual_score": [1, 3]}, {"actual_score": [3, 0]}, {"actual_score": [2, 2]}]
    forecasts = [standalone.poisson_forecast((1., 1.)) for _ in rows]
    arrays = standalone.domain_arrays(rows, forecasts)
    assert arrays["one_x_two"][1].tolist() == [2, 2, 0, 1]
    assert arrays["away_minus_two_three_way"][1].tolist() == [0, 1, 2, 2]
    assert arrays["total_goals_0_1_2_3_4plus"][1].tolist() == [3, 4, 3, 4]


def test_categorical_score_has_known_log_loss_brier_and_ordered_rps():
    result = standalone.categorical_score(np.array([[.5, .25, .25]]), np.array([1]))
    assert result == pytest.approx({"n": 1, "log_loss": math.log(4), "brier_sum": .875,
                                   "rps": .15625, "accuracy": 0.})


def test_independent_scripts_import_no_production_package():
    for source in (_PATH, _PATH.with_name("independent_goal_rate_audit.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("football_model") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                assert not (node.module or "").startswith("football_model")


def small_predictions():
    rows = [
        {"match_id": "m1", "date": "2025-08-09", "competition": "E0", "home": "H", "away": "A",
         "actual_score": [3, 1], "fit_before_date": "2025-08-01"},
        {"match_id": "m2", "date": "2025-08-16", "competition": "E0", "home": "A", "away": "H",
         "actual_score": [0, 2], "fit_before_date": "2025-08-01"}]
    raw_rates = np.array([[1., 2.], [2., 1.]])
    scales = {"shared_scale": np.array([.9, .9]), "side_scale": np.array([.8, 1.2])}
    for row, rate in zip(rows, raw_rates):
        row["forecast"] = standalone.poisson_forecast(rate)
    checker = standalone.independent.Auditor()
    rates, forecasts, stats, changes = standalone.prepared_predictions(rows, raw_rates, scales, checker)
    assert not checker.errors
    return rows, rates, forecasts, stats, changes


def test_later_cohort_exact_score_difference_uses_offset_identity():
    rows, rates, forecasts, stats, changes = small_predictions()
    cohort = standalone.cohort(rows, forecasts, stats, changes, [0, 1])
    for method in standalone.METHODS:
        expected = np.mean([d["joint_score_nll"] for d in changes[method]])
        assert cohort["mean_differences"][method+"_minus_original"]["exact_score"] == pytest.approx(expected)
        assert cohort["goal_summaries"][method]["n"] == 2
        assert set(cohort["scores"][method]) == {*standalone.DOMAIN_NAMES, "exact_score"}


def test_probability_calibration_boxes_retain_class_labels_and_membership():
    p = np.array([[.6, .2, .2], [.05, .15, .8]])
    result = standalone.probability_calibration(p, np.array([0, 1]))
    assert result["classes"]["home"]["mean_probability"] == pytest.approx(.325)
    assert result["classes"]["draw"]["observed_frequency"] == .5
    assert sum(row["n"] for row in result["equal_width_5_bins"]) == 6
    home = [row for row in result["equal_width_5_bins"] if row["outcome"] == "home"]
    assert [row["bin"] for row in home] == [0, 3]


def test_full_csv_audit_detects_one_changed_exact_score_field(tmp_path):
    rows, rates, forecasts, stats, changes = small_predictions()
    losses, arrays = standalone.per_row_losses(rows, forecasts, stats, changes)
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals",
              "base_fit_before_date", "calibration_before_date"]
    fields += [arm+"_"+key for arm in standalone.ARMS for key in (*standalone.STAT_FIELDS,
        "p_home", "p_draw", "p_away", *(domain+"_nll" for domain in (*standalone.DOMAIN_NAMES, "exact_score")))]
    records = []
    for i, row in enumerate(rows):
        record = {key: row[key] for key in fields[:5]}
        record.update({"home_goals": row["actual_score"][0], "away_goals": row["actual_score"][1],
                       "base_fit_before_date": row["fit_before_date"], "calibration_before_date": "2025-08-01"})
        for arm in standalone.ARMS:
            record.update({arm+"_"+key: value for key, value in stats[arm][i].items()})
            record.update({arm+"_p_"+label: arrays[arm]["one_x_two"][0][i, c] for c, label in enumerate(standalone.CLASSES)})
            record.update({arm+"_"+domain+"_nll": losses[arm][domain][i] for domain in (*standalone.DOMAIN_NAMES, "exact_score")})
        records.append(record)
    records[1]["side_scale_exact_score_nll"] += .02
    path = tmp_path/"full_rate_audit.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    checker = standalone.independent.Auditor()
    standalone.audit_csv(path, rows, stats, losses, arrays, checker)
    assert len(fields) == 102
    assert len(checker.errors) == 1
    assert checker.errors[0]["context"] == ["m2", "side_scale_exact_score_nll"]


def test_joint_resample_audit_detects_one_modified_interval_bound():
    dates = ["2025-08-04", "2025-08-25", "2025-09-15"]
    features = {"a": np.array([1., 2., 6.]), "b": np.array([2., 4., 12.])}
    saved = standalone.independent.block_bootstrap(dates, np.column_stack(list(features.values())), 2, 20, 5)
    means, intervals = saved.pop("mean"), saved.pop("interval_95")
    saved["features"] = {key: {"mean": means[i], "interval_95": intervals[i]} for i, key in enumerate(features)}
    saved["features"]["b"]["interval_95"][1] += .1
    checker = standalone.independent.Auditor()
    standalone.audit_bootstrap(dates, features, saved, checker, "sample")
    assert len(checker.errors) == 1
    assert checker.errors[0]["context"] == "sample.b.interval_95[1]"
