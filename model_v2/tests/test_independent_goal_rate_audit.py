"""Small checks for the standalone audit, without importing production code."""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1]/"src"/"football_model"/"independent_goal_rate_audit.py"
_SPEC = importlib.util.spec_from_file_location("standalone_goal_rate_audit", _PATH)
audit_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_module)


def test_all_24_statistics_have_independent_closed_form_values():
    actual = audit_module.statistics((0, 1), (1., 2.))
    assert set(actual) == set(audit_module.STAT_FIELDS)
    assert len(actual) == 24
    assert actual["home_residual"] == -1
    assert actual["away_residual"] == -1
    assert actual["home_pearson_square_minus_one"] == 0
    assert actual["away_pearson_square_minus_one"] == -.5
    assert actual["pearson_cross_product"] == pytest.approx(1/math.sqrt(2))
    assert actual["joint_score_nll"] == pytest.approx(3-math.log(2))
    assert actual["zero_zero_probability"] == pytest.approx(math.exp(-3))
    assert actual["one_one_probability"] == pytest.approx(2*math.exp(-3))
    expected_draw = sum(math.exp(-3)*2**g/math.factorial(g)**2 for g in range(30))
    assert actual["draw_probability"] == pytest.approx(expected_draw)
    assert actual["total_four_plus_probability"] == pytest.approx(1-13*math.exp(-3))
    assert actual["home_zero_residual"] == pytest.approx(1-math.exp(-1))
    assert actual["away_zero_residual"] == pytest.approx(-math.exp(-2))


def test_log_score_retains_a_large_legal_score_without_grid_or_clipping():
    actual = audit_module.statistics((21, 0), (2., 3.))
    assert actual["home_goals"] == 21
    assert actual["joint_score_nll"] == pytest.approx(5-21*math.log(2)+math.log(math.factorial(21)))


def test_summary_uses_mixture_of_conditional_probabilities_not_mean_rate():
    rows = [audit_module.statistics((0, 1), (.2, 1.)),
            audit_module.statistics((5, 0), (3.2, 2.))]
    result = audit_module.summarize(rows)
    assert result["home"]["observed_sum"] == 5
    assert result["home"]["expected_sum"] == pytest.approx(3.4)
    zero_count = result["home"]["goal_counts"]["0"]["expected"]
    assert zero_count == pytest.approx(math.exp(-.2)+math.exp(-3.2))
    assert zero_count != pytest.approx(2*math.exp(-1.7))
    for side in ("home", "away"):
        assert sum(r["observed"] for r in result[side]["goal_counts"].values()) == 2
        assert sum(r["expected"] for r in result[side]["goal_counts"].values()) == pytest.approx(2)
    assert audit_module.summarize(rows[:1])["scalar_descriptives"]["home_rate"]["sd"] is None


@pytest.mark.parametrize(("rate", "label"), [(.7499, "0-0.75"), (.75, "0.75-1"), (1., "1-1.25"),
    (1.25, "1.25-1.5"), (1.5, "1.5-2"), (2., "2-3"), (3., "3+"), (12., "3+")])
def test_rate_bin_boundaries(rate, label):
    assert audit_module.bin_label(rate) == label


@pytest.mark.parametrize("rate", [0., -1., math.nan, math.inf])
def test_rate_bin_rejects_nonpositive_or_nonfinite_values(rate):
    with pytest.raises(ValueError):
        audit_module.bin_label(rate)


def test_explicit_coefficient_replay_uses_provider_keys_and_defense_sign():
    row = {"competition": "E0", "home": "Home Team", "away": "Away Team"}
    parameters = {"intercept": .1, "coefficients": {
        "competition:E0": .2, "home:E0": .3,
        "attack:football-data:E0:Home Team": .4, "defense:football-data:E0:Away Team": .5,
        "attack:football-data:E0:Away Team": .6, "defense:football-data:E0:Home Team": .7}}
    assert audit_module.replay_rates(parameters, row) == pytest.approx((math.exp(.5), math.exp(.2)))


def test_joint_calendar_blocks_preserve_empty_weeks_and_shared_column_weights():
    dates = ["2024-08-05", "2024-08-06", "2024-08-26"]
    values = np.array([1., 2., 8.])
    boot = audit_module.block_bootstrap(dates, np.column_stack((values, 2*values)), 2, 100, 5)
    assert boot["weekly_match_counts"] == [2, 0, 0, 1]
    assert boot["n_matches"] == 3
    assert boot["valid_repeats"]+boot["zero_count_repeats"] == 100
    assert boot["mean"] == pytest.approx([11/3, 22/3])
    assert boot["interval_95"][1] == pytest.approx(2*np.array(boot["interval_95"][0]))


def test_unavailable_block_interval_is_explicit():
    result = audit_module.block_bootstrap(["2024-08-05"], np.array([[1.]]), 2, 10, 5)
    assert result["interval_95"] is None
    assert result["valid_repeats"] == 0
    assert result["unavailable_reason"] == "block_length_not_smaller_than_calendar_span"


def test_auditor_records_bad_numeric_missing_keys_and_null_mismatch():
    checker = audit_module.Auditor()
    checker.numeric(1., "1.25", "changed")
    checker.numeric(1., "nan", "nonfinite")
    checker.tree({"a": 1}, {}, "missing")
    checker.tree(None, {}, "null")
    assert len(checker.errors) == 4


def test_csv_audit_checks_every_statistic_and_detects_modified_value(tmp_path):
    rows = [{"match_id": "m", "date": "2024-08-08", "competition": "E0", "home": "H", "away": "A",
             "actual_score": [2, 1], "fit_before_date": "2024-08-01", "training_matches": 10}]
    stats = {arm: [audit_module.statistics((2, 1), (1.5, 1.))] for arm in audit_module.ARMS}
    tracks = {arm: {"m": (rows[0], stats[arm][0])} for arm in audit_module.ARMS}
    record = {key: rows[0][key] for key in ("match_id", "date", "competition", "home", "away")}
    record.update({"home_goals": 2, "away_goals": 1, "fixed_28_home_rate_bin": "1.5-2", "fixed_28_away_rate_bin": "1-1.25"})
    for arm in audit_module.ARMS:
        record.update({arm+"_fit_before_date": "2024-08-01", arm+"_training_matches": 10})
        record.update({arm+"_"+key: value for key, value in stats[arm][0].items()})
    record["days_14_home_residual"] += .01
    path = tmp_path/"audit_sample.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    checker = audit_module.Auditor()
    assert audit_module.audit_csv(path, rows, stats, tracks, checker) == 1
    assert len(checker.errors) == 1
    assert checker.errors[0]["context"] == ["m", "days_14_home_residual"]


def test_no_production_import_statement_in_standalone_script():
    import ast
    tree = ast.parse(_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("football_model") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("football_model")
