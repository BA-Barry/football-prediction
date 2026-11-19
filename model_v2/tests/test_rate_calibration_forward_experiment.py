"""Existing rate-scale families are re-estimated only on strict past folds.

Synthetic readable rows exercise orchestration and arithmetic, not historical
predictive performance.  Actual closed-form fits remain enabled in these tests.
"""

from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pytest

from football_model import rate_calibration_forward_experiment as experiment
from football_model.date_model import DateTeamGoalModel
from football_model.goal_rate_diagnostics import row_statistics
from football_model.penaltyblog_adapter import NativeGoalModel
from football_model.probability import score_grid, summarize
from football_model.rate_calibration import fit_rate_calibrator
from football_model.rate_calibration_experiment import loss_vectors
from football_model.refresh_experiment import prediction_arrays
from test_cmp_forward_experiment import archive_rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def install_runner(tmp_path, monkeypatch):
    originals = archive_rows()
    # Retain a full-coverage and a partial-coverage market independently from
    # the fitted team means.  The latter must not silently change the cohort.
    for i, row in enumerate(originals):
        row["market_prices"] = {"avg_preclosing": [2., 4., 4.]}
        row["market_probabilities"] = {"avg_preclosing": [.5, .25, .25]}
        if i % 3:
            row["market_prices"]["avg_closing"] = [2.5, 4., 3.5]
            inverses = 1 / np.asarray([2.5, 4., 3.5])
            row["market_probabilities"]["avg_closing"] = (inverses / inverses.sum()).tolist()
    output = tmp_path / "rate-forward-output"
    events, fitted = [], []
    real_fit, real_research = experiment.fit_window, experiment.research_tracks

    def load(root, parent, season):
        assert Path(root) == tmp_path
        assert Path(parent) == tmp_path / "source"
        assert (output / "实验设置.json").is_file(), "Declare settings before source loading"
        events.append(("load", season))
        if season == "2024-25":
            rows = [r for r in originals if r["date"] < "2025-08-01"]
            snapshots = 2
        else:
            assert season == "2025-26"
            rows = [r for r in originals if r["date"] >= "2025-08-01"]
            snapshots = 3
        return deepcopy(rows), {"parameter_snapshots": snapshots, "saved_prediction_rows": len(rows)}

    def fit(rows, **kwargs):
        cutoff = kwargs["before_date"]
        cutoff = cutoff.isoformat() if type(cutoff) is date else cutoff
        index = experiment.BOUNDARIES.index(cutoff)
        expected, _ = experiment.window_rows(originals, *experiment.WINDOWS[index])
        assert [r["match_id"] for r in rows] == [r["match_id"] for r in expected]
        assert all(experiment.TRAINING_START <= r["date"] < cutoff for r in rows)
        events.append(("fit", cutoff, tuple(r["match_id"] for r in rows)))
        parameters = real_fit(rows, **kwargs)
        assert set(parameters) == set(experiment.METHODS)
        for method in experiment.METHODS:
            reference = fit_rate_calibrator(rows, date.fromisoformat(cutoff), method)
            assert parameters[method]["parameters"] == reference["parameters"]
            assert parameters[method]["training_match_ids"] == reference["training_match_ids"]
        fitted.append(deepcopy(parameters))
        return parameters

    def research(rows, parameters, *, before_date):
        cutoff = before_date.isoformat() if type(before_date) is date else before_date
        path = output / cutoff / experiment.PARAMETER_FILE
        assert path.is_file(), "Save both fitted families before forecasting any target"
        assert read_json(path) == parameters == fitted[-1]
        assert all(cutoff <= r["date"] for r in rows)
        events.append(("research", cutoff, tuple(r["match_id"] for r in rows)))
        return real_research(rows, parameters, before_date=before_date)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rate-scale research must reuse saved team predictions, not refit teams")

    monkeypatch.setattr(experiment, "load_period", load)
    monkeypatch.setattr(experiment, "fit_window", fit)
    monkeypatch.setattr(experiment, "research_tracks", research)
    monkeypatch.setattr(DateTeamGoalModel, "fit", forbidden)
    monkeypatch.setattr(NativeGoalModel, "fit", forbidden)
    return originals, output, events, fitted


@pytest.fixture
def completed(tmp_path, monkeypatch):
    originals, output, events, fitted = install_runner(tmp_path, monkeypatch)
    unchanged = deepcopy(originals)
    result = experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert originals == unchanged
    return result, originals, output, events, fitted


def test_all_six_real_fits_precede_their_targets_and_keep_unique_complete_cohort(completed):
    result, originals, output, events, fitted = completed
    assert events[:2] == [("load", "2024-25"), ("load", "2025-26")]
    assert len(fitted) == len(result["folds"]) == len(experiment.WINDOWS) == 6
    assert [fold["cutoff"] for fold in result["folds"]] == list(experiment.BOUNDARIES[:-1])
    ids = []
    for i, (cutoff, end) in enumerate(experiment.WINDOWS):
        assert events[2 + 2*i][:2] == ("fit", cutoff)
        assert events[3 + 2*i][:2] == ("research", cutoff)
        folder = output / cutoff
        assert read_json(folder / experiment.PARAMETER_FILE) == fitted[i]
        assert read_json(folder / "同场分析.json") == result["folds"][i]["analysis"]
        tracks = read_json(folder / "完整研究预测.json")
        target_ids = [row["match_id"] for row in tracks["original"]]
        assert len(target_ids) == (1, 2, 3, 1, 2, 3)[i]
        for arm in experiment.ARMS:
            assert [row["match_id"] for row in tracks[arm]] == target_ids
            assert all(row["calibration_before_date"] == cutoff for row in tracks[arm])
            assert all(row["calibration_target_end_exclusive"] == end for row in tracks[arm])
        ids.extend(target_ids)
    expected = {row["match_id"] for row in originals
                if experiment.BOUNDARIES[0] <= row["date"] < experiment.BOUNDARIES[-1]}
    assert set(ids) == expected
    assert len(ids) == len(set(ids)) == result["pooled"]["n"] == 12
    assert result["counts"]["new_team_fits"] == 0
    assert result["counts"]["unique_scored_matches"] == 12
    assert result["counts"]["new_calibrator_fits"] == 12
    assert result["counts"]["new_scalar_estimates"] == 18
    assert result["counts"]["new_research_forecasts"] == 24
    assert result["counts"]["reused_original_forecasts"] == 12
    assert result["counts"]["reused_team_parameter_snapshots"] == 5
    assert read_json(output / experiment.RESULT_FILE) == result
    assert read_json(output / "实验设置.json") == result["settings"]
    assert not (output / "研究失败.json").exists()


def test_every_adjusted_forecast_uses_its_fold_scale_and_retains_original_rate_evidence(completed):
    result, originals, output, _, fitted = completed
    tracks = read_json(output / "完整前推预测.json")
    source = {row["match_id"]: row for row in originals}
    assert set(tracks) == set(experiment.ARMS) == {"original", "shared_scale", "side_scale"}
    for i, original in enumerate(tracks["original"]):
        raw = source[original["match_id"]]
        assert original["rates"] == raw["rates"]
        index = experiment.BOUNDARIES.index(original["calibration_before_date"])
        for arm in experiment.ARMS:
            row = tracks[arm][i]
            assert row["base_rates"] == raw["rates"]
            assert row["actual_score"] == raw["actual_score"]
            assert row["market_probabilities"] == raw["market_probabilities"]
            assert row["market_prices"] == raw["market_prices"]
            scales = [1., 1.] if arm == "original" else fitted[index][arm]["parameters"]["scales"]
            expected = [value*scale for value, scale in zip(raw["rates"], scales)]
            assert row["rates"] == pytest.approx(expected)
            assert row["forecast"] == summarize(score_grid(*row["rates"], rho=0))
            assert row["probabilities"] == list(row["forecast"]["one_x_two"].values())
            for j, side in enumerate(("home", "away")):
                assert row["forecast"]["expected_goals"][side] == pytest.approx(expected[j], abs=1e-8)
    assert result["settings"]["production_api_changed"] is False
    assert result["settings"]["new_model_structure"] is False
    assert result["settings"]["stacked_with_cmp_or_dependence"] is False
    assert any(row["rates"] != row["base_rates"] for row in tracks["side_scale"])


def test_csv_saved_rates_probabilities_cutoffs_and_four_losses_match_the_actual_forecasts(completed):
    _, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    statistics = {arm: [row_statistics(row, row["rates"]) for row in tracks[arm]]
                  for arm in experiment.ARMS}
    vectors = loss_vectors(tracks, statistics)
    with (output / "逐场前推概率.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 12
    for i, record in enumerate(saved):
        raw = tracks["original"][i]
        assert record["match_id"] == raw["match_id"]
        assert record["calibration_before_date"] == raw["calibration_before_date"]
        for arm in experiment.ARMS:
            row = tracks[arm][i]
            assert [float(record[f"{arm}_{side}_rate"]) for side in ("home", "away")] == row["rates"]
            assert [float(record[f"{arm}_p_{side}"]) for side in ("home", "draw", "away")] == row["probabilities"]
            for domain in experiment.DOMAINS:
                assert float(record[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]


def test_four_targets_are_exact_match_weighted_not_unequal_fold_mean_weighted(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    statistics = {arm: [row_statistics(row, row["rates"]) for row in tracks[arm]]
                  for arm in experiment.ARMS}
    vectors = loss_vectors(tracks, statistics)
    n = result["pooled"]["n"]
    assert n == sum(fold["analysis"]["n"] for fold in result["folds"]) == 12
    assert n != len(result["folds"])
    unequal_fold_differences = []
    for method in experiment.METHODS:
        name = method + "_minus_original"
        for domain in experiment.DOMAINS:
            values = vectors[method][domain] - vectors["original"][domain]
            actual = result["pooled"]["mean_differences"][name][domain]
            assert actual == pytest.approx(values.mean())
            weighted = sum(fold["analysis"]["n"] * fold["analysis"]["mean_differences"][name][domain]
                           for fold in result["folds"]) / n
            assert actual == pytest.approx(weighted)
            for group in (result["pooled"]["by_league"], result["pooled"]["by_month"], result["by_season"]):
                recomposed = sum(item["n"] * item["mean_differences"][name][domain]
                                 for item in group.values()) / n
                assert actual == pytest.approx(recomposed)
            equal_fold = np.mean([fold["analysis"]["mean_differences"][name][domain]
                                  for fold in result["folds"]])
            unequal_fold_differences.append(not np.isclose(actual, equal_fold))
            for weeks in ("2", "4", "8"):
                boot = result["pooled"]["bootstrap"][weeks]
                assert boot["n_matches"] == n
                assert boot["features"][name + "_" + domain]["mean"] == pytest.approx(values.mean())
    assert any(unequal_fold_differences)
    assert {season: item["n"] for season, item in result["by_season"].items()} == {
        "2024-25_partial": 3, "2025-26": 9}


def test_full_support_score_loss_and_categorical_losses_are_recomputed_without_clipping(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    for arm in experiment.ARMS:
        rows = tracks[arm]
        exact = [sum(rate - goal*math.log(rate) + math.lgamma(goal+1)
                     for goal, rate in zip(row["actual_score"], row["rates"])) for row in rows]
        assert result["pooled"]["scores"][arm]["exact_score"]["mean_nll"] == pytest.approx(np.mean(exact))
        for domain, (probabilities, actual) in prediction_arrays(rows).items():
            expected = [-math.log(probabilities[i, label]) for i, label in enumerate(actual)]
            assert result["pooled"]["scores"][arm][domain]["log_loss"] == pytest.approx(np.mean(expected))


def test_partial_and_missing_markets_keep_all_three_models_on_identical_subsets(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    rows = tracks["original"]
    markets = result["pooled"]["markets"]
    assert markets["avg_preclosing"]["n"] == 12
    expected_partial = [row["match_id"] for row in rows if "avg_closing" in row["market_probabilities"]]
    assert 0 < len(expected_partial) < 12
    assert markets["avg_closing"]["match_ids"] == expected_partial
    assert set(markets["avg_closing"]["arm_scores_on_same_matches"]) == set(experiment.ARMS)
    assert all(item["n"] == len(expected_partial)
               for item in markets["avg_closing"]["arm_scores_on_same_matches"].values())
    assert markets["b365_preclosing"]["n"] == 0
    assert set(markets["b365_preclosing"]["missing_ids"]) == {row["match_id"] for row in rows}


def test_report_is_generated_from_saved_results_and_states_research_scope(completed):
    result, _, output, _, _ = completed
    saved = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    assert saved == experiment.report(result)
    for phrase in ("逐场等权", "不是新盲测", "净胜至少3球", "正式默认未改"):
        assert phrase in saved
    assert "12" in saved
    assert "CMP" in saved
    assert "不确定" in saved or "区间" in saved


@pytest.mark.parametrize("interval,phrase", [
    (None, "区间不可估计"),
    ([.001, .002], "方向是恶化而非提升"),
    ([-.002, -.001], "仍不能确认未来提升"),
    ([-.001, .002], "区间跨零"),
])
def test_report_distinguishes_worsening_improvement_uncertainty_and_missing_interval(completed, interval, phrase):
    result = deepcopy(completed[0])
    field = f"{experiment.PRIMARY_METHOD}_minus_original_{experiment.PRIMARY_DOMAIN}"
    result["pooled"]["bootstrap"]["4"]["features"][field]["interval_95"] = interval
    saved = experiment.report(result)
    assert phrase in saved
    assert "区块长度敏感性" in saved
    assert "| shared_scale | 8周 |" in saved
    assert "| side_scale | 2周 |" in saved


def test_all_eight_paired_loss_distributions_and_both_histograms_retain_every_match(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    statistics = {arm: [row_statistics(row, row["rates"]) for row in tracks[arm]]
                  for arm in experiment.ARMS}
    vectors = loss_vectors(tracks, statistics)
    for method in experiment.METHODS:
        for domain in experiment.DOMAINS:
            values = vectors[method][domain] - vectors["original"][domain]
            item = result["pooled"]["paired_loss_descriptives"][method][domain]
            assert item["n"] == 12
            assert item["mean"] == pytest.approx(values.mean())
            assert item["sd"] == pytest.approx(values.std(ddof=1))
            assert item["median"] == pytest.approx(np.median(values))
            assert item["q25"] == pytest.approx(np.quantile(values, .25))
            assert item["q75"] == pytest.approx(np.quantile(values, .75))
            assert item["min"] == values.min() and item["max"] == values.max()
            assert item["n_negative"] == np.sum(values < 0)
            assert item["n_positive"] == np.sum(values > 0)
            assert item["n_zero"] == np.sum(values == 0)
        path = output / f"{method}_逐场损失分布.svg"
        svg = ElementTree.parse(path).getroot()
        assert svg.tag == "{http://www.w3.org/2000/svg}svg"
        texts = ["".join(node.itertext()) for node in svg.findall(".//{http://www.w3.org/2000/svg}text")]
        assert all(domain in texts for domain in experiment.DOMAINS)
        assert sum(text.startswith("n=12, mean=") for text in texts) == 4
        assert "nan" not in path.read_text(encoding="utf-8").lower()


def test_each_saved_grid_metadata_uses_all_adjusted_means_without_exceeding_tail_tolerance(completed):
    _, _, output, _, _ = completed
    for cutoff, _ in experiment.WINDOWS:
        tracks = read_json(output / cutoff / "完整研究预测.json")
        metadata = read_json(output / cutoff / "比分网格与来源.json")
        assert len(metadata) == len(tracks["original"])
        for i, item in enumerate(metadata):
            assert item["match_id"] == tracks["original"][i]["match_id"]
            assert item["calibration_before_date"] == cutoff
            for arm in experiment.ARMS:
                row = tracks[arm][i]
                assert item["rates"][arm] == row["rates"]
                assert item["research_forecasts"][arm] == row["forecast"]
                grid = item["grid_metadata"][arm]
                assert grid["rho"] == 0
                assert grid["normalized_finite_grid"] is True
                assert 0 <= grid["omitted_tail_mass_before_normalization"] < 1e-8


@pytest.mark.parametrize("fold_index", [0, 2, 4])
def test_target_and_later_scores_cannot_change_current_real_scale_fits(fold_index):
    records = archive_rows()
    cutoff, end = experiment.WINDOWS[fold_index]
    training, target = experiment.window_rows(records, cutoff, end)
    original = experiment.fit_window(training, training_start=experiment.TRAINING_START, before_date=cutoff)
    perturbed = deepcopy(records)
    for row in perturbed:
        if row["date"] >= cutoff:
            home, away = row["actual_score"]
            row["actual_score"] = [home + 7, away + 2]
            row["actual_outcome"] = 0 if home + 7 > away + 2 else 2 if home + 7 < away + 2 else 1
    training2, target2 = experiment.window_rows(perturbed, cutoff, end)
    assert training == training2 and target != target2
    changed = experiment.fit_window(training2, training_start=experiment.TRAINING_START, before_date=cutoff)
    assert original == changed
    for method in experiment.METHODS:
        assert changed[method]["training_max_date"] < cutoff


@pytest.mark.parametrize("fail_index", [0, 2])
def test_fit_failure_retains_earlier_completed_folds_without_default_scale_substitution(tmp_path, monkeypatch, fail_index):
    _, output, events, _ = install_runner(tmp_path, monkeypatch)
    actual_fit = experiment.fit_window
    cutoff = experiment.WINDOWS[fail_index][0]

    def fail(rows, **kwargs):
        if kwargs["before_date"] == cutoff:
            raise ValueError("deliberate rate-scale failure")
        return actual_fit(rows, **kwargs)

    monkeypatch.setattr(experiment, "fit_window", fail)
    with pytest.raises(ValueError, match="deliberate rate-scale"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert len([event for event in events if event[0] == "research"]) == fail_index
    failure = read_json(output / "研究失败.json")
    assert failure["cutoff"] == cutoff
    assert failure["type"] == "ValueError"
    assert failure["message"] == "deliberate rate-scale failure"
    assert failure["failed_no_fallback"] is True
    for earlier, _ in experiment.WINDOWS[:fail_index]:
        assert (output / earlier / experiment.PARAMETER_FILE).is_file()
        assert (output / earlier / "同场分析.json").is_file()
    assert not (output / experiment.RESULT_FILE).exists()
    assert not (output / experiment.REPORT_FILE).exists()
    assert not (output / "完整前推预测.json").exists()


def test_forecast_failure_keeps_both_saved_families_and_never_claims_success(tmp_path, monkeypatch):
    _, output, _, _ = install_runner(tmp_path, monkeypatch)
    cutoff = experiment.WINDOWS[0][0]

    def fail(rows, parameters, *, before_date):
        assert read_json(output / cutoff / experiment.PARAMETER_FILE) == parameters
        raise RuntimeError("deliberate rate-scale forecast failure")

    monkeypatch.setattr(experiment, "research_tracks", fail)
    with pytest.raises(RuntimeError, match="deliberate rate-scale forecast"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert (output / cutoff / experiment.PARAMETER_FILE).is_file()
    assert not (output / cutoff / "完整研究预测.json").exists()
    assert not (output / experiment.RESULT_FILE).exists()
    assert read_json(output / "研究失败.json")["failed_no_fallback"] is True


def test_empty_declared_target_fold_is_an_error_not_a_smaller_successful_cohort(tmp_path, monkeypatch):
    originals, output, _, _ = install_runner(tmp_path, monkeypatch)
    cutoff, end = experiment.WINDOWS[0]
    originals[:] = [row for row in originals if not cutoff <= row["date"] < end]
    with pytest.raises(ValueError):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert read_json(output / "研究失败.json")["cutoff"] == cutoff
    assert not (output / experiment.RESULT_FILE).exists()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    _, output, _, _ = install_runner(tmp_path, monkeypatch)
    output.mkdir()
    marker = output / "earlier-user-artifact.json"
    marker.touch()
    with pytest.raises(FileExistsError):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert marker.exists()
    assert not (output / "实验设置.json").exists()


def test_cli_defaults_point_to_g_project_archive_and_honor_explicit_output(tmp_path, monkeypatch, capsys):
    calls = []

    def run(root, parent, output):
        calls.append((root, parent, output))
        return {"pooled": {"n": 12, "mean_differences": {}}, "counts": {"new_calibrator_fits": 12}}

    monkeypatch.setattr(experiment, "run", run)
    monkeypatch.setattr(sys, "argv", ["rate_calibration_forward_experiment", "--output", str(tmp_path / "chosen")])
    experiment.main()
    root, parent, output = calls[0]
    assert root == Path(experiment.__file__).resolve().parents[3]
    assert (root / "model_v2" / "pyproject.toml").is_file()
    assert parent == root / "outputs" / "更新频率研究_20261023T073047979805Z"
    assert output == tmp_path / "chosen"
    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == summary["fits"] == 12
