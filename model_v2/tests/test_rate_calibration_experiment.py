"""Chronological coherent-score rate research, not a new historical blind test."""

from copy import deepcopy
import csv
from dataclasses import replace
from datetime import date
import json
import math

import numpy as np
import pytest

from football_model.date_model import DateTeamGoalModel
from football_model.forecast_diagnostics import moving_block_means
from football_model.goal_rate_diagnostics import STAT_FIELDS, row_statistics
import football_model.goal_rate_experiment as archived_diagnostics
from football_model.penaltyblog_adapter import NativeGoalModel
from football_model.probability import score_grid, summarize
import football_model.rate_calibration_experiment as experiment

# Reuse the existing readable-archive fixture instead of duplicating its long
# per-fit metadata builder. The rate runner reads only the 10 original trials;
# all real refresh validate_trial checks remain enabled.
from test_goal_rate_experiment import install_archive_environment, prediction


def earlier_rows():
    return [
        {"match_id": "old-a", "date": "2024-08-10", "actual_score": [3, 1], "rates": [1.5, 1.3]},
        {"match_id": "old-b", "date": "2025-05-01", "actual_score": [1, 3], "rates": [2.5, 1.7]},
    ]


def later_rows():
    dates = ("2025-08-04", "2025-08-11", "2025-08-11", "2025-09-08", "2025-10-06", "2025-11-10")
    scores = ((0, 0), (1, 1), (0, 3), (1, 2), (4, 2), (2, 1))
    rows = []
    for i, (day, score) in enumerate(zip(dates, scores)):
        rates = (1.1 + .2 * i, .8 + .1 * i)
        row = prediction(f"new-{i}", day, score, "D1" if i == 3 else "E0", 28, rates)
        row["rates"] = list(rates)
        # One market with full coverage and one with a missing match.
        row["market_prices"] = {"avg_preclosing": [2., 4., 4.]}
        row["market_probabilities"] = {"avg_preclosing": [.5, .25, .25]}
        if i:
            row["market_prices"]["avg_closing"] = [2.5, 4., 3.5]
            inverse = 1 / np.asarray([2.5, 4., 3.5])
            row["market_probabilities"]["avg_closing"] = (inverse / inverse.sum()).tolist()
        rows.append(row)
    return rows


def research(rows=None, parameters=None):
    return experiment.research_tracks(later_rows() if rows is None else rows,
        experiment.fit_earlier(earlier_rows()) if parameters is None else parameters)


def analysis():
    tracks, statistics, exported = research()
    result, aligned_tracks, aligned_stats = experiment.analyze_later(tracks, statistics, repeats=37)
    return result, aligned_tracks, aligned_stats, exported


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def install_rate_runner(tmp_path, monkeypatch):
    parent, sources, paths, events = install_archive_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "season_directories", archived_diagnostics.season_directories)
    monkeypatch.setattr(experiment, "load_seasons", archived_diagnostics.load_seasons)
    actual_validate, actual_fit = experiment.validate_trial, experiment.fit_rate_calibrator
    def validate(trial, records, start, end, comp, interval):
        events.append(("rate_validate", start.year, comp, interval))
        assert interval == 28
        return actual_validate(trial, records, start, end, comp, interval)
    def fit(rows, before_date, method):
        events.append(("calibrate", method, len(rows)))
        assert all(date.fromisoformat(row["date"]) < before_date for row in rows)
        assert before_date == experiment.TEST_START
        return actual_fit(rows, before_date, method)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved base parameters must not be fitted or predicted anew")
    monkeypatch.setattr(experiment, "validate_trial", validate)
    monkeypatch.setattr(experiment, "fit_rate_calibrator", fit)
    monkeypatch.setattr(NativeGoalModel, "fit", forbidden)
    monkeypatch.setattr(DateTeamGoalModel, "predict", forbidden)
    return parent, sources, paths, events


def test_fit_earlier_has_exact_season_boundaries_and_fixed_application_cutoff():
    parameters = experiment.fit_earlier(earlier_rows())
    assert tuple(parameters) == experiment.METHODS
    for p in parameters.values():
        assert p["before_date"] == "2025-08-01"
        assert p["training_match_ids"] == ["old-a", "old-b"]
    for day in ("2024-07-31", "2025-07-01", "2025-08-01"):
        bad = earlier_rows()
        bad[0]["date"] = day
        with pytest.raises(ValueError, match="declared earlier season"):
            experiment.fit_earlier(bad)


def test_research_is_nonmutating_and_all_forecasts_share_their_recorded_rate_distribution():
    rows, parameters = later_rows(), experiment.fit_earlier(earlier_rows())
    original = deepcopy((rows, parameters))
    tracks, stats, exported = research(rows, parameters)
    assert (rows, parameters) == original
    assert tracks["original"] == rows
    for i, export in enumerate(exported):
        assert export["match_id"] == rows[i]["match_id"]
        assert export["production_api_changed"] is False
        assert export["calibration_before_date"] == "2025-08-01"
        for arm in experiment.ARMS:
            item = tracks[arm][i]
            rates = item["rates"]
            grid = score_grid(*rates, rho=0)
            expected = summarize(grid)
            assert item["forecast"] == expected
            assert export["research_forecasts"][arm] == expected
            assert export["rates"][arm] == rates
            assert item["probabilities"] == list(expected["one_x_two"].values())
            assert stats[arm][i] == row_statistics(rows[i], rates)
            metadata = export["grid_metadata"][arm]
            assert metadata["rho"] == 0 and metadata["normalized_finite_grid"] is True
            assert metadata["max_goal_inclusive"] == len(grid.grid) - 1
            assert 0 <= metadata["omitted_tail_mass_before_normalization"] < 1e-8


def test_wrong_method_slot_or_calibration_cutoff_is_rejected_not_swapped_or_refitted():
    parameters = experiment.fit_earlier(earlier_rows())
    parameters["shared_scale"], parameters["side_scale"] = parameters["side_scale"], parameters["shared_scale"]
    with pytest.raises(ValueError, match="method/cutoff"):
        research(parameters=parameters)
    parameters = experiment.fit_earlier(earlier_rows())
    parameters["side_scale"]["before_date"] = "2025-08-02"
    with pytest.raises(ValueError, match="method/cutoff"):
        research(parameters=parameters)
    with pytest.raises(ValueError, match="both"):
        research(parameters={"side_scale": parameters["side_scale"]})


def test_malformed_rates_and_wrong_original_forecast_cannot_mix_distributions():
    for invalid in ([True, 1.], ["1", 1.], [float("nan"), 1.], [0, 1.]):
        rows = later_rows()
        rows[0]["rates"] = invalid
        with pytest.raises(ValueError):
            research(rows)
    rows = later_rows()
    rows[0]["rates"][0] += .5
    with pytest.raises(ValueError):
        research(rows)
    rows = later_rows()
    rows[0]["forecast"]["total_goals"]["4+"] += .1
    with pytest.raises(ValueError):
        research(rows)


def test_three_arm_ids_dates_and_source_identity_cannot_diverge_or_shrink():
    tracks, stats, _ = research()
    tracks["side_scale"] = tracks["side_scale"][:-1]
    stats["side_scale"] = stats["side_scale"][:-1]
    with pytest.raises(ValueError, match="coverage"):
        experiment.analyze_later(tracks, stats, repeats=7)
    tracks, stats, _ = research()
    tracks["side_scale"][0]["date"] = "2025-08-05"
    with pytest.raises(ValueError, match="Conflicting"):
        experiment.analyze_later(tracks, stats, repeats=7)
    tracks, stats, _ = research()
    tracks["side_scale"][0]["source_url"] = "https://example.invalid/other.csv"
    with pytest.raises(ValueError, match="Conflicting"):
        experiment.analyze_later(tracks, stats, repeats=7)


def test_alignment_respects_each_arms_original_statistics_order_without_mutation():
    tracks, stats, _ = research()
    for arm in experiment.ARMS[1:]:
        tracks[arm].reverse()
        stats[arm].reverse()
    before = deepcopy((tracks, stats))
    result, aligned, aligned_stats = experiment.analyze_later(tracks, stats, repeats=7)
    for arm in experiment.ARMS:
        assert [r["match_id"] for r in aligned[arm]] == result["match_ids"]
        for row, stat in zip(aligned[arm], aligned_stats[arm]):
            assert stat == row_statistics(row, row["rates"])
    assert (tracks, stats) == before


def test_all_four_domain_scores_and_pairwise_differences_use_the_same_matches():
    result, tracks, stats, _ = analysis()
    assert experiment.DOMAINS == ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way", "exact_score")
    for arm in experiment.ARMS:
        vectors = experiment.prediction_arrays(tracks[arm])
        for domain in experiment.DOMAINS:
            if domain == "exact_score":
                values = [sum(rate - goal * math.log(rate) + math.lgamma(goal + 1)
                    for goal, rate in zip(row["actual_score"], row["rates"])) for row in tracks[arm]]
                metric = "mean_nll"
            else:
                p, y = vectors[domain]
                values = [-math.log(p[i, label]) for i, label in enumerate(y)]
                metric = "log_loss"
            assert result["scores"][arm][domain][metric] == pytest.approx(np.mean(values))
            assert result["scores"][arm][domain]["n"] == result["n"] == 6
    for method in experiment.METHODS:
        for domain in experiment.DOMAINS:
            metric = "mean_nll" if domain == "exact_score" else "log_loss"
            assert result["mean_differences"][method + "_minus_original"][domain] == pytest.approx(
                result["scores"][method][domain][metric] - result["scores"]["original"][domain][metric])
    for groups in (result["by_league"], result["by_month"]):
        assert sum(g["n"] for g in groups.values()) == 6


def test_joint_blocks_include_all_domains_moments_and_probability_biases(monkeypatch):
    calls = []
    def observed(dates, matrix, **kwargs):
        calls.append((list(dates), np.asarray(matrix).copy(), kwargs))
        return moving_block_means(dates, matrix, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result, tracks, stats, _ = analysis()
    per_loss = experiment.loss_vectors(tracks, stats)
    assert [call[2]["block_weeks"] for call in calls[:3]] == [2, 4, 8]
    assert len(calls) == 5  # Three complete-cohort calls, two market-specific calls.
    for weeks, (dates, matrix, kwargs) in zip((2, 4, 8), calls):
        assert matrix.shape == (6, 50)
        np.testing.assert_array_equal(matrix, calls[0][1])
        oracle = moving_block_means(dates, matrix, **kwargs)
        saved = result["bootstrap"][str(weeks)]
        assert saved["n_calendar_weeks"] > saved["n_nonempty_weeks"]
        for i, (name, item) in enumerate(saved["features"].items()):
            assert item["mean"] == pytest.approx(oracle["mean"][i])
            assert item["interval_95"] == oracle["interval_95"][i]
            for method in experiment.METHODS:
                prefix = method + "_minus_original_"
                if name.startswith(prefix):
                    domain = name.removeprefix(prefix)
                    np.testing.assert_array_equal(matrix[:, i], per_loss[method][domain] - per_loss["original"][domain])
        for arm in experiment.ARMS:
            for field in experiment.MOMENT_FIELDS:
                assert f"{arm}_{field}" in saved["features"]
            for outcome in ("home", "draw", "away"):
                assert f"{arm}_{outcome}_probability_minus_frequency" in saved["features"]


def test_market_missingness_changes_only_its_matched_market_cohort():
    result, tracks, _, _ = analysis()
    assert result["markets"]["avg_preclosing"]["n"] == 6
    assert result["markets"]["avg_closing"]["n"] == 5
    assert result["markets"]["avg_closing"]["missing_ids"] == ["new-0"]
    assert result["markets"]["b365_preclosing"] == {"n": 0, "missing_ids": result["match_ids"]}
    rows = later_rows()
    for row in rows:
        row["market_prices"], row["market_probabilities"] = {}, {}
    raw, stats, _ = research(rows)
    no_market, _, _ = experiment.analyze_later(raw, stats, repeats=37)
    assert no_market["scores"] == result["scores"]
    assert no_market["mean_differences"] == result["mean_differences"]
    assert no_market["match_ids"] == result["match_ids"]


def test_csv_and_exported_json_match_all_24_scalars_probabilities_and_four_losses(tmp_path):
    result, tracks, stats, exported = analysis()
    path = tmp_path / "paired.csv"
    experiment.write_csv(path, tracks, stats)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    vectors = experiment.loss_vectors(tracks, stats)
    for i, row in enumerate(saved):
        assert row["match_id"] == result["match_ids"][i]
        assert row["calibration_before_date"] == "2025-08-01"
        for arm in experiment.ARMS:
            for field in STAT_FIELDS:
                assert float(row[f"{arm}_{field}"]) == stats[arm][i][field]
            for c, label in enumerate(("home", "draw", "away")):
                assert float(row[f"{arm}_p_{label}"]) == tracks[arm][i]["probabilities"][c]
            for domain in experiment.DOMAINS:
                assert float(row[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]
            assert exported[i]["research_forecasts"][arm] == tracks[arm][i]["forecast"]
    for arm in experiment.ARMS:
        assert np.mean([float(row[f"{arm}_joint_score_nll"]) for row in saved]) == pytest.approx(
            result["scores"][arm]["exact_score"]["mean_nll"])
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported


def test_zero_base_fit_run_finishes_both_earlier_calibrators_before_loading_later_csv(tmp_path, monkeypatch):
    parent, _, paths, events = install_rate_runner(tmp_path, monkeypatch)
    unchanged = {path: path.read_bytes() for path in paths.values()}
    output = experiment.run(tmp_path, parent, repeats=7)
    result = read_json(output / experiment.RESULT_FILE)
    assert read_json(output / "实验设置.json") == result["settings"]
    assert read_json(output / "进球率校准参数.json") == result["parameters"]
    later_load = next(i for i, event in enumerate(events) if event[0] == "load" and "2025/26" in event[1])
    assert [(e[1], e[2]) for e in events[:later_load] if e[0] == "calibrate"] == [("shared_scale", 5), ("side_scale", 5)]
    assert len([e for e in events if e[0] == "rate_validate"]) == 10
    assert result["counts"] == {"new_rate_calibrator_fits": 2, "new_scalar_rate_parameters": 3,
        "new_base_model_fits": 0, "source_parameter_snapshots_read": 10, "source_prediction_rows": 10,
        "new_research_forecasts": 10, "reused_later_source_forecasts": 5}
    assert result["earlier"]["n"] == result["later"]["n"] == 5
    assert result["production_api_changed"] is False
    assert not result["settings"]["market_fusion"] and not result["settings"]["model_selection"]
    assert all(path.read_bytes() == content for path, content in unchanged.items())
    assert (output / experiment.FORECAST_FILE).exists()
    assert (output / "较晚逐场概率与进球残差.csv").exists()
    report = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    for boundary in ("0次基础模型重训", "不是新盲测", "净胜至少3", "不是亚洲盘走盘", "不是验证成功", "无优化器"):
        assert boundary in report


def test_changed_valid_later_labels_and_odds_cannot_relearn_the_earlier_scales(tmp_path, monkeypatch):
    parent, sources, paths, _ = install_rate_runner(tmp_path, monkeypatch)
    first = experiment.run(tmp_path, parent, repeats=5)
    expected = read_json(first / "进球率校准参数.json")
    path = paths[("2025-26", "E0", 28)]
    trial = read_json(path)
    row = trial["predictions"][0]
    row.update(actual_score=[9, 0], actual_outcome=0,
               market_prices={"avg_preclosing": [3., 4., 5.]})
    inverse = 1 / np.asarray([3., 4., 5.])
    row["market_probabilities"] = {"avg_preclosing": (inverse / inverse.sum()).tolist()}
    path.write_text(json.dumps(trial), encoding="utf-8")
    index = next(i for i, r in enumerate(sources) if r.match_id == "2025-E0")
    sources[index] = replace(sources[index], full_time=(9, 0), markets={"avg_preclosing": (3., 4., 5.)})
    second = experiment.run(tmp_path, parent, repeats=5)
    assert read_json(second / "进球率校准参数.json") == expected
    assert read_json(first / experiment.RESULT_FILE)["later"]["scores"] != read_json(second / experiment.RESULT_FILE)["later"]["scores"]


def test_broken_source_or_parameter_archive_is_not_repaired_or_overwritten(tmp_path, monkeypatch):
    parent, _, paths, events = install_rate_runner(tmp_path, monkeypatch)
    path = paths[("2024-25", "D1", 28)]
    trial = read_json(path)
    trial["fits"][0]["coefficients"]["home:D1"] += .1
    path.write_text(json.dumps(trial), encoding="utf-8")
    unchanged = path.read_bytes()
    with pytest.raises(ValueError):
        experiment.run(tmp_path, parent, repeats=5)
    assert path.read_bytes() == unchanged
    assert not any(e[0] == "calibrate" for e in events)
    assert not list((tmp_path / "outputs").glob("进球率调整研究_*/进球率调整研究结果.json"))


def test_default_parent_and_missing_parent_never_trigger_new_base_fitting(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="completed refresh"):
        experiment.run(tmp_path)
    parent, _, _, _ = install_rate_runner(tmp_path, monkeypatch)
    output = experiment.run(tmp_path, repeats=5)
    assert read_json(output / "实验设置.json")["parent_directory"] == str(parent.resolve())
    for invalid in (True, 1, 1.5):
        with pytest.raises(ValueError, match="repeats"):
            experiment.run(tmp_path, parent, repeats=invalid)


def test_unit_scale_fits_leave_entire_score_distributions_and_paired_intervals_unchanged():
    rows = earlier_rows()
    for row in rows:
        row["rates"] = list(row["actual_score"])
    tracks, stats, _ = research(parameters=experiment.fit_earlier(rows))
    result, _, _ = experiment.analyze_later(tracks, stats, repeats=13)
    for method in experiment.METHODS:
        assert tracks[method] == tracks["original"]
        assert set(result["mean_differences"][method + "_minus_original"].values()) == {0.}
    for boot in result["bootstrap"].values():
        for name, value in boot["features"].items():
            if "_minus_original_" in name:
                assert value == {"mean": 0., "interval_95": [0., 0.]}


def test_inconsistent_statistics_and_duplicate_ids_are_not_hidden_by_alignment():
    tracks, stats, _ = research()
    stats["side_scale"].reverse()
    with pytest.raises(ValueError, match="disagree"):
        experiment.analyze_later(tracks, stats, repeats=5)
    tracks, stats, _ = research()
    tracks["shared_scale"].append(deepcopy(tracks["shared_scale"][0]))
    stats["shared_scale"].append(deepcopy(stats["shared_scale"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.analyze_later(tracks, stats, repeats=5)
