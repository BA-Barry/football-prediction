"""Read-only shape research keeps projected scores separate from probabilities."""

from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor

from football_model import core_tuning, core_tuning_experiment
from football_model.date_model import DateTeamGoalModel
from football_model.forecast_diagnostics import moving_block_means
import football_model.goal_rate_experiment as archive_fixture_target
from football_model.penaltyblog_adapter import NativeGoalModel
from football_model.probability import score_grid, summarize
from football_model import score_shape_experiment as experiment
from football_model.score_shape import EVENTS, STAT_FIELDS, row_shape_statistics
from football_model.team_model import TeamGoalModel

from test_goal_rate_experiment import install_archive_environment, prediction
from test_score_shape import indicator


def tracks():
    days = ("2025-08-04", "2025-08-11", "2025-08-11", "2025-09-08",
            "2025-10-06", "2025-11-03", "2025-11-10")
    scores = ((8, 0), (1, 1), (0, 3), (1, 2), (4, 2), (2, 1), (2, 2))
    competitions = ("E0", "E0", "D1", "E0", "D1", "I1", "E0")
    means = (.5, .75, 1., 1.25, 1.5, 2., 3.)
    result = {arm: [] for arm in experiment.ARMS}
    for i, (day, score, comp, home) in enumerate(zip(days, scores, competitions, means)):
        for arm in experiment.ARMS:
            rates = (home if arm == "original" else home * 1.6,
                     1.1 if arm == "original" else .7)
            row = prediction(f"m{i}", day, score, comp, 28, rates)
            row["rates"] = list(rates)
            result[arm].append(row)
    return result


def analysis(data=None, repeats=23):
    return experiment.analyze(tracks() if data is None else data, repeats=repeats)


def test_all_17_diagnostics_per_arm_match_per_row_statistics_without_mutation():
    data = tracks()
    before = deepcopy(data)
    result, aligned, stats = analysis(data)
    assert data == before
    assert result["n"] == 7 and len(experiment.DIAGNOSTICS) == 17
    assert len(STAT_FIELDS) == 39
    assert result["coverage"]["counts"] == {"original": 7, "selected": 7}
    for arm in experiment.ARMS:
        assert stats[arm] == [row_shape_statistics(row, row["rates"]) for row in aligned[arm]]
        for field in experiment.DIAGNOSTICS:
            values = np.asarray([s[field] for s in stats[arm]])
            summary = result["summaries"][arm][field]
            assert summary["mean"] == pytest.approx(values.mean())
            assert summary["sd"] == pytest.approx(values.std(ddof=1))
            assert summary["median"] == pytest.approx(np.median(values))
            assert summary["min"] == min(values) and summary["max"] == max(values)


def test_dispersions_and_projected_residuals_are_not_clipped_to_probability_ranges():
    result, aligned, stats = analysis()
    row, stat = aligned["original"][0], stats["original"][0]
    h, a = row["actual_score"]
    home, away = row["rates"]
    assert stat["home_dispersion_residual"] == pytest.approx(((h-home)**2-h)/home)
    assert stat["away_dispersion_residual"] == pytest.approx(((a-away)**2-a)/away)
    assert stat["home_dispersion_residual"] > 1
    assert stat["home_zero_mean_orthogonal_residual"] > 1
    for arm in experiment.ARMS:
        for statistic in stats[arm]:
            assert all(0 <= statistic[f"{event}_probability"] <= 1 for event in EVENTS)
        assert result["summaries"][arm]["home_zero_mean_orthogonal_residual"]["max"] > 1


def test_event_counts_use_raw_probabilities_and_actual_indicators_not_projected_scores():
    result, aligned, stats = analysis()
    for arm in experiment.ARMS:
        for event in EVENTS:
            expected_observed = sum(indicator(event, *row["actual_score"]) for row in aligned[arm])
            expected_count = math.fsum(stat[f"{event}_probability"] for stat in stats[arm])
            assert result["event_counts"][arm][event] == {
                "observed": expected_observed, "expected": pytest.approx(expected_count)}
    assert result["event_counts"]["original"]["home_zero"]["observed"] == 1
    # Deliberately make projections nonsensical; event counts still derive only
    # from raw I-p and p, and never interpret z+p as a new observed frequency.
    changed = deepcopy(stats)
    for arm in experiment.ARMS:
        for statistic in changed[arm]:
            for event in EVENTS:
                statistic[f"{event}_mean_orthogonal_residual"] = 900.
    local = experiment.cohort(changed, list(range(7)))
    assert local["event_counts"] == result["event_counts"]
    assert local["summaries"]["original"]["home_zero_mean_orthogonal_residual"]["mean"] == 900.


def test_alignment_is_by_match_identity_not_input_position_and_keeps_sources_unchanged():
    expected, expected_rows, expected_stats = analysis()
    data = tracks()
    data["original"] = data["original"][3:] + data["original"][:3]
    data["selected"].reverse()
    before = deepcopy(data)
    result, rows, stats = analysis(data)
    assert result == expected and rows == expected_rows and stats == expected_stats
    assert data == before


@pytest.mark.parametrize("change", ["missing_arm", "missing_match", "extra_match", "duplicate",
                                   "source_url", "score", "date", "market", "rates", "forecast", "probabilities"])
def test_missing_conflicting_or_incoherent_rows_are_not_hidden_by_intersection(change):
    data = tracks()
    if change == "missing_arm":
        data.pop("selected")
    elif change == "missing_match":
        data["selected"].pop()
    elif change == "extra_match":
        data["selected"].append({**deepcopy(data["selected"][0]), "match_id": "extra"})
    elif change == "duplicate":
        data["selected"].append(deepcopy(data["selected"][0]))
    elif change == "source_url":
        data["selected"][0]["source_url"] = "https://example.invalid/other.csv"
    elif change == "score":
        data["selected"][0]["actual_score"] = [9, 0]
    elif change == "date":
        data["selected"][0]["date"] = "2025-08-05"
    elif change == "market":
        data["selected"][0]["market_prices"] = {"avg_preclosing": [2., 4., 4.]}
        data["selected"][0]["market_probabilities"] = {"avg_preclosing": [.5, .25, .25]}
    elif change == "rates":
        data["selected"][0]["rates"][0] += .4
    elif change == "forecast":
        data["selected"][0]["forecast"]["total_goals"]["4+"] += .1
    elif change == "probabilities":
        data["selected"][0]["probabilities"] = [.5, .3, .2]
    with pytest.raises(ValueError):
        analysis(data)


def test_all_34_columns_use_one_joint_block_matrix_for_every_block_length(monkeypatch):
    calls = []
    def observed(dates, matrix, **kwargs):
        calls.append((list(dates), np.asarray(matrix).copy(), kwargs))
        return moving_block_means(dates, matrix, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result, aligned, stats = analysis()
    names = [f"{arm}_{field}" for arm in experiment.ARMS for field in experiment.DIAGNOSTICS]
    expected = np.column_stack([[s[field] for s in stats[arm]]
                               for arm in experiment.ARMS for field in experiment.DIAGNOSTICS])
    assert len(calls) == 3 and expected.shape == (7, 34)
    assert [call[2]["block_weeks"] for call in calls] == [2, 4, 8]
    for length, (dates, matrix, kwargs) in zip((2, 4, 8), calls):
        np.testing.assert_array_equal(matrix, expected)
        assert dates == [r["date"] for r in aligned["original"]]
        assert kwargs["repeats"] == 23 and kwargs["seed"] == experiment.SEED
        oracle = moving_block_means(dates, matrix, **kwargs)
        saved = result["bootstrap"][str(length)]
        assert list(saved["features"]) == names
        assert saved["n_matches"] == 7  # Not two arms or 34 features times n.
        assert saved["n_calendar_weeks"] > saved["n_nonempty_weeks"]
        assert 0 in saved["weekly_match_counts"]
        for i, name in enumerate(names):
            assert saved["features"][name]["mean"] == pytest.approx(expected[:, i].mean())
            assert saved["features"][name]["interval_95"] == oracle["interval_95"][i]


def test_fixed_original_rate_bins_use_the_same_members_for_both_policies_and_keep_empty_bins():
    result, aligned, stats = analysis()
    labels = ("0-0.75", "0.75-1", "1-1.25", "1.25-1.5", "1.5-2", "2-3", "3+")
    assert experiment.BIN_NAMES == labels
    home = result["fixed_original_rate_bins"]["home"]
    for i, label in enumerate(labels):
        assert home[label] == experiment.cohort(stats, [i])
        assert home[label]["n"] == 1
        for arm in experiment.ARMS:
            assert home[label]["summaries"][arm]["home_residual"]["mean"] == stats[arm][i]["home_residual"]
    # Original away means all equal 1.1, although selected means all equal .7.
    away = result["fixed_original_rate_bins"]["away"]
    assert away["1-1.25"]["n"] == 7
    assert away["0-0.75"] == {"n": 0, "summaries": None, "event_counts": None}
    assert away["0.75-1"]["n"] == away["3+"]["n"] == 0
    assert away["1-1.25"]["summaries"]["selected"]["away_residual"]["mean"] == pytest.approx(
        np.mean([row["actual_score"][1] - .7 for row in aligned["selected"]]))


def test_all_month_and_league_summaries_and_counts_recompose_the_same_matches():
    result, aligned, stats = analysis()
    for groups in (result["by_league"], result["by_month"]):
        assert sum(g["n"] for g in groups.values()) == 7
        for arm in experiment.ARMS:
            for field in experiment.DIAGNOSTICS:
                average = sum(g["n"] * g["summaries"][arm][field]["mean"] for g in groups.values())/7
                assert average == pytest.approx(result["summaries"][arm][field]["mean"])
            for event in EVENTS:
                observed = sum(g["event_counts"][arm][event]["observed"] for g in groups.values())
                expected = sum(g["event_counts"][arm][event]["expected"] for g in groups.values())
                assert observed == result["event_counts"][arm][event]["observed"]
                assert expected == pytest.approx(result["event_counts"][arm][event]["expected"])


def test_one_match_has_undefined_sample_sd_and_unavailable_block_intervals():
    data = {arm: rows[:1] for arm, rows in tracks().items()}
    result, _, _ = analysis(data, repeats=7)
    for arm in experiment.ARMS:
        assert all(item["sd"] is None for item in result["summaries"][arm].values())
    for boot in result["bootstrap"].values():
        assert boot["unavailable_reason"] == "block_length_not_smaller_than_calendar_span"
        assert all(item["interval_95"] is None for item in boot["features"].values())


def test_csv_retains_exact_39_statistics_per_arm_without_converting_z_to_probability(tmp_path):
    result, aligned, stats = analysis()
    path = tmp_path / "shape.csv"
    experiment.write_csv(path, aligned, stats)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields == ["match_id", "date", "competition", "home", "away"] + [
        f"{arm}_{field}" for arm in experiment.ARMS for field in STAT_FIELDS]
    assert len(fields) == 83 and len(rows) == 7
    assert [row["match_id"] for row in rows] == result["coverage"]["common_match_ids"]
    for i, row in enumerate(rows):
        for arm in experiment.ARMS:
            for field in STAT_FIELDS:
                assert float(row[f"{arm}_{field}"]) == stats[arm][i][field]
    assert float(rows[0]["original_home_zero_mean_orthogonal_residual"]) > 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def install_shape_archives(tmp_path, monkeypatch):
    old_parent, sources, old_paths, events = install_archive_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(core_tuning_experiment, "season_directories", archive_fixture_target.season_directories)
    monkeypatch.setattr(core_tuning_experiment, "load_seasons", archive_fixture_target.load_seasons)
    configs = core_tuning.candidate_configs()
    original_config = next(c for c in configs if c["id"] == experiment.ORIGINAL_ID)
    selected_config = next(c for c in configs if c["half_life_days"] == 365 and c["regularization"] == .001)
    parent = tmp_path / "outputs" / "核心参数训练_synthetic"
    parent.mkdir()
    (parent / experiment.POLICY_FILE).write_text(json.dumps({"config": selected_config}), encoding="utf-8")
    archive_paths = {}
    for period, (start, end) in experiment.PERIODS.items():
        records, _ = core_tuning_experiment.load_source(tmp_path, period)
        for comp in sorted(experiment.LEAGUES):
            local = [r for r in records if r.competition == comp]
            original = core_tuning_experiment.original_trial(local, old_parent, period, comp, original_config)
            selected = deepcopy(original)
            selected["config"] = dict(selected_config)
            selected.pop("reused_from")
            for fit in selected["fits"]:
                fit["regularization"] = selected_config["regularization"]
                fit["half_life_days"] = selected_config["half_life_days"]
            for arm, trial in (("original", original), ("selected", selected)):
                folder = parent / period / (trial["config"]["id"] if period == "2024-25" else arm)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{comp}.json"
                path.write_text(json.dumps(trial, allow_nan=False), encoding="utf-8")
                archive_paths[(period, arm, comp)] = path
    def forbidden(*args, **kwargs):
        raise AssertionError("Score shape research may not fit, generate forecasts, or load pickle")
    for model in (DateTeamGoalModel, TeamGoalModel, NativeGoalModel, PoissonRegressor):
        monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(DateTeamGoalModel, "predict", forbidden)
    monkeypatch.setattr(core_tuning, "run_candidate", forbidden)
    monkeypatch.setattr(core_tuning_experiment, "run_candidate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    return parent, archive_paths, events


def test_load_tracks_replays_both_policy_archives_with_earlier_source_history_and_zero_fits(tmp_path, monkeypatch):
    parent, paths, events = install_shape_archives(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in paths.values()}
    events.clear()
    raw, metadata = experiment.load_tracks(tmp_path, parent, "2024-25")
    assert {arm: len(rows) for arm, rows in raw.items()} == {"original": 5, "selected": 5}
    assert metadata["new_fits"] == metadata["new_forecasts"] == 0
    assert metadata["parameter_snapshots_read"] == {"original": 5, "selected": 5}
    loads = [event for event in events if event[0] == "load"]
    assert len(loads) == 1 and "2025/26" not in loads[0][1]
    for arm, rows in raw.items():
        assert all(row["training_max_date"] < row["fit_before_date"] <= row["date"] for row in rows)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("change", ["history_id", "history_size", "history_min", "history_max",
                                   "cutoff", "raw_rate", "coefficient", "period", "interval", "config_id",
                                   "policy_value"])
def test_load_tracks_refuses_corrupted_histories_parameters_or_policy_without_repair(tmp_path, monkeypatch, change):
    parent, paths, events = install_shape_archives(tmp_path, monkeypatch)
    path = paths[("2024-25", "selected", "D1")]
    trial = json.loads(path.read_text(encoding="utf-8"))
    if change == "history_id":
        trial["fits"][0]["reference_match_ids"][0] = "future-or-wrong-match"
    elif change == "history_size":
        trial["fits"][0]["n_matches"] += 1
    elif change == "history_min":
        trial["fits"][0]["training_min_date"] = "2023-05-31"
    elif change == "history_max":
        trial["fits"][0]["training_max_date"] = "2024-08-01"
    elif change == "cutoff":
        trial["predictions"][0]["fit_before_date"] = "2024-08-02"
    elif change == "raw_rate":
        trial["predictions"][0]["rates"][0] += .4
    elif change == "coefficient":
        trial["fits"][0]["coefficients"]["home:D1"] += .2
    elif change == "period":
        trial["end_exclusive"] = "2026-07-01"
    elif change == "interval":
        trial["update_interval_days"] = 7
    elif change == "config_id":
        trial["config"]["id"] = experiment.ORIGINAL_ID
    elif change == "policy_value":
        path = parent / experiment.POLICY_FILE
        trial = json.loads(path.read_text(encoding="utf-8"))
        trial["config"]["regularization"] = .1
    path.write_text(json.dumps(trial), encoding="utf-8")
    untouched = path.read_bytes()
    with pytest.raises(ValueError):
        experiment.load_tracks(tmp_path, parent, "2024-25")
    assert path.read_bytes() == untouched


def test_end_to_end_runner_is_zero_fit_zero_new_forecast_and_preserves_all_source_archives(tmp_path, monkeypatch):
    parent, paths, events = install_shape_archives(tmp_path, monkeypatch)
    untouched = {path: path.read_bytes() for path in paths.values()}
    policy = (parent / experiment.POLICY_FILE).read_bytes()
    events.clear()
    output = tmp_path / "shape-output"
    result = experiment.run(tmp_path, parent, output, repeats=11)
    assert result["counts"] == {"unique_match_ids": 10, "new_fits": 0, "new_forecasts": 0}
    assert result["settings"]["fit_new_model"] is False and result["settings"]["new_forecasts"] is False
    assert result["settings"]["production_changed"] is False
    assert result["settings"]["already_viewed_seasons"] is True
    assert result["settings"]["primary_policy"] == "original"
    assert [event[1] for event in events if event[0] == "load"] == [
        ("2022/23", "2023/24", "2024/25"), ("2022/23", "2023/24", "2024/25", "2025/26")]
    for period in experiment.PERIODS:
        local = result["periods"][period]
        assert local["n"] == 5
        assert result["source_metadata"][period]["parameter_snapshots_read"] == {"original": 5, "selected": 5}
        assert (output / f"{period}_逐场形态.csv").exists()
        assert json.loads((output / f"{period}_形态结果.json").read_text(encoding="utf-8")) == local
    assert json.loads((output / experiment.RESULT_FILE).read_text(encoding="utf-8")) == result
    assert json.loads((output / "诊断设置.json").read_text(encoding="utf-8")) == result["settings"]
    assert all(path.read_bytes() == content for path, content in untouched.items())
    assert (parent / experiment.POLICY_FILE).read_bytes() == policy
    report = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    for phrase in ("0次训练", "0条新预测", "不是概率", "选择自适应", "并非独立验证",
                   "不是去掉全部均值错误", "不把两臂当双倍样本", "非多指标联合检验"):
        assert phrase in report


def test_default_root_and_period_source_paths_are_project_local():
    root = Path(experiment.__file__).resolve().parents[3]
    assert root.name == "football-prediction-2026-10-22"
    assert (root / "model_v2" / "src" / "football_model").is_dir()
