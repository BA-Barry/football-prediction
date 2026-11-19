"""Saved-rate runner tests; synthetic archives do not replace historical evidence."""

from copy import deepcopy
import csv
from dataclasses import replace
from datetime import date, timedelta
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from football_model import native_experiment
from football_model.date_model import DateTeamGoalModel
from football_model.football_data import HistoricalMatch, LEAGUES
from football_model.forecast_diagnostics import align_tracks
from football_model.goal_rate_diagnostics import STAT_FIELDS, row_statistics, summary
import football_model.goal_rate_experiment as experiment
from football_model.probability import score_grid, summarize
import football_model.refresh_research as refresh


def prediction(key, day, score=(2, 1), comp="E0", interval=28, rates=(1.5, 1.1)):
    played = date.fromisoformat(day)
    anchor = date(played.year if played.month >= 8 else played.year - 1, 8, 1)
    cutoff = anchor + timedelta(days=(played - anchor).days // interval * interval)
    forecast = summarize(score_grid(*rates, rho=0))
    return {
        "match_id": key, "date": day, "competition": comp, "home": "A", "away": "B",
        "fit_before_date": cutoff.isoformat(), "training_max_date": (cutoff - timedelta(days=1)).isoformat(),
        "training_matches": 10, "actual_score": list(score),
        "actual_outcome": 0 if score[0] > score[1] else 2 if score[0] < score[1] else 1,
        "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
        "source_url": "https://example.invalid/synthetic.csv", "market_prices": {}, "market_probabilities": {},
    }


def example_tracks():
    days = ("2025-08-04", "2025-08-11", "2025-08-11", "2025-09-08",
            "2025-10-06", "2025-11-03", "2025-11-10")
    scores = ((0, 0), (1, 1), (0, 3), (1, 2), (4, 2), (2, 1), (2, 2))
    competitions = ("E0", "E0", "D1", "E0", "D1", "I1", "E0")
    base_home = (.5, .75, 1., 1.25, 1.5, 2., 3.)
    tracks, statistics = {}, {}
    for arm, interval in zip(experiment.ARMS, experiment.INTERVALS):
        tracks[arm], statistics[arm] = [], {}
        for i, (day, score, comp) in enumerate(zip(days, scores, competitions)):
            rates = (base_home[i] + (28 - interval) / 20, 1.1 + i / 10 + (28 - interval) / 100)
            row = prediction(f"m{i}", day, score, comp, interval, rates)
            tracks[arm].append(row)
            statistics[arm][row["match_id"]] = row_statistics(row, rates)
    return tracks, statistics


def analyze(tracks=None, statistics=None, repeats=37):
    if tracks is None:
        tracks, statistics = example_tracks()
    return experiment.analyze_period(tracks, statistics, repeats=repeats)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_common_alignment_and_extra_coverage_statistics_are_separate_and_nonmutating():
    tracks, stats = example_tracks()
    extra = prediction("extra", "2025-08-17", (0, 8), interval=7, rates=(.3, 4.))
    tracks["days_7"].append(extra)
    stats["days_7"]["extra"] = row_statistics(extra, (.3, 4.))
    for arm in tracks:
        tracks[arm].reverse()  # Alignment follows date and match ID, not input position.
    before = deepcopy((tracks, stats))
    result, entries, aligned_stats = analyze(tracks, stats)
    assert result["n"] == 7
    assert result["coverage"]["counts"] == {"days_7": 8, "days_14": 7, "days_28": 7}
    assert result["coverage"]["outside_common"] == {"days_7": ["extra"], "days_14": [], "days_28": []}
    assert result["extra_coverage"]["days_7"] == {
        "match_ids": ["extra"], "summary_separate_not_comparable_to_common": summary([extra], [stats["days_7"]["extra"]])}
    assert result["extra_coverage"]["days_28"]["summary_separate_not_comparable_to_common"] is None
    for arm in experiment.ARMS:
        assert [r["match_id"] for r in entries[arm]] == result["coverage"]["common_match_ids"]
        assert len(aligned_stats[arm]) == 7
        assert result["summaries"][arm] == summary(entries[arm], aligned_stats[arm])
    assert (tracks, stats) == before


def test_requires_all_declared_arms_with_exact_complete_statistics_id_sets():
    tracks, stats = example_tracks()
    with pytest.raises(ValueError, match="three"):
        analyze({key: value for key, value in tracks.items() if key != "days_14"}, stats)
    for action in ("missing", "extra"):
        broken = deepcopy(stats)
        if action == "missing":
            del broken["days_7"]["m0"]
        else:
            broken["days_7"]["not-predicted"] = broken["days_7"]["m0"]
        with pytest.raises(ValueError, match="IDs differ"):
            analyze(tracks, broken)


def test_duplicate_or_conflicting_source_identity_is_never_hidden_by_intersection():
    tracks, stats = example_tracks()
    tracks["days_7"].append(deepcopy(tracks["days_7"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(tracks, stats)
    tracks, stats = example_tracks()
    tracks["days_7"][0]["source_url"] = "https://example.invalid/different.csv"
    with pytest.raises(ValueError, match="Conflicting"):
        analyze(tracks, stats)


def test_bad_extra_coverage_statistic_is_rejected_even_if_not_in_common_matches():
    tracks, stats = example_tracks()
    extra = prediction("extra", "2025-08-17", (0, 8), interval=7)
    tracks["days_7"].append(extra)
    stats["days_7"]["extra"] = row_statistics(extra, (1.5, 1.1))
    stats["days_7"]["extra"]["home_residual"] = 123
    with pytest.raises(ValueError, match="disagree"):
        analyze(tracks, stats)


def test_fixed_28_rate_bins_use_same_members_for_all_arms_and_keep_empty_bins():
    tracks, stats = example_tracks()
    result, entries, aligned_stats = analyze(tracks, stats)
    expected_labels = ("0-0.75", "0.75-1", "1-1.25", "1.25-1.5", "1.5-2", "2-3", "3+")
    assert experiment.BIN_NAMES == expected_labels
    home = result["fixed_28_rate_bins"]["home"]
    assert tuple(home) == expected_labels
    for i, label in enumerate(expected_labels):
        assert home[label]["n"] == 1
        for arm in experiment.ARMS:
            assert home[label]["summaries"][arm] == summary([entries[arm][i]], [aligned_stats[arm][i]])
    # All away rates are >=1.1; retain empty bins explicitly rather than omit them.
    away = result["fixed_28_rate_bins"]["away"]
    assert away["0-0.75"] == {"n": 0, "summaries": None, "mean_nll_differences": None}
    assert away["3+"]["n"] == 0
    for side in ("home", "away"):
        assert sum(group["n"] for group in result["fixed_28_rate_bins"][side].values()) == 7


def test_league_month_and_nll_contributions_recompose_common_cohort():
    result, entries, stats = analyze()
    assert set(result["by_league"]) == {"E0", "D1", "I1"}
    assert set(result["by_month"]) == {"2025-08", "2025-09", "2025-10", "2025-11"}
    for arm in experiment.ARMS:
        for field in experiment.NLL_FIELDS:
            expected = np.mean([r[field] for r in stats[arm]])
            assert result["summaries"][arm]["scalar_descriptives"][field]["mean"] == pytest.approx(expected)
            for grouping in ("by_league", "by_month"):
                recomposed = sum(g["n"] * g["summaries"][arm]["scalar_descriptives"][field]["mean"]
                                 for g in result[grouping].values()) / result["n"]
                assert recomposed == pytest.approx(expected)
    for arm in experiment.CONTRASTS:
        for field in experiment.NLL_FIELDS:
            expected = np.mean([a[field] - b[field] for a, b in zip(stats[arm], stats["days_28"])])
            assert result["mean_nll_differences"][f"{arm}_minus_days_28"][field] == pytest.approx(expected)


def independent_interval(dates, matrix, length, repeats, seed):
    """Ordinary direct loop: same calendar-block resample for every column."""
    days = [date.fromisoformat(d) for d in dates]
    monday = min(days) - timedelta(days=min(days).weekday())
    week_id = [(d - monday).days // 7 for d in days]
    weeks = max(week_id) + 1
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, weeks - length + 1, size=(repeats, math.ceil(weeks / length)))
    means = []
    for draw in starts:
        sequence = [week for start in draw for week in range(start, start + length)][:weeks]
        indices = [i for week in sequence for i, member in enumerate(week_id) if member == week]
        if indices:
            means.append(np.mean(matrix[indices], axis=0))
    return np.quantile(means, [.025, .975], axis=0).T


def test_bootstrap_uses_one_joint_matrix_and_paired_differences_for_all_lengths(monkeypatch):
    calls = []
    original = experiment.moving_block_means
    def observed(dates, values, **kwargs):
        calls.append((list(dates), np.asarray(values).copy(), kwargs))
        return original(dates, values, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result, _, stats = analyze(repeats=43)
    assert len(calls) == 3
    assert [call[2]["block_weeks"] for call in calls] == [2, 4, 8]
    for length, (dates, matrix, kwargs) in zip(experiment.BLOCKS, calls):
        assert matrix.shape == (7, 48)
        assert kwargs["repeats"] == 43 and kwargs["seed"] == experiment.SEED
        np.testing.assert_array_equal(matrix, calls[0][1])
        boot = result["bootstrap"][str(length)]
        assert boot["n_calendar_weeks"] > boot["n_nonempty_weeks"]
        assert 0 in boot["weekly_match_counts"]
        expected = independent_interval(dates, matrix, length, 43, experiment.SEED)
        for index, (name, data) in enumerate(boot["features"].items()):
            assert data["mean"] == pytest.approx(matrix[:, index].mean())
            assert data["interval_95"] == pytest.approx(expected[index].tolist())
            for arm in experiment.CONTRASTS:
                prefix = f"{arm}_minus_days_28_"
                if name.startswith(prefix):
                    field = name.removeprefix(prefix)
                    np.testing.assert_allclose(matrix[:, index], [
                        a[field] - b[field] for a, b in zip(stats[arm], stats["days_28"])], rtol=0, atol=0)


def test_identical_rate_arms_produce_zero_paired_differences_and_intervals():
    tracks, stats = example_tracks()
    for arm in experiment.CONTRASTS:
        stats[arm] = deepcopy(stats["days_28"])
    result, _, _ = analyze(tracks, stats)
    for values in result["mean_nll_differences"].values():
        assert set(values.values()) == {0.}
    for boot in result["bootstrap"].values():
        for name, values in boot["features"].items():
            if "_minus_days_28_" in name:
                assert values == {"mean": 0., "interval_95": [0., 0.]}


def test_one_calendar_week_has_explicitly_unavailable_intervals():
    tracks, stats = example_tracks()
    for arm in experiment.ARMS:
        tracks[arm] = tracks[arm][:1]
        stats[arm] = {"m0": stats[arm]["m0"]}
    result, _, _ = analyze(tracks, stats)
    for boot in result["bootstrap"].values():
        assert boot["unavailable_reason"] == "block_length_not_smaller_than_calendar_span"
        assert all(data["interval_95"] is None for data in boot["features"].values())


def test_csv_exact_common_row_fields_roundtrip_and_summary_consistency(tmp_path):
    result, entries, stats = analyze()
    output = tmp_path / "rows.csv"
    experiment.write_csv(output, entries, stats)
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert [r["match_id"] for r in saved] == result["coverage"]["common_match_ids"]
    for i, row in enumerate(saved):
        assert [int(row["home_goals"]), int(row["away_goals"])] == entries["days_28"][i]["actual_score"]
        assert row["fixed_28_home_rate_bin"] == experiment.BIN_NAMES[i]
        for arm in experiment.ARMS:
            assert row[f"{arm}_fit_before_date"] == entries[arm][i]["fit_before_date"]
            assert int(row[f"{arm}_training_matches"]) == entries[arm][i]["training_matches"]
            for field in STAT_FIELDS:
                assert float(row[f"{arm}_{field}"]) == stats[arm][i][field]
    for arm in experiment.ARMS:
        for field in STAT_FIELDS:
            assert np.mean([float(row[f"{arm}_{field}"]) for row in saved]) == pytest.approx(
                result["summaries"][arm]["scalar_descriptives"][field]["mean"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def make_source(key, day, comp, score=(2, 1)):
    return HistoricalMatch(key, comp, "synthetic", date.fromisoformat(day), None,
        f"football-data:{comp}:A", f"football-data:{comp}:B", "A", "B", tuple(score), None,
        {}, "synthetic/not-a-source.csv", "https://example.invalid/synthetic.csv", None)


def install_archive_environment(tmp_path, monkeypatch):
    """Handmade readable parameters; real archive validation, never optimization."""
    parent = tmp_path / "outputs" / "更新频率研究_synthetic"
    parent.mkdir(parents=True)
    seasons = {season: tmp_path / season.replace("/", "-") for season in
               ("2022/23", "2023/24", "2024/25", "2025/26")}
    sources, trials, events, prior_periods = [], {}, [], {}
    for comp in LEAGUES:
        sources.extend([make_source(f"history-{comp}", "2023-06-01", comp),
                        make_source(f"2024-{comp}", "2024-08-15", comp),
                        make_source(f"2025-{comp}", "2025-08-15", comp, (1, 1))])
    for period, (start, end) in experiment.PERIODS.items():
        folder = parent / period
        folder.mkdir()
        tracks = {arm: [] for arm in experiment.ARMS}
        for comp in LEAGUES:
            target = next(r for r in sources if r.competition == comp and start <= r.match_date < end)
            for days in experiment.INTERVALS:
                rates = (1.5 + (28 - days) / 100, 1.1)
                prediction_row = prediction(target.match_id, target.match_date.isoformat(), target.full_time,
                                            comp, days, rates)
                cutoff = date.fromisoformat(prediction_row["fit_before_date"])
                history = sorted([r for r in sources if r.competition == comp and r.match_date < cutoff],
                                 key=lambda r: (r.match_date, r.match_id))
                teams = sorted({t for r in history for t in (r.home, r.away)})
                coefficients = {f"{kind}:{team}": 0. for team in teams for kind in ("attack", "defense")}
                coefficients.update({f"competition:{comp}": 0., f"home:{comp}": math.log(rates[0] / rates[1])})
                parameters = {
                    "model_type": "date_level_regularized_poisson", "regularization": .01,
                    "half_life_days": 180, "before_date": cutoff.isoformat(),
                    "training_min_date": history[0].match_date.isoformat(),
                    "training_max_date": history[-1].match_date.isoformat(), "n_matches": len(history),
                    "teams": teams, "competitions": [comp], "intercept": math.log(rates[1]),
                    "coefficients": coefficients, "optimizer_iterations": 0,
                    "reference_match_ids": [r.match_id for r in history], "basis": "team_marginal", "rho": 0,
                    "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
                }
                prediction_row.update(training_matches=len(history), training_max_date=parameters["training_max_date"])
                trial = {"config": dict(native_experiment.ORIGINAL), "start": start.isoformat(),
                    "end_exclusive": end.isoformat(), "update_interval_days": days,
                    "fits": [parameters], "predictions": [prediction_row], "failures": [], "skipped": []}
                path = folder / f"{comp}_days_{days}.json"
                path.write_text(json.dumps(trial, allow_nan=False), encoding="utf-8")
                trials[(period, comp, days)] = path
                tracks[f"days_{days}"].append(prediction_row)
        prior_periods[period] = {"coverage": align_tracks(tracks)[1]}
    prior = {"settings": {"intervals_days": list(experiment.INTERVALS)}, "periods": prior_periods}
    (parent / "更新频率研究结果.json").write_text(json.dumps(prior), encoding="utf-8")
    def load(selected):
        labels = tuple(selected)
        events.append(("load", labels))
        maximum = max(int(season[:4]) for season in labels)
        return [r for r in sources if r.match_date.year <= maximum], {s: {"synthetic": True} for s in labels}
    actual_validate = experiment.validate_trial
    def validate(trial, records, start, end, comp, interval):
        events.append(("validate", start.year, comp, interval))
        return actual_validate(trial, records, start, end, comp, interval)
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only diagnostics must not fit, predict anew, or load pickle")
    monkeypatch.setattr(experiment, "season_directories", lambda root: seasons)
    monkeypatch.setattr(experiment, "load_seasons", load)
    monkeypatch.setattr(experiment, "validate_trial", validate)
    monkeypatch.setattr(DateTeamGoalModel, "fit", forbidden)
    monkeypatch.setattr(native_experiment, "make_model", forbidden)
    monkeypatch.setattr(native_experiment, "run_config", forbidden)
    monkeypatch.setattr(refresh, "run_frequency", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    return parent, sources, trials, events


def test_zero_fit_archive_run_validates_all_30_trials_and_writes_consistent_outputs(tmp_path, monkeypatch):
    parent, sources, paths, events = install_archive_environment(tmp_path, monkeypatch)
    original_bytes = {path: path.read_bytes() for path in paths.values()}
    output = experiment.run(tmp_path, parent, repeats=17)
    result = read_json(output / experiment.RESULT_FILE)
    assert read_json(output / "分析设置.json") == result["settings"]
    assert result["counts"] == {"parameter_snapshots_read": 30, "saved_prediction_rows_validated": 30,
        "common_arm_diagnostic_rows": 30, "unique_common_matches": 10, "saved_trials_read": 30}
    assert result["new_model_fits"] == result["new_model_predictions"] == 0
    assert result["probabilities_changed"] is False
    assert result["settings"]["model_selection"] is False
    assert result["settings"]["rate_bin_edges"] == list(experiment.RATE_EDGES)
    assert result["settings"]["primary_block_weeks"] == 4
    assert [e for e in events if e[0] == "load"] == [
        ("load", ("2022/23", "2023/24", "2024/25")),
        ("load", ("2022/23", "2023/24", "2024/25", "2025/26"))]
    assert len([e for e in events if e[0] == "validate"]) == 30
    assert read_json(output / "阶段进度.json")["counts"] == result["counts"]
    for period, data in result["periods"].items():
        assert data["n"] == len(LEAGUES)
        with (output / period / "逐场进球率与残差.csv").open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert [r["match_id"] for r in csv_rows] == data["coverage"]["common_match_ids"]
        for arm in experiment.ARMS:
            for field in STAT_FIELDS:
                assert np.mean([float(r[f"{arm}_{field}"]) for r in csv_rows]) == pytest.approx(
                    data["summaries"][arm]["scalar_descriptives"][field]["mean"])
    assert all(path.read_bytes() == content for path, content in original_bytes.items())
    text = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    for boundary in ("0次拟合", "探索性", "不是三倍独立样本", "不是纯过度离散", "不是相关系数",
                     "不可估计", "半全场", "2024-25", "2025-26"):
        assert boundary in text


def test_default_parent_selects_completed_refresh_archive_without_training(tmp_path, monkeypatch):
    parent, _, _, _ = install_archive_environment(tmp_path, monkeypatch)
    output = experiment.run(tmp_path, repeats=5)
    assert read_json(output / "分析设置.json")["parent_directory"] == str(parent.resolve())


@pytest.mark.parametrize("damage", ["coefficient", "source", "missing-file", "coverage"])
def test_invalid_archive_or_source_is_not_skipped_repaired_or_refitted(tmp_path, monkeypatch, damage):
    parent, sources, paths, _ = install_archive_environment(tmp_path, monkeypatch)
    path = paths[("2024-25", "D1", 7)]
    if damage == "coefficient":
        trial = read_json(path)
        trial["fits"][0]["regularization"] = .5
        path.write_text(json.dumps(trial), encoding="utf-8")
    elif damage == "source":
        index = next(i for i, row in enumerate(sources) if row.match_id == "2024-D1")
        sources[index] = replace(sources[index], source_url="https://example.invalid/changed.csv")
    elif damage == "missing-file":
        # Change the expected path rather than deleting a material artifact.
        parent = tmp_path / "missing-parent"
        parent.mkdir()
        (parent / "更新频率研究结果.json").write_text(
            json.dumps({"settings": {"intervals_days": list(experiment.INTERVALS)}}), encoding="utf-8")
    else:
        prior = read_json(parent / "更新频率研究结果.json")
        prior["periods"]["2024-25"]["coverage"]["counts"]["days_7"] += 1
        (parent / "更新频率研究结果.json").write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        experiment.run(tmp_path, parent, repeats=5)
    assert not list((tmp_path / "outputs").glob("进球率诊断_*/进球率诊断结果.json"))


def test_duplicate_and_nonfinite_json_numbers_are_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            experiment._read_json(path)


def test_missing_parent_and_invalid_repeats_do_not_start_work(tmp_path):
    with pytest.raises(ValueError, match="completed refresh"):
        experiment.run(tmp_path)
    for repeats in (True, 1, 1.5):
        with pytest.raises(ValueError, match="repeats"):
            experiment.run(tmp_path, repeats=repeats)
    assert not (tmp_path / "outputs").exists()
