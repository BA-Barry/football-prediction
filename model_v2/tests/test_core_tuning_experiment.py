"""Same-match scores and actual runner ordering for core parameter research."""

from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_model import core_tuning
import football_model.core_tuning_experiment as experiment
from football_model.evaluation import probability_scores
from football_model.forecast_diagnostics import moving_block_means
import football_model.goal_rate_experiment as archive_fixture_target
from football_model.probability import score_grid, summarize

from test_core_tuning import history
from test_goal_rate_experiment import install_archive_environment, prediction
from test_rate_calibration_experiment import later_rows


def tracks():
    original = later_rows()
    selected = deepcopy(original)
    for i, row in enumerate(selected):
        row["rates"] = [row["rates"][0] * 1.06, row["rates"][1] * .93]
        row["forecast"] = summarize(score_grid(*row["rates"], rho=0))
        row["probabilities"] = list(row["forecast"]["one_x_two"].values())
    return {"original": original, "selected": selected}


def analysis():
    return experiment.analyze(tracks(), repeats=19)


def test_load_source_filters_later_season_before_csv_read(monkeypatch, tmp_path):
    directories = {name: tmp_path / name.replace("/", "-") for name in
                   ("2022/23", "2023/24", "2024/25", "2025/26")}
    called = []
    monkeypatch.setattr(experiment, "season_directories", lambda root: directories)
    def read(selected):
        called.append(dict(selected))
        assert "2025/26" not in selected
        return ["not-real-source"], {"read": list(selected)}
    monkeypatch.setattr(experiment, "load_seasons", read)
    records, sources = experiment.load_source(tmp_path, "2024-25")
    assert records == ["not-real-source"]
    assert list(called[0]) == ["2022/23", "2023/24", "2024/25"]
    assert sources["read"] == list(called[0])


def test_default_root_depth_points_to_project_not_model_or_src_directory():
    root = Path(experiment.__file__).resolve().parents[3]
    assert root.name == "football-prediction-2026-10-22"
    assert (root / "model_v2" / "src" / "football_model").is_dir()


def replay_fixture():
    rows = history()
    config = core_tuning.candidate_configs()[0]
    trial = core_tuning.run_candidate(rows, config, date(2024, 8, 1), date(2024, 9, 26))
    return trial, rows


def test_actual_new_fit_rates_replay_without_new_fits_and_do_not_mutate(monkeypatch):
    trial, records = replay_fixture()
    before = deepcopy(trial)
    def forbidden(*args, **kwargs):
        raise AssertionError("Parameter replay must not invoke fitting")
    monkeypatch.setattr(core_tuning.DateTeamGoalModel, "fit", forbidden)
    assert experiment.replay_rates(trial, records) is trial
    assert trial == before
    assert all(row["rates"] != row["forecast"]["expected_goals"] for row in trial["predictions"])


@pytest.mark.parametrize("change", ["rates", "coefficient", "alpha", "half_life", "rho",
                                   "score", "source_date", "forecast", "probabilities",
                                   "missing_match", "duplicate_fit", "failure"])
def test_replay_rejects_corrupted_raw_rates_coefficients_or_source(change):
    trial, records = replay_fixture()
    row = trial["predictions"][0]
    if change == "rates":
        row["rates"][0] *= 1.2
    elif change == "coefficient":
        trial["fits"][0]["coefficients"]["home:E0"] += .1
    elif change == "alpha":
        trial["fits"][0]["regularization"] = .1
    elif change == "half_life":
        trial["fits"][0]["half_life_days"] = 365
    elif change == "rho":
        trial["fits"][0]["rho"] = -.1
    elif change == "score":
        row["actual_score"] = [8, 7]
    elif change == "source_date":
        row["date"] = "2024-08-02"
    elif change == "forecast":
        row["forecast"]["total_goals"]["4+"] += .1
    elif change == "probabilities":
        row["probabilities"] = [.5, .3, .2]
    elif change == "missing_match":
        trial["predictions"].pop()
    elif change == "duplicate_fit":
        trial["fits"].append(deepcopy(trial["fits"][0]))
    elif change == "failure":
        trial["failures"] = [{"stage": "fit", "error": "not converged"}]
    with pytest.raises(ValueError):
        experiment.replay_rates(trial, records)


def test_all_four_domains_match_independent_per_match_log_scores():
    result, aligned = analysis()
    assert experiment.DOMAINS == ("one_x_two", "total_goals_0_1_2_3_4plus",
                                  "away_minus_two_three_way", "exact_score")
    assert result["n"] == 6
    for arm in experiment.ARMS:
        arrays = experiment.prediction_arrays(aligned[arm])
        for domain in experiment.DOMAINS:
            if domain == "exact_score":
                per_row = [sum(rate - goal*math.log(rate) + math.lgamma(goal + 1)
                               for rate, goal in zip(row["rates"], row["actual_score"]))
                           for row in aligned[arm]]
                field = "mean_nll"
            else:
                probabilities, labels = arrays[domain]
                per_row = [-math.log(probabilities[i, label]) for i, label in enumerate(labels)]
                field = "log_loss"
            assert result["scores"][arm][domain]["n"] == 6
            assert result["scores"][arm][domain][field] == pytest.approx(np.mean(per_row))
    for domain in experiment.DOMAINS:
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        assert result["selected_minus_original"][domain] == pytest.approx(
            result["scores"]["selected"][domain][field] - result["scores"]["original"][domain][field])


def test_zero_observed_outcome_probability_fails_without_tiny_clipping():
    rows = later_rows()
    observed = rows[0]["actual_outcome"]
    rows[0]["probabilities"] = [.5, .5, 0.]
    rows[0]["probabilities"][observed] = 0.
    rows[0]["probabilities"][(observed + 1) % 3] = .5
    rows[0]["probabilities"][(observed + 2) % 3] = .5
    with pytest.raises(ValueError, match="observed|positive|probability"):
        experiment.loss_vectors(rows)


def test_exact_score_nll_uses_untruncated_rates_not_expected_goals_field():
    data = tracks()
    row = prediction("high-rate", "2025-11-17", (30, 5), rates=(20., 12.))
    row["rates"] = [20., 12.]
    for arm in data:
        data[arm].append(deepcopy(row))
    result, aligned = experiment.analyze(data, repeats=7)
    values = experiment.loss_vectors(aligned["original"])["exact_score"]
    expected = 20 - 30*math.log(20) + math.lgamma(31) + 12 - 5*math.log(12) + math.lgamma(6)
    assert values[-1] == pytest.approx(expected)
    assert result["scores"]["original"]["exact_score"]["mean_nll"] == pytest.approx(values.mean())


def test_alignment_is_per_match_not_array_position_and_does_not_mutate_inputs():
    data = tracks()
    result, aligned = experiment.analyze(data, repeats=7)
    data["selected"].reverse()
    before = deepcopy(data)
    reversed_result, reversed_aligned = experiment.analyze(data, repeats=7)
    assert reversed_result == result
    assert reversed_aligned == aligned
    assert data == before


@pytest.mark.parametrize("change", ["missing_arm", "missing_match", "extra_match", "duplicate",
                                   "source_url", "score", "date", "market", "rates", "forecast"])
def test_analysis_does_not_hide_incomplete_conflicted_or_incoherent_forecasts(change):
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
        data["selected"][0]["actual_score"] = [8, 0]
        data["selected"][0]["actual_outcome"] = 0
    elif change == "date":
        data["selected"][0]["date"] = "2025-08-05"
    elif change == "market":
        data["selected"][0]["market_prices"] = {}
        data["selected"][0]["market_probabilities"] = {}
    elif change == "rates":
        data["selected"][0]["rates"][0] += .2
    elif change == "forecast":
        data["selected"][0]["forecast"]["over_2_5"] += .2
    with pytest.raises(ValueError):
        experiment.analyze(data, repeats=7)


def test_market_scores_use_only_identical_available_quotes_not_all_matches():
    result, aligned = analysis()
    assert result["markets"]["avg_preclosing"]["n"] == 6
    closing = result["markets"]["avg_closing"]
    assert closing["n"] == 5 and closing["missing_ids"] == ["new-0"]
    assert result["markets"]["b365_preclosing"] == {"n": 0, "missing_ids": result["coverage"]["common_match_ids"]}
    indices = [i for i, row in enumerate(aligned["original"]) if "avg_closing" in row["market_probabilities"]]
    labels = [aligned["original"][i]["actual_outcome"] for i in indices]
    for arm in experiment.ARMS:
        expected = probability_scores([aligned[arm][i]["probabilities"] for i in indices], labels)
        assert closing["model_scores_on_same_matches"][arm] == expected
    data = tracks()
    for rows in data.values():
        for row in rows:
            row["market_probabilities"], row["market_prices"] = {}, {}
    no_quotes, _ = experiment.analyze(data, repeats=19)
    assert no_quotes["scores"] == result["scores"]
    assert no_quotes["selected_minus_original"] == result["selected_minus_original"]


def test_paired_blocks_share_all_four_domains_and_preserve_empty_calendar_weeks(monkeypatch):
    calls = []
    def observed(dates, values, **kwargs):
        calls.append((list(dates), np.asarray(values).copy(), dict(kwargs)))
        return moving_block_means(dates, values, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result, aligned = analysis()
    assert len(calls) == 5  # Three complete-cohort blocks plus two available markets.
    assert [c[2]["block_weeks"] for c in calls[:3]] == [2, 4, 8]
    expected = np.column_stack([experiment.loss_vectors(aligned["selected"])[domain]
                               - experiment.loss_vectors(aligned["original"])[domain]
                               for domain in experiment.DOMAINS])
    for length, (dates, values, kwargs) in zip((2, 4, 8), calls):
        assert values.shape == (6, 4)
        np.testing.assert_allclose(values, expected, rtol=0, atol=0)
        assert kwargs["repeats"] == 19 and kwargs["seed"] == experiment.SEED
        oracle = moving_block_means(dates, values, **kwargs)
        saved = result["bootstrap"][str(length)]
        assert saved["n_calendar_weeks"] > saved["n_nonempty_weeks"]
        assert 0 in saved["weekly_match_counts"]
        for i, domain in enumerate(experiment.DOMAINS):
            assert saved["differences"][domain]["mean"] == pytest.approx(expected[:, i].mean())
            assert saved["differences"][domain]["interval_95"] == oracle["interval_95"][i]


def test_identical_policy_has_zero_differences_and_grouped_scores_recompose():
    data = tracks()
    data["selected"] = deepcopy(data["original"])
    result, _ = experiment.analyze(data, repeats=13)
    assert set(result["selected_minus_original"].values()) == {0.}
    for boot in result["bootstrap"].values():
        assert all(item == {"mean": 0., "interval_95": [0., 0.]} for item in boot["differences"].values())
    for groups in (result["by_league"], result["by_month"]):
        assert sum(g["n"] for g in groups.values()) == 6
        for arm in experiment.ARMS:
            for domain in experiment.DOMAINS:
                field = "mean_nll" if domain == "exact_score" else "log_loss"
                recomposed = sum(g["n"] * g["scores"][arm][domain][field] for g in groups.values()) / 6
                assert recomposed == pytest.approx(result["scores"][arm][domain][field])


def test_single_week_has_explicit_unavailable_intervals():
    data = {arm: rows[:1] for arm, rows in tracks().items()}
    result, _ = experiment.analyze(data, repeats=7)
    for boot in result["bootstrap"].values():
        assert boot["unavailable_reason"] == "block_length_not_smaller_than_calendar_span"
        assert all(item["interval_95"] is None for item in boot["differences"].values())


def test_csv_preserves_all_raw_rates_probabilities_and_four_per_match_losses(tmp_path):
    result, aligned = analysis()
    path = tmp_path / "paired.csv"
    experiment.write_csv(path, aligned)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["match_id"] for r in rows] == result["coverage"]["common_match_ids"]
    for arm in experiment.ARMS:
        values = experiment.loss_vectors(aligned[arm])
        for i, row in enumerate(rows):
            assert float(row[f"{arm}_home_rate"]) == aligned[arm][i]["rates"][0]
            assert float(row[f"{arm}_away_rate"]) == aligned[arm][i]["rates"][1]
            for j, label in enumerate(("home", "draw", "away")):
                assert float(row[f"{arm}_p_{label}"]) == aligned[arm][i]["probabilities"][j]
            for domain in experiment.DOMAINS:
                assert float(row[f"{arm}_{domain}_nll"]) == values[domain][i]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def install_runner(tmp_path, monkeypatch, *, selected_is_original):
    parent, sources, paths, events = install_archive_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "season_directories", archive_fixture_target.season_directories)
    monkeypatch.setattr(experiment, "load_seasons", archive_fixture_target.load_seasons)
    original_config = next(c for c in core_tuning.candidate_configs() if c["id"] == experiment.ORIGINAL_ID)
    actual_load, actual_original = experiment.load_source, experiment.original_trial
    def load(root, period):
        if period == "2025-26":
            output = tmp_path / "output"
            assert (output / experiment.POLICY_FILE).exists()
            assert (output / "较早候选完整成绩.json").exists()
            events.append(("later_policy_already_written",))
        return actual_load(root, period)
    def original(records, original_parent, period, comp, config):
        events.append(("reuse_original", period, comp))
        return actual_original(records, original_parent, period, comp, config)
    def simulated_candidate(records, config, start, end):
        period = "2024-25" if start.year == 2024 else "2025-26"
        comp = records[0].competition
        trial = deepcopy(actual_original(records, parent, period, comp, original_config))
        trial["config"] = dict(config)
        trial.pop("reused_from")
        events.append(("simulated_fit", period, comp, config["id"]))
        for fit in trial["fits"]:
            fit["regularization"], fit["half_life_days"] = config["regularization"], config["half_life_days"]
            if selected_is_original:
                fit["intercept"] = math.log(3.)
                fit["coefficients"] = {k: 0. for k in fit["coefficients"]}
                fit["coefficients"][f"home:{comp}"] = math.log(.2/3.)
        if selected_is_original:
            for row in trial["predictions"]:
                row["rates"] = [.2, 3.]
                row["forecast"] = summarize(score_grid(*row["rates"], rho=0))
                row["probabilities"] = list(row["forecast"]["one_x_two"].values())
        return trial
    monkeypatch.setattr(experiment, "load_source", load)
    monkeypatch.setattr(experiment, "original_trial", original)
    monkeypatch.setattr(experiment, "run_candidate", simulated_candidate)
    return parent, paths, events


@pytest.mark.parametrize("selected_is_original", [True, False])
def test_full_orchestration_writes_policy_before_later_read_and_counts_real_vs_reused(tmp_path, monkeypatch, selected_is_original):
    parent, paths, events = install_runner(tmp_path, monkeypatch, selected_is_original=selected_is_original)
    untouched = {path: path.read_bytes() for path in paths.values()}
    output = tmp_path / "output"
    result = experiment.run(tmp_path, parent, output, repeats=7)
    later_read = next(i for i, event in enumerate(events) if event[0] == "later_policy_already_written")
    assert len([event for event in events[:later_read] if event[0] == "simulated_fit"]) == 40
    assert len([event for event in events[:later_read] if event[0] == "reuse_original"]) == 5
    assert all(event[1] == "2024-25" for event in events[:later_read] if event[0] in ("simulated_fit", "reuse_original"))
    assert result["counts"]["reused_parameter_snapshots"] == 10
    assert result["counts"]["new_fits"] == (40 if selected_is_original else 45)
    assert result["counts"]["later_policies_fitted"] == int(not selected_is_original)
    assert result["selection"]["expected_matches"] == 5
    assert result["earlier"]["n"] == result["later"]["n"] == 5
    assert (result["selection"]["config"]["id"] == experiment.ORIGINAL_ID) is selected_is_original
    assert len(result["selection"]["candidate_scores"]) == 9
    assert result["settings"]["production_api_changed"] is False
    assert all(path.read_bytes() == content for path, content in untouched.items())
    assert json.loads((output / experiment.RESULT_FILE).read_text(encoding="utf-8")) == result
    assert json.loads((output / experiment.POLICY_FILE).read_text(encoding="utf-8")) == result["selection"]
    assert (output / "较早同场概率.csv").exists() and (output / "较晚同场概率.csv").exists()
    report = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    for phrase in ("不是全新盲测", "净胜至少3球", "不是亚洲盘走水", "不自动更换正式参数"):
        assert phrase in report
