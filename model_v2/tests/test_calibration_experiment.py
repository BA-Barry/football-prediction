"""Independent research-runner tests, not a substitute for the real historical run."""

from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import football_model.calibration_experiment as experiment
from football_model.evaluation import probability_scores
from football_model.football_data import LEAGUES, MARKET_COLUMNS
from football_model.forecast_diagnostics import BLOCKS, CLASS_NAMES, moving_block_means
from football_model.native_experiment import ORIGINAL, TEST_START, VALID_START, VALID_END
from football_model.prematch_calibration import apply_calibrator


def prediction(key="m1", day="2025-08-15", outcome=0, probabilities=(0.4, 0.3, 0.3), competition="E0"):
    cutoff = "2024-08-01" if day < "2025-07-01" else "2025-08-01"
    last = "2024-07-31" if cutoff == "2024-08-01" else "2025-07-31"
    score = [(2, 0), (1, 1), (0, 2)][outcome]
    return {"match_id": key, "date": day, "competition": competition,
        "home": "Home " + key, "away": "Away " + key,
        "fit_before_date": cutoff, "training_max_date": last,
        "actual_score": list(score), "actual_outcome": outcome,
        "probabilities": list(probabilities), "source_url": "https://example.invalid/synthetic.csv",
        "market_prices": {name: [2.0, 4.0, 4.0] for name in MARKET_COLUMNS},
        "market_probabilities": {name: [0.5, 0.25, 0.25] for name in MARKET_COLUMNS},
        "forecast": {"one_x_two": {"home": probabilities[0], "draw": probabilities[1], "away": probabilities[2]},
            "over2_5": 0.55, "total_goals": {"0": 0.1, "1": 0.2}, "expected_goals": {"home": 1.4, "away": 1.1}}}


def earlier_rows():
    days = ["2024-08-15", "2024-09-09", "2024-11-04", "2025-01-06", "2025-03-03", "2025-05-05"]
    probabilities = [(0.4, 0.3, 0.3), (0.4, 0.3, 0.3), (0.4, 0.3, 0.3),
                     (0.45, 0.35, 0.2), (0.2, 0.5, 0.3), (0.2, 0.3, 0.5)]
    outcomes = [0, 1, 2, 0, 2, 1]
    return [prediction("early-" + str(i), day, outcome, p, list(LEAGUES)[i % len(LEAGUES)])
            for i, (day, outcome, p) in enumerate(zip(days, outcomes, probabilities))]


def later_rows():
    days = ["2025-08-04", "2025-08-11", "2025-08-11", "2025-09-08", "2025-10-06", "2025-11-03"]
    probabilities = [(0.5, 0.3, 0.2), (0.25, 0.35, 0.4), (0.3, 0.5, 0.2),
                     (0.6, 0.25, 0.15), (0.2, 0.3, 0.5), (0.3, 0.4, 0.3)]
    return [prediction("late-" + str(i), day, i % 3, p, "D1" if i == 4 else "E0")
            for i, (day, p) in enumerate(zip(days, probabilities))]


def declared_parameters():
    return {method: {"method": method,
        "parameters": {"bias": [0.2, -0.1, 0.0]} if method == "class_bias" else {"beta": 0.8, "temperature": 1.25},
        "before_date": TEST_START.isoformat(), "training_min_date": "2024-08-15",
        "training_max_date": "2025-05-05", "n_matches": 6,
        "training_match_ids": [r["match_id"] for r in earlier_rows()], "class_counts": [2, 2, 2],
        "raw_training_log_loss": 1.1, "calibrated_training_log_loss": 1.05,
        "optimization": {"success": True}}
        for method in experiment.METHODS}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_ordered_rows_sorts_date_then_id_and_keeps_sources_unchanged():
    rows = [prediction("z", "2025-09-08"), prediction("b"), prediction("a")]
    before = deepcopy(rows)
    assert [r["match_id"] for r in experiment.ordered_rows(rows)] == ["a", "b", "z"]
    assert rows == before


@pytest.mark.parametrize("rows", [[], [prediction(), prediction()]])
def test_ordered_rows_refuses_empty_or_duplicate_cohort(rows):
    with pytest.raises(ValueError):
        experiment.ordered_rows(rows)


@pytest.mark.parametrize("field,value", [
    ("date", "not-a-date"), ("fit_before_date", "2025-08-16"),
    ("training_max_date", "2025-08-01"), ("training_max_date", "2025-08-02"),
    ("actual_outcome", 1), ("probabilities", [0.4, 0.3, 0.2]),
])
def test_ordered_rows_preserves_strict_earlier_training_and_saved_identity_checks(field, value):
    row = prediction()
    row[field] = value
    with pytest.raises(ValueError):
        experiment.ordered_rows([row])


@pytest.mark.parametrize("field", ["date", "fit_before_date", "training_max_date"])
def test_missing_dates_are_errors_not_fabricated_or_skipped(field):
    row = prediction()
    del row[field]
    with pytest.raises((KeyError, ValueError)):
        experiment.ordered_rows([row])


def test_calibrator_training_uses_sorted_declared_season_and_fixed_before_date(monkeypatch):
    rows = list(reversed(earlier_rows()))
    calls = []

    def observed_fit(supplied, cutoff, method):
        calls.append((deepcopy(supplied), cutoff, method))
        return declared_parameters()[method]

    monkeypatch.setattr(experiment, "fit_calibrator", observed_fit)
    result = experiment.fit_research_calibrators(rows)
    assert set(result) == set(experiment.METHODS)
    assert [method for _, _, method in calls] == list(experiment.METHODS)
    assert all(cutoff == TEST_START for _, cutoff, _ in calls)
    assert all([r["match_id"] for r in supplied] == [r["match_id"] for r in earlier_rows()]
               for supplied, _, _ in calls)


@pytest.mark.parametrize("day", ["2024-07-31", "2025-07-01", "2025-08-01", "2026-05-01"])
def test_calibrator_training_rejects_other_seasons_before_calling_optimizer(monkeypatch, day):
    row = prediction("outside", day)
    row["fit_before_date"] = "2024-07-01"
    row["training_max_date"] = "2024-06-30"
    called = []
    monkeypatch.setattr(experiment, "fit_calibrator", lambda *a: called.append(a))
    with pytest.raises(ValueError, match="declared earlier forecast season"):
        experiment.fit_research_calibrators([row])
    assert called == []


def test_calibrator_training_failure_propagates_without_identity_fallback(monkeypatch):
    def failed(*args):
        raise RuntimeError("deliberate failed fit")

    monkeypatch.setattr(experiment, "fit_calibrator", failed)
    with pytest.raises(RuntimeError, match="deliberate failed fit"):
        experiment.fit_research_calibrators(earlier_rows())


def test_make_arms_creates_three_vectors_and_never_mutates_forecast_or_parameters():
    rows, parameters = later_rows(), declared_parameters()
    before_rows, before_parameters = deepcopy(rows), deepcopy(parameters)
    arms = experiment.make_arms(rows, parameters)
    assert list(arms) == ["original", *experiment.METHODS]
    np.testing.assert_array_equal(arms["original"], [r["probabilities"] for r in rows])
    for method in experiment.METHODS:
        np.testing.assert_allclose(arms[method], apply_calibrator(arms["original"], parameters[method]))
    assert rows == before_rows
    assert parameters == before_parameters
    arms["original"][0, 0] = 0
    assert rows == before_rows
    assert all(r["forecast"] == original["forecast"] for r, original in zip(rows, before_rows))


@pytest.mark.parametrize("methods", [{}, {"class_bias"}, {"temperature"}, {"class_bias", "temperature", "extra"}])
def test_make_arms_requires_exactly_two_predeclared_calibrators(methods):
    parameters = declared_parameters()
    parameters = {method: parameters.get(method, {}) for method in methods}
    with pytest.raises(ValueError, match="both predeclared"):
        experiment.make_arms(later_rows(), parameters)


@pytest.mark.parametrize("method", experiment.METHODS)
def test_make_arms_refuses_mislabeled_parameters(method):
    parameters = declared_parameters()
    parameters[method]["method"] = "temperature" if method == "class_bias" else "class_bias"
    with pytest.raises(ValueError, match="label mismatch"):
        experiment.make_arms(later_rows(), parameters)


@pytest.mark.parametrize("day", ["2025-08-01", "2025-08-02"])
def test_make_arms_requires_calibrator_training_strictly_before_cutoff(day):
    parameters = declared_parameters()
    parameters["class_bias"]["training_max_date"] = day
    with pytest.raises(ValueError, match="training must precede"):
        experiment.make_arms(later_rows(), parameters)


def test_make_arms_allows_cutoff_day_but_never_earlier_applications():
    rows = [prediction(day="2025-08-01")]
    assert experiment.make_arms(rows, declared_parameters())["original"].shape == (1, 3)
    rows[0]["date"] = "2025-07-31"
    with pytest.raises(ValueError, match="later-fitted calibrator"):
        experiment.make_arms(rows, declared_parameters())


def test_make_arms_calibration_failure_is_not_replaced_with_raw_probabilities(monkeypatch):
    def failed(*args):
        raise ValueError("deliberate underflow")

    monkeypatch.setattr(experiment, "apply_calibrator", failed)
    with pytest.raises(ValueError, match="deliberate underflow"):
        experiment.make_arms(later_rows(), declared_parameters())


def test_research_vectors_refuse_score_grid_interpretation_and_do_not_copy_forecast():
    rows = later_rows()
    before = deepcopy(rows)
    arms = experiment.make_arms(rows, declared_parameters())
    result = experiment.research_predictions(rows, arms)
    assert len(result) == len(rows)
    for i, saved in enumerate(result):
        assert saved["not_a_calibrated_score_grid"] is True
        assert saved["base_fit_before_date"] == rows[i]["fit_before_date"]
        assert set(saved["research_one_x_two"]) == set(experiment.ARM_NAMES)
        assert not {"forecast", "probabilities", "total_goals", "over2_5", "score_grid", "one_x_two"} & set(saved)
        for name, p in arms.items():
            np.testing.assert_array_equal(saved["research_one_x_two"][name], p[i])
    result[0]["research_one_x_two"]["original"][0] = 0
    assert rows == before


@pytest.mark.parametrize("mutation", ["missing-arm", "extra-arm", "empty-rows", "wrong-length", "bad-sum", "nonfinite"])
def test_analyze_arms_rejects_incomplete_or_invalid_probabilities(mutation):
    rows = later_rows()
    arms = experiment.make_arms(rows, declared_parameters())
    if mutation == "missing-arm":
        del arms["temperature"]
    elif mutation == "extra-arm":
        arms["extra"] = arms["original"].copy()
    elif mutation == "empty-rows":
        rows = []
    elif mutation == "wrong-length":
        arms["temperature"] = arms["temperature"][:-1]
    elif mutation == "bad-sum":
        arms["temperature"][0] = [0.4, 0.3, 0.2]
    else:
        arms["temperature"][0, 0] = np.nan
    with pytest.raises(ValueError):
        experiment.analyze_arms(rows, arms, repeats=8)


def test_missing_markets_do_not_change_three_arm_cohort_or_primary_scores():
    rows = later_rows()
    arms = experiment.make_arms(rows, declared_parameters())
    expected = experiment.analyze_arms(rows, arms, repeats=16)
    for i, row in enumerate(rows):
        for market in MARKET_COLUMNS:
            if market == "avg_closing" or i < 2:
                row["market_probabilities"].pop(market)
                row["market_prices"].pop(market)
    result = experiment.analyze_arms(rows, arms, repeats=16)
    for field in ("n", "match_ids", "scores", "calibration", "contrasts", "bootstrap", "by_league", "by_month"):
        assert result[field] == expected[field]
    assert result["markets"]["avg_closing"] == {"n": 0, "missing_ids": result["match_ids"]}
    market = result["markets"]["avg_preclosing"]
    assert market["n"] == len(rows) - 2
    assert market["match_ids"] == [r["match_id"] for r in rows[2:]]
    assert market["missing_ids"] == [r["match_id"] for r in rows[:2]]
    assert all(s["n"] == len(rows) - 2 for s in market["arm_scores_on_identical_matches"].values())
    y = np.asarray([r["actual_outcome"] for r in rows[2:]])
    for name, p in arms.items():
        assert market["arm_scores_on_identical_matches"][name] == probability_scores(p[2:], y)


def test_identical_arms_have_exact_zero_loss_differences_and_intervals():
    rows = later_rows()
    p = np.asarray([r["probabilities"] for r in rows])
    arms = {name: p.copy() for name in experiment.ARM_NAMES}
    result = experiment.analyze_arms(rows, arms, repeats=16)
    for name, contrast in result["contrasts"].items():
        assert contrast["mean"] == 0
        for length in BLOCKS:
            feature = result["bootstrap"][str(length)]["features"][name]
            assert feature == {"mean": 0.0, "interval_95": [0.0, 0.0]}


def test_same_match_scores_and_group_and_outcome_contributions_reconstruct_totals():
    rows = later_rows()
    arms = experiment.make_arms(rows, declared_parameters())
    before = deepcopy(rows)
    result = experiment.analyze_arms(rows, arms, repeats=32)
    assert rows == before
    assert result["match_ids"] == [r["match_id"] for r in rows]
    y = np.asarray([r["actual_outcome"] for r in rows])
    for name, probabilities in arms.items():
        scores = result["scores"][name]
        assert scores == probability_scores(probabilities, y)
        assert sum(c["binary_brier"] for c in result["calibration"][name]["classes"].values()) == pytest.approx(scores["brier_sum"])
        for grouping in ("by_league", "by_month"):
            assert sum(g["n"] for g in result[grouping].values()) == len(rows)
            mean = sum(g["scores"][name]["log_loss"] * g["n"] for g in result[grouping].values()) / len(rows)
            assert mean == pytest.approx(scores["log_loss"])
    for name, contrast in result["contrasts"].items():
        method = name.removesuffix("_minus_original")
        expected = sum(math.log(arms["original"][i, row["actual_outcome"]] /
                                arms[method][i, row["actual_outcome"]]) for i, row in enumerate(rows)) / len(rows)
        assert contrast["mean"] == pytest.approx(expected)
        assert sum(c["n"] for c in contrast["by_actual_outcome"].values()) == len(rows)
        assert sum(c["weighted_contribution"] for c in contrast["by_actual_outcome"].values()) == pytest.approx(expected)
        for grouping in ("by_league", "by_month"):
            mean = sum(g["mean_differences"][name] * g["n"] for g in result[grouping].values()) / len(rows)
            assert mean == pytest.approx(expected)
    # Deliberately unequal league/month sizes: unweighted group means are not the target.
    grouped = result["by_league"]
    assert len({g["n"] for g in grouped.values()}) > 1
    assert np.mean([g["mean_differences"]["class_bias_minus_original"] for g in grouped.values()]) != pytest.approx(
        result["contrasts"]["class_bias_minus_original"]["mean"])


def test_analyze_arms_passes_all_arms_classes_and_leagues_jointly_to_2_4_8_week_bootstrap(monkeypatch):
    rows = later_rows()
    for row in rows:
        row["market_probabilities"] = {}
        row["market_prices"] = {}
    arms = experiment.make_arms(rows, declared_parameters())
    calls = []

    def observed(dates, values, **kwargs):
        calls.append((list(dates), values.copy(), dict(kwargs)))
        return moving_block_means(dates, values, **kwargs)

    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result = experiment.analyze_arms(rows, arms, repeats=23, seed=17)
    assert [call[2]["block_weeks"] for call in calls] == [2, 4, 8]
    expected_names = [f"{method}_minus_original" for method in experiment.METHODS]
    expected_names += [f"{arm}_{outcome}_bias" for arm in arms for outcome in CLASS_NAMES]
    for dates, values, kwargs in calls:
        assert dates == [r["date"] for r in rows]
        assert values.shape == (len(rows), 11)
        assert kwargs["repeats"] == 23 and kwargs["seed"] == 17
        boot = result["bootstrap"][str(kwargs["block_weeks"])]
        assert list(boot["features"]) == expected_names
        oracle = moving_block_means(dates, values, **kwargs)
        for i, feature in enumerate(expected_names):
            assert boot["features"][feature]["mean"] == pytest.approx(oracle["mean"][i])
            assert boot["features"][feature]["interval_95"] == oracle["interval_95"][i]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    np.testing.assert_array_equal(calls[1][1], calls[2][1])
    assert result["bootstrap"]["4"]["n_calendar_weeks"] == 14
    assert 0 in result["bootstrap"]["4"]["weekly_match_counts"]


def test_raw_summary_is_unmodified_probability_diagnostic_not_calibrated_training_score():
    rows = earlier_rows()
    before = deepcopy(rows)
    result = experiment.raw_summary(rows)
    assert rows == before
    y = np.asarray([r["actual_outcome"] for r in rows])
    p = np.asarray([r["probabilities"] for r in rows])
    assert result["scores"] == probability_scores(p, y)
    assert result["n"] == sum(v["n"] for v in result["by_league"].values())
    assert result["first_date"] == min(r["date"] for r in rows)
    assert result["last_date"] == max(r["date"] for r in rows)


def test_histogram_uses_earlier_probabilities_without_relabeling_class_counts_independent():
    text = experiment.probability_histogram(earlier_rows())
    assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert text.endswith("</svg>")
    assert text.count("<rect") == 31
    assert "not independent class samples" in text
    assert "common y maximum" in text
    assert "equal-width 0.1 bins" in text
    assert "n=6" in text


def test_csv_reproduces_standalone_vectors_loss_and_base_cutoff(tmp_path):
    rows = later_rows()
    arms = experiment.make_arms(rows, declared_parameters())
    path = tmp_path / "research.csv"
    experiment.write_csv(path, rows, arms)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert [r["match_id"] for r in saved] == [r["match_id"] for r in rows]
    for i, record in enumerate(saved):
        assert record["base_fit_before_date"] == rows[i]["fit_before_date"]
        assert int(record["actual_outcome"]) == rows[i]["actual_outcome"]
        assert not {"forecast", "over2_5", "total_goals"} & set(record)
        for method, p in arms.items():
            np.testing.assert_allclose([float(record[f"{method}_p_{c}"]) for c in CLASS_NAMES], p[i])
            assert float(record[f"{method}_log_loss"]) == pytest.approx(-math.log(p[i, rows[i]["actual_outcome"]]))


def test_fit_earlier_original_reads_only_development_directories_and_same_fixed_config(tmp_path, monkeypatch):
    directories = {s: tmp_path / s.replace("/", "-") for s in ("2022/23", "2023/24", "2024/25", "2025/26")}
    records = [SimpleNamespace(competition=comp) for comp in LEAGUES]
    expected_calls = []
    monkeypatch.setattr(experiment, "season_directories", lambda root: directories)

    def observed_load(supplied):
        assert set(supplied) == {"2022/23", "2023/24", "2024/25"}
        assert supplied == {s: p for s, p in directories.items() if s != "2025/26"}
        return records, {"synthetic": True}

    def trial(league, config, start, end):
        assert len(league) == 1
        assert config == ORIGINAL and start == VALID_START and end == VALID_END
        comp = league[0].competition
        expected_calls.append(comp)
        return {"predictions": [prediction("early-" + comp, "2024-09-01", competition=comp)],
                "fits": [{"synthetic": comp}], "failures": [], "skipped": [{"reason": "unseen_team_no_estimated_prior"}]}

    monkeypatch.setattr(experiment, "load_seasons", observed_load)
    monkeypatch.setattr(experiment, "run_config", trial)
    monkeypatch.setattr(experiment, "expected_target_ids", lambda league, *a: {"early-" + league[0].competition})
    output = tmp_path / "output"
    (output / "较早旧模型预测").mkdir(parents=True)
    rows, coverage = experiment.fit_earlier_original(tmp_path, output)
    assert expected_calls == sorted(LEAGUES)
    assert len(rows) == len(LEAGUES)
    assert coverage["new_base_model_fits"] == len(LEAGUES)
    assert coverage["sources"] == {"synthetic": True}
    assert all(m["predicted"] == 1 and m["fits"] == 1 for m in coverage["leagues"].values())
    assert len(list((output / "较早旧模型预测").glob("*.json"))) == len(LEAGUES)


@pytest.mark.parametrize("failure", ["missing", "extra", "duplicate", "fit-failure"])
def test_fit_earlier_original_refuses_incomplete_runs_instead_of_shrinking_training_cohort(tmp_path, monkeypatch, failure):
    records = [SimpleNamespace(competition=comp) for comp in LEAGUES]
    monkeypatch.setattr(experiment, "season_directories", lambda root: {"2024/25": tmp_path})
    monkeypatch.setattr(experiment, "load_seasons", lambda supplied: (records, {}))
    monkeypatch.setattr(experiment, "expected_target_ids", lambda *a: {"expected"})
    rows = [prediction("expected", "2024-09-01")]
    if failure == "missing":
        rows = []
    elif failure == "extra":
        rows.append(prediction("extra", "2024-09-01"))
    elif failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    trial = {"predictions": rows, "fits": [], "skipped": [],
             "failures": [{"stage": "fit"}] if failure == "fit-failure" else []}
    monkeypatch.setattr(experiment, "run_config", lambda *a: deepcopy(trial))
    output = tmp_path / "output"
    (output / "较早旧模型预测").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete earlier original forecasts"):
        experiment.fit_earlier_original(tmp_path, output)
    assert len(list((output / "较早旧模型预测").glob("*.json"))) == 1


def install_small_runner(monkeypatch, early, late, events):
    original_fit = experiment.fit_calibrator

    def earlier(root, output):
        events.append("earlier-base-predictions")
        assert (output / "实验设置.json").exists()
        return deepcopy(early), {"new_base_model_fits": 5, "sources": {}, "leagues": {}}

    def observed_fit(rows, cutoff, method):
        events.append("fit-" + method)
        assert all(date.fromisoformat(r["date"]) < cutoff for r in rows)
        assert {r["match_id"] for r in rows} == {r["match_id"] for r in early}
        return original_fit(rows, cutoff, method)

    def later(parent):
        events.append("load-later")
        assert events.index("fit-class_bias") < events.index("load-later")
        assert events.index("fit-temperature") < events.index("load-later")
        rows = {r["match_id"]: deepcopy(r) for r in late}
        return {"original": rows}, {"common_match_ids": list(rows)}

    monkeypatch.setattr(experiment, "fit_earlier_original", earlier)
    monkeypatch.setattr(experiment, "fit_calibrator", observed_fit)
    monkeypatch.setattr(experiment, "load_saved_tracks", later)


def test_small_runner_fits_before_loading_later_labels_exports_only_isolated_research(tmp_path, monkeypatch):
    early, late, events = earlier_rows(), later_rows(), []
    early_before, late_before = deepcopy(early), deepcopy(late)
    install_small_runner(monkeypatch, early, late, events)
    parent = tmp_path / "prior-source"
    parent.mkdir()
    source = parent / "source.json"
    source.write_text('{"preserve": true}', encoding="utf-8")
    before = source.read_bytes()
    root = tmp_path / "research-project"
    output = experiment.run(root, parent, repeats=12)
    assert output.is_relative_to(root / "outputs")
    assert events == ["earlier-base-predictions", "fit-class_bias", "fit-temperature", "load-later"]
    assert source.read_bytes() == before
    assert early == early_before and late == late_before
    result = read_json(output / experiment.RESULT_FILE)
    settings = result["settings"]
    assert settings["base_config"] == ORIGINAL
    assert settings["methods"] == list(experiment.METHODS)
    assert settings["primary_contrast"] == "class_bias_minus_original"
    for key in ("calibrator_updated_in_later_season", "winner_selected_on_later_results", "formal_probability_api_changed",
                "score_grid_reweighted", "market_fusion", "fit_per_league"):
        assert settings[key] is False
    assert settings["previously_viewed_later_season"] is True
    assert settings["match_equal_weight"] is True
    assert result["counts"] == {"new_base_model_fits": 5, "new_calibrator_fits": 2,
        "new_earlier_base_predictions": 6, "reused_later_base_predictions": 6,
        "new_later_base_model_fits": 0, "new_research_calibrated_vectors": 12}
    for method in experiment.METHODS:
        saved = read_json(output / f"校准参数_{method}.json")
        assert saved == result["parameters"][method]
        assert saved["training_match_ids"] == [r["match_id"] for r in early]
        assert saved["training_max_date"] < saved["before_date"] == TEST_START.isoformat()
    vectors = read_json(output / "较晚研究概率.json")
    assert all(r["not_a_calibrated_score_grid"] for r in vectors)
    assert all("forecast" not in r for r in vectors)
    assert (output / "较早原始概率直方图.svg").exists()
    with (output / "较晚研究概率与损失.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    for method, scores in result["later"]["scores"].items():
        assert sum(float(r[f"{method}_log_loss"]) for r in saved) / len(saved) == pytest.approx(scores["log_loss"])
    text = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    assert text == experiment.report(result)
    for phrase in ("不按已查看的2025/26选赢家", "不是当年逐时已选模型", "不是时间外验证成功", "独立研究向量", "不能把校准1X2与旧比分分布混用", "不含校准参数估计"):
        assert phrase in text


def test_changing_later_labels_cannot_change_earlier_fitted_parameters(tmp_path, monkeypatch):
    early, late = earlier_rows(), later_rows()
    original_fit = experiment.fit_calibrator
    events = []
    install_small_runner(monkeypatch, early, late, events)
    first = experiment.run(tmp_path / "first", tmp_path / "parent", repeats=8)
    first_parameters = read_json(first / experiment.RESULT_FILE)["parameters"]
    # Reset the wrapper to avoid nested observers; only future labels/scores change.
    monkeypatch.setattr(experiment, "fit_calibrator", original_fit)
    changed = deepcopy(late)
    for row in changed:
        row["actual_outcome"] = (row["actual_outcome"] + 1) % 3
        row["actual_score"] = [[2, 0], [1, 1], [0, 2]][row["actual_outcome"]]
    install_small_runner(monkeypatch, early, changed, [])
    second = experiment.run(tmp_path / "second", tmp_path / "parent", repeats=8)
    a, b = read_json(first / experiment.RESULT_FILE), read_json(second / experiment.RESULT_FILE)
    assert first_parameters == b["parameters"]
    assert a["earlier_raw"] == b["earlier_raw"]
    assert a["later"]["scores"] != b["later"]["scores"]


def test_runner_does_not_load_later_data_after_calibration_fit_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "fit_earlier_original", lambda *a: (earlier_rows(), {"new_base_model_fits": 5}))
    called = []

    def failed(*args):
        raise RuntimeError("research fit failed")

    monkeypatch.setattr(experiment, "fit_calibrator", failed)
    monkeypatch.setattr(experiment, "load_saved_tracks", lambda *a: called.append("later-read"))
    with pytest.raises(RuntimeError, match="research fit failed"):
        experiment.run(tmp_path / "root", tmp_path / "parent", repeats=8)
    assert called == []
    output = next((tmp_path / "root" / "outputs").iterdir())
    assert not (output / experiment.RESULT_FILE).exists()


def test_runner_rejects_train_eval_identity_overlap_before_research_predictions(tmp_path, monkeypatch):
    early, late = earlier_rows(), later_rows()
    late[0]["match_id"] = early[0]["match_id"]
    install_small_runner(monkeypatch, early, late, [])
    with pytest.raises(ValueError, match="identities overlap"):
        experiment.run(tmp_path / "root", tmp_path / "parent", repeats=8)
    output = next((tmp_path / "root" / "outputs").iterdir())
    assert not (output / "较晚研究概率.json").exists()


def test_runner_default_requires_completed_parent_without_starting_new_fits(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    called = []
    monkeypatch.setattr(experiment, "fit_earlier_original", lambda *a: called.append(True))
    with pytest.raises(ValueError, match="completed original/native"):
        experiment.run(tmp_path)
    assert called == []


def test_report_handles_missing_markets_and_too_short_calendar_without_claiming_blind_test():
    rows = [prediction()]
    for market in MARKET_COLUMNS:
        rows[0]["market_probabilities"].pop(market)
        rows[0]["market_prices"].pop(market)
    result = {"earlier_raw": experiment.raw_summary(earlier_rows()),
        "later": experiment.analyze_arms(rows, experiment.make_arms(rows, declared_parameters()), repeats=8),
        "parameters": declared_parameters(), "counts": {"new_base_model_fits": 5}, "settings": {"repeats": 8}}
    text = experiment.report(result)
    assert "不可估计" in text
    assert "不能把同一后期赛季重新称为新盲测" in text
    assert "训练内" in text and "未来有效" in text
    assert "β=0时为均匀分布极限" in text
    for market in MARKET_COLUMNS:
        assert f"| {market} | 0 | — |" in text
