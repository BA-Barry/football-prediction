"""Fixed-window research uses strictly past fits and match-weighted evaluation."""

from copy import deepcopy
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pytest

from football_model import cmp_forward_experiment as experiment
from football_model.cmp_fit import CMPFitError, fit_cmp_window
from test_goal_rate_experiment import prediction


def archive_rows():
    """Unequal fold sizes deliberately distinguish match from fold weighting."""
    records = []
    warmup = ("2024-08-01", "2024-09-12", "2024-11-21", "2025-01-31")
    days = [(day, "warmup") for day in warmup]
    for index, ((cutoff, _), count) in enumerate(zip(experiment.WINDOWS, (1, 2, 3, 1, 2, 3))):
        start = date.fromisoformat(cutoff)
        days += [((start + timedelta(days=j)).isoformat(), f"fold-{index}") for j in range(count)]
    days += [(experiment.BOUNDARIES[-1], "unscored-future")]
    scores = ((0, 0), (1, 0), (2, 1), (0, 3), (4, 2), (1, 1), (1, 2))
    for i, (day, label) in enumerate(days):
        rates = (.7 + .17*(i % 7), .8 + .13*(i % 6))
        row = prediction(f"{label}-{i:02d}", day, scores[i % len(scores)],
                         ("E0", "D1", "I1", "SP1", "F1")[i % 5], rates=rates)
        row["rates"] = list(rates)
        records.append(row)
    return records


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def install_runner(tmp_path, monkeypatch):
    originals = archive_rows()
    output = tmp_path / "forward-output"
    events, fits = [], []
    real_research = experiment.research_tracks

    def load(root, parent, season):
        assert Path(root) == tmp_path
        assert Path(parent) == tmp_path / "source"
        events.append(("load", season))
        if season == "2024-25":
            subset = [r for r in originals if r["date"] < "2025-08-01"]
            snapshots = 2
        else:
            assert season == "2025-26"
            subset = [r for r in originals if r["date"] >= "2025-08-01"]
            snapshots = 3
        return deepcopy(subset), {"parameter_snapshots": snapshots, "saved_prediction_rows": len(subset)}

    def fit(rows, *, training_start, before_date):
        index = experiment.BOUNDARIES.index(before_date)
        assert training_start == experiment.TRAINING_START
        assert all(training_start <= r["date"] < before_date for r in rows)
        expected, _ = experiment.window_rows(originals, *experiment.WINDOWS[index])
        assert [r["match_id"] for r in rows] == [r["match_id"] for r in expected]
        events.append(("fit", before_date, tuple(r["match_id"] for r in rows)))
        result = {"nu": 1.04 + .065*index, "n_matches": len(rows),
                  "training_start": training_start, "before_date": before_date,
                  "training_match_ids": [r["match_id"] for r in rows],
                  "training_min_date": rows[0]["date"], "training_max_date": rows[-1]["date"],
                  "bounds": [.25, 4.], "at_boundary": False, "success": True}
        fits.append(deepcopy(result))
        return result

    def research(rows, nu):
        cutoff = fits[-1]["before_date"]
        parameter = output / cutoff / "CMP参数.json"
        assert parameter.is_file(), "The fold parameter must be saved before target predictions"
        assert read_json(parameter) == fits[-1]
        assert nu == fits[-1]["nu"]
        assert all(r["date"] >= cutoff for r in rows)
        events.append(("research", cutoff, tuple(r["match_id"] for r in rows)))
        return real_research(rows, nu)

    monkeypatch.setattr(experiment, "load_period", load)
    monkeypatch.setattr(experiment, "fit_cmp_window", fit)
    monkeypatch.setattr(experiment, "research_tracks", research)
    return originals, output, events, fits


def test_fixed_windows_partition_by_date_only_without_mutating_rows():
    records = list(reversed(archive_rows()))
    before = deepcopy(records)
    scored = []
    for cutoff, end in experiment.WINDOWS:
        training, target = experiment.window_rows(records, cutoff, end)
        assert training and target
        assert training == sorted(training, key=lambda r: (r["date"], r["match_id"]))
        assert all(experiment.TRAINING_START <= r["date"] < cutoff for r in training)
        assert all(cutoff <= r["date"] < end for r in target)
        assert not set(r["match_id"] for r in training) & set(r["match_id"] for r in target)
        scored.extend(r["match_id"] for r in target)
    expected = [r["match_id"] for r in records if experiment.BOUNDARIES[0] <= r["date"] < experiment.BOUNDARIES[-1]]
    assert len(scored) == len(set(scored)) == len(expected) == 12
    assert set(scored) == set(expected)
    assert records == before


@pytest.mark.parametrize("fold_index", [0, 2, 4])
def test_future_score_perturbations_do_not_change_actual_fitted_nu_or_history(fold_index):
    records = archive_rows()
    cutoff, end = experiment.WINDOWS[fold_index]
    original_training, original_target = experiment.window_rows(records, cutoff, end)
    original_fit = fit_cmp_window(original_training, training_start=experiment.TRAINING_START, before_date=cutoff)
    perturbed = deepcopy(records)
    for row in perturbed:
        if row["date"] >= cutoff:
            h, a = row["actual_score"]
            row["actual_score"] = [h + 7, a + 1]
            row["actual_outcome"] = 0 if h + 7 > a + 1 else 2 if h + 7 < a + 1 else 1
    new_training, new_target = experiment.window_rows(perturbed, cutoff, end)
    assert original_training == new_training
    assert original_target != new_target
    perturbed_fit = fit_cmp_window(new_training, training_start=experiment.TRAINING_START, before_date=cutoff)
    assert perturbed_fit == original_fit
    assert all(row["date"] < cutoff for row in new_training)
    assert original_fit["training_max_date"] < cutoff


@pytest.mark.parametrize("problem", ["unknown_window", "duplicate", "invalid_future_identity", "no_training", "no_target"])
def test_partition_errors_do_not_silently_skip_or_deduplicate_folds(problem):
    records = archive_rows()
    cutoff, end = experiment.WINDOWS[0]
    if problem == "unknown_window":
        cutoff = "2025-02-02"
    elif problem == "duplicate":
        records.append(deepcopy(records[0]))
    elif problem == "invalid_future_identity":
        records[-1]["source_url"] = ""
    elif problem == "no_training":
        records = [r for r in records if r["date"] >= cutoff]
    elif problem == "no_target":
        records = [r for r in records if not cutoff <= r["date"] < end]
    with pytest.raises(ValueError):
        experiment.window_rows(records, cutoff, end)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    originals, output, events, fits = install_runner(tmp_path, monkeypatch)
    unchanged = deepcopy(originals)
    result = experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert originals == unchanged
    return result, originals, output, events, fits


def test_runner_saves_each_fold_parameter_before_research_and_all_six_folds_are_scored_once(completed):
    result, originals, output, events, fits = completed
    assert events[:2] == [("load", "2024-25"), ("load", "2025-26")]
    assert len(fits) == len(result["folds"]) == len(experiment.WINDOWS) == 6
    assert [f["cutoff"] for f in result["folds"]] == list(experiment.BOUNDARIES[:-1])
    all_target_ids = []
    for index, (cutoff, end) in enumerate(experiment.WINDOWS):
        assert events[2 + 2*index][0:2] == ("fit", cutoff)
        assert events[3 + 2*index][0:2] == ("research", cutoff)
        folder = output / cutoff
        assert read_json(folder / "CMP参数.json") == result["folds"][index]["fit"] == fits[index]
        assert read_json(folder / "同场分析.json") == result["folds"][index]["analysis"]
        tracks = read_json(folder / "完整研究预测.json")
        ids = [r["match_id"] for r in tracks["cmp"]]
        all_target_ids.extend(ids)
        assert ids == [r["match_id"] for r in tracks["original"]]
        assert len(ids) == (1, 2, 3, 1, 2, 3)[index]
        for arm in experiment.ARMS:
            assert all(r["cmp_fit_before_date"] == cutoff and r["cmp_target_end_exclusive"] == end
                       and r["cmp_training_start"] == experiment.TRAINING_START for r in tracks[arm])
    expected = [r["match_id"] for r in originals if experiment.BOUNDARIES[0] <= r["date"] < experiment.BOUNDARIES[-1]]
    assert set(all_target_ids) == set(expected)
    assert len(all_target_ids) == len(set(all_target_ids)) == result["pooled"]["n"] == 12
    assert result["counts"] == {
        "new_dispersion_fits": 6, "new_team_fits": 0, "unique_scored_matches": 12,
        "research_predictions": 12,
        "warmup_only_forecast_ids": [r["match_id"] for r in originals if r["date"] < experiment.BOUNDARIES[0]],
        "reused_team_parameter_snapshots": 5}
    assert read_json(output / experiment.RESULT_FILE) == result
    assert read_json(output / "实验设置.json") == result["settings"]
    assert len(list(output.rglob("*.json"))) == 21
    assert len(list(output.rglob("*"))) == 30  # 6 folders, 21 JSON, CSV, SVG, Markdown.
    assert not (output / "研究失败.json").exists()


def test_saved_forward_probabilities_keep_source_raw_means_and_all_four_losses(completed):
    result, originals, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    source = {r["match_id"]: r for r in originals}
    for old, new in zip(tracks["original"], tracks["cmp"]):
        raw = source[old["match_id"]]
        assert old["rates"] == new["rates"] == raw["rates"]
        assert old["actual_score"] == new["actual_score"] == raw["actual_score"]
        assert old["market_probabilities"] == new["market_probabilities"] == raw["market_probabilities"]
        assert old["market_prices"] == new["market_prices"] == raw["market_prices"]
        for i, side in enumerate(("home", "away")):
            assert new["forecast"]["expected_goals"][side] == pytest.approx(raw["rates"][i], abs=1e-8)
    vectors = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    assert set(vectors["cmp"]) == set(experiment.DOMAINS)
    with (output / "逐场前推概率.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        assert len(reader.fieldnames) == 23
    assert len(saved) == 12
    for i, item in enumerate(saved):
        assert item["match_id"] == tracks["original"][i]["match_id"]
        assert [float(item[k]) for k in ("home_mean", "away_mean")] == tracks["original"][i]["rates"]
        for arm in experiment.ARMS:
            for domain in experiment.DOMAINS:
                assert float(item[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]
    assert result["settings"]["raw_means_changed"] is False
    assert result["settings"]["production_api_changed"] is False


def test_pooled_results_are_match_weighted_not_six_window_equal_weight_or_six_observations(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    n = result["pooled"]["n"]
    assert n == sum(f["analysis"]["n"] for f in result["folds"]) == 12
    assert n != len(result["folds"])
    per_loss = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    differences = {}
    for domain in experiment.DOMAINS:
        differences[domain] = per_loss["cmp"][domain] - per_loss["original"][domain]
        pooled = result["pooled"]["cmp_minus_original"][domain]
        assert pooled == pytest.approx(differences[domain].mean())
        weighted = sum(f["analysis"]["n"] * f["analysis"]["cmp_minus_original"][domain]
                       for f in result["folds"]) / n
        assert pooled == pytest.approx(weighted)
        for grouping in ("by_league", "by_month"):
            groups = result["pooled"][grouping]
            recomposed = sum(g["n"] * g["cmp_minus_original"][domain] for g in groups.values()) / n
            assert pooled == pytest.approx(recomposed)
        seasonal = sum(s["n"] * s["cmp_minus_original"][domain] for s in result["by_season"].values()) / n
        assert pooled == pytest.approx(seasonal)
    assert any(not np.isclose(result["pooled"]["cmp_minus_original"][domain],
                             np.mean([f["analysis"]["cmp_minus_original"][domain] for f in result["folds"]]))
               for domain in experiment.DOMAINS)
    for weeks in ("2", "4", "8"):
        boot = result["pooled"]["bootstrap"][weeks]
        assert boot["n_matches"] == n
        assert set(boot["differences"]) == set(experiment.DOMAINS)
        for domain in experiment.DOMAINS:
            assert boot["differences"][domain]["mean"] == pytest.approx(differences[domain].mean())


def test_all_loss_descriptives_retain_tail_negative_and_positive_matches(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    per_loss = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    for domain in experiment.DOMAINS:
        values = per_loss["cmp"][domain] - per_loss["original"][domain]
        item = result["pooled"]["paired_loss_descriptives"][domain]
        assert item["n"] == len(values) == 12
        assert item["mean"] == pytest.approx(values.mean())
        assert item["sd"] == pytest.approx(values.std(ddof=1))
        assert item["median"] == pytest.approx(np.median(values))
        assert item["q25"] == pytest.approx(np.quantile(values, .25))
        assert item["q75"] == pytest.approx(np.quantile(values, .75))
        assert item["min"] == values.min() and item["max"] == values.max()
        assert item["n_negative"] == np.sum(values < 0)
        assert item["n_positive"] == np.sum(values > 0)
    assert result["nu_path_descriptives"]["n"] == 6
    assert result["nu_path_descriptives"]["mean"] == pytest.approx(np.mean([f["fit"]["nu"] for f in result["folds"]]))


def test_report_explains_conditioning_and_histogram_is_parseable_without_dropping_observations(completed):
    result, _, output, _, _ = completed
    report = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    assert report == experiment.report(result)
    for phrase in ("6次共享ν重估", "12场不重叠评分", "不是总共只估计一个标量", "逐场等权",
                   "不是六次独立实验", "不是新盲测", "不包含每次重新估计ν", "均值/SD/中位数/分位数/极值",
                   "没有裁掉高比分", "净胜至少3球", "不是亚洲盘退款", "没有深度学习", "正式模型未替换"):
        assert phrase in report
    root = ElementTree.parse(output / "逐场损失分布.svg").getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    texts = ["".join(node.itertext()) for node in root.findall(".//{http://www.w3.org/2000/svg}text")]
    assert all(domain in texts for domain in experiment.DOMAINS)
    assert sum(text.startswith("n=12, mean=") for text in texts) == 4
    assert len(root.findall(".//{http://www.w3.org/2000/svg}rect")) == 121
    assert "nan" not in (output / "逐场损失分布.svg").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("fail_index", [0, 2])
def test_fit_failure_retains_completed_folds_and_trace_without_fallback_or_success_report(tmp_path, monkeypatch, fail_index):
    _, output, events, _ = install_runner(tmp_path, monkeypatch)
    actual_fit = experiment.fit_cmp_window
    trace = [{"nu": .25, "nll": None, "numerical_status": "error"}]
    cutoff = experiment.WINDOWS[fail_index][0]
    def fail(rows, *, training_start, before_date):
        if before_date == cutoff:
            events.append(("failed_fit", before_date))
            raise CMPFitError("deliberate forward numerical failure", history=trace)
        return actual_fit(rows, training_start=training_start, before_date=before_date)
    monkeypatch.setattr(experiment, "fit_cmp_window", fail)
    with pytest.raises(CMPFitError, match="deliberate forward"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert len([e for e in events if e[0] == "research"]) == fail_index
    assert not (output / cutoff).exists()
    assert read_json(output / "研究失败.json") == {
        "cutoff": cutoff, "type": "CMPFitError", "message": "deliberate forward numerical failure",
        "history": trace, "failed_no_fallback": True}
    assert (output / "实验设置.json").exists()
    for earlier, _ in experiment.WINDOWS[:fail_index]:
        assert (output / earlier / "CMP参数.json").exists()
        assert (output / earlier / "同场分析.json").exists()
    assert not (output / experiment.RESULT_FILE).exists()
    assert not (output / experiment.REPORT_FILE).exists()
    assert not (output / "完整前推预测.json").exists()


def test_research_failure_keeps_saved_parameter_and_stops_without_poisson_substitution(tmp_path, monkeypatch):
    _, output, events, _ = install_runner(tmp_path, monkeypatch)
    cutoff = experiment.WINDOWS[0][0]
    def fail(rows, nu):
        assert (output / cutoff / "CMP参数.json").exists()
        raise RuntimeError("deliberate target forecast failure")
    monkeypatch.setattr(experiment, "research_tracks", fail)
    with pytest.raises(RuntimeError, match="deliberate target"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert len([e for e in events if e[0] == "fit"]) == 1
    assert (output / cutoff / "CMP参数.json").exists()
    assert not (output / cutoff / "完整研究预测.json").exists()
    assert not (output / experiment.RESULT_FILE).exists()
    failure = read_json(output / "研究失败.json")
    assert failure["cutoff"] == cutoff and failure["failed_no_fallback"]


def test_unsupported_empty_fold_stops_instead_of_dropping_a_window(tmp_path, monkeypatch):
    originals, output, _, _ = install_runner(tmp_path, monkeypatch)
    cutoff, end = experiment.WINDOWS[0]
    originals[:] = [r for r in originals if not cutoff <= r["date"] < end]
    with pytest.raises(ValueError, match="no skipped folds"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert read_json(output / "研究失败.json")["cutoff"] == cutoff
    assert not (output / experiment.RESULT_FILE).exists()


def test_cli_default_paths_are_anchored_at_g_project_not_package(tmp_path, monkeypatch, capsys):
    calls = []
    def run(root, parent, output):
        calls.append((root, parent, output))
        return {"pooled": {"n": 12, "cmp_minus_original": {}}, "folds": [{"fit": {"nu": 1.2}}]}
    monkeypatch.setattr(experiment, "run", run)
    monkeypatch.setattr(sys, "argv", ["cmp_forward_experiment", "--output", str(tmp_path / "chosen")])
    experiment.main()
    root, parent, output = calls[0]
    assert root == Path(experiment.__file__).resolve().parents[3]
    assert root.name == "football-prediction-2026-10-22"
    assert root.drive.upper() == "G:"
    assert parent == root / "outputs" / "更新频率研究_20261023T073047979805Z"
    assert output == tmp_path / "chosen"
    assert json.loads(capsys.readouterr().out)["nu_path"] == [1.2]
