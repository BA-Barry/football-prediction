"""Dependence research keeps original means and scores complete fixed windows."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pytest

from football_model import bivariate_poisson_experiment as experiment
from football_model import cmp_experiment, cmp_forward_experiment
from football_model.bivariate_poisson_fit import BivariatePoissonFitError
from test_cmp_forward_experiment import archive_rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def install_runner(tmp_path, monkeypatch):
    """Fake only archived input and scalar fits; predictions and analysis run."""
    originals = archive_rows()
    output = tmp_path / "dependence-output"
    events, fits = [], []
    real_research = experiment.research_tracks

    def load(root, parent, season):
        assert Path(root) == tmp_path
        assert Path(parent) == tmp_path / "source"
        assert (output / "实验设置.json").is_file()
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
        expected, _ = experiment.window_rows(originals, *experiment.WINDOWS[index])
        assert [r["match_id"] for r in rows] == [r["match_id"] for r in expected]
        assert all(training_start <= r["date"] < before_date for r in rows)
        events.append(("fit", before_date, tuple(r["match_id"] for r in rows)))
        result = {"k": .02 + .06*index, "n_matches": len(rows),
            "training_start": training_start, "before_date": before_date,
            "training_match_ids": [r["match_id"] for r in rows],
            "training_min_date": rows[0]["date"], "training_max_date": rows[-1]["date"],
            "bounds": [0., .95], "at_boundary": False, "success": True}
        fits.append(deepcopy(result))
        return result

    def research(rows, k):
        cutoff = fits[-1]["before_date"]
        parameter = output / cutoff / experiment.PARAMETER_FILE
        assert parameter.is_file(), "Save the scalar before generating target predictions"
        assert read_json(parameter) == fits[-1]
        assert k == fits[-1]["k"]
        assert all(r["date"] >= cutoff for r in rows)
        events.append(("research", cutoff, tuple(r["match_id"] for r in rows)))
        return real_research(rows, k)

    monkeypatch.setattr(experiment, "load_period", load)
    monkeypatch.setattr(experiment, "fit_window", fit)
    monkeypatch.setattr(experiment, "research_tracks", research)
    return originals, output, events, fits


@pytest.fixture
def completed(tmp_path, monkeypatch):
    originals, output, events, fits = install_runner(tmp_path, monkeypatch)
    unchanged = deepcopy(originals)
    result = experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert originals == unchanged
    return result, originals, output, events, fits


def test_settings_precede_source_load_and_all_six_fixed_windows_score_every_target_once(completed):
    result, originals, output, events, fits = completed
    assert events[:2] == [("load", "2024-25"), ("load", "2025-26")]
    assert len(fits) == len(result["folds"]) == len(experiment.WINDOWS) == 6
    assert [f["cutoff"] for f in result["folds"]] == list(experiment.BOUNDARIES[:-1])
    scored = []
    for index, (cutoff, end) in enumerate(experiment.WINDOWS):
        assert events[2 + 2*index][:2] == ("fit", cutoff)
        assert events[3 + 2*index][:2] == ("research", cutoff)
        folder = output / cutoff
        assert read_json(folder / experiment.PARAMETER_FILE) == fits[index] == result["folds"][index]["fit"]
        assert read_json(folder / "同场分析.json") == result["folds"][index]["analysis"]
        tracks = read_json(folder / "完整研究预测.json")
        old_ids = [r["match_id"] for r in tracks["original"]]
        new_ids = [r["match_id"] for r in tracks[experiment.CANDIDATE]]
        assert old_ids == new_ids
        assert len(new_ids) == (1, 2, 3, 1, 2, 3)[index]
        scored.extend(new_ids)
        for arm in experiment.ARMS:
            assert all(r["dependence_fit_before_date"] == cutoff
                and r["dependence_training_start"] == experiment.TRAINING_START
                and r["dependence_target_end_exclusive"] == end for r in tracks[arm])
    expected = [r["match_id"] for r in originals
        if experiment.BOUNDARIES[0] <= r["date"] < experiment.BOUNDARIES[-1]]
    assert set(scored) == set(expected)
    assert len(scored) == len(set(scored)) == result["pooled"]["n"] == 12
    assert result["counts"] == {"new_dependence_fits": 6, "new_team_fits": 0,
        "unique_scored_matches": 12, "research_predictions": 12,
        "warmup_only_forecast_ids": [r["match_id"] for r in originals if r["date"] < experiment.BOUNDARIES[0]],
        "reused_team_parameter_snapshots": 5}
    assert read_json(output / experiment.RESULT_FILE) == result
    assert read_json(output / "实验设置.json") == result["settings"]
    settings = result["settings"]
    assert settings["k_bounds"] == [0., .95]
    assert settings["objective_scan"] == [i/20 for i in range(20)]
    assert settings["stacked_with_cmp"] is False
    assert settings["raw_means_changed"] is False
    assert settings["poisson_marginals_changed"] is False
    assert settings["production_api_changed"] is False
    assert settings["positive_dependence_only"] is True
    assert not (output / "研究失败.json").exists()


def test_saved_probabilities_preserve_source_means_scores_and_market_separation(completed):
    result, originals, output, _, fits = completed
    tracks = read_json(output / "完整前推预测.json")
    source = {r["match_id"]: r for r in originals}
    by_cutoff = {item["before_date"]: item["k"] for item in fits}
    for old, new in zip(tracks["original"], tracks[experiment.CANDIDATE]):
        raw = source[old["match_id"]]
        for key in ("rates", "actual_score", "actual_outcome", "market_probabilities", "market_prices"):
            assert old[key] == new[key] == raw[key]
        assert old["forecast"] == raw["forecast"]
        assert old["probabilities"] == raw["probabilities"]
        assert new["dependence_k"] == by_cutoff[new["dependence_fit_before_date"]]
        assert "dependence_numerics" in new
        assert "dependence_k" not in old
        assert not any("cmp_nu" in r or "cmp_numerics" in r for r in (old, new))
        for i, side in enumerate(("home", "away")):
            assert new["forecast"]["expected_goals"][side] == pytest.approx(raw["rates"][i], abs=1e-8)
    assert set(result["pooled"]["scores"]) == set(experiment.ARMS)


def test_event_diagnostics_use_joint_00_and_11_without_changing_marginal_zero_rates():
    rows = archive_rows()[:5]
    k = .4
    tracks = experiment.research_tracks(rows, k)
    old = experiment.event_summary(tracks["original"])
    new = experiment.event_summary(tracks[experiment.CANDIDATE])
    expected_00, expected_11, covariances, correlations = [], [], [], []
    for row in rows:
        mh, ma = row["rates"]
        c = k*min(mh, ma)
        p00 = np.exp(-(mh+ma-c))
        expected_00.append(p00)
        expected_11.append(p00*((mh-c)*(ma-c)+c))
        covariances.append(c)
        correlations.append(c/np.sqrt(mh*ma))
    for key, expected in (("zero_zero", expected_00), ("one_one", expected_11)):
        assert new["events"][key]["expected_count"] == pytest.approx(sum(expected), rel=1e-12)
    assert new["events"]["zero_zero"]["expected_count"] > old["events"]["zero_zero"]["expected_count"]
    for key, index in (("home_zero", 0), ("away_zero", 1)):
        assert new["events"][key] == old["events"][key]
        assert new["events"][key]["expected_count"] == pytest.approx(sum(np.exp(-r["rates"][index]) for r in rows))
    assert new["conditional_covariance"]["mean"] == pytest.approx(np.mean(covariances))
    assert new["conditional_pearson_correlation"]["mean"] == pytest.approx(np.mean(correlations))
    assert new["conditional_pearson_correlation"]["mean"] != pytest.approx(k)
    assert new["raw_goal_means"] == old["raw_goal_means"]
    assert new["observed_goal_means"] == old["observed_goal_means"]


def test_pooled_results_weight_all_matches_not_each_fold_equally(completed):
    result, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    vectors = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    n = result["pooled"]["n"]
    assert n == sum(f["analysis"]["n"] for f in result["folds"]) == 12
    assert n != len(result["folds"])
    for domain in experiment.DOMAINS:
        difference = vectors[experiment.CANDIDATE][domain]-vectors["original"][domain]
        pooled = result["pooled"][experiment.DELTA][domain]
        assert pooled == pytest.approx(difference.mean())
        weighted = sum(f["analysis"]["n"]*f["analysis"][experiment.DELTA][domain] for f in result["folds"])/n
        assert pooled == pytest.approx(weighted)
        for grouping in ("by_league", "by_month"):
            groups = result["pooled"][grouping]
            assert pooled == pytest.approx(sum(g["n"]*g[experiment.DELTA][domain] for g in groups.values())/n)
        assert pooled == pytest.approx(sum(g["n"]*g[experiment.DELTA][domain] for g in result["by_season"].values())/n)
        for weeks in ("2", "4", "8"):
            boot = result["pooled"]["bootstrap"][weeks]
            assert boot["n_matches"] == n
            assert boot["differences"][domain]["mean"] == pytest.approx(difference.mean())
    assert any(not np.isclose(result["pooled"][experiment.DELTA][domain],
        np.mean([f["analysis"][experiment.DELTA][domain] for f in result["folds"]])) for domain in experiment.DOMAINS)


def test_csv_contains_every_raw_mean_probability_and_full_support_loss(completed):
    _, _, output, _, _ = completed
    tracks = read_json(output / "完整前推预测.json")
    vectors = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    with (output / "逐场前推概率.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        assert len(reader.fieldnames) == 23
        assert all(not field.startswith("cmp_") for field in reader.fieldnames)
    assert len(saved) == 12
    for i, item in enumerate(saved):
        assert item["match_id"] == tracks["original"][i]["match_id"]
        assert [float(item[key]) for key in ("home_mean", "away_mean")] == tracks["original"][i]["rates"]
        for arm in experiment.ARMS:
            assert [float(item[f"{arm}_p_{side}"]) for side in ("home", "draw", "away")] == tracks[arm][i]["probabilities"]
            for domain in experiment.DOMAINS:
                assert float(item[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]


def test_all_loss_descriptives_keep_tails_and_signed_match_counts(completed):
    result, _, output, _, fits = completed
    tracks = read_json(output / "完整前推预测.json")
    vectors = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    for domain in experiment.DOMAINS:
        values = vectors[experiment.CANDIDATE][domain]-vectors["original"][domain]
        item = result["pooled"]["paired_loss_descriptives"][domain]
        assert item["n"] == len(values) == 12
        assert item["mean"] == pytest.approx(values.mean())
        assert item["sd"] == pytest.approx(values.std(ddof=1))
        for name, q in (("q25", .25), ("median", .5), ("q75", .75)):
            assert item[name] == pytest.approx(np.quantile(values, q))
        assert item["min"] == values.min() and item["max"] == values.max()
        assert item["n_negative"] == np.sum(values < 0)
        assert item["n_positive"] == np.sum(values > 0)
        assert item["n_zero"] == np.sum(values == 0)
    assert result["k_path_descriptives"]["n"] == 6
    assert result["k_path_descriptives"]["mean"] == pytest.approx(np.mean([f["k"] for f in fits]))


def test_report_and_parseable_histogram_describe_exploration_without_losing_matches(completed):
    result, _, output, _, _ = completed
    report = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    assert report == experiment.report(result)
    for phrase in ("6次共享比例k重估", "12场不重叠评分", "每个窗口仅一个活跃k", "没有叠加CMP",
        "不是Pearson相关系数", "未检验负相关", "逐场等权", "不是六次独立实验", "而非新盲测",
        "不包含k重估", "均值/SD/中位数/分位数/极值", "不裁掉高比分", "净胜至少3球",
        "不是亚洲盘退款", "没有神经网络", "正式模型未替换"):
        assert phrase in report
    root = ElementTree.parse(output / "逐场损失分布.svg").getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    texts = ["".join(node.itertext()) for node in root.findall(".//{http://www.w3.org/2000/svg}text")]
    assert all(domain in texts for domain in experiment.DOMAINS)
    assert sum(text.startswith("n=12, mean=") for text in texts) == 4
    assert experiment.CANDIDATE.upper() in texts[0]
    assert len(root.findall(".//{http://www.w3.org/2000/svg}rect")) == 121
    assert "nan" not in (output / "逐场损失分布.svg").read_text(encoding="utf-8").lower()


def test_null_research_matches_original_on_all_four_scoring_domains_without_mutation():
    rows = archive_rows()
    before = deepcopy(rows)
    tracks = experiment.research_tracks(rows, 0.)
    assert rows == before
    result, aligned = experiment.analyze(tracks, repeats=10)
    for old, new in zip(aligned["original"], aligned[experiment.CANDIDATE]):
        assert new["rates"] == old["rates"]
        assert new["probabilities"] == pytest.approx(old["probabilities"], abs=1e-12)
        assert new["exact_score_nll"] == pytest.approx(old["exact_score_nll"], abs=1e-12)
    assert all(abs(value) < 1e-12 for value in result[experiment.DELTA].values())
    assert result["event_diagnostics"][experiment.CANDIDATE]["events"] == result["event_diagnostics"]["original"]["events"]


@pytest.mark.parametrize("marker,value", [("cmp_nu", 1.1), ("dependence_k", .2)])
def test_research_rejects_stacking_instead_of_silently_overwriting_a_distribution(marker, value):
    rows = archive_rows()[:1]
    rows[0][marker] = value
    with pytest.raises(ValueError, match="no stacked"):
        experiment.research_tracks(rows, .2)


def test_research_retains_real_score_outside_grid_with_finite_exact_loss():
    rows = archive_rows()[:1]
    rows[0]["actual_score"], rows[0]["actual_outcome"] = [40, 37], 0
    tracks = experiment.research_tracks(rows, .3)
    assert all(len(tracks[arm]) == 1 for arm in experiment.ARMS)
    new = tracks[experiment.CANDIDATE][0]
    assert new["actual_score"] == [40, 37]
    assert np.isfinite(new["exact_score_nll"]) and new["exact_score_nll"] > 50
    assert new["exact_score_nll"] != tracks["original"][0]["exact_score_nll"]
    assert set(experiment.loss_vectors([new])) == set(experiment.DOMAINS)


@pytest.mark.parametrize("problem", ["missing_candidate", "changed_means"])
def test_analysis_does_not_hide_coverage_loss_or_changed_raw_means(problem):
    tracks = experiment.research_tracks(archive_rows()[:3], .2)
    if problem == "missing_candidate":
        tracks[experiment.CANDIDATE].pop()
    else:
        tracks[experiment.CANDIDATE][0]["rates"] = [8., 7.]
    with pytest.raises(ValueError):
        experiment.analyze(tracks, repeats=10)


def test_shared_cmp_analysis_csv_histogram_defaults_remain_cmp(tmp_path):
    tracks = cmp_experiment.research_tracks(archive_rows()[:5], 1.08)
    default, _ = cmp_experiment.analyze(tracks, repeats=10)
    explicit, _ = cmp_experiment.analyze(tracks, repeats=10, candidate="cmp",
        event_summary_fn=cmp_experiment.event_summary)
    assert default == explicit
    assert cmp_experiment.ARMS == cmp_forward_experiment.ARMS == ("original", "cmp")
    assert "cmp_minus_original" in default and experiment.DELTA not in default
    assert set(default["scores"]) == {"original", "cmp"}
    cmp_experiment.write_csv(tmp_path / "cmp.csv", tracks)
    with (tmp_path / "cmp.csv").open(encoding="utf-8-sig", newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    assert "cmp_exact_score_nll" in fields
    assert not any(field.startswith(experiment.CANDIDATE) for field in fields)
    cmp_forward_experiment.write_histogram(tmp_path / "cmp.svg", tracks)
    assert "CMP minus original" in (tmp_path / "cmp.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize("fail_index", [0, 2])
def test_fit_failure_preserves_completed_folds_and_trace_without_fallback(tmp_path, monkeypatch, fail_index):
    _, output, events, _ = install_runner(tmp_path, monkeypatch)
    real_fit = experiment.fit_window
    cutoff = experiment.WINDOWS[fail_index][0]
    trace = [{"k": .05, "nll": None, "numerical_status": "error"}]
    def fail(rows, *, training_start, before_date):
        if before_date == cutoff:
            events.append(("failed_fit", before_date))
            raise BivariatePoissonFitError("deliberate dependence fit failure", history=trace)
        return real_fit(rows, training_start=training_start, before_date=before_date)
    monkeypatch.setattr(experiment, "fit_window", fail)
    with pytest.raises(BivariatePoissonFitError, match="deliberate dependence"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert len([event for event in events if event[0] == "research"]) == fail_index
    assert not (output / cutoff).exists()
    assert read_json(output / "研究失败.json") == {"cutoff": cutoff, "type": "BivariatePoissonFitError",
        "message": "deliberate dependence fit failure", "history": trace, "failed_no_fallback": True}
    for earlier, _ in experiment.WINDOWS[:fail_index]:
        assert (output / earlier / experiment.PARAMETER_FILE).exists()
        assert (output / earlier / "同场分析.json").exists()
    for filename in (experiment.RESULT_FILE, experiment.REPORT_FILE, "完整前推预测.json"):
        assert not (output / filename).exists()


def test_research_failure_retains_saved_parameter_and_never_substitutes_original(tmp_path, monkeypatch):
    _, output, events, _ = install_runner(tmp_path, monkeypatch)
    cutoff = experiment.WINDOWS[0][0]
    def fail(rows, k):
        assert (output / cutoff / experiment.PARAMETER_FILE).is_file()
        raise RuntimeError("deliberate target distribution failure")
    monkeypatch.setattr(experiment, "research_tracks", fail)
    with pytest.raises(RuntimeError, match="deliberate target"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert len([event for event in events if event[0] == "fit"]) == 1
    assert (output / cutoff / experiment.PARAMETER_FILE).exists()
    assert not (output / cutoff / "完整研究预测.json").exists()
    assert not (output / experiment.RESULT_FILE).exists()
    failure = read_json(output / "研究失败.json")
    assert failure["cutoff"] == cutoff and failure["failed_no_fallback"] is True


def test_empty_target_window_stops_instead_of_dropping_a_fold(tmp_path, monkeypatch):
    originals, output, _, _ = install_runner(tmp_path, monkeypatch)
    cutoff, end = experiment.WINDOWS[0]
    originals[:] = [row for row in originals if not cutoff <= row["date"] < end]
    with pytest.raises(ValueError, match="no skipped folds"):
        experiment.run(tmp_path, tmp_path / "source", output, repeats=10)
    assert read_json(output / "研究失败.json")["cutoff"] == cutoff
    assert not (output / experiment.RESULT_FILE).exists()


def test_cli_default_root_and_source_use_current_checkout(tmp_path, monkeypatch, capsys):
    calls = []
    def run(root, parent, output):
        calls.append((root, parent, output))
        return {"pooled": {"n": 12, experiment.DELTA: {}}, "folds": [{"fit": {"k": .1}}]}
    monkeypatch.setattr(experiment, "run", run)
    monkeypatch.setattr(sys, "argv", ["bivariate_poisson_experiment", "--output", str(tmp_path / "chosen")])
    experiment.main()
    root, parent, output = calls[0]
    assert root == Path(experiment.__file__).resolve().parents[3]
    assert (root / "model_v2" / "pyproject.toml").is_file()
    assert parent == root / "outputs" / "更新频率研究_20261023T073047979805Z"
    assert output == tmp_path / "chosen"
    assert json.loads(capsys.readouterr().out)["k_path"] == [.1]
