"""Distribution-only CMP research: ordinary same-match and chronology tests."""

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor

import football_model.cmp_experiment as experiment
from football_model.cmp_fit import CMPFitError
from football_model.date_model import DateTeamGoalModel
from football_model.evaluation import probability_scores
from football_model.forecast_diagnostics import losses
import football_model.goal_rate_experiment as archive_fixture_target
from football_model.mean_cmp import mean_cmp
from football_model.penaltyblog_adapter import NativeGoalModel
import football_model.rate_calibration_experiment as calibration
from football_model.refresh_experiment import prediction_arrays
from football_model.team_model import TeamGoalModel

from test_goal_rate_experiment import install_archive_environment
from test_rate_calibration_experiment import later_rows


def research(nu=1.35):
    return experiment.research_tracks(later_rows(), nu)


def analysis(nu=1.35, repeats=10):
    return experiment.analyze(research(nu), repeats=repeats)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_research_preserves_raw_means_identity_and_inputs_but_changes_only_distribution():
    rows = later_rows()
    original = deepcopy(rows)
    tracks = experiment.research_tracks(rows, 1.35)
    assert rows == original
    assert tuple(tracks) == experiment.ARMS == ("original", "cmp")
    for source, old, new in zip(rows, tracks["original"], tracks["cmp"]):
        assert {k: old[k] for k in source} == source
        for key in ("rates", "match_id", "date", "fit_before_date", "training_max_date",
                    "training_matches", "actual_score", "actual_outcome", "source_url",
                    "market_prices", "market_probabilities"):
            assert old[key] == new[key] == source[key]
        assert new["cmp_nu"] == 1.35
        assert new["cmp_numerics"]
        assert new["probabilities"] == list(new["forecast"]["one_x_two"].values())
        assert new["forecast"] != old["forecast"]
        for index, side in enumerate(("home", "away")):
            assert new["forecast"]["expected_goals"][side] == pytest.approx(source["rates"][index], abs=1e-8)


def test_nu_one_recovers_original_probabilities_and_all_four_loss_domains():
    tracks = research(1.)
    for a, b in zip(tracks["original"], tracks["cmp"]):
        np.testing.assert_allclose(a["probabilities"], b["probabilities"], rtol=0, atol=1e-10)
    result, _ = experiment.analyze(tracks, repeats=10)
    for value in result["cmp_minus_original"].values():
        assert value == pytest.approx(0., abs=1e-9)
    for boot in result["bootstrap"].values():
        for item in boot["differences"].values():
            assert item["mean"] == pytest.approx(0., abs=1e-9)
            if item["interval_95"] is not None:
                assert item["interval_95"] == pytest.approx([0., 0.], abs=1e-9)


def test_exact_score_nll_uses_full_support_not_finite_forecast_grid():
    rows = later_rows()
    rows[0].update(actual_score=[100, 0], actual_outcome=0)
    tracks = experiment.research_tracks(rows, 1.4)
    for arm in experiment.ARMS:
        row = tracks[arm][0]
        nu = 1. if arm == "original" else 1.4
        expected = -math.fsum(mean_cmp(mu, nu).logpmf(y) for mu, y in zip(row["rates"], row["actual_score"]))
        assert np.isfinite(expected) and expected > 100
        assert row["exact_score_nll"] == pytest.approx(expected)


def test_analysis_aligns_by_identity_and_keeps_original_inputs_unchanged():
    expected, expected_rows = analysis()
    tracks = research()
    tracks["original"] = tracks["original"][2:] + tracks["original"][:2]
    tracks["cmp"].reverse()
    original = deepcopy(tracks)
    result, aligned = experiment.analyze(tracks, repeats=10)
    assert tracks == original
    assert result == expected and aligned == expected_rows
    assert result["coverage"]["counts"] == {"original": 6, "cmp": 6}
    assert all(not v for v in result["coverage"]["outside_common"].values())


@pytest.mark.parametrize("change", ["missing_arm", "missing", "extra", "duplicate", "source", "date", "score", "market", "rates"])
def test_analysis_rejects_missing_conflicting_or_mean_changed_matches(change):
    tracks = research()
    tracks["cmp"] = deepcopy(tracks["cmp"])
    if change == "missing_arm":
        tracks.pop("cmp")
    elif change == "missing":
        tracks["cmp"].pop()
    elif change == "extra":
        tracks["cmp"].append({**deepcopy(tracks["cmp"][0]), "match_id": "unmatched"})
    elif change == "duplicate":
        tracks["cmp"].append(deepcopy(tracks["cmp"][0]))
    elif change == "source":
        tracks["cmp"][0]["source_url"] += "?different"
    elif change == "date":
        tracks["cmp"][0]["date"] = "2025-08-05"
    elif change == "score":
        tracks["cmp"][0].update(actual_score=[5, 0], actual_outcome=0)
    elif change == "market":
        tracks["cmp"][0]["market_prices"] = {}
        tracks["cmp"][0]["market_probabilities"] = {}
    else:
        tracks["cmp"][0]["rates"][0] += .1
    with pytest.raises(ValueError):
        experiment.analyze(tracks, repeats=10)


@pytest.mark.parametrize("change", ["bool_rate", "negative_rate", "nan_rate", "incoherent_grid", "wrong_probabilities"])
def test_research_rejects_invalid_raw_rates_and_incoherent_original_forecast(change):
    rows = later_rows()
    if change == "bool_rate":
        rows[0]["rates"][0] = True
    elif change == "negative_rate":
        rows[0]["rates"][0] = -1.
    elif change == "nan_rate":
        rows[0]["rates"][0] = float("nan")
    elif change == "incoherent_grid":
        rows[0]["rates"][0] += .1
    else:
        rows[0]["probabilities"] = [.3, .3, .4]
    with pytest.raises(ValueError):
        experiment.research_tracks(rows, 1.35)


def test_four_domain_losses_and_all_group_summaries_recompose_same_matches():
    result, rows = analysis()
    assert result["n"] == 6
    assert tuple(result["cmp_minus_original"]) == experiment.DOMAINS
    vectors = {arm: experiment.loss_vectors(rows[arm]) for arm in experiment.ARMS}
    for domain in experiment.DOMAINS:
        assert result["cmp_minus_original"][domain] == pytest.approx(np.mean(vectors["cmp"][domain]-vectors["original"][domain]))
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        for arm in experiment.ARMS:
            expected = np.mean(vectors[arm][domain])
            assert result["scores"][arm][domain][field] == pytest.approx(expected)
            for grouping in ("by_league", "by_month"):
                recomposed = sum(g["n"] * g["scores"][arm][domain][field] for g in result[grouping].values()) / result["n"]
                assert recomposed == pytest.approx(expected)


def test_all_four_paired_loss_columns_share_calendar_block_resampling(monkeypatch):
    calls = []
    actual = experiment.moving_block_means
    def observed(dates, values, **kwargs):
        calls.append((list(dates), np.asarray(values).copy(), kwargs))
        return actual(dates, values, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result, rows = analysis(repeats=11)
    expected = np.column_stack([experiment.loss_vectors(rows["cmp"])[d]-experiment.loss_vectors(rows["original"])[d]
                                for d in experiment.DOMAINS])
    for weeks, (dates, matrix, kwargs) in zip((2, 4, 8), calls[:3]):
        assert dates == [r["date"] for r in rows["original"]]
        np.testing.assert_array_equal(matrix, expected)
        assert matrix.shape == (6, 4)
        assert kwargs == {"block_weeks": weeks, "repeats": 11, "seed": experiment.SEED}
        assert result["bootstrap"][str(weeks)]["n_matches"] == 6


def test_market_comparisons_use_same_observed_subset_for_both_models():
    result, rows = analysis()
    reference = rows["original"]
    for market, n in (("avg_preclosing", 6), ("avg_closing", 5)):
        item = result["markets"][market]
        indices = [i for i, r in enumerate(reference) if market in r["market_probabilities"]]
        assert item["n"] == n
        assert item["match_ids"] == [reference[i]["match_id"] for i in indices]
        assert item["missing_ids"] == [r["match_id"] for r in reference if market not in r["market_probabilities"]]
        p = np.asarray([reference[i]["market_probabilities"][market] for i in indices])
        y = np.asarray([reference[i]["actual_outcome"] for i in indices])
        assert item["market_scores"] == probability_scores(p, y)
        for arm in experiment.ARMS:
            model = prediction_arrays(rows[arm])["one_x_two"][0][indices]
            assert item["model_scores_on_same_matches"][arm] == probability_scores(model, y)
            assert item["model_minus_market"][arm]["mean"] == pytest.approx(float(np.mean(losses(model, y)-losses(p, y))))
    for item in result["markets"].values():
        if item["n"] == 0:
            assert "market_scores" not in item and "model_minus_market" not in item


def test_event_counts_use_cmp_probabilities_at_fixed_original_means_not_poisson_shortcuts():
    result, rows = analysis(nu=1.8)
    for arm in experiment.ARMS:
        diagnostics = result["event_diagnostics"][arm]
        np.testing.assert_array_equal(diagnostics["raw_goal_means"], np.mean([r["rates"] for r in rows[arm]], axis=0))
        probabilities = {k: [] for k in diagnostics["events"]}
        observed = {k: [] for k in diagnostics["events"]}
        for row in rows[arm]:
            h, a = row["actual_score"]
            hd, ad = (mean_cmp(mu, 1. if arm == "original" else 1.8) for mu in row["rates"])
            probabilities["home_zero"].append(math.exp(hd.logpmf(0)))
            probabilities["away_zero"].append(math.exp(ad.logpmf(0)))
            probabilities["zero_zero"].append(math.exp(hd.logpmf(0)+ad.logpmf(0)))
            probabilities["one_one"].append(math.exp(hd.logpmf(1)+ad.logpmf(1)))
            probabilities["draw"].append(row["probabilities"][1])
            probabilities["total_four_plus"].append(row["forecast"]["total_goals"]["4+"])
            for key, value in zip(("home_zero", "away_zero", "draw", "zero_zero", "one_one", "total_four_plus"),
                                  (h == 0, a == 0, h == a, h == a == 0, h == a == 1, h+a >= 4)):
                observed[key].append(value)
        for key, item in diagnostics["events"].items():
            assert item["expected_count"] == pytest.approx(sum(probabilities[key]))
            assert item["actual_count"] == sum(observed[key])
            assert item["actual_minus_expected_rate"] == pytest.approx(np.mean(observed[key])-np.mean(probabilities[key]))
    assert result["event_diagnostics"]["cmp"]["events"]["home_zero"]["expected_count"] != pytest.approx(
        result["event_diagnostics"]["original"]["events"]["home_zero"]["expected_count"])


@pytest.mark.parametrize("problem", ["zero_probability", "nan_exact_score"])
def test_invalid_observed_losses_are_not_silently_clipped(problem):
    rows = research()["cmp"]
    if problem == "zero_probability":
        draw = rows[0]["probabilities"][1]
        rows[0]["probabilities"][0] += draw
        rows[0]["probabilities"][1] = 0.
    else:
        rows[0]["exact_score_nll"] = float("nan")
    with pytest.raises(ValueError):
        experiment.loss_vectors(rows)


def test_csv_preserves_raw_means_and_every_arm_probability_and_loss(tmp_path):
    result, rows = analysis()
    path = tmp_path / "cmp.csv"
    experiment.write_csv(path, rows)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        assert len(reader.fieldnames) == 23
    assert len(saved) == result["n"]
    vectors = {arm: experiment.loss_vectors(rows[arm]) for arm in experiment.ARMS}
    for i, item in enumerate(saved):
        assert item["match_id"] == rows["original"][i]["match_id"]
        assert [float(item[k]) for k in ("home_mean", "away_mean")] == rows["original"][i]["rates"]
        for arm in experiment.ARMS:
            assert [float(item[f"{arm}_p_{k}"]) for k in ("home", "draw", "away")] == rows[arm][i]["probabilities"]
            for domain in experiment.DOMAINS:
                assert float(item[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]


def install_runner(tmp_path, monkeypatch):
    parent, sources, paths, events = install_archive_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(calibration, "season_directories", archive_fixture_target.season_directories)
    monkeypatch.setattr(calibration, "load_seasons", archive_fixture_target.load_seasons)
    output = tmp_path / "cmp-output"
    real_load, real_fit = experiment.load_period, experiment.fit_cmp
    def load(root, source, period):
        if period == "2025-26":
            assert (output / experiment.PARAMETER_FILE).is_file()
            assert events[-1][0] == "dispersion_fit_completed"
        events.append(("cmp_load", period))
        return real_load(root, source, period)
    def fit(rows):
        events.append(("dispersion_fit", tuple(r["match_id"] for r in rows)))
        assert all("2024-" in r["match_id"] for r in rows)
        result = real_fit(rows)
        events.append(("dispersion_fit_completed", result["nu"]))
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("CMP research must not train teams or regenerate team means")
    monkeypatch.setattr(experiment, "load_period", load)
    monkeypatch.setattr(experiment, "fit_cmp", fit)
    for model in (DateTeamGoalModel, NativeGoalModel, TeamGoalModel, PoissonRegressor):
        monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(DateTeamGoalModel, "predict", forbidden)
    return parent, sources, paths, events, output


def test_end_to_end_readable_archive_run_saves_earlier_fit_before_later_load_without_team_fits(tmp_path, monkeypatch):
    parent, _, paths, events, output = install_runner(tmp_path, monkeypatch)
    unchanged = {path: path.read_bytes() for path in paths.values()}
    result = experiment.run(tmp_path, parent, output, repeats=10)
    assert read_json(output / experiment.RESULT_FILE) == result
    assert read_json(output / experiment.PARAMETER_FILE) == result["fit"]
    assert read_json(output / "实验设置.json") == result["settings"]
    assert result["counts"] == {"new_team_fits": 0, "new_dispersion_fits": 1,
        "reused_parameter_snapshots": 10, "new_research_forecasts": 10}
    assert result["earlier"]["n"] == result["later"]["n"] == 5
    parameter = result["fit"]
    assert parameter["training_match_ids"] == sorted([f"2024-{c}" for c in ("D1", "E0", "F1", "I1", "SP1")])
    fit_position = next(i for i, e in enumerate(events) if e[0] == "dispersion_fit_completed")
    later_position = next(i for i, e in enumerate(events) if e == ("cmp_load", "2025-26"))
    assert fit_position < later_position
    for period in ("earlier", "later"):
        tracks = read_json(output / f"{period}_完整研究预测.json")
        assert len(tracks["original"]) == len(tracks["cmp"]) == 5
        assert [r["rates"] for r in tracks["original"]] == [r["rates"] for r in tracks["cmp"]]
        assert (output / f"{period}_同场概率.csv").is_file()
    assert all(path.read_bytes() == contents for path, contents in unchanged.items())
    assert not (output / "研究失败.json").exists()
    text = (output / experiment.REPORT_FILE).read_text(encoding="utf-8")
    for limitation in ("全部保留", "不是新盲测", "不能解释为足球固有离散规律", "主目标", "不自动采纳", "净胜至少3球", "不是亚洲盘走水", "没有加入神经网络"):
        assert limitation in text


def test_fit_failure_records_trace_and_never_opens_later_or_falls_back(tmp_path, monkeypatch):
    parent, _, _, events, output = install_runner(tmp_path, monkeypatch)
    trace = [{"nu": .25, "nll": None, "numerical_status": "error"}]
    def fail(rows):
        raise CMPFitError("synthetic numerical failure", history=trace)
    monkeypatch.setattr(experiment, "fit_cmp", fail)
    with pytest.raises(CMPFitError, match="synthetic"):
        experiment.run(tmp_path, parent, output, repeats=10)
    assert not any(e == ("cmp_load", "2025-26") for e in events)
    assert not (output / experiment.PARAMETER_FILE).exists()
    assert not (output / experiment.RESULT_FILE).exists()
    assert not (output / experiment.REPORT_FILE).exists()
    assert read_json(output / "研究失败.json") == {
        "type": "CMPFitError", "message": "synthetic numerical failure", "history": trace, "failed_no_fallback": True}


def test_later_source_failure_retains_saved_earlier_parameter_without_success_report(tmp_path, monkeypatch):
    parent, _, paths, _, output = install_runner(tmp_path, monkeypatch)
    path = paths[("2025-26", "D1", 28)]
    trial = read_json(path)
    trial["fits"][0]["coefficients"]["home:D1"] += .2
    path.write_text(json.dumps(trial), encoding="utf-8")
    unchanged = path.read_bytes()
    with pytest.raises(ValueError):
        experiment.run(tmp_path, parent, output, repeats=10)
    assert (output / experiment.PARAMETER_FILE).exists()
    assert path.read_bytes() == unchanged
    assert not (output / experiment.RESULT_FILE).exists()
    assert read_json(output / "研究失败.json")["failed_no_fallback"] is True


def test_cli_default_root_is_project_not_model_v2_or_source(tmp_path, monkeypatch, capsys):
    calls = []
    def run(root, parent, output):
        calls.append((root, parent, output))
        return {"fit": {"nu": 1.2}, "later": {"cmp_minus_original": {}}}
    monkeypatch.setattr(experiment, "run", run)
    monkeypatch.setattr(sys, "argv", ["cmp_experiment", "--output", str(tmp_path / "chosen")])
    experiment.main()
    root, parent, output = calls[0]
    assert root == Path(experiment.__file__).resolve().parents[3]
    assert root.name == "football-prediction-2026-10-22"
    assert parent == root / "outputs" / "更新频率研究_20261023T073047979805Z"
    assert output == tmp_path / "chosen"
    assert json.loads(capsys.readouterr().out)["nu"] == 1.2
