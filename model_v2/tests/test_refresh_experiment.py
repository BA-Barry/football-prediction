"""Independent refresh-analysis tests; mocked runs do not replace historical fits."""

from copy import deepcopy
import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

import football_model.refresh_experiment as experiment
from football_model.evaluation import probability_scores
from football_model.football_data import HistoricalMatch, LEAGUES, MARKET_COLUMNS
from football_model.forecast_diagnostics import BLOCKS, CLASS_NAMES, moving_block_means
from football_model.probability import score_grid, summarize


def prediction(key="m1", day="2025-08-15", score=(2, 0), comp="E0", interval=28,
               rates=(1.5, 1.1)):
    played = date.fromisoformat(day)
    anchor = date(played.year if played.month >= 8 else played.year-1, 8, 1)
    cutoff = anchor + timedelta(days=(played-anchor).days//interval*interval)
    forecast = summarize(score_grid(*rates, rho=0))
    return {"match_id": key, "date": day, "competition": comp,
        "home": "Home "+key, "away": "Away "+key,
        "fit_before_date": cutoff.isoformat(),
        "training_max_date": (cutoff-timedelta(days=1)).isoformat(),
        "training_matches": 100+(cutoff-anchor).days,
        "actual_score": list(score),
        "actual_outcome": 0 if score[0] > score[1] else 2 if score[0] < score[1] else 1,
        "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
        "source_url": "https://example.invalid/synthetic.csv",
        "market_prices": {market: [2., 4., 4.] for market in MARKET_COLUMNS},
        "market_probabilities": {market: [.5, .25, .25] for market in MARKET_COLUMNS}}


def source(row, *, key=None, day=None, comp=None, home=None, away=None, exclusion=None):
    return HistoricalMatch(key or row["match_id"], comp or row["competition"], "synthetic",
        date.fromisoformat(day or row["date"]), None, home or row["home"], away or row["away"],
        home or row["home"], away or row["away"], tuple(row["actual_score"]), None, {},
        "synthetic", row["source_url"], exclusion)


def example_tracks(year=2025):
    days = [f"{year}-08-04", f"{year}-08-11", f"{year}-08-11", f"{year}-09-08",
            f"{year}-10-06", f"{year}-11-03", f"{year}-11-10"]
    scores = [(2, 0), (1, 1), (0, 3), (1, 2), (0, 2), (2, 1), (2, 2)]
    comps = ["E0", "E0", "E0", "E0", "D1", "I1", "E0"]
    tracks = {arm: [prediction(f"{year}-m{i}", day, score, comp, interval,
                  rates=(1.3+.13*i+(28-interval)/120, 1.0+.07*i))
                  for i, (day, score, comp) in enumerate(zip(days, scores, comps))]
              for arm, interval in zip(experiment.ARMS, experiment.INTERVALS)}
    return tracks, [source(row) for row in tracks["days_28"]]


def analyze(tracks=None, records=None, repeats=31):
    if tracks is None:
        tracks, records = example_tracks()
    return experiment.analyze_period(tracks, records, repeats=repeats)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_five_bucket_scores_match_independent_log_brier_and_ordered_rps():
    p = np.array([[.1, .2, .3, .2, .2], [.25, .25, .2, .2, .1], [.05, .1, .15, .3, .4]])
    y = np.array([0, 3, 4])
    actual = experiment.categorical_scores(p, y)
    truth = np.eye(5)[y]
    assert actual["n"] == 3
    assert actual["log_loss"] == pytest.approx(-sum(math.log(p[i, label]) for i, label in enumerate(y))/3)
    assert actual["brier_sum"] == pytest.approx(sum(sum((p[i, j]-truth[i, j])**2 for j in range(5)) for i in range(3))/3)
    expected_rps = sum(sum((sum(p[i, :j+1])-sum(truth[i, :j+1]))**2 for j in range(4)) for i in range(3))/12
    assert actual["rps"] == pytest.approx(expected_rps)
    assert actual["accuracy"] == pytest.approx(1/3)


@pytest.mark.parametrize("p,y", [
    ([], []), ([[.25]*4], [0]), ([[.2]*5], [5]), ([[.2]*5], [-1]),
    ([[.2]*5], [1.0]), ([[.2]*5], [True]), ([[.2]*5], [[1]]),
    ([[.2]*5], [0, 1]), ([[.2]*5], []), ([[.2, .2, .2, .2, .3]], [0]),
    ([[-.1, .2, .2, .2, .5]], [0]), ([[1.1, -.1, 0, 0, 0]], [0]),
    ([[float("nan"), .2, .2, .2, .2]], [0]), ([[float("inf"), 0, 0, 0, 0]], [0]),
])
def test_five_bucket_scores_reject_invalid_probabilities_or_labels(p, y):
    with pytest.raises(ValueError):
        experiment.categorical_scores(p, y)


def test_zero_probability_event_is_scored_not_dropped_and_perfect_scores_zero():
    perfect = experiment.categorical_scores([[1, 0, 0, 0, 0]], [0])
    assert all(perfect[key] == 0 for key in ("log_loss", "brier_sum", "rps"))
    failed = experiment.categorical_scores([[1, 0, 0, 0, 0]], [4])
    assert failed["n"] == 1
    assert failed["log_loss"] == pytest.approx(-math.log(np.finfo(float).tiny))
    assert failed["brier_sum"] == 2
    assert failed["rps"] == 1


@pytest.mark.parametrize("score,total,handicap", [
    ((0, 0), 0, 2), ((1, 0), 1, 2), ((0, 2), 2, 1), ((0, 3), 3, 0),
    ((1, 3), 4, 1), ((1, 4), 4, 0), ((4, 7), 4, 0), ((4, 6), 4, 1),
    ((10, 9), 4, 2), ((1, 2), 3, 2),
])
def test_actual_total_buckets_and_away_minus_two_require_margin_at_least_three(score, total, handicap):
    row = prediction(score=score)
    arrays = experiment.prediction_arrays([row])
    assert arrays["total_goals_0_1_2_3_4plus"][1].tolist() == [total]
    assert arrays["away_minus_two_three_way"][1].tolist() == [handicap]
    assert arrays["away_minus_two_three_way"][0].tolist() == [[row["forecast"]["away_minus_two_jincai"][k] for k in ("win", "draw", "lose")]]
    assert experiment.scores_for([row])["one_x_two"] == probability_scores([row["probabilities"]], [row["actual_outcome"]])


def test_freshness_strictly_earlier_dates_same_league_exclusions_and_two_team_count():
    row = prediction(day="2025-08-15", interval=28)
    records = [source(row), source(row, key="before", day="2025-07-31"),
        source(row, key="at-cutoff", day="2025-08-01"),
        source(row, key="one-team", day="2025-08-10", away="Other"),
        source(row, key="neither", day="2025-08-14", home="Another", away="Else"),
        source(row, key="same-day", day="2025-08-15"),
        source(row, key="future", day="2025-08-16"),
        source(row, key="other-league", day="2025-08-10", comp="D1"),
        source(row, key="awarded", day="2025-08-10", exclusion="administratively_awarded_result"),
        source(row, key="resumed", day="2025-08-10", exclusion="resumed_partially_played_match")]
    result = experiment.freshness([row], records)
    assert result == [{"match_id": "m1", "snapshot_age_days": 14,
        "last_included_result_age_days": 15,
        "eligible_results_since_cutoff_before_match_date": 3,
        "involving_either_team_since_cutoff_before_match_date": 2}]
    # A match involving both targets is one result, not two player/team appearances.
    assert experiment.freshness([row], list(reversed(records))) == result


@pytest.mark.parametrize("field,value", [("date", "2025-08-16"),
    ("training_max_date", "2025-08-01"), ("fit_before_date", "2025-08-16")])
def test_freshness_invalid_chronology_not_silently_fixed(field, value):
    row = prediction()
    record = source(row)
    row[field] = value
    with pytest.raises(ValueError):
        experiment.freshness([row], [record])


def test_freshness_duplicate_source_ids_rejected():
    row = prediction()
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.freshness([row], [source(row), source(row)])


def test_describe_is_sample_sd_and_single_observation_has_no_fabricated_sd():
    assert experiment.describe([2]) == {"n": 1, "mean": 2., "sd": None,
        "min": 2., "q25": 2., "median": 2., "q75": 2., "max": 2.}
    actual = experiment.describe([1, 2, 3, 9])
    assert actual["sd"] == pytest.approx(np.std([1, 2, 3, 9], ddof=1))
    assert actual["median"] == 2.5


@pytest.mark.parametrize("values", [[], [[1, 2]], [float("nan")], [float("inf")]])
def test_describe_refuses_empty_nonfinite_or_nonscalar_values(values):
    with pytest.raises(ValueError):
        experiment.describe(values)


def test_period_requires_all_arms_and_rejects_duplicate_identity_conflicts():
    tracks, records = example_tracks()
    incomplete = {key: value for key, value in tracks.items() if key != "days_14"}
    with pytest.raises(ValueError, match="three"):
        analyze(incomplete, records)
    tracks["days_7"].append(deepcopy(tracks["days_7"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(tracks, records)
    tracks, records = example_tracks()
    tracks["days_7"][0]["actual_score"] = [0, 2]
    tracks["days_7"][0]["actual_outcome"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        analyze(tracks, records)


def test_common_cohort_and_separate_extra_coverage_not_mix_in_main_scores():
    tracks, records = example_tracks()
    extra = prediction("extra", "2025-08-17", (0, 5), interval=7)
    tracks["days_7"].append(extra)
    records.append(source(extra))
    before = deepcopy((tracks, records))
    result, entries, fresh = analyze(tracks, records)
    assert result["n"] == 7
    assert result["coverage"]["counts"] == {"days_7": 8, "days_14": 7, "days_28": 7}
    assert result["coverage"]["outside_common"]["days_7"] == ["extra"]
    assert result["extra_coverage"]["days_7"]["scores_separate_not_comparable_to_common"] == experiment.scores_for([extra])
    assert result["extra_coverage"]["days_28"]["scores_separate_not_comparable_to_common"] is None
    for arm in experiment.ARMS:
        assert result["scores"][arm]["one_x_two"]["n"] == 7
        assert [row["match_id"] for row in entries[arm]] == result["coverage"]["common_match_ids"]
        assert len(fresh[arm]) == 7
    assert (tracks, records) == before


def test_missing_market_does_not_change_main_cohort_and_uses_same_market_subcohort():
    tracks, records = example_tracks()
    original = analyze(tracks, records)[0]
    for rows in tracks.values():
        for row in rows[:2]:
            row["market_probabilities"].pop("avg_preclosing")
            row["market_prices"].pop("avg_preclosing")
        for row in rows:
            row["market_probabilities"].pop("avg_closing")
            row["market_prices"].pop("avg_closing")
    result = analyze(tracks, records)[0]
    assert result["n"] == original["n"]
    assert result["scores"] == original["scores"]
    assert result["markets"]["avg_preclosing"]["n"] == 5
    assert result["markets"]["avg_preclosing"]["missing_ids"] == [row["match_id"] for row in tracks["days_28"][:2]]
    assert result["markets"]["avg_closing"] == {"n": 0, "missing_ids": [row["match_id"] for row in tracks["days_28"]]}
    for arm in experiment.ARMS:
        market = result["markets"]["avg_preclosing"]
        rows = tracks[arm][2:]
        scores = probability_scores([row["probabilities"] for row in rows], [row["actual_outcome"] for row in rows])
        assert market["arm_scores_on_same_matches"][arm] == scores
        assert market["differences"][arm]["mean"] == pytest.approx(scores["log_loss"]-market["market_scores"]["log_loss"])


def test_joint_bootstrap_all_arms_domains_classes_same_weights_and_no_empty_week_compression(monkeypatch):
    tracks, records = example_tracks()
    calls = []
    def observed(dates, values, **kwargs):
        calls.append((list(dates), np.asarray(values).copy(), dict(kwargs)))
        return moving_block_means(dates, values, **kwargs)
    monkeypatch.setattr(experiment, "moving_block_means", observed)
    result = analyze(tracks, records, repeats=43)[0]
    assert [call[2]["block_weeks"] for call in calls[:3]] == list(BLOCKS)
    assert all(call[1].shape == (7, 15) for call in calls[:3])
    assert all(call[2]["repeats"] == 43 and call[2]["seed"] == experiment.SEED for call in calls)
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    np.testing.assert_array_equal(calls[1][1], calls[2][1])
    for weeks, call in zip(BLOCKS, calls):
        oracle = moving_block_means(call[0], call[1], **call[2])
        saved = result["bootstrap"][str(weeks)]
        assert saved["n_calendar_weeks"] > saved["n_nonempty_weeks"]
        assert 0 in saved["weekly_match_counts"]
        for i, feature in enumerate(saved["features"].values()):
            assert feature["mean"] == pytest.approx(oracle["mean"][i])
            assert feature["interval_95"] == oracle["interval_95"][i]
    assert len(calls) == 3+len(MARKET_COLUMNS)


def test_identical_forecast_arms_have_zero_paired_changes_and_intervals():
    tracks, records = example_tracks()
    for arm in experiment.CONTRASTS:
        for row, original in zip(tracks[arm], tracks["days_28"]):
            row["probabilities"] = list(original["probabilities"])
            row["forecast"] = deepcopy(original["forecast"])
    result = analyze(tracks, records)[0]
    for contrast in result["mean_differences"].values():
        assert all(value == 0 for value in contrast.values())
    for boot in result["bootstrap"].values():
        for name, feature in boot["features"].items():
            if "minus" in name:
                assert feature == {"mean": 0., "interval_95": [0., 0.]}
    assert all(result["probability_change_from_28"][arm]["total_variation"]["max"] == 0 for arm in experiment.CONTRASTS)


def test_paired_weighted_outcome_contributions_groups_and_freshness_reconstruct():
    tracks, records = example_tracks()
    result, entries, fresh = analyze(tracks, records)
    for arm in experiment.CONTRASTS:
        expected = np.array([math.log(old["probabilities"][old["actual_outcome"]]/new["probabilities"][new["actual_outcome"]])
                             for old, new in zip(entries["days_28"], entries[arm])])
        assert result["mean_differences"][f"{arm}_minus_days_28"]["one_x_two"] == pytest.approx(expected.mean())
        contributions = result["one_x_two_outcome_contributions"][arm]
        assert sum(v["n"] for v in contributions.values()) == result["n"]
        assert sum(v["weighted_contribution"] for v in contributions.values()) == pytest.approx(expected.mean())
        for c, label in enumerate(CLASS_NAMES):
            assert contributions[label]["weighted_contribution"] == pytest.approx(sum(expected[i] for i, row in enumerate(entries[arm]) if row["actual_outcome"] == c)/len(expected))
        assert result["probability_change_from_28"][arm]["added_training_matches"]["mean"] == pytest.approx(np.mean([new["training_matches"]-old["training_matches"] for new, old in zip(entries[arm], entries["days_28"])]))
    for grouping in ("by_league", "by_month", "by_28_day_snapshot_age"):
        groups = result[grouping]
        assert sum(group["n"] for group in groups.values()) == result["n"]
        for arm in experiment.ARMS:
            for domain, score in result["scores"][arm].items():
                for metric in ("log_loss", "brier_sum", "rps", "accuracy"):
                    reconstructed = sum(g["n"]*g["scores"][arm][domain][metric] for g in groups.values())/result["n"]
                    assert reconstructed == pytest.approx(score[metric])
    for arm in experiment.ARMS:
        for field, stats in result["freshness"][arm].items():
            assert stats == experiment.describe([row[field] for row in fresh[arm]])


def test_record_order_does_not_change_common_analysis():
    tracks, records = example_tracks()
    expected = analyze(tracks, records)[0]
    for rows in tracks.values():
        rows.reverse()
    assert analyze(tracks, list(reversed(records)))[0] == expected


def test_csv_reproduces_all_arm_probabilities_freshness_and_three_domain_losses(tmp_path):
    result, entries, fresh = analyze()
    target = tmp_path/"paired.csv"
    experiment.write_csv(target, entries, fresh)
    assert target.read_bytes().startswith(b"\xef\xbb\xbf")
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["match_id"] for row in rows] == result["coverage"]["common_match_ids"]
    for i, row in enumerate(rows):
        assert int(row["actual_total_goal_bucket"]) == min(sum(entries["days_28"][i]["actual_score"]), 4)
        for arm in experiment.ARMS:
            for c, label in enumerate(CLASS_NAMES):
                assert float(row[f"{arm}_p_{label}"]) == entries[arm][i]["probabilities"][c]
            assert row[f"{arm}_fit_before_date"] == entries[arm][i]["fit_before_date"]
            for field, value in fresh[arm][i].items():
                if field != "match_id":
                    assert int(row[f"{arm}_{field}"]) == value
            for domain, field in (("one_x_two", "one_x_two_log_loss"),
                    ("total_goals_0_1_2_3_4plus", "total_goal_bucket_log_loss"),
                    ("away_minus_two_three_way", "away_minus_two_log_loss")):
                p, y = experiment.prediction_arrays([entries[arm][i]])[domain]
                assert float(row[f"{arm}_{field}"]) == pytest.approx(-math.log(p[0, y[0]]))


def test_histogram_is_raw_paired_changes_two_panels_not_independent_class_samples():
    _, entries, _ = analyze()
    svg = experiment.difference_histogram(entries)
    root = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    text = " ".join(node.text or "" for node in root.findall("svg:text", ns))
    assert "common n=7" in text and "negative better" in text
    assert "days_7 minus days_28" in text and "days_14 minus days_28" in text
    assert len(root.findall("svg:rect", ns)) == 61
    assert all(math.isfinite(float(node.attrib["height"])) for node in root.findall("svg:rect", ns))


def test_all_zero_histogram_has_finite_axis_and_heights():
    _, entries, _ = analyze()
    for arm in experiment.CONTRASTS:
        entries[arm] = deepcopy(entries["days_28"])
    svg = experiment.difference_histogram(entries)
    root = ElementTree.fromstring(svg)
    assert "nan" not in svg.lower() and "inf" not in svg.lower()
    assert len(root) > 60


def install_mock_runner(monkeypatch, tmp_path):
    events = []
    seasons = {season: tmp_path/season.replace("/", "-") for season in ("2022/23", "2023/24", "2024/25", "2025/26")}
    all_records = []
    for year in (2024, 2025):
        for comp in LEAGUES:
            all_records.append(source(prediction(f"{year}-{comp}", f"{year}-08-15", comp=comp)))
    earlier, later = tmp_path/"earlier", tmp_path/"later"
    earlier.mkdir()
    later.mkdir()
    (earlier/"较早校准研究结果.json").write_text(json.dumps({"parent_directory": str(later)}), encoding="utf-8")
    def load(selected):
        labels = tuple(selected)
        events.append(("load", labels))
        cutoff_year = max(int(season[:4]) for season in labels)
        return [r for r in all_records if r.match_date.year <= cutoff_year], {season: {"synthetic": True} for season in labels}
    def trial(records, start, end, interval):
        comp = records[0].competition
        events.append(("reference" if interval == 28 else "fit", start.year, comp, interval))
        selected = [r for r in records if start <= r.match_date < end]
        rows = [prediction(r.match_id, r.match_date.isoformat(), r.full_time, comp, interval,
                           rates=(1.5+(28-interval)/100, 1.1)) for r in selected]
        return {"config": experiment.ORIGINAL, "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "update_interval_days": interval, "predictions": rows,
            "fits": [{"synthetic": True}], "failures": [], "skipped": []}
    def reference(path, records, start, end, comp):
        assert all(r.competition == comp for r in records)
        assert ("较早旧模型预测" in str(path)) == (start.year == 2024)
        return trial(records, start, end, 28)
    def new_frequency(records, start, end, interval):
        assert interval in (7, 14), "28-day archives must never be refitted"
        return trial(records, start, end, interval)
    monkeypatch.setattr(experiment, "season_directories", lambda root: seasons)
    monkeypatch.setattr(experiment, "load_seasons", load)
    monkeypatch.setattr(experiment, "load_reference", reference)
    monkeypatch.setattr(experiment, "run_frequency", new_frequency)
    return earlier, later, events


def test_mock_end_to_end_filters_earlier_seasons_reuses_28_all_outputs_and_no_promotion(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier, later, repeats=17)
    result = read_json(output/experiment.RESULT_FILE)
    settings = read_json(output/"实验设置.json")
    assert output.parent == tmp_path/"outputs"
    assert settings == result["settings"]
    assert settings["primary_contrast"] == "days_7_minus_days_28"
    assert settings["secondary_contrast"] == "days_14_minus_days_28"
    assert settings["previously_viewed_seasons"] == list(experiment.PERIODS)
    for key in ("model_selection", "calibration", "formal_probability_api_changed", "market_fusion"):
        assert settings[key] is False
    assert [e for e in events if e[0] == "load"] == [("load", ("2022/23", "2023/24", "2024/25")),
        ("load", ("2022/23", "2023/24", "2024/25", "2025/26"))]
    first_later = next(i for i, event in enumerate(events) if event[0] == "load" and "2025/26" in event[1])
    assert len([e for e in events[:first_later] if e[0] == "fit"]) == 10
    assert len([e for e in events if e[0] == "fit"]) == 20
    assert len([e for e in events if e[0] == "reference"]) == 10
    assert result["counts"] == {"new_base_model_fits": 20, "reused_base_parameter_snapshots": 10,
        "new_prediction_rows": 20, "reused_prediction_rows": 10}
    assert list(result["periods"]) == list(experiment.PERIODS)
    assert read_json(output/"阶段进度.json")["completed_periods"] == list(experiment.PERIODS)
    for period in experiment.PERIODS:
        assert result["periods"][period]["n"] == len(LEAGUES)
        assert (output/period/"同场概率与损失.csv").exists()
        assert (output/period/"原始逐场损失差.svg").exists()
        for comp in LEAGUES:
            for interval in experiment.INTERVALS:
                saved = read_json(output/period/f"{comp}_days_{interval}.json")
                assert saved["update_interval_days"] == interval
                assert "forecast" in saved["predictions"][0]
    text = (output/experiment.REPORT_FILE).read_text(encoding="utf-8")
    assert "探索性" in text and "不是新盲测" in text and "同日结果不用" in text
    assert "净胜≥3" in text and "不是亚洲盘走盘" in text
    assert "没有市场融合" in text and "半全场" in text
    assert "不可估计" in text  # One calendar week cannot support these block intervals.
    assert "2024-25" in text and "2025-26" in text


def test_runner_infers_later_parent_from_earlier_result(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier_parent=earlier, repeats=7)
    assert read_json(output/"实验设置.json")["later_parent"] == str(later.resolve())
    assert len([e for e in events if e[0] == "fit"]) == 20


def test_runner_reference_failure_does_not_refit_28_or_silently_continue(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    def broken(*args):
        raise ValueError("Broken archive")
    monkeypatch.setattr(experiment, "load_reference", broken)
    with pytest.raises(ValueError, match="Broken archive"):
        experiment.run(tmp_path, earlier, later, repeats=7)
    assert not any(event[0] == "fit" for event in events)


def test_runner_missing_earlier_archives_does_not_start_training(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "season_directories", lambda root: calls.append(True))
    with pytest.raises(ValueError, match="earlier original"):
        experiment.run(tmp_path)
    assert calls == []


def test_report_handles_wholly_missing_markets_without_substitution(tmp_path, monkeypatch):
    earlier, later, _ = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier, later, repeats=7)
    result = read_json(output/experiment.RESULT_FILE)
    for period in result["periods"].values():
        for market in MARKET_COLUMNS:
            period["markets"][market] = {"n": 0, "missing_ids": period["coverage"]["common_match_ids"]}
    text = experiment.report(result)
    assert "| 2024-25 | avg_preclosing | 0 | — | — | — | — |" in text
    assert "| 2025-26 | avg_closing | 0 | — | — | — | — |" in text


def test_json_list_and_source_tuple_containers_are_equivalent_without_mutating_inputs():
    tracks, records = example_tracks()
    expected = analyze(tracks, records)[0]
    for row in tracks["days_7"]:
        row["market_prices"] = {key: tuple(value) for key, value in row["market_prices"].items()}
        row["actual_score"] = tuple(row["actual_score"])
    original = deepcopy(tracks)
    assert analyze(tracks, records)[0] == expected
    assert tracks == original


def test_equivalent_containers_do_not_hide_real_quote_changes():
    tracks, records = example_tracks()
    row = tracks["days_7"][0]
    row["market_prices"]["avg_preclosing"] = (3., 4., 4.)
    inverse = 1/np.asarray(row["market_prices"]["avg_preclosing"])
    row["market_probabilities"]["avg_preclosing"] = (inverse/inverse.sum()).tolist()
    with pytest.raises(ValueError, match="Conflicting"):
        analyze(tracks, records)


def test_resume_reloads_completed_arms_without_refits_and_keeps_counts(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier, later, repeats=17)
    initial = read_json(output/experiment.RESULT_FILE)
    events.clear()
    validations = []
    def validate(trial, records, start, end, comp, interval):
        assert trial["update_interval_days"] == interval
        validations.append((start.year, comp, interval))
        return trial
    monkeypatch.setattr(experiment, "validate_trial", validate)
    resumed = experiment.run(tmp_path, repeats=17, resume=output)
    assert resumed == output
    result = read_json(output/experiment.RESULT_FILE)
    assert not any(event[0] == "fit" for event in events)
    assert len(validations) == 30
    assert result["counts"] == initial["counts"]
    assert result["periods"] == initial["periods"]
    assert result["execution"] == {"resumed": True, "new_fits_in_this_invocation": 0,
        "new_frequency_fits_reloaded_without_refitting": 20}


def test_partial_resume_keeps_valid_fits_and_fits_only_missing_arm(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    original_frequency = experiment.run_frequency
    def interrupt_after_one(records, start, end, interval):
        if len([event for event in events if event[0] == "fit"]) >= 1:
            raise RuntimeError("simulated interruption before next fit")
        return original_frequency(records, start, end, interval)
    monkeypatch.setattr(experiment, "run_frequency", interrupt_after_one)
    with pytest.raises(RuntimeError):
        experiment.run(tmp_path, earlier, later, repeats=17)
    output, = (tmp_path/"outputs").glob("更新频率研究_*")
    first_saved = next((output/"2024-25").glob("*_days_7.json"))
    saved_bytes = first_saved.read_bytes()
    monkeypatch.setattr(experiment, "run_frequency", original_frequency)
    monkeypatch.setattr(experiment, "validate_trial", lambda trial, *args, **kwargs: trial)
    events.clear()
    experiment.run(tmp_path, repeats=17, resume=output)
    result = read_json(output/experiment.RESULT_FILE)
    assert len([event for event in events if event[0] == "fit"]) == 19
    assert first_saved.read_bytes() == saved_bytes
    assert result["counts"]["new_base_model_fits"] == 20
    assert result["execution"]["new_fits_in_this_invocation"] == 19
    assert result["execution"]["new_frequency_fits_reloaded_without_refitting"] == 1


def test_bad_resume_does_not_retrain_or_replace_invalid_saved_arm(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier, later, repeats=17)
    events.clear()
    def reject(*args, **kwargs):
        raise ValueError("Saved interval is damaged")
    monkeypatch.setattr(experiment, "validate_trial", reject)
    with pytest.raises(ValueError, match="damaged"):
        experiment.run(tmp_path, repeats=17, resume=output)
    assert not any(event[0] == "fit" for event in events)


def test_resume_settings_mismatch_refused_before_loading_or_fitting(tmp_path, monkeypatch):
    earlier, later, events = install_mock_runner(monkeypatch, tmp_path)
    output = experiment.run(tmp_path, earlier, later, repeats=17)
    events.clear()
    with pytest.raises(ValueError, match="settings differ"):
        experiment.run(tmp_path, repeats=19, resume=output)
    assert not events


def test_resume_refuses_unrelated_output_directory(tmp_path):
    with pytest.raises(ValueError, match="explicitly named"):
        experiment.run(tmp_path, resume=tmp_path/"other")


@pytest.mark.parametrize("repeats", [True, 1, 1.5, 0, -1])
def test_invalid_repeats_refused_before_creating_output(tmp_path, repeats):
    with pytest.raises(ValueError, match="repeats"):
        experiment.run(tmp_path, repeats=repeats)
    assert not (tmp_path/"outputs").exists()

