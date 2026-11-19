"""Small synthetic examples only; no real-season fitting in these tests."""

import csv
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import math

import pytest

from football_model.football_data import HistoricalMatch
from football_model import second_season_team_experiment as experiment_module
from football_model.second_season_team_experiment import (
    MARKET, evaluate, full_time_forecast, load_season_csv, monday, weekly_walk_forward,
)


def history():
    start = date(2018, 1, 1)
    return [HistoricalMatch(
        f"synthetic-{i}", "E0", "2017/18", start + timedelta(days=i), None,
        "A" if i % 2 else "B", "B" if i % 2 else "A",
        "A" if i % 2 else "B", "B" if i % 2 else "A",
        (i % 4, (i + 1) % 3), None, {MARKET: (2., 3.5, 4.)}, "synthetic", "synthetic",
    ) for i in range(49)]


def test_thursday_targets_use_preceding_monday_not_same_week_results():
    records = history()
    result = weekly_walk_forward(records, evaluation_start=date(2018, 2, 1))
    assert monday(date(2018, 2, 1)) == date(2018, 1, 29)
    assert len(result["predictions"]) == 18
    assert len(result["fits"]) == 3
    first = result["fits"][0]
    assert first["before_date"] == "2018-01-29"
    assert first["training_max_date"] == "2018-01-28"
    assert first["n_matches"] == 28
    for fit in result["fits"]:
        cutoff = date.fromisoformat(fit["before_date"])
        assert cutoff.weekday() == 0
        assert fit["regularization"] == .01
        assert fit["half_life_days"] == 180
        assert set(fit["reference_match_ids"]) == {r.match_id for r in records if r.match_date < cutoff}
    for row in result["predictions"]:
        assert row["training_max_date"] < row["fit_before_date"] <= row["date"]
    json.dumps(result["fits"], allow_nan=False)


def test_changing_any_week_score_cannot_change_predictions_within_that_week():
    records = history()
    a = weekly_walk_forward(records, evaluation_start=date(2018, 2, 1))
    changed = [replace(r, full_time=(19, 0))
               if date(2018, 1, 29) <= r.match_date < date(2018, 2, 5) else r for r in records]
    b = weekly_walk_forward(changed, evaluation_start=date(2018, 2, 1))
    first_a = [p for p in a["predictions"] if p["fit_before_date"] == "2018-01-29"]
    first_b = [p for p in b["predictions"] if p["fit_before_date"] == "2018-01-29"]
    assert len(first_a) == len(first_b) == 4
    for x, y in zip(first_a, first_b):
        assert x["probabilities"] == pytest.approx(y["probabilities"], abs=1e-12)
    assert a["fits"][0] == b["fits"][0]
    assert a["fits"][1]["coefficients"] != b["fits"][1]["coefficients"]


def test_unknown_teams_listed_and_market_comparison_is_paired():
    records = history()
    records[32] = replace(records[32], home="UNKNOWN")
    records[33] = replace(records[33], markets={})
    result = weekly_walk_forward(records, evaluation_start=date(2018, 2, 1))
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "unseen_team_no_estimated_prior"
    assert result["skipped"][0]["unknown_teams"] == ["UNKNOWN"]
    metrics = evaluate(result["predictions"])
    assert metrics["model_all_supported"]["n"] == 17
    paired = metrics["market_comparisons"][MARKET]
    assert paired["n_matches"] == paired["market"]["n"] == paired["model_on_identical_matches"]["n"] == 16
    assert records[33].match_id not in paired["match_ids"]
    assert records[32].match_id not in paired["match_ids"]
    assert {row["match_id"] for row in result["predictions"] if MARKET in row["market_probabilities"]} == set(paired["match_ids"])


def test_full_score_distribution_and_eight_total_buckets_are_normalized():
    forecast = full_time_forecast(4.5, 3.5)
    assert sum(forecast["one_x_two"].values()) == pytest.approx(1)
    assert list(forecast["total_goals"]) == ["0", "1", "2", "3", "4", "5", "6", "7+"]
    assert sum(forecast["total_goals"].values()) == pytest.approx(1)
    assert forecast["total_goals"]["7+"] > .5
    row = {"actual_score": [5, 3], "actual_outcome": 0,
        "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
        "market_probabilities": {}}
    total = evaluate([row])["total_goals"]
    assert total["n"] == 1
    assert total["log_loss"] == pytest.approx(-math.log(forecast["total_goals"]["7+"]))
    assert 0 <= total["rps"] <= 1
    assert len(total["calibration"]) == 8


def write_source(path, rows):
    columns = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_csv_reuses_parser_without_inventing_avg_odds_or_times(tmp_path):
    path = tmp_path / "E0.csv"
    rows = [dict(Div="E0", Date="01/02/2018", HomeTeam="A", AwayTeam="B",
        FTHG="2", FTAG="1", FTR="H", B365H="2", B365D="3.5", B365A="4"),
        dict(Div="E0", Date="02/02/2018", HomeTeam="B", AwayTeam="A",
        FTHG="0", FTAG="0", FTR="D", B365H="2", B365D="", B365A="4")]
    write_source(path, rows)
    records, source = load_season_csv(path, "2017/18")
    assert records[0].markets == {MARKET: (2., 3.5, 4.)}
    assert records[1].markets == {}
    assert source["market_rows"] == 1
    assert source["time_semantics"]["individual_odds_observed_at"] is None
    assert source["source_url"].endswith("1718/E0.csv")
    assert not source["duplicate_keys"] and not source["rejected_rows"]
    assert all(issue["field"] in {MARKET, "half_time"} for issue in source["field_issues"])


def test_csv_bad_score_and_duplicate_identity_remain_explicit(tmp_path):
    path = tmp_path / "E0.csv"
    good = dict(Div="E0", Date="01/02/2018", HomeTeam="A", AwayTeam="B",
        FTHG="2", FTAG="1", FTR="H", B365H="2", B365D="3.5", B365A="4")
    write_source(path, [good, good, {**good, "HomeTeam": "C", "FTHG": "-1"}])
    records, report = load_season_csv(path, "2017/18")
    assert len(records) == 2
    assert len(report["duplicate_keys"]) == 1
    assert len(report["rejected_rows"]) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        weekly_walk_forward(records, evaluation_start=date(2018, 2, 1))


def test_independent_season_requirement_and_no_history_are_explicit():
    records = history()
    with pytest.raises(ValueError, match="independently"):
        weekly_walk_forward(records + [replace(records[0], season="2015/16", match_id="other")],
                            evaluation_start=date(2018, 2, 1))
    with pytest.raises(ValueError, match="calendar date"):
        monday(datetime(2018, 2, 1))
    empty = weekly_walk_forward([], evaluation_start=date(2018, 2, 1))
    assert empty["fits"] == []
    assert evaluate([])["model_all_supported"] is None
    no_history = weekly_walk_forward(records[:3], evaluation_start=date(2018, 1, 1))
    assert len(no_history["skipped"]) == 3
    assert {r["reason"] for r in no_history["skipped"]} == {"no_earlier_history"}


def test_excluded_results_stay_out_of_training_and_target_scores():
    records = history()
    records[2] = replace(records[2], goal_fit_exclusion="not_on_field_result")
    records[31] = replace(records[31], goal_fit_exclusion="not_on_field_result")
    result = weekly_walk_forward(records, evaluation_start=date(2018, 2, 1))
    assert result["fits"][0]["n_matches"] == 27
    assert len(result["predictions"]) == 17
    assert result["skipped"][0]["reason"] == "not_on_field_result"
    assert all(records[2].match_id not in f["reference_match_ids"] for f in result["fits"])
    assert all(records[31].match_id not in f["reference_match_ids"] for f in result["fits"])


def test_small_two_season_run_writes_readable_results_and_parameters(tmp_path, monkeypatch):
    specs = []
    for year, season in ((2016, "2015/16"), (2018, "2017/18")):
        path = tmp_path / f"synthetic_{year}.csv"
        rows = []
        for day, home, away, score in ((f"01/01/{year}", "A", "B", (2, 1)),
            (f"02/01/{year}", "C", "D", (1, 0)), (f"01/02/{year}", "A", "C", (1, 1))):
            h, a = score
            rows.append(dict(Div="E0", Date=day, HomeTeam=home, AwayTeam=away,
                FTHG=str(h), FTAG=str(a), FTR="H" if h > a else "D",
                B365H="2", B365D="3.5", B365A="4"))
        write_source(path, rows)
        specs.append((season, path.name, date(year, 2, 1)))
    monkeypatch.setattr(experiment_module, "SEASON_SPECS", tuple(specs))
    output = tmp_path / "synthetic_outputs"
    result = experiment_module.run(tmp_path, output)
    assert result["status"]["current_predictions"] is False
    assert result["status"]["player_validation"] is False
    assert result["protocol"]["market_blending"] is False
    assert result["protocol"]["hyperparameter_search"] is False
    assert len(list(output.iterdir())) == 6
    parameters = json.loads((output / "逐周完整参数.json").read_text(encoding="utf-8"))
    assert {p["season"] for p in parameters} == {"2015/16", "2017/18"}
    assert all(p["n_matches"] == len(p["reference_match_ids"]) == 2 for p in parameters)
    predictions = json.loads((output / "逐场球队验证.json").read_text(encoding="utf-8"))
    assert len(predictions) == 2
    assert all(p["market_quote_observed_at"] is None for p in predictions)
    with (output / "逐场球队验证.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2
    assert "p_goals_7_plus" in table[0]
    assert "b365_preclosing_home" in table[0]
    report = (output / "球队层验证说明.md").read_text(encoding="utf-8")
    assert "不是当前比赛预测" in report
    assert "不是球员模型验证" in report
    assert "同一批比赛" in report
    assert "没有调参" in report
    assert "7+" in report
