"""Same-match comparisons and real small chronological runs, not live evidence."""

from copy import deepcopy
import csv
from dataclasses import replace
from datetime import date
import json

import numpy as np
import pytest

from football_model import native_experiment
from football_model.date_model import DateTeamGoalModel
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.schedule_model import DateActivityIndex, DateScheduleGoalModel
import football_model.schedule_experiment as experiment
from test_schedule_research import record


def tracks():
    days = ("2025-08-05", "2025-08-12", "2025-08-12", "2025-09-02", "2025-10-01", "2025-11-15")
    source = [record(f"m{i}", day, score=(i % 4, (i+2) % 3)) for i, day in enumerate(days)]
    source += [record("old", "2024-08-01")]
    index = DateActivityIndex(source)
    output = {arm: [] for arm in experiment.ARMS}
    for arm in output:
        for i, raw in enumerate(source[:-1]):
            rates = [1.2+i*.1+(arm == "schedule")*.05, .9+i*.05]
            forecast = summarize(score_grid(*rates))
            prices = {"avg_preclosing": [2., 4., 4.]}
            market = {"avg_preclosing": [.5, .25, .25]}
            output[arm].append({"match_id": raw.match_id, "competition": "E0", "date": raw.match_date.isoformat(),
                "home": "A", "away": "B", "actual_score": list(raw.full_time),
                "actual_outcome": 0 if raw.full_time[0] > raw.full_time[1] else 2 if raw.full_time[0] < raw.full_time[1] else 1,
                "fit_before_date": "2025-08-01", "training_max_date": "2025-07-31", "training_matches": 100,
                "rates": rates, "forecast": forecast, "probabilities": list(forecast["one_x_two"].values()),
                "market_prices": prices, "market_probabilities": market,
                "source_url": raw.source_url, "context_before_date": raw.match_date.isoformat(),
                "schedule_context": index.context(raw.home, raw.away, raw.competition, raw.match_date)})
    return output


def test_same_match_complete_scoring_and_shared_calendar_matrix():
    raw = tracks()
    before = deepcopy(raw)
    result, entries, stats = experiment.analyze_period(raw, repeats=31)
    assert raw == before and result["n"] == 6
    losses, features = experiment.feature_vectors(entries, stats)
    assert len(features) == 32
    for domain in experiment.DOMAINS:
        assert result["mean_differences"][domain] == pytest.approx(np.mean(losses["schedule"][domain]-losses["original"][domain]))
    for weeks in experiment.BLOCKS:
        expected = moving_block_means([r["date"] for r in entries["original"]], np.column_stack(list(features.values())),
            block_weeks=weeks, repeats=31, seed=experiment.SEED)
        for i, name in enumerate(features):
            assert result["bootstrap"][str(weeks)]["features"][name]["interval_95"] == pytest.approx(expected["interval_95"][i])
    for group in ("by_month", "by_league", "by_observed_count_comparison"):
        assert sum(g["n"] for g in result[group].values()) == 6


@pytest.mark.parametrize("issue", ["missing_arm", "lost_match", "bad_rates", "bad_forecast", "cutoff", "context", "date_context"])
def test_mismatched_information_or_distribution_cannot_silently_shrink_cohort(issue):
    raw = tracks()
    if issue == "missing_arm":
        del raw["schedule"]
    elif issue == "lost_match":
        raw["schedule"].pop()
    elif issue == "bad_rates":
        raw["schedule"][0]["rates"][0] = True
    elif issue == "bad_forecast":
        raw["schedule"][0]["forecast"]["total_goals"]["4+"] += .1
    elif issue == "cutoff":
        raw["schedule"][0]["training_matches"] += 1
    elif issue == "context":
        raw["schedule"][0]["schedule_context"]["home"]["source_fixture_count_prev14d"] += 1
    else:
        raw["schedule"][0]["context_before_date"] = "2025-08-04"
    with pytest.raises(ValueError):
        experiment.analyze_period(raw, repeats=13)


def test_market_missingness_does_not_change_model_cohort_and_csv_roundtrips(tmp_path):
    raw = tracks()
    for arm in raw:
        raw[arm][0]["market_prices"] = {}
        raw[arm][0]["market_probabilities"] = {}
    result, entries, stats = experiment.analyze_period(raw, repeats=17)
    assert result["n"] == 6
    assert result["markets"]["avg_preclosing"]["n"] == 5
    assert result["markets"]["avg_closing"]["n"] == 0
    path = tmp_path/"paired.csv"
    experiment.write_csv(path, entries, stats)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and len(rows[0]) == 79
    for i, row in enumerate(rows):
        for arm in experiment.ARMS:
            assert float(row[f"{arm}_joint_score_nll"]) == pytest.approx(stats[arm][i]["joint_score_nll"])


def environment(tmp_path, monkeypatch):
    source = []
    for comp in experiment.LEAGUES:
        for season, dates in (("2023/24", ("2024-06-01", "2024-06-07", "2024-06-13")),
                              ("2024/25", ("2024-08-05", "2024-08-12", "2024-10-01", "2025-05-01")),
                              ("2025/26", ("2025-08-05", "2025-08-12", "2025-10-01", "2026-05-01"))):
            for i, day in enumerate(dates):
                row = record(f"{comp}-{day}", day, "A" if i % 2 else "B", "B" if i % 2 else "A", (i % 3, (i+1) % 3))
                source.append(replace(row, competition=comp, season=season,
                    home=f"football-data:{comp}:{row.home_label}", away=f"football-data:{comp}:{row.away_label}"))
    directories = {s: tmp_path/s.replace("/", "-") for s in ("2022/23", "2023/24", "2024/25", "2025/26")}
    parent = tmp_path/"parent"
    for period, (start, end) in experiment.PERIODS.items():
        folder = parent/period
        folder.mkdir(parents=True)
        for comp in experiment.LEAGUES:
            rows = [r for r in source if r.competition == comp and int(r.season[:4]) <= start.year]
            old = native_experiment.run_config(rows, native_experiment.ORIGINAL, start, end)
            (folder/f"{comp}_days_28.json").write_text(json.dumps(old), encoding="utf-8")
    events = []
    def load(selected):
        events.append(tuple(selected))
        return [r for r in source if r.season in selected], {}
    monkeypatch.setattr(experiment, "season_directories", lambda root: directories)
    monkeypatch.setattr(experiment, "load_seasons", load)
    return parent, events


def test_full_runner_actually_fits_then_saved_replay_never_refits(tmp_path, monkeypatch):
    parent, events = environment(tmp_path, monkeypatch)
    source_before = {str(p): p.read_text() for p in parent.rglob("*.json")}
    output = experiment.run(tmp_path, parent, repeats=13)
    result = json.loads((output/experiment.RESULT_FILE).read_text(encoding="utf-8"))
    assert result["counts"]["new_schedule_fits"] == 30
    assert result["counts"]["new_schedule_predictions"] == 40
    assert result["counts"]["new_fits_this_call"] == 30
    assert result["model_adopted"] is False and result["production_api_changed"] is False
    assert "2025/26" not in events[0] and "2025/26" in events[1]
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing complete archives must never refit")
    monkeypatch.setattr(DateScheduleGoalModel, "fit", forbidden)
    monkeypatch.setattr(DateTeamGoalModel, "fit", forbidden)
    experiment.run(tmp_path, parent, output=output, repeats=13)
    replay = json.loads((output/experiment.RESULT_FILE).read_text(encoding="utf-8"))
    assert replay["periods"] == result["periods"]
    assert replay["counts"]["new_fits_this_call"] == 0
    assert replay["counts"]["complete_schedule_trials_reloaded_this_call"] == 10
    assert {str(p): p.read_text() for p in parent.rglob("*.json")} == source_before
    assert "不是新增盲测" in (output/experiment.REPORT_FILE).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="settings differ"):
        experiment.run(tmp_path, parent, output=output, repeats=15)
