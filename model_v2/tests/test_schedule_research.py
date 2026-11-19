"""Actual small-sample fits and source-bound replay of schedule research trials."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from football_model.date_model import DateTeamGoalModel
from football_model.football_data import HistoricalMatch
from football_model import native_experiment
import football_model.schedule_research as research


START, END = date(2024, 8, 1), date(2024, 11, 1)


def record(key, day, home="A", away="B", score=(2, 1), *, exclusion=None, markets=None):
    return HistoricalMatch(
        key, "E0", "2024/25", date.fromisoformat(day), None,
        "football-data:E0:" + home, "football-data:E0:" + away,
        home, away, score, None,
        {"avg_preclosing": (2.0, 3.5, 4.0), "avg_closing": (2.1, 3.4, 3.8)}
        if markets is None else markets,
        "G:/synthetic-not-a-source/E0.csv", "https://example.invalid/E0.csv", exclusion,
    )


@pytest.fixture(scope="module")
def sources():
    history = [record("history-" + str(i), (date(2024, 6, 1) + timedelta(days=i * 4)).isoformat(),
                      "A" if i % 2 else "B", "B" if i % 2 else "A",
                      ((i * 2) % 4, (i + 1) % 3)) for i in range(12)]
    return history + [
        record("anchor-day", "2024-08-01", score=(1, 2)),
        record("same-day-peer", "2024-08-01", "B", "A", (0, 1)),
        record("day-two", "2024-08-02", score=(2, 2)),
        record("unknown-C", "2024-08-03", "C", "B", (1, 1)),
        record("excluded", "2024-08-09", exclusion="administratively_awarded_result"),
        record("new-known-next-block", "2024-08-10", "A", "C", (3, 1)),
        record("first-28-boundary", "2024-08-29", score=(0, 1), markets={}),
        record("C-known-next-block", "2024-08-30", "A", "C", (2, 1)),
        record("after-empty-weeks", "2024-10-10", score=(4, 2)),
        record("exclusive-end", "2024-11-01", score=(1, 0)),
        record("far-future", "2026-01-05", score=(5, 4)),
    ]


@pytest.fixture(scope="module")
def trial(sources):
    return research.run_trial(sources, START, END)


def check(trial, sources):
    return research.validate_trial(trial, sources, START, END, "E0")


def test_actual_full_trial_replay_is_nonmutating_and_keeps_original_cohort(sources, trial):
    before_source, before_trial = deepcopy(sources), deepcopy(trial)
    assert check(trial, sources) is trial
    assert sources == before_source and trial == before_trial
    assert {row["match_id"] for row in trial["predictions"]} == native_experiment.expected_target_ids(
        sources, START, END, 28)
    assert trial["failures"] == []
    assert trial["config"] == research.CONFIG
    assert trial["metrics"]["model"]["n"] == len(trial["predictions"])


def test_every_nonempty_block_is_really_refitted_from_full_source_records(sources, monkeypatch):
    fit_calls = []
    original_fit = research.DateScheduleGoalModel.fit

    def observed(self, records, cutoff):
        fit_calls.append((list(records), cutoff, self.regularization, self.half_life_days))
        return original_fit(self, records, cutoff)

    monkeypatch.setattr(research.DateScheduleGoalModel, "fit", observed)
    output = research.run_trial(list(reversed(sources)), START, END)
    assert len(fit_calls) == len(output["fits"]) == 3
    assert [call[1].isoformat() for call in fit_calls] == [row["before_date"] for row in output["fits"]]
    assert all(supplied == sorted(sources, key=lambda row: (row.match_date, row.match_id))
               and alpha == .01 and half_life == 180
               for supplied, _, alpha, half_life in fit_calls)
    assert all({research.SELF_COUNT_FEATURE, research.OPPONENT_COUNT_FEATURE}
               <= set(fit["coefficients"]) for fit in output["fits"])
    original = DateTeamGoalModel(.01, 180).fit(sources, START).parameters()
    assert any(original["coefficients"][key] != output["fits"][0]["coefficients"][key]
               for key in original["coefficients"])


def test_within_block_date_counts_update_while_parameters_remain_fixed(trial):
    rows = {row["match_id"]: row for row in trial["predictions"]}
    first, peer, second = (rows[key] for key in ("anchor-day", "same-day-peer", "day-two"))
    assert {row["fit_before_date"] for row in (first, peer, second)} == {START.isoformat()}
    assert first["schedule_context"]["home"]["source_fixture_count_prev14d"] == 0
    assert peer["schedule_context"]["home"]["source_fixture_count_prev14d"] == 0
    assert second["schedule_context"]["home"]["source_fixture_count_prev14d"] == 2
    assert second["context_before_date"] == second["date"] == "2024-08-02"
    assert set(second["schedule_context"]["home"]["previous_14d_ids"]) == {"anchor-day", "same-day-peer"}
    assert first["rates"] != second["rates"]


def test_same_day_future_goals_cannot_change_earlier_fit_or_initial_block_forecasts(sources, trial):
    changed = [replace(row, full_time=(19, 18)) if row.match_date >= START else row for row in sources]
    other = research.run_trial(changed, START, END)
    assert other["fits"][0] == trial["fits"][0]
    def initial(run):
        return {row["match_id"]: (row["rates"], row["forecast"], row["schedule_context"])
                for row in run["predictions"] if row["fit_before_date"] == START.isoformat()}
    assert initial(trial) == initial(other)


def test_odds_and_excluded_goal_labels_never_affect_any_trained_parameters(sources, trial):
    changed = [replace(row, full_time=(17, 16) if row.goal_fit_exclusion else row.full_time,
                       markets={"avg_preclosing": (5.0, 5.0, 5.0)}) for row in sources]
    other = research.run_trial(changed, START, END)
    assert other["fits"] == trial["fits"]
    assert [row["rates"] for row in other["predictions"]] == [row["rates"] for row in trial["predictions"]]


def test_unknown_team_whole_match_skips_and_excluded_dates_are_kept_without_goals(trial):
    skipped = {row["match_id"]: row["reason"] for row in trial["skipped"]}
    assert skipped == {"unknown-C": "unseen_team_no_estimated_prior",
                       "excluded": "administratively_awarded_result",
                       "new-known-next-block": "unseen_team_no_estimated_prior"}
    rows = {row["match_id"]: row for row in trial["predictions"]}
    assert "C-known-next-block" in rows
    second_fit = trial["fits"][1]
    assert "unknown-C" in second_fit["reference_match_ids"]
    assert "new-known-next-block" in second_fit["reference_match_ids"]
    assert all("excluded" not in fit["reference_match_ids"] for fit in trial["fits"])
    # August 9's excluded source row is within the preceding 14d of an
    # independent query even though its goals never enter a fitted target.
    index = research.DateActivityIndex([record("special", "2024-08-09", exclusion="special")])
    context = index.context("football-data:E0:A", "football-data:E0:B", "E0", date(2024, 8, 10))
    assert context["home"]["source_fixture_count_prev14d"] == 1
    assert context["home"]["special_previous_14d_ids"] == ["special"]


def test_empty_blocks_do_not_fit_or_move_the_august_anchor(sources, trial, monkeypatch):
    assert [fit["before_date"] for fit in trial["fits"]] == ["2024-08-01", "2024-08-29", "2024-09-26"]
    assert all((date.fromisoformat(fit["before_date"]) - START).days % 28 == 0 for fit in trial["fits"])
    monkeypatch.setattr(research.DateScheduleGoalModel, "fit", lambda *a, **k: pytest.fail("No eligible block"))
    empty = research.run_trial([row for row in sources if row.match_date < START], START, END)
    assert not empty["fits"] and not empty["predictions"] and not empty["skipped"]
    unknown = research.run_trial([record("unknown", "2024-08-03")], START, END)
    assert not unknown["fits"] and not unknown["predictions"]
    assert unknown["skipped"] == [{"match_id": "unknown", "reason": "unseen_team_no_estimated_prior"}]


def test_load_and_validate_replay_perform_zero_fits_and_preserve_archive(tmp_path, sources, trial, monkeypatch):
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(trial, allow_nan=False), encoding="utf-8-sig")
    before = path.read_bytes()
    monkeypatch.setattr(research.DateScheduleGoalModel, "fit", lambda *a, **k: pytest.fail("No fitting in replay"))
    assert research.load_trial(path, sources, START, END, "E0") == json.loads(before.decode("utf-8-sig"))
    assert check(trial, sources) is trial and path.read_bytes() == before


@pytest.mark.parametrize("text", ["{\"x\":NaN}", "{\"x\":Infinity}", "{\"config\":{},\"config\":{}}", "null", "[]"])
def test_json_loader_rejects_nonfinite_duplicate_and_nonmapping_data(tmp_path, sources, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        research.load_trial(path, sources, START, END)


@pytest.mark.parametrize("error", [ConvergenceWarning("optimizer failed"), ValueError("fit failed"),
                                  FloatingPointError("numeric failure")])
def test_fit_failures_abort_explicitly_without_fallback(sources, monkeypatch, error):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        if isinstance(error, ConvergenceWarning):
            warnings.warn(error)
        else:
            raise error
    monkeypatch.setattr(research.DateScheduleGoalModel, "fit", failed)
    with pytest.raises(ValueError, match="no fallback"):
        research.run_trial(sources, START, END)
    assert calls == [1]


def test_prediction_failure_aborts_without_alternative_model(sources, monkeypatch):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ValueError("Bad rate")
    monkeypatch.setattr(research.DateScheduleGoalModel, "predict", failed)
    with pytest.raises(ValueError, match="prediction failed.*no fallback"):
        research.run_trial(sources, START, END)
    assert calls == [1]


@pytest.mark.parametrize("start,end", [("2024-08-01", END), (datetime(2024, 8, 1), END),
                                        (date(2024, 8, 2), END), (END, START), (START, START)])
def test_rejects_invalid_or_shifted_date_anchor(sources, trial, start, end):
    with pytest.raises(ValueError):
        research.run_trial(sources, start, end)
    with pytest.raises(ValueError):
        research.validate_trial(trial, sources, start, end)


@pytest.mark.parametrize("damage", ["empty", "duplicate", "mixed", "score", "date", "market"])
def test_rejects_invalid_source_records_without_training(sources, monkeypatch, damage):
    damaged = list(sources)
    if damage == "empty":
        damaged = []
    elif damage == "duplicate":
        damaged.append(damaged[0])
    elif damage == "mixed":
        damaged[0] = replace(damaged[0], competition="D1")
    elif damage == "score":
        damaged[0] = replace(damaged[0], full_time=(True, 0))
    elif damage == "date":
        damaged[0] = replace(damaged[0], match_date="2024-06-01")
    else:
        damaged[0] = replace(damaged[0], markets={"bad": (2, 3, 4)})
    monkeypatch.setattr(research.DateScheduleGoalModel, "fit", lambda *a, **k: pytest.fail("Invalid source"))
    with pytest.raises(ValueError):
        research.run_trial(damaged, START, END)


@pytest.mark.parametrize("damage", ["missing-prediction", "duplicate-prediction", "unexpected-prediction",
                                    "missing-fit", "duplicate-fit", "empty-block-fit", "missing-skip",
                                    "duplicate-skip", "skip-reason", "failed", "config", "interval"])
def test_complete_coverage_fit_and_skip_records_cannot_be_silently_reduced(sources, trial, damage):
    run = deepcopy(trial)
    if damage == "missing-prediction":
        run["predictions"].pop()
    elif damage == "duplicate-prediction":
        run["predictions"].append(deepcopy(run["predictions"][0]))
    elif damage == "unexpected-prediction":
        run["predictions"][0]["match_id"] = "far-future"
    elif damage == "missing-fit":
        run["fits"].pop()
    elif damage == "duplicate-fit":
        run["fits"].append(deepcopy(run["fits"][0]))
    elif damage == "empty-block-fit":
        run["fits"].append({**deepcopy(run["fits"][-1]), "before_date": "2024-10-24"})
    elif damage == "missing-skip":
        run["skipped"].pop()
    elif damage == "duplicate-skip":
        run["skipped"].append(deepcopy(run["skipped"][0]))
    elif damage == "skip-reason":
        run["skipped"][0]["reason"] = "assumed_zero"
    elif damage == "failed":
        run["failures"] = [{"stage": "fit"}]
    elif damage == "config":
        run["config"]["window_days"] = 7
    else:
        run["update_interval_days"] = 28.0
    with pytest.raises(ValueError):
        check(run, sources)


@pytest.mark.parametrize("field,value", [
    ("model_type", "original"), ("regularization", .02), ("half_life_days", 90), ("rho", .01),
    ("n_matches", True), ("n_matches", 12.0), ("optimizer_iterations", 1000),
    ("optimizer_iterations", True), ("intercept", float("nan")), ("intercept", "0"),
    ("basis", "player_conditional"), ("training_max_date", "2024-08-01"),
    ("reference_match_ids", ["anchor-day"]), ("competitions", ["D1"]),
    ("availability_convention", "same_day"), ("not_for", "live use"),
])
def test_saved_fit_metadata_and_numeric_settings_match_source_and_model(sources, trial, field, value):
    run = deepcopy(trial)
    run["fits"][0][field] = value
    with pytest.raises(ValueError):
        check(run, sources)


@pytest.mark.parametrize("damage", ["missing", "extra", "changed", "nan", "bool", "string", "definition"])
def test_both_new_and_original_coefficients_and_feature_definition_are_replayed(sources, trial, damage):
    run = deepcopy(trial)
    fit = run["fits"][0]
    key = research.SELF_COUNT_FEATURE
    if damage == "missing":
        del fit["coefficients"][key]
    elif damage == "extra":
        fit["coefficients"]["unused"] = 0.0
    elif damage == "definition":
        fit["schedule_feature_definition"]["same_date_records_included"] = 0
    else:
        # Attack has a nonzero first prediction exposure even when count=0.
        key = next(key for key in fit["coefficients"] if key.startswith("attack:"))
        fit["coefficients"][key] = {"changed": fit["coefficients"][key] + .2,
                                     "nan": float("nan"), "bool": True, "string": "0"}[damage]
    with pytest.raises(ValueError):
        check(run, sources)


@pytest.mark.parametrize("field,value", [
    ("source_fixture_count_prev14d", 99), ("source_fixture_count_prev14d", 0.0),
    ("source_fixture_count_prev14d", False), ("previous_date", "2024-08-01"),
    ("gap_days", 0), ("previous_14d_ids", ["anchor-day"]),
    ("special_previous_14d_ids", ["excluded"]), ("window_left_censored", 0),
    ("has_previous_source_record", 0), ("team", "football-data:E0:other"),
])
def test_full_prediction_context_is_checked_against_source_not_just_feature_values(sources, trial, field, value):
    run = deepcopy(trial)
    run["predictions"][0]["schedule_context"]["home"][field] = value
    with pytest.raises(ValueError):
        check(run, sources)


@pytest.mark.parametrize("field,value", [
    ("context_before_date", "2024-07-31"), ("fit_before_date", "2024-07-31"),
    ("date", "2024-08-02"), ("source_url", "https://example.invalid/wrong.csv"),
    ("training_matches", True), ("training_matches", 12.0),
    ("rates", [1, 1]), ("rates", [True, 1]), ("actual_score", [19, 18]),
])
def test_rates_identity_dates_labels_and_training_metadata_are_source_bound(sources, trial, field, value):
    run = deepcopy(trial)
    run["predictions"][0][field] = value
    with pytest.raises(ValueError):
        check(run, sources)


@pytest.mark.parametrize("path", [
    ("one_x_two", "home"), ("expected_goals", "away"), ("total_goals", "4+"),
    ("away_minus_two_jincai", "win"), ("away_minus_two_asian", "push"),
    ("over_2_5",), ("both_score",), ("fair_odds_unbuffered", "away_win"),
])
def test_complete_forecast_is_regenerated_from_same_score_grid(sources, trial, path):
    run = deepcopy(trial)
    value = run["predictions"][0]["forecast"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] += .01
    with pytest.raises(ValueError, match="forecast"):
        check(run, sources)


def test_changed_odds_even_with_correct_devig_do_not_match_original_source(sources, trial):
    run = deepcopy(trial)
    row = run["predictions"][0]
    row["market_prices"]["avg_preclosing"] = [3.0, 4.0, 5.0]
    row["market_probabilities"]["avg_preclosing"] = list(map(float, research.de_vig([3, 4, 5])))
    with pytest.raises(ValueError, match="market_prices"):
        check(run, sources)


@pytest.mark.parametrize("field", ["rates", "schedule_context", "context_before_date", "forecast"])
def test_missing_new_prediction_fields_are_explicit_value_errors(sources, trial, field):
    run = deepcopy(trial)
    del run["predictions"][0][field]
    with pytest.raises(ValueError):
        check(run, sources)


def test_future_and_exclusive_end_label_changes_are_irrelevant_to_replay(sources, trial):
    changed = [replace(row, full_time=(40, 39)) if row.match_date >= END else row for row in sources]
    assert check(trial, changed) is trial
    all_training_ids = {key for fit in trial["fits"] for key in fit["reference_match_ids"]}
    assert not {"far-future", "exclusive-end"} & all_training_ids
