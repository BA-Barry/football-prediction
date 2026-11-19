"""Strict chronology and source/replay checks for refresh-frequency research."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import math
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from football_model import native_experiment
from football_model.football_data import HistoricalMatch
import football_model.refresh_research as research


START, END = date(2024, 8, 1), date(2024, 11, 1)


def record(key, day, home="A", away="B", score=(2, 1), *, exclusion=None, markets=None):
    return HistoricalMatch(
        key, "E0", "2024/25", date.fromisoformat(day), None,
        "football-data:E0:" + home, "football-data:E0:" + away,
        home, away, score, None,
        {"avg_preclosing": (2.0, 3.5, 4.0), "avg_closing": (2.1, 3.4, 3.8)}
        if markets is None else markets,
        "synthetic/not-a-source/E0.csv", "https://example.invalid/E0.csv", exclusion,
    )


@pytest.fixture(scope="module")
def sources():
    history = [record("history-" + str(i), (date(2024, 6, 1) + timedelta(days=i * 4)).isoformat(),
                      "A" if i % 2 else "B", "B" if i % 2 else "A",
                      ((i * 2) % 4, (i + 1) % 3)) for i in range(12)]
    return history + [
        record("anchor-day", "2024-08-01", score=(1, 2)),
        record("day-two", "2024-08-02", score=(2, 2)),
        record("unknown-C", "2024-08-03", "C", "B", (1, 1)),
        record("excluded", "2024-08-09", exclusion="administratively_awarded_result"),
        record("new-known-weekly", "2024-08-10", "A", "C", (3, 1)),
        record("first-28-boundary", "2024-08-29", score=(0, 1), markets={}),
        record("after-empty-blocks", "2024-10-10", score=(4, 2)),
        record("exclusive-end", "2024-11-01", score=(1, 0)),
        record("far-future", "2026-01-05", score=(5, 4)),
    ]


@pytest.fixture(scope="module")
def trials(sources):
    return {interval: native_experiment.run_config(sources, dict(native_experiment.ORIGINAL),
                                                   START, END, interval_days=interval)
            for interval in research.INTERVALS}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def check(trial, sources, interval=28):
    return research.validate_trial(trial, sources, START, END, "E0", interval)


def test_predeclared_intervals_and_periods():
    assert research.INTERVALS == (7, 14, 28)
    assert research.PERIODS == {
        "2024-25": (native_experiment.VALID_START, native_experiment.VALID_END),
        "2025-26": (native_experiment.TEST_START, native_experiment.TEST_END),
    }


@pytest.mark.parametrize("interval", research.INTERVALS)
def test_real_small_sample_full_trial_roundtrip_is_nonmutating(sources, trials, interval):
    trial = deepcopy(trials[interval])
    before_trial, before_source = deepcopy(trial), deepcopy(sources)
    assert check(trial, sources, interval) is trial
    assert trial == before_trial
    assert sources == before_source
    assert not trial["failures"]
    assert {r["match_id"] for r in trial["predictions"]} == native_experiment.expected_target_ids(
        sources, START, END, interval)


def test_loaded_28_reference_performs_no_fit_or_pickle_load(tmp_path, sources, trials, monkeypatch):
    path = write_json(tmp_path / "complete.json", trials[28])

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference archives must not invoke fitting")

    monkeypatch.setattr(native_experiment, "make_model", forbidden)
    monkeypatch.setattr(native_experiment, "run_config", forbidden)
    loaded = research.load_reference(path, sources, START, END, "E0")
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    assert path.read_bytes().startswith(b"{")


def test_loader_accepts_utf8_bom(tmp_path, sources, trials):
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(trials[28]), encoding="utf-8-sig")
    assert research.load_reference(path, sources, START, END, "E0")["config"] == native_experiment.ORIGINAL


@pytest.mark.parametrize("text", ["{\"config\":{},\"config\":{}}", "{\"x\":NaN}", "{\"x\":Infinity}", "[]", "null"])
def test_loader_refuses_duplicate_keys_nonfinite_and_nonmapping_json(tmp_path, sources, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        research.load_reference(path, sources, START, END, "E0")


@pytest.mark.parametrize("interval", [7, 14])
def test_new_frequency_calls_existing_original_model_without_parameter_reuse(sources, trials, monkeypatch, interval):
    calls = []

    def observed(records, config, start, end, interval_days):
        calls.append((records, deepcopy(config), start, end, interval_days))
        return deepcopy(trials[interval])

    monkeypatch.setattr(native_experiment, "run_config", observed)
    output = research.run_frequency(list(reversed(sources)), START, END, interval)
    assert len(calls) == 1
    supplied, config, start, end, used_interval = calls[0]
    assert config == native_experiment.ORIGINAL
    assert config["half_life_days"] == 180
    assert (start, end, used_interval) == (START, END, interval)
    assert supplied == sorted(sources, key=lambda r: (r.match_date, r.match_id))
    assert output == trials[interval]


def test_new_frequency_does_not_refit_28_reference(sources, monkeypatch):
    monkeypatch.setattr(native_experiment, "run_config", lambda *args, **kwargs: pytest.fail("No refit"))
    with pytest.raises(ValueError, match="saved 28-day reference"):
        research.run_frequency(sources, START, END, 28)


@pytest.mark.parametrize("interval", [7, 14])
def test_new_frequency_real_fit_has_complete_source_bound_results(sources, interval):
    output = research.run_frequency(sources, START, END, interval)
    assert not output["failures"]
    assert len(output["fits"]) >= 3
    assert all(fit["regularization"] == 0.01 and fit["half_life_days"] == 180 and fit["rho"] == 0
               for fit in output["fits"])
    assert all(p["date"] < END.isoformat() for p in output["predictions"])


def test_new_frequency_optimizer_warning_is_failure_not_fallback(sources, monkeypatch):
    calls = []

    def unfit(*args, **kwargs):
        calls.append(1)
        warnings.warn("Deliberate line-search failure", ConvergenceWarning)
        pytest.fail("Convergence warnings must abort")

    monkeypatch.setattr(native_experiment, "run_config", unfit)
    with pytest.raises(ValueError, match="did not converge"):
        research.run_frequency(sources, START, END, 7)
    assert calls == [1]


def test_failed_run_is_not_replaced_with_another_interval(sources, trials, monkeypatch):
    trial = deepcopy(trials[7])
    trial["failures"] = [{"cutoff": START.isoformat(), "stage": "fit", "error": "failed"}]
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["interval_days"])
        return trial

    monkeypatch.setattr(native_experiment, "run_config", failed)
    with pytest.raises(ValueError, match="failures"):
        research.run_frequency(sources, START, END, 7)
    assert calls == [7]


@pytest.mark.parametrize("start,end,interval", [
    ("2024-08-01", END, 7), (datetime(2024, 8, 1), END, 7),
    (date(2024, 8, 2), END, 7), (date(2024, 7, 1), END, 14),
    (START, START, 7), (END, START, 7), (START, "2024-11-01", 14),
    (START, END, True), (START, END, 7.0), (START, END, 0), (START, END, 21),
])
def test_declared_settings_refuse_shifted_anchor_bad_dates_and_other_intervals(sources, trials, start, end, interval):
    with pytest.raises(ValueError):
        research.validate_trial(trials[28], sources, start, end, "E0", interval)
    with pytest.raises(ValueError):
        research.run_frequency(sources, start, end, interval)


@pytest.mark.parametrize("key,value", [
    ("config", {"id": "poisson_h180", "family": "poisson", "half_life_days": 180}),
    ("start", "2024-08-02"), ("end_exclusive", "2024-10-31"),
    ("update_interval_days", 7), ("update_interval_days", 28.0),
    ("failures", [{"stage": "predict", "error": "failed"}]),
    ("predictions", {}), ("fits", None), ("skipped", {}),
])
def test_trial_settings_failures_and_shape_must_match_declared_arm(sources, trials, key, value):
    trial = deepcopy(trials[28])
    trial[key] = value
    with pytest.raises(ValueError):
        check(trial, sources)


@pytest.mark.parametrize("field", ["config", "start", "end_exclusive", "update_interval_days", "failures", "predictions", "fits", "skipped"])
def test_missing_trial_fields_are_explicit_errors(sources, trials, field):
    trial = deepcopy(trials[28])
    del trial[field]
    with pytest.raises(ValueError):
        check(trial, sources)


def test_full_prediction_coverage_cannot_be_silently_reduced(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"].pop()
    with pytest.raises(ValueError, match="full expected prediction coverage"):
        check(trial, sources)


def test_duplicate_prediction_ids_are_not_overwritten(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"].append(deepcopy(trial["predictions"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        check(trial, sources)


def test_duplicate_fit_cutoffs_are_not_overwritten(sources, trials):
    trial = deepcopy(trials[28])
    trial["fits"].append(deepcopy(trial["fits"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        check(trial, sources)


def test_missing_fit_cannot_be_hidden_by_complete_predictions(sources, trials):
    trial = deepcopy(trials[28])
    trial["fits"].pop()
    with pytest.raises(ValueError, match="Missing fit"):
        check(trial, sources)


@pytest.mark.parametrize("interval", research.INTERVALS)
def test_fixed_anchor_survives_empty_calendar_blocks(sources, trials, interval):
    trial = trials[interval]
    prediction = next(p for p in trial["predictions"] if p["match_id"] == "after-empty-blocks")
    expected = START + timedelta(days=((date(2024, 10, 10) - START).days // interval) * interval)
    assert prediction["fit_before_date"] == expected.isoformat()
    assert all((date.fromisoformat(f["before_date"]) - START).days % interval == 0 for f in trial["fits"])
    assert len(trial["fits"]) < math.ceil((END - START).days / interval)
    assert check(trial, sources, interval) is trial


def test_empty_block_must_not_add_a_fit_or_shift_later_fit(sources, trials):
    trial = deepcopy(trials[28])
    fit = deepcopy(trial["fits"][0])
    fit["before_date"] = "2024-09-26"  # no eligible target until October 10's block
    # Use a truly empty 28-day block after the last target but before END.
    fit["before_date"] = "2024-10-24"
    trial["fits"].append(fit)
    with pytest.raises(ValueError, match="empty-block"):
        check(trial, sources)


def test_prediction_cannot_use_shifted_nonanchored_cutoff(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"][0]["fit_before_date"] = "2024-07-31"
    with pytest.raises(ValueError, match="anchor"):
        check(trial, sources)


def test_unknown_team_is_whole_match_skip_then_learned_from_earlier_results(sources, trials):
    weekly, monthly = trials[7], trials[28]
    assert {r["match_id"]: r["reason"] for r in weekly["skipped"]}["unknown-C"] == "unseen_team_no_estimated_prior"
    assert "new-known-weekly" in {r["match_id"] for r in weekly["predictions"]}
    assert "new-known-weekly" not in {r["match_id"] for r in monthly["predictions"]}
    learned = next(f for f in weekly["fits"] if f["before_date"] == "2024-08-08")
    assert "unknown-C" in learned["reference_match_ids"]
    assert "football-data:E0:C" in learned["teams"]
    assert "excluded" not in {key for f in weekly["fits"] for key in f["reference_match_ids"]}


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong-reason", "extra-field", "predicted-as-skipped"])
def test_skipped_coverage_and_reasons_are_complete_and_not_fabricated(sources, trials, damage):
    trial = deepcopy(trials[28])
    if damage == "missing":
        trial["skipped"].pop()
    elif damage == "duplicate":
        trial["skipped"].append(deepcopy(trial["skipped"][0]))
    elif damage == "wrong-reason":
        trial["skipped"][0]["reason"] = "unestimated_zero_effect"
    elif damage == "extra-field":
        trial["skipped"][0]["unverifiable"] = 1
    else:
        trial["skipped"].append({"match_id": "anchor-day", "reason": "unseen_team_no_estimated_prior"})
    with pytest.raises(ValueError, match="[Ss]kipped"):
        check(trial, sources)


def test_exclusive_end_and_future_results_do_not_enter_targets_or_saved_history(sources, trials):
    trial = trials[7]
    represented = {r["match_id"] for r in trial["predictions"]}
    history = {key for fit in trial["fits"] for key in fit["reference_match_ids"]}
    assert not {"exclusive-end", "far-future"} & (represented | history)
    changed = [replace(r, full_time=(99, 98)) if r.match_id in {"exclusive-end", "far-future"} else r for r in sources]
    assert check(trial, changed, 7) is trial


def test_same_day_and_later_results_cannot_enter_initial_fit(sources, trials):
    initial = trials[7]["fits"][0]
    assert initial["before_date"] == START.isoformat()
    assert "anchor-day" not in initial["reference_match_ids"]
    assert all(r.match_date < START for r in sources if r.match_id in initial["reference_match_ids"])
    changed = [replace(r, full_time=(7, 6)) if r.match_date >= START else r for r in sources]
    rerun = research.run_frequency(changed, START, END, 7)
    assert rerun["fits"][0] == initial
    # Initial-block forecasts cannot change when same/later actual results change.
    initial_rows = {r["match_id"]: r["forecast"] for r in trials[7]["predictions"] if r["fit_before_date"] == START.isoformat()}
    changed_rows = {r["match_id"]: r["forecast"] for r in rerun["predictions"] if r["fit_before_date"] == START.isoformat()}
    assert initial_rows == changed_rows


@pytest.mark.parametrize("field,value", [
    ("model_type", "dixon_coles"), ("before_date", "2024-08-02"),
    ("training_min_date", "2024-06-02"), ("training_max_date", "2024-08-01"),
    ("n_matches", 999), ("n_matches", True), ("n_matches", 12.0),
    ("teams", ["football-data:E0:A"]), ("competitions", ["D1"]),
    ("basis", "player_conditional"), ("availability_convention", "same_day_included"),
    ("regularization", 0.02), ("regularization", True),
    ("half_life_days", 90), ("rho", 0.01), ("rho", False),
    ("optimizer_iterations", 1000), ("optimizer_iterations", -1),
    ("optimizer_iterations", 1.5), ("optimizer_iterations", True),
    ("intercept", float("inf")), ("intercept", float("nan")),
    ("intercept", "0"), ("intercept", False),
])
def test_fit_metadata_model_settings_and_optimizer_state_are_validated(sources, trials, field, value):
    trial = deepcopy(trials[28])
    trial["fits"][0][field] = value
    with pytest.raises(ValueError):
        check(trial, sources)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "reorder", "same-day", "future", "excluded"])
def test_full_strictly_earlier_history_ids_cannot_be_forged(sources, trials, damage):
    trial = deepcopy(trials[28])
    ids = trial["fits"][0]["reference_match_ids"]
    if damage == "duplicate":
        ids.append(ids[0])
    elif damage == "missing":
        ids.pop()
    elif damage == "reorder":
        ids.reverse()
    else:
        ids[-1] = {"same-day": "anchor-day", "future": "far-future", "excluded": "excluded"}[damage]
    with pytest.raises(ValueError, match="reference_match_ids"):
        check(trial, sources)


@pytest.mark.parametrize("damage", ["missing", "extra", "nan", "inf", "bool", "string", "changed"])
def test_saved_coefficients_require_exact_finite_trained_features(sources, trials, damage):
    trial = deepcopy(trials[28])
    coefficients = trial["fits"][0]["coefficients"]
    key = next(iter(coefficients))
    if damage == "missing":
        del coefficients[key]
    elif damage == "extra":
        coefficients["attack:unseen"] = 0.0
    else:
        coefficients[key] = {"nan": float("nan"), "inf": float("inf"), "bool": True,
                             "string": "0", "changed": coefficients[key] + 0.2}[damage]
    with pytest.raises(ValueError):
        check(trial, sources)


@pytest.mark.parametrize("field,value", [
    ("competition", "D1"), ("date", "2024-08-02"), ("home", "Other"), ("away", "Other"),
    ("actual_score", [5, 0]), ("source_url", "https://example.invalid/wrong.csv"),
    ("training_matches", 999), ("training_matches", True), ("training_matches", 12.0),
    ("training_max_date", "2024-07-14"), ("actual_outcome", 0),
])
def test_prediction_identity_score_and_training_metadata_must_match_source(sources, trials, field, value):
    trial = deepcopy(trials[28])
    trial["predictions"][0][field] = value
    with pytest.raises(ValueError):
        check(trial, sources)


def test_jointly_changed_score_and_outcome_still_fail_source_check(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"][0]["actual_score"] = [5, 0]
    trial["predictions"][0]["actual_outcome"] = 0
    with pytest.raises(ValueError, match="actual_score"):
        check(trial, sources)


def test_changed_odds_and_correctly_recomputed_devig_still_fail_source_check(sources, trials):
    trial = deepcopy(trials[28])
    row = trial["predictions"][0]
    row["market_prices"]["avg_preclosing"] = [3.0, 3.5, 4.0]
    inverse = 1 / np.array(row["market_prices"]["avg_preclosing"])
    row["market_probabilities"]["avg_preclosing"] = list(inverse / inverse.sum())
    with pytest.raises(ValueError, match="market_prices"):
        check(trial, sources)


@pytest.mark.parametrize("damage", ["missing-both", "changed-devig", "invalid-price", "extra-market"])
def test_markets_cannot_be_dropped_changed_or_invented(sources, trials, damage):
    trial = deepcopy(trials[28])
    row = trial["predictions"][0]
    if damage == "missing-both":
        del row["market_prices"]["avg_preclosing"]
        del row["market_probabilities"]["avg_preclosing"]
    elif damage == "changed-devig":
        row["market_probabilities"]["avg_preclosing"] = [0.4, 0.3, 0.3]
    elif damage == "invalid-price":
        row["market_prices"]["avg_preclosing"] = [1.0, 3.5, 4.0]
    else:
        row["market_prices"]["invented"] = [2.0, 4.0, 4.0]
        row["market_probabilities"]["invented"] = [0.5, 0.25, 0.25]
    with pytest.raises(ValueError):
        check(trial, sources)


@pytest.mark.parametrize("path", [
    ("one_x_two", "home"), ("one_x_two", "draw"), ("expected_goals", "away"),
    ("total_goals", "0"), ("total_goals", "4+"), ("over_2_5",), ("both_score",),
    ("away_minus_two_jincai", "win"), ("away_minus_two_jincai", "draw"),
    ("away_minus_two_asian", "push"), ("fair_odds_unbuffered", "away_win"),
    ("fair_odds_unbuffered", "away_minus_two_asian"),
])
def test_entire_forecast_not_only_1x2_is_replayed_from_saved_parameters(sources, trials, path):
    trial = deepcopy(trials[28])
    value = trial["predictions"][0]["forecast"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] += 0.01
    with pytest.raises(ValueError, match="forecast"):
        check(trial, sources)


@pytest.mark.parametrize("damage", ["missing-section", "extra-section", "missing-total-cell", "nonfinite", "string", "bool"])
def test_forecast_shape_and_numeric_domain_are_not_silently_repaired(sources, trials, damage):
    trial = deepcopy(trials[28])
    forecast = trial["predictions"][0]["forecast"]
    if damage == "missing-section":
        del forecast["both_score"]
    elif damage == "extra-section":
        forecast["made_up"] = 0.4
    elif damage == "missing-total-cell":
        del forecast["total_goals"]["4+"]
    else:
        forecast["both_score"] = {"nonfinite": float("inf"), "string": "0.5", "bool": True}[damage]
    with pytest.raises(ValueError, match="forecast"):
        check(trial, sources)


def test_well_normalized_but_changed_saved_probability_vector_is_rejected(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"][0]["probabilities"] = [0.5, 0.3, 0.2]
    with pytest.raises(ValueError, match="probabilities"):
        check(trial, sources)


def test_roundoff_only_differences_are_accepted_without_changing_input(sources, trials):
    trial = deepcopy(trials[28])
    trial["predictions"][0]["forecast"]["both_score"] += 1e-13
    before = deepcopy(trial)
    assert check(trial, sources) is trial
    assert trial == before


@pytest.mark.parametrize("damage", ["duplicate", "mixed-league", "empty", "datetime", "negative-goals", "missing-url", "empty-exclusion", "bad-odds", "unknown-market"])
def test_source_data_are_validated_before_fitting_or_replay(sources, trials, damage, monkeypatch):
    records = deepcopy(sources)
    if damage == "duplicate":
        records.append(records[0])
    elif damage == "mixed-league":
        records[-1] = replace(records[-1], competition="D1")
    elif damage == "empty":
        records = []
    elif damage == "datetime":
        records[0] = replace(records[0], match_date=datetime(2024, 6, 1))
    elif damage == "negative-goals":
        records[0] = replace(records[0], full_time=(-1, 0))
    elif damage == "missing-url":
        records[0] = replace(records[0], source_url="")
    elif damage == "empty-exclusion":
        records[0] = replace(records[0], goal_fit_exclusion="")
    elif damage == "bad-odds":
        records[0] = replace(records[0], markets={"avg_closing": (True, 2.0, 3.0)})
    else:
        records[0] = replace(records[0], markets={"made-up": (2.0, 3.0, 4.0)})
    monkeypatch.setattr(native_experiment, "run_config", lambda *args, **kwargs: pytest.fail("Bad source must not be fitted"))
    with pytest.raises(ValueError):
        check(trials[28], records)
    with pytest.raises(ValueError):
        research.run_frequency(records, START, END, 7)


def test_wrong_declared_competition_is_not_filtered_to_a_smaller_source_set(sources, trials):
    with pytest.raises(ValueError, match="one declared league"):
        research.validate_trial(trials[28], sources, START, END, "D1", 28)


def test_source_changed_historical_score_cannot_change_metadata_or_replay_by_itself(sources, trials):
    # Archives are parameter replays, not independent re-optimizations. Identity
    # checks cover target scores; earlier source score truth is an upstream duty.
    # This test explicitly does not claim a parameter replay can prove fitting.
    changed = [replace(row, full_time=(8, 7)) if row.match_id == "history-0" else row for row in sources]
    assert check(trials[28], changed) is trials[28]


def test_no_target_matches_means_no_fits_but_is_still_a_valid_complete_empty_arm(sources):
    only_history = [row for row in sources if row.match_date < START]
    trial = native_experiment.run_config(only_history, dict(native_experiment.ORIGINAL), START, END, interval_days=7)
    assert trial["predictions"] == trial["fits"] == trial["failures"] == trial["skipped"] == []
    assert check(trial, only_history, 7) is trial


def test_all_targets_unknown_means_explicit_skips_not_default_zero_strength():
    sources = [record("old", "2024-07-01"), record("unknown", "2024-08-02", "C", "D")]
    trial = native_experiment.run_config(sources, dict(native_experiment.ORIGINAL), START, END, interval_days=7)
    assert not trial["predictions"] and not trial["fits"] and not trial["failures"]
    assert trial["skipped"] == [{"match_id": "unknown", "reason": "unseen_team_no_estimated_prior"}]
    assert check(trial, sources, 7) is trial
