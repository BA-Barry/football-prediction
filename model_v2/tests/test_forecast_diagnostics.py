"""Independent ordinary tests for saved-forecast diagnostic calculations."""

from copy import deepcopy
from datetime import date, datetime, timedelta
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_model.evaluation import probability_scores
from football_model.football_data import MARKET_COLUMNS
from football_model.forecast_diagnostics import (
    BLOCKS,
    CLASS_NAMES,
    MODEL_IDS,
    PRIMARY_MARKET,
    REPORT_FILE,
    RESULT_FILE,
    SOURCE_RESULT,
    align_tracks,
    calibration_summary,
    diagnose,
    losses,
    load_saved_tracks,
    moving_block_means,
    probabilities,
    report,
    run,
    validate_prediction,
    write_paired_csv,
)
from football_model.native_experiment import CANDIDATES, ORIGINAL, TEST_END, TEST_START
from football_model.football_data import LEAGUES


TRACK_NAMES = (*MODEL_IDS, "validation_selected")


def row(key="m1", day="2025-08-15", score=(2, 0), p=(0.4, 0.3, 0.3)):
    h, a = score
    return {
        "match_id": key,
        "competition": "E0",
        "date": day,
        "home": "Home " + key,
        "away": "Away " + key,
        "fit_before_date": "2025-08-01",
        "training_max_date": "2025-07-31",
        "actual_score": list(score),
        "actual_outcome": 0 if h > a else 2 if a > h else 1,
        "probabilities": list(p),
        "source_url": "https://example.invalid/synthetic.csv",
        "market_prices": {name: [2.0, 4.0, 4.0] for name in MARKET_COLUMNS},
        "market_probabilities": {name: [0.5, 0.25, 0.25] for name in MARKET_COLUMNS},
    }


def tracks(rows=None):
    if rows is None:
        rows = [row(), row("m2", "2025-08-18", (1, 1)),
                row("m3", "2025-09-08", (0, 1)), row("m4", "2025-10-06", (3, 1))]
    return {name: deepcopy(rows) for name in TRACK_NAMES}


def remove_market(all_tracks, market, ids=None):
    for rows in all_tracks.values():
        for item in rows:
            if ids is None or item["match_id"] in ids:
                del item["market_probabilities"][market]
                del item["market_prices"][market]


@pytest.mark.parametrize("bad", [
    [0.5, 0.5], [[0.5, 0.25, 0.25]], [0.5, 0.25, 0.25, 0],
    [-0.1, 0.5, 0.6], [1.01, 0, -0.01], [np.nan, 0.5, 0.5],
    [np.inf, 0, 0], [0.4, 0.3, 0.2], [0.4, 0.3, 0.30000001],
])
def test_probability_validation_rejects_bad_without_normalizing(bad):
    with pytest.raises(ValueError):
        probabilities(bad)


@pytest.mark.parametrize("p", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.25, 0.25]])
def test_probability_validation_accepts_endpoints(p):
    np.testing.assert_array_equal(probabilities(p), p)


@pytest.mark.parametrize("field,value", [
    ("match_id", ""), ("home", ""), ("away", ""), ("source_url", ""),
    ("home", None), ("competition", "unknown"),
    ("training_max_date", "2025-08-01"),
    ("training_max_date", "2025-08-02"),
    ("fit_before_date", "2025-08-16"),
    ("date", "not-a-date"),
    ("actual_score", [-1, 0]), ("actual_score", [True, 0]),
    ("actual_score", [1.0, 0]), ("actual_score", [1, 0, 2]),
    ("actual_outcome", 1), ("actual_outcome", True), ("actual_outcome", 0.0),
    ("probabilities", [0.5, 0.5, 0.1]),
])
def test_saved_prediction_ordinary_validation(field, value):
    item = row()
    item[field] = value
    with pytest.raises(ValueError):
        validate_prediction(item)


def test_saved_prediction_accepts_fit_cutoff_on_match_date_but_not_training():
    item = row(day="2025-08-01")
    assert validate_prediction(item) is None
    item["training_max_date"] = item["date"]
    with pytest.raises(ValueError, match="strictly earlier"):
        validate_prediction(item)


def test_saved_prediction_rejects_identical_teams():
    item = row()
    item["away"] = item["home"]
    with pytest.raises(ValueError, match="fixture identity"):
        validate_prediction(item)


@pytest.mark.parametrize("bad", [[1, 4, 4], [0, 4, 4], [np.nan, 4, 4], [2, 4], [np.inf, 4, 4]])
def test_saved_market_decimal_odds_must_be_valid(bad):
    item = row()
    item["market_prices"][PRIMARY_MARKET] = bad
    with pytest.raises(ValueError, match="decimal odds"):
        validate_prediction(item)


def test_saved_market_probability_price_keys_must_match():
    item = row()
    del item["market_prices"][PRIMARY_MARKET]
    with pytest.raises(ValueError, match="keys disagree"):
        validate_prediction(item)


def test_saved_unknown_market_is_not_silently_accepted():
    item = row()
    item["market_prices"]["invented"] = [2.0, 4.0, 4.0]
    item["market_probabilities"]["invented"] = [0.5, 0.25, 0.25]
    with pytest.raises(ValueError, match="keys disagree"):
        validate_prediction(item)


def test_saved_market_probability_must_match_declared_devig():
    item = row()
    item["market_probabilities"][PRIMARY_MARKET] = [0.4, 0.3, 0.3]
    with pytest.raises(ValueError, match="proportional de-vig"):
        validate_prediction(item)


@pytest.mark.parametrize("change", [
    {"date": "2025-08-16"}, {"competition": "D1"},
    {"home": "Different home"}, {"away": "Different away"},
    {"actual_score": [3, 0]}, {"actual_score": [1, 1], "actual_outcome": 1},
    {"source_url": "https://example.invalid/different.csv"},
])
def test_shared_id_identity_disagreement_is_rejected(change):
    all_tracks = tracks()
    all_tracks["validation_selected"][0].update(change)
    with pytest.raises(ValueError, match="Conflicting match identity"):
        align_tracks(all_tracks)


def test_shared_id_valid_but_different_market_is_rejected():
    all_tracks = tracks()
    item = all_tracks["validation_selected"][0]
    item["market_prices"][PRIMARY_MARKET] = [2.0, 3.0, 6.0]
    item["market_probabilities"][PRIMARY_MARKET] = [0.5, 1 / 3, 1 / 6]
    with pytest.raises(ValueError, match="Conflicting match identity"):
        align_tracks(all_tracks)


def test_duplicate_ids_are_not_overwritten():
    all_tracks = tracks()
    all_tracks["original"].append(deepcopy(all_tracks["original"][0]))
    with pytest.raises(ValueError, match="Duplicate match ID"):
        align_tracks(all_tracks)


def test_conflicting_overlap_outside_final_cohort_is_still_rejected():
    all_tracks = tracks()
    extra = row("outside", "2025-08-20")
    all_tracks["original"].append(deepcopy(extra))
    extra["home"] = "Conflicting outside home"
    all_tracks["native_dc_180"].append(extra)
    with pytest.raises(ValueError, match="Conflicting match identity"):
        align_tracks(all_tracks)


def test_alignment_reports_outside_ids_and_sorts_chronologically():
    all_tracks = tracks([row("z", "2025-08-18"), row("b"), row("a")])
    all_tracks["original"].append(row("outside"))
    aligned, coverage = align_tracks(all_tracks)
    assert coverage["common_match_ids"] == ["a", "b", "z"]
    assert coverage["outside_common"]["original"] == ["outside"]
    assert coverage["counts"]["original"] == 4
    assert all(list(track) == ["a", "b", "z"] for track in aligned.values())


def test_record_order_does_not_change_alignment_or_diagnostics():
    all_tracks = tracks()
    before = deepcopy(all_tracks)
    expected = diagnose(all_tracks, repeats=32)
    for index, rows in enumerate(all_tracks.values()):
        rows[:] = rows[index:] + rows[:index]
        rows.reverse()
    assert align_tracks(all_tracks) == align_tracks(before)
    assert diagnose(all_tracks, repeats=32) == expected


@pytest.mark.parametrize("bad", [{}, {"one": []}, {"one": [row("a")], "two": [row("b")]}])
def test_empty_tracks_or_empty_intersection_rejected(bad):
    with pytest.raises(ValueError):
        align_tracks(bad)


def test_diagnose_requires_all_four_declared_tracks():
    all_tracks = tracks()
    del all_tracks["native_dc_180"]
    with pytest.raises(ValueError, match="all four"):
        diagnose(all_tracks, repeats=8)


def test_market_missing_in_only_one_track_is_identity_conflict():
    all_tracks = tracks()
    remove_market({"original": all_tracks["original"]}, PRIMARY_MARKET, {"m1"})
    with pytest.raises(ValueError, match="Conflicting match identity"):
        diagnose(all_tracks, repeats=8)


def test_missing_market_is_explicit_and_every_comparator_uses_same_subcohort():
    all_tracks = tracks()
    remove_market(all_tracks, PRIMARY_MARKET, {"m1"})
    result = diagnose(all_tracks, repeats=16)
    assert result["coverage"]["common_match_ids"] == ["m1", "m2", "m3", "m4"]
    assert result["coverage"]["missing_market_ids"][PRIMARY_MARKET] == ["m1"]
    primary = result["markets"][PRIMARY_MARKET]
    assert primary["match_ids"] == ["m2", "m3", "m4"]
    assert primary["n"] == 3
    assert all(score["n"] == 3 for score in primary["scores"].values())
    assert result["markets"]["avg_closing"]["n"] == 4


def test_wholly_missing_secondary_market_is_none_not_imputed():
    all_tracks = tracks()
    remove_market(all_tracks, "avg_closing")
    result = diagnose(all_tracks, repeats=8)
    assert result["markets"]["avg_closing"] is None
    assert result["coverage"]["missing_market_ids"]["avg_closing"] == ["m1", "m2", "m3", "m4"]


def test_wholly_missing_primary_market_does_not_substitute_closing():
    all_tracks = tracks()
    remove_market(all_tracks, PRIMARY_MARKET)
    with pytest.raises(ValueError, match="do not substitute closing"):
        diagnose(all_tracks, repeats=8)


def test_diagnostics_never_mutate_source_rows():
    all_tracks = tracks()
    before = deepcopy(all_tracks)
    diagnose(all_tracks, repeats=8)
    assert all_tracks == before


def expanded_match_bootstrap(dates, values, length, repeats, seed):
    """Independent oracle: enumerate sampled weeks, expand actual match rows."""
    dates = [date.fromisoformat(d) if isinstance(d, str) else d for d in dates]
    first_day = min(dates)
    first_monday = first_day - timedelta(days=first_day.weekday())
    calendar = []
    cursor = first_monday
    while cursor <= max(dates):
        calendar.append([i for i, day in enumerate(dates) if cursor <= day < cursor + timedelta(days=7)])
        cursor += timedelta(days=7)
    rng = np.random.default_rng(seed)
    output, empty = [], 0
    for _ in range(repeats):
        starts = rng.integers(0, len(calendar) - length + 1, size=math.ceil(len(calendar) / length))
        sampled_weeks = [w for start in starts for w in range(int(start), int(start) + length)][:len(calendar)]
        indices = [i for week in sampled_weeks for i in calendar[week]]
        if indices:
            output.append(np.asarray(values)[indices].mean(axis=0))
        else:
            empty += 1
    return np.asarray(output), empty


def test_moving_blocks_weight_matches_not_equal_week_means_and_joint_columns():
    dates = ["2025-08-04"] + ["2025-08-11"] * 5 + ["2025-08-18"] + ["2025-08-25"] * 3
    first = np.asarray([0] + [10] * 5 + [20] + [30] * 3, dtype=float)
    values = np.column_stack([first, first, 3 * first - 5, np.zeros(len(first))])
    result = moving_block_means(dates, values, block_weeks=2, repeats=61, seed=7)
    expected, empty = expanded_match_bootstrap(dates, values, 2, 61, 7)
    assert result["weekly_match_counts"] == [1, 5, 1, 3]
    assert result["mean"][0] == 16
    assert result["mean"][0] != np.mean([0, 10, 20, 30])
    assert empty == result["zero_count_repeats"] == 0
    np.testing.assert_allclose(result["interval_95"], np.quantile(expected, [0.025, 0.975], axis=0).T)
    assert result["interval_95"][0] == result["interval_95"][1]
    np.testing.assert_allclose(result["interval_95"][2], 3 * np.asarray(result["interval_95"][0]) - 5)
    assert result["interval_95"][3] == [0, 0]


def test_empty_calendar_weeks_cross_year_and_empty_resamples_are_preserved():
    dates = ["2025-12-29", "2026-02-02"]
    values = np.asarray([[1.0, -2.0], [3.0, -6.0]])
    result = moving_block_means(dates, values, block_weeks=1, repeats=100, seed=11)
    expected, empty = expanded_match_bootstrap(dates, values, 1, 100, 11)
    assert result["calendar_week_starts"] == ["2025-12-29", "2026-01-05", "2026-01-12",
                                              "2026-01-19", "2026-01-26", "2026-02-02"]
    assert result["weekly_match_counts"] == [1, 0, 0, 0, 0, 1]
    assert result["n_calendar_weeks"] == 6
    assert result["n_nonempty_weeks"] == 2
    assert result["zero_count_repeats"] == empty > 0
    assert result["valid_repeats"] + empty == 100
    np.testing.assert_allclose(result["interval_95"], np.quantile(expected, [0.025, 0.975], axis=0).T)


def test_bootstrap_seed_reproducible_with_date_objects():
    dates = [date(2025, 8, 4) + timedelta(days=7 * w) for w in range(6)]
    values = np.arange(12).reshape(6, 2)
    a = moving_block_means(dates, values, block_weeks=2, repeats=101, seed=17)
    b = moving_block_means([d.isoformat() for d in dates], values, block_weeks=2, repeats=101, seed=17)
    assert a == b


@pytest.mark.parametrize("dates,length", [
    (["2025-08-04"], 1), (["2025-08-04", "2025-08-08"], 2),
    (["2025-08-04", "2025-08-11"], 2), (["2025-08-04", "2025-08-11"], 8),
])
def test_single_week_or_non_smaller_block_has_no_spurious_interval(dates, length):
    result = moving_block_means(dates, np.ones((len(dates), 1)), block_weeks=length, repeats=10)
    assert result["interval_95"] is None
    assert result["valid_repeats"] == 0
    assert result["unavailable_reason"] == "block_length_not_smaller_than_calendar_span"
    assert result["mean"] == [1.0]


@pytest.mark.parametrize("settings", [
    {"block_weeks": 0}, {"block_weeks": True}, {"block_weeks": 1.5},
    {"repeats": 1}, {"repeats": True}, {"repeats": 3.5},
    {"seed": -1}, {"seed": True}, {"seed": 0.5},
])
def test_bootstrap_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        moving_block_means(["2025-08-04", "2025-08-18"], [[1], [2]], **settings)


@pytest.mark.parametrize("dates,values", [
    ([], []), (["2025-08-04"], [1]), (["2025-08-04"], [[1], [2]]),
    (["2025-08-04"], [[]]), (["2025-08-04"], [[np.nan]]),
    (["2025-08-04"], [[np.inf]]), ([datetime(2025, 8, 4)], [[1]]),
    (["not-a-date"], [[1]]),
])
def test_bootstrap_rejects_bad_dates_or_arrays(dates, values):
    with pytest.raises(ValueError):
        moving_block_means(dates, values)


def test_identical_model_and_market_probabilities_produce_exact_zero_contrasts():
    all_tracks = tracks()
    for rows in all_tracks.values():
        for item in rows:
            item["probabilities"] = [0.5, 0.25, 0.25]
    result = diagnose(all_tracks, repeats=32)
    for cohort in result["markets"].values():
        for name, contrast in cohort["contrasts"].items():
            assert contrast["mean_log_loss_difference"] == 0
            for length in BLOCKS:
                feature = cohort["bootstrap"][str(length)]["features"][name]
                assert feature["mean"] == 0
                assert feature["interval_95"] == [0, 0]


def test_calibration_bin_endpoints_and_class_binary_brier_sum():
    p = np.eye(3)
    y = np.asarray([0, 1, 2])
    result = calibration_summary(p, y)
    for name in CLASS_NAMES:
        bins = [entry for entry in result["equal_width_5_bins"] if entry["outcome"] == name]
        assert [(entry["bin"], entry["n"]) for entry in bins] == [(0, 2), (4, 1)]
        assert sum(entry["n"] for entry in bins) == 3
        assert bins[0]["mean_probability"] == bins[0]["observed_frequency"] == 0
        assert bins[1]["mean_probability"] == bins[1]["observed_frequency"] == 1
        assert result["classes"][name]["binary_brier"] == 0
    p = np.asarray([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
    result = calibration_summary(p, y)
    assert sum(c["binary_brier"] for c in result["classes"].values()) == pytest.approx(
        probability_scores(p, y)["brier_sum"])
    assert sum(c["mean_probability_minus_frequency"] for c in result["classes"].values()) == pytest.approx(0)


def test_outcome_and_group_contributions_reconstruct_overall_difference():
    all_tracks = tracks()
    for name, rows in all_tracks.items():
        rows[2]["competition"] = "D1"
        if name == "validation_selected":
            for item in rows:
                item["probabilities"] = [0.45, 0.25, 0.3]
    primary = diagnose(all_tracks, repeats=16)["markets"][PRIMARY_MARKET]
    for name, contrast in primary["contrasts"].items():
        by_outcome = contrast["by_actual_outcome"]
        assert sum(c["n"] for c in by_outcome.values()) == primary["n"]
        assert sum(c["weighted_contribution_to_overall"] for c in by_outcome.values()) == pytest.approx(
            contrast["mean_log_loss_difference"])
        for c in by_outcome.values():
            assert c["weighted_contribution_to_overall"] == pytest.approx(
                c["within_outcome_mean_difference"] * c["n"] / primary["n"])
        for grouping in ("by_league", "by_calendar_month"):
            groups = primary[grouping]
            assert sum(g["n"] for g in groups.values()) == primary["n"]
            assert sum(g["contrasts"][name] * g["n"] for g in groups.values()) / primary["n"] == pytest.approx(
                contrast["mean_log_loss_difference"])
    for method, score in primary["scores"].items():
        for grouping in ("by_league", "by_calendar_month"):
            assert sum(g["scores"][method]["log_loss"] * g["n"] for g in primary[grouping].values()) / primary["n"] == pytest.approx(
                score["log_loss"])
    expected = sum(math.log([0.5, 0.25, 0.25][item["actual_outcome"]] /
                            item["probabilities"][item["actual_outcome"]]) for item in all_tracks["original"]) / 4
    assert primary["contrasts"][f"original_minus_{PRIMARY_MARKET}"]["mean_log_loss_difference"] == pytest.approx(expected)


def test_absent_outcome_has_no_fabricated_conditional_mean():
    primary = diagnose(tracks([row("m1"), row("m2", "2025-10-06")]), repeats=8)["markets"][PRIMARY_MARKET]
    for contrast in primary["contrasts"].values():
        for name in ("draw", "away"):
            part = contrast["by_actual_outcome"][name]
            assert part == {"n": 0, "within_outcome_mean_difference": None,
                            "weighted_contribution_to_overall": 0}


def test_loss_zero_probability_is_explicit_finite_clipped_loss_not_deleted():
    result = losses(np.asarray([[0, 0.5, 0.5], [1, 0, 0]]), np.asarray([0, 0]))
    np.testing.assert_allclose(result, [-math.log(np.finfo(float).tiny), 0])
    assert np.isfinite(result).all()


def write_payload(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read_payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def saved_parent(tmp_path, selected="poisson_h180"):
    """Small ordinary saved files, not fitted models or a frozen contract."""
    parent = tmp_path / "source"
    parent.mkdir()
    (parent / "较晚赛季").mkdir()
    configurations = {item["id"]: item for item in [*CANDIDATES, ORIGINAL]}
    selection = {comp: {"config": deepcopy(configurations[selected])} for comp in LEAGUES}
    ids = []
    for index, comp in enumerate(sorted(LEAGUES)):
        prediction = row("match-" + comp, ["2025-08-15", "2025-08-18", "2025-09-08", "2025-10-06", "2025-10-13"][index])
        prediction["competition"] = comp
        ids.append(prediction["match_id"])
        for config_id in {*MODEL_IDS.values(), selected}:
            item = deepcopy(prediction)
            if config_id == "poisson_h180":
                item["probabilities"] = [0.45, 0.25, 0.3]
            elif config_id == "dixon_coles_h180":
                item["probabilities"] = [0.5, 0.3, 0.2]
            trial = {"config": deepcopy(configurations[config_id]), "start": TEST_START.isoformat(),
                     "end_exclusive": TEST_END.isoformat(), "update_interval_days": 28,
                     "predictions": [item], "failures": [], "skipped": []}
            write_payload(parent / "较晚赛季" / f"{comp}_{config_id}.json", trial)
    write_payload(parent / "验证期选择.json", selection)
    write_payload(parent / SOURCE_RESULT, {"selection": selection, "common_test_match_ids": ids})
    return parent


def alter_selection_both(parent, change):
    selection = read_payload(parent / "验证期选择.json")
    change(selection)
    prior = read_payload(parent / SOURCE_RESULT)
    prior["selection"] = deepcopy(selection)
    write_payload(parent / "验证期选择.json", selection)
    write_payload(parent / SOURCE_RESULT, prior)


@pytest.mark.parametrize("selected", ["poisson_h180", "dixon_coles_h180"])
def test_saved_loader_reuses_same_config_and_reconstructs_four_tracks(tmp_path, monkeypatch, selected):
    parent = saved_parent(tmp_path, selected)
    reads = []
    ordinary_read = Path.read_text

    def observed_read(path, *args, **kwargs):
        reads.append(path)
        return ordinary_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observed_read)
    matched, coverage = load_saved_tracks(parent)
    assert set(matched) == set(TRACK_NAMES)
    assert len(coverage["common_match_ids"]) == len(LEAGUES)
    assert all(len(rows) == len(LEAGUES) for rows in matched.values())
    selected_fixed = "native_poisson_180" if selected == "poisson_h180" else "native_dc_180"
    assert matched["validation_selected"] == matched[selected_fixed]
    trial_reads = [p for p in reads if p.parent.name == "较晚赛季"]
    assert len(trial_reads) == 3 * len(LEAGUES)
    assert len(set(trial_reads)) == len(trial_reads)


def test_saved_loader_selection_file_must_agree_with_result(tmp_path):
    parent = saved_parent(tmp_path)
    selection = read_payload(parent / "验证期选择.json")
    selection["E0"]["config"] = deepcopy(ORIGINAL)
    write_payload(parent / "验证期选择.json", selection)
    with pytest.raises(ValueError, match="selection disagrees"):
        load_saved_tracks(parent)


def test_saved_loader_selection_must_have_the_expected_leagues(tmp_path):
    parent = saved_parent(tmp_path)
    alter_selection_both(parent, lambda selection: selection.pop("E0"))
    with pytest.raises(ValueError):
        load_saved_tracks(parent)


@pytest.mark.parametrize("field,value", [("id", "unknown"), ("family", "dixon_coles"), ("half_life_days", 365)])
def test_saved_loader_selection_config_label_must_match_actual_known_settings(tmp_path, field, value):
    parent = saved_parent(tmp_path)

    def alter(selection):
        selection["E0"]["config"][field] = value

    alter_selection_both(parent, alter)
    with pytest.raises(ValueError):
        load_saved_tracks(parent)


@pytest.mark.parametrize("change", [
    {"start": "2025-08-02"}, {"end_exclusive": "2026-06-30"},
    {"update_interval_days": 7}, {"update_interval_days": 28.5},
    {"config": {"id": "poisson_h180", "family": "dixon_coles", "half_life_days": 180}},
    {"config": {"id": "poisson_h180", "family": "poisson", "half_life_days": 365}},
    {"config": {"id": "poisson_h90", "family": "poisson", "half_life_days": 90}},
])
def test_saved_loader_rejects_run_setting_or_model_label_mismatch(tmp_path, change):
    parent = saved_parent(tmp_path)
    path = parent / "较晚赛季" / "E0_poisson_h180.json"
    trial = read_payload(path)
    trial.update(change)
    write_payload(path, trial)
    with pytest.raises(ValueError):
        load_saved_tracks(parent)


@pytest.mark.parametrize("change", [
    {"competition": "D1"}, {"date": "2025-07-31"}, {"date": TEST_END.isoformat()},
])
def test_saved_loader_rejects_prediction_outside_declared_league_or_period(tmp_path, change):
    parent = saved_parent(tmp_path)
    path = parent / "较晚赛季" / "E0_poisson_h180.json"
    trial = read_payload(path)
    trial["predictions"][0].update(change)
    write_payload(path, trial)
    with pytest.raises(ValueError, match="outside declared"):
        load_saved_tracks(parent)


@pytest.mark.parametrize("edit", ["extra", "missing"])
def test_saved_loader_rejects_changed_original_common_ids(tmp_path, edit):
    parent = saved_parent(tmp_path)
    prior = read_payload(parent / SOURCE_RESULT)
    if edit == "extra":
        prior["common_test_match_ids"].append("missing-from-all-tracks")
    else:
        prior["common_test_match_ids"].pop()
    write_payload(parent / SOURCE_RESULT, prior)
    with pytest.raises(ValueError, match="common cohort differs"):
        load_saved_tracks(parent)


def test_paired_csv_is_readable_and_reproduces_each_saved_loss_with_blank_missing_market(tmp_path):
    all_tracks = tracks()
    remove_market(all_tracks, "avg_closing", {"m1"})
    matched, coverage = align_tracks(all_tracks)
    path = tmp_path / "paired.csv"
    write_paired_csv(path, matched)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    assert [r["match_id"] for r in saved_rows] == coverage["common_match_ids"]
    for saved in saved_rows:
        item = matched["original"][saved["match_id"]]
        y = item["actual_outcome"]
        assert saved["date"] == item["date"]
        assert saved["competition"] == item["competition"]
        assert int(saved["actual_outcome"]) == y
        for method in TRACK_NAMES:
            expected = -math.log(matched[method][saved["match_id"]]["probabilities"][y])
            assert float(saved[method + "_log_loss"]) == pytest.approx(expected)
        for market in MARKET_COLUMNS:
            if market in item["market_probabilities"]:
                assert float(saved[market + "_log_loss"]) == pytest.approx(-math.log(item["market_probabilities"][market][y]))
            else:
                assert saved[market + "_log_loss"] == ""


def test_report_handles_unavailable_intervals_and_empty_secondary_market():
    all_tracks = tracks([row()])
    remove_market(all_tracks, "avg_closing")
    result = diagnose(all_tracks, repeats=8)
    result.update({"parent_directory": "G:/synthetic-source", "settings": {"repeats": 8, "seed": 20261023}})
    content = report(result)
    assert "不可估计" in content
    assert "| avg_closing | 0 |" in content
    assert "G:/synthetic-source" in content
    assert "探索性" in content
    assert "0次拟合" in content


def test_small_saved_file_run_exports_json_report_and_csv_without_changing_sources(tmp_path):
    parent = saved_parent(tmp_path)
    before = {path.relative_to(parent): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    root = tmp_path / "project"
    output = run(root, parent, repeats=8)
    assert output.is_relative_to(root / "outputs")
    result = read_payload(output / RESULT_FILE)
    assert result["new_model_fits"] == result["new_model_predictions"] == 0
    assert result["probabilities_changed"] is False
    primary = result["markets"][PRIMARY_MARKET]
    assert primary["n"] == len(LEAGUES)
    assert result["source_loading_coverage"]["common_match_ids"] == primary["match_ids"]
    with (output / "同场逐场损失.csv").open(encoding="utf-8-sig", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == primary["n"]
    for name, scores in primary["scores"].items():
        observed = sum(float(item[name + "_log_loss"]) for item in pairs) / len(pairs)
        assert observed == pytest.approx(scores["log_loss"])
    saved_report = (output / REPORT_FILE).read_text(encoding="utf-8")
    assert saved_report == report(result)
    assert str(parent) in saved_report
    assert f"{primary['scores']['original']['log_loss']:.6f}" in saved_report
    assert {path.relative_to(parent): path.read_bytes() for path in parent.rglob("*") if path.is_file()} == before
