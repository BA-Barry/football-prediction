"""Small synthetic timing, exposure, selection and reuse checks; no real fit."""

from copy import deepcopy
from datetime import date
import json

import pytest

import football_model.scaling_experiment as scaling


def match(mid, day, clock="early", stints=1):
    return {"match_id": mid, "match_date": day, "clock": clock,
            "stints": [{"duration_seconds": 5400 / stints,
                        "home_players": ["p"], "away_players": ["q"]} for _ in range(stints)]}


def predictions(rows, loss=2):
    return {"predictions": [{"match_id": m["match_id"], "match_date": m["match_date"],
        "n_stints": len(m["stints"]), "actual_home_target": 1, "actual_away_target": 0,
        "expected_home_target": 1.2, "expected_away_target": .8, "point_process_loss": loss,
        "loss_kind": "point_process_nll_up_to_model_independent_constants"} for m in rows],
        "skipped": [], "requested_matches": len(rows)}


def paths():
    rows = [("old", "2017-11-20"), ("boundary", "2017-11-27"),
            ("val1a", "2017-12-02"), ("val1b", "2017-12-03"),
            ("val2", "2017-12-09"), ("val3", "2018-01-28"),
            ("gap1", "2018-01-29"), ("gap2", "2018-01-30"), ("gap3", "2018-01-31"),
            ("later1", "2018-02-03"), ("later2", "2018-02-10")]
    return {c: [match(mid, day, c, index + 1) for mid, day in (reversed(rows) if index == 1 else rows)]
            for index, c in enumerate(scaling.SEASONS["wyscout_2017_18"]["clocks"])}


def test_effective_ridge_is_player_base_times_physical_exposure_ratio():
    assert scaling.effective_ridge(100, 5400, 5400) == 100
    assert scaling.effective_ridge(100, 16200, 5400) == 300
    assert scaling.effective_ridge(10, 5400, 10800) == 5


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("bad", [True, False, 0, -1, float("nan"), float("inf"), "100", None])
def test_effective_ridge_requires_positive_finite_numeric_not_boolean(position, bad):
    values = [100., 5400., 5400.]
    values[position] = bad
    with pytest.raises(ValueError, match="Positive finite"):
        scaling.effective_ridge(*values)


@pytest.mark.parametrize("values", [(1e308, 1e308, 1), (1, 1e308, 1e-308), (1e-300, 1e-300, 1e300)])
def test_effective_ridge_rejects_product_ratio_overflow_or_underflow(values):
    with pytest.raises(ValueError, match="remain finite and positive"):
        scaling.effective_ridge(*values)


def test_exposure_counts_physical_stints_once_not_both_sides_or_player_rows():
    rows = [match("one", "2017-11-20"), match("two", "2017-11-21", stints=3)]
    rows[0]["stints"][0]["home_players"] = [f"home{i}" for i in range(11)]
    rows[0]["stints"][0]["away_players"] = [f"away{i}" for i in range(11)]
    rows[0]["player_minutes"] = [{"on_pitch_seconds": 999999} for _ in range(22)]
    before = deepcopy(rows)
    assert scaling.training_exposure(rows) == 10800
    assert rows == before
    assert scaling.training_exposure([match("one", "2017-11-20", stints=30)]) == 5400


@pytest.mark.parametrize("duration", [0, -1, True, False, float("nan"), float("inf"), "5400", None])
def test_exposure_rejects_nonpositive_nonfinite_or_non_numeric_durations(duration):
    rows = [match("one", "2017-11-20")]
    rows[0]["stints"][0]["duration_seconds"] = duration
    with pytest.raises(ValueError, match="Positive finite stint"):
        scaling.training_exposure(rows)


def test_exposure_rejects_empty_duplicate_histories_and_missing_stints():
    row = match("one", "2017-11-20")
    with pytest.raises(ValueError, match="Distinct nonempty history"):
        scaling.training_exposure([])
    with pytest.raises(ValueError, match="Distinct nonempty history"):
        scaling.training_exposure([row, deepcopy(row)])
    row["stints"] = []
    with pytest.raises(ValueError, match="Positive finite stint"):
        scaling.training_exposure([row])


def test_validation_windows_are_strict_monday_histories_before_january29_selection():
    data, config = paths(), scaling.SEASONS["wyscout_2017_18"]
    before = deepcopy(data)
    windows = scaling.validation_windows(data, config)
    assert [cutoff for cutoff, _, _ in windows] == [date(2017, 11, 27), date(2017, 12, 4), date(2018, 1, 22)]
    cutoff, history, target = windows[0]
    assert [m["match_id"] for m in history] == ["old"]
    assert {m["match_id"] for m in target} == {"val1a", "val1b"}
    assert all(m["match_id"] != "boundary" for m in history)
    assert {m["match_id"] for _, _, target in windows for m in target} == {"val1a", "val1b", "val2", "val3"}
    for cutoff, history, target in windows:
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(config["train_before"] <= date.fromisoformat(m["match_date"]) < config["selection_before"] for m in target)
    assert windows[-1][0] < date(2018, 1, 29)
    assert data == before


def test_selection_cutoff_cannot_extend_past_first_later_weekly_fit():
    config = {**scaling.SEASONS["wyscout_2017_18"], "selection_before": date(2018, 1, 30)}
    with pytest.raises(ValueError, match="precede the first weekly fit cutoff"):
        scaling.validation_windows(paths(), config)


def policy_scores():
    data = paths()
    rows = {c: [m for m in values if m["match_id"] in {"val1a", "val2"}] for c, values in data.items()}
    fixed = {c: {"10": predictions(values, 1), "100": predictions(values, 2)} for c, values in rows.items()}
    scaled = {c: {"10": predictions(values, 4), "100": predictions(values, 3)} for c, values in rows.items()}
    return fixed, scaled


def test_policy_selections_use_one_all_candidate_clock_and_policy_intersection():
    fixed, scaled = policy_scores()
    for c in fixed:
        for p in fixed[c]["10"]["predictions"]:
            if p["match_id"] == "val2":
                p["point_process_loss"] = 100
        for p in fixed[c]["100"]["predictions"]:
            if p["match_id"] == "val2":
                p["point_process_loss"] = 0
    scaled["late"]["100"]["predictions"] = [p for p in scaled["late"]["100"]["predictions"] if p["match_id"] == "val1a"]
    before = deepcopy((fixed, scaled))
    result = scaling.common_policy_selections(fixed, scaled)
    assert result["common_match_ids"] == ["val1a"]
    assert result["fixed"]["common_match_ids"] == result["scaled"]["common_match_ids"] == ["val1a"]
    assert result["fixed"]["selected_player_ridge"] == 10
    assert result["scaled"]["selected_player_ridge"] == 100
    assert result["fixed"]["candidate_metrics"]["10"]["mean_clock_match_loss"] == 1
    assert (fixed, scaled) == before


def test_policy_selection_ties_choose_larger_base_with_no_policy_averaging():
    fixed, scaled = policy_scores()
    for policy in (fixed, scaled):
        for tracks in policy.values():
            tracks["100"] = deepcopy(tracks["10"])
    result = scaling.common_policy_selections(fixed, scaled)
    assert result["fixed"]["selected_player_ridge"] == result["scaled"]["selected_player_ridge"] == 100
    assert result["fixed"]["candidate_metrics"]["100"]["mean_clock_match_loss"] == 1
    assert result["scaled"]["candidate_metrics"]["100"]["mean_clock_match_loss"] == 4


@pytest.mark.parametrize("outside_common", [False, True])
def test_cross_policy_observed_target_identity_must_agree_even_outside_common(outside_common):
    fixed, scaled = policy_scores()
    if outside_common:
        scaled["late"]["100"]["predictions"] = [p for p in scaled["late"]["100"]["predictions"] if p["match_id"] == "val1a"]
    for tracks in scaled.values():
        for score in tracks.values():
            for p in score["predictions"]:
                if p["match_id"] == "val2":
                    p["actual_home_target"] = 2
    with pytest.raises(ValueError, match="same match date, target and loss kind"):
        scaling.common_policy_selections(fixed, scaled)


def test_policy_selection_empty_global_intersection_is_not_zero_loss_success():
    fixed, scaled = policy_scores()
    for tracks in fixed.values():
        for score in tracks.values():
            score["predictions"] = [p for p in score["predictions"] if p["match_id"] == "val1a"]
    for tracks in scaled.values():
        for score in tracks.values():
            score["predictions"] = [p for p in score["predictions"] if p["match_id"] == "val2"]
    with pytest.raises(ValueError, match="No common validation matches across policies"):
        scaling.common_policy_selections(fixed, scaled)


def test_policy_clocks_candidates_duplicates_and_within_clock_paths_checked():
    fixed, scaled = policy_scores()
    wrong_clock = deepcopy(scaled)
    del wrong_clock["late"]
    with pytest.raises(ValueError, match="same declared clocks"):
        scaling.common_policy_selections(fixed, wrong_clock)
    wrong_candidates = {c: {**v, "1000": deepcopy(v["100"])} for c, v in scaled.items()}
    with pytest.raises(ValueError, match="same candidate base-ridge keys"):
        scaling.common_policy_selections(fixed, wrong_candidates)
    duplicate = deepcopy(scaled)
    duplicate["early"]["10"]["predictions"].append(deepcopy(duplicate["early"]["10"]["predictions"][0]))
    with pytest.raises(ValueError, match="distinct nonempty match IDs"):
        scaling.common_policy_selections(fixed, duplicate)
    wrong_path = deepcopy(scaled)
    wrong_path["early"]["10"]["predictions"][0]["n_stints"] += 1
    with pytest.raises(ValueError, match="same actual state path"):
        scaling.common_policy_selections(fixed, wrong_path)


def test_scaled_validation_first_week_reuses_ratio_one_later_fits_scale_without_future(tmp_path, monkeypatch):
    data, config = paths(), scaling.SEASONS["wyscout_2017_18"]
    original = deepcopy(data)
    parent = tmp_path / "parent"
    parent_models = parent / "validation" / "models"
    parent_models.mkdir(parents=True)
    val_rows = {c: [m for m in rows if m["match_id"].startswith("val")] for c, rows in data.items()}
    for c in config["clocks"]:
        for base in scaling.PLAYER_RIDGES:
            scaling.write_json(parent_models / f"2017-11-27_{c}_ridge{base:g}.json", {})
            scaling.write_json(parent / "validation" / f"validation_{c}_ridge{base:g}_predictions.json",
                               predictions(val_rows[c], base / 10000))
    calls = []

    def score_call(history, target, cutoff, ridge, players, destination, progress, cached=None):
        assert players is True
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(config["train_before"] <= date.fromisoformat(m["match_date"]) < config["selection_before"] for m in target)
        assert all(m["clock"] == target[0]["clock"] for m in history)
        base = float(destination.stem.rsplit("ridge", 1)[1])
        assert ridge == pytest.approx(base * len(history))
        calls.append({"cutoff": cutoff, "cached": cached, "ridge": ridge, "base": base,
                      "clock": target[0]["clock"], "history": [m["match_id"] for m in history]})
        if cached is not None:
            assert cutoff == date(2017, 11, 27)
            assert ridge == base
            assert cached.is_file()
            progress["reused_model_evaluations"] += 1
        else:
            assert cutoff > date(2017, 11, 27)
            progress["actual_model_fits"] += 1
        scaling.write_json(destination, {"stub": True})
        return predictions(target, base / 10000), {"include_players": players, "player_ridge": ridge}

    def reused(cached, history, target, cutoff, ridge, players, destination, progress):
        return score_call(history, target, cutoff, ridge, players, destination, progress, cached)

    monkeypatch.setattr(scaling, "fit_saved", score_call)
    monkeypatch.setattr(scaling, "reuse_saved", reused)
    progress = {"actual_model_fits": 7, "reused_model_evaluations": 9}
    result = scaling.scaled_validation(data, config, parent, tmp_path / "validation", progress)
    assert result["active_weeks"] == 3
    assert result["requested_matches"] == 4
    assert result["common_match_ids"] == ["val1a", "val1b", "val2", "val3"]
    assert result["reference"]["cutoff"] == "2017-11-27"
    assert result["reference"]["history_match_ids"] == ["old"]
    assert result["reference"]["exposure_seconds_by_clock"] == {c: 5400 for c in config["clocks"]}
    assert len(calls) == 36
    assert sum(call["cached"] is not None for call in calls) == 12
    assert progress == {"actual_model_fits": 31, "reused_model_evaluations": 21}
    assert result["fixed"]["selected_player_ridge"] == result["scaled"]["selected_player_ridge"] == 10
    assert len(list((tmp_path / "validation" / "models").glob("*.json"))) == 36
    assert all(not mid.startswith("gap") for call in calls for mid in call["history"])
    assert data == original


@pytest.mark.parametrize("old_base,new_base,missing_cache,expected_fits,expected_reuses", [
    (100, 100, False, 6, 12),
    (10, 10, False, 3, 18),
    (10, 1000, False, 15, 18),
    (100, 100, True, 7, 11),
])
def test_later_two_by_two_reuses_aliases_and_fits_only_distinct_missing_settings(
        tmp_path, monkeypatch, old_base, new_base, missing_cache, expected_fits, expected_reuses):
    name, data = "wyscout_2017_18", paths()
    config = scaling.SEASONS[name]
    source, parent, output = tmp_path / "source", tmp_path / "parent", tmp_path / "output"
    source.mkdir()
    parent.mkdir()
    output.mkdir()
    scaling.write_json(source / config["source_file"], data)
    parent_models = parent / name / "models"
    parent_models.mkdir(parents=True)
    train_ids, val_ids, first_cutoff = scaling.selection_split(data, config["train_before"], config["selection_before"], config["target_from"])
    scaling.write_json(parent / name / "分段与来源.json", {"source_directory": str(source),
        "source_file": config["source_file"], "training_match_ids": train_ids,
        "validation_match_ids": val_ids, "first_weekly_cutoff": first_cutoff.isoformat()})
    prior = {"seasons": {name: {"source_directory": str(source), "source_matches": len(data["early"]),
                               "selection": {"selected_player_ridge": old_base}}}}
    for cutoff in ("2018-01-29", "2018-02-05"):
        for clock in config["clocks"]:
            for track in ("team", "fixed100", "selected"):
                if not (missing_cache and cutoff == "2018-01-29" and clock == "early" and track == "team"):
                    scaling.write_json(parent_models / f"{cutoff}_{clock}_{track}.json", {})
    events, calls = [], []

    def selected_validation(source_paths, settings, parent_season, destination, progress):
        assert source_paths == data
        assert parent_season == parent / name
        events.append("selected")
        destination.mkdir()
        return {"fixed": {"selected_player_ridge": old_base},
                "scaled": {"selected_player_ridge": new_base},
                "reference": {"exposure_seconds_by_clock": {c: 5400 for c in config["clocks"]}}}

    def score_call(history, target, cutoff, ridge, players, destination, progress, cached=None):
        assert events == ["selected"]
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(date.fromisoformat(m["match_date"]) >= config["target_from"] for m in target)
        assert all(m["clock"] == target[0]["clock"] for m in history)
        calls.append((cutoff, target[0]["clock"], players, ridge, cached))
        if cached is not None:
            assert cached.is_file()
            progress["reused_model_evaluations"] += 1
        else:
            progress["actual_model_fits"] += 1
        scaling.write_json(destination, {"stub": True})
        return predictions(target, 3 if not players else ridge / 10000), {"player_ridge": ridge}

    def reused(cached, history, target, cutoff, ridge, players, destination, progress):
        return score_call(history, target, cutoff, ridge, players, destination, progress, cached)

    monkeypatch.setattr(scaling, "scaled_validation", selected_validation)
    monkeypatch.setattr(scaling, "fit_saved", score_call)
    monkeypatch.setattr(scaling, "reuse_saved", reused)
    progress = {"actual_model_fits": 2, "reused_model_evaluations": 4}
    result = scaling.run_season(name, parent, prior, output, progress)
    assert result["actual_model_fits"] == expected_fits
    assert result["reused_model_evaluations"] == expected_reuses
    assert len(calls) == expected_fits + expected_reuses
    assert len({call[:4] for call in calls}) == len(calls)
    assert progress == {"actual_model_fits": 2 + expected_fits, "reused_model_evaluations": 4 + expected_reuses}
    assert result["comparison"]["common_match_ids"] == ["later1", "later2"]
    assert result["active_later_weeks"] == 2
    assert len(list((output / name / "models").glob("*.json"))) == expected_fits + expected_reuses
    weeks = json.loads((output / name / "逐周缩放与覆盖.json").read_text(encoding="utf-8"))
    for week in weeks:
        for clock in config["clocks"]:
            row = week["clocks"][clock]
            assert row["scaled_new"]["effective_player_ridge"] == new_base * week["history_matches"]
            assert row["scaled_old"]["effective_player_ridge"] == old_base * week["history_matches"]
            if old_base == new_base:
                assert row["fixed_new"]["alias_of_track"] == ("fixed100" if old_base == 100 else "fixed_old")
                assert "alias_of_track" in row["scaled_new"]
    if old_base == new_base:
        assert result["comparison"]["clock_averaged"]["decomposition"]["interaction_policy_by_base"] == 0
