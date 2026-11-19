"""Ordinary synthetic bookkeeping tests; no statistical model fits or downloads."""

from copy import deepcopy
from datetime import date, datetime
import json
from types import SimpleNamespace

import pytest

import football_model.regularization_experiment as experiment


def match(mid, day, clock="exact", n_stints=1):
    return {"match_id": mid, "match_date": day, "clock": clock,
            "stints": [{"duration_seconds": 5400 / n_stints,
                        "home_players": ["home"], "away_players": ["away"]}
                       for _ in range(n_stints)]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_source(tmp_path, name, content):
    source = tmp_path / "source"
    source.mkdir()
    experiment.write_json(source / experiment.SEASONS[name]["source_file"], content)
    return source


def test_selection_split_strict_january_29_cutoff_excludes_all_gap_days():
    paths = {"early": [match("train", "2017-11-30"), match("val1", "2017-12-01"),
                        match("val2", "2018-01-28"), match("gap1", "2018-01-29"),
                        match("gap2", "2018-01-30"), match("gap3", "2018-01-31"),
                        match("later", "2018-02-03")]}
    original = deepcopy(paths)
    train, validation, cutoff = experiment.selection_split(
        paths, date(2017, 12, 1), date(2018, 1, 29), date(2018, 2, 1))
    assert train == ["train"]
    assert validation == ["val1", "val2"]
    assert cutoff == date(2018, 1, 29)
    assert paths == original


def test_february_1_selection_would_leak_beyond_first_january_29_weekly_fit():
    paths = {"early": [match("train", "2017-11-30"), match("val", "2018-01-28"),
                        match("lateval", "2018-01-31"), match("later", "2018-02-03")]}
    with pytest.raises(ValueError, match="precede the first weekly fit cutoff"):
        experiment.selection_split(paths, date(2017, 12, 1), date(2018, 2, 1), date(2018, 2, 1))


@pytest.mark.parametrize("dates,error", [
    (("2017-12-01", date(2018, 1, 29), date(2018, 2, 1)), "Calendar dates"),
    ((datetime(2017, 12, 1), date(2018, 1, 29), date(2018, 2, 1)), "Calendar dates"),
    ((date(2017, 12, 1), date(2017, 12, 1), date(2018, 2, 1)), "Ordered"),
    ((date(2018, 1, 29), date(2017, 12, 1), date(2018, 2, 1)), "Ordered"),
    ((date(2017, 12, 1), date(2018, 2, 2), date(2018, 2, 1)), "Ordered"),
])
def test_selection_split_date_types_and_order(dates, error):
    with pytest.raises(ValueError, match=error):
        experiment.selection_split({"early": []}, *dates)


@pytest.mark.parametrize("rows", [
    [match("train", "2017-11-30"), match("val", "2018-01-28")],
    [match("train", "2017-11-30"), match("later", "2018-02-03")],
    [match("val", "2018-01-28"), match("later", "2018-02-03")],
])
def test_selection_split_requires_late_windows_and_nonempty_earlier_segments(rows):
    with pytest.raises(ValueError):
        experiment.selection_split({"early": rows}, date(2017, 12, 1), date(2018, 1, 29), date(2018, 2, 1))


def test_duplicate_source_ids_cannot_enter_selection_windows():
    rows = [match("train", "2017-11-30"), match("val", "2018-01-28"),
            match("later", "2018-02-03"), match("later", "2018-02-03")]
    with pytest.raises(ValueError, match="Duplicate match"):
        experiment.selection_split({"early": rows}, date(2017, 12, 1), date(2018, 1, 29), date(2018, 2, 1))


def test_statsbomb_loader_reads_fixed_exact_not_full_or_other_clocks(tmp_path):
    content = {"full": {"exact": [match("full_only", "2015-11-01")]},
               "fixed45": {"exact": [match("fixed", "2015-11-01")],
                           "ceil": [match("rounded", "2015-11-01", "ceil")]}}
    source = write_source(tmp_path, "statsbomb_2015_16", content)
    before = (source / experiment.SEASONS["statsbomb_2015_16"]["source_file"]).read_bytes()
    paths = experiment.load_paths(source, "statsbomb_2015_16")
    assert list(paths) == ["exact"]
    assert [m["match_id"] for m in paths["exact"]] == ["fixed"]
    assert (source / experiment.SEASONS["statsbomb_2015_16"]["source_file"]).read_bytes() == before


def test_wyscout_loader_retains_all_three_clocks_and_matches_by_id_not_order(tmp_path):
    content = {clock: [match(mid, "2017-11-01", clock) for mid in ids]
               for clock, ids in (("early", ("a", "b")), ("middle", ("b", "a")), ("late", ("a", "b")))}
    source = write_source(tmp_path, "wyscout_2017_18", content)
    paths = experiment.load_paths(source, "wyscout_2017_18")
    assert tuple(paths) == experiment.SEASONS["wyscout_2017_18"]["clocks"]
    assert [m["match_id"] for m in paths["middle"]] == ["b", "a"]
    assert all(m["clock"] == c for c, rows in paths.items() for m in rows)


@pytest.mark.parametrize("mutation,error", [("duplicate", "Distinct source match IDs"),
                                           ("missing", "same source cohort"),
                                           ("replacement", "same source cohort")])
def test_loader_rejects_duplicate_or_different_clock_ids(tmp_path, mutation, error):
    content = {c: [match("a", "2017-11-01", c), match("b", "2017-11-02", c)]
               for c in experiment.SEASONS["wyscout_2017_18"]["clocks"]}
    if mutation == "duplicate":
        content["late"].append(deepcopy(content["late"][0]))
    elif mutation == "missing":
        content["late"].pop()
    else:
        content["late"][1]["match_id"] = "different"
    source = write_source(tmp_path, "wyscout_2017_18", content)
    with pytest.raises(ValueError, match=error):
        experiment.load_paths(source, "wyscout_2017_18")


def test_loader_does_not_silently_drop_a_missing_declared_clock(tmp_path):
    source = write_source(tmp_path, "wyscout_2017_18", {"early": [], "late": []})
    with pytest.raises(KeyError, match="middle"):
        experiment.load_paths(source, "wyscout_2017_18")


def test_find_source_uses_latest_complete_local_pair_not_a_partial_directory(tmp_path):
    config = experiment.SEASONS["wyscout_2017_18"]
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    complete = outputs / "第二赛季固定窗口球员探索_20260101"
    older = outputs / "第二赛季固定窗口球员探索_20250101"
    partial = outputs / "第二赛季固定窗口球员探索_20270101"
    for folder in (complete, older, partial):
        folder.mkdir()
        experiment.write_json(folder / config["source_file"], {})
    for folder in (complete, older):
        experiment.write_json(folder / config["completed_file"], {})
    assert experiment.find_source(tmp_path, config) == complete


def fake_coefficients(include_players=True):
    coefficients = {"intercept": 99, "home_advantage": 98, "team_attack:t": 3, "team_defence:t": -4}
    if include_players:
        coefficients.update({f"player_{kind}:{pid}": value
                             for pid, value in (("small", 1), ("edge450", 2), ("edge1350", 4))
                             for kind in ("attack", "defence")})
    return SimpleNamespace(include_players=include_players, players={"small", "edge450", "edge1350"},
                           feature_names=list(coefficients), coefficients=list(coefficients.values()))


def test_coefficient_summary_uses_training_stint_exposure_and_exact_bin_edges():
    history = [{"stints": [
        {"duration_seconds": 449 * 60, "home_players": ["small"], "away_players": []},
        {"duration_seconds": 450 * 60, "home_players": ["edge450"], "away_players": []},
        {"duration_seconds": 1350 * 60, "home_players": [], "away_players": ["edge1350"]}],
        "player_minutes": [{"player_id": "small", "on_pitch_seconds": 99999999}]}]
    before = deepcopy(history)
    result = experiment.coefficient_summary(fake_coefficients(), history)
    bins = result["player_exposure_bins"]
    assert bins["under_450_minutes"]["coefficients"] == 2
    assert bins["under_450_minutes"]["rms"] == 1
    assert bins["450_to_under_1350_minutes"]["rms"] == 2
    assert bins["at_least_1350_minutes"]["rms"] == 4
    assert result["player_group"]["coefficients"] == 6
    assert result["player_group"]["rms"] == pytest.approx(7 ** .5)
    assert result["player_group"]["l2"] == pytest.approx(42 ** .5)
    assert result["player_group"]["max_abs"] == 4
    assert result["team_group"]["l2"] == 5
    assert result["training_exposure_seconds"] == (449 + 450 + 1350) * 60
    assert result["training_matches"] == 1
    assert result["training_stints"] == 3
    assert result["known_players"] == 3
    assert history == before


def test_team_only_coefficient_summary_has_no_fabricated_player_norm():
    result = experiment.coefficient_summary(fake_coefficients(False), [{"stints": []}])
    assert result["player_group"] == {"coefficients": 0, "rms": None, "l2": 0, "max_abs": None}
    assert all(row["coefficients"] == 0 and row["rms"] is None
               for row in result["player_exposure_bins"].values())
    assert result["team_group"]["l2"] == 5


@pytest.mark.parametrize("chosen_ridge,expected_fits", [(100.0, 24), (1000.0, 30)])
def test_run_season_selects_before_late_fit_and_counts_reuse_without_real_training(
        tmp_path, monkeypatch, chosen_ridge, expected_fits):
    name = "wyscout_2017_18"
    config = experiment.SEASONS[name]
    rows = [("train", "2017-11-30"), ("val1", "2017-12-01"), ("val2", "2018-01-28"),
            ("gap1", "2018-01-29"), ("gap2", "2018-01-30"), ("gap3", "2018-01-31"),
            ("late1", "2018-02-01"), ("late2", "2018-02-03"),
            ("late3", "2018-02-05"), ("unknown", "2018-02-06")]
    paths = {clock: [match(mid, day, clock, i + 1) for mid, day in (reversed(rows) if i == 1 else rows)]
             for i, clock in enumerate(config["clocks"])}
    source = write_source(tmp_path, name, paths)
    output = tmp_path / "output"
    output.mkdir()
    calls, events = [], []
    real_select = experiment.select_regularization

    def select(scores):
        assert len(calls) == len(config["clocks"]) * len(experiment.PLAYER_RIDGES)
        assert all(c["validation"] for c in calls)
        events.append("select")
        return real_select(scores)

    def fit(history, target, cutoff, ridge, players, destination, progress):
        validation = destination.name.startswith("validation_")
        clock = next(c for c in config["clocks"] if f"_{c}_" in destination.name)
        assert all(m["clock"] == clock for m in history + target)
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(date.fromisoformat(m["match_date"]) >= cutoff for m in target)
        assert not {m["match_id"] for m in history} & {m["match_id"] for m in target}
        if validation:
            assert cutoff == config["train_before"]
            assert {m["match_id"] for m in history} == {"train"}
            assert {m["match_id"] for m in target} == {"val1", "val2"}
            assert all(date.fromisoformat(m["match_date"]) < date(2018, 1, 29) for m in target)
            assert not events
        else:
            assert events == ["select"]
            assert (output / name / "较早验证选择.json").is_file()
            assert all(date.fromisoformat(m["match_date"]) >= config["target_from"] for m in target)
        calls.append({"cutoff": cutoff, "history": {m["match_id"] for m in history},
                      "target": {m["match_id"] for m in target}, "ridge": ridge,
                      "players": players, "validation": validation, "destination": destination})
        progress["actual_model_fits"] += 1
        experiment.write_json(destination, {"stub": True, "cutoff": cutoff.isoformat()})
        predictions, skipped = [], []
        for m in target:
            if players and m["match_id"] == "unknown":
                skipped.append({"match_id": m["match_id"], "reason": "unknown identity"})
                continue
            predictions.append({"match_id": m["match_id"], "match_date": m["match_date"],
                "n_stints": len(m["stints"]), "actual_home_target": 1, "actual_away_target": 0,
                "expected_home_target": 1.2, "expected_away_target": .8,
                "point_process_loss": 3.0 if not players else 1.0 if ridge == chosen_ridge else 2.0,
                "loss_kind": "point_process_nll_up_to_model_independent_constants"})
        score = {"predictions": predictions, "skipped": skipped, "requested_matches": len(target)}
        return score, {"player_ridge": ridge, "include_players": players,
                       "supported": len(predictions), "requested": len(target), "skipped": skipped}

    monkeypatch.setattr(experiment, "fit_saved", fit)
    monkeypatch.setattr(experiment, "select_regularization", select)
    progress = {"actual_model_fits": 7}
    result = experiment.run_season(name, source, output, progress)
    assert result["actual_model_fits"] == expected_fits
    assert len(calls) == expected_fits
    assert progress["actual_model_fits"] == expected_fits + 7
    assert len(list((output / name / "models").glob("*.json"))) == expected_fits
    assert result["active_weeks"] == 2
    assert result["selection"]["selected_player_ridge"] == chosen_ridge
    assert result["selection"]["validation_requested_matches"] == 2
    assert result["selection"]["first_weekly_cutoff"] == "2018-01-29"
    assert result["comparison"]["common_match_ids"] == ["late1", "late2", "late3"]
    expected_delta = 0 if chosen_ridge == 100 else -1
    assert result["comparison"]["clock_averaged"]["selected_minus_fixed100"] == expected_delta
    weekly = [c for c in calls if not c["validation"]]
    assert {c["cutoff"] for c in weekly} == {date(2018, 1, 29), date(2018, 2, 5)}
    assert all(c["history"] == {"train", "val1", "val2"} and c["target"] == {"late1", "late2"}
               for c in weekly if c["cutoff"] == date(2018, 1, 29))
    assert all(c["history"] == {"train", "val1", "val2", "gap1", "gap2", "gap3", "late1", "late2"}
               and c["target"] == {"late3", "unknown"}
               for c in weekly if c["cutoff"] == date(2018, 2, 5))
    weeks = read_json(output / name / "逐周拟合与覆盖.json")
    for week in weeks:
        for diagnostics in week["clocks"].values():
            assert diagnostics["selected"].get("reused_fixed100_model", False) == (chosen_ridge == 100)
    for clock, coverage in result["coverage"].items():
        assert coverage["team"] == {"requested": 4, "supported": 4, "skipped": 0}
        assert coverage["selected"] == coverage["fixed100"] == {"requested": 4, "supported": 3, "skipped": 1}
        selected_scores = read_json(output / name / f"later_{clock}_selected_predictions.json")
        assert selected_scores["skipped"] == [{"match_id": "unknown", "reason": "unknown identity"}]
    saved = read_json(output / name / "正则对照结果.json")
    assert saved == result
