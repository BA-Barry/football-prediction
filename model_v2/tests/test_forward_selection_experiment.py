"""Synthetic weekly validation, cache inputs and reuse bookkeeping tests."""

from copy import deepcopy
from datetime import date
import json
from types import SimpleNamespace

import pytest

import football_model.forward_selection_experiment as forward


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
    return {c: [match(mid, day, c, i + 1) for mid, day in (reversed(rows) if i == 1 else rows)]
            for i, c in enumerate(forward.SEASONS["wyscout_2017_18"]["clocks"])}


@pytest.mark.parametrize("skip_one_clock", [False, True])
def test_weekly_validation_uses_monday_history_gap_exclusion_and_equal_match_loss(
        tmp_path, monkeypatch, skip_one_clock):
    data = paths()
    original = deepcopy(data)
    config = forward.SEASONS["wyscout_2017_18"]
    calls = []

    def fake_fit(history, target, cutoff, ridge, players, destination, progress):
        assert players is True
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(config["train_before"] <= date.fromisoformat(m["match_date"]) < config["selection_before"] for m in target)
        assert all(m["clock"] == target[0]["clock"] for m in history)
        if cutoff == date(2017, 11, 27):
            assert {m["match_id"] for m in history} == {"old"}
            assert {m["match_id"] for m in target} == {"val1a", "val1b"}
        calls.append((cutoff, ridge, target[0]["clock"], [m["match_id"] for m in target]))
        progress["actual_model_fits"] += 1
        forward.write_json(destination, {"stub": True})
        loss = (0 if cutoff == date(2017, 11, 27) else 8) if ridge == 10 else {100: 6, 1000: 5, 10000: 7}[ridge]
        score = predictions(target, loss)
        if skip_one_clock and target[0]["clock"] == "late":
            excluded = [p for p in score["predictions"] if p["match_id"] == "val2"]
            score["predictions"] = [p for p in score["predictions"] if p["match_id"] != "val2"]
            score["skipped"] = [{"match_id": p["match_id"], "reason": "unknown"} for p in excluded]
        return score, {"requested": len(target), "supported": len(score["predictions"])}

    monkeypatch.setattr(forward, "fit_saved", fake_fit)
    progress = {"actual_model_fits": 9, "reused_model_evaluations": 2}
    selected = forward.weekly_validation(data, config, tmp_path / "validation", progress)
    assert selected["active_weeks"] == 3
    assert selected["validation_requested_matches"] == 4
    assert selected["actual_model_fits"] == len(calls) == 36
    assert progress == {"actual_model_fits": 45, "reused_model_evaluations": 2}
    assert selected["validation_first_fit_cutoff"] == "2017-11-27"
    assert selected["validation_last_fit_cutoff"] == "2018-01-22"
    assert selected["first_weekly_cutoff"] == "2018-01-29"
    assert selected["selected_player_ridge"] == 10
    expected_ids = ["val1a", "val1b", "val3"] if skip_one_clock else ["val1a", "val1b", "val2", "val3"]
    assert selected["common_match_ids"] == expected_ids
    assert selected["candidate_metrics"]["10"]["mean_clock_match_loss"] == pytest.approx(8 / 3 if skip_one_clock else 4)
    assert selected["candidate_metrics"]["1000"]["mean_clock_match_loss"] == 5
    # Equal-week averaging would pick 1000 (10's weekly mean is 16/3),
    # while the required equal-match aggregation selects 10 on four IDs.
    assert data == original
    assert len(list((tmp_path / "validation" / "models").glob("*.json"))) == 36
    assert all("gap" not in mid for _, _, _, mids in calls for mid in mids)
    assert selected["coverage"]["late"]["10"]["skipped"] == int(skip_one_clock)


def cached_model(history, cutoff=date(2018, 1, 29), ridge=100, players=True):
    ordered = sorted(history, key=lambda m: (m["match_date"], m["match_id"]))
    parameters = {"before_date": cutoff.isoformat(), "include_players": players,
        "response_kind": "goals", "team_ridge": 1, "context_ridge": 1, "player_ridge": ridge,
        "reference_match_ids": [m["match_id"] for m in ordered], "n_matches": len(ordered),
        "n_stints": sum(len(m["stints"]) for m in ordered),
        "training_min_date": ordered[0]["match_date"], "training_max_date": ordered[-1]["match_date"]}
    model = SimpleNamespace(parameters=lambda: deepcopy(parameters), include_players=players,
        players={"p", "q"}, feature_names=["team_attack:t", "player_attack:p", "player_defence:p",
                                          "player_attack:q", "player_defence:q"],
        coefficients=[0, .1, -.1, .2, -.2], optimizer_diagnostics={"success": True})
    return model, parameters


@pytest.mark.parametrize("field,value", [
    ("before_date", "2018-02-05"), ("include_players", False), ("response_kind", "xg"),
    ("team_ridge", 2), ("context_ridge", 2), ("player_ridge", 1000),
    ("reference_match_ids", ["wrong"]), ("reference_match_ids", ["b", "a"]),
    ("n_matches", 999), ("n_stints", 999),
    ("training_min_date", "2017-01-01"), ("training_max_date", "2018-01-29"),
])
def test_cache_requires_exact_history_order_cutoff_flags_target_and_regularization(field, value):
    history = [match("b", "2017-12-02"), match("a", "2017-12-01")]
    model, parameters = cached_model(history)
    forward.validate_cached_model(model, history, date(2018, 1, 29), 100, True)
    parameters[field] = value
    with pytest.raises(ValueError, match=f"mismatch: {field}"):
        forward.validate_cached_model(model, history, date(2018, 1, 29), 100, True)


def test_cache_rejects_future_or_duplicate_history_and_calendar_type():
    history = [match("a", "2017-12-01")]
    model, _ = cached_model(history)
    with pytest.raises(ValueError, match="strictly earlier"):
        forward.validate_cached_model(model, history + [match("late", "2018-01-29")], date(2018, 1, 29), 100, True)
    with pytest.raises(ValueError, match="Distinct history"):
        forward.validate_cached_model(model, history + history, date(2018, 1, 29), 100, True)
    with pytest.raises(ValueError, match="calendar cutoff"):
        forward.validate_cached_model(model, history, "2018-01-29", 100, True)


def test_reuse_saved_scores_and_copies_without_incrementing_actual_fits(tmp_path, monkeypatch):
    history = [match("a", "2017-12-01")]
    target = [match("later", "2018-02-03")]
    model, _ = cached_model(history)
    model.predict_match = lambda m: predictions([m], 1.8)["predictions"][0]
    model.save = lambda destination: forward.write_json(destination, model.parameters())
    monkeypatch.setattr(forward.JointStintIntensityModel, "load", lambda path: model)
    progress = {"actual_model_fits": 3, "reused_model_evaluations": 4}
    destination = tmp_path / "copy.json"
    score, diagnostic = forward.reuse_saved(tmp_path / "source.json", history, target, date(2018, 1, 29),
        100, True, destination, progress)
    assert destination.is_file()
    assert progress == {"actual_model_fits": 3, "reused_model_evaluations": 5}
    assert score["requested_matches"] == 1
    assert diagnostic["saved_parameter_and_witness_roundtrip"] is True
    assert diagnostic["coefficient_summary"]["training_matches"] == 1


def test_direct_selection_comparison_uses_global_all_clock_intersection():
    data = paths()
    late = {c: [m for m in rows if m["match_id"].startswith("later")] for c, rows in data.items()}
    previous = {c: predictions(rows, 2) for c, rows in late.items()}
    weekly = {c: predictions(rows, 1.8) for c, rows in late.items()}
    weekly["late"]["predictions"] = weekly["late"]["predictions"][:1]
    result = forward.compare_selected_tracks(previous, weekly)
    assert result["common_match_ids"] == ["later1"]
    assert result["mean_clock_weekly_minus_previous_static_loss"] == pytest.approx(-.2)
    assert all(row["previous_static_selected_on_common"]["matches"] == 1 for row in result["per_clock"].values())
    assert all(row["weekly_minus_previous_static_loss"] == pytest.approx(-.2) for row in result["per_clock"].values())
    weekly["late"]["predictions"] = []
    empty = forward.compare_selected_tracks(previous, weekly)
    assert empty["common_match_ids"] == []
    assert empty["mean_clock_weekly_minus_previous_static_loss"] is None
    assert all(row["paired_matches"] == [] and row["weekly_selected_on_common"] is None for row in empty["per_clock"].values())


def test_direct_selection_comparison_checks_targets_paths_and_duplicate_ids():
    data = paths()
    previous = {c: predictions([m for m in rows if m["match_id"].startswith("later")]) for c, rows in data.items()}
    for mutation, error in (("target", "same match date, target"), ("path", "same actual state path"),
                            ("duplicate", "distinct nonempty match IDs")):
        weekly = deepcopy(previous)
        if mutation == "target":
            weekly["late"]["predictions"][0]["actual_home_target"] = 2
        elif mutation == "path":
            weekly["late"]["predictions"][0]["n_stints"] += 1
        else:
            weekly["late"]["predictions"].append(deepcopy(weekly["late"]["predictions"][0]))
        with pytest.raises(ValueError, match=error):
            forward.compare_selected_tracks(previous, weekly)


@pytest.mark.parametrize("selected,expected_fits,expected_reuses", [(100, 0, 12), (1000, 0, 18), (10, 6, 12)])
def test_later_reuse_selected100_alias_parent_selected_or_new_fits(
        tmp_path, monkeypatch, selected, expected_fits, expected_reuses):
    name = "wyscout_2017_18"
    config, data = forward.SEASONS[name], paths()
    source = tmp_path / "source"
    source.mkdir()
    forward.write_json(source / config["source_file"], data)
    parent, output = tmp_path / "parent", tmp_path / "output"
    parent.mkdir()
    output.mkdir()
    season_parent = parent / name
    season_parent.mkdir()
    parent_models = season_parent / "models"
    parent_models.mkdir()
    train_ids, val_ids, first_cutoff = forward.selection_split(data, config["train_before"],
        config["selection_before"], config["target_from"])
    forward.write_json(season_parent / "分段与来源.json", {"source_directory": str(source),
        "source_file": config["source_file"], "training_match_ids": train_ids,
        "validation_match_ids": val_ids, "first_weekly_cutoff": first_cutoff.isoformat()})
    old_result = {"seasons": {name: {"source_directory": str(source), "source_matches": len(data["early"]),
        "selection": {"selected_player_ridge": 1000, "validation_requested_matches": len(val_ids), "common_match_ids": val_ids}}}}
    summary = {"player_group": {"rms": .1}}
    for cutoff in ("2018-01-29", "2018-02-05"):
        for clock in config["clocks"]:
            for kind in ("team", "fixed100", "selected"):
                forward.write_json(parent_models / f"{cutoff}_{clock}_{kind}.json", {})
    for clock, rows in data.items():
        later = [m for m in rows if m["match_id"].startswith("later")]
        forward.write_json(season_parent / f"later_{clock}_selected_predictions.json", predictions(later, 2))
    forward.write_json(season_parent / "逐周拟合与覆盖.json", [{"clocks": {
        c: {"selected": {"coefficient_summary": summary}} for c in config["clocks"]}}])
    calls, events = [], []

    def weekly_validation(source_paths, settings, destination, progress):
        assert source_paths == data
        destination.mkdir()
        events.append("selected")
        return {"selected_player_ridge": selected, "validation_requested_matches": len(val_ids),
                "common_match_ids": val_ids, "active_weeks": 3}

    def score_call(history, target, cutoff, ridge, players, destination, progress, cached=None):
        assert events == ["selected"]
        assert (output / name / "多周较早验证选择.json").is_file()
        assert all(date.fromisoformat(m["match_date"]) < cutoff for m in history)
        assert all(date.fromisoformat(m["match_date"]) >= config["target_from"] for m in target)
        assert all(date.fromisoformat(m["match_date"]) >= cutoff for m in target)
        assert all(m["clock"] == target[0]["clock"] for m in history)
        calls.append({"cached": cached, "cutoff": cutoff, "ridge": ridge, "players": players})
        if cached:
            assert cached.parent == parent_models
            progress["reused_model_evaluations"] += 1
        else:
            progress["actual_model_fits"] += 1
        forward.write_json(destination, {"stub": True})
        score = predictions(target, 3 if not players else 2 if ridge == 1000 else 1 if ridge == selected else 2)
        return score, {"coefficient_summary": summary, "player_ridge": ridge, "include_players": players}

    def reused(cached, history, target, cutoff, ridge, players, destination, progress):
        return score_call(history, target, cutoff, ridge, players, destination, progress, cached)

    monkeypatch.setattr(forward, "weekly_validation", weekly_validation)
    monkeypatch.setattr(forward, "fit_saved", score_call)
    monkeypatch.setattr(forward, "reuse_saved", reused)
    progress = {"actual_model_fits": 7, "reused_model_evaluations": 9}
    result = forward.run_season(name, parent, old_result, output, progress)
    assert result["actual_model_fits"] == expected_fits
    assert result["reused_model_evaluations"] == expected_reuses
    assert len(calls) == expected_fits + expected_reuses
    assert progress == {"actual_model_fits": 7 + expected_fits, "reused_model_evaluations": 9 + expected_reuses}
    assert result["selected_vs_prior"]["common_match_ids"] == ["later1", "later2"]
    assert result["selected_vs_prior"]["mean_clock_weekly_minus_previous_static_loss"] == (0 if selected == 1000 else -1)
    assert result["comparison"]["clock_averaged"]["selected_minus_fixed100"] == (-1 if selected == 10 else 0)
    weeks = json.loads((output / name / "逐周较晚拟合与覆盖.json").read_text(encoding="utf-8"))
    assert all(values["selected"].get("reused_fixed100_alias", False) == (selected == 100)
               for week in weeks for values in week["clocks"].values())
    assert len(list((output / name / "models").glob("*.json"))) == expected_fits + expected_reuses


def test_find_parent_prefers_latest_completed_result(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    complete, partial = outputs / "球员收缩验证_20260101", outputs / "球员收缩验证_20270101"
    complete.mkdir()
    partial.mkdir()
    forward.write_json(complete / forward.PARENT_RESULT, {})
    assert forward.find_parent(tmp_path) == complete
    assert forward.find_parent(tmp_path, complete) == complete
    with pytest.raises(ValueError, match="completed regularization"):
        forward.find_parent(tmp_path, partial)


@pytest.mark.parametrize("field,value", [
    ("source_directory", "different_source"), ("source_file", "different_file.json"),
    ("training_match_ids", []), ("validation_match_ids", []),
    ("first_weekly_cutoff", "2018-02-05"), ("source_matches", 999),
])
def test_parent_source_or_segment_changes_rejected_before_any_validation_fit(tmp_path, monkeypatch, field, value):
    name, data = "wyscout_2017_18", paths()
    config = forward.SEASONS[name]
    source, parent, output = tmp_path / "source", tmp_path / "parent", tmp_path / "output"
    source.mkdir()
    parent.mkdir()
    output.mkdir()
    (parent / name).mkdir()
    forward.write_json(source / config["source_file"], data)
    train_ids, val_ids, first_cutoff = forward.selection_split(data, config["train_before"],
        config["selection_before"], config["target_from"])
    metadata = {"source_directory": str(source), "source_file": config["source_file"],
        "training_match_ids": train_ids, "validation_match_ids": val_ids,
        "first_weekly_cutoff": first_cutoff.isoformat()}
    old = {"source_directory": str(source), "source_matches": len(data["early"])}
    if field == "source_matches":
        old[field] = value
    else:
        metadata[field] = str(tmp_path / value) if field == "source_directory" else value
    forward.write_json(parent / name / "分段与来源.json", metadata)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A source/segment mismatch must be rejected before fitting")

    monkeypatch.setattr(forward, "weekly_validation", unexpected_fit)
    with pytest.raises(ValueError, match="exact source cohort and date splits"):
        forward.run_season(name, parent, {"seasons": {name: old}}, output,
            {"actual_model_fits": 0, "reused_model_evaluations": 0})
