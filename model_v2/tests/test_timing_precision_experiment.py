"""Ordinary synthetic replay tests; no held-out data or real model training."""

from copy import deepcopy
from datetime import date
import json

import numpy as np
import pytest

from football_model.joint_stint_model import JointStintIntensityModel, UnknownStintIdentityError, _match
import football_model.timing_precision_experiment as timing_module
from football_model.timing_precision_experiment import (
    SCENARIOS, TimingReplayError, measure_match, replay_match, rounded_second,
    run, training_path_intensities,
)


def fixture_stream(changes=(), half_end=120):
    """A tiny valid StatsBomb source with exact clocks and two full lineups."""
    metadata = {"match_id": 1, "match_date": "2015-08-08", "home_team": {"home_team_id": 1},
                "away_team": {"away_team_id": 2}, "home_score": 0, "away_score": 0}
    lineups = [{"team_id": t, "lineup": [{"player_id": p, "player_name": f"Player {p}"}
                for p in range(1 if t == 1 else 21, 15 if t == 1 else 35)]} for t in (1, 2)]
    rows = []
    for t, first in ((1, 1), (2, 21)):
        rows.append({"period": 1, "second": 0, "type": {"name": "Starting XI"}, "team": {"id": t},
                     "tactics": {"lineup": [{"player": {"id": p}} for p in range(first, first + 11)]}})
    for period in (1, 2):
        rows.append({"period": period, "second": 0, "type": {"name": "Half Start"}})
    for change in changes:
        row = deepcopy(change)
        row.setdefault("period", 1)
        row.setdefault("team", {"id": 1})
        row.setdefault("player", {"id": 1})
        if row["type"]["name"] == "Shot":
            row.setdefault("shot", {"statsbomb_xg": .2, "outcome": {"name": "Saved"}})
            if row["shot"]["outcome"]["name"] == "Goal":
                metadata["home_score" if row["team"]["id"] == 1 else "away_score"] += 1
        elif row["type"]["name"] == "Own Goal Against":
            metadata["away_score" if row["team"]["id"] == 1 else "home_score"] += 1
        rows.append(row)
    for period in (1, 2):
        rows.append({"period": period, "second": half_end, "type": {"name": "Half End"}})
    rows.sort(key=lambda row: (row["period"], row["second"]))
    for index, row in enumerate(rows, 1):
        seconds = row.pop("second")
        row["timestamp"] = f"{int(seconds // 3600):02}:{int(seconds // 60) % 60:02}:{seconds % 60:09.6f}"
        row["index"], row["id"] = index, f"event-{index}"
    return metadata, rows, lineups


def substitution(second, outgoing=1, incoming=12, period=1):
    return {"period": period, "second": second, "type": {"name": "Substitution"},
            "player": {"id": outgoing}, "substitution": {"replacement": {"id": incoming}}}


def shot(second, player, goal=False, period=1):
    return {"period": period, "second": second, "type": {"name": "Shot"}, "player": {"id": player},
            "shot": {"statsbomb_xg": .2, "outcome": {"name": "Goal" if goal else "Saved"}}}


@pytest.mark.parametrize("scenario,second,expected", [
    ("floor", 29.99, 0), ("nearest_half_up", 29.99, 0), ("nearest_half_up", 30, 60),
    ("nearest_half_up", 90, 120), ("nearest_half_up", 150, 180),
    ("ceil", 60, 60), ("ceil", 60.001, 120), ("exact", 14.25, 14.25),
])
def test_fixed_rounding_is_period_local_and_half_up_not_bankers(scenario, second, expected):
    assert rounded_second(second, scenario, 200) == (expected, False)


def test_rounding_clips_only_beyond_true_half_end_and_records_it():
    assert rounded_second(119, "ceil", 119.5) == (119.5, True)
    assert rounded_second(119, "floor", 119.5) == (60, False)
    assert rounded_second(119.5, "exact", 119.5) == (119.5, False)


def test_all_goals_and_exposure_survive_while_off_pitch_scorers_are_measured():
    metadata, events, lineups = fixture_stream([shot(12, 1), substitution(14), shot(20, 12, True),
                                              shot(119, 2, True, period=2)])
    raw_before = deepcopy(events)
    measured = measure_match(metadata, events, lineups)
    assert events == raw_before
    assert measured["common_supported"]
    assert set(measured["scenarios"]) == set(SCENARIOS)
    for scenario, path in measured["scenarios"].items():
        assert path["diagnostics"]["event_goals"] == [2, 0]
        assert path["diagnostics"]["total_duration_seconds"] == 240
        assert sum(s["home_goals"] + s["away_goals"] for s in path["stints"]) == 2
        assert measured["comparisons"][scenario]["duration_difference_seconds"] == 0
        assert measured["comparisons"][scenario]["goal_difference"] == 0
        assert not measured["comparisons"][scenario]["starter_ids_changed"]
        assert all("home_xg" not in s and "away_xg" not in s for s in path["stints"])
    assert measured["comparisons"]["floor"]["shot_actor_conflicts"] == 1
    assert measured["comparisons"]["floor"]["goal_actor_conflicts"] == 0
    assert measured["comparisons"]["ceil"]["goal_actor_conflicts"] == 1
    assert len(measured["comparisons"]["ceil"]["changed_goal_lineup_event_ids"]) == 1
    assert measured["comparisons"]["ceil"]["max_abs_player_minute_difference"] == pytest.approx(46 / 60)


def test_goals_keep_raw_clock_score_and_goal_before_lineup_attribution():
    args = fixture_stream([shot(10, 1, True), substitution(14), shot(20, 12, True)])
    measured = measure_match(*args)
    for path in measured["scenarios"].values():
        goals = path["diagnostics"]["goal_attribution"]
        assert [(g["second"], g["score_before"]) for g in goals] == [(10, [0, 0]), (20, [1, 0])]
        stints = path["stints"]
        assert next(s for s in stints if s["end_second"] == 10)["home_score_difference"] == 0
        assert next(s for s in stints if s["end_second"] == 20)["home_score_difference"] == 1


def test_substitution_rounding_does_not_drop_stoppage_time_goals():
    args = fixture_stream([substitution(119, period=2), shot(121, 12, True, period=2)], half_end=125)
    measured = measure_match(*args)
    assert measured["common_supported"]
    for path in measured["scenarios"].values():
        assert path["diagnostics"]["period_seconds"] == {"1": 125, "2": 125}
        assert path["diagnostics"]["total_duration_seconds"] == 250
        assert path["diagnostics"]["goal_attribution"][0]["second"] == 121
    # Incoming player has positive exposure in this case; the next test clips
    # a different late appearance to true Half End rather than inventing time.


def test_half_end_clipping_can_remove_positive_exposure_identity_without_removing_starter_identity():
    args = fixture_stream([substitution(124, period=2)], half_end=125)
    measured = measure_match(*args)
    assert measured["common_supported"]
    assert measured["comparisons"]["ceil"]["clip_count"] == 1
    assert measured["comparisons"]["ceil"]["positive_exposure_player_ids_lost"] == ["statsbomb:player:12"]
    assert not measured["comparisons"]["ceil"]["starter_ids_changed"]


def test_original_source_index_is_explicit_tie_policy_for_rounded_chain():
    args = fixture_stream([substitution(14), substitution(25, outgoing=12, incoming=13)])
    measured = measure_match(*args)
    assert measured["common_supported"]
    for scenario in ("floor", "nearest_half_up", "ceil"):
        path = measured["scenarios"][scenario]
        first_positive_after_both = next(s for s in path["stints"] if "statsbomb:player:13" in s["home_players"])
        assert "statsbomb:player:1" not in first_positive_after_both["home_players"]
        assert "statsbomb:player:12" not in first_positive_after_both["home_players"]


def test_temporary_exit_interleaving_rejects_whole_scenario_instead_of_fixing_it():
    changes = [{"second": 10, "type": {"name": "Player Off"}},
               {"second": 13, "type": {"name": "Player On"}}, substitution(14)]
    measured = measure_match(*fixture_stream(changes))
    assert not measured["common_supported"]
    assert "floor" in measured["failures"]
    assert measured["comparisons"] == {}
    with pytest.raises(TimingReplayError, match="Player Off"):
        replay_match(*fixture_stream(changes), scenario="floor")


def test_dismissal_status_changes_are_rejected_but_exact_card_time_is_never_rounded():
    red = {"second": 20, "type": {"name": "Bad Behaviour"},
           "bad_behaviour": {"card": {"name": "Red Card"}}}
    measured = measure_match(*fixture_stream([substitution(14), red]))
    assert not measured["common_supported"]
    assert "ceil" in measured["failures"]
    assert "dismissal status" in measured["failures"]["ceil"]


def test_unaffected_red_card_changes_man_state_at_original_second():
    red = {"second": 35, "type": {"name": "Foul Committed"}, "player": {"id": 2},
           "foul_committed": {"card": {"name": "Second Yellow"}}}
    measured = measure_match(*fixture_stream([substitution(14), red]))
    assert measured["common_supported"]
    for path in measured["scenarios"].values():
        assert any(s["end_second"] == 35 and s["home_man_difference"] == 0 for s in path["stints"])
        assert all(s["home_man_difference"] == -1 for s in path["stints"] if s["period"] == 2)


def synthetic_model_for_path(path):
    """Construct a fitted-parameter-shaped object without invoking an optimizer."""
    model = JointStintIntensityModel(include_players=True, response_kind="goals")
    model.before_date = date(2015, 12, 1)
    model.reference_match_ids = [path["match_id"]]
    model.teams = {path["home_team_id"], path["away_team_id"]}
    model.players = {p for s in path["stints"] for c in ("home_players", "away_players") for p in s[c]}
    model.reference_fractions = {team: {p["player_id"]: p["on_pitch_seconds"] / path["diagnostics"]["total_duration_seconds"]
        for p in path["player_minutes"] if p["team_id"] == team and p["on_pitch_seconds"] > 0} for team in model.teams}
    model.feature_names = ["intercept", "home_advantage", "own_score_difference", "own_man_difference"]
    model.feature_names += [f"team_{kind}:{t}" for kind in ("attack", "defence") for t in sorted(model.teams)]
    model.feature_names += [f"player_{kind}:{p}" for kind in ("attack", "defence") for p in sorted(model.players)]
    model.feature_index = {name: i for i, name in enumerate(model.feature_names)}
    model.coefficients = np.linspace(-.25, .3, len(model.feature_names))
    return model


def test_training_path_linear_predictor_is_numerically_identical_to_model_design():
    path = replay_match(*fixture_stream([shot(10, 1, True), substitution(14), shot(20, 12, True)]))
    model = synthetic_model_for_path(path)
    evaluated = training_path_intensities(model, path)
    design, exposure, _, _ = model._design([_match(path, "goals")])
    expected_rates = np.exp(np.asarray(design @ model.coefficients).ravel())
    measured_rates = np.array([r[col] for r in evaluated["stints"] for col in ("home_intensity_per90", "away_intensity_per90")])
    np.testing.assert_allclose(measured_rates, expected_rates, rtol=1e-13, atol=1e-13)
    assert evaluated["path_integrated_home_intensity"] == pytest.approx((expected_rates * exposure)[::2].sum())
    assert evaluated["path_integrated_away_intensity"] == pytest.approx((expected_rates * exposure)[1::2].sum())
    with pytest.raises(ValueError, match="at or after"):
        model.predict_match(path)  # Production interface still refuses a training-window date.


def test_training_path_requires_fitted_match_and_never_defaults_unknown_players_to_zero():
    path = replay_match(*fixture_stream([substitution(14)]))
    model = synthetic_model_for_path(path)
    model.players.remove("statsbomb:player:12")
    with pytest.raises(UnknownStintIdentityError, match="Unknown exact-path"):
        training_path_intensities(model, path)
    model.reference_match_ids = []
    with pytest.raises(ValueError, match="fitted earlier"):
        training_path_intensities(model, path)


def test_run_reads_only_earlier_event_files_and_does_not_require_later_events(tmp_path):
    args = fixture_stream([substitution(14), shot(20, 12, True)])
    metadata, events, lineups = args
    source = tmp_path / "data/statsbomb/premier-league-2015-16"
    source.mkdir(parents=True)
    later = deepcopy(metadata)
    later["match_id"], later["match_date"] = 999, "2015-12-01"
    (source / "matches_2_27.json").write_text(json.dumps([metadata, later]), encoding="utf-8")
    (source / "events_1.json").write_text(json.dumps(events), encoding="utf-8")
    (source / "lineups_1.json").write_text(json.dumps(lineups), encoding="utf-8")
    # Later event/lineup files deliberately do not exist.
    output = run(tmp_path, fit_models=False)
    summary = json.loads((output / "分钟精度测量汇总.json").read_text(encoding="utf-8"))
    assert summary["requested_matches"] == summary["common_matches"] == 1
    assert summary["later_event_files_read"] == 0
    assert json.loads((output / "读取事件比赛ID.json").read_text(encoding="utf-8")) == [1]
    assert not (output / "models").exists()
    report = (output / "换人分钟精度测量说明.md").read_text(encoding="utf-8")
    assert "原始1场解析失败0场" in report
    assert "后1场事件文件未读取" in report
    assert "本次没有进行模型拟合" in report


def test_measurement_report_handles_no_substitutions_without_inventing_zero_error_estimate(tmp_path):
    metadata, events, lineups = fixture_stream([shot(20, 1, True)])
    source = tmp_path / "data/statsbomb/premier-league-2015-16"
    source.mkdir(parents=True)
    for name, payload in (("matches_2_27.json", [metadata]), ("events_1.json", events), ("lineups_1.json", lineups)):
        (source / name).write_text(json.dumps(payload), encoding="utf-8")
    output = run(tmp_path, fit_models=False)
    summary = json.loads((output / "全部较早场次换人秒移动汇总.json").read_text(encoding="utf-8"))
    assert summary["source_substitutions"] == 0
    assert summary["scenarios"]["floor"]["mean_abs_shift_seconds"] is None
    assert "| 向下取整 | — | — | — | 0 |" in (output / "换人分钟精度测量说明.md").read_text(encoding="utf-8")


def test_fit_interface_makes_exactly_four_fixed_fits_and_pairs_supported_exact_paths(monkeypatch, tmp_path):
    first = measure_match(*fixture_stream([shot(5, 1, True), substitution(124, period=2)], half_end=125))
    second_args = list(fixture_stream([shot(8, 1, True)], half_end=125))
    second_args[0]["match_id"] = 2
    second = measure_match(*second_args)
    common_paths = {s: [first["scenarios"][s], second["scenarios"][s]] for s in SCENARIOS}
    fitted, saved = [], []

    class RecordedModel:
        def __init__(self, **kwargs):
            assert kwargs == {"include_players": True, "response_kind": "goals", "player_ridge": 10.0}

        def fit(self, matches, before_date):
            # No optimizer: exercise the public orchestration with synthetic
            # coefficients. The late ceil debut intentionally has zero exposure.
            assert before_date == date(2015, 12, 1)
            self.__dict__.update(synthetic_model_for_path(matches[0]).__dict__)
            self.reference_match_ids = [m["match_id"] for m in matches]
            fitted.append(list(self.reference_match_ids))
            return self

        def save(self, path):
            saved.append(path.name)

    monkeypatch.setattr(timing_module, "JointStintIntensityModel", RecordedModel)
    result = timing_module.fit_sensitivity(common_paths, tmp_path / "parameters")
    assert len(fitted) == len(saved) == result["model_fits"] == 4
    assert set(saved) == {f"{s}_goals_players_ridge10.json" for s in SCENARIOS}
    assert result["fixed_player_ridge"] == 10
    assert result["common_exact_evaluation_match_ids"] == [second["match_id"]]
    assert result["skipped_exact_path_evaluations"]["ceil"][0]["match_id"] == first["match_id"]
    for scenario in SCENARIOS:
        assert [m["match_id"] for m in result["scenarios"][scenario]["matches"]] == [second["match_id"]]
    assert result["scenarios"]["exact"]["mean_abs_integrated_intensity_difference"] == 0
    assert result["scenarios"]["exact"]["max_abs_relative_stint_intensity_change"] == 0
