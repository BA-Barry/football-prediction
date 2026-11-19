"""Synthetic full timing paths only; these tests perform no fitting."""

from copy import deepcopy
import json

import pytest

from football_model.joint_stint_model import _match
from football_model.statsbomb_fixed_window import FixedWindowCropError, crop_fixed_window
from football_model.timing_precision_experiment import measure_match


def source_path():
    h = [f"statsbomb:player:{i}" for i in range(1, 12)]
    a = [f"statsbomb:player:{i}" for i in range(21, 32)]
    stints, attributions = [], []
    score = [0, 0]

    def stint(period, start, end, home, away, goal=None):
        targets = [0, 0]
        ids = []
        if goal is not None:
            side, eid = goal
            targets[side] = 1
            ids.append(eid)
            attributions.append({"event_id": eid, "period": period, "second": end,
                "scoring_team_id": (1, 2)[side], "score_before": list(score),
                "home_players": list(home), "away_players": list(away), "actor_on_assumed_pitch": True})
        stints.append({"stint_id": f"source-stint-{len(stints)}", "period": period,
            "start_second": start, "end_second": end, "duration_seconds": end-start,
            "home_players": list(home), "away_players": list(away),
            "start_home_goals": score[0], "start_away_goals": score[1],
            "home_score_difference": score[0]-score[1], "home_man_difference": len(home)-len(away),
            "home_goals": targets[0], "away_goals": targets[1], "goal_event_ids": ids})
        for i in (0, 1):
            score[i] += targets[i]

    stint(1, 0., 2700., h, a, (0, "at-2700"))
    stint(1, 2700., 2750., h, a, (1, "stoppage-goal"))
    # First-half stoppage dismissal and substitution carry into half two.
    h_after = h[1:]
    a_after = a[1:] + ["statsbomb:player:32"]
    stint(1, 2750., 2800., h_after, a_after)
    stint(2, 0., 100., h_after, a_after, (0, "second-half-goal"))
    stint(2, 100., 2800., h_after, a_after, (1, "crossing-goal"))
    player_minutes = []
    for team, pids in ((1, h + ["statsbomb:player:12"]), (2, a + ["statsbomb:player:32", "statsbomb:player:33"])):
        for pid in pids:
            seconds = sum(r["duration_seconds"] for r in stints if pid in r["home_players"] + r["away_players"])
            player_minutes.append({"team_id": team, "player_id": pid, "player_name": pid,
                "started": pid in h + a, "on_pitch_seconds": seconds, "minutes": seconds/60})
    return {"match_id": "statsbomb:match:1", "source_match_id": 1, "match_date": "2015-08-08",
        "home_team_id": 1, "away_team_id": 2, "stints": stints, "player_minutes": player_minutes,
        "diagnostics": {"scenario": "floor", "event_goals": list(score), "period_seconds": {"1": 2800., "2": 2800.},
            "total_duration_seconds": 5600., "all_source_goals_preserved": True,
            "goal_attribution": attributions,
            "actor_conflicts": [{"event_id": "boundary-actor", "period": 1, "second": 2700., "is_goal": True},
                {"event_id": "post-window-actor", "period": 1, "second": 2750., "is_goal": True}],
            "substitutions": [{"event_id": "sub", "period": 1, "original_second": 2761.2,
                "scenario_second": 2760., "clipped_at_true_half_end": False}],
            "tie_policy": "source timing policy", "source_on_pitch_card_membership": {"red": True}}}


def test_inclusive_boundary_crossing_goals_and_exact_window_exposure():
    result = crop_fixed_window(source_path())
    assert sum(s["duration_seconds"] for s in result["stints"]) == 5400
    assert result["diagnostics"]["full_goals"] == [2, 2]
    assert result["diagnostics"]["window_goals"] == [2, 0]
    assert result["diagnostics"]["window_excluded_goals"] == [0, 2]
    assert result["diagnostics"]["excluded_goal_event_ids"] == ["stoppage-goal", "crossing-goal"]
    assert result["stints"][0]["goal_event_ids"] == ["at-2700"]
    crossing = result["stints"][-1]
    assert (crossing["start_second"], crossing["end_second"], crossing["duration_seconds"]) == (100., 2700., 2600.)
    assert crossing["goal_event_ids"] == []
    assert crossing["home_goals"] == crossing["away_goals"] == 0
    assert crossing["source_end_second"] == 2800.
    assert _match(result, "goals")["match_id"] == result["match_id"]


def test_second_half_keeps_full_first_half_score_lineup_and_man_state():
    source = source_path()
    result = crop_fixed_window(source)
    h2 = next(s for s in result["stints"] if s["period"] == 2)
    assert [h2["start_home_goals"], h2["start_away_goals"]] == [1, 1]
    assert h2["home_score_difference"] == 0  # Not 1-0 after removing the stoppage target.
    assert h2["home_man_difference"] == -1
    assert len(h2["home_players"]) == 10
    assert "statsbomb:player:1" not in h2["home_players"]
    assert "statsbomb:player:32" in h2["away_players"]


def test_no_source_mutation_zero_exposure_and_original_clock_provenance():
    source = source_path()
    before = deepcopy(source)
    result = crop_fixed_window(source)
    assert source == before
    assert result["match_id"] == source["match_id"]
    assert [p["player_id"] for p in result["player_minutes"]] == [p["player_id"] for p in source["player_minutes"]]
    minutes = {p["player_id"]: p for p in result["player_minutes"]}
    assert minutes["statsbomb:player:12"]["minutes"] == 0
    assert minutes["statsbomb:player:33"]["on_pitch_seconds"] == 0
    assert minutes["statsbomb:player:1"]["minutes"] == 45
    assert minutes["statsbomb:player:32"]["minutes"] == 45
    assert "not full-match" in minutes["statsbomb:player:32"]["minutes_basis"]
    d = result["diagnostics"]
    assert d["source_diagnostics"] == source["diagnostics"]
    assert d["substitutions"] == source["diagnostics"]["substitutions"]
    assert [c["event_id"] for c in d["actor_conflicts"]] == ["boundary-actor"]
    assert d["source_period_seconds"] == {"1": 2800., "2": 2800.}
    assert d["period_seconds"] == {"1": 2700., "2": 2700.}
    json.dumps(result, allow_nan=False)
    result["diagnostics"]["source_diagnostics"]["substitutions"][0]["original_second"] = 0
    assert source == before


@pytest.mark.parametrize("change", ["missing_id", "missing_clock", "wrong_clock", "side_mismatch", "unsplit", "short_half", "gap", "orphan_attribution"])
def test_refuse_ambiguous_or_incomplete_goal_paths(change):
    source = source_path()
    if change == "missing_id":
        source["stints"][0]["goal_event_ids"] = []
    elif change == "missing_clock":
        del source["diagnostics"]["goal_attribution"][0]["second"]
    elif change == "wrong_clock":
        source["diagnostics"]["goal_attribution"][0]["second"] = 2600.
    elif change == "side_mismatch":
        source["diagnostics"]["goal_attribution"][0]["scoring_team_id"] = 2
    elif change == "unsplit":
        source["stints"][0]["end_second"] = 2800.
        source["stints"][0]["duration_seconds"] = 2800.
    elif change == "short_half":
        source["diagnostics"]["period_seconds"]["1"] = 2699.
    elif change == "gap":
        source["stints"][0]["start_second"] = 1.
        source["stints"][0]["duration_seconds"] -= 1.
    else:
        goal = deepcopy(source["diagnostics"]["goal_attribution"][0])
        goal["event_id"] = "not-in-stints"
        source["diagnostics"]["goal_attribution"].append(goal)
    with pytest.raises(FixedWindowCropError):
        crop_fixed_window(source)


def test_actual_measure_match_shape_with_zero_goals_is_supported():
    meta = {"match_id": 9, "match_date": "2015-08-08", "home_team": {"home_team_id": 1},
        "away_team": {"away_team_id": 2}, "home_score": 0, "away_score": 0}
    lineups, events = [], []
    for team, start in ((1, 1), (2, 21)):
        lineups.append({"team_id": team, "lineup": [{"player_id": i, "player_name": str(i)} for i in range(start, start+12)]})
        events.append({"period": 1, "timestamp": "00:00:00.000", "type": {"name": "Starting XI"},
            "team": {"id": team}, "tactics": {"lineup": [{"player": {"id": i}} for i in range(start, start+11)]}})
    for period in (1, 2):
        events.extend([{"period": period, "timestamp": "00:00:00.000", "type": {"name": "Half Start"}},
            {"period": period, "timestamp": "00:46:00.000", "type": {"name": "Half End"}}])
    events.sort(key=lambda r: (r["period"], r["timestamp"]))
    for index, event in enumerate(events, 1):
        event.update(index=index, id=f"event-{index}")
    measured = measure_match(meta, events, lineups)
    assert measured["common_supported"]
    for path in measured["scenarios"].values():
        cropped = crop_fixed_window(path)
        assert len(cropped["stints"]) == 2
        assert cropped["diagnostics"]["all_source_goals_preserved"]
        assert cropped["diagnostics"]["window_goals"] == [0, 0]
