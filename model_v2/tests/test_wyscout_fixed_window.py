"""Synthetic fixed-window scenarios; no real-data fitting or clock inference."""

from copy import deepcopy

import pytest

from football_model.joint_stint_model import _match
from football_model.wyscout_fixed_window import SCENARIOS, parse_match, substitution_clock


def raw(second, period=1, player=1, team=10, marked=(), kind="Pass", subevent=None):
    return {"eventId": 10 if kind == "Shot" else 3 if kind == "Free Kick" else 8,
            "eventName": kind, "subEventId": subevent if subevent is not None else 100 if kind == "Shot" else 85,
            "matchId": 1, "matchPeriod": "1H" if period == 1 else "2H", "eventSec": second,
            "playerId": player, "teamId": team, "tags": [{"id": tag} for tag in marked]}


def fixture_stream(extra=(), home_subs=(), away_subs=(), red_minutes=None, tail=2800):
    red_minutes = red_minutes or {}
    teams = {}
    for team, first, side, subs in ((10, 1, "home", home_subs), (20, 21, "away", away_subs)):
        teams[str(team)] = {"teamId": team, "side": side, "score": 0, "scoreHT": 0,
            "formation": {"lineup": [{"playerId": pid, "redCards": str(red_minutes.get(pid, 0))}
                                      for pid in range(first, first + 11)],
                          "bench": [{"playerId": pid, "redCards": str(red_minutes.get(pid, 0))}
                                    for pid in range(first + 11, first + 15)],
                          "substitutions": list(subs)}}
    metadata = {"wyId": 1, "dateutc": "2017-08-11 18:45:00", "duration": "Regular", "teamsData": teams}
    events = [raw(1), raw(1, period=2)] + deepcopy(list(extra)) + [raw(tail), raw(tail, period=2)]
    events.sort(key=lambda e: (e["matchPeriod"], e["eventSec"]))
    for index, event in enumerate(events, 1):
        event["id"] = index
        tags = {t["id"] for t in event["tags"]}
        awarded = (20 if event["teamId"] == 10 else 10) if 102 in tags else (
            event["teamId"] if 101 in tags and event["eventName"] != "Save attempt" else None)
        if awarded is not None:
            teams[str(awarded)]["score"] += 1
            teams[str(awarded)]["scoreHT"] += event["matchPeriod"] == "1H"
    return metadata, events


def sub(minute, outgoing=1, incoming=12):
    return {"minute": minute, "playerOut": outgoing, "playerIn": incoming}


@pytest.mark.parametrize("minute,early,middle,late,period", [
    (0, 0, 0, 0, 1), (1, 0, 30, 60, 1), (45, 2640, 2670, 2700, 1),
    (46, 0, 30, 60, 2), (90, 2640, 2670, 2700, 2), (91, 2700, 2730, 2760, 2),
    (95, 2940, 2970, 3000, 2),
])
def test_predeclared_clock_rules_do_not_clip_or_infer_period_from_actions(minute, early, middle, late, period):
    assert substitution_clock(minute, "early") == (period, early)
    assert substitution_clock(minute, "middle") == (period, middle)
    assert substitution_clock(minute, "late") == (period, late)


def test_fixed_exposure_is_exactly_5400_not_the_last_action_and_source_is_unchanged():
    metadata, events = fixture_stream([raw(100, marked=(101,), kind="Shot")], home_subs=[sub(30)])
    originals = deepcopy((metadata, events))
    result = parse_match(metadata, events, player_names={1: "Starter", "wyscout:player:12": "Replacement"})
    assert result["common_supported"]
    assert (metadata, events) == originals
    for scenario, path in result["scenarios"].items():
        assert _match(path, "goals")["match_id"] == "wyscout:match:1"
        assert path["diagnostics"]["total_duration_seconds"] == 5400
        assert sum(s["duration_seconds"] for s in path["stints"] if s["period"] == 1) == 2700
        assert sum(s["duration_seconds"] for s in path["stints"] if s["period"] == 2) == 2700
        assert path["diagnostics"]["window_goals"] == [1, 0]
        assert path["diagnostics"]["full_source_goals"] == [1, 0]
        assert path["diagnostics"]["observed_last_event_seconds_not_whistle"] == {"1": 2800, "2": 2800}
        assert all("xg" not in k for s in path["stints"] for k in s)
        assert all(p.startswith("wyscout:player:") for s in path["stints"] for p in s["home_players"] + s["away_players"])
        minutes = {p["player_id"]: p for p in path["player_minutes"]}
        assert minutes["wyscout:player:1"]["player_name"] == "Starter"
        assert minutes["wyscout:player:12"]["player_name"] == "Replacement"
        assert "wyscout:player:13" not in minutes  # Unused bench identity is not treated as an appearance.


def test_first_half_stoppage_goal_and_red_are_not_targets_but_set_second_half_initial_state():
    metadata, events = fixture_stream([
        raw(2710, marked=(101,), kind="Shot"), raw(2720, player=2, marked=(1703,), kind="Foul"),
        raw(20, period=2, player=21, team=20, marked=(101,), kind="Shot"),
    ], red_minutes={2: 46})
    result = parse_match(metadata, events)
    assert result["common_supported"]
    for path in result["scenarios"].values():
        assert path["diagnostics"]["full_source_goals"] == [1, 1]
        assert path["diagnostics"]["window_goals"] == [0, 1]
        second_half = next(s for s in path["stints"] if s["period"] == 2)
        assert second_half["start_second"] == 0
        assert second_half["home_score_difference"] == 1
        assert second_half["home_man_difference"] == -1
        assert "wyscout:player:2" not in second_half["home_players"]


def test_seventy_five_stoppage_goals_remain_in_full_source_and_second_half_state_not_target():
    metadata, events = fixture_stream([raw(2701 + i, marked=(101,), kind="Shot") for i in range(75)])
    result = parse_match(metadata, events)
    assert result["common_supported"]
    assert result["diagnostics"]["full_source_goals"] == [75, 0]
    assert result["diagnostics"]["window_goals"] == [0, 0]
    for path in result["scenarios"].values():
        second_half = next(s for s in path["stints"] if s["period"] == 2)
        assert second_half["home_score_difference"] == 75
        assert sum(s["home_goals"] for s in path["stints"]) == 0


def test_second_half_post_window_goals_and_false_red_are_recorded_not_applied_or_rejected():
    metadata, events = fixture_stream([raw(2710, period=2, marked=(101,), kind="Shot"),
                                      raw(2720, period=2, player=2, marked=(1701,), kind="Foul")])
    result = parse_match(metadata, events)
    assert result["common_supported"]
    assert result["diagnostics"]["full_source_goals"] == [1, 0]
    assert result["diagnostics"]["window_goals"] == [0, 0]
    assert len(result["diagnostics"]["post_second_half_red_issues"]) == 1
    for path in result["scenarios"].values():
        assert all(s["home_man_difference"] == 0 for s in path["stints"])
        assert any(r["is_goal"] for r in path["diagnostics"]["post_second_half_records_not_applied"])
        assert any(r["is_red"] for r in path["diagnostics"]["post_second_half_records_not_applied"])


def test_goal_at_2700_is_retained_and_same_second_goal_precedes_substitution():
    metadata, events = fixture_stream([raw(2700, marked=(101,), kind="Shot"),
                                      raw(2701, player=12, marked=(101,), kind="Shot")], home_subs=[sub(45)])
    result = parse_match(metadata, events)
    assert result["common_supported"]
    late = result["scenarios"]["late"]
    goals = late["diagnostics"]["goal_attribution"]
    assert late["diagnostics"]["window_goals"] == [1, 0]
    assert late["diagnostics"]["full_source_goals"] == [2, 0]
    assert "wyscout:player:1" in goals[0]["home_players"]
    assert "wyscout:player:12" not in goals[0]["home_players"]
    assert "wyscout:player:12" in goals[1]["home_players"]
    second_half = next(s for s in late["stints"] if s["period"] == 2)
    assert second_half["home_score_difference"] == 2


def test_same_second_raw_card_precedes_substitution_and_invalid_replacement_is_rejected():
    metadata, events = fixture_stream([raw(1800, player=1, marked=(1701,), kind="Foul")],
                                      home_subs=[sub(30)], red_minutes={1: 30})
    result = parse_match(metadata, events)
    assert not result["common_supported"]
    assert set(result["scenarios"]) == {"early", "middle"}
    assert "Illegal" in result["rejects"]["late"][0]
    assert all(s["home_man_difference"] == 0 for s in result["scenarios"]["early"]["stints"])


def test_goal_scorer_outside_approximate_lineup_is_diagnostic_and_not_a_reason_to_drop_goal():
    metadata, events = fixture_stream([raw(1790, player=12, marked=(101,), kind="Shot")], home_subs=[sub(30)])
    result = parse_match(metadata, events)
    assert result["common_supported"]
    assert result["scenarios"]["late"]["diagnostics"]["actor_conflicts"][0]["is_goal"]
    assert result["scenarios"]["late"]["diagnostics"]["window_goals"] == [1, 0]
    assert result["scenarios"]["early"]["diagnostics"]["actor_conflicts"] == []


def test_keeper_duplicate_and_own_goal_perspectives_are_not_double_counted():
    metadata, events = fixture_stream([raw(100, marked=(101,), kind="Free Kick", subevent=33),
        raw(101, player=21, team=20, marked=(101,), kind="Save attempt"),
        raw(200, player=2, marked=(102,)), raw(201, player=21, team=20, marked=(101,), kind="Save attempt")])
    result = parse_match(metadata, events)
    assert result["common_supported"]
    assert result["diagnostics"]["full_source_goals"] == [1, 1]
    for path in result["scenarios"].values():
        assert path["diagnostics"]["window_goals"] == [1, 1]
        assert len(path["diagnostics"]["goal_attribution"]) == 2


@pytest.mark.parametrize("period,second", [(1, 100), (1, 2750), (2, 100), (2, 2700)])
def test_unsupported_raw_red_affecting_either_window_rejects_whole_match(period, second):
    metadata, events = fixture_stream([raw(second, period=period, player=2, marked=(1701,), kind="Foul")])
    result = parse_match(metadata, events)
    assert not result["common_supported"]
    assert not result["scenarios"]
    assert set(result["rejects"]) == set(SCENARIOS)
    assert "unsupported" in result["rejects"]["middle"][0]


def test_metadata_red_missing_raw_clock_rejects_if_not_explicitly_post90():
    assert not parse_match(*fixture_stream(red_minutes={2: 65}))["common_supported"]
    result = parse_match(*fixture_stream(red_minutes={2: 93}))
    assert result["common_supported"]
    assert result["diagnostics"]["post_second_half_red_issues"][0]["reported_minute"] == 93


def test_off_pitch_card_never_reduces_man_count_or_adds_unused_player_to_lineup():
    args = fixture_stream([raw(1000, player=13, marked=(1701,), kind="Foul")], red_minutes={13: 17})
    result = parse_match(*args)
    assert result["common_supported"]
    for path in result["scenarios"].values():
        assert all(s["home_man_difference"] == 0 for s in path["stints"])
        assert not any(p["player_id"] == "wyscout:player:13" for p in path["player_minutes"])
        assert path["diagnostics"]["off_pitch_reds_no_man_change"][0]["man_count_changed"] is False


def test_reported_source_order_is_kept_for_same_minute_substitution_chain():
    good = parse_match(*fixture_stream(home_subs=[sub(30), sub(30, outgoing=12, incoming=13)]))
    assert good["common_supported"]
    assert all(any("wyscout:player:13" in s["home_players"] for s in p["stints"])
               for p in good["scenarios"].values())
    bad = parse_match(*fixture_stream(home_subs=[sub(30, outgoing=12, incoming=13), sub(30)]))
    assert not bad["common_supported"]
    assert not bad["scenarios"]


def test_post_second_half_substitution_is_not_clipped_back_into_target_window():
    # Even the impossible outgoing identity does not affect any future window.
    result = parse_match(*fixture_stream(home_subs=[sub(95, outgoing=999, incoming=12)]))
    assert result["common_supported"]
    for path in result["scenarios"].values():
        assert not any("wyscout:player:12" in s["home_players"] for s in path["stints"])
        assert path["diagnostics"]["substitution_clocks"][0]["assumed_second"] > 2700


def test_short_half_tail_and_full_time_goal_disagreement_reject_without_repair():
    short = parse_match(*fixture_stream(tail=2699))
    assert not short["common_supported"]
    assert "shorter" in short["rejects"]["early"][0]
    metadata, events = fixture_stream()
    metadata["teamsData"]["10"]["score"] = 1
    missing = parse_match(metadata, events)
    assert not missing["common_supported"]
    assert "FT" in missing["rejects"]["early"][0]


def test_metadata_half_time_disagreement_is_recorded_without_fabricating_a_goal():
    metadata, events = fixture_stream([raw(100, marked=(101,), kind="Shot")])
    metadata["teamsData"]["10"]["scoreHT"] = 0
    result = parse_match(metadata, events)
    assert result["common_supported"]
    assert result["diagnostics"]["source_halftime_agrees"] is False
    assert result["diagnostics"]["full_source_halftime_goals"] == [1, 0]


def test_duplicate_identity_or_nonunique_starting_eleven_is_not_repaired():
    metadata, events = fixture_stream()
    metadata["teamsData"]["10"]["formation"]["lineup"][1]["playerId"] = 1
    result = parse_match(metadata, events)
    assert not result["common_supported"]
    assert "Duplicate roster" in result["rejects"]["early"][0]
