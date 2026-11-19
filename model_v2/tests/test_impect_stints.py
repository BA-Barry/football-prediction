from copy import deepcopy

import pytest

from football_model.impect_stints import ImpectParseError, parse_match
from football_model.joint_stint_model import _match


def event(index, period, seconds, kind, team=None, player=None, **extra):
    row = {"id": index + 1000, "index": index, "periodId": period,
           "gameTime": {"gameTimeInSec": (period - 1) * 10000 + seconds},
           "actionType": kind, "action": kind, "squadId": team,
           "player": {"id": player} if player is not None else None}
    row.update(extra)
    return row


def change(player, second, before, after, period=2):
    return {"playerId": player, "gameTime": {"gameTimeInSec": (period - 1) * 10000 + second},
            "fromPosition": before, "toPosition": after,
            "exchangedPlayerId": None, "substitutionType": None}


def source():
    metadata = {"id": 1, "iterationId": 743, "homeSquadId": 1, "awaySquadId": 2,
                "scheduledDate": "2024-01-01T23:30:00Z"}
    lineups = {"id": 1, "iterationId": 743, "dateTime": metadata["scheduledDate"]}
    for key, first in (("squadHome", 1), ("squadAway", 101)):
        lineups[key] = {"id": 1 if first == 1 else 2,
                        "players": [{"id": p} for p in range(first, first + 14)],
                        "startingPositions": [{"playerId": p, "position": "CENTRAL_DEFENDER"}
                                              for p in range(first, first + 11)],
                        "substitutions": []}
    events = [event(0, 1, 0, "KICK_OFF", 1, 1),
              event(1, 1, 2820, "FINAL_WHISTLE", duration=7180),
              event(2, 2, 0, "KICK_OFF", 2, 101), event(3, 2, 3000, "FINAL_WHISTLE")]
    return metadata, events, lineups


def add(events, *rows):
    result = sorted(events + list(rows), key=lambda r: (r["periodId"], r["gameTime"]["gameTimeInSec"], r["index"]))
    for index, row in enumerate(result):
        row["index"] = index
    return result


def replacement(lineups, *, second=600, period=2):
    lineups["squadHome"]["substitutions"] += [
        change(1, second, "CENTRAL_DEFENDER", "BANK", period),
        change(12, second, "BANK", "CENTRAL_DEFENDER", period)]


def test_actual_half_ends_not_event_durations_and_unused_zero_minutes():
    result = parse_match(*source())
    assert result["diagnostics"]["total_duration_seconds"] == 5820
    assert result["stints"][1]["start_second"] == 0
    assert result["player_minutes"][0]["minutes"] == 97
    assert result["player_minutes"][-1]["minutes"] == 0
    assert result["match_date"] == "2024-01-01"
    assert result["team_identity_namespace"] == "impect:squad"
    assert result["diagnostics"]["score_matches_independent_source"] is None
    assert "home_xg" not in result["stints"][0]
    assert _match(result, "goals")["home"] == 1
    with pytest.raises(ValueError):
        _match(result, "xg")


@pytest.mark.parametrize("period,offset", [(1, .325), (2, .625)])
def test_nonzero_recorded_kickoff_sets_exposure_start(period, offset):
    metadata, events, lineups = source()
    opening = next(row for row in events if row["periodId"] == period and row["actionType"] == "KICK_OFF")
    opening["action"] = "KICKOFF_WHISTLE"
    opening["gameTime"] = {"gameTimeInSec": (period - 1) * 10000 + offset,
                           "gameTime": f"recorded opening {offset}"}
    result = parse_match(metadata, events, lineups)
    diagnostics = result["diagnostics"]
    stints = [row for row in result["stints"] if row["period"] == period]
    assert stints[0]["start_second"] == pytest.approx(offset)
    assert diagnostics["period_start_seconds"][str(period)] == pytest.approx(offset)
    assert diagnostics["period_duration_seconds"][str(period)] == pytest.approx((2820 if period == 1 else 3000) - offset)
    assert diagnostics["total_duration_seconds"] == pytest.approx(5820 - offset)
    assert all(row["on_pitch_seconds"] == pytest.approx(5820 - offset)
               for row in result["player_minutes"] if row["started"])
    assert sum(row["duration_seconds"] for row in result["stints"]) == pytest.approx(5820 - offset)
    source_opening = diagnostics["opening_whistle_events"][str(period)]
    assert source_opening["event_id"] == opening["id"]
    assert source_opening["event_index"] == opening["index"]
    assert source_opening["source_action"] == "KICKOFF_WHISTLE"
    assert source_opening["source_game_time"] == opening["gameTime"]
    assert diagnostics["period_seconds"][str(period)] == (2820 if period == 1 else 3000)
    assert _match(result, "goals")["stints"][period - 1]["start_second"] == pytest.approx(offset)


def test_later_goal_restart_does_not_reset_period_start():
    metadata, events, lineups = source()
    events[0]["action"] = "KICKOFF_WHISTLE"
    events[0]["gameTime"]["gameTimeInSec"] = .325
    events = add(events, event(10, 1, 600, "GOAL", 1, 1),
                 event(11, 1, 630, "KICK_OFF", 2, 101, action="KICKOFF_WHISTLE"))
    result = parse_match(metadata, events, lineups)
    assert result["diagnostics"]["total_duration_seconds"] == pytest.approx(5819.675)
    assert result["diagnostics"]["event_goals"] == [1, 0]
    assert result["diagnostics"]["period_start_seconds"]["1"] == .325


@pytest.mark.parametrize("kind", ["GOAL", "RED_CARD", "PASS"])
def test_event_before_recorded_kickoff_is_not_silently_discarded(kind):
    metadata, events, lineups = source()
    events[0]["gameTime"]["gameTimeInSec"] = .325
    # Keep the opening first by index: a later indexed event with an earlier
    # clock must also be rejected rather than omitted from the exposure window.
    early = event(10, 1, .1, kind, 1, 1)
    early["index"] = 1
    for row in events[1:]:
        row["index"] += 1
    events.append(early)
    with pytest.raises(ImpectParseError, match="before recorded period kickoff"):
        parse_match(metadata, events, lineups)


def test_valid_match_action_before_later_kickoff_cannot_be_relabelled_as_opening():
    metadata, events, lineups = source()
    events[0]["gameTime"]["gameTimeInSec"] = .325
    events = add(events, event(10, 1, .1, "PASS", 1, 1))
    with pytest.raises(ImpectParseError, match="begin with an identifiable KICK_OFF"):
        parse_match(metadata, events, lineups)


@pytest.mark.parametrize("case", ["missing", "wrong_action", "at_final_whistle", "change_before_opening"])
def test_unidentifiable_or_inconsistent_kickoff_rejected(case):
    metadata, events, lineups = source()
    if case == "missing":
        events = [row for row in events if not (row["periodId"] == 1 and row["actionType"] == "KICK_OFF")]
    elif case == "wrong_action":
        events[0]["action"] = "PASS"
    elif case == "at_final_whistle":
        events[0]["gameTime"]["gameTimeInSec"] = 2820
    elif case == "change_before_opening":
        events[0]["gameTime"]["gameTimeInSec"] = .325
        replacement(lineups, second=.1, period=1)
    with pytest.raises(ImpectParseError):
        parse_match(metadata, events, lineups)


def test_legacy_kickoff_action_label_remains_explicit():
    result = parse_match(*source())
    assert result["diagnostics"]["period_start_seconds"] == {"1": 0, "2": 0}
    assert result["diagnostics"]["opening_whistle_events"]["1"]["opening_label"] == "legacy_KICK_OFF_action_label"


def test_goal_targets_previous_state_before_score_updates():
    metadata, events, lineups = source()
    events = add(events, event(10, 1, 600.125, "GOAL", 1, 1))
    result = parse_match(metadata, events, lineups, expected_score=[1, 0])
    assert result["stints"][0]["home_goals"] == 1
    assert result["stints"][0]["home_score_difference"] == 0
    assert result["stints"][1]["home_score_difference"] == 1
    assert result["diagnostics"]["score_matches_independent_source"] is True


def test_own_goal_credits_opponent_without_counting_twice():
    metadata, events, lineups = source()
    result = parse_match(metadata, add(events, event(10, 1, 900, "OWN_GOAL", 1, 1)), lineups,
                         expected_score=(0, 1))
    assert result["diagnostics"]["event_goals"] == [0, 1]
    assert result["stints"][0]["away_goals"] == 1


@pytest.mark.parametrize("period,second,first_seconds,incoming_seconds", [(2, 600, 3420, 2400), (2, 0, 2820, 3000), (1, 600, 600, 5220)])
def test_atomic_membership_and_halftime_clocks(period, second, first_seconds, incoming_seconds):
    metadata, events, lineups = source()
    replacement(lineups, second=second, period=period)
    result = parse_match(metadata, events, lineups)
    minutes = {p["player_id"]: p["on_pitch_seconds"] for p in result["player_minutes"]}
    assert minutes["impect:player:1"] == first_seconds
    assert minutes["impect:player:12"] == incoming_seconds
    assert all(len(s["home_players"]) == 11 for s in result["stints"])


def test_tactical_change_does_not_make_exposure_boundary():
    metadata, events, lineups = source()
    lineups["squadHome"]["substitutions"] = [change(1, 600, "CENTRAL_DEFENDER", "CENTER_FORWARD")]
    result = parse_match(metadata, events, lineups)
    assert len(result["stints"]) == 2
    assert result["diagnostics"]["status_events"]["tactical_position_changes"] == 1


def test_goal_membership_exact_tie_rejected_without_cross_list_order():
    metadata, events, lineups = source()
    replacement(lineups)
    events = add(events, event(10, 2, 600, "GOAL", 2, 101))
    with pytest.raises(ImpectParseError, match="cross-list order"):
        parse_match(metadata, events, lineups)


def test_red_card_removes_once_with_coherent_bank_exit():
    metadata, events, lineups = source()
    lineups["squadHome"]["substitutions"] = [change(1, 600, "CENTRAL_DEFENDER", "BANK")]
    events = add(events, event(10, 2, 600, "RED_CARD", 1, 1))
    result = parse_match(metadata, events, lineups)
    assert len(result["stints"][-1]["home_players"]) == 10
    assert result["stints"][-1]["home_man_difference"] == -1
    assert result["diagnostics"]["status_events"]["on_pitch_dismissals"] == 1


def test_bench_red_does_not_reduce_field():
    metadata, events, lineups = source()
    result = parse_match(metadata, add(events, event(10, 2, 600, "RED_CARD", 1, 12)), lineups)
    assert all(len(s["home_players"]) == 11 for s in result["stints"])
    assert len(result["stints"]) == 2
    assert result["diagnostics"]["status_events"]["off_pitch_dismissals"] == 1


def test_unreconciled_red_card_rejected():
    metadata, events, lineups = source()
    with pytest.raises(ImpectParseError, match="BANK departure"):
        parse_match(metadata, add(events, event(10, 2, 600, "RED_CARD", 1, 1)), lineups)


def test_duration_kpis_sum_positions_and_report_not_correct_difference():
    metadata, events, lineups = source()
    kpis = {"matchId": 1, "squadHome": {"id": 1, "players": []},
            "squadAway": {"id": 2, "players": []}}
    for key, first in (("squadHome", 1), ("squadAway", 101)):
        kpis[key]["players"] = [{"id": p, "position": "CENTRAL_DEFENDER", "playDuration": 5820}
                                for p in range(first, first + 11)]
    kpis["squadHome"]["players"][0]["playDuration"] = 3000
    kpis["squadHome"]["players"].append({"id": 1, "position": "CENTER_FORWARD", "playDuration": 2819.5})
    result = parse_match(metadata, events, lineups, kpis)
    audit = result["diagnostics"]["player_duration_audit"]
    assert audit["max_absolute_difference_seconds"] == .5
    assert audit["all_active_players_have_duration"]
    assert result["player_minutes"][0]["on_pitch_seconds"] == 5820
    assert audit["source_position_rows"] == 23


def test_source_objects_unchanged():
    values = source()
    before = deepcopy(values)
    parse_match(*values)
    assert values == before


def test_missing_active_duration_is_explicit_not_assumed_zero():
    metadata, events, lineups = source()
    kpis = {"matchId": 1, "squadHome": {"id": 1, "players": []},
            "squadAway": {"id": 2, "players": []}}
    result = parse_match(metadata, events, lineups, kpis)
    audit = result["diagnostics"]["player_duration_audit"]
    assert not audit["all_active_players_have_duration"]
    assert len(audit["missing_active_player_ids"]) == 22
    assert audit["compared_players"] == 0
    assert len(audit["unused_roster_without_duration_ids"]) == 6
    assert audit["max_absolute_difference_seconds"] is None
    assert parse_match(metadata, events, lineups)["diagnostics"]["player_duration_audit"]["provided"] is False


def test_goal_and_tactical_change_same_clock_preserve_membership():
    metadata, events, lineups = source()
    lineups["squadHome"]["substitutions"] = [change(1, 600, "CENTRAL_DEFENDER", "CENTER_FORWARD")]
    result = parse_match(metadata, add(events, event(10, 2, 600, "GOAL", 1, 1)), lineups)
    assert result["diagnostics"]["event_goals"] == [1, 0]


def test_conflicting_critical_action_labels_rejected_without_expected_score():
    metadata, events, lineups = source()
    events = add(events, event(10, 1, 600, "SHOT", 1, 1, action="GOAL"))
    with pytest.raises(ImpectParseError, match="critical action/actionType"):
        parse_match(metadata, events, lineups)


def test_scheduled_utc_date_is_used_not_local_calendar_date():
    metadata, events, lineups = source()
    metadata["scheduledDate"] = "2024-01-02T07:30:00+08:00"
    result = parse_match(metadata, events, lineups)
    assert result["match_date"] == "2024-01-01"
    assert result["scheduled_kickoff_utc"] == "2024-01-01T23:30:00+00:00"
    assert result["source_scheduled_datetime_text"] == metadata["scheduledDate"]


def test_goal_at_final_whistle_included_and_stints_contiguous():
    metadata, events, lineups = source()
    result = parse_match(metadata, add(events, event(10, 2, 3000, "GOAL", 2, 101)), lineups)
    assert result["diagnostics"]["event_goals"] == [0, 1]
    for period in (1, 2):
        stints = [row for row in result["stints"] if row["period"] == period]
        assert stints[0]["start_second"] == 0
        assert stints[-1]["end_second"] == result["diagnostics"]["period_seconds"][str(period)]
        assert sum(row["duration_seconds"] for row in stints) == stints[-1]["end_second"]
        assert all(a["end_second"] == b["start_second"] for a, b in zip(stints, stints[1:]))


@pytest.mark.parametrize("case", ["duplicate_event_id", "duplicate_event_index", "duplicate_start",
                                  "duplicate_roster", "wrong_lineup", "bad_clock", "extra_period",
                                  "missing_whistle", "wrong_position", "unknown_actor", "off_pitch_goal",
                                  "goal_zero", "score_conflict", "late_change", "unpaired_departure",
                                  "reentry", "duplicate_change", "model_clock_inversion", "unknown_sendoff"])
def test_source_conflicts_rejected(case):
    metadata, events, lineups = source()
    expected = None
    if case == "duplicate_event_id":
        events[1]["id"] = events[0]["id"]
    elif case == "duplicate_event_index":
        events[1]["index"] = events[0]["index"]
    elif case == "duplicate_start":
        lineups["squadHome"]["startingPositions"][-1] = lineups["squadHome"]["startingPositions"][0]
    elif case == "duplicate_roster":
        lineups["squadAway"]["players"][0]["id"] = 1
    elif case == "wrong_lineup":
        lineups["id"] = 2
    elif case == "bad_clock":
        events[2]["gameTime"]["gameTimeInSec"] = 2700
    elif case == "extra_period":
        events[-1]["periodId"] = 3
    elif case == "missing_whistle":
        events.pop()
    elif case == "wrong_position":
        lineups["squadHome"]["substitutions"] = [change(1, 600, "CENTER_FORWARD", "LEFT_WINGER")]
    elif case in {"unknown_actor", "off_pitch_goal", "goal_zero"}:
        events = add(events, event(10, 1, 0 if case == "goal_zero" else 600, "GOAL", 1,
                                   999 if case == "unknown_actor" else 12 if case == "off_pitch_goal" else 1))
    elif case == "score_conflict":
        expected = (1, 0)
    elif case == "late_change":
        replacement(lineups, second=3001)
    elif case == "unpaired_departure":
        lineups["squadHome"]["substitutions"] = [change(1, 600, "CENTRAL_DEFENDER", "BANK")]
    elif case == "reentry":
        replacement(lineups)
        lineups["squadHome"]["substitutions"] += [change(12, 1200, "CENTRAL_DEFENDER", "BANK"),
                                                  change(1, 1200, "BANK", "CENTRAL_DEFENDER")]
    elif case == "duplicate_change":
        replacement(lineups)
        lineups["squadHome"]["substitutions"].append(deepcopy(lineups["squadHome"]["substitutions"][0]))
    elif case == "model_clock_inversion":
        events = add(events, event(10, 1, 600, "GOAL", 1, 1))
        events[1]["gameTime"]["gameTimeInSec"] = 2900
    elif case == "unknown_sendoff":
        events = add(events, event(10, 1, 600, "SECOND_YELLOW", 1, 1))
    with pytest.raises(ImpectParseError):
        parse_match(metadata, events, lineups, expected_score=expected)


@pytest.mark.parametrize("case", ["match", "squad", "unknown_player", "duplicate_position", "negative_duration"])
def test_bad_duration_audit_inputs_rejected(case):
    metadata, events, lineups = source()
    kpis = {"matchId": 1, "squadHome": {"id": 1, "players": [
        {"id": 1, "position": "CENTRAL_DEFENDER", "playDuration": 5820}]},
            "squadAway": {"id": 2, "players": []}}
    if case == "match":
        kpis["matchId"] = 99
    elif case == "squad":
        kpis["squadHome"]["id"] = 99
    elif case == "unknown_player":
        kpis["squadHome"]["players"][0]["id"] = 999
    elif case == "duplicate_position":
        kpis["squadHome"]["players"].append(deepcopy(kpis["squadHome"]["players"][0]))
    elif case == "negative_duration":
        kpis["squadHome"]["players"][0]["playDuration"] = -1
    with pytest.raises(ImpectParseError):
        parse_match(metadata, events, lineups, kpis)
