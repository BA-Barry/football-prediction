from copy import deepcopy

import pytest

from football_model.statsbomb_stints import StatsBombParseError, parse_match


def event(index, period, seconds, kind, team=1, player=None, **extra):
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(int(minutes), 60)
    value = {"id": f"event-{index}", "index": index, "period": period,
             "timestamp": f"{hours:02}:{minute:02}:{sec:06.3f}",
             "type": {"name": kind}, "team": {"id": team}}
    if player is not None:
        value["player"] = {"id": player}
    value.update(extra)
    return value


def source():
    metadata = {"match_id": 1, "match_date": "2024-01-01", "kick_off": "15:00:00.000",
                "home_team": {"home_team_id": 1}, "away_team": {"away_team_id": 2},
                "home_score": 0, "away_score": 0}
    rosters = [{"team_id": team, "lineup": [
        {"player_id": pid, "player_name": f"Player {pid}"}
        for pid in range(1 if team == 1 else 101, 15 if team == 1 else 115)]}
        for team in (1, 2)]
    events = [event(1, 1, 0, "Starting XI", tactics={"lineup": [
        {"player": {"id": pid}} for pid in range(1, 12)]}),
        event(2, 1, 0, "Starting XI", team=2, tactics={"lineup": [
            {"player": {"id": pid}} for pid in range(101, 112)]}),
        event(3, 1, 0, "Half Start"), event(4, 1, 2820, "Half End"),
        event(5, 2, 0, "Half Start"), event(6, 2, 3000, "Half End")]
    return metadata, events, rosters


def insert_events(base, additions):
    # Real source indexes govern ties; construct sequential synthetic indexes.
    values = sorted(base + additions, key=lambda e: (e["period"], e["timestamp"], e["index"]))
    for index, value in enumerate(values, 1):
        value["index"] = index
    return values


def test_two_halves_keep_stoppage_time_and_do_not_invent_absolute_timestamps():
    result = parse_match(*source())
    assert len(result["stints"]) == 2
    assert result["diagnostics"]["total_duration_seconds"] == 5820
    assert result["stints"][1]["start_second"] == 0
    assert result["player_minutes"][0]["minutes"] == 97
    assert not any(key in result for key in ("kickoff", "observed_at", "reference_asof"))
    assert result["player_minutes"][-1]["minutes"] == 0


def test_goal_belongs_to_previous_score_state_and_current_lineup():
    metadata, events, rosters = source()
    metadata["home_score"] = 1
    events = insert_events(events, [event(20, 1, 600.25, "Shot", player=1,
                                         shot={"statsbomb_xg": .4, "outcome": {"name": "Goal"}}),
                                    event(21, 1, 700, "Shot", team=2, player=101,
                                          shot={"statsbomb_xg": .1, "outcome": {"name": "Saved"}})])
    result = parse_match(metadata, events, rosters)
    first, second = result["stints"][:2]
    assert first["end_second"] == 600.25
    assert first["home_goals"] == 1 and first["home_xg"] == .4
    assert first["home_score_difference"] == 0
    assert second["home_score_difference"] == 1 and second["away_xg"] == .1


def test_substitution_and_temporary_absence_integrate_actual_player_exposure():
    metadata, events, rosters = source()
    events = insert_events(events, [
        event(20, 1, 600, "Player Off", player=1),
        event(21, 1, 660, "Player On", player=1),
        event(22, 2, 1200, "Substitution", player=1,
              substitution={"replacement": {"id": 12}})])
    result = parse_match(metadata, events, rosters)
    minutes = {p["player_id"]: p["minutes"] for p in result["player_minutes"]}
    assert minutes["statsbomb:player:1"] == 66
    assert minutes["statsbomb:player:12"] == 30
    assert any(row["home_man_difference"] == -1 for row in result["stints"])
    assert sum(row["duration_seconds"] for row in result["stints"]) == 5820


def test_substituting_temporarily_absent_player_is_not_a_second_removal():
    metadata, events, rosters = source()
    events = insert_events(events, [event(20, 1, 600, "Player Off", player=1),
        event(21, 1, 660, "Substitution", player=1,
              substitution={"replacement": {"id": 12}})])
    result = parse_match(metadata, events, rosters)
    assert result["stints"][1]["home_man_difference"] == -1
    assert result["stints"][2]["home_man_difference"] == 0


def test_red_card_is_applied_once_and_bench_red_does_not_reduce_field_count():
    metadata, events, rosters = source()
    events = insert_events(events, [
        event(20, 1, 600, "Foul Committed", player=1,
              foul_committed={"card": {"name": "Second Yellow"}}),
        event(21, 1, 600, "Bad Behaviour", player=1,
              bad_behaviour={"card": {"name": "Red Card"}}),
        event(22, 1, 700, "Bad Behaviour", team=2, player=114,
              bad_behaviour={"card": {"name": "Red Card"}})])
    result = parse_match(metadata, events, rosters)
    assert result["stints"][-1]["home_man_difference"] == -1
    assert result["diagnostics"]["status_events"] == {
        "on_pitch_dismissals": 1, "duplicate_dismissal_events": 1, "off_pitch_dismissals": 1}


def test_linked_own_goal_views_are_counted_once_without_shot_xg():
    metadata, events, rosters = source()
    metadata["away_score"] = 1
    against = event(20, 1, 600, "Own Goal Against", player=1, related_events=["event-21"])
    scored = event(21, 1, 600, "Own Goal For", team=2, related_events=["event-20"])
    result = parse_match(metadata, insert_events(events, [against, scored]), rosters)
    assert sum(row["away_goals"] for row in result["stints"]) == 1
    assert result["diagnostics"]["away_xg"] == 0


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_index", "clock_order", "missing_xi",
                                          "bad_xi", "extra_period", "missing_half_end", "score_mismatch"])
def test_ambiguous_source_states_fail_explicitly(change):
    metadata, events, rosters = source()
    if change == "duplicate_id":
        events[1]["id"] = events[0]["id"]
    elif change == "duplicate_index":
        events[1]["index"] = events[0]["index"]
    elif change == "clock_order":
        events[3]["timestamp"] = "00:00:00.000"
        events[2]["timestamp"] = "00:01:00.000"
    elif change == "missing_xi":
        events.pop(0)
    elif change == "bad_xi":
        events[0]["tactics"]["lineup"][0]["player"]["id"] = 999
    elif change == "extra_period":
        events[-1]["period"] = 3
    elif change == "missing_half_end":
        events.pop()
    else:
        metadata["home_score"] = 1
    with pytest.raises(StatsBombParseError):
        parse_match(metadata, events, rosters)


@pytest.mark.parametrize("action", [
    event(20, 1, 600, "Player On", player=1),
    event(20, 1, 600, "Substitution", player=1, substitution={"replacement": {"id": 2}}),
    event(20, 1, 600, "Shot", player=12, shot={"statsbomb_xg": .1}),
    event(20, 1, 600, "Shot", player=1, shot={"statsbomb_xg": float("nan")}),
    event(20, 1, 600, "Own Goal For", team=2),
])
def test_illegal_returns_replacements_and_unattributable_targets_fail(action):
    metadata, events, rosters = source()
    with pytest.raises(StatsBombParseError):
        parse_match(metadata, insert_events(events, [deepcopy(action)]), rosters)


def test_same_timestamp_events_use_source_index_for_substitution_then_shot():
    metadata, events, rosters = source()
    events = insert_events(events, [event(20, 1, 600, "Substitution", player=1,
                                         substitution={"replacement": {"id": 12}}),
        event(21, 1, 600, "Shot", player=12,
              shot={"statsbomb_xg": .2, "outcome": {"name": "Saved"}})])
    result = parse_match(metadata, events, rosters)
    assert "statsbomb:player:12" in result["stints"][1]["home_players"]
    assert result["stints"][1]["home_xg"] == .2


def test_irrelevant_trailing_receipt_clock_anomaly_is_recorded_not_used_or_repaired():
    metadata, events, rosters = source()
    # Preserve source index order: a pass near half-time followed by a receipt
    # with the provider's reset clock. Neither event affects model stints.
    pass_event = event(20, 1, 2819, "Pass", player=1)
    receipt = event(21, 1, .2, "Ball Receipt*", player=2)
    events[3:3] = [pass_event, receipt]
    for index, item in enumerate(events, 1):
        item["index"] = index
    result = parse_match(metadata, events, rosters)
    assert result["diagnostics"]["total_duration_seconds"] == 5820
    assert len(result["stints"]) == 2
    assert result["diagnostics"]["all_source_event_clock_anomalies"][0]["event_id"] == "event-21"
    assert receipt["timestamp"] == "00:00:00.200"


def test_nonmodel_event_beyond_half_end_is_reported_without_extending_exposure():
    metadata, events, rosters = source()
    events.insert(3, event(20, 1, 2821, "Ball Receipt*", player=1))
    for index, item in enumerate(events, 1):
        item["index"] = index
    result = parse_match(metadata, events, rosters)
    assert result["diagnostics"]["total_duration_seconds"] == 5820
    assert any(row["issue"] == "nonmodel_event_beyond_half_end"
               for row in result["diagnostics"]["all_source_event_clock_anomalies"])


def test_relevant_clock_inversion_still_fails_even_with_irrelevant_events_between():
    metadata, events, rosters = source()
    events[3:3] = [event(20, 1, 600, "Player Off", player=1),
                   event(21, 1, 610, "Pass", player=2),
                   event(22, 1, 590, "Player On", player=1)]
    for index, item in enumerate(events, 1):
        item["index"] = index
    with pytest.raises(StatsBombParseError, match="model-relevant"):
        parse_match(metadata, events, rosters)
