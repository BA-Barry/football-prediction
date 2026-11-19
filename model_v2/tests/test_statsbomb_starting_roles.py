from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from football_model import statsbomb_starting_roles as subject
from football_model.full_season_experiment import TEAM_NAMES


def sources():
    metadata = {"match_id": 10, "match_date": "2015-08-08", "home_score": 1, "away_score": 0,
                "home_team": {"home_team_id": 1, "home_team_name": "Arsenal"},
                "away_team": {"away_team_id": 33, "away_team_name": "Chelsea"}}
    events, lineups = [], []
    positions = [1, 2, 3, 5, 6, 9, 11, 12, 17, 21, 23]
    for team, first in ((1, 1), (33, 101)):
        rows, roster = [], []
        for i, position in enumerate(positions):
            pid, label = first + i, subject.POSITION_MAPPING[position][0]
            rows.append({"player": {"id": pid}, "position": {"id": position, "name": label}})
            roster.append({"player_id": pid, "positions": [{"from_period": 1, "from": "00:00",
                           "position_id": position, "position": label, "start_reason": "Starting XI"}]})
        events.append({"id": str(team), "type": {"name": "Starting XI"}, "team": {"id": team},
                       "period": 1, "timestamp": "00:00:00.000", "minute": 0, "second": 0,
                       "tactics": {"lineup": rows}})
        lineups.append({"team_id": team, "lineup": roster})
    events.append({"id": "goal", "type": {"name": "Shot"}, "team": {"id": 1}, "period": 1,
                   "shot": {"outcome": {"name": "Goal"}}})
    return metadata, events, lineups


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def scores(path, rows=None):
    if rows is None:
        rows = [["E0", "08/08/2015", "Arsenal", "Chelsea", "1", "0", "1", "0"]]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"])
        writer.writerows(rows)
    return path


def test_nine_fields_namespaced_stable_ids_and_independent_starting_roles():
    meta, events, lineups = sources()
    original = deepcopy((meta, events, lineups))
    result = subject.extract_roles(meta, events, lineups, [1, 0])
    assert set(result) == subject.BASE_FIELDS | {"home_roles", "away_roles"}
    assert result["match_id"] == "statsbomb:match:10"
    assert (result["home_team_id"], result["away_team_id"]) == (1, 33)
    assert result["home_starters"] == [f"statsbomb:player:{i}" for i in range(1, 12)]
    assert result["home_roles"]["statsbomb:player:1"] == "GK"
    assert result["home_roles"]["statsbomb:player:8"] == "MID"
    assert result["home_roles"]["statsbomb:player:9"] == "FWD"
    assert (meta, events, lineups) == original


@pytest.mark.parametrize("pid,expected", [(1, "GK"), (7, "DEF"), (8, "DEF"),
    (12, "MID"), (16, "MID"), (18, "MID"), (20, "MID"), (17, "FWD"), (21, "FWD"), (25, "FWD")])
def test_explicit_coarse_role_semantics(pid, expected):
    assert subject.role(pid, subject.POSITION_MAPPING[pid][0]) == expected


@pytest.mark.parametrize("pid,label", [(99, "Unknown"), (17, "Right Midfield"), (True, "Goalkeeper")])
def test_unknown_or_inconsistent_positions_are_not_filled(pid, label):
    with pytest.raises(subject.StartingRolesSourceError, match="Unknown or conflicting"):
        subject.role(pid, label)


def test_only_initial_observations_used_not_later_minutes_roles_cards_or_event_order():
    meta, events, lineups = sources()
    expected = subject.extract_roles(meta, events, lineups, [1, 0])
    for event in events:
        if event["type"]["name"] == "Starting XI":
            event["tactics"]["formation"] = "arbitrary ignored formation"
    events += [{"type": {"name": "Substitution"}, "minute": -999}, {"type": {"name": "Bad Behaviour"}, "red": True}]
    for lineup in lineups:
        for player in lineup["lineup"]:
            player["cards"] = {"arbitrary": "ignored"}
            player["positions"][0].update(to="19:03", to_period=1, end_reason="Red Card")
            player["positions"].append({"from_period": 2, "from": "45:00", "position_id": 99,
                                       "position": "unknown future role"})
    assert subject.extract_roles(meta, list(reversed(events)), lineups, [1, 0]) == expected


@pytest.mark.parametrize("mutation", ["missing_xi", "duplicate_xi", "late_xi", "wrong_period", "short_xi",
    "duplicate_player", "cross_team_player", "unknown_role", "changed_opening", "late_roster_start",
    "missing_roster", "duplicate_roster", "wrong_reason", "ambiguous_opening", "duplicate_roster_player"])
def test_source_ambiguity_rejected(mutation):
    meta, events, lineups = sources()
    if mutation == "missing_xi": events.pop(0)
    elif mutation == "duplicate_xi": events.append(deepcopy(events[0]))
    elif mutation == "late_xi": events[0]["timestamp"] = "00:00:01.000"
    elif mutation == "wrong_period": events[0]["period"] = 2
    elif mutation == "short_xi": events[0]["tactics"]["lineup"].pop()
    elif mutation == "duplicate_player": events[0]["tactics"]["lineup"][1]["player"]["id"] = 1
    elif mutation == "cross_team_player": events[1]["tactics"]["lineup"][0]["player"]["id"] = 1
    elif mutation == "unknown_role": events[0]["tactics"]["lineup"][1]["position"]["id"] = 99
    elif mutation == "changed_opening": lineups[0]["lineup"][0]["positions"][0].update(position_id=2, position="Right Back")
    elif mutation == "late_roster_start": lineups[0]["lineup"][0]["positions"][0]["from"] = "00:01"
    elif mutation == "missing_roster": lineups.pop()
    elif mutation == "duplicate_roster": lineups.append(deepcopy(lineups[0]))
    elif mutation == "wrong_reason": lineups[0]["lineup"][0]["positions"][0]["start_reason"] = "Substitution"
    elif mutation == "ambiguous_opening": lineups[0]["lineup"][0]["positions"] *= 2
    elif mutation == "duplicate_roster_player": lineups[0]["lineup"].append(deepcopy(lineups[0]["lineup"][0]))
    with pytest.raises(subject.StartingRolesSourceError):
        subject.extract_roles(meta, events, lineups, [1, 0])


def test_explicit_aliases_match_existing_twenty_team_id_mapping():
    meta, _, _ = sources()
    assert len(subject.TEAM_ALIASES) == len(TEAM_NAMES) == 20
    assert set(subject.TEAM_ALIASES.values()) == set(TEAM_NAMES.values())


def test_exact_independent_score_and_half_time(tmp_path):
    meta, _, _ = sources()
    score = subject.independent_score(meta, subject.load_scores(scores(tmp_path / "scores.csv")))
    assert score == {"full_time": [1, 0], "half_time": [1, 0], "csv_line": 2,
                     "exact_join_key": ["2015-08-08", "Arsenal", "Chelsea"]}


@pytest.mark.parametrize("change", ["date", "reversed", "unknown_alias", "wrong_team_id", "duplicates", "wrong_ft", "bad_goals", "wrong_div", "ht_gt_ft"])
def test_score_join_does_not_use_fuzzy_names_or_silently_pick_duplicate(tmp_path, change):
    meta, _, _ = sources()
    rows = [["E0", "08/08/2015", "Arsenal", "Chelsea", "1", "0", "1", "0"]]
    if change == "date": meta["match_date"] = "2015-08-09"
    elif change == "reversed": rows[0][2:4] = ["Chelsea", "Arsenal"]
    elif change == "unknown_alias": meta["home_team"]["home_team_name"] = "Arsenal FC"
    elif change == "wrong_team_id": meta["home_team"]["home_team_id"] = 22
    elif change == "duplicates": rows *= 2
    elif change == "wrong_ft": rows[0][4] = "2"
    elif change == "bad_goals": rows[0][4] = "1.0"
    elif change == "wrong_div": rows[0][0] = "E1"
    elif change == "ht_gt_ft": rows[0][6] = "2"
    with pytest.raises(subject.StartingRolesSourceError):
        subject.independent_score(meta, subject.load_scores(scores(tmp_path / "scores.csv", rows)))


def test_own_goal_counted_once_and_half_time_kept_independent():
    meta, events, _ = sources()
    events += [{"id": "against", "type": {"name": "Own Goal Against"}, "team": {"id": 33},
                "period": 2, "timestamp": "00:10:00.000", "related_events": ["for"]},
               {"id": "for", "type": {"name": "Own Goal For"}, "team": {"id": 1},
                "period": 2, "timestamp": "00:10:00.000", "related_events": ["against"]}]
    result = subject.event_scores(meta, events)
    assert result["full_time"] == [2, 0] and result["half_time"] == [1, 0]
    assert result["goal_event_ids"] == ["goal", "against"]
    events[-1]["timestamp"] = "00:10:01.000"
    with pytest.raises(subject.StartingRolesSourceError, match="consistent linked"):
        subject.event_scores(meta, events)


def test_inspection_retains_independent_score_and_opening_failures_together(tmp_path):
    meta, events, lineups = sources()
    lineups[0]["lineup"][0]["positions"][0]["from"] = "00:01"
    put(tmp_path / "events_10.json", events)
    put(tmp_path / "lineups_10.json", lineups)
    observed, converted = subject.inspect_match(meta, tmp_path, {})
    assert converted is None and observed["included"] is False
    assert {e["stage"] for e in observed["errors"]} == {"score_join", "opening_xi_roles"}


def test_all_zero_time_ambiguities_are_retained_without_using_end_times():
    _, _, lineups = sources()
    first, second = lineups[0]["lineup"][:2]
    first["positions"][0]["start_reason"] = "Tactical Shift"
    second["positions"].append(dict(second["positions"][0], position_id=4, position="Center Back",
                                    start_reason="Tactical Shift", to="00:00", to_period=1))
    anomalies = subject.opening_position_anomalies(lineups)
    assert len(anomalies) == 2
    assert anomalies[0]["issues"] == ["zero_time_position_not_labelled_starting_xi"]
    assert anomalies[1]["issues"] == ["multiple_zero_time_opening_positions", "zero_time_position_not_labelled_starting_xi"]
    assert len(anomalies[1]["zero_time_positions"]) == 2
    assert all("to" not in p for p in anomalies[1]["zero_time_positions"])


@pytest.mark.parametrize("field", ["period", "minute", "second"])
def test_booleans_are_not_accepted_as_opening_clock_integers(field):
    meta, events, lineups = sources()
    events[0][field] = bool(events[0][field])
    with pytest.raises(subject.StartingRolesSourceError):
        subject.extract_roles(meta, events, lineups, [1, 0])


def test_run_full_denominator_provenance_and_no_source_rewrites(tmp_path):
    meta, events, lineups = sources()
    source = tmp_path / "source"
    put(source / "events_10.json", events)
    put(source / "lineups_10.json", lineups)
    missing = deepcopy(meta)
    missing["match_id"] = 11
    missing["match_date"] = "2015-08-09"
    catalogue = put(source / "catalogue.json", [meta, missing])
    receipt = put(source / "下载来源_old.json", {"original": "receipt unchanged"})
    score_path = scores(tmp_path / "scores.csv")
    before = {p: p.read_bytes() for p in (catalogue, receipt, score_path)}
    result = subject.run(tmp_path, source, catalogue, score_path)
    inspection = subject.read(result["inspection_file"])
    assert result["expected_matches"] == 2 and result["included_matches"] == 1
    assert inspection["expected_catalogue_matches"] == 2 and len(inspection["matches"]) == 2
    assert inspection["source_receipt_files"] == [str(receipt)]
    assert inspection["historical_first_available_at"] is None
    assert inspection["historical_opening_position_announcement_at"] is None
    assert inspection["fits"] == inspection["downloads"] == inspection["forecasts"] == 0
    assert {p: p.read_bytes() for p in before} == before
    assert subject.read(result["matches_file"])[0]["match_id"] == "statsbomb:match:10"
    assert inspection["matches"][1]["match_date"] == "2015-08-09"
    assert {e["stage"] for e in inspection["matches"][1]["errors"]} == {"events", "lineups", "score_join"}


def test_duplicate_catalogue_rows_are_all_retained_as_failures(tmp_path):
    meta, events, lineups = sources()
    put(tmp_path / "events_10.json", events)
    put(tmp_path / "lineups_10.json", lineups)
    catalogue = put(tmp_path / "catalogue.json", [meta, meta])
    result = subject.run(tmp_path, tmp_path, catalogue, scores(tmp_path / "scores.csv"))
    inspection = subject.read(result["inspection_file"])
    assert result["expected_matches"] == 2 and result["included_matches"] == 0
    assert all(any(e["stage"] == "catalogue" for e in row["errors"]) for row in inspection["matches"])
