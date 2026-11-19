from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_model import impect_starting_xi as subject


def source():
    metadata = {"id": 1, "iterationId": 743, "homeSquadId": 38, "awaySquadId": 33,
                "scheduledDate": "2024-01-02T00:30:00+02:00"}
    lineups = {"id": 1, "iterationId": 743, "dateTime": "2024-01-01T22:30:00Z",
               "lastCalculationDate": "2025-08-01T00:00:00Z"}
    for key, team, first in (("squadHome", 38, 1), ("squadAway", 33, 101)):
        lineups[key] = {"id": team, "players": [{"id": player} for player in range(first, first + 15)],
                        "startingPositions": [{"playerId": player, "position": "CENTRAL_DEFENDER"}
                                              for player in range(first, first + 11)],
                        "substitutions": []}
    return metadata, lineups


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_extracts_only_pure_source_xi_and_independent_full_time():
    metadata, lineups = source()
    match = subject.extract_match(metadata, lineups, [2, 1])
    assert set(match) == {"match_id", "match_date", "home_team_id", "away_team_id",
                          "home_starters", "away_starters", "full_time"}
    assert match["match_date"] == "2024-01-01"
    assert match["home_starters"] == [f"impect:player:{i}" for i in range(1, 12)]
    assert match["away_starters"] == [f"impect:player:{i}" for i in range(101, 112)]
    assert match["full_time"] == [2, 1]


@pytest.mark.parametrize("later_path", [None, "malformed", {"RED_CARD": True},
    [{"fromPosition": "BANK", "toPosition": "SOMETHING", "playerId": -5}], [{"future": "outcome"}]])
def test_later_substitutions_and_malicious_current_squad_are_ignored(later_path):
    metadata, lineups = source()
    expected = subject.extract_match(metadata, lineups, [2, 1])
    for key in ("squadHome", "squadAway"):
        lineups[key]["substitutions"] = later_path
        lineups[key]["currentSquadId"] = -999
        for player in lineups[key]["players"]:
            player["currentSquadId"] = {"malicious": "future transfer"}
    lineups["events"] = {"RED_CARD": 99}
    lineups["player_kpis"] = ["not read"]
    assert subject.extract_match(metadata, lineups, [2, 1]) == expected


@pytest.mark.parametrize("mutation,pattern", [
    ("match_id", "identity"), ("iteration", "identity"), ("kickoff", "identity"),
    ("home_id", "squad identity"), ("duplicate_starter", "duplicate starter"),
    ("unknown_roster", "Unknown roster"), ("cross_roster", "both squads"),
    ("duplicate_roster", "Duplicate roster"), ("too_few", "exactly 11"),
    ("too_many", "exactly 11"), ("bank", "non-BANK"), ("blank_position", "non-BANK"),
    ("bool_player", "integer"), ("negative_player", "integer"),
])
def test_invalid_source_identity_or_starters_fail(mutation, pattern):
    metadata, lineups = source()
    home, away = lineups["squadHome"], lineups["squadAway"]
    if mutation == "match_id":
        lineups["id"] = 2
    elif mutation == "iteration":
        lineups["iterationId"] = 1
    elif mutation == "kickoff":
        lineups["dateTime"] = "2024-01-02T22:30:00Z"
    elif mutation == "home_id":
        home["id"] = 99
    elif mutation == "duplicate_starter":
        home["startingPositions"][1] = deepcopy(home["startingPositions"][0])
    elif mutation == "unknown_roster":
        home["startingPositions"][0]["playerId"] = 999
    elif mutation == "cross_roster":
        away["players"][0]["id"] = home["players"][0]["id"]
    elif mutation == "duplicate_roster":
        home["players"].append(deepcopy(home["players"][0]))
    elif mutation == "too_few":
        home["startingPositions"].pop()
    elif mutation == "too_many":
        home["startingPositions"].append({"playerId": 12, "position": "STRIKER"})
    elif mutation == "bank":
        home["startingPositions"][0]["position"] = "BANK"
    elif mutation == "blank_position":
        home["startingPositions"][0]["position"] = " "
    elif mutation == "bool_player":
        home["startingPositions"][0]["playerId"] = True
    elif mutation == "negative_player":
        home["players"][0]["id"] = -1
    with pytest.raises(subject.StartingXiSourceError, match=pattern):
        subject.extract_match(metadata, lineups, [2, 1])


@pytest.mark.parametrize("field,target", [("scheduledDate", "metadata"), ("dateTime", "lineups")])
def test_naive_timestamp_never_invents_utc(field, target):
    metadata, lineups = source()
    (metadata if target == "metadata" else lineups)[field] = "2024-01-01T22:30:00"
    with pytest.raises(subject.StartingXiSourceError, match="timezone"):
        subject.extract_match(metadata, lineups, [2, 1])


@pytest.mark.parametrize("full_time", [None, [1], [1, 2, 3], [-1, 0], [True, 0], [1.5, 0]])
def test_invalid_full_time_not_imputed(full_time):
    with pytest.raises(subject.StartingXiSourceError):
        subject.extract_match(*source(), full_time)


def test_starters_source_order_does_not_change_identity_features():
    metadata, lineups = source()
    expected = subject.extract_match(metadata, lineups, [2, 1])
    lineups["squadHome"]["startingPositions"].reverse()
    assert subject.extract_match(metadata, lineups, [2, 1]) == expected


def test_only_lineups_file_read_even_when_other_downloads_incomplete(tmp_path, monkeypatch):
    metadata, lineups = source()
    lineup_file = put(tmp_path / "lineups.json", lineups)
    record = {"match_id": 1, "status": "incomplete", "files": [
        {"kind": "events", "file": "do-not-read-events.json", "error": "source path unresolved"},
        {"kind": "lineups", "file": lineup_file, "http_status": 200, "received_at": "2026-01-01T00:00:00Z"},
        {"kind": "player_kpis", "file": "do-not-read-kpis.json"}]}
    monkeypatch.setattr(subject, "independent_score", lambda *args, **kwargs: {"full_time": [2, 1], "half_time": [1, 0]})
    original_read, opened = subject.read, []
    def observe(path):
        opened.append(str(path))
        return original_read(path)
    monkeypatch.setattr(subject, "read", observe)
    item, match = subject.inspect_match(metadata, record, "unused.csv")
    assert opened == [lineup_file]
    assert item["status"] == "included" and match["full_time"] == [2, 1]
    assert "independent_score" not in match
    assert item["lineups_last_calculation_date"] == lineups["lastCalculationDate"]


@pytest.mark.parametrize("case", ["no_record", "no_lineups", "missing_file", "failed_http"])
def test_missing_lineups_retains_source_status_without_output(tmp_path, case):
    metadata, _ = source()
    record = {"match_id": 1, "files": [{"kind": "lineups", "file": str(tmp_path / "missing.json"), "http_status": 200}]}
    if case == "no_record":
        record = None
    elif case == "no_lineups":
        record["files"] = []
    elif case == "failed_http":
        record["files"][0]["http_status"] = 429
    item, match = subject.inspect_match(metadata, record, "unused.csv")
    assert match is None and item["included"] is False
    assert item["status"] in {"lineups_missing", "lineups_download_failed"}


@pytest.mark.parametrize("case", ["duplicate", "modified", "version", "partial"])
def test_saved_selection_must_be_full_original_v3_catalogue(monkeypatch, case):
    metadata, _ = source()
    second = {**metadata, "id": 2, "scheduledDate": "2023-12-01T00:00:00Z"}
    catalogue = [metadata, second]
    selection = {"documented_data_version": "V3", "selected_matches": deepcopy(catalogue)}
    monkeypatch.setattr(subject, "CATALOGUE_MATCH_COUNT", 2)
    if case == "duplicate":
        selection["selected_matches"][1] = deepcopy(metadata)
    elif case == "modified":
        selection["selected_matches"][0]["lastCalculationDate"] = "different source revision"
    elif case == "version":
        selection["documented_data_version"] = "V4"
    else:
        selection["selected_matches"].pop()
    with pytest.raises(ValueError, match="selection"):
        subject.selected_catalogue(catalogue, selection)


def test_run_preserves_full_target_denominator_and_pure_schema(tmp_path, monkeypatch):
    metadata, lineups = source()
    earlier = {**metadata, "id": 2, "scheduledDate": "2023-12-01T00:00:00Z"}
    catalogue = [earlier, metadata]
    season = tmp_path / "saved_explicit_season"
    put(tmp_path / "data/impect/feasibility_20261023/matches_743.json", catalogue)
    put(season / "selection.json", {"documented_data_version": "V3", "selected_matches": catalogue})
    lineup_file = put(season / "lineups.json", lineups)
    put(season / "matches/1.json", {"match_id": 1, "files": [{"kind": "lineups", "file": lineup_file, "http_status": 200}]})
    monkeypatch.setattr(subject, "CATALOGUE_MATCH_COUNT", 2)
    monkeypatch.setattr(subject, "independent_score", lambda *args, **kwargs: {"full_time": [2, 1], "half_time": [1, 0]})
    result = subject.run(tmp_path, season)
    output = Path(result["output_directory"])
    assert result["expected_matches"] == 2 and result["included_matches"] == 1
    assert result["expected_targets"] == result["included_targets"] == 1
    saved = subject.read(output / "matches.json")
    assert saved == [subject.extract_match(metadata, lineups, [2, 1])]
    inspection = subject.read(output / "inspection.json")
    assert inspection["fits"] == 0 and inspection["statuses"] == {"lineups_missing": 1, "included": 1}
    assert inspection["historical_first_available_at"] is None
    assert all(item["historical_lineup_announcement_at"] is None for item in inspection["matches"])
