from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_model import impect_starting_roles as subject
from football_model import impect_starting_xi as xi


def source():
    metadata = {"id": 1, "iterationId": 743, "homeSquadId": 38, "awaySquadId": 33,
                "scheduledDate": "2024-01-02T00:30:00+02:00"}
    lineups = {"id": 1, "iterationId": 743, "dateTime": "2024-01-01T22:30:00Z"}
    labels = ["GOALKEEPER", "CENTRAL_DEFENDER", "CENTRAL_DEFENDER", "LEFT_WINGBACK_DEFENDER",
              "RIGHT_WINGBACK_DEFENDER", "DEFENSE_MIDFIELD", "CENTRAL_MIDFIELD", "ATTACKING_MIDFIELD",
              "CENTER_FORWARD", "LEFT_WINGER", "RIGHT_WINGER"]
    for key, team, first in (("squadHome", 38, 1), ("squadAway", 33, 101)):
        lineups[key] = {"id": team, "players": [{"id": player} for player in range(first, first + 13)],
                        "startingPositions": [{"playerId": first + index, "position": label} for index, label in enumerate(labels)],
                        "substitutions": []}
    return metadata, lineups, xi.extract_match(metadata, lineups, [2, 1])


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_adds_only_two_role_maps_and_preserves_original_seven_fields():
    metadata, lineups, expected = source()
    before = deepcopy((metadata, lineups, expected))
    result = subject.extract_roles(metadata, lineups, expected)
    assert set(result) == subject.BASE_FIELDS | {"home_roles", "away_roles"}
    assert {key: result[key] for key in expected} == expected
    assert result["home_roles"]["impect:player:1"] == "GK"
    assert result["home_roles"]["impect:player:4"] == "DEF"
    assert result["home_roles"]["impect:player:8"] == "MID"
    assert result["home_roles"]["impect:player:10"] == "FWD"
    assert set(result["away_roles"]) == set(expected["away_starters"])
    assert (metadata, lineups, expected) == before
    result["full_time"][0] = 9
    result["home_starters"].reverse()
    assert expected == before[2]


def test_same_player_can_have_different_roles_in_different_matches():
    metadata, lineups, expected = source()
    first = subject.extract_roles(metadata, lineups, expected)
    lineups["squadHome"]["startingPositions"][9]["position"] = "ATTACKING_MIDFIELD"
    metadata["id"] = lineups["id"] = 2
    second_expected = xi.extract_match(metadata, lineups, [0, 0])
    second = subject.extract_roles(metadata, lineups, second_expected)
    assert first["home_roles"]["impect:player:10"] == "FWD"
    assert second["home_roles"]["impect:player:10"] == "MID"


@pytest.mark.parametrize("future", [None, ["arbitrary"], {"RED_CARD": "unusable"}, -100])
def test_future_paths_and_current_squad_attributes_do_not_affect_roles(future):
    metadata, lineups, expected = source()
    original = subject.extract_roles(metadata, lineups, expected)
    for side in ("squadHome", "squadAway"):
        lineups[side]["substitutions"] = future
        for player in lineups[side]["players"]:
            player["currentSquadId"] = future
    lineups["events"] = lineups["player_kpis"] = future
    assert subject.extract_roles(metadata, lineups, expected) == original


@pytest.mark.parametrize("field,value", [("id", 5), ("iterationId", 7), ("dateTime", "2024-01-03T00:00:00Z")])
def test_old_xi_identity_validation_is_reused(field, value):
    metadata, lineups, expected = source()
    lineups[field] = value
    with pytest.raises(xi.StartingXiSourceError, match="identity"):
        subject.extract_roles(metadata, lineups, expected)


def test_unknown_nonbank_position_has_no_default_role():
    metadata, lineups, expected = source()
    lineups["squadHome"]["startingPositions"][3]["position"] = "LIBERO_NEW_LABEL"
    with pytest.raises(subject.StartingRolesSourceError, match="Unmapped"):
        subject.extract_roles(metadata, lineups, expected)


@pytest.mark.parametrize("field,value", [("match_id", "impect:match:2"), ("match_date", "2024-01-02"),
    ("home_team_id", 99), ("away_team_id", 99)])
def test_saved_base_identity_mismatch_is_not_repaired(field, value):
    metadata, lineups, expected = source()
    expected[field] = value
    with pytest.raises(subject.StartingRolesSourceError, match="disagrees"):
        subject.extract_roles(metadata, lineups, expected)


@pytest.mark.parametrize("case", ["different", "duplicate", "short", "extra_field"])
def test_saved_starter_set_or_schema_mismatch_is_not_silently_selected(case):
    metadata, lineups, expected = source()
    if case == "different":
        expected["home_starters"][0] = "impect:player:12"
    elif case == "duplicate":
        expected["home_starters"][0] = expected["home_starters"][1]
    elif case == "short":
        expected["home_starters"].pop()
    else:
        expected["future_role"] = "MID"
    with pytest.raises(subject.StartingRolesSourceError):
        subject.extract_roles(metadata, lineups, expected)


def inspection_item(metadata, receipt):
    return {"source_match_id": metadata["id"], "included": True,
            "home_team_id": metadata["homeSquadId"], "away_team_id": metadata["awaySquadId"],
            "source_scheduled_date_text": metadata["scheduledDate"],
            "scheduled_kickoff_utc": "2024-01-01T22:30:00+00:00", "lineups_receipt": receipt}


def test_run_reads_explicit_receipt_and_keeps_provenance_outside_features(tmp_path, monkeypatch):
    metadata, lineups, expected = source()
    lineups_file = put(tmp_path / "lineups.json", lineups)
    receipt = {"kind": "lineups", "file": lineups_file, "http_status": 200}
    item = inspection_item(metadata, receipt)
    original_source = {"matches": [item], "source_score_file": str(tmp_path / "explicit-score.csv"),
                       "source_terms": "local custom terms"}
    source_path = Path(put(tmp_path / "source_inspection.json", original_source))
    matches_path = Path(put(tmp_path / "matches.json", [expected]))
    initial_source, initial_matches, initial_lineups = (path.read_bytes() for path in (source_path, matches_path, Path(lineups_file)))
    monkeypatch.setattr(subject, "EXPECTED_MATCHES", 1)
    monkeypatch.setattr(xi, "independent_score", lambda *args, **kwargs: {"full_time": [2, 1], "half_time": [1, 0]})
    result = subject.run(tmp_path, source_path, matches_path)
    output = Path(result["output_directory"])
    assert result["included_matches"] == result["included_targets"] == 1
    converted = subject.read(output / "matches.json")
    assert converted == [subject.extract_roles(metadata, lineups, expected)]
    saved = subject.read(output / "inspection.json")
    assert saved["historical_first_available_at"] is None
    assert saved["historical_opening_position_announcement_at"] is None
    assert saved["source_terms"] == "local custom terms"
    assert saved["fits"] == saved["downloads"] == 0
    assert source_path.read_bytes() == initial_source and matches_path.read_bytes() == initial_matches
    assert Path(lineups_file).read_bytes() == initial_lineups


def test_independent_full_time_join_conflict_is_retained_not_refilled(tmp_path, monkeypatch):
    metadata, lineups, expected = source()
    item = inspection_item(metadata, {"kind": "lineups", "http_status": 200,
                                    "file": put(tmp_path / "lineups.json", lineups)})
    monkeypatch.setattr(xi, "independent_score", lambda *args, **kwargs: {"full_time": [9, 9], "half_time": [0, 0]})
    observation, match = subject.inspect_match(item, expected, "unused.csv")
    assert match is None and observation["status"] == "saved_full_time_conflict"


def test_unknown_label_is_an_explicit_source_exclusion(tmp_path, monkeypatch):
    metadata, lineups, expected = source()
    lineups["squadAway"]["startingPositions"][3]["position"] = "UNKNOWN_ROLE"
    item = inspection_item(metadata, {"kind": "lineups", "http_status": 200,
                                    "file": put(tmp_path / "lineups.json", lineups)})
    monkeypatch.setattr(xi, "independent_score", lambda *args, **kwargs: {"full_time": [2, 1], "half_time": [1, 0]})
    observation, match = subject.inspect_match(item, expected, "unused.csv")
    assert match is None and observation["status"] == "roles_or_saved_xi_conflict"
    assert "Unmapped" in observation["error"]


def test_mismatched_match_ids_in_explicit_inputs_raise(tmp_path, monkeypatch):
    metadata, _, expected = source()
    item = inspection_item({**metadata, "id": 2}, {})
    source_path = put(tmp_path / "source.json", {"matches": [item], "source_score_file": "unused.csv"})
    matches_path = put(tmp_path / "matches.json", [expected])
    monkeypatch.setattr(subject, "EXPECTED_MATCHES", 1)
    with pytest.raises(ValueError, match="identical distinct"):
        subject.run(tmp_path, source_path, matches_path)
