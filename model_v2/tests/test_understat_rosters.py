from copy import deepcopy
import json
from pathlib import Path

import pytest

from football_model import understat_rosters as subject


def source():
    identity = {"match_id": "football-data:D1:2023/24:Augsburg|M'gladbach",
                "date": "2023-08-19", "home": "football-data:D1:Augsburg",
                "away": "football-data:D1:M'gladbach", "understat_id": "23069"}
    payload = {"rosters": {}}
    for side, team, first in (("h", 10, 1), ("a", 20, 101)):
        payload["rosters"][side] = {
            str(600000 + player): {
                "id": str(600000 + player), "player_id": str(player),
                "team_id": str(team), "h_a": side,
                "position": "DC" if player < first + 11 else "Sub",
                "positionOrder": str(player - first + 1), "time": "90",
                "goals": "0", "xG": "0", "player": f"Player {player}",
            } for player in range(first, first + 14)
        }
    return payload, identity


def row(payload, side="h", offset=0):
    return list(payload["rosters"][side].values())[offset]


def test_returns_only_identity_starters_and_explicit_source_limitations():
    payload, identity = source()
    original = deepcopy((payload, identity))
    result = subject.extract_starting_xi(payload, identity)
    assert result["match_id"] == identity["match_id"]
    assert result["match_date"] == "2023-08-19"
    assert result["home"] == identity["home"] and result["away"] == identity["away"]
    assert result["home_starters"] == [f"understat:player:{i}" for i in range(1, 12)]
    assert result["away_starters"] == [f"understat:player:{i}" for i in range(101, 112)]
    assert result["understat_id"] == "23069"
    assert result["source_url"] == "https://understat.com/match/23069"
    assert result["historical_lineup_announcement_at"] is None
    assert result["historical_first_available_at"] is None
    assert result["identity_basis"] == "caller_supplied_exact_fixture_join_not_verified_by_roster_payload"
    assert result["availability"] == "retrospective_starting_xi_known; original_publication_time_unknown"
    assert set(result) == {"match_id", "match_date", "home", "away", "understat_id",
                           "home_starters", "away_starters", "source_url", "identity_basis",
                           "historical_lineup_announcement_at", "historical_first_available_at", "availability"}
    assert (payload, identity) == original


def test_namespaced_player_id_is_not_roster_appearance_id():
    payload, identity = source()
    result = subject.extract_starting_xi(payload, identity)
    assert "understat:player:1" in result["home_starters"]
    assert "understat:player:600001" not in result["home_starters"]


def test_sub_with_90_minutes_and_low_position_order_remains_a_sub():
    payload, identity = source()
    expected = subject.extract_starting_xi(payload, identity)
    row(payload, offset=11).update(time="90", positionOrder="1")
    assert subject.extract_starting_xi(payload, identity) == expected


@pytest.mark.parametrize("order", ["15", "16", None, {"not": "a feature"}])
def test_starter_position_order_does_not_define_starting_status(order):
    payload, identity = source()
    expected = subject.extract_starting_xi(payload, identity)
    row(payload)["positionOrder"] = order
    assert subject.extract_starting_xi(payload, identity) == expected


@pytest.mark.parametrize("position", sorted(subject.STARTING_POSITIONS))
def test_all_sixteen_observed_field_position_codes_are_supported(position):
    payload, identity = source()
    row(payload)["position"] = position
    assert len(subject.extract_starting_xi(payload, identity)["home_starters"]) == 11


def test_performance_minutes_names_cards_and_substitution_fields_are_not_inspected():
    payload, identity = source()
    expected = subject.extract_starting_xi(payload, identity)
    for side in payload["rosters"].values():
        for entry in side.values():
            for field in ("time", "goals", "own_goals", "shots", "xG", "key_passes", "assists",
                          "xA", "xGChain", "xGBuildup", "yellow_card", "red_card",
                          "roster_in", "roster_out", "player", "positionOrder"):
                entry[field] = {"deliberately": "not parsed"}
    payload.update(shots=None, tmpl=["postmatch rendering"])
    identity.update(goals=[99, 98], xg=[100, 100], source_url="untrusted external URL",
                    historical_lineup_announcement_at="2023-08-19T12:00:00Z")
    assert subject.extract_starting_xi(payload, identity) == expected


def test_roster_order_is_not_a_feature():
    payload, identity = source()
    expected = subject.extract_starting_xi(payload, identity)
    payload["rosters"] = {side: dict(reversed(list(roster.items())))
                           for side, roster in reversed(list(payload["rosters"].items()))}
    assert subject.extract_starting_xi(payload, identity) == expected


def test_parser_cannot_claim_to_verify_external_fixture_date_or_match_id():
    payload, identity = source()
    identity.update(date="2024-08-19", match_id="externally-joined-fixture", understat_id="999")
    result = subject.extract_starting_xi(payload, identity)
    assert result["match_date"] == "2024-08-19"
    assert result["source_url"] == "https://understat.com/match/999"
    assert "not_verified_by_roster_payload" in result["identity_basis"]


@pytest.mark.parametrize("field", ["match_id", "date", "home", "away", "understat_id"])
def test_missing_external_identity_field_is_explicit(field):
    payload, identity = source()
    del identity[field]
    with pytest.raises(subject.UnderstatRosterSourceError):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("field,value", [
    ("match_id", 3), ("home", ""), ("away", " "), ("understat_id", True),
    ("understat_id", "0"), ("understat_id", "01"), ("date", "20230819"),
    ("date", "2023-02-30"), ("date", None),
])
def test_invalid_external_identity_is_not_repaired(field, value):
    payload, identity = source()
    identity[field] = value
    with pytest.raises(subject.UnderstatRosterSourceError):
        subject.extract_starting_xi(payload, identity)


def test_same_external_home_and_away_is_invalid():
    payload, identity = source()
    identity["away"] = identity["home"]
    with pytest.raises(subject.UnderstatRosterSourceError, match="Distinct external"):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("payload", [None, [], {}, {"rosters": []},
    {"rosters": {"h": {}}}, {"rosters": {"h": {}, "a": {}, "neutral": {}}}])
def test_missing_or_unsupported_side_structure_is_explicit(payload):
    with pytest.raises(subject.UnderstatRosterSourceError):
        subject.extract_starting_xi(payload, source()[1])


@pytest.mark.parametrize("field", ["id", "player_id", "team_id", "h_a", "position"])
def test_required_row_fields_are_not_inferred(field):
    payload, identity = source()
    del row(payload)[field]
    with pytest.raises(subject.UnderstatRosterSourceError):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("position", ["UNKNOWN", "BANK", "sub", "", None, 15, {}])
def test_unknown_position_is_unsupported_not_a_guess_from_minutes(position):
    payload, identity = source()
    row(payload).update(position=position, time="90", positionOrder="1")
    with pytest.raises(subject.UnderstatRosterSourceError, match="Unsupported"):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("field,value", [("id", True), ("id", -1), ("player_id", ""),
    ("player_id", True), ("player_id", 1.5), ("team_id", "0"), ("team_id", True)])
def test_invalid_identifiers_are_not_coerced(field, value):
    payload, identity = source()
    row(payload)[field] = value
    with pytest.raises(subject.UnderstatRosterSourceError, match="decimal ID"):
        subject.extract_starting_xi(payload, identity)


def test_dictionary_key_must_equal_the_raw_roster_row_id():
    payload, identity = source()
    row(payload)["id"] = "999999"
    with pytest.raises(subject.UnderstatRosterSourceError, match="key must equal"):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("side", ["a", "", None])
def test_row_side_must_match_its_container(side):
    payload, identity = source()
    row(payload)["h_a"] = side
    with pytest.raises(subject.UnderstatRosterSourceError, match="h_a"):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("side,offset", [("h", 11), ("h", 12), ("a", 0), ("a", 11)])
def test_duplicate_players_including_bench_and_other_team_are_rejected(side, offset):
    payload, identity = source()
    row(payload, side, offset)["player_id"] = row(payload)["player_id"]
    with pytest.raises(subject.UnderstatRosterSourceError, match="Duplicate player_id"):
        subject.extract_starting_xi(payload, identity)


def test_team_id_must_be_consistent_even_for_a_sub():
    payload, identity = source()
    row(payload, offset=11)["team_id"] = "999"
    with pytest.raises(subject.UnderstatRosterSourceError, match="consistent"):
        subject.extract_starting_xi(payload, identity)


def test_both_sides_cannot_have_the_same_team_id():
    payload, identity = source()
    for entry in payload["rosters"]["a"].values():
        entry["team_id"] = "10"
    with pytest.raises(subject.UnderstatRosterSourceError, match="Distinct home and away source"):
        subject.extract_starting_xi(payload, identity)


@pytest.mark.parametrize("count", [10, 12])
def test_exactly_eleven_starters_are_required(count):
    payload, identity = source()
    row(payload, offset=10 if count == 10 else 11)["position"] = "Sub" if count == 10 else "DC"
    with pytest.raises(subject.UnderstatRosterSourceError, match="Exactly 11"):
        subject.extract_starting_xi(payload, identity)


def test_real_saved_pilot_if_present_contains_20_valid_pure_xis():
    pilot = Path(__file__).resolve().parents[2] / "data/understat/roster_pilot_20261024T183946498125Z"
    if not pilot.is_dir():
        pytest.skip("Personal saved pilot is not redistributed with the parser")
    selection = json.loads((pilot / "试样选择.json").read_text(encoding="utf-8"))["selected"]
    assert len(selection) == 20
    for identity in selection:
        payload = json.loads((pilot / f'{identity["understat_id"]}.json').read_text(encoding="utf-8"))
        result = subject.extract_starting_xi(payload, identity)
        assert len(result["home_starters"]) == len(result["away_starters"]) == 11
        assert not set(result["home_starters"]) & set(result["away_starters"])
