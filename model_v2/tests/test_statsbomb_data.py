import json

import pytest

from football_model.statsbomb_data import _fetch, season_profile


def metadata(mid, home, away):
    return {"match_id": mid, "match_date": "2015-08-08",
            "home_team": {"home_team_id": home}, "away_team": {"away_team_id": away}}


def test_complete_schedule_is_measured_without_assuming_a_fixed_team_count():
    profile = season_profile([metadata(i, h, a) for i, (h, a) in enumerate(
        [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], 1)])
    assert profile["complete_double_round_robin"]
    assert profile["team_match_counts"] == {"1": 4, "2": 4, "3": 4}


def test_catalog_label_cannot_hide_missing_or_repeated_pairs_and_match_ids():
    profile = season_profile([metadata(1, 1, 2), metadata(1, 1, 2)])
    assert not profile["complete_double_round_robin"]
    assert profile["duplicate_match_ids"] == [1]
    assert profile["missing_home_away_pairs"] == [[2, 1]]
    with pytest.raises(ValueError):
        season_profile([])


def test_cached_source_json_is_validated_and_not_assigned_a_fake_download_time(tmp_path):
    path = tmp_path / "events_1.json"
    path.write_text("[]", encoding="utf-8")
    result = _fetch(("data/events/1.json", "events_1.json"), tmp_path)
    assert result["reused_local_file"]
    assert result["retrieved_at_utc"] is None and result["historical_first_available_at"] is None
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _fetch(("data/events/1.json", "events_1.json"), tmp_path)
