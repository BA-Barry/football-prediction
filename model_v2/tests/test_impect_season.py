"""Ordinary source inspection tests, independent of full local data downloads."""

from copy import deepcopy
import json

import pytest

from football_model import impect_season as subject
from football_model.impect_stints import ImpectParseError


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    metadata = {"id": 123087, "scheduledDate": "2024-04-26T18:30:00Z",
                "homeSquadId": 416, "awaySquadId": 31}
    record = {"match_id": metadata["id"], "status": "completed", "files": []}
    for kind in ("events", "lineups", "player_kpis"):
        value = [{"actionType": "GOAL"}, {"actionType": "OWN_GOAL"}] if kind == "events" else {}
        record["files"].append({"kind": kind, "file": put(tmp_path / f"{kind}.json", value)})
    parsed = {"stints": [{"period": 1, "home_goals": 1, "away_goals": 0}],
              "diagnostics": {"total_duration_seconds": 5700, "event_goals": [1, 0],
                              "status_events": {}, "all_source_event_clock_anomalies": [],
                              "player_duration_audit": {"compared_players": 22,
                                  "missing_active_player_ids": [], "max_absolute_difference_seconds": 0.0,
                                  "player_comparisons": []}}}
    score = {"full_time": [1, 0], "half_time": [1, 0]}
    monkeypatch.setattr(subject, "independent_score", lambda *args, **kwargs: deepcopy(score))
    monkeypatch.setattr(subject, "parse_match", lambda *args, **kwargs: deepcopy(parsed))
    return metadata, record, parsed, score


def test_missing_download_is_recorded_not_parsed(inputs):
    metadata, _, _, _ = inputs
    item, parsed = subject.inspect_match(metadata, None, "unused.csv")
    assert parsed is None
    assert item["status"] == "download_incomplete"
    assert item["target_period"] is True


def test_eligible_source_path_keeps_goals_and_independent_join(inputs):
    metadata, record, expected, _ = inputs
    item, parsed = subject.inspect_match(metadata, record, "unused.csv")
    assert parsed == expected
    assert item["training_eligible"] is True
    assert item["raw_goal_events"] == 2
    assert item["raw_own_goal_events"] == 1
    assert item["independent_score"]["half_time"] == [1, 0]


@pytest.mark.parametrize("case,status", [("halftime", "half_time_conflict"),
    ("missing", "active_duration_source_missing"), ("duration", "duration_source_conflict")])
def test_inconsistent_source_is_retained_but_not_returned_for_fit(inputs, case, status):
    metadata, record, parsed, score = inputs
    audit = parsed["diagnostics"]["player_duration_audit"]
    if case == "halftime":
        score["half_time"] = [0, 0]
    elif case == "missing":
        audit["missing_active_player_ids"] = ["impect:player:1"]
    else:
        audit["max_absolute_difference_seconds"] = .0003
    item, result = subject.inspect_match(metadata, record, "unused.csv")
    assert result is None
    assert item["status"] == status
    assert item["training_eligible"] is False


def test_parse_failure_keeps_exact_reason(inputs, monkeypatch):
    metadata, record, _, _ = inputs
    def fail(*args, **kwargs):
        raise ImpectParseError("red and BANK have different clocks")
    monkeypatch.setattr(subject, "parse_match", fail)
    item, parsed = subject.inspect_match(metadata, record, "unused.csv")
    assert parsed is None
    assert item["error"] == "red and BANK have different clocks"


def test_unexpected_programming_error_is_not_masked_as_source_exclusion(inputs, monkeypatch):
    metadata, record, _, _ = inputs
    def fail(*args, **kwargs):
        raise RuntimeError("implementation error")
    monkeypatch.setattr(subject, "parse_match", fail)
    with pytest.raises(RuntimeError, match="implementation error"):
        subject.inspect_match(metadata, record, "unused.csv")


@pytest.mark.parametrize("case", ["identity", "file_kind", "duplicate"])
def test_conflicting_manifest_does_not_silently_select_a_file(inputs, case):
    metadata, record, _, _ = inputs
    if case == "identity":
        record["match_id"] += 1
    elif case == "file_kind":
        record["files"][0]["kind"] = "unknown"
    else:
        record["files"].append(record["files"][0])
    with pytest.raises(ValueError, match="identity or file kinds"):
        subject.inspect_match(metadata, record, "unused.csv")


def test_run_preserves_raw_target_denominator_in_partial_download(tmp_path, inputs):
    metadata, record, _, _ = inputs
    earlier = {**metadata, "id": 122838, "scheduledDate": "2023-08-18T18:30:00Z"}
    catalogue = [earlier, metadata]
    source = tmp_path / "data/impect/season_test"
    put(tmp_path / "data/impect/feasibility_20261023/matches_743.json", catalogue)
    put(source / "selection.json", {"selected_matches": catalogue})
    put(source / "matches/123087.json", record)
    result = subject.run(tmp_path, source)
    inspection = subject.read(subject.Path(result["output_directory"]) / "inspection.json")
    assert inspection["expected_catalogue_matches"] == 2
    assert inspection["expected_target_matches"] == inspection["eligible_target_matches"] == 1
    assert inspection["statuses"] == {"download_incomplete": 1, "eligible": 1}
    assert inspection["fits"] == 0


@pytest.mark.parametrize("case", ["duplicate", "modified"])
def test_selection_must_match_actual_catalogue_not_only_id_set(tmp_path, case):
    metadata = {"id": 1, "scheduledDate": "2024-01-01T12:00:00Z"}
    source = tmp_path / "source"
    put(tmp_path / "data/impect/feasibility_20261023/matches_743.json", [metadata])
    selected = [metadata, metadata] if case == "duplicate" else [{**metadata, "scheduledDate": "2024-01-02T12:00:00Z"}]
    put(source / "selection.json", {"selected_matches": selected})
    with pytest.raises(ValueError, match="selection"):
        subject.run(tmp_path, source)
