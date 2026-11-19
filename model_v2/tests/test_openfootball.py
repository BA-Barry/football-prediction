from datetime import datetime, timezone

import pytest

from football_model.openfootball import inspect_payload, parse_match


NOW = datetime(2026, 10, 22, 8, tzinfo=timezone.utc)
OPTIONS = {"competition": "en.1", "season": "2024-25", "source_url": "https://example.test/fixture.json", "observed_at": NOW}


def match(**changes):
    return {"round": "Matchday 1", "date": "2024-08-16", "time": "20:00",
            "team1": "Team H", "team2": "Team A", "score": {"ht": [0, 0], "ft": [1, 0]}, **changes}


def test_real_shape_record_keeps_clock_timezone_and_publication_unknown():
    row = parse_match(match(), **OPTIONS)
    assert row.home_label == "Team H" and row.full_time == (1, 0)
    assert row.source_time_text == "20:00"
    assert row.kickoff_utc is None and row.historical_first_published_at is None
    assert row.observed_at == NOW
    assert row.timezone_status == "unverified"
    assert row.as_record()["match_date"] == "2024-08-16"


@pytest.mark.parametrize("value", [None, {}, {"ft": None}, {"ht": [0, 0]}, {"et": [2, 1], "p": [4, 3]}])
def test_missing_full_time_is_unknown_not_zero_or_extra_time(value):
    row = parse_match(match(score=value), **OPTIONS)
    assert row.full_time is None


@pytest.mark.parametrize("value", [[], [0], [1, 0, 0], [True, 0], [-1, 0], [0.5, 0], "1-0"])
def test_bad_scores_are_reported_not_silently_converted(value):
    result = inspect_payload({"matches": [match(score={"ft": value})]}, **OPTIONS)
    assert result["rejected_rows"] == 1
    assert result["full_time_rows"] == 0
    assert len(result["issues"]) == 1


def test_missing_clock_preserves_date_and_no_fabricated_midnight():
    row = match()
    del row["time"]
    parsed = parse_match(row, **OPTIONS)
    assert parsed.source_time_text is None and parsed.kickoff_utc is None


def test_missing_halftime_and_impossible_halftime():
    assert parse_match(match(score={"ft": [1, 0]}), **OPTIONS).half_time is None
    with pytest.raises(ValueError, match="exceeds"):
        parse_match(match(score={"ht": [2, 0], "ft": [1, 0]}), **OPTIONS)


def test_duplicate_keys_and_date_changes_are_not_silently_overwritten():
    result = inspect_payload({"matches": [match(), match(date="2024-08-17", score={"ft": [2, 0]})]}, **OPTIONS)
    assert result["parsed_rows"] == 2
    assert len(result["duplicate_keys"]) == 1
    assert len(result["issues"]) == 1


def test_team_source_identity_persists_across_seasons_not_global_aliases():
    current = parse_match(match(), **OPTIONS)
    later = parse_match(match(date="2025-08-16"), **{**OPTIONS, "season": "2025-26"})
    assert current.home_key == later.home_key
    assert current.source_match_key != later.source_match_key
    assert current.home_key != parse_match(match(team1="Team H FC"), **OPTIONS).home_key


@pytest.mark.parametrize("row", [None, {}, match(date="2024-02-30"), match(time=20), match(team2="Team H")])
def test_bad_rows_rejected_with_location(row):
    result = inspect_payload({"matches": [row]}, **OPTIONS)
    assert result["issues"][0]["row_index"] == 0


def test_empty_file_and_bad_top_level():
    empty = inspect_payload({"matches": []}, **OPTIONS)
    assert empty["date_min"] is None and empty["input_rows"] == 0
    with pytest.raises(ValueError, match="matches array"):
        inspect_payload({"matches": {}}, **OPTIONS)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_match(match(), **{**OPTIONS, "observed_at": NOW.replace(tzinfo=None)})


def test_missing_results_are_reported_separately_from_parse_issues():
    result = inspect_payload({"matches": [match(round="Matchday 38", score={})]}, **OPTIONS)
    assert result["issues"] == []
    assert result["missing_full_time_rows"] == 1
    assert result["missing_half_time_rows"] == 1
    assert result["missing_full_time_rounds"] == {"Matchday 38": 1}


@pytest.mark.parametrize("value, expected_format", [({"ft": [2, 1]}, "object"), ([2, 1], "array")])
def test_reported_full_time_shapes_have_explicit_metadata(value, expected_format):
    parsed = parse_match(match(score=value), **OPTIONS)
    assert parsed.full_time == (2, 1)
    assert parsed.score_format == expected_format
    assert parsed.as_record()["score_format"] == expected_format
    assert parsed.kickoff_utc is None and parsed.historical_first_published_at is None


def test_score_array_does_not_invent_halftime_or_regulation_semantics():
    parsed = parse_match(match(score=[2, 1]), **OPTIONS)
    assert parsed.half_time is None
    report = inspect_payload({"matches": [match(score=[2, 1])]}, **OPTIONS)
    assert any("regulation-90" in note for note in report["notes"])


def test_absent_null_and_additional_time_only_shapes_remain_distinct():
    absent = match()
    del absent["score"]
    rows = [absent, match(score=None), match(score={"et": [3, 2], "p": [5, 4]})]
    report = inspect_payload({"matches": rows}, **OPTIONS)
    assert report["raw_score_formats"] == {"absent": 1, "null": 1, "object": 1}
    assert [row["score_format"] for row in report["records"]] == ["absent", "null", "object"]
    assert report["full_time_rows"] == 0


@pytest.mark.parametrize("value", [[], [0], [1, 0, 0], [True, 0], [-1, 0], [0.5, 0], "1-0", 0, False])
def test_invalid_top_level_scores_are_rejected_not_zero_filled(value):
    report = inspect_payload({"matches": [match(score=value)]}, **OPTIONS)
    assert report["rejected_rows"] == 1
    assert report["full_time_rows"] == 0


def test_raw_score_shape_counts_include_rejected_rows():
    report = inspect_payload({"matches": [match(score=[1, 0]), match(score="1-0"), None]}, **OPTIONS)
    assert report["raw_score_formats"] == {"array": 1, "string": 1, "invalid_match_row": 1}
    assert report["rejected_rows"] == 2
