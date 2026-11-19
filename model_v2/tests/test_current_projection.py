"""Current date-only source adapters, without fitting or downloading anything."""

from dataclasses import replace
from datetime import date, datetime, timezone
import json

import pytest

from football_model.current_projection import (
    ALIASES,
    convert_current,
    load_current_context,
    team_descriptive,
)


CUTOFF = datetime(2026, 10, 23, 17, tzinfo=timezone.utc)


def source_row(**changes):
    return {
        "observed_at": "2026-10-23T16:00:00Z",
        "match_date": "2026-10-18",
        "home_label": "Manchester United FC",
        "away_label": "AFC Bournemouth",
        "full_time": [2, 1],
        "source_url": "https://example.test/official/result",
    } | changes


def write_context(directory, records=(), supplements=(), **inspection_changes):
    inspection = {
        "rejected_rows": [],
        "duplicate_keys": [],
        "records": list(records),
    } | inspection_changes
    supplement_payload = {"records": list(supplements)}
    (directory / "inspection.json").write_text(
        json.dumps(inspection), encoding="utf-8"
    )
    (directory / "supplements.json").write_text(
        json.dumps(supplement_payload), encoding="utf-8"
    )
    return inspection, supplement_payload


@pytest.mark.parametrize("label,expected", [
    ("AFC Bournemouth", "Bournemouth"),
    ("Arsenal FC", "Arsenal"),
    ("Aston Villa FC", "Aston Villa"),
    ("Brentford FC", "Brentford"),
    ("Brighton & Hove Albion FC", "Brighton"),
    ("Chelsea FC", "Chelsea"),
    ("Coventry City FC", "Coventry"),
    ("Crystal Palace FC", "Crystal Palace"),
    ("Everton FC", "Everton"),
    ("Fulham FC", "Fulham"),
    ("Hull City AFC", "Hull"),
    ("Ipswich Town FC", "Ipswich"),
    ("Leeds United FC", "Leeds"),
    ("Liverpool FC", "Liverpool"),
    ("Manchester City FC", "Man City"),
    ("Manchester United FC", "Man United"),
    ("Newcastle United FC", "Newcastle"),
    ("Nottingham Forest FC", "Nott'm Forest"),
    ("Sunderland AFC", "Sunderland"),
    ("Tottenham Hotspur FC", "Tottenham"),
])
def test_explicit_source_aliases_preserve_known_identity(label, expected):
    opponent = "Arsenal FC" if label != "Arsenal FC" else "Chelsea FC"
    record = convert_current(
        source_row(home_label=label, away_label=opponent), "synthetic.json", CUTOFF
    )
    assert ALIASES[label] == expected
    assert record.home_label == expected
    assert record.home == f"football-data:E0:{expected}"
    assert record.away == f"football-data:E0:{ALIASES[opponent]}"
    assert record.match_id == (
        f"football-data:E0:2026/27:{expected}|{ALIASES[opponent]}"
    )


@pytest.mark.parametrize("label", [
    "Manchester United", "manchester united fc", "Manchester United FC ",
    "Unknown FC",
])
def test_unresolved_alias_is_not_fuzzily_guessed(label):
    with pytest.raises(KeyError):
        convert_current(source_row(away_label=label), "synthetic.json", CUTOFF)


def test_record_preserves_date_only_and_source_without_inventing_metadata():
    row = source_row(full_time=(0, 0))
    record = convert_current(row, "synthetic.json", CUTOFF)
    assert record.competition == "E0" and record.season == "2026/27"
    assert record.match_date == date(2026, 10, 18)
    assert record.full_time == (0, 0)
    assert record.source_path == "synthetic.json"
    assert record.source_url == row["source_url"]
    assert record.source_time_text is None and record.half_time is None
    assert record.markets == {} and record.goal_fit_exclusion is None
    assert row["full_time"] == (0, 0)


@pytest.mark.parametrize("observed", [
    "2026-10-23T17:00:00Z", "2026-10-24T01:00:00+08:00",
])
def test_source_observed_exactly_at_cutoff_is_available(observed):
    assert convert_current(
        source_row(observed_at=observed), "synthetic.json", CUTOFF
    ).full_time == (2, 1)


def test_future_source_observation_is_rejected():
    with pytest.raises(ValueError, match="not observed"):
        convert_current(
            source_row(observed_at="2026-10-23T17:00:00.000001Z"),
            "synthetic.json", CUTOFF,
        )


@pytest.mark.parametrize("observed", [
    "2026-10-23T16:00:00", "2026-10-23", "not-a-time", None,
])
def test_source_observation_needs_real_timezone_aware_time(observed):
    with pytest.raises(ValueError, match="timezone-aware"):
        convert_current(source_row(observed_at=observed), "synthetic.json", CUTOFF)


@pytest.mark.parametrize("match_date", ["2026-10-23", "2026-10-24"])
def test_same_calendar_day_and_later_results_are_not_assumed_available(match_date):
    with pytest.raises(ValueError, match="strictly earlier calendar-date"):
        convert_current(source_row(match_date=match_date), "synthetic.json", CUTOFF)


@pytest.mark.parametrize("match_date", ["2026-02-30", "2026-10-18T12:00:00Z"])
def test_match_date_must_be_valid_date_without_inferred_kickoff(match_date):
    with pytest.raises(ValueError):
        convert_current(source_row(match_date=match_date), "synthetic.json", CUTOFF)


@pytest.mark.parametrize("score", [
    None, [], [1], [1, 2, 3], {"home": 1, "away": 2}, "1-2",
    [-1, 0], [0, -1], [True, 0], [1, False], [1.0, 0], [1, "0"],
])
def test_goals_require_exactly_two_nonnegative_integers(score):
    with pytest.raises(ValueError, match="two nonnegative integer goals"):
        convert_current(source_row(full_time=score), "synthetic.json", CUTOFF)


def test_loading_keeps_completed_matches_and_adds_supplements(tmp_path):
    completed = source_row()
    supplement = source_row(
        home_label="Leeds United FC", away_label="Manchester United FC",
        full_time=[1, 1],
    )
    # Missing full-time scores are ignored, not converted to zero scores. Their
    # unavailable/future metadata must not accidentally enter the fit either.
    unfinished = source_row(
        full_time=None, match_date="2026-10-25", observed_at="2026-10-26T00:00:00Z",
        home_label="Unresolved fixture label",
    )
    inspection, supplements = write_context(
        tmp_path, records=[completed, unfinished], supplements=[supplement]
    )
    rows, actual_inspection, actual_supplements = load_current_context(tmp_path, CUTOFF)
    assert len(rows) == 2
    assert [row.full_time for row in rows] == [(2, 1), (1, 1)]
    assert rows[0].source_path == str(tmp_path / "en.1.json")
    assert rows[1].source_path == str(tmp_path / "supplements.json")
    assert actual_inspection == inspection and actual_supplements == supplements


@pytest.mark.parametrize("field,value", [
    ("rejected_rows", ["invalid score"]),
    ("duplicate_keys", ["same fixture"]),
])
def test_loading_does_not_silently_ignore_unresolved_source_errors(tmp_path, field, value):
    write_context(tmp_path, records=[source_row()], **{field: value})
    with pytest.raises(ValueError, match="unresolved invalid or duplicate"):
        load_current_context(tmp_path, CUTOFF)


@pytest.mark.parametrize("placement", ["within_source", "across_sources"])
def test_repeated_fixture_is_not_counted_twice_even_if_score_conflicts(tmp_path, placement):
    first, repeated = source_row(), source_row(full_time=[9, 9])
    if placement == "within_source":
        write_context(tmp_path, records=[first, repeated])
    else:
        write_context(tmp_path, records=[first], supplements=[repeated])
    with pytest.raises(ValueError, match="not double counting"):
        load_current_context(tmp_path, CUTOFF)


def test_loading_rejects_future_supplement_observation(tmp_path):
    write_context(tmp_path, supplements=[
        source_row(observed_at="2026-10-23T17:00:01Z")
    ])
    with pytest.raises(ValueError, match="not observed"):
        load_current_context(tmp_path, CUTOFF)


def test_loading_accepts_empty_available_history_without_making_up_matches(tmp_path):
    write_context(tmp_path)
    rows, _, _ = load_current_context(tmp_path, CUTOFF)
    assert rows == []


def sample_team_records():
    details = [
        ("2026-10-01", "Manchester United FC", "Arsenal FC", [2, 1]),
        ("2026-10-04", "Chelsea FC", "Manchester United FC", [0, 2]),
        ("2026-10-08", "Tottenham Hotspur FC", "Manchester United FC", [1, 1]),
        ("2026-10-12", "Manchester United FC", "AFC Bournemouth", [0, 3]),
        ("2026-10-18", "Manchester United FC", "Liverpool FC", [0, 0]),
    ]
    return [convert_current(
        source_row(match_date=day, home_label=home, away_label=away, full_time=score),
        "synthetic.json", CUTOFF,
    ) for day, home, away, score in details]


def test_team_statistics_account_for_away_orientation_and_filter_season():
    records = sample_team_records()
    other_season = replace(records[0], season="2025/26", full_time=(99, 0))
    unrelated = convert_current(source_row(
        home_label="Leeds United FC", away_label="Sunderland AFC", full_time=[7, 8]
    ), "synthetic.json", CUTOFF)
    result = team_descriptive([unrelated, other_season, *reversed(records)], "Man United", "2026/27")
    assert {key: result[key] for key in (
        "played", "won", "drawn", "lost", "goals_for", "goals_against", "points"
    )} == {
        "played": 5, "won": 2, "drawn": 2, "lost": 1,
        "goals_for": 5, "goals_against": 5, "points": 8,
    }
    assert result["team"] == "Man United" and result["season"] == "2026/27"
    assert [game["date"] for game in result["results"]] == [
        record.match_date.isoformat() for record in records
    ]
    # Descriptive source scores retain the reported home/away orientation.
    assert result["results"][1]["score"] == [0, 2]
    assert result["results"][1]["source"] == records[1].source_url


def test_other_team_view_reverses_goal_direction():
    result = team_descriptive(sample_team_records(), "Bournemouth", "2026/27")
    assert result["played"] == result["won"] == 1
    assert result["drawn"] == result["lost"] == 0
    assert result["goals_for"] == result["points"] == 3
    assert result["goals_against"] == 0


def test_empty_team_history_reports_zero_counts_and_no_fabricated_games():
    result = team_descriptive(sample_team_records(), "Man United", "2024/25")
    assert result == {
        "team": "Man United", "season": "2024/25", "played": 0,
        "won": 0, "drawn": 0, "lost": 0, "goals_for": 0,
        "goals_against": 0, "points": 0, "results": [],
    }
