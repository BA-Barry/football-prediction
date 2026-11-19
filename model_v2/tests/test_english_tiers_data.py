from datetime import date
import json
from pathlib import Path

import pytest

from football_model.english_tiers_data import (
    DIVISIONS, SEASONS, TEAM_ALIASES, parse_text, reinspect_saved,
    run_intake, transition_summary,
)


def parse(body, season="2025-26", competition="en.2"):
    return parse_text(body, season=season, competition=competition,
                      source_url="https://example.invalid/source", source_path="raw.txt")


def sample(line="12:30  Coventry City FC  v Hull City AFC  2-0 (1-0)", heading="Matchday 1"):
    return f"= English Championship\n# Source header\n▪ {heading}\nSat Aug 9 2025\n  {line}\n"


@pytest.mark.parametrize("line,score", [
    ("12:30  Coventry City FC  v Hull City AFC  2-0 (1-0)", [2, 0]),
    ("12:30  Coventry City FC  2-0 (1-0)  Hull City AFC", [2, 0]),
    ("Coventry City FC v Hull City AFC  0-0", [0, 0]),
    ("Coventry City FC  v Hull City AFC  12-1 (5-0)", [12, 1]),
])
def test_supported_numeric_layouts(line, score):
    result = parse(sample(line))
    assert result["issues"] == []
    row, = result["records"]
    assert row["reported_score"] == row["regulation_time_score"] == score
    assert row["eligible_regular_result"] is True
    assert row["home"] == {"source_label": "Coventry City FC", "canonical_label": "Coventry", "team_id": "england:Coventry"}
    assert row["away"]["team_id"] == "england:Hull"
    assert row["kickoff_utc"] is None
    assert row["source_url"].startswith("https://example.invalid")


def test_dates_are_completed_from_season_not_input_order_or_machine_clock():
    text = """▪ Matchday 1
Sat Aug 9
12:30  Coventry City FC  v Hull City AFC  1-0
▪ Matchday 30
Sat Jan 10
12:30  Hull City AFC  v Coventry City FC  1-1
▪ Matchday 7
Sat Sep 20
12:30  Ipswich Town FC  v Coventry City FC  0-0
"""
    result = parse(text)
    assert result["issues"] == []
    assert [x["match_date"] for x in result["records"]] == ["2025-08-09", "2026-01-10", "2025-09-20"]
    assert result["time_semantics"]["historical_first_availability"] is None
    assert result["time_semantics"]["download_time_is_historical_availability"] is False


@pytest.mark.parametrize("value", ["2025/26", "2025-27", "2001-00", "2025-6"])
def test_invalid_season_raises(value):
    with pytest.raises(ValueError):
        parse("", season=value)


@pytest.mark.parametrize("line", ["Sun Aug 9 2025", "Sat Feb 30 2026", "Sat Aug 9 2024", "Sat Flop 9", "Sat Aug 09 25"])
def test_bad_date_never_inherits_previous_date(line):
    body = sample() + line + "\n12:30  Hull City AFC  v Ipswich Town FC  1-0\n"
    result = parse(body)
    assert len(result["records"]) == 1
    assert any(x["kind"] == "match_without_valid_date" for x in result["issues"])
    assert any("date" in x["kind"] for x in result["issues"])


@pytest.mark.parametrize("clock", ["24:00", "12:60", "99:99"])
def test_invalid_clock_not_accepted(clock):
    result = parse(sample(f"{clock}  Hull City AFC  v Coventry City FC  1-0"))
    assert not result["records"]
    assert result["issues"][0]["kind"] == "invalid_clock"


@pytest.mark.parametrize("score", ["-1-0", "1-2 awarded", "2-1 postponed", "1-0 2-0", "1-0 (0-0) (0-0)", "unknown", "1-0 abandoned", "3:1"])
def test_unknown_score_statuses_are_audited_not_interpreted(score):
    result = parse(sample(f"Hull City AFC  v Coventry City FC  {score}"))
    assert result["records"] == []
    assert len(result["issues"]) == 1
    assert "score" in result["issues"][0]["kind"]
    assert score in result["issues"][0]["raw"]


def test_unknown_team_is_retained_with_no_invented_alias():
    result = parse(sample("Hull City AFC  v Fake Team FC  1-0"))
    row, = result["records"]
    assert row["away"]["team_id"] is None
    assert not row["eligible_regular_result"]
    assert result["issues"][0]["kind"] == "unknown_team_alias"


@pytest.mark.parametrize("heading", ["Playoffs", "Finals", "Finals, Semifinals", "Play-off Final"])
@pytest.mark.parametrize("score", ["1-0", "1-1 a.e.t. (0-1, 0-0)", "5-6 pen. 1-1 a.e.t. (1-1, 0-1)"])
def test_playoff_and_extended_scores_never_supply_regular_rows(heading, score):
    result = parse(sample(f"Coventry City FC  v Hull City AFC  {score}", heading=heading))
    row, = result["records"]
    assert row["stage"] == "playoff"
    assert row["eligible_regular_result"] is False
    assert row["raw_score_text"] == score
    assert row["neutral_status"] == "unknown"
    if "a.e.t." in score:
        assert row["regulation_time_score"] is None
        assert row["reported_score"] == [1, 1]
    else:
        assert row["regulation_time_score"] == [1, 0]
    assert row["extra_time"] == ("a.e.t." in score)
    assert row["penalties"] == ("pen." in score)


def test_extra_time_in_regular_stage_is_visible_but_ineligible():
    result = parse(sample("Hull City AFC  v Coventry City FC  2-1 a.e.t. (1-1)"))
    assert not result["records"][0]["eligible_regular_result"]
    assert result["issues"][0]["kind"] == "extra_time_in_regular_stage"


def test_unknown_stage_is_not_assumed_regular():
    result = parse(sample(heading="Relegation tie"))
    assert result["records"][0]["stage"] == "unknown"
    assert not result["records"][0]["eligible_regular_result"]
    assert result["issues"][0]["kind"] == "unrecognized_stage"


def test_annotation_text_is_retained_not_treated_as_player_events():
    body = sample() + "                  (Player One 90+4';\n                   Player Two 11')\n"
    result = parse(body)
    assert result["issues"] == []
    assert len(result["records"]) == 1
    assert len(result["records"][0]["source_annotations"]) == 2
    assert "90+4'" in result["records"][0]["source_annotations"][0]["raw"]


def test_unterminated_annotation_is_reported():
    result = parse(sample() + "                  (Player One 90+4'\n")
    assert result["issues"][0]["kind"] == "unterminated_annotation_at_eof"


def test_missing_status_lines_do_not_disappear():
    result = parse(sample() + "This unfamiliar format was not interpreted\n")
    assert result["issues"][0]["raw"] == "This unfamiliar format was not interpreted"


def test_half_time_conflict_retains_observation_without_eligibility():
    result = parse(sample("Hull City AFC  v Coventry City FC  0-0 (1-0)"))
    assert result["records"][0]["reported_score"] == [0, 0]
    assert not result["records"][0]["eligible_regular_result"]
    assert result["issues"][0]["kind"] == "half_time_exceeds_full_time"


def test_duplicate_source_rows_are_not_overwritten():
    line = "12:30  Hull City AFC  v Coventry City FC  1-0"
    result = parse(sample(line) + line)
    assert len(result["records"]) == 2
    assert len(result["duplicate_keys"]) == 1


def test_team_identity_is_shared_across_division_but_match_keys_are_distinct():
    first = parse(sample(), competition="en.1")
    second = parse(sample(), competition="en.2")
    assert first["records"][0]["home"]["team_id"] == second["records"][0]["home"]["team_id"]
    assert first["records"][0]["match_id"] != second["records"][0]["match_id"]
    assert TEAM_ALIASES["Manchester United"] == TEAM_ALIASES["Manchester United FC"]


def test_transition_summary_describes_only_observed_adjacent_seasons():
    first = parse("▪ Matchday 1\nSat Aug 17 2024\nHull City AFC  v Ipswich Town FC  1-0", season="2024-25", competition="en.1")
    second = parse(sample("Hull City AFC  v Ipswich Town FC  1-0"))
    result = transition_summary([first, second])
    assert result["focus_team_membership"]["england:Ipswich"] == {"2024-25": ["en.1"], "2025-26": ["en.2"]}
    assert len(result["adjacent_season_division_changes"]) == 2
    assert result["model_fit_performed"] is False
    assert "Disconnected" in result["identifiability_note"]


def test_narrow_intake_and_offline_reinspection_keep_raws_and_times(tmp_path):
    calls = []
    def fake_fetch(url):
        calls.append(url)
        if url.endswith("LICENSE.md"):
            return b"CC0 1.0 Universal", {"status": 200}
        season = next(s for s in SEASONS if s in url)
        start = int(season[:4])
        weekday = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[date(start, 8, 9).weekday()]
        text = f"▪ Matchday 1\n{weekday} Aug 9 {start}\nHull City AFC  v Ipswich Town FC  1-0\n"
        return text.encode(), {"status": 200, "http_date": "test source response date"}
    result, directory = run_intake(tmp_path, fetch=fake_fetch)
    assert len(calls) == 9
    assert len(list(directory.glob("*.txt"))) == 8
    assert (directory / "LICENSE.md").read_bytes() == b"CC0 1.0 Universal"
    assert all(x["received_at"] for x in result["files"])
    assert all(x["git_ref"] == "master" for x in result["files"])
    assert result["trained"] is result["existing_forecasts_modified"] is False
    original = (directory / "2025-26_en.2.txt").read_bytes()
    recheck, out = reinspect_saved(directory)
    assert len(calls) == 9
    assert recheck["new_network_requests"] == 0
    assert out.parent == directory
    assert (directory / "2025-26_en.2.txt").read_bytes() == original
    assert (directory / "2025-26_en.2.inspection.json").exists()
    assert (out / "coverage_summary.json").exists()


def test_reinspection_refuses_source_path_outside_intake(tmp_path):
    raw = tmp_path / "data"
    raw.mkdir()
    meta = {"files": [{"competition": "en.2", "path": str(tmp_path / "elsewhere.txt")}]}
    (raw / "intake_summary.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        reinspect_saved(raw)
