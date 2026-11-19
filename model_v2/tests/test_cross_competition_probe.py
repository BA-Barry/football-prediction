from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from football_model.cross_competition_probe import (
    ALIASES, COMPETITIONS, measure_bridges, parse_cup_text,
    resolve_domestic_team, run_probe,
)


NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)
URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/2023-24/cl.txt"


def inspect(text, season="2023-24", code="cl"):
    return parse_cup_text(text, competition=code, season=season, source_url=URL, observed_at=NOW)


def test_modern_file_preserves_clock_raw_source_and_date_basis():
    raw = "    21:00  Arsenal FC (ENG) v AC Milan (ITA) 2-1 (1-1)"
    result = inspect("= UEFA Champions League 2023/24\n# Matches 1\n▪ Group A\nTue Sep 19 2023\n" + raw)
    assert result["parsed_rows"] == 1 and result["header_count_difference"] == 0
    row = result["records"][0]
    assert row["raw_line"] == raw and row["source_line_number"] == 5
    assert row["source_url"] == URL and row["observed_at"] == NOW.isoformat()
    assert row["match_date"] == "2023-09-19" and row["year_basis"] == "explicit_date_line"
    assert row["source_clock_text"] == "21:00" and row["kickoff_utc"] is None
    assert row["historical_first_published_at"] is None and row["neutral_venue"] is None
    assert row["reported_unmarked_score"] == [2, 1] and row["reported_unmarked_half_time"] == [1, 1]
    assert row["verified_regulation_90"] is False and row["regulation_90_score"] is None


def test_same_season_month_window_not_line_order_prevents_false_year_rollover():
    result = inspect("▪ Group A\nThu Dec 14\nArsenal FC (ENG) v AC Milan (ITA) 2-0\n"
                     "▪ Group B\nTue Sep 19\nArsenal FC (ENG) v AS Roma (ITA) 1-0\n"
                     "▪ Playoffs\nThu Feb 22\nAS Roma (ITA) v Arsenal FC (ENG) 0-1")
    assert [r["match_date"] for r in result["records"]] == ["2023-12-14", "2023-09-19", "2024-02-22"]
    assert {r["year_basis"] for r in result["records"]} == {"derived_season_month_window"}
    assert result["weekday_conflict_rows"] == 0


@pytest.mark.parametrize("text,unknown", [
    ("Arsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
    ("Tue Sep 19\n▪ New Group\nArsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
    ("Tue Sep 19\nWed Xxx 20\nArsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
    ("Tue Sep 19\nWed Sep 99\nArsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
    ("Tue Sep 19\nWed Sep 20 2025\nArsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
    ("Tue Sep 19\nWed September 20\nArsenal FC (ENG) v AC Milan (ITA) 1-0", 1),
])
def test_bad_or_missing_date_context_is_not_inherited_or_fabricated(text, unknown):
    result = inspect(text)
    assert result["date_unknown_rows"] == unknown
    assert result["records"][0]["match_date"] is None


def test_weekday_conflict_retained_and_counted_not_silently_corrected():
    result = inspect("Mon Sep 19 2023\nArsenal FC (ENG) v AC Milan (ITA) 1-0")
    assert result["weekday_conflict_rows"] == 1
    assert result["context_issues"][0]["issue"] == "weekday_mismatch"
    assert result["records"][0]["source_date_line"] == "Mon Sep 19 2023"


@pytest.mark.parametrize("score,kind,pairs,penalties", [
    ("5-4 a.e.t. (2-2, 2-0)", "extra_time_annotation", [[5, 4], [2, 2], [2, 0]], None),
    ("3-2 pen. 0-2 a.e.t. (0-2, 0-1)", "penalty_shootout_annotation", [[3, 2], [0, 2], [0, 2], [0, 1]], [3, 2]),
    ("4-2 pen. 1-1 a.e.t. (1-1, 1-1)", "penalty_shootout_annotation", [[4, 2], [1, 1], [1, 1], [1, 1]], [4, 2]),
    ("1-0 a.e.t. (0-0)", "extra_time_annotation", [[1, 0], [0, 0]], None),
    ("3-1 (2-1)", "reported_unmarked_score", [[3, 1], [2, 1]], None),
    ("0-0", "reported_unmarked_score", [[0, 0]], None),
    ("", "missing_score", [], None),
    ("postponed", "unsupported_score_annotation", [], None),
    ("2-1 awarded", "unsupported_score_annotation", [[2, 1]], None),
    ("1-0 (2-0)", "inconsistent_unmarked_halftime", [[1, 0], [2, 0]], None),
])
def test_score_annotations_keep_all_numbers_never_make_90min(score, kind, pairs, penalties):
    result = inspect("Tue Sep 19\nArsenal FC (ENG) v AC Milan (ITA) " + score)
    row = result["records"][0]
    assert row["score_kind"] == kind and row["source_score_pairs_in_order"] == pairs
    assert row["reported_penalty_shootout_score"] == penalties
    assert row["regulation_90_score"] is None and row["verified_regulation_90"] is False
    if kind != "reported_unmarked_score":
        assert row["reported_unmarked_score"] is row["reported_unmarked_half_time"] is None


def test_source_venue_not_automatically_neutral_or_home_and_explicit_leg_retained():
    row = inspect("▪ Quarter-finals - 2nd Leg\nTue Sep 19\n"
                  "21:00 Arsenal FC (ENG) v AC Milan (ITA) 1-0 @ Example Stadium, London")["records"][0]
    assert row["source_venue_text"] == "Example Stadium, London"
    assert row["neutral_venue"] is None and row["home_advantage_verified"] is False
    assert row["explicit_leg_number"] == 2 and row["tie_format_status"] == "explicit_leg_label"


def test_round_not_explicit_leg_and_missing_clock_not_filled_from_previous_row():
    rows = inspect("▪ Finals, Final\nTue Sep 19\n21:00 Arsenal FC (ENG) v AC Milan (ITA) 1-0\n"
                   "AS Roma (ITA) v Liverpool FC (ENG) 2-0")["records"]
    assert rows[0]["explicit_leg_number"] is None and rows[0]["neutral_venue"] is None
    assert rows[1]["source_clock_text"] is None


@pytest.mark.parametrize("line", [
    "25:00 Arsenal FC (ENG) v AC Milan (ITA) 1-0",
    "21:60 Arsenal FC (ENG) v AC Milan (ITA) 1-0",
    "Arsenal FC (ENG) v Arsenal FC (ENG) 1-0",
    "Arsenal FC v AC Milan 1-0",
    "21:00 Arsenal FC 1-0 AC Milan",
])
def test_unsupported_or_bad_match_rows_retained_as_rejections(line):
    result = inspect("Tue Sep 19\n" + line)
    assert result["parsed_rows"] == 0 and len(result["rejected_match_lines"]) == 1
    assert result["rejected_match_lines"][0]["raw_line"] == line


def test_header_count_and_duplicates_visible_and_not_independent_proof():
    line = "Arsenal FC (ENG) v AC Milan (ITA) 1-0"
    result = inspect("# Matches 3\nTue Sep 19\n" + line + "\n" + line)
    assert result["parsed_rows"] == 2 and result["header_count_difference"] == -1
    assert len(result["duplicate_keys"]) == 1 and len(result["records"]) == 2


@pytest.mark.parametrize("season", ["2023/24", "2023-25", "2024-24", "x", "2023-2024"])
def test_invalid_season_rejected(season):
    with pytest.raises(ValueError):
        inspect("", season=season)


def test_observation_time_must_be_aware():
    with pytest.raises(ValueError):
        parse_cup_text("", competition="cl", season="2023-24", source_url=URL,
                       observed_at=datetime(2026, 10, 23))


def test_unknown_competition_rejected():
    with pytest.raises(ValueError):
        inspect("", code="super")


def test_exact_and_explicit_alias_methods_country_scoped_no_fuzzy_matching():
    teams = {"E0": {"Arsenal", "Aston Villa"}, "I1": {"Atalanta"}, "F1": {"Monaco"}}
    assert resolve_domestic_team("Aston Villa", "ENG", teams)["status"] == "exact_country_scoped_label"
    assert resolve_domestic_team("Arsenal FC", "ENG", teams)["status"] == "explicit_reviewed_alias"
    assert resolve_domestic_team("arsenal fc", "ENG", teams)["status"] == "unmapped_source_name"
    assert resolve_domestic_team("Arsenal", "ARG", teams)["status"] == "outside_five_league_country_scope"
    assert resolve_domestic_team("Arsenal FC", "ITA", teams)["status"] == "unmapped_source_name"
    assert resolve_domestic_team("AS Monaco FC", "MCO", teams)["team_key"] == "football-data:F1:Monaco"
    missing = resolve_domestic_team("AC Milan", "ITA", teams)
    assert missing["status"] == "target_not_in_same_season_domestic_file" and missing["team_key"] is None


def test_bridge_edges_require_both_actual_domestic_targets_and_distinct_leagues():
    source = inspect("Tue Sep 19\nArsenal FC (ENG) v AC Milan (ITA) 1-0\n"
                     "Aston Villa (ENG) v Arsenal FC (ENG) 1-0\n"
                     "PSV (NED) v AS Roma (ITA) 1-0\n"
                     "ACF Fiorentina (ITA) v Feyenoord (NED) 1-0\n"
                     "AS Monaco FC (MCO) v Arsenal FC (ENG) 1-0")
    result = measure_bridges(source, {"E0": {"Arsenal", "Aston Villa"}, "I1": {"Milan", "Roma"}, "F1": {"Monaco"}})
    assert result["both_teams_linked_rows"] == 3 and result["one_team_linked_rows"] == 1
    assert result["neither_team_linked_rows"] == 1 and result["cross_domestic_league_rows"] == 2
    assert result["cross_domestic_league_edges"] == [
        {"leagues": ["E0", "F1"], "match_rows": 1}, {"leagues": ["E0", "I1"], "match_rows": 1}]
    assert result["verified_regulation_90_rows"] == 0


def domestic_file(directory):
    directory.mkdir()
    (directory / "E0.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,19/09/2023,Arsenal,Aston Villa,1,0,H\n", encoding="utf-8")
    return directory


def fake_fetch(url):
    if url.endswith(".txt"):
        code = Path(url).stem
        text = (f"= {COMPETITIONS[code]} 2023/24\n# Matches 1\n▪ Group A\nTue Sep 19 2023\n"
                "Arsenal FC (ENG) v Aston Villa (ENG) 1-0\n")
        return text.encode(), {"Last-Modified": "Fri, 23 Oct 2026 00:00:00 GMT"}
    return b"Upstream documentation fixture", {}


def test_run_downloads_small_files_writes_reports_and_never_trains(tmp_path):
    directory = domestic_file(tmp_path / "domestic")
    result, report = run_probe(tmp_path, {"2023-24": directory}, fetch=fake_fetch)
    assert result["status"] == "complete" and result["trained"] is False
    assert result["predictions_generated"] is False and result["totals"]["parsed_rows"] == 3
    assert result["totals"]["both_teams_linked_rows"] == 3
    assert result["totals"]["cross_domestic_league_rows"] == 0
    assert len(result["sources"]) == 5
    assert Path(result["raw_directory"]).parent == tmp_path / "data" / "openfootball-cups"
    assert Path(result["result_json"]).is_file() and report.is_file()
    stored = json.loads(Path(result["result_json"]).read_text(encoding="utf-8"))
    assert stored["totals"] == result["totals"]
    assert "不是UEFA官方" in report.read_text(encoding="utf-8")
    assert stored["domestic_sources"]["2023-24"]["leagues"]["E0"]["source_url"].endswith("2324/E0.csv")
    assert len(stored["alias_entries"]) == len(ALIASES)


def test_download_failure_preserved_no_retry_or_alternative_source(tmp_path):
    directory = domestic_file(tmp_path / "domestic")
    calls = []
    def fetch(url):
        calls.append(url)
        if url.endswith("el.txt"):
            raise HTTPError(url, 403, "Forbidden", {}, None)
        return fake_fetch(url)
    result, _ = run_probe(tmp_path, {"2023-24": directory}, fetch=fetch)
    assert result["status"] == "partial" and len(calls) == len(set(calls)) == 5
    failure = next(s for s in result["sources"] if s["status"] == "error")
    assert failure["http_status"] == 403 and failure["url"].endswith("el.txt")
    assert result["totals"]["parsed_rows"] == 2


def test_decoding_error_preserves_downloaded_source_and_other_measurements(tmp_path):
    directory = domestic_file(tmp_path / "domestic")
    def fetch(url):
        return (b"\xff", {}) if url.endswith("conf.txt") else fake_fetch(url)
    result, _ = run_probe(tmp_path, {"2023-24": directory}, fetch=fetch)
    assert result["status"] == "partial" and result["totals"]["parsed_rows"] == 2
    assert result["inspection_errors"][0]["error_type"] == "UnicodeDecodeError"
    assert (Path(result["raw_directory"]) / "2023-24_conf.txt").read_bytes() == b"\xff"


def test_separate_runs_do_not_overwrite_data_or_reports(tmp_path):
    directory = domestic_file(tmp_path / "domestic")
    first, first_report = run_probe(tmp_path, {"2023-24": directory}, fetch=fake_fetch)
    second, second_report = run_probe(tmp_path, {"2023-24": directory}, fetch=fake_fetch)
    assert first["raw_directory"] != second["raw_directory"] and first_report != second_report
    assert first_report.exists() and second_report.exists()


def test_missing_domestic_inputs_not_silently_treated_as_zero_coverage(tmp_path):
    with pytest.raises(ValueError):
        run_probe(tmp_path, {}, fetch=fake_fetch)


@pytest.mark.parametrize("body", [
    b"<html>temporary error page returned with status 200</html>",
    b"= UEFA Champions League 2024/25\n",
    b"= UEFA Europa League 2023/24\n",
    b"= UEFA Champions League 2023/24\n= UEFA Europa League 2023/24\n",
    b"= UEFA Champions League 2023/24\n",
    b"= UEFA Champions League 2023/24\n# Matches 125\n",
    b"= UEFA Champions League 2023/24\nTue Sep 19 2023\nArsenal FC (ENG) v AC Milan (ITA) 1-0\n",
])
def test_wrong_or_missing_source_title_not_a_successful_zero_match_file(tmp_path, body):
    directory = domestic_file(tmp_path / "domestic")
    def fetch(url):
        return (body, {}) if url.endswith("cl.txt") else fake_fetch(url)
    result, _ = run_probe(tmp_path, {"2023-24": directory}, fetch=fetch)
    assert result["status"] == "partial" and result["totals"]["parsed_rows"] == 2
    assert result["inspection_errors"][0]["competition"] == "cl"
    assert (Path(result["raw_directory"]) / "2023-24_cl.txt").read_bytes() == body
