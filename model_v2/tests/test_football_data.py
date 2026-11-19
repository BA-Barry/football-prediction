import csv
from datetime import date

import pytest

from football_model.football_data import MARKET_COLUMNS, load_directory, parse_row


def row(**changes):
    result = {"Div": "E0", "Date": "16/08/2024", "Time": "20:00",
              "HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "2", "FTAG": "1",
              "FTR": "H", "HTHG": "1", "HTAG": "0", "HTR": "H"}
    result.update({c: v for columns in MARKET_COLUMNS.values()
                   for c, v in zip(columns, ("2.0", "3.5", "4.0"))})
    result.update(changes)
    return result


def parse(value, competition="E0", season="2024/25"):
    return parse_row(value, competition=competition, season=season,
                     source_path="synthetic/local.csv", source_url="https://example.test/source.csv")


def write_csv(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def test_date_level_record_does_not_invent_exact_timestamps():
    record, issues = parse(row())
    assert record.match_date == date(2024, 8, 16)
    assert record.source_time_text == "20:00"
    assert not hasattr(record, "kickoff") and not hasattr(record, "observed_at")
    assert record.home == "football-data:E0:Home"
    assert record.markets["avg_closing"] == (2.0, 3.5, 4.0)
    assert not issues
    assert parse(row(Date="16/08/24"))[0].match_date == record.match_date


@pytest.mark.parametrize("changes", [
    {"Date": "2024-08-16"}, {"Date": "31/02/2024"}, {"FTHG": "1.5"},
    {"FTAG": "-1"}, {"FTHG": ""}, {"FTR": "A"}, {"Div": "D1"},
    {"HTHG": "3"}, {"HTR": "D"}, {"HomeTeam": "Away"}, {"Time": "24:01"},
])
def test_bad_rows_are_not_silently_coerced(changes):
    with pytest.raises(ValueError):
        parse(row(**changes))


def test_missing_halftime_and_missing_or_invalid_market_preserved():
    record, issues = parse(row(HTAG="", AvgA="", B365CH="nan"))
    assert record.half_time is None
    assert "avg_preclosing" not in record.markets
    assert "b365_closing" not in record.markets
    assert len(issues) == 3
    assert {i["issue"] for i in issues} == {"partial_missing", "missing", "invalid_prices"}


def test_administrative_result_retains_source_and_excludes_goal_fit():
    record, _ = parse(row(Div="D1", Date="14/12/2024", HomeTeam="Union Berlin",
                          AwayTeam="Bochum", FTHG="0", FTAG="2", FTR="A",
                          HTHG="1", HTAG="1", HTR="D"), competition="D1")
    assert record.full_time == (0, 2)
    assert record.goal_fit_exclusion == "administratively_awarded_result"


@pytest.mark.parametrize("competition,day,home,away,score", [
    ("SP1", "11/12/2023", "Granada", "Ath Bilbao", (1, 1)),
    ("I1", "25/04/2024", "Udinese", "Roma", (1, 2)),
])
def test_known_partially_resumed_matches_keep_source_date_and_score_but_exclude_fit(
        competition, day, home, away, score):
    record, _ = parse(row(Div=competition, Date=day, HomeTeam=home, AwayTeam=away,
                          FTHG=str(score[0]), FTAG=str(score[1]),
                          FTR="D" if score[0] == score[1] else "A",
                          HTHG="0", HTAG="0", HTR="D"),
                      competition=competition, season="2023/24")
    assert record.full_time == score
    assert record.match_date == date(int(day[6:]), int(day[3:5]), int(day[:2]))
    assert record.goal_fit_exclusion == "resumed_partially_played_match"


@pytest.mark.parametrize('day,home,away,score,season', [
    ('16/03/2025', 'Montpellier', 'St Etienne', (0, 2), '2024/25'),
    ('17/05/2026', 'Nantes', 'Toulouse', (0, 0), '2025/26'),
])
def test_confirmed_incomplete_results_stay_readable_but_do_not_train_full_goals(day, home, away, score, season):
    from football_model.date_model import DateTeamGoalModel
    partial, _ = parse(row(Div='F1', Date=day, HomeTeam=home, AwayTeam=away,
                          FTHG=str(score[0]), FTAG=str(score[1]),
                          FTR='D' if score[0] == score[1] else 'A',
                          HTHG='0', HTAG='0', HTR='D'), competition='F1', season=season)
    ordinary, _ = parse(row(Div='F1', HomeTeam='Home', AwayTeam='Away'), competition='F1')
    assert partial.full_time == score
    assert partial.goal_fit_exclusion == 'administratively_confirmed_incomplete_match'
    fit = DateTeamGoalModel().fit([partial, ordinary], date(2026, 6, 1))
    assert fit.reference_match_ids == [ordinary.match_id]
    assert partial.home not in fit.teams and partial.away not in fit.teams


def test_local_load_reports_duplicates_rejections_and_coverage(tmp_path):
    write_csv(tmp_path / "E0.csv", [row(), row(Date="17/08/2024"), row(FTR="D")])
    records, report = load_directory(tmp_path)
    assert len(records) == 2
    assert report["leagues"]["E0"]["input_rows"] == 3
    assert report["leagues"]["E0"]["market_rows"]["avg_closing"] == 2
    assert len(report["duplicate_keys"]) == 1
    assert len(report["rejected_rows"]) == 1
    assert report["time_semantics"]["odds_observation_times"] == "unknown"


def test_team_identity_survives_season_change_and_match_identity_does_not():
    first, _ = parse(row())
    later, _ = parse(row(Date="16/08/2025"), season="2025/26")
    assert first.home == later.home
    assert first.match_id != later.match_id


def test_empty_directory_and_missing_required_header_fail(tmp_path):
    with pytest.raises(ValueError, match="No known-league"):
        load_directory(tmp_path)
    write_csv(tmp_path / "E0.csv", [{"Div": "E0"}])
    with pytest.raises(ValueError, match="missing required"):
        load_directory(tmp_path)
