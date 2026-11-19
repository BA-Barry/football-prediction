from copy import deepcopy
import csv

import pytest

from football_model.impect_pilot import independent_score


def score_file(tmp_path, rows):
    path = tmp_path / "D1.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def row():
    return {"Div": "D1", "Date": "18/08/2023", "HomeTeam": "Werder Bremen", "AwayTeam": "Bayern Munich",
            "FTHG": "0", "FTAG": "4", "HTHG": "0", "HTAG": "1"}


def metadata():
    return {"id": 122838, "scheduledDate": "2023-08-18T18:30:00Z", "homeSquadId": 38, "awaySquadId": 33}


def test_exact_join_includes_independent_halftime_and_physical_csv_line(tmp_path):
    result = independent_score(metadata(), score_file(tmp_path, [row()]))
    assert result["full_time"] == [0, 4]
    assert result["half_time"] == [0, 1]
    assert result["csv_line"] == 2


@pytest.mark.parametrize("change", ["reverse", "wrong_date", "unknown_id", "naive_date"])
def test_no_fuzzy_or_implicit_timezone_matching(tmp_path, change):
    match = metadata()
    if change == "reverse":
        match["homeSquadId"], match["awaySquadId"] = 33, 38
    elif change == "wrong_date":
        match["scheduledDate"] = "2023-08-19T18:30:00Z"
    elif change == "unknown_id":
        match["homeSquadId"] = 999
    else:
        match["scheduledDate"] = "2023-08-18T18:30:00"
    with pytest.raises(ValueError):
        independent_score(match, score_file(tmp_path, [row()]))


def test_duplicate_scores_do_not_pick_first(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        independent_score(metadata(), score_file(tmp_path, [row(), row()]))


def test_wrong_division_not_accepted(tmp_path):
    record = row()
    record["Div"] = "E0"
    with pytest.raises(ValueError, match="Bundesliga"):
        independent_score(metadata(), score_file(tmp_path, [record]))


def test_only_date_and_names_are_used_not_source_kickoff_clock(tmp_path):
    match = deepcopy(metadata())
    match["scheduledDate"] = "2023-08-18T20:30:00+02:00"
    assert independent_score(match, score_file(tmp_path, [row()]))["source_date"] == "18/08/2023"


def test_explicit_extended_mapping_does_not_expand_the_default_pilot(tmp_path):
    match = metadata()
    match["homeSquadId"] = 432
    record = row()
    record["HomeTeam"] = "Heidenheim"
    path = score_file(tmp_path, [record])
    with pytest.raises(ValueError, match="mapping"):
        independent_score(match, path)
    assert independent_score(match, path, team_names={432: "Heidenheim", 33: "Bayern Munich"})["full_time"] == [0, 4]
