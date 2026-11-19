import csv
from datetime import date
from types import SimpleNamespace

import pytest

from football_model.player_probe import coverage_simulation, reconcile_match, write_minutes_csv


def match(**changes):
    fields = dict(match_id="fd:one", competition="D1", season="2023/24",
                  match_date=date(2023, 8, 19), home_label="Leverkusen", away_label="RB Leipzig",
                  home="D1:A", away="D1:B", full_time=(3, 2), goal_fit_exclusion=None)
    return SimpleNamespace(**(fields | changes))


def metadata(**changes):
    return dict(home_team={"home_team_id": 904}, away_team={"away_team_id": 182},
                match_date="2023-08-19", home_score=3, away_score=2) | changes


def test_cross_provider_link_requires_explicit_ids_date_home_away_and_equal_score():
    record = match()
    assert reconcile_match(metadata(), [record]) is record
    with pytest.raises(ValueError, match="score conflict"):
        reconcile_match(metadata(home_score=2), [record])
    with pytest.raises(ValueError, match="explicit"):
        reconcile_match(metadata(home_team={"home_team_id": 99999}), [record])


@pytest.mark.parametrize("records", [[], [match(), match()],
    [match(match_date=date(2023, 8, 20))], [match(home_label="RB Leipzig", away_label="Leverkusen")],
    [match(season="2024/25")], [match(competition="E0")]])
def test_no_date_only_fuzzy_reverse_or_duplicated_join(records):
    with pytest.raises(ValueError, match="one exact"):
        reconcile_match(metadata(), records)


def test_frequency_simulation_never_invents_first_match_strength_or_uses_excluded_result():
    records = [match(match_id="old", match_date=date(2023, 8, 1)),
        match(match_id="first_C", match_date=date(2024, 8, 16), home="D1:C"),
        match(match_id="second_C", match_date=date(2024, 8, 24), home="D1:C"),
        match(match_id="excluded_D", match_date=date(2024, 8, 20), home="D1:D",
              goal_fit_exclusion="administratively_awarded_result"),
        match(match_id="still_unknown_D", match_date=date(2024, 8, 27), home="D1:D")]
    report = coverage_simulation(records)
    summary = report["seasons"]["2024/25"]["totals"]
    assert summary == {"eligible_target_matches": 3, "known_team_targets_28d": 0,
                       "known_team_targets_7d": 1}
    assert report["not_model_performance"] and not report["model_or_prior_changed"]
    unknown = report["seasons"]["2024/25"]["leagues"]["D1"]["unknown_team_matches_7d"]
    assert unknown == ["first_C", "still_unknown_D"]


def test_exposure_csv_does_not_require_forecast_fields_or_hide_zero_minutes(tmp_path):
    path = tmp_path / "minutes.csv"
    write_minutes_csv(path, [{"player_id": "statsbomb:player:1", "minutes": 0,
                              "football_data_match_id": None}])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"player_id": "statsbomb:player:1", "minutes": "0",
                     "football_data_match_id": ""}]
    with pytest.raises(ValueError):
        write_minutes_csv(tmp_path / "empty.csv", [])
