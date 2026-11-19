from datetime import date, datetime

import pytest

from football_model.rolling_player_experiment import compare_supported_week, rolling_windows, static_vs_rolling, week_start


def row(mid, day):
    return {"match_id": mid, "match_date": day}


def test_week_is_monday_and_datetime_does_not_invent_a_timezone():
    assert week_start(date(2016, 2, 1)) == date(2016, 2, 1)
    assert week_start(date(2016, 2, 7)) == date(2016, 2, 1)
    with pytest.raises(ValueError, match="calendar"):
        week_start(datetime(2016, 2, 1))


def test_tuesday_cannot_train_same_week_sunday_but_enters_next_week():
    matches = [row("old", "2016-01-31"), row("tuesday", "2016-02-02"),
               row("sunday", "2016-02-07"), row("next", "2016-02-08")]
    weeks = list(rolling_windows(matches))
    assert len(weeks) == 2
    cutoff, history, targets = weeks[0]
    assert cutoff == date(2016, 2, 1)
    assert [m["match_id"] for m in history] == ["old"]
    assert [m["match_id"] for m in targets] == ["tuesday", "sunday"]
    assert [m["match_id"] for m in weeks[1][1]] == ["old", "tuesday", "sunday"]
    assert [m["match_id"] for m in weeks[1][2]] == ["next"]
    with pytest.raises(ValueError, match="Duplicate"):
        list(rolling_windows(matches + [matches[0]]))


def test_rolling_static_comparison_does_not_credit_more_easy_matches():
    def prediction(mid, loss):
        return {"match_id": mid, "match_date": "2016-02-02", "n_stints": 10,
                "point_process_loss": loss, "expected_home_target": 1., "expected_away_target": 1.,
                "actual_home_target": 1., "actual_away_target": 1., "loss_kind": "test"}
    static = {"predictions": [prediction("common", 2.)]}
    rolling = {"predictions": [prediction("common", 3.), prediction("additional", -20.)]}
    result = static_vs_rolling(static, rolling)
    assert result["common_match_ids"] == ["common"]
    assert result["mean_rolling_minus_static_loss"] == 1.


def test_entire_week_of_unknown_players_is_zero_coverage_not_a_failed_experiment():
    team = {"predictions": [{"match_id": "new-player-game"}], "skipped": []}
    player = {"predictions": [], "skipped": [{"match_id": "new-player-game", "reason": "unknown"}]}
    result = compare_supported_week(team, player)
    assert result["common_match_ids"] == []
    assert result["mean_player_minus_team_loss"] is None
    assert len(player["skipped"]) == 1
