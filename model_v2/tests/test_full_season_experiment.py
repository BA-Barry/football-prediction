"""Ordinary chronological/cohort tests; no claims about model accuracy."""

from copy import deepcopy
from datetime import date

import pytest

from football_model.full_season_experiment import (
    choose_penalty, paired_comparison, reconcile_football_data, score_matches, temporal_split,
)
from football_model.joint_stint_model import UnknownStintIdentityError


def observation(mid, day):
    return {"match_id": mid, "match_date": day}


def prediction(mid, loss):
    return {"match_id": mid, "match_date": "2016-02-02", "n_stints": 10,
            "point_process_loss": loss, "expected_home_target": 1.0, "expected_away_target": 1.5,
            "actual_home_target": 2, "actual_away_target": 1, "loss_kind": "test"}


def test_boundaries_are_calendar_dates_and_not_overlapping():
    rows = [observation("train", "2015-11-30"), observation("v1", "2015-12-01"),
            observation("v2", "2016-01-31"), observation("test", "2016-02-01")]
    a, b, c = temporal_split(list(reversed(rows)))
    assert [m["match_id"] for m in a] == ["train"]
    assert [m["match_id"] for m in b] == ["v1", "v2"]
    assert [m["match_id"] for m in c] == ["test"]
    with pytest.raises(ValueError, match="Duplicate"):
        temporal_split(rows + [rows[0]])
    with pytest.raises(ValueError, match="ordered calendar"):
        temporal_split(rows, date(2016, 2, 1), date(2015, 12, 1))


def test_pairing_excludes_extra_easy_team_games_and_preserves_targets():
    team = {"predictions": [prediction("common", 3), prediction("extra", -100)]}
    player = {"predictions": [prediction("common", 2.5)]}
    result = paired_comparison(team, player)
    assert result["team_on_common"]["mean_match_loss"] == 3
    assert result["mean_player_minus_team_loss"] == -.5
    assert result["common_match_ids"] == ["common"]
    bad = deepcopy(player)
    bad["predictions"][0]["actual_home_target"] = 3
    with pytest.raises(ValueError, match="same response"):
        paired_comparison(team, bad)


def test_penalty_uses_validation_loss_and_requires_identical_cohorts():
    records = [{"player_ridge": penalty, "comparison": {
        "common_match_ids": ["a", "b"], "player_on_common": {"mean_match_loss": loss}}}
               for penalty, loss in [(10, 3), (100, 2), (1000, 2)]]
    assert choose_penalty(records) == 1000
    records[2]["comparison"]["common_match_ids"] = ["a"]
    with pytest.raises(ValueError, match="identical"):
        choose_penalty(records)


def test_unknown_player_rejects_entire_match_and_other_errors_are_not_hidden():
    class UnknownModel:
        def predict_match(self, match):
            raise UnknownStintIdentityError("new player")
    result = score_matches(UnknownModel(), [observation("one", "2016-02-01")])
    assert result["predictions"] == []
    assert result["skipped"][0]["match_id"] == "one"
    class BrokenModel:
        def predict_match(self, match):
            raise RuntimeError("real defect")
    with pytest.raises(RuntimeError, match="real defect"):
        score_matches(BrokenModel(), [observation("one", "2016-02-01")])


def test_exact_independent_join_rejects_score_disagreement(tmp_path):
    source = tmp_path / "e0.csv"
    source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n08/08/2015,Arsenal,Leicester,2,1\n", encoding="utf-8")
    rows = [{"match_id": "sb-one", "match_date": "2015-08-08", "home_team_id": 1,
             "away_team_id": 22, "diagnostics": {"event_goals": [2, 1]}}]
    assert reconcile_football_data(rows, source)["exact_joins"] == 1
    rows[0]["diagnostics"]["event_goals"] = [1, 1]
    with pytest.raises(ValueError, match="score disagreement"):
        reconcile_football_data(rows, source)
