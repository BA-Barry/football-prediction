from datetime import date
import copy
import json
import math

import pytest

from football_model.multi_rating import build_rating_features


def event(day, *, home="A", away="B", hg=2, ag=0, league="E0", shots=None):
    result = {"date": day, "league": league, "home_team": home, "away_team": away,
              "home_goals": hg, "away_goals": ag}
    if shots is not None:
        result.update(home_shots=shots[0], away_shots=shots[1],
                      home_shots_on_target=shots[0] // 2,
                      away_shots_on_target=shots[1] // 2)
    return result


def target(day, *, identifier="target", home="A", away="B", league="E0"):
    return {"match_id": identifier, "forecast_date": day, "league": league,
            "home_team": home, "away_team": away}


def build(obs, ts):
    return build_rating_features(obs, ts, include_adjusted_shots=False)


def test_original_single_match_rating_equations_and_strict_cutoff():
    obs = [event("2024-01-01"), event("2024-01-02", hg=0, ag=9)]
    result = build(obs, [target("2024-01-02")])
    f = result["rows"][0]["features"]
    assert f["mr_results_home"] == 10.0
    assert f["mr_results_away"] == -10.0
    assert f["mr_goals_home"] == pytest.approx(5.0 * 3.0 ** 1.25)
    change = 0.035 * 3.0 * math.log10(3.0)
    assert f["mr_pi_home_home"] == pytest.approx(change)
    assert f["mr_pi_home_away"] == pytest.approx(0.7 * change)
    assert f["mr_home_history_count"] == 1.0
    assert result["rows"][0]["history_max_date"] == "2024-01-01"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_unknown_team_is_none_not_known_zero_and_leagues_are_isolated():
    result = build([event("2024-01-01", league="D1")], [target("2024-01-03")])
    f = result["rows"][0]["features"]
    assert f["mr_results_home"] is None
    assert f["mr_pi_expected_goal_diff"] is None
    assert f["mr_home_history_count"] == 0.0
    assert f["mr_home_shots_update_count"] == 0.0


def test_future_information_cannot_modify_earlier_features_and_inputs_unchanged():
    obs = [event("2024-01-01"), event("2024-06-01", shots=(99, 1))]
    ts = [target("2024-01-03"), target("2024-07-02", identifier="later")]
    original = copy.deepcopy(obs)
    first = build(obs, ts)["rows"][0]
    changed = copy.deepcopy(obs)
    changed[1]["home_goals"] = 100
    assert first == build(changed, ts)["rows"][0]
    assert original == obs


def test_target_input_order_is_preserved_and_missing_shots_stay_unknown():
    ts = [target("2024-01-05", identifier="late"), target(date(2024, 1, 2), identifier="early")]
    result = build([event("2024-01-01")], ts)
    assert [r["match_id"] for r in result["rows"]] == ["late", "early"]
    assert result["rows"][0]["features"]["mr_shots_home"] is None
    assert result["metadata"]["new_shot_adjustment_fits"] == 0


def test_shot_adjustment_is_trained_strictly_before_prediction_season():
    pytest.importorskip("xgboost")
    obs = [event("2022-08-01", hg=0, ag=2, shots=(5, 12)),
           event("2022-09-01", hg=1, ag=1, shots=(10, 10)),
           event("2022-10-01", hg=2, ag=0, shots=(12, 5)),
           event("2023-08-01", hg=2, ag=0, shots=(15, 4)),
           event("2023-09-01", hg=0, ag=1, shots=(4, 15))]
    ts = [target("2023-08-03", identifier="early"), target("2023-10-01", identifier="late")]
    result = build_rating_features(obs, ts)
    fits = [r for r in result["metadata"]["shot_adjustment_splits"] if r["fitted"]]
    assert len(fits) == 2
    assert all(r["training_max_date"] < r["prediction_period_start"] for r in fits)
    assert result["rows"][0]["features"]["mr_home_shots_update_count"] == 1.0
    changed = copy.deepcopy(obs)
    changed[-1].update(home_goals=12, away_goals=0, home_shots=90)
    assert result["rows"][0] == build_rating_features(changed, ts)["rows"][0]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_duplicate_observations_are_not_silently_counted_twice():
    with pytest.raises(ValueError, match="Duplicate"):
        build([event("2024-01-01"), event("2024-01-01")], [target("2024-01-03")])
