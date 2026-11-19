"""Tiny research orchestration tests; not real-season prediction evidence."""

import copy
import json
import math

import pytest

from football_model import hierarchical_xi_research as research
from football_model import starting_xi_research as old_research
from football_model.gaussian_goal_mixture import score


def match(mid, day, changed=False):
    home = [f"impect:player:{i}" for i in range(1,12)]
    if changed:
        home[-1] = "impect:player:23"
    away = [f"impect:player:{i}" for i in range(12,23)]
    return {"match_id": f"impect:match:{mid}", "match_date": day, "home_team_id":1,"away_team_id":2,
            "home_starters":home,"away_starters":away,"full_time":[1,0],
            "home_roles": {p:"GK" if i==0 else "DEF" if i<5 else "MID" if i<9 else "FWD" for i,p in enumerate(home)},
            "away_roles": {p:"GK" if i==0 else "DEF" if i<5 else "MID" if i<9 else "FWD" for i,p in enumerate(away)}}


def test_fixture_projection_excludes_all_realized_information():
    row = {**match("x","2024-01-02"),"substitutions":[1],"actual_minutes":[999]}
    original = copy.deepcopy(row)
    fixture = research.fixture_only(row)
    assert set(fixture) == set(old_research.FIXTURE_FIELDS) | {"home_roles","away_roles"}
    fixture["home_roles"][fixture["home_starters"][0]] = "MID"
    fixture["home_starters"].append("changed")
    assert row == original


def test_policy_fixes_main_and_reports_sensitivity_without_winner_selection():
    policy = research.fixed_policy()
    assert policy["main_prior_sd"] == .10
    assert policy["sensitivity_prior_sd"] == [.05,.10,.20]
    assert policy["searches"] == 0
    assert "no winner" in policy["prior_scale_selection"]
    assert policy["quadrature"]["primary_order"] == 15
    assert policy["quadrature"]["fixed_check_order"] == 25
    assert "not learned" in policy["components"]


def test_empty_and_fixed_cohort_comparisons_do_not_fill_zero():
    assert research.paired_comparison([],[])["mean_players_minus_team"]["full_score_nll"] is None
    base = {"match_id":"x","match_date":"2024-01-02","status":"predicted",**score([0,0],[[0,0],[0,0]],[1,0])}
    changed = {**base,"full_score_nll":base["full_score_nll"]+.2}
    assert research.paired_comparison([base],[changed],set())["common_matches"] == 0
    assert research.paired_comparison([base],[changed],{"x"})["mean_players_minus_team"]["full_score_nll"] == pytest.approx(.2)


def test_scoring_only_passes_fixture_to_predictor():
    class Recorder:
        def predict_distribution(self, fixture):
            assert "full_time" not in fixture and "substitutions" not in fixture
            assert len(fixture) == 8
            return {"log_mean":[math.log(1.5),0],"log_cov":[[.01,.002],[.002,.02]],"unknown_player_ids":[]}
    rows, errors = research.score_targets(Recorder(),[match("x","2024-01-02")],{"impect:match:x"})
    assert rows[0]["old_v36_cohort"] == "known"
    assert rows[0]["status"] == "predicted"
    assert errors[0]["full_score_nll"] < 1e-8
    assert errors[0]["total_goals_probability"] < 1e-8
    assert errors[0]["away_minus_two_jincai_probability"] < 1e-8


def test_numerical_check_measures_even_unrealized_category_probabilities():
    primary = score([0.,0.], [[.01,0.],[0.,.01]], [1,0])
    check = copy.deepcopy(primary)
    check["forecast"]["total_goals"]["4+"] += .03
    check["forecast"]["away_minus_two_jincai"]["win"] += .02
    errors = research._numerical_difference(primary, check)
    assert errors["total_goals_probability"] == pytest.approx(.03)
    assert errors["away_minus_two_jincai_probability"] == pytest.approx(.02)
    assert errors["full_score_nll"] == 0.


def prepare(tmp_path):
    rows = [match("old","2023-12-30"),match("new","2024-01-02",True),match("same","2024-01-07",True),match("next","2024-01-08",True)]
    source, inspection = tmp_path / "matches.json", tmp_path / "inspection.json"
    source.write_text(json.dumps(rows),encoding="utf-8")
    inspection.write_text(json.dumps({"expected_catalogue_matches":306,"expected_target_matches":163}),encoding="utf-8")
    old = old_research.run(tmp_path,source,inspection)
    return source,inspection,old


def test_tiny_actual_run_unknowns_integrated_both_tracks_same_history(tmp_path):
    source,inspection,old = prepare(tmp_path)
    original = {path:path.read_bytes() for path in [source,inspection,*old.rglob("*.json")]}
    output = research.run(tmp_path,source,inspection,old)
    assert all(path.read_bytes()==data for path,data in original.items())
    result = json.loads((output / "分层首发前推结果.json").read_text(encoding="utf-8"))
    assert result["model_fits_completed"] == 8 and result["weeks"] == 2
    assert result["target_matches"] == 3 and result["hyperparameter_searches"] == 0
    assert result["track_summaries"]["team_laplace"]["n"] == 3
    for scale in (.05,.10,.20):
        comparison = result["comparisons"][f"s{scale:.2f}"]
        assert comparison["all"]["common_matches"] == 3
        assert comparison["old_known"]["common_matches"] == 1
        assert comparison["old_unknown"]["common_matches"] == 2
    rows = json.loads((output / "players_s0.10_predictions.json").read_text(encoding="utf-8"))
    assert rows[0]["distribution"]["unknown_player_ids"] == ["impect:player:23"]
    assert rows[-1]["distribution"]["unknown_player_ids"] == []
    assert rows[0]["distribution"]["unknown_player_log_cov"][0][0] == pytest.approx(.005)
    assert max(result["team_map_reconstruction_max_absolute_metric_difference_from_v36"].values()) < 1e-8
    assert len(list((output / "models").glob("*.npz"))) == 8
    assert "不按成绩选赢家" in (output / "分层首发前推说明.md").read_text(encoding="utf-8")


def test_mismatched_old_target_set_fails_before_fit(tmp_path,monkeypatch):
    source,inspection,old = prepare(tmp_path)
    path = old / "full_match_players_predictions.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(rows[:-1]),encoding="utf-8")
    def nofit(*args,**kwargs):
        raise AssertionError("fit should not start on mismatched experiment")
    monkeypatch.setattr(research.HierarchicalStartingXIModel,"fit",nofit)
    with pytest.raises(ValueError,match="all original"):
        research.run(tmp_path,source,inspection,old)
