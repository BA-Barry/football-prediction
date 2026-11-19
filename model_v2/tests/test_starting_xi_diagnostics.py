"""Diagnostic arithmetic and tiny archived synthetic pipeline tests."""

import copy
import json
import math

import pytest

from football_model import starting_xi_diagnostics as diagnostics
from football_model import starting_xi_research as research


def match(mid, day, changed=False):
    home = [f"impect:player:{i}" for i in range(1, 12)]
    if changed:
        home[-1] = "impect:player:23"
    return {"match_id": f"impect:match:{mid}", "match_date": day, "home_team_id": 1, "away_team_id": 2,
            "home_starters": home, "away_starters": [f"impect:player:{i}" for i in range(12, 23)],
            "full_time": [1, 0]}


def payload(include_players=True):
    names = ["intercept", "home_advantage", "team_attack:1", "team_attack:2", "team_defence:1", "team_defence:2"]
    names += [f"player_{kind}:impect:player:{i}" for kind in ("attack", "defence") for i in range(1, 24)]
    coefficients = {name: 0.0 for name in names}
    coefficients.update({"intercept": math.log(1.2), "home_advantage": .1,
                         "team_attack:1": .2, "team_attack:2": .3, "team_defence:1": .4, "team_defence:2": .5,
                         "player_attack:impect:player:11": .1, "player_attack:impect:player:23": .3,
                         "player_defence:impect:player:11": .05, "player_defence:impect:player:23": .15})
    return {"feature_names": names, "coefficients": [coefficients[name] for name in names],
            "include_players": include_players, "players": [f"impect:player:{i}" for i in range(1, 24)],
            "reference_frequencies": {"1": {f"impect:player:{i}": 1 for i in range(1,12)},
                                      "2": {f"impect:player:{i}": 1 for i in range(12,23)}}}


def test_reference_xi_has_zero_deviation_and_correct_defensive_sign():
    parts = diagnostics.rate_parts(payload(), match("x", "2024-01-02"))
    assert parts["xi_log"] == [0, 0]
    assert parts["means"] == pytest.approx([1.2*math.exp(-.2), 1.2*math.exp(-.1)])


def test_starter_replacement_attack_and_defence_separated():
    parts = diagnostics.rate_parts(payload(), match("x", "2024-01-02", True))
    assert parts["xi_log"] == pytest.approx([.2, -.1])
    assert parts["means"] == pytest.approx([1.2, 1.2*math.exp(-.2)])
    team = diagnostics.rate_parts(payload(False), match("x", "2024-01-02", True))
    assert team["xi_log"] == [0,0]


def test_unknown_not_zero_imputed_and_corrupt_feature_lengths_fail():
    p = payload()
    p["players"].remove("impect:player:23")
    with pytest.raises(ValueError, match="Unknown"):
        diagnostics.rate_parts(p, match("x", "2024-01-02", True))
    p = payload()
    p["coefficients"].pop()
    with pytest.raises(ValueError, match="lengths"):
        diagnostics.rate_parts(p, match("x", "2024-01-02"))


def test_nll_includes_factorials_and_empty_descriptives_are_unknown():
    assert diagnostics.full_nll([2,1], [3,0]) == pytest.approx(3 - 3*math.log(2) + math.lgamma(4))
    assert diagnostics.describe([]) == {"n":0,"mean":None,"median":None,"sample_sd":None,"min":None,"max":None}
    assert diagnostics.describe([1])["sample_sd"] is None


def test_historical_context_never_reads_target_goal_or_paths():
    history = [match("a", "2023-12-30"), match("b", "2023-12-31")]
    context = diagnostics.history_context(history)
    future = match("c", "2024-01-02", True)
    original = copy.deepcopy(future)
    features = diagnostics.fixture_context(future, context)
    assert features["unknown_starting_ids"] == ["impect:player:23"]
    assert features["minimum_earlier_global_starts"] == 0
    assert features["sum_changes_from_latest_training_xi"] == 1
    assert features["sum_expected_replacements_from_training_reference"] == 1
    future.update(full_time=[99,99], actual_minutes=[9000], substitutions=["future"], red_cards=[1])
    assert diagnostics.fixture_context(future, context) == features
    assert original["full_time"] == [1,0]


def test_fixed_groups_keep_unknown_denominator_and_empty_cells():
    rows = [{"status":"skipped", "starts":0}, {"status":"predicted", "starts":2,
             "full_score_delta": .2, "team_full_nll":3, "players_full_nll":3.2}]
    groups = diagnostics.group_rows(rows, "starts", [("0",0,1),("1-3",1,4),("4+",4,None)])
    assert [g["requested_matches"] for g in groups] == [1,1,0]
    assert [g["common_matches"] for g in groups] == [0,1,0]
    assert groups[0]["primary_delta"]["mean"] is None
    assert groups[1]["primary_delta"]["mean"] == .2


def test_moments_do_not_compare_different_cohorts():
    rows = [{"status":"predicted", "team_means":[2,1], "actual":[3,0]}, {"status":"skipped"}]
    moments = diagnostics.calibration_moments(rows, "team")
    assert moments["matches"] == 1 and moments["team_sides"] == 2
    assert moments["predicted_goals"] == moments["observed_goals"] == 3
    assert moments["mean_pearson_square"] == .75


def test_actual_saved_pipeline_no_refit_or_mutation_during_diagnostics(tmp_path, monkeypatch):
    rows = [match("old", "2023-12-30"), match("new", "2024-01-02", True),
            match("sameweek", "2024-01-07", True), match("nextweek", "2024-01-08", True)]
    source, inspection = tmp_path / "matches.json", tmp_path / "inspection.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    inspection.write_text(json.dumps({"expected_catalogue_matches":306,"expected_target_matches":163}),encoding="utf-8")
    archived = research.run(tmp_path, source, inspection)
    original = {path: path.read_bytes() for path in [source, inspection, *archived.rglob("*.json")]}
    def no_fit(*args, **kwargs):
        raise AssertionError("diagnostics must not fit")
    monkeypatch.setattr(research.FullMatchStartingXIModel, "fit", no_fit)
    output = diagnostics.run(tmp_path, source, archived)
    assert all(path.read_bytes() == value for path,value in original.items())
    result = json.loads((output / "机制诊断结果.json").read_text(encoding="utf-8"))
    details = json.loads((output / "逐场历史信息与误差.json").read_text(encoding="utf-8"))
    assert result["target_matches"] == 3 and result["common_matches"] == 1 and result["unknown_skips"] == 2
    assert result["fits"] == result["parameter_searches"] == result["new_forecasts"] == 0
    assert result["max_reconstructed_saved_mean_error"] < 1e-12
    assert [w["history_matches"] for w in result["training"]] == [1,3]
    assert all(row["minimum_earlier_global_starts"] == 0 for row in details[:2])
    assert details[-1]["minimum_earlier_global_starts"] == 2
    assert sum(g["requested_matches"] for g in result["groups_by_minimum_earlier_starts"]) == 3
    assert sum(g["common_matches"] for g in result["groups_by_xi_changes"]) == 1
    report = (output / "首发全场机制诊断说明.md").read_text(encoding="utf-8")
    assert "不能当预测成绩" in report
    assert "不根据分组挑参数" in report
