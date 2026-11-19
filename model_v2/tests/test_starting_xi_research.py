"""Tiny synthetic pipeline checks, not empirical prediction results."""

import json
import math

import pytest

from football_model import starting_xi_research as research
from football_model.starting_xi_model import FullMatchStartingXIModel


def match(mid, day, new_player=False):
    home = [f"impect:player:{i}" for i in range(1, 12)]
    if new_player:
        home[-1] = "impect:player:99"
    return {"match_id": f"impect:match:{mid}", "match_date": day, "home_team_id": 1, "away_team_id": 2,
            "home_starters": home, "away_starters": [f"impect:player:{i}" for i in range(12, 23)],
            "full_time": [1, 0]}


def inputs(tmp_path, rows):
    matches, inspection = tmp_path / "matches.json", tmp_path / "inspection.json"
    matches.write_text(json.dumps(rows), encoding="utf-8")
    inspection.write_text(json.dumps({"expected_catalogue_matches": 306,
                                      "expected_target_matches": 163, "excluded": ["source-only"]}), encoding="utf-8")
    return matches, inspection


def test_fixture_boundary_copies_only_pre_target_fields():
    row = {**match("x", "2024-01-02"), "actual_minutes": [999], "red_cards": [1], "stints": ["future"]}
    fixture = research.fixture_only(row)
    assert set(fixture) == set(research.FIXTURE_FIELDS)
    assert "full_time" not in fixture
    fixture["home_starters"].append("impect:player:1000")
    assert len(row["home_starters"]) == 11


def test_predictor_never_receives_goal_or_realized_paths():
    class Recorder:
        def predict_match(self, row):
            assert set(row) == set(research.FIXTURE_FIELDS)
            return {"home_mean": 1.5, "away_mean": 1.0}
    scored = research.score_matches(Recorder(), [{**match("x", "2024-01-02"), "stints": ["hostile"]}])
    assert scored[0]["full_score_nll"] == pytest.approx(2.5 - math.log(1.5))
    assert sum(scored[0]["forecast"]["one_x_two"].values()) == pytest.approx(1)


def test_only_identity_exception_is_a_model_skip():
    class Broken:
        def predict_match(self, row):
            raise ValueError("numerical corruption")
    with pytest.raises(ValueError, match="corruption"):
        research.score_matches(Broken(), [match("x", "2024-01-02")])


def test_empty_common_is_unknown_not_zero():
    comparison = research.compare([], [])
    assert comparison["common_matches"] == 0
    assert comparison["mean_players_minus_team"]["full_score_nll"] is None
    assert comparison["team_on_common"]["n"] == 0
    assert comparison["team_on_common"]["full_score_nll"] is None
    assert research.coverage([])["supported_fraction"] is None


def test_pairing_uses_common_ids_not_all_team_mean():
    class Constant:
        def predict_match(self, row):
            return {"home_mean": 1.5, "away_mean": 1.0}
    rows = research.score_matches(Constant(), [match("a", "2024-01-02"), match("b", "2024-01-03")])
    other = [{**rows[1], "full_score_nll": rows[1]["full_score_nll"] - 0.2}]
    comparison = research.compare(rows, other)
    assert comparison["common_match_ids"] == ["impect:match:b"]
    assert comparison["team_on_common"]["n"] == 1
    assert comparison["mean_players_minus_team"]["full_score_nll"] == pytest.approx(-0.2)
    assert comparison["paired_full_score_delta_descriptive"]["sample_sd"] is None


def test_policy_is_fixed_full_score_not_old_path_loss():
    policy = research.fixed_policy()
    assert policy["penalties"] == {"intercept": 0, "home": 1, "team": 1, "players": 100}
    assert policy["parameter_searches"] == 0
    assert "summed" in policy["loss"]
    assert "no duration offset" in policy["target"]
    assert "not a reconstructed T-60" in policy["availability"]


def test_actual_tiny_rolling_run_saves_restores_and_updates_only_next_week(tmp_path):
    rows = [match("old", "2023-12-30"), match("new", "2024-01-02", True),
            match("sameweek", "2024-01-07", True), match("nextweek", "2024-01-08", True)]
    matches, inspection = inputs(tmp_path, rows)
    original = matches.read_bytes()
    output = research.run(tmp_path, matches, inspection)
    assert matches.read_bytes() == original
    result = json.loads((output / "首发全场前推结果.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["weeks"] == 2 and result["model_fits_completed"] == 4
    assert result["coverage"]["team"]["supported_matches"] == 3
    assert result["coverage"]["players"]["supported_matches"] == 1
    assert result["comparison"]["common_match_ids"] == ["impect:match:nextweek"]
    scores = json.loads((output / "full_match_players_predictions.json").read_text(encoding="utf-8"))
    assert [r["match_id"] for r in scores if r["status"] == "skipped"] == ["impect:match:new", "impect:match:sameweek"]
    assert all("99" in r["reason"] for r in scores if r["status"] == "skipped")
    first = FullMatchStartingXIModel.load(output / "models" / "2024-01-01_players.json")
    later = FullMatchStartingXIModel.load(output / "models" / "2024-01-08_players.json")
    assert "impect:player:99" not in first.players
    assert "impect:player:99" in later.players
    weeks = json.loads((output / "逐周拟合与比较.json").read_text(encoding="utf-8"))
    assert [r["history_matches"] for r in weeks] == [1, 3]
    assert weeks[0]["history_max_date"] < weeks[0]["week_start"]
    assert result["source_denominators"]["expected_target_matches"] == 163
    assert "source-only" in (output / "源检验说明副本.json").read_text(encoding="utf-8")


def test_all_unknown_targets_complete_without_invented_mean(tmp_path):
    matches, inspection = inputs(tmp_path, [match("old", "2023-12-30"), match("new", "2024-01-02", True)])
    output = research.run(tmp_path, matches, inspection)
    result = json.loads((output / "首发全场前推结果.json").read_text(encoding="utf-8"))
    assert result["comparison"]["common_matches"] == 0
    assert result["comparison"]["mean_players_minus_team"]["full_score_nll"] is None
    assert "损失差为未知" in (output / "首发全场前推说明.md").read_text(encoding="utf-8")


def test_failed_fit_leaves_incomplete_not_completed_result(tmp_path, monkeypatch):
    matches, inspection = inputs(tmp_path, [match("old", "2023-12-30"), match("new", "2024-01-02")])
    def broken(*args, **kwargs):
        raise RuntimeError("failed fit")
    monkeypatch.setattr(FullMatchStartingXIModel, "fit", broken)
    with pytest.raises(RuntimeError, match="failed fit"):
        research.run(tmp_path, matches, inspection)
    outputs = list((tmp_path / "outputs").iterdir())
    assert len(outputs) == 1
    assert (outputs[0] / "运行未完成说明.json").exists()
    assert not (outputs[0] / "首发全场前推结果.json").exists()
