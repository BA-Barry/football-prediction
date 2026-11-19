"""Tiny synthetic implementation tests; not real football training claims."""

from datetime import date
import json
from pathlib import Path

import pytest

from football_model import impect_player_research as research
from football_model.joint_stint_model import JointStintIntensityModel


def match(mid, day, home_players=None, home_goals=1):
    return {"match_id": mid, "match_date": day, "home_team_id": 1, "away_team_id": 2,
            "stints": [{"stint_id": f"{mid}:p1", "period": 1, "start_second": 0,
                        "end_second": 2800, "duration_seconds": 2800,
                        "home_players": home_players or ["impect:player:h"],
                        "away_players": ["impect:player:a"],
                        "home_score_difference": 0, "home_man_difference": 0,
                        "home_goals": home_goals, "away_goals": 0}]}


def write_inputs(tmp_path, matches, inspection=None):
    parsed, checked = tmp_path / "parsed_matches.json", tmp_path / "inspection.json"
    parsed.write_text(json.dumps(matches), encoding="utf-8")
    checked.write_text(json.dumps(inspection or {"matches": [], "excluded": ["bad-source"]}), encoding="utf-8")
    return parsed, checked


def test_fixed_policy_retains_earlier_penalty_and_goals_only():
    policy = research.fixed_policy()
    assert policy["response_kind"] == "goals"
    assert policy["player_ridge"] == 100
    assert policy["team_ridge"] == policy["context_ridge"] == 1
    assert policy["max_iterations"] == 2000
    assert "2015/16" in policy["player_penalty_provenance"]
    assert "No 2023/24" in policy["player_penalty_provenance"]
    assert "not pre-match" in policy["limits"]


def test_training_observations_measure_exposure_once_and_absent_red_card_signal():
    summary = research.training_observations([match("a", "2023-12-30"), match("b", "2023-12-31", home_goals=2)])
    assert summary["matches"] == 2
    assert summary["stints"] == 2
    assert summary["total_goals"] == 3
    assert summary["total_actual_seconds"] == 5600
    assert summary["home_man_difference_observed_values"] == [0]
    assert summary["home_man_difference_varies"] is False
    assert summary["unequal_player_count_stints"] == 0
    assert "do not claim" in summary["man_difference_limit"]


def test_training_observations_describe_actual_count_variation_not_causality():
    regular = match("a", "2023-12-30")
    dismissal = match("b", "2023-12-31")
    dismissal["stints"][0]["home_man_difference"] = -1
    summary = research.training_observations([regular, dismissal])
    assert summary["home_man_difference_varies"] is True
    assert summary["home_man_difference_observed_values"] == [-1, 0]
    assert summary["unequal_player_count_seconds"] == 2800
    assert summary["matches_with_unequal_player_count"] == 1
    assert "not itself a causal" in summary["man_difference_limit"]


def test_raw_source_and_credible_target_denominators_remain_distinct():
    source = {"expected_catalogue_matches": 306, "expected_target_matches": 163,
              "eligible_matches": 10, "eligible_target_matches": 5}
    rows = [match("old", "2023-12-31"), match("new", "2024-01-02")]
    summary = research.source_denominators(source, rows)
    assert summary["expected_catalogue_matches"] == 306
    assert summary["expected_target_matches"] == 163
    assert summary["eligible_target_matches_in_parsed_input"] == 1
    assert summary["source_ineligible_target_matches"] == 162
    assert summary["inspection_declared_eligible_target_matches"] == 5
    assert len(rows) == 2  # The reporting helper never reselects or filters.


def test_weekly_dates_strictly_precede_monday_and_fixed_target_end():
    rows = [match("old", "2023-12-31"), match("tue", "2024-01-02"),
            match("sun", "2024-01-07"), match("next", "2024-01-08"),
            match("last", "2024-05-18"), match("after", "2024-05-19")]
    windows = list(research.weekly_windows(rows))
    assert windows[0][0] == date(2024, 1, 1)
    assert [r["match_id"] for r in windows[0][1]] == ["old"]
    assert [r["match_id"] for r in windows[0][2]] == ["tue", "sun"]
    assert [r["match_id"] for r in windows[1][1]] == ["old", "tue", "sun"]
    assert windows[-1][2][0]["match_id"] == "last"
    assert all(r["match_id"] != "after" for _, history, targets in windows for r in history + targets)


@pytest.mark.parametrize("rows,error", [
    ([], "nonempty"),
    ([match("x", "2023-12-31")], "No target"),
    ([match("x", "2024-01-02")], "earlier"),
    ([match("x", "2024-01-02"), match("x", "2024-01-03")], "duplicate"),
    ([match("", "2024-01-02")], "Missing"),
    ([match("x", "2024-01-02T00:00:00Z")], "calendar"),
])
def test_bad_chronology_is_not_silently_repaired(rows, error):
    with pytest.raises(ValueError, match=error):
        list(research.weekly_windows(rows))


def test_zero_common_coverage_is_unknown_not_zero_loss():
    team = {"predictions": [{"match_id": "new"}]}
    players = {"predictions": []}
    comparison = research.supported_comparison(team, players)
    assert comparison["common_match_ids"] == []
    assert comparison["mean_player_minus_team_loss"] is None
    assert comparison["team_on_common"] is None
    assert comparison["paired_matches"] == []
    empty = research.coverage_summary({"requested_matches": 2, "predictions": [], "skipped": [{}, {}]})
    assert empty["supported_fraction"] == 0
    assert empty["summary_on_supported"] is None


def test_tiny_real_run_preserves_input_and_keeps_unknowns_until_next_week(tmp_path):
    rows = [match("old", "2023-12-30"), match("new", "2024-01-02", ["new-player"]),
            match("sameweek", "2024-01-07", ["new-player"]),
            match("nextweek", "2024-01-08", ["new-player"])]
    parsed, inspection = write_inputs(tmp_path, rows)
    original = parsed.read_bytes()
    output = research.run(tmp_path, parsed, inspection)
    result = json.loads((output / "固定前推结果.json").read_text(encoding="utf-8"))
    assert parsed.read_bytes() == original
    assert result["model_fits_completed"] == 4
    assert result["weeks"] == 2
    assert result["coverage"]["team"]["supported_matches"] == 3
    assert result["coverage"]["players"]["supported_matches"] == 1
    assert result["comparison"]["common_match_ids"] == ["nextweek"]
    assert json.loads((output / "源检验说明副本.json").read_text(encoding="utf-8"))["excluded"] == ["bad-source"]
    players = json.loads((output / "goals_weekly_players_predictions.json").read_text(encoding="utf-8"))
    assert [r["match_id"] for r in players["skipped"]] == ["new", "sameweek"]
    assert "new-player" in players["skipped"][0]["reason"]
    first = JointStintIntensityModel.load(output / "models" / "2024-01-01_goals_players.json")
    later = JointStintIntensityModel.load(output / "models" / "2024-01-08_goals_players.json")
    assert first.reference_match_ids == ["old"]
    assert "new-player" not in first.players
    assert set(later.reference_match_ids) == {"old", "new", "sameweek"}
    assert "new-player" in later.players
    assert first.training_max_date < first.before_date
    weeks = json.loads((output / "逐周拟合与比较.json").read_text(encoding="utf-8"))
    assert weeks[0]["training_observations"]["matches"] == 1
    assert weeks[1]["training_observations"]["matches"] == 3
    assert weeks[1]["training_observations"]["total_actual_seconds"] == 8400
    report = (output / "较新球员固定前推说明.md").read_text(encoding="utf-8")
    assert "不作显著性或稳定提升" in report
    assert "不能声称训练学到了红牌" in report


def test_all_unknown_target_week_can_complete_and_report_no_average(tmp_path):
    parsed, inspection = write_inputs(tmp_path, [match("old", "2023-12-31"),
                                               match("future", "2024-01-02", ["unknown"])])
    output = research.run(tmp_path, parsed, inspection)
    result = json.loads((output / "固定前推结果.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["comparison"]["mean_player_minus_team_loss"] is None
    assert result["coverage"]["players"]["summary_on_supported"] is None
    assert "损失差为未知" in (output / "较新球员固定前推说明.md").read_text(encoding="utf-8")


def test_scoring_occurs_only_after_save_and_load(tmp_path, monkeypatch):
    parsed, inspection = write_inputs(tmp_path, [match("old", "2023-12-31"), match("next", "2024-01-02")])
    saved = []
    original_save = JointStintIntensityModel.save
    original_load = JointStintIntensityModel.load
    original_score = research.score_matches

    def save(self, path):
        original_save(self, path)
        saved.append((str(path), "save"))

    def load(path):
        restored = original_load(path)
        restored.loaded_from_saved_path = Path(path)
        saved.append((str(path), "load"))
        return restored

    def score(model, targets):
        assert model.loaded_from_saved_path.is_file()
        assert saved[-1] == (str(model.loaded_from_saved_path), "load")
        saved.append((str(model.loaded_from_saved_path), "score"))
        return original_score(model, targets)

    monkeypatch.setattr(JointStintIntensityModel, "save", save)
    monkeypatch.setattr(JointStintIntensityModel, "load", staticmethod(load))
    monkeypatch.setattr(research, "score_matches", score)
    research.run(tmp_path, parsed, inspection)
    assert [event for _, event in saved] == ["save", "load", "score", "save", "load", "score"]


def test_optimizer_failure_preserves_completed_work_and_no_complete_result(tmp_path, monkeypatch):
    parsed, inspection = write_inputs(tmp_path, [match("old", "2023-12-31"), match("next", "2024-01-02")])
    original_fit = JointStintIntensityModel.fit

    def fit(self, *args):
        if self.include_players:
            raise RuntimeError("synthetic optimizer failure")
        return original_fit(self, *args)

    monkeypatch.setattr(JointStintIntensityModel, "fit", fit)
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        research.run(tmp_path, parsed, inspection)
    output = next((tmp_path / "outputs").iterdir())
    failed = json.loads((output / "运行未完成说明.json").read_text(encoding="utf-8"))
    assert failed["completed_fits"] == 1
    assert failed["completed_weeks"] == 0
    assert not (output / "固定前推结果.json").exists()
    team = json.loads((output / "goals_weekly_team_predictions.json").read_text(encoding="utf-8"))
    assert team["predictions"][0]["match_id"] == "next"


def test_inspection_is_required_and_not_implicitly_discovered(tmp_path):
    parsed, inspection = write_inputs(tmp_path, [match("old", "2023-12-31"), match("next", "2024-01-02")])
    inspection.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="inspection"):
        research.run(tmp_path, parsed, inspection)
    assert not (tmp_path / "outputs").exists()


def test_models_share_same_input_paths_and_do_not_read_xg(tmp_path):
    rows = [match("old", "2023-12-31"), match("next", "2024-01-02")]
    for row in rows:
        row["stints"][0]["home_xg"] = "invalid and deliberately irrelevant"
    parsed, inspection = write_inputs(tmp_path, rows)
    output = research.run(tmp_path, parsed, inspection)
    weeks = json.loads((output / "逐周拟合与比较.json").read_text(encoding="utf-8"))
    assert weeks[0]["comparison"]["team_on_common"]["stints"] == weeks[0]["comparison"]["player_on_common"]["stints"]
    for path in (output / "models").iterdir():
        params = json.loads(path.read_text(encoding="utf-8"))
        assert params["response_kind"] == "goals"
        assert params["reference_match_ids"] == ["old"]
