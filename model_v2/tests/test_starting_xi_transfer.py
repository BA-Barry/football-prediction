"""Ordinary small transfer-run checks, not real-match prediction evidence."""

from copy import deepcopy
from datetime import date
import json
import math

import pytest

from football_model import starting_xi_transfer as transfer
from football_model.english_tiers_experiment import METRICS
from football_model.hierarchical_xi_model import UnknownHierarchicalTeamError


def match(mid, day, changed=False):
    home = [f"statsbomb:player:{i}" for i in range(1, 12)]
    away = [f"statsbomb:player:{i}" for i in range(12, 23)]
    if changed:
        home[-1] = "statsbomb:player:23"
    roles = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
    return {"match_id": f"statsbomb:match:{mid}", "match_date": day,
            "home_team_id": 1, "away_team_id": 2,
            "home_starters": home, "away_starters": away,
            "home_roles": dict(zip(home, roles)), "away_roles": dict(zip(away, roles)), "full_time": [1, 0]}


def prepare(tmp_path):
    rows = [match(1, "2016-01-25"), match(2, "2016-01-31"),
            match(3, "2016-02-02", True), match(4, "2016-02-07", True), match(5, "2016-02-08", True)]
    source, inspection = tmp_path / "matches.json", tmp_path / "inspection.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    observations = [{"source_match_id": int(r["match_id"].split(":")[-1]),
                     "match_date": r["match_date"], "included": True} for r in rows]
    observations += [{"source_match_id": 6, "match_date": "2016-02-09", "included": False,
                      "status": "opening_position_conflict"}]
    inspection.write_text(json.dumps({"expected_catalogue_matches": 6, "included_matches": 5,
                                      "matches": observations}), encoding="utf-8")
    return source, inspection


def test_weekly_history_uses_no_same_week_or_future_result():
    rows = [match(1, "2016-01-31"), match(2, "2016-02-02"),
            match(3, "2016-02-07"), match(4, "2016-02-08")]
    windows = list(transfer.weekly_windows(rows))
    assert len(windows) == 2
    assert windows[0][0] == date(2016, 2, 1)
    assert [r["match_id"] for r in windows[0][1]] == ["statsbomb:match:1"]
    assert [r["match_id"] for r in windows[0][2]] == ["statsbomb:match:2", "statsbomb:match:3"]
    assert len(windows[1][1]) == 3


def test_target_endpoint_inclusive_postponed_fixture_kept():
    rows = [match(1, "2016-01-31"), match(2, "2016-05-17"), match(3, "2016-05-18")]
    windows = list(transfer.weekly_windows(rows))
    assert [r["match_id"] for _, _, target in windows for r in target] == ["statsbomb:match:2"]


@pytest.mark.parametrize("rows", [[], [match(1, "2015-12-01")],
    [match(1, "2016-02-02")], [match(1, "2016-01-01"), match(1, "2016-02-02")]])
def test_bad_or_empty_windows_fail_explicitly(rows):
    with pytest.raises(ValueError):
        list(transfer.weekly_windows(rows))


def test_fixed_policy_not_a_new_search_or_coefficient_transfer():
    policy = transfer.fixed_policy()
    assert policy["candidate_scales"] == [.10]
    assert policy["hyperparameter_searches"] == 0
    assert policy["quadrature"] == {"primary_order": 15, "check_order": 25}
    assert "no minute offset or time decay" in policy["loss"]
    assert "never relabel" in policy["source_identities"]
    assert "not pure independent" in policy["anchor_limit"]


def test_prediction_receives_no_completed_path_or_target():
    class Recorder:
        def predict_distribution(self, fixture):
            assert len(fixture) == 8 and "full_time" not in fixture
            assert "actual_minutes" not in fixture
            return {"log_mean": [math.log(1.5), 0.], "log_cov": [[.01, .002], [.002, .02]],
                    "unknown_player_ids": ["statsbomb:player:23"]}
    target = {**match(3, "2016-02-02", True), "actual_minutes": "must not enter"}
    original = deepcopy(target)
    rows, errors = transfer.score_targets(Recorder(), [target])
    assert target == original
    assert rows[0]["has_unknown_player"] is True
    assert errors[0]["full_score_nll"] < 1e-8


def test_unknown_team_is_explicit_missing_not_zero_or_fallback():
    class Missing:
        def predict_distribution(self, fixture):
            raise UnknownHierarchicalTeamError("unknown team")
    rows, errors = transfer.score_targets(Missing(), [match(3, "2016-02-02")])
    assert rows[0]["status"] == "missing" and "full_score_nll" not in rows[0]
    assert errors == []


def test_comparison_empty_is_unknown_not_zero():
    comparisons = transfer.compare_tracks({"team_laplace": [], "players_laplace": []})
    assert comparisons["all"]["common_matches"] == 0
    assert comparisons["all"]["mean_players_minus_team"]["full_score_nll"] is None
    assert transfer.conditional_intervals(comparisons["all"], [])["status"] == "unavailable"


def test_real_tiny_fits_same_history_and_unknowns_become_known(tmp_path):
    source, inspection = prepare(tmp_path)
    before = {p: p.read_bytes() for p in (source, inspection)}
    output = transfer.run(tmp_path, source, inspection)
    result = json.loads((output / "首发固定结构迁移结果.json").read_text(encoding="utf-8"))
    assert all(p.read_bytes() == content for p, content in before.items())
    assert result["model_fits_completed"] == 4 and result["weeks"] == 2
    assert result["catalogue_targets"] == 4 and result["included_targets"] == 3
    assert len(result["source_missing_targets"]) == 1
    assert result["comparisons"]["all"]["common_matches"] == 3
    assert result["comparisons"]["unknown"]["common_matches"] == 2
    assert result["comparisons"]["known"]["common_matches"] == 1
    for arm in transfer.ARMS:
        rows = json.loads((output / f"{arm}_predictions.json").read_text(encoding="utf-8"))
        assert len(rows) == 3
        assert result["tracks"][arm]["summary"]["n"] == 3
    players = json.loads((output / "players_laplace_predictions.json").read_text(encoding="utf-8"))
    assert players[0]["distribution"]["unknown_player_log_cov"][0][0] == pytest.approx(.005)
    assert players[-1]["distribution"]["unknown_player_ids"] == []
    assert len(list((output / "models").glob("*.npz"))) == 4
    for length in ("2", "4", "8"):
        assert result["conditional_block_intervals"][length]["metrics"]["full_score_nll"]["interval_95"] is None
    assert set(result["quadrature_max_absolute_difference_15_vs_25"]["players_laplace"]) >= set(METRICS)
    assert "不是新的盲测" in (output / "首发固定结构迁移说明.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("change", ["included_matches", "expected_catalogue_matches", "source_match_id"])
def test_source_count_or_identity_mismatch_before_fit(tmp_path, monkeypatch, change):
    source, inspection = prepare(tmp_path)
    value = json.loads(inspection.read_text(encoding="utf-8"))
    if change == "source_match_id":
        value["matches"][0]["source_match_id"] = 98765
    else:
        value[change] += 1
    inspection.write_text(json.dumps(value), encoding="utf-8")
    def nofit(*args, **kwargs):
        raise AssertionError("mismatched source should not be fitted")
    monkeypatch.setattr(transfer.HierarchicalStartingXIModel, "fit", nofit)
    with pytest.raises(ValueError):
        transfer.run(tmp_path, source, inspection)


def test_conditional_intervals_keep_paired_calendar_structure():
    differences = [{"match_id": str(i), **{m: -.01 for m in METRICS}} for i in range(12)]
    dates = [{"match_id": str(i), "match_date": date(2016, 2, 1).fromordinal(date(2016, 2, 1).toordinal()+7*i).isoformat()}
             for i in range(12)]
    result = transfer.conditional_intervals({"paired_matches": differences}, dates)
    for length in ("2", "4", "8"):
        assert result[length]["n_calendar_weeks"] == 12
        assert result[length]["metrics"]["full_score_nll"]["interval_95"] == pytest.approx([-.01, -.01])


def test_incomplete_run_retains_progress_without_auto_retry(tmp_path, monkeypatch):
    source, inspection = prepare(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate optimizer failure")
    monkeypatch.setattr(transfer.HierarchicalStartingXIModel, "fit", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        transfer.run(tmp_path, source, inspection)
    path = next((tmp_path / "outputs").glob("*/运行未完成说明.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["completed_fits"] == 0 and value["completed_weeks"] == 0
