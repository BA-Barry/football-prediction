"""Replay/export checks, not predictive-performance observations.

The real example is never edited or settled here. Any fabricated result is
confined to a temporary copy and labelled as a synthetic scoring exercise.
"""

from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import socket
import urllib.request

import pytest

from football_model import current_projection
from football_model.multiline import PAIRS, TRACKS
from football_model.multiline_pilot import main, render_report, run_multiline


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/prematch-intelligence/20261024_multiline/multiline_case.json"


def source_payload():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def export_input(tmp_path, payload):
    path = tmp_path / "reviewed-multiline.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def read_export(result, name):
    return json.loads((Path(result["output_directory"]) / name).read_text(encoding="utf-8"))


def test_real_sample_exports_five_available_lines_and_no_settled_observations(tmp_path):
    payload = source_payload()
    result = run_multiline(tmp_path, SAMPLE)
    scores = read_export(result, "同场评分与覆盖.json")
    assert result["records"] == 1 and result["settled"] == 0
    assert result["all_five_common_scored"] == 0
    assert result["available"] == dict.fromkeys(TRACKS, 1)
    assert scores["n_records"] == 1 and scores["n_settled"] == 0
    assert scores["pending_fixture_ids"] == [payload["records"][0]["fixture"]["match_id"]]
    assert len(scores["comparisons"]) == len(PAIRS) == 10
    for subset in [scores["all_five_common"], *scores["comparisons"].values()]:
        assert subset["n"] == 0 and subset["fixture_ids"] == []
        for metrics in subset["metrics"].values():
            assert metrics["n"] == 0
            assert metrics["mean_log_loss"] is None
            assert metrics["mean_brier"] is None
            assert metrics["top1_accuracy"] is None
    assert all(c["available_with_result"] == 0 for c in scores["coverage"].values())
    assert read_export(result, "五线赛前记录.json") == payload["records"]


def test_real_sample_keeps_original_llm_responses_market_source_and_provenance(tmp_path):
    payload = source_payload()
    result = run_multiline(tmp_path, SAMPLE)
    saved = read_export(result, "输入与原始LLM回答.json")
    assert saved == payload
    for key in ("llm_responses", "market_observation", "market_history", "packet_paths", "policy"):
        assert saved[key] == payload[key]
    assert saved["market_observation"]["source_url"].startswith("https://skybet.com/")
    assert saved["market_observation"]["opening_status"] == "unknown"
    assert saved["market_observation"]["quote_history_points"] == 1
    assert saved["policy"]["old_abstention_preserved"] is True
    assert all(response["model_identifier"] == "unknown" for response in saved["llm_responses"].values())
    for name, response in saved["llm_responses"].items():
        track = saved["records"][0]["tracks"][name]
        assert track["probabilities"] == response["probabilities"]
        assert datetime.fromisoformat(track["generated_at"]) == datetime.fromisoformat(response["generated_at_utc"])
        assert track["exposed_to_statistical"] is response["exposed_to_statistical"]
        assert track["exposed_to_market"] is response["exposed_to_market"] is False


def test_real_sample_reports_actual_untrained_pool_and_no_change_informed_line(tmp_path):
    result = run_multiline(tmp_path, SAMPLE)
    records = read_export(result, "五线赛前记录.json")
    tracks = records[0]["tracks"]
    assert tracks["llm_independent"]["probabilities"] == [.44, .31, .25]
    assert tracks["llm_model_informed"]["probabilities"] == tracks["statistical"]["probabilities"]
    expected = [(a + b) / 2 for a, b in zip(tracks["statistical"]["probabilities"],
                                          tracks["llm_independent"]["probabilities"])]
    assert tracks["joint_pool"]["probabilities"] == pytest.approx(expected)
    assert tracks["joint_pool"]["weight_source"] == "explicit_untrained_research"
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "保留原模型数字" in report
    assert "不是额外的独立预测证据" in report
    assert "50:50只是未训练的算术候选" in report
    assert "1/p" in report and "不是推荐买入价" in report
    assert "相邻2 UP" in report and "不是不同厂商模型的比较" in report


def test_report_is_rendered_from_saved_payload_and_scores(tmp_path):
    result = run_multiline(tmp_path, SAMPLE)
    scores = read_export(result, "同场评分与覆盖.json")
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert report == render_report(source_payload(), scores)
    assert "已结算 0 场" in report
    assert "五线共同且已结算 0 场" in report
    assert "主目标胜平负Log loss" in report
    assert "不能当作五份独立证据" in report


def test_reexport_preserves_cutoff_and_all_forecast_times_and_never_edits_source(tmp_path):
    original_bytes = SAMPLE.read_bytes()
    first = run_multiline(tmp_path, SAMPLE)
    second = run_multiline(tmp_path, SAMPLE)
    assert first["output_directory"] != second["output_directory"]
    a = read_export(first, "五线赛前记录.json")
    b = read_export(second, "五线赛前记录.json")
    assert a == b == source_payload()["records"]
    assert SAMPLE.read_bytes() == original_bytes
    for result in (first, second):
        metadata = read_export(result, "导出信息.json")
        assert metadata["export_time_is_not_prediction_time"] is True
        assert metadata["training_runs"] == metadata["llm_calls"] == 0
        assert metadata["input_path"] == str(SAMPLE.resolve())
        assert result["training_runs_in_export"] == result["llm_calls_in_export"] == 0
        assert datetime.fromisoformat(metadata["exported_at"]) > datetime.fromisoformat(a[0]["as_of"])


def test_real_replay_needs_no_projection_files_no_refit_and_no_network(tmp_path, monkeypatch):
    payload = source_payload()
    payload["statistical_projection_path"] = str(tmp_path / "not-present-statistical-model.json")
    payload["packet_paths"] = {"independent": str(tmp_path / "not-present-packet.json")}

    def prohibited(*args, **kwargs):
        pytest.fail("Export must not fit a model or call network/LLM services")

    monkeypatch.setattr(current_projection.DateTeamGoalModel, "fit", prohibited)
    monkeypatch.setattr(socket, "create_connection", prohibited)
    monkeypatch.setattr(urllib.request, "urlopen", prohibited)
    result = run_multiline(tmp_path, export_input(tmp_path, payload))
    assert result["available"] == dict.fromkeys(TRACKS, 1)
    assert result["training_runs_in_export"] == result["llm_calls_in_export"] == 0
    assert read_export(result, "五线赛前记录.json") == payload["records"]


def test_explicit_missing_informed_line_is_retained_without_filling_or_dropping_game(tmp_path):
    payload = source_payload()
    record = payload["records"][0]
    record["tracks"]["llm_model_informed"] = None
    record["missing_reasons"]["llm_model_informed"] = "abstained: synthetic export test, insufficient evidence"
    del payload["llm_responses"]["llm_model_informed"]
    result = run_multiline(tmp_path, export_input(tmp_path, payload))
    scores = read_export(result, "同场评分与覆盖.json")
    saved = read_export(result, "五线赛前记录.json")[0]
    assert result["records"] == 1 and result["available"]["llm_model_informed"] == 0
    assert saved["tracks"]["llm_model_informed"] is None
    assert saved["missing_reasons"] == record["missing_reasons"]
    assert scores["coverage"]["llm_model_informed"]["abstained"] == 1
    assert "缺失／弃权：abstained:" in Path(result["report"]).read_text(encoding="utf-8")


def test_synthetic_settlement_copy_scores_each_line_once_and_report_is_consistent(tmp_path):
    payload = source_payload()
    payload["pilot_id"] = "synthetic-settlement-export-test-not-an-observed-result"
    fixture = payload["records"][0]["fixture"]
    fixture_id = fixture["match_id"]
    payload["results"][fixture_id] = {
        "fixture_id": fixture_id, "home_goals": 2, "away_goals": 1,
        "finished_at": "2026-10-25T16:00:00Z", "available_at": "2026-10-25T16:01:00Z",
        "settlement": "regulation_including_stoppage",
    }
    result = run_multiline(tmp_path, export_input(tmp_path, payload))
    scores = read_export(result, "同场评分与覆盖.json")
    assert result["records"] == result["settled"] == result["all_five_common_scored"] == 1
    assert len(scores["per_fixture"]) == 1
    for comparison in scores["comparisons"].values():
        assert comparison["n"] == 1 and comparison["fixture_ids"] == [fixture_id]
    for name, track in payload["records"][0]["tracks"].items():
        metrics = scores["all_five_common"]["metrics"][name]
        assert metrics["n"] == 1
        assert metrics["mean_log_loss"] == pytest.approx(-math.log(track["probabilities"][0]))
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "已结算 1 场" in report
    assert "五线共同且已结算 1 场" in report
    assert "所有比较n=0" not in report
    assert "当前没有已结束的真实目标比赛" not in report
    assert source_payload()["results"] == {}


@pytest.mark.parametrize("records", [[], None, {}, "not-a-list"])
def test_empty_or_malformed_records_rejected_before_output(tmp_path, records):
    with pytest.raises(ValueError, match="nonempty records"):
        run_multiline(tmp_path, export_input(tmp_path, {"pilot_id": "synthetic-invalid", "records": records}))
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("pilot_id", [None, False, 4])
def test_explicit_string_pilot_id_is_required(tmp_path, pilot_id):
    payload = source_payload()
    payload["pilot_id"] = pilot_id
    with pytest.raises(ValueError, match="pilot_id"):
        run_multiline(tmp_path, export_input(tmp_path, payload))
    assert not (tmp_path / "outputs").exists()


def test_duplicate_fixture_not_counted_as_second_game(tmp_path):
    payload = source_payload()
    payload["records"].append(deepcopy(payload["records"][0]))
    with pytest.raises(ValueError, match="duplicate fixture_id"):
        run_multiline(tmp_path, export_input(tmp_path, payload))
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("arm", TRACKS)
def test_forecast_or_quote_after_record_cutoff_is_rejected_before_output(tmp_path, arm):
    payload = source_payload()
    record = payload["records"][0]
    key = "observed_at" if arm == "market" else "generated_at"
    record["tracks"][arm][key] = (datetime.fromisoformat(record["as_of"]) + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        run_multiline(tmp_path, export_input(tmp_path, payload))
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("change", ["probability", "exposure", "pool_provenance", "settlement", "kickoff"])
def test_invalid_prediction_semantics_rejected_without_export(tmp_path, change):
    payload = source_payload()
    record = payload["records"][0]
    if change == "probability":
        record["tracks"]["market"]["probabilities"] = [.4, .3, .4]
    elif change == "exposure":
        record["tracks"]["llm_independent"]["exposed_to_statistical"] = True
    elif change == "pool_provenance":
        record["tracks"]["joint_pool"]["components"]["llm"]["id"] = "not-the-actual-input-line"
    elif change == "settlement":
        record["settlement"] = "including_extra_time"
    else:
        record["as_of"] = record["fixture"]["kickoff"]
    with pytest.raises(ValueError):
        run_multiline(tmp_path, export_input(tmp_path, payload))
    assert not (tmp_path / "outputs").exists()


def test_command_line_supplied_input_exports_saved_predictions(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "--input", str(SAMPLE)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["records"] == 1 and result["settled"] == 0
    assert result["available"] == dict.fromkeys(TRACKS, 1)
    assert result["training_runs_in_export"] == result["llm_calls_in_export"] == 0

