import copy
import json
from pathlib import Path

import pytest

from football_model.prospective_pilot import main, render_report, run_pilot


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "prematch-intelligence" / "20261024_可行性样本.json"


def source_payload():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def export_input(tmp_path, payload):
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_sample_export_keeps_abstention_and_empty_metrics(tmp_path):
    result = run_pilot(tmp_path, SAMPLE)
    directory = Path(result["output_directory"])
    scores = json.loads((directory / "同场比较与覆盖.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "三轨赛前记录.json").read_text(encoding="utf-8"))
    packets = json.loads((directory / "LLM新闻输入包.json").read_text(encoding="utf-8"))
    assert result["n_records"] == 1 and result["paired_n"] == 0
    assert scores["n_settled"] == 0
    assert all(scores["metrics"][arm]["mean_log_loss"] is None for arm in scores["metrics"])
    assert all(track is None for track in records[0]["tracks"].values())
    assert len(packets[0]["evidence"]) == 6
    assert all(item["published_at"] is None for item in packets[0]["evidence"])
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "abstain" in report and "三轨完整且已结算的同场比较 0 场" in report
    assert report == render_report(source_payload(), packets, records, scores)


def test_reexport_is_not_new_llm_forecast(tmp_path):
    original = SAMPLE.read_bytes()
    first, second = run_pilot(tmp_path, SAMPLE), run_pilot(tmp_path, SAMPLE)
    assert first["output_directory"] != second["output_directory"]
    records = [json.loads((Path(result["output_directory"]) / "三轨赛前记录.json").read_text(encoding="utf-8"))
               for result in (first, second)]
    assert records[0] == records[1]
    assert SAMPLE.read_bytes() == original
    assert first["llm_api_calls_in_export"] == second["llm_api_calls_in_export"] == 0


def test_future_news_does_not_change_packet(tmp_path):
    payload = source_payload()
    baseline = run_pilot(tmp_path, export_input(tmp_path, payload))
    payload["cases"][0]["evidence"].append({"observed_at": "2026-10-26T12:00:00Z",
                                             "claim": "post-match data never supplied to LLM"})
    changed = run_pilot(tmp_path, export_input(tmp_path, payload))
    name = "LLM新闻输入包.json"
    assert (Path(baseline["output_directory"]) / name).read_bytes() == (Path(changed["output_directory"]) / name).read_bytes()


def test_abstention_cannot_be_replaced_with_forecast(tmp_path):
    payload = source_payload()
    case = payload["cases"][0]
    case["tracks"]["llm"] = {"fixture_id": case["fixture"]["match_id"],
                              "generated_at": "2026-10-23T17:05:45Z",
                              "probabilities": [.5, .3, .2],
                              "exposed_to_other_track_probabilities": False}
    del case["missing_reasons"]["llm"]
    with pytest.raises(ValueError, match="abstained"):
        run_pilot(tmp_path, export_input(tmp_path, payload))
    assert not (tmp_path / "outputs").exists()


def test_duplicate_fixture_not_counted_twice(tmp_path):
    payload = source_payload()
    payload["cases"].append(copy.deepcopy(payload["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        run_pilot(tmp_path, export_input(tmp_path, payload))


@pytest.mark.parametrize("cases", [[], None, {}])
def test_empty_cases_rejected(tmp_path, cases):
    with pytest.raises(ValueError, match="at least one"):
        run_pilot(tmp_path, export_input(tmp_path, {"pilot_id": "empty", "cases": cases}))


def test_command_line_uses_supplied_input(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "--input", str(SAMPLE)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["paired_n"] == 0 and result["trained"] is False
