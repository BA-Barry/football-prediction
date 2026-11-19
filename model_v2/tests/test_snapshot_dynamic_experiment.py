"""Small real-fit integration checks, not another historical model search."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from football_model import snapshot_dynamic_experiment as experiment
from football_model.snapshot_dynamic import DynamicFitError
from test_snapshot_dynamic import row


def install(tmp_path, monkeypatch, late_scores=((3, 0), (1, 1), (0, 2))):
    early = [row(f"early-{i}", f"2024-08-{i+2:02d}", score=score)
             for i, score in enumerate(((3, 0), (2, 1), (1, 2)))]
    late = [row(f"late-{i}", f"2025-08-{i+2:02d}", score=score)
            for i, score in enumerate(late_scores)]
    output, events = tmp_path / "research", []

    def load(root, parent, season):
        assert (output / "实验设置.json").is_file()
        events.append(season)
        if season == "2024-25":
            assert not (output / experiment.PARAMETER_FILE).exists()
            return deepcopy(early), {"parameter_snapshots": 1}
        assert season == "2025-26"
        assert (output / experiment.PARAMETER_FILE).is_file()
        saved = json.loads((output / experiment.PARAMETER_FILE).read_text(encoding="utf-8"))
        assert saved["training_match_ids"] == [r["match_id"] for r in early]
        return deepcopy(late), {"parameter_snapshots": 1}

    monkeypatch.setattr(experiment, "load_period", load)
    return early, late, output, events


def test_one_real_fit_saved_before_later_loading_and_complete_cohort(tmp_path, monkeypatch):
    early, late, output, events = install(tmp_path, monkeypatch)
    original = deepcopy((early, late))
    result = experiment.run(tmp_path, tmp_path / "archive", output, repeats=10)
    assert events == ["2024-25", "2025-26"]
    assert (early, late) == original
    assert result["counts"]["new_shared_update_speed_fits"] == 1
    assert result["counts"]["new_team_fits"] == 0
    assert result["parameter"]["n_evaluations"] <= 14
    assert result["later"]["n"] == len(late) == 3
    assert result["earlier"]["n"] == len(early) == 3
    assert result["counts"]["new_dynamic_forecasts"] == 6
    assert result["counts"]["reused_team_snapshots"] == 2
    assert json.loads((output / experiment.RESULT_FILE).read_text(encoding="utf-8")) == result
    assert (output / experiment.REPORT_FILE).read_text(encoding="utf-8") == experiment.report(result)
    tracks = json.loads((output / "较晚动态预测.json").read_text(encoding="utf-8"))
    vectors = {arm: experiment.loss_vectors(tracks[arm]) for arm in experiment.ARMS}
    with (output / "逐场较晚动态概率.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 3
    for i, item in enumerate(saved):
        for arm in experiment.ARMS:
            for domain in experiment.DOMAINS:
                assert float(item[f"{arm}_{domain}_nll"]) == vectors[arm][domain][i]
    assert (output / "动态逐场损失分布.svg").is_file()


def test_later_results_cannot_select_eta_or_change_earlier_forecasts(tmp_path, monkeypatch):
    a = tmp_path / "a"
    _, _, output, _ = install(a, monkeypatch)
    first = experiment.run(a, a / "archive", output, repeats=10)
    first_early = json.loads((output / "较早训练预测.json").read_text(encoding="utf-8"))
    b = tmp_path / "b"
    _, _, output2, _ = install(b, monkeypatch, late_scores=((0, 4), (4, 3), (1, 0)))
    second = experiment.run(b, b / "archive", output2, repeats=10)
    assert first["parameter"] == second["parameter"]
    assert first_early == json.loads((output2 / "较早训练预测.json").read_text(encoding="utf-8"))


def test_fit_failure_keeps_trace_without_opening_later_or_fallback(tmp_path, monkeypatch):
    _, _, output, events = install(tmp_path, monkeypatch)

    def fail(rows):
        raise DynamicFitError("test failure", evaluations=[{"eta": 0., "nll": 2.}])

    monkeypatch.setattr(experiment, "fit_learning_rate", fail)
    with pytest.raises(DynamicFitError):
        experiment.run(tmp_path, tmp_path / "archive", output, repeats=10)
    failure = json.loads((output / "研究失败.json").read_text(encoding="utf-8"))
    assert events == ["2024-25"]
    assert failure["history"] == [{"eta": 0., "nll": 2.}]
    assert failure["no_expanded_search_or_fallback"] is True
    assert not (output / experiment.RESULT_FILE).exists()


def test_existing_output_not_overwritten(tmp_path, monkeypatch):
    _, _, output, events = install(tmp_path, monkeypatch)
    output.mkdir()
    with pytest.raises(FileExistsError):
        experiment.run(tmp_path, tmp_path / "archive", output, repeats=10)
    assert not events


def test_missing_or_reordered_dynamic_targets_are_not_silently_compared(monkeypatch):
    rows = [row("a", "2025-08-02"), row("b", "2025-08-03")]
    monkeypatch.setattr(experiment, "filter_rows", lambda rows, eta: rows[:-1])
    with pytest.raises(ValueError, match="every source match"):
        experiment.tracks_for(rows, .04)
