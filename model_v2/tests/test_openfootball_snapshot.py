from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest

from football_model.openfootball_snapshot import COMPETITIONS, run_snapshot


def fixture(day, score=None, **changes):
    return {"round": "Matchday 1", "date": day, "team1": "Home", "team2": "Away",
            "score": score, **changes}


def fake_fetcher(rows, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        if url.endswith(".json"):
            body = json.dumps({"name": "Example league", "matches": rows}).encode()
        else:
            body = b"Official upstream documentation fixture"
        return body, {"last-modified": "Fri, 23 Oct 2026 00:00:00 GMT"}
    return fetch


def test_snapshot_measures_calendar_coverage_without_historical_asof_claims(tmp_path):
    calls = []
    rows = [fixture("2026-10-20", [1, 0]), fixture("2026-10-21"),
            fixture("2026-10-22"), fixture("2026-10-24", {"ft": [2, 0]}),
            fixture("2026-10-25")]
    result, report = run_snapshot(tmp_path, calendar_date=date(2026, 10, 22),
                                  fetch=fake_fetcher(rows, calls))
    assert result["status"] == "complete"
    assert result["trained"] is False and result["predictions_generated"] is False
    assert result["season"] == "2026-27"
    assert len(calls) == 7 and len(result["competitions"]) == 5
    assert result["totals"]["past_dated_without_result_rows"] == 5
    assert result["totals"]["future_scored_anomaly_rows"] == 5
    for row in result["competitions"]:
        assert row["past_dated_without_result_rows"] == 1
        assert row["on_calendar_date_without_result_rows"] == 1
        assert row["latest_scored_date"] == "2026-10-24"
        assert row["latest_scored_date_gap_days"] == -2
        assert row["latest_scored_on_or_before_calendar_date"] == "2026-10-20"
        assert row["latest_scored_on_or_before_gap_days"] == 2
        assert row["raw_score_formats"] == {"array": 1, "null": 3, "object": 1}
        assert row["verified_utc_kickoff_rows"] == row["historical_first_availability_rows"] == 0
        assert row["odds_rows"] == row["lineup_rows"] == row["injury_rows"] == 0
    assert report.parent.name.startswith("openfootball接入_") and report.exists()
    assert report.parent.parent == tmp_path / "outputs"
    raw = Path(result["raw_directory"])
    assert raw.parent == tmp_path / "data" / "openfootball" / "2026-27"
    assert {p.name for p in raw.glob("*.json")} >= {f"{code}.json" for code in COMPETITIONS}
    assert (raw / "README.md").is_file() and (raw / "LICENSE.md").is_file()
    assert (report.parent / "覆盖报告.json").is_file()
    assert "不是真实历史 as-of" in report.read_text(encoding="utf-8")
    assert "已完成比赛" not in report.read_text(encoding="utf-8").split("| 联赛 |")[1].split("\n\n")[0]
    for source in result["sources"]:
        assert source["http_last_modified_not_match_publication_time"] == "Fri, 23 Oct 2026 00:00:00 GMT"
        assert datetime.fromisoformat(source["observed_at"]).tzinfo == timezone.utc


def test_schema_failures_and_duplicates_are_not_silently_overwritten(tmp_path):
    rows = [fixture("2026-10-20", [1, 0]), fixture("2026-10-21", [2, 0]),
            fixture("2026-10-19", "1-0"), None]
    result, _ = run_snapshot(tmp_path, calendar_date=date(2026, 10, 22), fetch=fake_fetcher(rows))
    assert result["status"] == "complete"
    for row in result["competitions"]:
        assert row["rejected_rows"] == 2 and len(row["duplicate_keys"]) == 1
        assert row["input_rows"] == 4 and row["parsed_rows"] == 2
    assert result["totals"]["rejected_rows"] == 10
    assert result["totals"]["duplicate_source_keys"] == 5


def test_fetch_failure_is_explicit_and_not_retried_or_replaced(tmp_path):
    calls = []
    normal = fake_fetcher([])
    def fetch(url):
        calls.append(url)
        if url.endswith("de.1.json"):
            raise HTTPError(url, 403, "Forbidden", {}, None)
        return normal(url)
    result, report = run_snapshot(tmp_path, fetch=fetch)
    assert result["status"] == "partial"
    assert result["totals"]["download_failure_count"] == 1
    assert len(result["competitions"]) == 4
    failed = [s for s in result["sources"] if s["status"] == "error"]
    assert len(failed) == 1 and failed[0]["http_status"] == 403
    assert sum(url.endswith("de.1.json") for url in calls) == 1
    assert report.exists()


def test_bad_json_keeps_original_and_reports_failed_inspection(tmp_path):
    normal = fake_fetcher([])
    def fetch(url):
        return (b"not valid json", {}) if url.endswith("en.1.json") else normal(url)
    result, _ = run_snapshot(tmp_path, fetch=fetch)
    assert result["status"] == "partial" and len(result["inspection_errors"]) == 1
    assert (Path(result["raw_directory"]) / "en.1.json").read_bytes() == b"not valid json"
    assert len(result["competitions"]) == 4


def test_empty_scores_do_not_fabricate_a_latest_result(tmp_path):
    result, _ = run_snapshot(tmp_path, season="2024-25", calendar_date=date(2025, 5, 1),
                             fetch=fake_fetcher([fixture("2025-04-01")]))
    assert result["source_calendar_date"] == "2025-05-01"
    for row in result["competitions"]:
        assert row["latest_scored_date"] is None
        assert row["latest_scored_date_gap_days"] is None
        assert row["past_dated_without_result_rows"] == 1


def test_repeated_run_preserves_earlier_outputs_and_does_not_touch_old_probe(tmp_path):
    old = tmp_path / "outputs" / "model-v2" / "真实赛果样本_字段检查.json"
    old.parent.mkdir(parents=True)
    old.write_text("keep existing output", encoding="utf-8")
    first, report1 = run_snapshot(tmp_path, fetch=fake_fetcher([]))
    second, report2 = run_snapshot(tmp_path, fetch=fake_fetcher([]))
    assert report1 != report2 and report1.exists() and report2.exists()
    assert first["raw_directory"] != second["raw_directory"]
    assert old.read_text(encoding="utf-8") == "keep existing output"


@pytest.mark.parametrize("season", ["../2026-27", "2026/27", "2026-28", "2026", "2026-27/../../"])
def test_invalid_season_rejected_before_fetch(tmp_path, season):
    calls = []
    with pytest.raises(ValueError, match="season"):
        run_snapshot(tmp_path, season=season, fetch=fake_fetcher([], calls))
    assert calls == []


def test_datetime_is_not_a_source_calendar_date(tmp_path):
    with pytest.raises(ValueError, match="calendar_date"):
        run_snapshot(tmp_path, calendar_date=datetime.now(timezone.utc), fetch=fake_fetcher([]))


@pytest.mark.skipif(os.name != "nt", reason="Windows G-drive requirement")
def test_c_drive_rejected_without_network_or_file_creation():
    calls = []
    with pytest.raises(ValueError, match="on G:"):
        run_snapshot(Path("C:/not-created-football-snapshot"), fetch=fake_fetcher([], calls))
    assert calls == []


def test_cli_exposes_reference_date_and_exits_on_partial(monkeypatch, tmp_path, capsys):
    from football_model import openfootball_snapshot
    captured = {}
    def fake_run(task_root, season, calendar_date):
        captured.update(task_root=task_root, season=season, calendar_date=calendar_date)
        return {"coverage_json": "coverage.json", "raw_directory": "raw", "status": "partial",
                "totals": {"rejected_rows": 0}}, Path("report.md")
    monkeypatch.setattr(openfootball_snapshot, "run_snapshot", fake_run)
    status = openfootball_snapshot.main(["--task-root", str(tmp_path), "--season", "2024-25",
                                        "--source-calendar-date", "2025-05-01"])
    assert status == 1
    assert captured == {"task_root": tmp_path, "season": "2024-25", "calendar_date": date(2025, 5, 1)}
    assert json.loads(capsys.readouterr().out)["trained"] is False
