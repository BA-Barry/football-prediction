"""Batch research adapter tests use synthetic records and a single mocked fit."""

from dataclasses import replace
from datetime import date, datetime, timezone
import json

import numpy as np
import pytest

import football_model.batch_projection as batch
from football_model.current_projection import convert_current


NOW = datetime(2026, 10, 23, 18, tzinfo=timezone.utc)


def row(**changes):
    return {"match_id": "future:a:b", "home": "team:a", "away": "team:b",
            "competition": "Premier League", "kickoff": "2026-10-25T14:00:00Z",
            "home_source_label": "Manchester United FC", "away_source_label": "AFC Bournemouth",
            "source_url": "https://example.test/official/fixtures",
            "observed_at": "2026-10-23T17:00:00Z", "source_note": "Official UTC kickoff"} | changes


def fixture_file(path, rows=None, **extra):
    path.write_text(json.dumps({"fixtures": [row()] if rows is None else rows} | extra), encoding="utf-8")
    return path


def source_row(**changes):
    return {"observed_at": "2026-10-23T17:00:00Z", "match_date": "2026-10-18",
            "home_label": "Manchester United FC", "away_label": "AFC Bournemouth",
            "full_time": [2, 1], "source_url": "https://example.test/official/result"} | changes


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    clock = [NOW]

    class Clock:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return clock[0]

    monkeypatch.setattr(batch, "datetime", Clock)
    context = tmp_path / "context"
    context.mkdir()
    first = source_row()
    supplement = source_row(home_label="Chelsea FC", away_label="Arsenal FC", full_time=[1, 0])
    inspection = {"rejected_rows": [], "duplicate_keys": [],
                  "records": [first, supplement | {"full_time": None},
                              source_row(home_label="Coventry City FC", away_label="Hull City AFC", full_time=None)]}
    (context / "inspection.json").write_text(json.dumps(inspection), encoding="utf-8")
    (context / "supplements.json").write_text(json.dumps({"records": [supplement]}), encoding="utf-8")
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    manifest = {"files": [{"path": str(history_dir / "E0.csv"), "downloaded_at": "2026-10-23T16:00:00Z"}]}
    (history_dir / "下载来源.json").write_text(json.dumps(manifest), encoding="utf-8")
    history = replace(convert_current(first, "synthetic.json", NOW),
                      match_id="historical:2025:a:b", season="2025/26", match_date=date(2026, 5, 1))
    future_history = replace(history, match_id="future-result", match_date=date(2026, 10, 24))
    same_day = replace(history, match_id="same-day-result", match_date=NOW.date())
    other_competition = replace(history, match_id="other-league", competition="D1")
    monkeypatch.setattr(batch, "season_directories", lambda root: {"2025/26": history_dir})
    monkeypatch.setattr(batch, "load_directory", lambda directory, season: (
        [history, future_history, same_day, other_competition], {"rejected_rows": [], "duplicate_keys": []}))
    fits, predicts = [], []

    class Model:
        def __init__(self, regularization, half_life):
            assert regularization == .01 and half_life == 180

        def fit(self, records, before_date):
            fits.append((list(records), before_date))
            self.n_matches = len(records)
            self.teams = {team for record in records for team in (record.home, record.away)}
            return self

        def predict(self, home, away, competition, match_date):
            predicts.append((home, away, competition, match_date))
            return 1.8, 1.1

        def parameters(self):
            return {"regularization": .01, "half_life_days": 180, "n_matches": self.n_matches,
                    "not_for": "Historical estimator warning remains intact"}

    monkeypatch.setattr(batch, "DateTeamGoalModel", Model)
    return {"root": tmp_path, "context": context, "fixtures": fixture_file(tmp_path / "fixtures.json"),
            "history_dir": history_dir, "fits": fits, "predicts": predicts, "clock": clock, "history": history,
            "model_class": Model}


def run(sandbox):
    return batch.run_batch_projection(sandbox["root"], sandbox["context"], sandbox["fixtures"])


def test_shared_one_fit_supplies_every_fixture_and_preserves_policy(sandbox):
    fixtures = [row(), row(match_id="future:c:d", home="team:c", away="team:d",
                           home_source_label="Chelsea FC", away_source_label="Arsenal FC",
                           kickoff="2026-10-26T20:00:00+01:00")]
    fixture_file(sandbox["fixtures"], fixtures, policy={"parameter_changes": "trend-based only"})
    result, output = run(sandbox)
    assert len(sandbox["fits"]) == result["fit_count"] == 1
    assert len(sandbox["predicts"]) == result["fixture_count"] == result["predicted_fixtures"] == 2
    assert result["missing_fixtures"] == result["parameter_searches"] == 0
    assert result["training_matches"] == 3 and result["current_season_matches"] == 2
    assert result["policy"] == {"parameter_changes": "trend-based only"}
    assert sandbox["predicts"][1] == ("football-data:E0:Chelsea", "football-data:E0:Arsenal", "E0", date(2026, 10, 26))
    assert result["model_parameters"]["not_for"] == "Historical estimator warning remains intact"
    assert output.parent == sandbox["root"] / "outputs"
    assert output.name.startswith("批量球队投影_")


def test_numerical_full_score_distribution_agrees_with_summary_and_track(sandbox):
    result, _ = run(sandbox)
    case = result["fixtures"][0]
    distribution = case["score_distribution"]
    matrix = np.array(distribution["probabilities"])
    assert distribution["home_goal_axis"] == list(range(matrix.shape[0]))
    assert distribution["away_goal_axis"] == list(range(matrix.shape[1]))
    assert matrix.sum() == pytest.approx(1)
    assert case["rates"] == {"home": 1.8, "away": 1.1}
    h, a = np.indices(matrix.shape)
    p = [matrix[h > a].sum(), matrix[h == a].sum(), matrix[h < a].sum()]
    assert case["track"]["probabilities"] == pytest.approx(p)
    assert case["forecast"]["one_x_two"] == dict(zip(("home", "draw", "away"), case["track"]["probabilities"]))
    assert case["track"]["fixture_id"] == row()["match_id"]
    assert datetime.fromisoformat(case["track"]["generated_at"]) < datetime.fromisoformat(case["fixture"]["kickoff"])
    assert case["missing_reasons"] == {} and case["status"] == "predicted"


def test_output_identity_source_descriptions_and_missing_coverage_are_retained(sandbox):
    result, output = run(sandbox)
    case = result["fixtures"][0]
    assert case["fixture"]["home"] == "team:a" and case["fixture"]["away"] == "team:b"
    assert case["fixture_source"]["source_note"] == "Official UTC kickoff"
    assert case["fixture_source"]["source_url"] == row()["source_url"]
    assert case["fixture_source"]["observed_at"] == "2026-10-23T17:00:00+00:00"
    assert case["resolved_team_labels"] == {"home": "Man United", "away": "Bournemouth"}
    assert result["current_missing_reported_results"] == 1
    assert len(case["descriptive_team_results"]) == 4
    assert case["descriptive_team_results"][1]["played"] == 1
    assert case["descriptive_team_results"][1]["goals_for"] == 2
    assert case["descriptive_team_results"][3]["goals_against"] == 2
    saved = json.loads((output / "批量球队投影与来源.json").read_text(encoding="utf-8"))
    assert saved == result
    assert json.loads((output / "逐场统计线路.json").read_text(encoding="utf-8"))[0]["track"] == case["track"]
    desc = json.loads((output / "LLM描述性球队资料.json").read_text(encoding="utf-8"))
    assert desc[0]["descriptive_team_results"] == case["descriptive_team_results"]


def test_only_earlier_dates_and_e0_enter_fit(sandbox):
    run(sandbox)
    records, before = sandbox["fits"][0]
    assert before == NOW.date()
    assert all(record.match_date < before and record.competition == "E0" for record in records)
    assert {record.match_id for record in records} == {
        "historical:2025:a:b", "football-data:E0:2026/27:Man United|Bournemouth",
        "football-data:E0:2026/27:Chelsea|Arsenal"}


@pytest.mark.parametrize("competition", ["E0", "premier-league", "Premier League"])
def test_explicit_competition_aliases_are_supported(sandbox, competition):
    fixture_file(sandbox["fixtures"], [row(competition=competition)])
    result, _ = run(sandbox)
    assert result["fixtures"][0]["status"] == "predicted"
    assert sandbox["predicts"][0][2] == "E0"


@pytest.mark.parametrize("changes,reason", [
    ({"home_source_label": "Unknown FC"}, "Unresolved explicit team aliases"),
    ({"home_source_label": "Coventry City FC"}, "Unseen team requires an explicitly estimated prior"),
    ({"competition": "Champions League"}, "outside this E0-only"),
    ({"neutral": True}, "Neutral venue is unsupported"),
])
def test_unsupported_cases_remain_missing_without_zero_substitution(sandbox, changes, reason):
    fixture_file(sandbox["fixtures"], [row(**changes)])
    result, _ = run(sandbox)
    assert len(sandbox["fits"]) == 1 and sandbox["predicts"] == []
    assert result["fixture_count"] == result["missing_fixtures"] == 1
    assert result["predicted_fixtures"] == 0
    case = result["fixtures"][0]
    assert case["status"] == "missing" and reason in case["missing_reasons"]["statistical"]
    assert all(case[key] is None for key in ("track", "forecast", "rates", "score_distribution"))
    if changes.get("home_source_label") == "Unknown FC":
        assert case["resolved_team_labels"]["home"] is None
        assert len(case["descriptive_team_results"]) == 2


@pytest.mark.parametrize("changes,message", [
    ({"kickoff": "2026-10-25T14:00:00"}, "timezone-aware"),
    ({"kickoff": "2026-10-23T18:00:00Z"}, "not future"),
    ({"kickoff": "2026-10-23T17:59:59Z"}, "not future"),
    ({"observed_at": "2026-10-23T18:00:01Z"}, "not observed"),
    ({"observed_at": "2026-10-23T17:00:00"}, "timezone-aware"),
    ({"published_at": "2026-10-24T12:00:00Z"}, "publication"),
    ({"source_url": "file:///fixture.json"}, "HTTP"),
    ({"home_source_label": ""}, "nonempty"),
    ({"home": ""}, "nonempty"),
    ({"neutral": "false"}, "boolean"),
    ({"away": "team:a"}, "Teams must differ"),
    ({"away_source_label": "Manchester United FC"}, "source teams must differ"),
])
def test_invalid_or_unavailable_fixture_data_are_rejected_before_fit(sandbox, changes, message):
    fixture_file(sandbox["fixtures"], [row(**changes)])
    with pytest.raises(ValueError, match=message):
        run(sandbox)
    assert sandbox["fits"] == []


@pytest.mark.parametrize("rows", [[], [row(), row()], [row(), row(match_id="different-id")]])
def test_empty_and_duplicate_batches_are_not_fitted(sandbox, rows):
    fixture_file(sandbox["fixtures"], rows)
    with pytest.raises(ValueError, match="nonempty|Duplicate"):
        run(sandbox)
    assert sandbox["fits"] == []


def test_historical_manifest_must_be_available(sandbox):
    manifest_path = sandbox["history_dir"] / "下载来源.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["downloaded_at"] = "2026-10-23T18:00:01Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="downloaded after"):
        run(sandbox)
    assert sandbox["fits"] == []


@pytest.mark.parametrize("error_field", ["rejected_rows", "duplicate_keys"])
def test_historical_unresolved_rows_are_not_silently_fitted(sandbox, monkeypatch, error_field):
    monkeypatch.setattr(batch, "load_directory", lambda directory, season: (
        [sandbox["history"]], {"rejected_rows": [], "duplicate_keys": [], error_field: ["unresolved"]}))
    with pytest.raises(ValueError, match="Unresolved historical"):
        run(sandbox)
    assert sandbox["fits"] == []


def test_current_source_future_observation_is_not_fitted(sandbox):
    path = sandbox["context"] / "supplements.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["records"][0]["observed_at"] = "2026-10-23T18:00:01Z"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="not observed"):
        run(sandbox)
    assert sandbox["fits"] == []


def test_fixture_that_starts_during_fit_is_missing_not_live_prediction(sandbox, monkeypatch):
    original_fit = sandbox["model_class"].fit

    def slow_fit(self, records, before_date):
        result = original_fit(self, records, before_date)
        sandbox["clock"][0] = datetime(2026, 10, 25, 14, tzinfo=timezone.utc)
        return result

    monkeypatch.setattr(sandbox["model_class"], "fit", slow_fit)
    result, _ = run(sandbox)
    assert result["fixtures"][0]["track"] is None and sandbox["predicts"] == []
    assert "elapsed during batch fit" in result["fixtures"][0]["missing_reasons"]["statistical"]


def test_fixture_that_starts_during_grid_calculation_is_not_retained(sandbox, monkeypatch):
    original_score = batch.score_grid

    def slow_grid(*args, **kwargs):
        result = original_score(*args, **kwargs)
        sandbox["clock"][0] = datetime(2026, 10, 25, 14, tzinfo=timezone.utc)
        return result

    monkeypatch.setattr(batch, "score_grid", slow_grid)
    result, _ = run(sandbox)
    case = result["fixtures"][0]
    assert case["track"] is case["forecast"] is case["rates"] is None
    assert "elapsed during projection" in case["missing_reasons"]["statistical"]


def test_policy_and_fixture_list_are_ordinary_explicit_objects(sandbox):
    fixture_file(sandbox["fixtures"], policy="frozen")
    with pytest.raises(ValueError, match="policy must be an object"):
        run(sandbox)
    fixture_file(sandbox["fixtures"], [None])
    with pytest.raises(ValueError, match="fixture must be an object"):
        run(sandbox)


def test_main_prints_compact_summary_without_another_fit(sandbox, capsys):
    assert batch.main(["--root", str(sandbox["root"]), "--context", str(sandbox["context"]),
                       "--fixtures", str(sandbox["fixtures"])]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["fit_count"] == 1 and summary["fixture_count"] == summary["predicted_fixtures"] == 1
    assert summary["missing_fixtures"] == 0 and len(sandbox["fits"]) == 1
