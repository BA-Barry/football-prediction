"""Synthetic saved-source integration checks, never real forecast performance."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import socket
import urllib.request

import pytest

import football_model.batch_scoring as scoring
from football_model.date_model import DateTeamGoalModel
from football_model.domain import Fixture
from football_model.multiline import LLM_TRACKS, TRACKS, arithmetic_pool, multi_record
from football_model.prospective import SETTLEMENT


NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
OBSERVED = NOW - timedelta(minutes=1)
FIXTURES = [Fixture(f"synthetic:score:{i}", f"Synthetic home {i}", f"Synthetic away {i}",
                    "synthetic league", datetime(2026, 1, 2 + 7*i, 18, tzinfo=timezone.utc))
            for i in range(2)]


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def saved_record(fixture, *, missing_pool=False):
    instant = fixture.kickoff - timedelta(hours=1)
    tracks = {arm: {"id": f"{fixture.match_id}:{arm}", "fixture_id": fixture.match_id,
                    "probabilities": [.5, .3, .2], "model": "synthetic-model",
                    "observed_at" if arm == "market" else "generated_at": instant,
                    **({"exposed_to_statistical": arm == "llm_model_informed",
                        "exposed_to_market": False} if arm in LLM_TRACKS else {})}
              for arm in TRACKS if arm != "joint_pool"}
    if not missing_pool:
        tracks["joint_pool"], reason = arithmetic_pool(
            tracks["statistical"], tracks["llm_independent"],
            {"statistical": .5, "llm": .5}, fixture, instant)
        assert reason is None
    return multi_record(fixture, instant, tracks=tracks,
                        missing_reasons={"joint_pool": "not generated before kickoff"} if missing_pool else {})


def provider_row(fixture, *, final=False, goals=(1, 1)):
    return {"match_id": fixture.match_id, "home": fixture.home, "away": fixture.away,
            "expected_kickoff": fixture.kickoff.isoformat(), "kickoff_time": fixture.kickoff.isoformat(),
            "verified": True, "finished": final, "finished_provisional": True,
            "home_goals": goals[0], "away_goals": goals[1]}


def provider(*, final=False, goals=(1, 1)):
    return {"recorded_at": NOW.isoformat(),
            "source_metadata": {"fixtures": {"status": 200, "received_at": OBSERVED.isoformat(),
                                               "source_url": "https://example.test/official/provider"}},
            "fixture_observations": [provider_row(FIXTURES[0], final=final, goals=goals),
                                     provider_row(FIXTURES[1])]}


def club(raw_path, *, goals=(1, 1)):
    fixture = FIXTURES[0]
    return {"match_id": fixture.match_id, "home": fixture.home, "away": fixture.away,
            "kickoff": fixture.kickoff.isoformat(), "completion_supported": True,
            "source_kind": "official_home_club_match_report",
            "body_completion_evidence": ["Synthetic report explicitly states that the match ended."],
            "original_raw_path": str(raw_path), "score": {"home": goals[0], "away": goals[1]},
            "finished_by_observation": OBSERVED.isoformat(), "recorded_at": NOW.isoformat(),
            "original_received_at": OBSERVED.isoformat(),
            "source_url": "https://example.test/official/club/full-time",
            "published_at": None, "fpl_crosscheck": {"finished": False, "finished_provisional": True}}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return NOW

    monkeypatch.setattr(scoring, "datetime", Clock)
    payload = {"pilot_id": "synthetic:two-record-study", "records": [
        saved_record(FIXTURES[0], missing_pool=True), saved_record(FIXTURES[1])], "results": {},
        "raw_responses": {"synthetic_original": "retain unchanged"}}
    raw = tmp_path / "synthetic-original-body.json"
    write(raw, {"body": "Synthetic official report says the match finished 1-1."})
    return {"root": tmp_path, "case": write(tmp_path / "case.json", payload), "payload": payload,
            "club": club(raw), "club_path": write(tmp_path / "club.json", club(raw)),
            "provider": provider(), "provider_path": write(tmp_path / "provider.json", provider()),
            "fixtures": {fixture.match_id: fixture for fixture in FIXTURES}, "raw": raw}


@pytest.mark.parametrize("source_kind", ["official_home_club_match_report", "official_away_club_match_report"])
def test_club_full_time_can_settle_while_preserving_provisional_provider_and_originals(sandbox, monkeypatch, source_kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-source scoring must not fit models or fetch sources")

    monkeypatch.setattr(DateTeamGoalModel, "fit", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    write(sandbox["club_path"], sandbox["club"] | {"source_kind": source_kind})
    old_bytes = {key: sandbox[key].read_bytes() for key in ("case", "club_path", "provider_path", "raw")}
    output = scoring.run(sandbox["root"], sandbox["case"],
                         [sandbox["provider_path"], sandbox["club_path"]])
    assert output["n_records"] == 2 and output["n_settled"] == output["pending"] == 1
    assert output["all_five_settled"] == 0
    saved = read(output["case"])
    truth = saved["results"][FIXTURES[0].match_id]
    assert truth["finished_by"] == OBSERVED.isoformat() and "finished_at" not in truth
    assert truth["completion_basis"] == "official_completed_match_report"
    assert truth["source_kind"] == source_kind
    assert truth["corroborating_provider"] == {"finished": False, "finished_provisional": True}
    assert saved["records"] == sandbox["payload"]["records"]
    assert saved["raw_responses"] == sandbox["payload"]["raw_responses"]
    assert {key: sandbox[key].read_bytes() for key in old_bytes} == old_bytes
    statuses = [row["status"] for row in saved["result_observations"]]
    assert statuses.count("provisional") == 2
    assert "confirmed_full_time_by_club_report" in statuses
    scores = read(Path(output["output_directory"]) / "同场评分与覆盖.json")
    values = scores["per_fixture"][0]["scores"]["statistical"]
    assert values["log_loss"] == pytest.approx(-math.log(.3))
    assert values["brier"] == pytest.approx(.5**2 + (.3-1)**2 + .2**2)
    assert values["top1_accuracy"] == 0.
    assert scores["comparisons"]["statistical_vs_llm_independent"]["n"] == 1
    assert scores["coverage"]["joint_pool"]["available_with_result"] == 0
    assert saved["records"][0]["tracks"]["joint_pool"] is None
    execution = read(Path(output["output_directory"]) / "执行信息.json")
    assert execution["model_fits"] == execution["llm_calls"] == execution["parameter_changes"] == 0
    assert execution["original_forecasts_changed"] is False


def test_provider_provisional_only_stays_pending(sandbox):
    normalized, observations = scoring.source_results(sandbox["provider_path"], sandbox["fixtures"], NOW)
    assert normalized == {}
    assert all(row["status"] == "provisional" for row in observations)
    outcome = scoring.run(sandbox["root"], sandbox["case"], [sandbox["provider_path"]])
    assert outcome["n_settled"] == 0 and outcome["pending"] == 2


def test_provider_final_flag_provides_completion_bound_not_invented_whistle(sandbox):
    path = write(sandbox["provider_path"], provider(final=True))
    complete, observations = scoring.source_results(path, sandbox["fixtures"], NOW)
    assert list(complete) == [FIXTURES[0].match_id]
    assert complete[FIXTURES[0].match_id]["completion_basis"] == "official_fpl_final_flag"
    assert complete[FIXTURES[0].match_id]["finished_by"] == OBSERVED.isoformat()
    assert "finished_at" not in complete[FIXTURES[0].match_id]
    assert observations[0]["status"] == "confirmed_full_time"
    assert scoring.run(sandbox["root"], sandbox["case"], [path])["n_settled"] == 1


@pytest.mark.parametrize("case", ["wrong_home", "wrong_kickoff", "unknown", "duplicate", "provider_failed"])
def test_invalid_provider_identity_duplicates_or_failed_request_rejected(sandbox, case):
    value = provider(final=True)
    if case == "wrong_home":
        value["fixture_observations"][0]["home"] = "Other home"
    elif case == "wrong_kickoff":
        value["fixture_observations"][0]["kickoff_time"] = (FIXTURES[0].kickoff + timedelta(hours=1)).isoformat()
    elif case == "unknown":
        value["fixture_observations"][0]["match_id"] = "synthetic:unknown"
    elif case == "duplicate":
        value["fixture_observations"].append(deepcopy(value["fixture_observations"][0]))
    else:
        value["source_metadata"]["fixtures"]["status"] = 503
    with pytest.raises(ValueError):
        scoring.source_results(write(sandbox["provider_path"], value), sandbox["fixtures"], NOW)


def test_unverified_provider_identity_is_explicit_missing_not_accepted_final(sandbox):
    value = provider(final=True)
    value["fixture_observations"][0]["verified"] = False
    complete, observations = scoring.source_results(write(sandbox["provider_path"], value), sandbox["fixtures"], NOW)
    assert complete == {}
    assert observations[0]["status"] == "unverified_source_identity"


@pytest.mark.parametrize("changes", [
    {"recorded_at": (NOW + timedelta(seconds=1)).isoformat()},
    {"original_received_at": (NOW + timedelta(seconds=1)).isoformat()},
    {"finished_by_observation": (OBSERVED - timedelta(seconds=1)).isoformat()},
    {"finished_by_observation": FIXTURES[0].kickoff.isoformat()},
    {"score": {"home": True, "away": 1}},
    {"score": {"home": 1, "away": -1}},
    {"score": {"home": 1., "away": 1}},
    {"source_url": "file:///synthetic-report"},
    {"source_url": "https:///missing-host"},
    {"source_kind": "anonymous_summary"},
    {"body_completion_evidence": []},
])
def test_completed_source_requires_valid_observation_goals_url_and_body_support(sandbox, changes):
    source = sandbox["club"] | changes
    with pytest.raises(ValueError):
        scoring.source_results(write(sandbox["club_path"], source), sandbox["fixtures"], NOW)


def test_club_saved_raw_body_must_exist(sandbox):
    source = sandbox["club"] | {"original_raw_path": str(sandbox["root"] / "does-not-exist.json")}
    with pytest.raises(ValueError, match="original report body is missing"):
        scoring.source_results(write(sandbox["club_path"], source), sandbox["fixtures"], NOW)


def test_noncompletion_club_report_does_not_settle(sandbox):
    source = sandbox["club"] | {"completion_supported": False}
    complete, observations = scoring.source_results(write(sandbox["club_path"], source), sandbox["fixtures"], NOW)
    assert complete == {}
    assert observations[0]["status"] == "report_does_not_confirm_completion"


def test_conflicting_final_sources_require_review_and_leave_case_unchanged(sandbox):
    path = write(sandbox["provider_path"], provider(final=True, goals=(2, 1)))
    before = sandbox["case"].read_bytes()
    with pytest.raises(ValueError, match="Conflicting completed results"):
        scoring.run(sandbox["root"], sandbox["case"], [sandbox["club_path"], path])
    assert sandbox["case"].read_bytes() == before
    assert not (sandbox["root"] / "outputs").exists()


@pytest.mark.parametrize("final,goals", [(True, (1, 1)), (False, (9, 9))])
def test_agreeing_final_or_different_pending_source_does_not_replace_existing_result(sandbox, final, goals):
    fixture = FIXTURES[0]
    original = {"fixture_id": fixture.match_id, "home_goals": 1, "away_goals": 1,
                "finished_at": (fixture.kickoff + timedelta(hours=2)).isoformat(),
                "available_at": (fixture.kickoff + timedelta(hours=3)).isoformat(),
                "settlement": SETTLEMENT, "source_url": "https://example.test/original-result",
                "completion_basis": "synthetic_prior_saved_result", "retain_extra": "unchanged"}
    payload = deepcopy(sandbox["payload"])
    payload["results"][fixture.match_id] = original
    write(sandbox["case"], payload)
    path = write(sandbox["provider_path"], provider(final=final, goals=goals))
    output = scoring.run(sandbox["root"], sandbox["case"], [path, sandbox["club_path"]])
    saved = read(output["case"])
    assert saved["results"][fixture.match_id] == original
    assert read(sandbox["case"]) == payload
    assert output["n_settled"] == 1


def test_weekly_scores_and_report_keep_fixture_counts_not_line_counts(sandbox):
    results, _ = scoring.source_results(sandbox["club_path"], sandbox["fixtures"], NOW)
    records = sandbox["payload"]["records"]
    periods = scoring.weekly_scores(records, results)
    assert list(periods) == ["2026-W01", "2026-W02"]
    assert periods["2026-W01"]["n_records"] == periods["2026-W01"]["n_settled"] == 1
    assert periods["2026-W01"]["all_five_common"]["n"] == 0
    assert periods["2026-W02"]["n_records"] == 1 and periods["2026-W02"]["n_settled"] == 0
    scores = scoring.score_multi(records, results)
    payload = sandbox["payload"] | {"results": results}
    rendered = scoring.report(payload, scores, periods)
    assert "比赛记录2场，已录入完赛1场，待结算1场" in rendered
    assert "2026-W01：已结算1场；全五线共同已结算0场" in rendered
    assert "2026-W02：已结算0场；全五线共同已结算0场" in rendered
    assert "[synthetic:score:0](https://example.test/official/club/full-time)" in rendered
    assert "finished_by只是该时点已确认比赛结束的上界" in rendered
    assert "本次球队拟合0次、LLM调用0次、参数搜索／变更0次" in rendered
