"""Synthetic-only future-version assembly checks; never forecasting evidence."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

import football_model.prospective_revision as revision
from football_model.domain import Fixture
from football_model.multiline import arithmetic_pool, build_packet, multi_record, score_multi


NOW = datetime(2026, 10, 24, 5, 30, tzinfo=timezone.utc)
PREPARED = NOW - timedelta(minutes=10)


def write(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def case(tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return NOW
    monkeypatch.setattr(revision, "datetime", Clock)
    earlier = NOW - timedelta(days=1)
    fixtures = [Fixture("old", "Old H", "Old A", "Synthetic", NOW - timedelta(hours=5)),
                Fixture("future", "Future H", "Future A", "Synthetic", NOW + timedelta(hours=5))]
    records = []
    for f in fixtures:
        stat = {"id": "stat:" + f.match_id, "fixture_id": f.match_id, "generated_at": earlier.isoformat(),
                "probabilities": [.5, .3, .2], "model": "synthetic", "description": "old statistical"}
        llm = {"id": "llm:" + f.match_id, "fixture_id": f.match_id, "generated_at": earlier.isoformat(),
               "probabilities": [.4, .35, .25], "model": "synthetic", "exposed_to_statistical": False,
               "exposed_to_market": False}
        informed = llm | {"id": "informed:" + f.match_id, "exposed_to_statistical": True}
        market = {"id": "market:" + f.match_id, "fixture_id": f.match_id, "observed_at": earlier.isoformat(),
                  "probabilities": [.6, .25, .15], "bookmaker": "Synthetic"}
        pool, _ = arithmetic_pool(stat, llm, revision.WEIGHTS, f, earlier)
        records.append(multi_record(f, earlier, tracks={"statistical": stat, "llm_independent": llm,
            "llm_model_informed": informed, "market": market, "joint_pool": pool}))
    payload = {"pilot_id": "synthetic", "records": records,
        "results": {"old": {"an_existing_outcome": "not imported"}},
        "policy": {"pool_weights": deepcopy(revision.WEIGHTS)},
        "preparation": {"prepared_at": earlier.isoformat()}, "projection_path": "synthetic saved projection",
        "market_source": {"market_observations": [{"source": "old cached prices"}]}}
    old_path = tmp_path / "old.json"
    write(old_path, payload)
    directory = tmp_path / "packets"
    directory.mkdir()
    documents = {}
    responses = {}
    f = fixtures[1]
    evidence = [{"id": "fact", "url": "https://example.test/news", "claim": "Synthetic player news",
        "claim_type": "fact", "team": f.home, "category": "availability", "status": "source assertion",
        "published_at": None, "observed_at": (PREPARED-timedelta(minutes=2)).isoformat()}]
    context = [{"id": "raw", "text": "Synthetic descriptive results, not model probabilities",
                "observed_at": earlier.isoformat()}]
    for mode in ("independent", "model_informed"):
        packet = build_packet(f, PREPARED, evidence, ["Unknown lineup"], descriptive_context=context,
            statistical=records[1]["tracks"]["statistical"] if mode == "model_informed" else None, mode=mode)
        documents[mode] = {"prepared_at": PREPARED.isoformat(), "mode": mode, "packets": [packet]}
        responses[mode] = {"forecasts": [{"fixture_id": f.match_id,
            "generated_at_utc": (PREPARED+timedelta(minutes=2)).isoformat(),
            "exposed_to_statistical": mode == "model_informed", "exposed_to_market": False,
            "model_identifier": "unknown", "decision": "forecast", "probabilities": [.3, .4, .3],
            "reasoning_summary": "Synthetic, not real forecasting", "calibration_status": "unvalidated"}]}
    return {"root": tmp_path, "old_path": old_path, "old": payload, "directory": directory,
            "documents": documents, "responses": responses}


def execute(case):
    for mode in case["documents"]:
        write(case["directory"] / f"{mode}_packets.json", case["documents"][mode])
        write(case["directory"] / f"{mode}_response.json", case["responses"][mode])
    return revision.run(case["root"], case["old_path"], case["directory"])


def test_reuses_old_observations_without_importing_results_or_changing_input(case):
    before = case["old_path"].read_bytes()
    original = deepcopy(case)
    result = execute(case)
    payload = read(result["case"])
    assert case["old_path"].read_bytes() == before
    assert case == original
    assert result["records"] == 1 and result["settled"] == 0
    assert payload["results"] == {}
    row = payload["records"][0]
    for arm in ("statistical", "market"):
        assert row["tracks"][arm] == case["old"]["records"][1]["tracks"][arm]
    assert row["tracks"]["joint_pool"]["weights"] == revision.WEIGHTS
    assert row["tracks"]["joint_pool"]["probabilities"] == pytest.approx([.4, .35, .25])
    assert payload["revision"]["model_fits"] == payload["revision"]["parameter_changes"] == 0
    assert payload["revision"]["old_market_refreshed"] is False
    assert score_multi(payload["records"], {})["n_records"] == 1
    assert "原独立LLM" in Path(result["report"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("field,value", [("home", "Other"), ("away", "Other"),
    ("competition", "Other"), ("kickoff", (NOW+timedelta(hours=6)).isoformat()), ("neutral", True)])
def test_packet_identity_cannot_change(case, field, value):
    case["documents"]["independent"]["packets"][0]["fixture"][field] = value
    with pytest.raises(ValueError, match="identity"):
        execute(case)


@pytest.mark.parametrize("mode", ["independent", "model_informed"])
def test_duplicate_or_unknown_packet_is_rejected(case, mode):
    case["documents"][mode]["packets"].append(deepcopy(case["documents"][mode]["packets"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        execute(case)


def test_old_case_duplicate_fixture_is_not_an_extra_observation(case):
    case["old"]["records"].append(deepcopy(case["old"]["records"][1]))
    write(case["old_path"], case["old"])
    with pytest.raises(ValueError, match="duplicate"):
        execute(case)


@pytest.mark.parametrize("at", [PREPARED-timedelta(seconds=1), PREPARED, NOW, NOW+timedelta(seconds=1)])
def test_response_strict_time_order(case, at):
    case["responses"]["independent"]["forecasts"][0]["generated_at_utc"] = at.isoformat()
    with pytest.raises(ValueError):
        execute(case)


def test_packet_after_current_time_rejected(case):
    case["documents"]["independent"]["prepared_at"] = (NOW+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="mode/time"):
        execute(case)


def test_packet_and_document_times_must_match(case):
    case["documents"]["independent"]["packets"][0]["as_of"] = (PREPARED-timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="prepared_at"):
        execute(case)


def test_both_strategies_use_same_preparation(case):
    document = case["documents"]["model_informed"]
    document["prepared_at"] = (PREPARED-timedelta(seconds=1)).isoformat()
    document["packets"][0]["as_of"] = document["prepared_at"]
    with pytest.raises(ValueError, match="same preparation"):
        execute(case)


def test_started_fixture_not_revised(case):
    row = case["old"]["records"][1]
    row["fixture"]["kickoff"] = NOW.isoformat()
    write(case["old_path"], case["old"])
    for doc in case["documents"].values():
        doc["packets"][0]["fixture"]["kickoff"] = NOW.isoformat()
    with pytest.raises(ValueError, match="precede every"):
        execute(case)


@pytest.mark.parametrize("change", [{"exposed_to_market": True}, {"exposed_to_statistical": True},
    {"probabilities": [.5, .5, .5]}, {"decision": "abstain"}, {"model_identifier": ""}])
def test_response_semantics_reuse_existing_rules(case, change):
    case["responses"]["independent"]["forecasts"][0].update(change)
    with pytest.raises(ValueError):
        execute(case)


def test_abstention_stays_missing_and_does_not_impute_pool(case):
    case["responses"]["independent"]["forecasts"][0].update(decision="abstain", probabilities=None)
    result = execute(case)
    payload = read(result["case"])
    tracks = payload["records"][0]["tracks"]
    assert tracks["llm_independent"] is tracks["joint_pool"] is None
    assert tracks["llm_model_informed"] is not None
    scores = read(Path(result["run_directory"]) / "initial_scores.json")
    assert scores["coverage"]["llm_independent"]["abstained"] == 1


def test_abstention_response_still_requires_later_time(case):
    case["responses"]["independent"]["forecasts"][0].update(decision="abstain", probabilities=None,
                                                                          generated_at_utc=PREPARED.isoformat())
    with pytest.raises(ValueError, match="strictly"):
        execute(case)


def test_missing_response_keeps_missing_line_and_pool(case):
    case["responses"]["independent"]["forecasts"] = []
    payload = read(execute(case)["case"])
    assert payload["records"][0]["tracks"]["llm_independent"] is None
    assert payload["records"][0]["tracks"]["joint_pool"] is None


def test_fixed_policy_cannot_be_tuned(case):
    case["old"]["policy"]["pool_weights"] = {"statistical": .6, "llm": .4}
    write(case["old_path"], case["old"])
    with pytest.raises(ValueError, match="50:50"):
        execute(case)


def test_policy_must_precede_new_packets(case):
    case["old"]["preparation"]["prepared_at"] = NOW.isoformat()
    write(case["old_path"], case["old"])
    with pytest.raises(ValueError, match="policy must precede"):
        execute(case)


def test_informed_cannot_swap_statistical_reference(case):
    case["documents"]["model_informed"]["packets"][0]["statistical"]["probabilities"] = [.2, .3, .5]
    with pytest.raises(ValueError, match="unchanged"):
        execute(case)


def test_independent_cannot_receive_statistical_line(case):
    case["documents"]["independent"]["packets"][0]["statistical"] = case["old"]["records"][1]["tracks"]["statistical"]
    with pytest.raises(ValueError, match="cannot contain"):
        execute(case)


@pytest.mark.parametrize("section,timefield", [("evidence", "observed_at"), ("evidence", "published_at"),
                                              ("descriptive_context", "observed_at")])
def test_future_source_evidence_not_silently_removed(case, section, timefield):
    case["documents"]["independent"]["packets"][0][section][0][timefield] = (PREPARED+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="unavailable"):
        execute(case)


def test_two_modes_cannot_sneak_different_news(case):
    case["documents"]["model_informed"]["packets"][0]["evidence"][0]["claim"] = "Different input"
    with pytest.raises(ValueError, match="more than statistical"):
        execute(case)


def test_duplicate_response_rejected(case):
    case["responses"]["independent"]["forecasts"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        execute(case)


def test_different_versions_cannot_be_concatenated_as_more_samples(case):
    payload = read(execute(case)["case"])
    with pytest.raises(ValueError, match="One fixture"):
        score_multi([case["old"]["records"][1], payload["records"][0]], {})


def test_cli_defaults_use_packet_directory_responses(case, capsys):
    for mode in case["documents"]:
        write(case["directory"] / f"{mode}_packets.json", case["documents"][mode])
        write(case["directory"] / f"{mode}_response.json", case["responses"][mode])
    assert revision.main(["--root", str(case["root"]), "--case", str(case["old_path"]),
                          "--packet-dir", str(case["directory"])]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == 1
