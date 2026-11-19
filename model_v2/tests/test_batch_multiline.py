"""Synthetic batch comparison tests; no real responses/results are edited.

Every packet, forecast, market quote and settlement below belongs to a temporary
test project. None is an observation of real forecasting performance.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import socket
import urllib.request

import pytest

import football_model.batch_multiline as batch
from football_model.date_model import DateTeamGoalModel
from football_model.domain import OddsSnapshot
from football_model.multiline import PAIRS, TRACKS
from football_model.probability import implied


NOW = datetime(2026, 10, 23, 18, tzinfo=timezone.utc)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def description(team, season, empty=False):
    games = [] if empty else [
        {"date": "2026-10-10", "home": team, "away": "Opponent", "score": [2, 1],
         "source": "https://example.test/results/1"},
        {"date": "2026-10-18", "home": "Opponent", "away": team, "score": [3, 1],
         "source": "https://example.test/results/2"},
    ]
    return {"team": team, "season": season, "played": len(games), "won": int(not empty),
            "drawn": 0, "lost": int(not empty), "goals_for": 0 if empty else 3,
            "goals_against": 0 if empty else 4, "points": 0 if empty else 3, "results": games}


def projection_case(index):
    fid = f"synthetic-fixture:{index}"
    home, away = f"Home{index}", f"Away{index}"
    return {
        "fixture": {"match_id": fid, "home": home, "away": away, "competition": "Premier League",
                    "kickoff": "2026-10-24T14:00:00+00:00", "neutral": False},
        "track": {"id": f"statistical:{fid}", "fixture_id": fid,
                  "generated_at": "2026-10-23T17:49:00+00:00", "probabilities": [.5, .3, .2],
                  "model": "synthetic_fixed_poisson", "description": "Synthetic earlier-data projection"},
        "rates": {"home": 1.4, "away": 1.0},
        "forecast": {"total_goals": {"0": .1, "1": .2, "2": .3, "3": .2, "4+": .2}, "over_2_5": .4},
        "missing_reasons": {},
        "descriptive_team_results": [description(home, "2025/26", empty=True), description(home, "2026/27"),
                                     description(away, "2025/26"), description(away, "2026/27")],
    }


def response(case, mode, **changes):
    return {"fixture_id": case["fixture"]["match_id"], "generated_at_utc": "2026-10-23T18:01:00Z",
            "exposed_to_statistical": mode == "model_informed", "exposed_to_market": False,
            "model_identifier": "unknown", "decision": "forecast", "probabilities": [.4, .35, .25],
            "reasoning_summary": "Synthetic response: descriptive form is uncertain, not calibrated strength.",
            "calibration_status": "uncalibrated_research_forecast"} | changes


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    clock = [NOW]

    class Clock:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return clock[0]

    monkeypatch.setattr(batch, "datetime", Clock)
    cases = [projection_case(1), projection_case(2)]
    projection = {"generated_at": "2026-10-23T17:50:00Z", "fit_count": 1, "parameter_searches": 0,
                  "training_matches": 10, "current_season_matches": 2,
                  "current_missing_reported_results": 0, "fixtures": cases,
                  "model_parameters": {"regularization": .01, "half_life_days": 180}}
    evidence = {case["fixture"]["match_id"]: [{
        "id": f"availability:{index}", "url": "https://example.test/official/news",
        "claim": "Player is flagged doubtful; no confirmed starting lineup.", "claim_type": "fact",
        "team": case["fixture"]["home"], "category": "availability", "status": "doubtful",
        "observed_at": "2026-10-23T17:40:00Z", "published_at": None,
        "source_kind": "official_cached_metadata", "retrieval_note": "Publication time unknown"}]
        for index, case in enumerate(cases)}
    news = {"evidence_by_fixture": evidence,
            "unknowns_by_fixture": {cases[0]["fixture"]["match_id"]: ["Synthetic unknown exact goalkeeper availability"]}}
    observations = [{"fixture_id": case["fixture"]["match_id"], "decimal_prices": {"home": 2.2, "draw": 3.1, "away": 3.4},
                     "market": "regulation_full_time_one_x_two", "observed_at": "2026-10-23T17:45:00Z",
                     "bookmaker": "SyntheticBook", "source": "bookmaker_listing_page", "source_event_id": None,
                     "source_url": "https://example.test/bookmaker/fixtures", "opening_status": "unknown"}
                    for case in cases]
    old_single = tmp_path / "data/prematch-intelligence/20261024_multiline/multiline_case.json"
    old_single.parent.mkdir(parents=True)
    old_single.write_text('{"legacy_single_case":"never edit"}', encoding="utf-8")
    return {"root": tmp_path, "clock": clock, "projection": projection,
            "projection_path": write(tmp_path / "projection.json", projection),
            "news": news, "news_path": write(tmp_path / "news.json", news),
            "market": {"market_observations": observations},
            "market_path": write(tmp_path / "market.json", {"market_observations": observations}),
            "old_single": old_single}


def prepared(sandbox):
    result = batch.prepare(sandbox["root"], sandbox["projection_path"], sandbox["news_path"])
    directory = Path(result["run_directory"])
    for mode in ("independent", "model_informed"):
        packets = read(directory / f"{mode}_packets.json")["packets"]
        by_id = {case["fixture"]["match_id"]: case for case in sandbox["projection"]["fixtures"]}
        write(directory / f"{mode}_response.json", {"forecasts": [response(by_id[p["fixture"]["match_id"]], mode)
                                                                  for p in packets]})
    sandbox["clock"][0] = NOW + timedelta(minutes=2)
    return directory


def assembled(sandbox):
    directory = prepared(sandbox)
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    return result, read(result["case"]), directory


def update_response(directory, mode, mutator):
    path = directory / f"{mode}_response.json"
    payload = read(path)
    mutator(payload)
    write(path, payload)


def test_prepare_isolates_probabilities_and_saves_weights_before_responses(sandbox):
    result = batch.prepare(sandbox["root"], sandbox["projection_path"], sandbox["news_path"])
    directory = Path(result["run_directory"])
    assert result["independent_packets"] == result["model_informed_packets"] == 2
    preparation = read(directory / "preparation.json")
    assert preparation["policy"]["pool_weights"] == {"statistical": .5, "llm": .5}
    assert "untrained" in preparation["policy"]["pool_weight_status"]
    assert preparation["prepared_at"] == NOW.isoformat()
    assert not (directory / "independent_response.json").exists()
    assert not (directory / "model_informed_response.json").exists()
    independents = read(directory / "independent_packets.json")["packets"]
    informed = read(directory / "model_informed_packets.json")["packets"]
    for independent, known, case in zip(independents, informed, sandbox["projection"]["fixtures"]):
        assert "statistical" not in independent and "market" not in independent
        assert "probabilities" not in json.dumps([json.loads(row["text"]) for row in independent["descriptive_context"]])
        assert known["statistical"]["probabilities"] == case["track"]["probabilities"]
        assert "market" not in known
        assert independent["evidence"] == known["evidence"]
        assert independent["descriptive_context"] == known["descriptive_context"]
        assert independent["as_of"] == known["as_of"] == NOW.isoformat()
        assert "exposed_to_statistical=false" in independent["instructions"]
        assert "exposed_to_statistical=true" in known["instructions"]
        assert "untrusted data" in independent["instructions"]
        assert all("official_cached_metadata" == fact["source_kind"] for fact in independent["evidence"])
    assert "Synthetic unknown exact goalkeeper availability" in independents[0]["unknowns"]


def test_descriptive_summary_handles_no_history_without_zero_strength_or_imputed_probability(sandbox):
    result = batch.prepare(sandbox["root"], sandbox["projection_path"], sandbox["news_path"])
    packets = read(Path(result["run_directory"]) / "independent_packets.json")["packets"]
    compact = [json.loads(row["text"]) for row in packets[0]["descriptive_context"]]
    empty = next(row for row in compact if row["season"] == "2025/26" and row["team"] == "Home1")
    assert empty["played"] == 0 and "not zero ability" in empty["history_note"]
    assert "last_five_results" not in empty and "probabilities" not in empty and "strength" not in empty
    home = next(row for row in compact if row["season"] == "2026/27" and row["team"] == "Home1")
    assert home["home_descriptive"] == {"played": 1, "goals_for": 2, "goals_against": 1}
    assert home["away_descriptive"] == {"played": 1, "goals_for": 1, "goals_against": 3}
    assert len(home["last_five_results"]) == 2 and "results" not in home


def test_prepare_selects_only_news_observed_and_published_by_packet_time(sandbox):
    fid = sandbox["projection"]["fixtures"][0]["fixture"]["match_id"]
    original = sandbox["news"]["evidence_by_fixture"][fid][0]
    sandbox["news"]["evidence_by_fixture"][fid] += [
        original | {"id": "future-observation", "observed_at": "2026-10-23T18:00:01Z"},
        original | {"id": "future-publication", "published_at": "2026-10-23T18:00:01Z"},
    ]
    write(sandbox["news_path"], sandbox["news"])
    directory = prepared(sandbox)
    for mode in ("independent", "model_informed"):
        packet = read(directory / f"{mode}_packets.json")["packets"][0]
        assert [row["id"] for row in packet["evidence"]] == [original["id"]]


def test_complete_assembly_reuses_quote_validation_and_proportional_devig(sandbox, monkeypatch):
    directory = prepared(sandbox)
    seen_snapshots, calls = [], []

    def constructor(**kwargs):
        snapshot = OddsSnapshot(**kwargs)
        seen_snapshots.append(snapshot)
        return snapshot

    def devig(snapshot):
        calls.append(snapshot)
        return implied(snapshot)

    monkeypatch.setattr(batch, "OddsSnapshot", constructor)
    monkeypatch.setattr(batch, "implied", devig)
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload, scores = read(result["case"]), read(directory / "initial_scores.json")
    assert result["records"] == 2 and result["settled"] == 0
    assert result["available"] == dict.fromkeys(TRACKS, 2)
    assert len(seen_snapshots) == len(calls) == 2
    assert all(a is b for a, b in zip(seen_snapshots, calls))
    expected_market = [(1 / price) / sum(1 / p for p in (2.2, 3.1, 3.4)) for price in (2.2, 3.1, 3.4)]
    for record in payload["records"]:
        tracks = record["tracks"]
        assert tracks["market"]["probabilities"] == pytest.approx(expected_market)
        assert tracks["joint_pool"]["probabilities"] == pytest.approx([.45, .325, .225])
        assert tracks["joint_pool"]["weights"] == {"statistical": .5, "llm": .5}
        assert tracks["joint_pool"]["weight_source"] == "explicit_untrained_research"
        assert tracks["llm_independent"]["exposed_to_statistical"] is False
        assert tracks["llm_model_informed"]["exposed_to_statistical"] is True
        assert tracks["llm_model_informed"]["exposed_to_market"] is False
        assert record["missing_reasons"] == {}
    assert scores["n_settled"] == scores["all_five_common"]["n"] == 0
    assert len(scores["comparisons"]) == len(PAIRS) == 10
    assert all(item["n"] == 0 for item in scores["comparisons"].values())


def test_original_response_identity_times_reasoning_and_unknown_event_ids_are_preserved(sandbox):
    result, payload, directory = assembled(sandbox)
    for mode in ("independent", "model_informed"):
        raw = read(directory / f"{mode}_response.json")["forecasts"]
        assert payload["llm_responses"][mode] == {row["fixture_id"]: row for row in raw}
        for row in raw:
            arm = "llm_independent" if mode == "independent" else "llm_model_informed"
            track = next(r for r in payload["records"] if r["fixture"]["match_id"] == row["fixture_id"])["tracks"][arm]
            assert track["probabilities"] == row["probabilities"]
            assert datetime.fromisoformat(track["generated_at"]) == datetime.fromisoformat(row["generated_at_utc"])
            assert track["description"] == row["calibration_status"]
            assert "model_id=unknown" in track["model"]
    assert payload["market_source"] == sandbox["market"]
    assert all(q["source_event_id"] is None and q["opening_status"] == "unknown"
               for q in payload["market_source"]["market_observations"])
    assert all("source_event_id" not in r["tracks"]["market"] for r in payload["records"])


def early_fixture_prepared(sandbox, minutes_to_kickoff=60):
    """Prepare before kickoff, then let the synthetic clock advance normally."""
    sandbox["projection"]["fixtures"][0]["fixture"]["kickoff"] = (
        NOW + timedelta(minutes=minutes_to_kickoff)).isoformat()
    write(sandbox["projection_path"], sandbox["projection"])
    return prepared(sandbox)


def test_late_assembly_archives_existing_prematch_tracks_without_backdating_a_new_pool(sandbox):
    directory = early_fixture_prepared(sandbox)
    original_responses = {mode: read(directory / f"{mode}_response.json")
                          for mode in ("independent", "model_informed")}
    sandbox["clock"][0] = NOW + timedelta(hours=2)
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload = read(result["case"])
    archived, future = payload["records"]
    assert payload["assembled_at"] == sandbox["clock"][0].isoformat()
    assert archived["information_origin"] == "archived_existing_prematch_tracks"
    assert datetime.fromisoformat(archived["as_of"]) == NOW + timedelta(minutes=1)
    assert archived["tracks"]["joint_pool"] is None
    assert archived["missing_reasons"]["joint_pool"].startswith("not_prematch:")
    assert future["information_origin"] == "assembled_before_kickoff"
    assert future["as_of"] == payload["assembled_at"]
    assert future["tracks"]["joint_pool"] is not None
    assert future["tracks"]["joint_pool"]["generated_at"] == payload["assembled_at"]
    assert result["available"] == dict.fromkeys(TRACKS, 2) | {"joint_pool": 1}
    for arm in ("statistical", "market", "llm_independent", "llm_model_informed"):
        timestamp = ("observed_at" if arm == "market" else "generated_at")
        assert datetime.fromisoformat(archived["tracks"][arm][timestamp]) < datetime.fromisoformat(archived["fixture"]["kickoff"])
    for mode in ("independent", "model_informed"):
        assert read(directory / f"{mode}_response.json") == original_responses[mode]
        original = original_responses[mode]["forecasts"][0]
        arm = "llm_independent" if mode == "independent" else "llm_model_informed"
        assert datetime.fromisoformat(archived["tracks"][arm]["generated_at"]) == datetime.fromisoformat(original["generated_at_utc"])


@pytest.mark.parametrize("mode,arm", [("independent", "llm_independent"), ("model_informed", "llm_model_informed")])
@pytest.mark.parametrize("seconds_after_kickoff", [0, 1])
def test_late_llm_response_is_kept_raw_but_excluded_from_prematch_coverage(sandbox, mode, arm, seconds_after_kickoff):
    directory = early_fixture_prepared(sandbox, minutes_to_kickoff=1)
    other_mode = "model_informed" if mode == "independent" else "independent"
    update_response(directory, other_mode, lambda p: p["forecasts"][0].update(generated_at_utc="2026-10-23T18:00:30Z"))
    observed = NOW + timedelta(minutes=1, seconds=seconds_after_kickoff)
    update_response(directory, mode, lambda p: p["forecasts"][0].update(generated_at_utc=observed.isoformat()))
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload = read(result["case"])
    archived = payload["records"][0]
    assert archived["information_origin"] == "archived_existing_prematch_tracks"
    assert archived["tracks"][arm] is None
    assert archived["missing_reasons"][arm].startswith("not_prematch:")
    assert archived["tracks"]["joint_pool"] is None
    assert datetime.fromisoformat(archived["as_of"]) == NOW + timedelta(seconds=30)
    assert payload["llm_responses"][mode][archived["fixture"]["match_id"]]["generated_at_utc"] == observed.isoformat()
    assert result["available"][arm] == 1
    assert result["available"]["joint_pool"] == 1
    scores = read(directory / "initial_scores.json")
    assert scores["coverage"][arm]["available"] == scores["coverage"][arm]["missing"] == 1
    assert scores["coverage"][arm]["abstained"] == 0


@pytest.mark.parametrize("seconds_after_kickoff", [0, 1])
def test_late_market_quote_is_not_used_as_prematch_information_or_archive_time(sandbox, seconds_after_kickoff):
    directory = early_fixture_prepared(sandbox, minutes_to_kickoff=1)
    for mode in ("independent", "model_informed"):
        update_response(directory, mode, lambda p: p["forecasts"][0].update(generated_at_utc="2026-10-23T18:00:30Z"))
    observed = NOW + timedelta(minutes=1, seconds=seconds_after_kickoff)
    sandbox["market"]["market_observations"][0]["observed_at"] = observed.isoformat()
    write(sandbox["market_path"], sandbox["market"])
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload = read(result["case"])
    archived = payload["records"][0]
    assert archived["tracks"]["market"] is None
    assert archived["missing_reasons"]["market"].startswith("not_prematch:")
    assert archived["tracks"]["joint_pool"] is None
    assert datetime.fromisoformat(archived["as_of"]) == NOW + timedelta(seconds=30)
    assert payload["market_source"] == sandbox["market"]
    assert result["available"]["market"] == result["available"]["joint_pool"] == 1
    assert payload["records"][1]["tracks"]["market"] is not None


def test_archive_as_of_uses_prepared_information_when_no_later_prematch_track_survives(sandbox):
    directory = early_fixture_prepared(sandbox, minutes_to_kickoff=1)
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload = read(result["case"])
    archived = payload["records"][0]
    assert datetime.fromisoformat(archived["as_of"]) == NOW
    assert archived["tracks"]["llm_independent"] is archived["tracks"]["llm_model_informed"] is None
    assert archived["tracks"]["statistical"] is not None and archived["tracks"]["market"] is not None
    assert payload["assembled_at"] == (NOW + timedelta(minutes=2)).isoformat()
    assert datetime.fromisoformat(archived["as_of"]) < datetime.fromisoformat(payload["assembled_at"])


def test_assembly_exactly_at_kickoff_is_archival_and_does_not_create_new_pool(sandbox):
    directory = early_fixture_prepared(sandbox)
    sandbox["clock"][0] = NOW + timedelta(hours=1)
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    archived = read(result["case"])["records"][0]
    assert archived["information_origin"] == "archived_existing_prematch_tracks"
    assert archived["tracks"]["joint_pool"] is None
    assert result["available"]["joint_pool"] == 1


def test_saved_pre_response_weights_not_later_module_default_drive_pool(sandbox, monkeypatch):
    directory = prepared(sandbox)
    monkeypatch.setattr(batch, "POLICY", deepcopy(batch.POLICY) | {"pool_weights": {"statistical": .9, "llm": .1}})
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload = read(result["case"])
    assert payload["policy"]["pool_weights"] == {"statistical": .5, "llm": .5}
    assert payload["records"][0]["tracks"]["joint_pool"]["probabilities"] == pytest.approx([.45, .325, .225])
    prepared_at = datetime.fromisoformat(payload["preparation"]["prepared_at"])
    assert all(prepared_at < datetime.fromisoformat(row["generated_at_utc"])
               for rows in payload["llm_responses"].values() for row in rows.values())


def test_missing_market_remains_missing_without_copying_other_match_or_equal_probabilities(sandbox):
    sandbox["market"]["market_observations"] = sandbox["market"]["market_observations"][:1]
    write(sandbox["market_path"], sandbox["market"])
    result, payload, directory = assembled(sandbox)
    assert result["available"]["market"] == 1
    assert payload["records"][0]["tracks"]["market"] is not None
    second = payload["records"][1]
    assert second["tracks"]["market"] is None
    assert "no imputation" in second["missing_reasons"]["market"]
    assert second["tracks"]["joint_pool"] is not None
    scores = read(directory / "initial_scores.json")
    assert scores["coverage"]["market"]["missing"] == 1


@pytest.mark.parametrize("mode,arm", [("independent", "llm_independent"), ("model_informed", "llm_model_informed")])
def test_explicit_abstention_is_preserved_and_only_independent_abstention_blocks_pool(sandbox, mode, arm):
    directory = prepared(sandbox)
    update_response(directory, mode, lambda payload: payload["forecasts"][0].update(
        decision="abstain", probabilities=None, reasoning_summary="Insufficient sourced evidence"))
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload, scores = read(result["case"]), read(directory / "initial_scores.json")
    record = payload["records"][0]
    assert record["tracks"][arm] is None
    assert record["missing_reasons"][arm] == "abstained: Insufficient sourced evidence"
    assert scores["coverage"][arm]["abstained"] == 1
    assert (record["tracks"]["joint_pool"] is None) == (mode == "independent")
    assert result["available"][arm] == 1


def test_unsupplied_response_is_distinct_from_abstention(sandbox):
    directory = prepared(sandbox)
    update_response(directory, "independent", lambda payload: payload["forecasts"].pop())
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    payload, scores = read(result["case"]), read(directory / "initial_scores.json")
    record = payload["records"][1]
    assert record["tracks"]["llm_independent"] is record["tracks"]["joint_pool"] is None
    assert record["missing_reasons"]["llm_independent"] == "not_supplied"
    assert scores["coverage"]["llm_independent"]["abstained"] == 0
    assert "llm" in record["missing_reasons"]["joint_pool"]


def test_no_statistical_prediction_still_gets_independent_packet_but_no_informed_packet(sandbox):
    case = sandbox["projection"]["fixtures"][1]
    case["track"] = None
    case["missing_reasons"] = {"statistical": "Unseen team without an estimated prior"}
    write(sandbox["projection_path"], sandbox["projection"])
    directory = prepared(sandbox)
    assert len(read(directory / "independent_packets.json")["packets"]) == 2
    assert len(read(directory / "model_informed_packets.json")["packets"]) == 1
    result = batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    record = read(result["case"])["records"][1]
    assert record["tracks"]["statistical"] is record["tracks"]["llm_model_informed"] is record["tracks"]["joint_pool"] is None
    assert record["tracks"]["llm_independent"] is not None
    assert record["missing_reasons"]["statistical"] == "Unseen team without an estimated prior"


def test_numerical_informed_response_cannot_claim_an_unsupplied_statistical_packet(sandbox):
    case = sandbox["projection"]["fixtures"][1]
    case["track"] = None
    case["missing_reasons"] = {"statistical": "Unknown prior"}
    write(sandbox["projection_path"], sandbox["projection"])
    directory = prepared(sandbox)
    update_response(directory, "model_informed", lambda payload: payload["forecasts"].append(response(case, "model_informed")))
    with pytest.raises(ValueError, match="Unknown|packet"):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])


@pytest.mark.parametrize("changes,message", [
    ({"fixture_id": "not-in-the-batch"}, "Unknown|duplicate"),
    ({"generated_at_utc": "2026-10-23T17:59:59Z"}, "follow its packet"),
    ({"generated_at_utc": "2026-10-23T18:02:01Z"}, "precede assembly"),
    ({"generated_at_utc": "2026-10-23T18:01:00"}, "timezone-aware"),
    ({"exposed_to_statistical": True}, "exposure"),
    ({"exposed_to_market": True}, "exposure"),
    ({"exposed_to_statistical": 0}, "exposure"),
    ({"model_identifier": ""}, "nonempty"),
    ({"reasoning_summary": ""}, "nonempty"),
    ({"calibration_status": ""}, "nonempty"),
    ({"decision": "abstain", "probabilities": [.4, .35, .25]}, "abstention"),
    ({"decision": "forecast", "probabilities": None}, "abstention"),
    ({"probabilities": [.4, .4, .4]}, "sum to one"),
    ({"probabilities": [-.1, .6, .5]}, "finite real"),
    ({"probabilities": [True, 0, 0]}, "finite real"),
    ({"probabilities": [float("nan"), .5, .5]}, "finite real"),
])
def test_invalid_response_identity_time_exposure_and_probabilities_are_rejected(sandbox, changes, message):
    directory = prepared(sandbox)
    update_response(directory, "independent", lambda payload: payload["forecasts"][0].update(changes))
    with pytest.raises(ValueError, match=message):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    assert not (directory / "batch_case.json").exists()


def test_duplicate_response_is_not_counted_as_a_second_observation(sandbox):
    directory = prepared(sandbox)
    update_response(directory, "independent", lambda payload: payload["forecasts"].append(deepcopy(payload["forecasts"][0])))
    with pytest.raises(ValueError, match="duplicate"):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])


@pytest.mark.parametrize("changes,message", [
    ({"fixture_id": "wrong-fixture"}, "Unknown|duplicate"),
    ({"market": "early_payout_2up"}, "standard regulation"),
    ({"observed_at": "2026-10-23T18:02:01Z"}, "not available"),
    ({"observed_at": "2026-10-23T17:45:00"}, "timezone-aware"),
    ({"decimal_prices": {"home": 1., "draw": 3., "away": 4.}}, "greater than 1"),
    ({"decimal_prices": {"home": float("inf"), "draw": 3., "away": 4.}}, "finite"),
])
def test_unavailable_wrong_fixture_wrong_market_or_invalid_quote_is_rejected(sandbox, changes, message):
    directory = prepared(sandbox)
    sandbox["market"]["market_observations"][0].update(changes)
    write(sandbox["market_path"], sandbox["market"])
    with pytest.raises(ValueError, match=message):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])


def test_incomplete_quote_and_duplicate_quote_are_not_filled_or_chosen_by_value(sandbox):
    directory = prepared(sandbox)
    market = deepcopy(sandbox["market"])
    del market["market_observations"][0]["decimal_prices"]["draw"]
    write(sandbox["market_path"], market)
    with pytest.raises(TypeError):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    market = deepcopy(sandbox["market"])
    market["market_observations"].append(deepcopy(market["market_observations"][0]))
    write(sandbox["market_path"], market)
    with pytest.raises(ValueError, match="duplicate"):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])


@pytest.mark.parametrize("change", ["add", "remove", "home", "kickoff"])
def test_reloaded_projection_must_match_the_actually_prepared_fixture_identities(sandbox, change):
    directory = prepared(sandbox)
    projection = deepcopy(sandbox["projection"])
    if change == "add":
        projection["fixtures"].append(projection_case(3))
    elif change == "remove":
        projection["fixtures"].pop()
    elif change == "home":
        projection["fixtures"][0]["fixture"]["home"] = "Different fixture team"
    else:
        projection["fixtures"][0]["fixture"]["kickoff"] = "2026-10-24T15:00:00Z"
    write(sandbox["projection_path"], projection)
    with pytest.raises(ValueError, match="fixtures differ|identity differs"):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])


@pytest.mark.parametrize("change,message", [
    ("duplicate", "Duplicate fixture"), ("future", "Future statistical"),
    ("started", "as_of must precede"), ("wrong_track_id", "fixture_id"),
])
def test_bad_projection_is_rejected_at_prepare(sandbox, change, message):
    projection = deepcopy(sandbox["projection"])
    if change == "duplicate":
        projection["fixtures"].append(deepcopy(projection["fixtures"][0]))
    elif change == "future":
        projection["generated_at"] = "2026-10-23T18:00:01Z"
    elif change == "started":
        projection["fixtures"][0]["fixture"]["kickoff"] = "2026-10-23T17:59:59Z"
    else:
        projection["fixtures"][0]["track"]["fixture_id"] = "different-fixture"
    write(sandbox["projection_path"], projection)
    with pytest.raises(ValueError, match=message):
        batch.prepare(sandbox["root"], sandbox["projection_path"], sandbox["news_path"])


def test_assembly_cannot_be_repeated_to_overwrite_an_existing_saved_batch(sandbox):
    result, payload, directory = assembled(sandbox)
    original = Path(result["case"]).read_bytes()
    with pytest.raises(FileExistsError):
        batch.assemble(sandbox["root"], directory, sandbox["market_path"])
    assert Path(result["case"]).read_bytes() == original


def test_prepare_assemble_export_preserve_old_single_case_and_all_inputs(sandbox):
    paths = [sandbox["old_single"], sandbox["projection_path"], sandbox["news_path"], sandbox["market_path"]]
    before = {path: path.read_bytes() for path in paths}
    result, payload, directory = assembled(sandbox)
    response_bytes = {directory / f"{mode}_response.json": (directory / f"{mode}_response.json").read_bytes()
                      for mode in ("independent", "model_informed")}
    batch.export(sandbox["root"], result["case"])
    for path, content in before.items() | response_bytes.items():
        assert path.read_bytes() == content


def test_export_replays_without_fit_network_or_refreshed_prediction_times(sandbox, monkeypatch):
    result, payload, directory = assembled(sandbox)
    source_bytes = Path(result["case"]).read_bytes()

    def prohibited(*args, **kwargs):
        pytest.fail("Batch export must not train or call a network/LLM service")

    monkeypatch.setattr(DateTeamGoalModel, "fit", prohibited)
    monkeypatch.setattr(socket, "create_connection", prohibited)
    monkeypatch.setattr(urllib.request, "urlopen", prohibited)
    sandbox["clock"][0] += timedelta(seconds=1)
    first = batch.export(sandbox["root"], result["case"])
    sandbox["clock"][0] += timedelta(minutes=1)
    second = batch.export(sandbox["root"], result["case"])
    assert first["output_directory"] != second["output_directory"]
    assert Path(result["case"]).read_bytes() == source_bytes
    for exported in (first, second):
        output = Path(exported["output_directory"])
        assert read(output / "五线赛前记录及原始回答.json") == payload
        assert read(output / "球队完整比分分布与来源.json") == sandbox["projection"]
        metadata = read(output / "导出信息.json")
        assert metadata["fit_count"] == metadata["llm_calls"] == 0
        assert metadata["prediction_times_refreshed"] is False
        assert datetime.fromisoformat(metadata["exported_at"]) > datetime.fromisoformat(payload["records"][0]["as_of"])


def test_pending_report_has_no_settled_performance_claims_and_uses_saved_scores(sandbox):
    result, payload, directory = assembled(sandbox)
    exported = batch.export(sandbox["root"], result["case"])
    output = Path(exported["output_directory"])
    scores = read(output / "同场评分与覆盖.json")
    report = Path(exported["report"]).read_text(encoding="utf-8")
    assert report == batch.render_report(payload, sandbox["projection"], scores)
    assert "真实比赛2场，已结算0场" in report
    assert "尚未录入可结算的真实赛果，没有准确率" in report
    assert "五线完整且已结算0场" in report
    assert "单场是否命中不直接触发" in report and "更晚真实比赛" in report
    assert "50:50权重在这两批LLM回答前" in report
    assert "部分列表页没有书商event ID，保留未知" in report
    assert "不能从LLM三类概率补造" in report
    assert "10-24 22:00" in report
    assert "50.0/30.0/20.0" in report and "40.0/35.0/25.0" in report
    assert scores["pending_fixture_ids"] == [row["fixture"]["match_id"] for row in payload["records"]]


def test_synthetic_settlement_scores_same_fixtures_and_report_does_not_remain_pending(sandbox):
    result, payload, directory = assembled(sandbox)
    fid = payload["records"][0]["fixture"]["match_id"]
    payload["results"][fid] = {"fixture_id": fid, "home_goals": 2, "away_goals": 1,
                               "finished_at": "2026-10-24T16:00:00Z", "available_at": "2026-10-24T16:01:00Z",
                               "settlement": "regulation_including_stoppage"}
    synthetic_case = write(sandbox["root"] / "synthetic-settlement-not-real.json", payload)
    sandbox["clock"][0] = datetime(2026, 10, 24, 16, 2, tzinfo=timezone.utc)
    exported = batch.export(sandbox["root"], synthetic_case)
    scores = read(Path(exported["output_directory"]) / "同场评分与覆盖.json")
    report = Path(exported["report"]).read_text(encoding="utf-8")
    assert exported["records"] == 2 and exported["settled"] == 1
    assert scores["all_five_common"]["n"] == 1
    assert len(scores["per_fixture"]) == 1
    for comparison in scores["comparisons"].values():
        assert comparison["n"] == 1 and comparison["fixture_ids"] == [fid]
    for arm, track in payload["records"][0]["tracks"].items():
        assert scores["all_five_common"]["metrics"][arm]["mean_log_loss"] == pytest.approx(-math.log(track["probabilities"][0]))
    assert "真实比赛2场，已结算1场" in report and "五线完整且已结算1场" in report
    assert "尚未录入可结算的真实赛果" not in report
    assert read(result["case"])["results"] == {}


@pytest.mark.parametrize("action,required", [("prepare", "--projection and --news"),
                                               ("assemble", "--run-dir and --market"), ("export", "--case")])
def test_cli_missing_required_inputs_exits_without_running(action, required, sandbox, capsys):
    with pytest.raises(SystemExit) as error:
        batch.main([action, "--root", str(sandbox["root"])])
    assert error.value.code == 2 and required in capsys.readouterr().err


def test_cli_prepare_assemble_export_report_counts_saved_observations(sandbox, capsys):
    assert batch.main(["prepare", "--root", str(sandbox["root"]), "--projection", str(sandbox["projection_path"]),
                       "--news", str(sandbox["news_path"])]) == 0
    prepared_result = json.loads(capsys.readouterr().out)
    directory = Path(prepared_result["run_directory"])
    for mode in ("independent", "model_informed"):
        write(directory / f"{mode}_response.json", {"forecasts": [response(case, mode) for case in sandbox["projection"]["fixtures"]]})
    sandbox["clock"][0] += timedelta(minutes=2)
    assert batch.main(["assemble", "--root", str(sandbox["root"]), "--run-dir", str(directory),
                       "--market", str(sandbox["market_path"])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["records"] == 2 and result["settled"] == 0
    assert batch.main(["export", "--root", str(sandbox["root"]), "--case", result["case"]]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["available"] == dict.fromkeys(TRACKS, 2)
    assert exported["records"] == 2 and exported["settled"] == 0
