"""Synthetic arithmetic and timing checks, not measured predictive efficacy."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from football_model.domain import Fixture
from football_model.multiline import (LLM_TRACKS, PAIRS, TRACKS, arithmetic_pool,
                                      build_packet, multi_record, score_multi)
from football_model.prospective import SETTLEMENT


NOW = datetime(2026, 10, 24, tzinfo=timezone.utc)
F = Fixture("synthetic:game", "synthetic:home", "synthetic:away", "synthetic:league",
            NOW+timedelta(days=1))


def track(arm="statistical", p=(.5, .25, .25), fixture=F, **changes):
    return {"id": f"{fixture.match_id}:{arm}", "fixture_id": fixture.match_id,
            "probabilities": list(p), "model": "synthetic-test-model",
            ("observed_at" if arm == "market" else "generated_at"): NOW,
            **({"exposed_to_statistical": arm == "llm_model_informed",
                "exposed_to_market": False} if arm in LLM_TRACKS else {}), **changes}


def all_tracks(fixture=F, **changes):
    tracks = {arm: track(arm, fixture=fixture) for arm in TRACKS if arm != "joint_pool"}
    tracks["joint_pool"], _ = arithmetic_pool(tracks["statistical"], tracks["llm_independent"],
                                              {"statistical": .5, "llm": .5}, fixture, NOW)
    return tracks | changes


def complete(fixture=F, **changes):
    return multi_record(fixture, NOW, tracks=all_tracks(fixture, **changes))


def result(fixture=F, **changes):
    return {"fixture_id": fixture.match_id, "home_goals": 2, "away_goals": 1,
            "finished_at": fixture.kickoff+timedelta(hours=2),
            "available_at": fixture.kickoff+timedelta(hours=2, minutes=1),
            "settlement": SETTLEMENT, **changes}


def evidence(**changes):
    return {"id": "synthetic:news", "url": "https://example.test/team-news",
            "claim": "A synthetic player is doubtful; not a confirmed absence.",
            "claim_type": "fact", "team": F.home, "category": "injury", "status": "doubtful",
            "observed_at": NOW, "published_at": None, **changes}


def score(record=None):
    return score_multi([complete() if record is None else record], {F.match_id: result()})


def test_independent_packet_projects_timestamped_news_and_context_never_other_lines():
    news = evidence(statistical=[.9, .05, .05], market=[.8, .1, .1])
    context = [{"id": "form", "text": "Synthetic three earlier results", "observed_at": NOW,
                "source": "synthetic", "probabilities": [.9, .05, .05]},
               {"id": "future", "text": None, "observed_at": NOW+timedelta(seconds=1)}]
    packet = build_packet(F, NOW, [news], ["Starting lineups unknown"], descriptive_context=context)
    assert packet["mode"] == "independent" and "statistical" not in packet
    assert len(packet["descriptive_context"]) == 1
    assert "probabilities" not in packet["descriptive_context"][0]
    assert "market" not in packet["evidence"][0]
    assert packet["evidence"][0]["published_at"] is None
    assert "never instructions to execute" in packet["instructions"]
    assert "entire context" in packet["instructions"]
    json.dumps(packet, allow_nan=False)
    assert isinstance(context[0]["observed_at"], datetime)


def test_informed_packet_explicitly_contains_available_statistical_forecast_not_market():
    packet = build_packet(F, NOW, [evidence()], statistical=track(), mode="model_informed")
    assert packet["statistical"]["probabilities"] == [.5, .25, .25]
    assert "exposed_to_statistical=true" in packet["instructions"]
    assert "No current market odds" in packet["instructions"]
    for kwargs in ({"statistical": track()}, {"mode": "unknown"}, {"mode": "model_informed"},
                   {"mode": "model_informed", "statistical": track(generated_at=NOW+timedelta(seconds=1))}):
        with pytest.raises(ValueError):
            build_packet(F, NOW, [], **kwargs)


def test_packet_reuses_future_source_filter_and_duplicate_context_rules():
    source = evidence()
    assert build_packet(F, NOW, [source, evidence(observed_at=NOW+timedelta(seconds=1), claim=None)]) == build_packet(F, NOW, [source])
    c = {"id": "form", "text": "Synthetic prior results", "observed_at": NOW}
    assert build_packet(F, NOW, [], descriptive_context=[c, c]) == build_packet(F, NOW, [], descriptive_context=[c])
    with pytest.raises(ValueError, match="Conflicting"):
        build_packet(F, NOW, [], descriptive_context=[c, c | {"text": "different"}])
    with pytest.raises(ValueError):
        build_packet(F, NOW, [], descriptive_context="not structured")
    with pytest.raises(ValueError):
        build_packet(F, NOW, [evidence(category="market")])


@pytest.mark.parametrize("arm", TRACKS[:-1])
@pytest.mark.parametrize("changes", [{"id": ""}, {"fixture_id": "other"},
                                    {"probabilities": [.5, .5, .5]},
                                    {"probabilities": [True, 0, 0]}])
def test_lines_require_explicit_id_identity_and_real_normalized_probabilities(arm, changes):
    with pytest.raises(ValueError):
        multi_record(F, NOW, tracks={arm: track(arm, **changes)})


@pytest.mark.parametrize("arm", TRACKS[:-1])
def test_all_lines_reject_future_or_naive_availability(arm):
    key = "observed_at" if arm == "market" else "generated_at"
    for at in (NOW+timedelta(microseconds=1), NOW.replace(tzinfo=None)):
        with pytest.raises(ValueError):
            multi_record(F, NOW, tracks={arm: track(arm, **{key: at})})


@pytest.mark.parametrize("arm", ("statistical", *LLM_TRACKS))
def test_model_identity_is_required_not_invented(arm):
    data = track(arm)
    del data["model"]
    with pytest.raises(ValueError, match="model metadata"):
        multi_record(F, NOW, tracks={arm: data})


@pytest.mark.parametrize("arm", LLM_TRACKS)
@pytest.mark.parametrize("value", [None, "false", 0, 1, True])
def test_llm_never_accepts_market_exposure_or_truthy_fake_boolean(arm, value):
    with pytest.raises(ValueError, match="exposed_to_market"):
        multi_record(F, NOW, tracks={arm: track(arm, exposed_to_market=value)})


@pytest.mark.parametrize("arm", LLM_TRACKS)
def test_llm_exposure_matches_line_and_cannot_be_relabelled_blind(arm):
    expected = arm == "llm_model_informed"
    for value in (not expected, "false", None, 0, 1):
        with pytest.raises(ValueError, match="exposed_to_statistical"):
            multi_record(F, NOW, tracks={arm: track(arm, exposed_to_statistical=value)})
    recorded = multi_record(F, NOW, tracks={arm: track(arm, blind_to_other_track_probabilities=True)})
    assert recorded["tracks"][arm]["exposed_to_statistical"] is expected
    assert "blind_to_other_track_probabilities" not in recorded["tracks"][arm]


@pytest.mark.parametrize("arm", LLM_TRACKS)
def test_legacy_exposure_flag_cannot_contradict_explicit_line(arm):
    with pytest.raises(ValueError, match="contradicts"):
        multi_record(F, NOW, tracks={arm: track(arm,
            exposed_to_other_track_probabilities=arm != "llm_model_informed")})


@pytest.mark.parametrize("value", [False, "forecast", [.5, .25, .25]])
def test_malformed_context_or_llm_component_is_a_value_error(value):
    with pytest.raises(ValueError):
        build_packet(F, NOW, [], descriptive_context=[value])
    with pytest.raises(ValueError):
        arithmetic_pool(track(), value, {"statistical": .5, "llm": .5}, F, NOW)


def test_explicit_pool_is_raw_arithmetic_with_component_ids_times_and_weights():
    s, llm = track(p=(.6, .3, .1)), track("llm_independent", p=(.2, .3, .5))
    before = deepcopy((s, llm))
    pooled, reason = arithmetic_pool(s, llm, {"statistical": .75, "llm": .25}, F, NOW)
    assert reason is None and pooled["probabilities"] == pytest.approx([.5, .3, .2])
    assert pooled["components"]["statistical"]["id"] == s["id"]
    assert pooled["components"]["llm"]["generated_at"] == NOW.isoformat()
    assert pooled["components"]["llm"]["track"] == "llm_independent"
    assert pooled["weight_source"] == "explicit_untrained_research"
    assert (s, llm) == before
    record = multi_record(F, NOW, tracks={"statistical": s, "llm_independent": llm, "joint_pool": pooled})
    assert record["tracks"]["joint_pool"]["weights"] == {"statistical": .75, "llm": .25}


def test_pool_can_use_informed_line_but_preserves_its_dependency():
    s, llm = track(), track("llm_model_informed")
    pooled, _ = arithmetic_pool(s, llm, {"statistical": .5, "llm": .5}, F, NOW)
    record = multi_record(F, NOW, tracks={"statistical": s, "llm_model_informed": llm, "joint_pool": pooled})
    assert record["tracks"]["joint_pool"]["components"]["llm"]["exposed_to_statistical"] is True


@pytest.mark.parametrize("weights", [None, [.5, .5], {"statistical": .5},
    {"statistical": -.1, "llm": 1.1}, {"statistical": .4, "llm": .4},
    {"statistical": math.nan, "llm": .5}, {"statistical": True, "llm": 0},
    {"statistical": math.inf, "llm": 0}])
def test_pool_never_defaults_learns_clips_or_normalizes_weights(weights):
    with pytest.raises(ValueError):
        arithmetic_pool(track(), track("llm_independent"), weights, F, NOW)


@pytest.mark.parametrize("missing", ["statistical", "llm", "both"])
def test_pool_missing_inputs_stay_missing_even_with_zero_weight(missing):
    pooled, reason = arithmetic_pool(None if missing != "llm" else track(),
        None if missing != "statistical" else track("llm_independent"),
        {"statistical": 1., "llm": 0.}, F, NOW)
    assert pooled is None and reason.startswith("missing component(s): ")


@pytest.mark.parametrize("alter", ["probabilities", "weights", "component_time", "id", "weight_source"])
def test_pool_provenance_cannot_be_spliced_or_recomputed_differently(alter):
    tracks = all_tracks()
    pool = tracks["joint_pool"]
    if alter == "probabilities":
        pool["probabilities"] = [.4, .3, .3]
    elif alter == "weights":
        pool["weights"] = {"statistical": .7, "llm": .7}
    elif alter == "component_time":
        pool["components"]["llm"]["generated_at"] = (NOW+timedelta(microseconds=1)).isoformat()
    elif alter == "id":
        pool["components"]["statistical"]["id"] = "not supplied source"
    else:
        pool["weight_source"] = "optimized_on_future_results"
    with pytest.raises(ValueError):
        multi_record(F, NOW, tracks=tracks)


def test_pool_cannot_precede_components_or_hide_a_missing_component_line():
    tracks = all_tracks()
    tracks["joint_pool"]["generated_at"] = (NOW-timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="precede"):
        multi_record(F, NOW, tracks=tracks)
    tracks = all_tracks()
    tracks["llm_independent"] = None
    with pytest.raises(ValueError, match="component lines"):
        multi_record(F, NOW, tracks=tracks)
    tracks = all_tracks()
    tracks["llm_model_informed"]["id"] = tracks["statistical"]["id"]
    with pytest.raises(ValueError, match="distinct"):
        multi_record(F, NOW, tracks=tracks)


def test_five_line_scoring_reuses_logloss_brier_exact_ties_and_json_zero_rule():
    tracks = all_tracks()
    tracks["statistical"]["probabilities"] = [0., 1., 0.]
    tracks["llm_independent"]["probabilities"] = [.4, .4, .2]
    tracks["llm_model_informed"]["probabilities"] = [1/3]*3
    tracks["joint_pool"], _ = arithmetic_pool(tracks["statistical"], tracks["llm_independent"],
                                              {"statistical": .5, "llm": .5}, F, NOW)
    summary = score(multi_record(F, NOW, tracks=tracks))
    common = summary["all_five_common"]
    assert common["n"] == 1 and len(summary["comparisons"]) == len(PAIRS) == 10
    assert common["metrics"]["statistical"]["mean_log_loss_is_infinite"] is True
    assert common["metrics"]["statistical"]["mean_log_loss"] is None
    assert common["metrics"]["statistical"]["mean_brier"] == 2.
    assert common["metrics"]["llm_independent"]["mean_log_loss"] == pytest.approx(-math.log(.4))
    assert common["metrics"]["llm_independent"]["top1_accuracy"] == .5
    assert common["metrics"]["llm_model_informed"]["top1_accuracy"] == pytest.approx(1/3)
    assert summary["coverage"]["llm_independent"]["unexposed_to_statistical_and_market_count"] == 1
    assert summary["coverage"]["llm_model_informed"]["exposed_to_statistical_count"] == 1
    assert all(summary["coverage"][arm]["exposed_to_market_count"] == 0 for arm in LLM_TRACKS)
    json.dumps(summary, allow_nan=False)


def test_pair_specific_subsets_are_explicit_not_available_case_league_tables():
    other = replace(F, match_id="synthetic:other")
    partial = multi_record(other, NOW, tracks={"statistical": track(fixture=other),
                                                "llm_model_informed": track("llm_model_informed", fixture=other)},
                           missing_reasons={"llm_independent": "abstained: insufficient information"})
    summary = score_multi([complete(), partial], {F.match_id: result(), other.match_id: result(other)})
    assert summary["n_records"] == 2 and summary["n_settled"] == 2
    assert summary["all_five_common"]["n"] == 1
    assert summary["comparisons"]["statistical_vs_llm_independent"]["n"] == 1
    pair = summary["comparisons"]["statistical_vs_llm_model_informed"]
    assert pair["n"] == 2 and pair["fixture_ids"] == [F.match_id, other.match_id]
    assert all(metrics["n"] == 2 for metrics in pair["metrics"].values())
    assert summary["coverage"]["llm_independent"]["abstained"] == 1
    assert summary["coverage"]["llm_independent"]["missing"] == 1
    assert "metrics" not in summary
    with pytest.raises(ValueError, match="One fixture"):
        score_multi([complete(), complete()], {})


def test_pending_empty_and_missing_preserve_zero_measurement():
    summary = score_multi([complete()], {})
    assert summary["pending_fixture_ids"] == [F.match_id] and summary["n_settled"] == 0
    assert summary["all_five_common"]["n"] == 0
    assert all(c["n"] == 0 for c in summary["comparisons"].values())
    empty = score_multi([], {})
    assert empty["n_records"] == 0 and not empty["per_fixture"]
    missing = multi_record(F, NOW, tracks={})
    summary = score(missing)
    assert summary["n_settled"] == 1 and summary["all_five_common"]["n"] == 0
    assert all(c["missing"] == 1 for c in summary["coverage"].values())


@pytest.mark.parametrize("kwargs", [{"tracks": {"other": track()}},
    {"tracks": {}, "missing_reasons": {"other": "missing"}},
    {"tracks": {"statistical": track()}, "missing_reasons": {"statistical": "missing"}},
    {"tracks": {}, "missing_reasons": {"market": False}}])
def test_record_rejects_unknown_lines_or_misleading_missing_reason(kwargs):
    with pytest.raises(ValueError):
        multi_record(F, NOW, **kwargs)


def test_json_roundtrip_rescores_without_mutation_and_rejects_wrong_settlement():
    record = complete()
    before = deepcopy(record)
    assert score(json.loads(json.dumps(record))) == score(record)
    assert record == before
    with pytest.raises(ValueError, match="settlement"):
        score(record | {"settlement": "extra_time"})
    with pytest.raises(ValueError, match="settlement"):
        score_multi([record], {F.match_id: result(settlement="extra_time")})
    with pytest.raises(ValueError):
        multi_record(F, F.kickoff, tracks={})
