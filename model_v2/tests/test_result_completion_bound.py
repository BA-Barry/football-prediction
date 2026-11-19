"""Synthetic exact-finish versus observed-completion-bound scoring checks.

These tests contain no real match result, source request or efficacy claim.
An observed upper bound is not an exact final-whistle timestamp.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from football_model.domain import Fixture
from football_model.multiline import (LLM_TRACKS, TRACKS, arithmetic_pool,
                                      multi_record, score_multi)
from football_model.prospective import (ARMS, SETTLEMENT, _settled,
                                        prediction_record, score_records)


AS_OF = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
F = Fixture("synthetic:completion-bound", "synthetic:home", "synthetic:away",
            "synthetic:competition", AS_OF + timedelta(hours=8))
EXACT_FINISH = F.kickoff + timedelta(hours=2)
FINISHED_BY = F.kickoff + timedelta(hours=3)
AVAILABLE = F.kickoff + timedelta(hours=4)


def result(*, bound=False, goals=(2, 1), **changes):
    row = {"fixture_id": F.match_id, "settlement": SETTLEMENT,
           "home_goals": goals[0], "away_goals": goals[1],
           "available_at": AVAILABLE,
           "finished_by" if bound else "finished_at": FINISHED_BY if bound else EXACT_FINISH}
    return row | changes


def record(kind, p=(.5, .25, .25)):
    if kind == "legacy":
        tracks = {arm: {"fixture_id": F.match_id, "probabilities": list(p),
                        "observed_at" if arm == "market" else "generated_at": AS_OF,
                        **({"exposed_to_other_track_probabilities": False} if arm == "llm" else {})}
                  for arm in ARMS}
        return prediction_record(F, AS_OF, **tracks)
    tracks = {arm: {"id": f"synthetic:{arm}", "fixture_id": F.match_id,
                    "probabilities": list(p), "model": "synthetic-model",
                    "observed_at" if arm == "market" else "generated_at": AS_OF,
                    **({"exposed_to_statistical": arm == "llm_model_informed",
                        "exposed_to_market": False} if arm in LLM_TRACKS else {})}
              for arm in TRACKS if arm != "joint_pool"}
    tracks["joint_pool"], missing = arithmetic_pool(
        tracks["statistical"], tracks["llm_independent"],
        {"statistical": .5, "llm": .5}, F, AS_OF)
    assert missing is None
    return multi_record(F, AS_OF, tracks=tracks)


def score(kind, truth, p=(.5, .25, .25)):
    scorer = score_records if kind == "legacy" else score_multi
    return scorer([record(kind, p)], {F.match_id: truth})


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("goals,outcome", [((2, 1), "home"), ((0, 0), "draw"), ((1, 3), "away")])
def test_exact_and_upper_bound_give_identical_probability_scores(kind, goals, outcome):
    exact = score(kind, result(goals=goals))
    bound = score(kind, result(bound=True, goals=goals))
    assert bound == exact
    assert bound["n_records"] == bound["n_settled"] == 1
    assert bound["pending_fixture_ids"] == []
    assert bound["per_fixture"][0]["outcome"] == outcome


@pytest.mark.parametrize("bound", [False, True])
def test_completion_types_distinguish_missing_null_and_present_without_mutation(bound):
    truth = result(bound=bound)
    absent_key = "finished_at" if bound else "finished_by"
    for explicit_null in (False, True):
        if explicit_null:
            truth[absent_key] = None
        original = deepcopy(truth)
        assert _settled(truth, F) == 0
        assert truth == original
        assert (absent_key in truth) is explicit_null


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_scoring_keeps_original_result_bound_and_prediction_input_unchanged(kind):
    truth = result(bound=True, source={"url": "https://example.test/synthetic-full-time",
                                      "meaning": "already complete when observed"})
    records = [record(kind)]
    results = {F.match_id: truth}
    before_records, before_results = deepcopy(records), deepcopy(results)
    scorer = score_records if kind == "legacy" else score_multi
    scores = scorer(records, results)
    assert records == before_records
    assert results == before_results
    assert "finished_at" not in results[F.match_id]
    assert results[F.match_id]["finished_by"] == FINISHED_BY
    assert scores["n_settled"] == 1
    # The time representation must not enter or alter the class-probability score.
    row = scores["per_fixture"][0]["scores"]["statistical"]
    assert row["log_loss"] == pytest.approx(-math.log(.5))
    assert row["brier"] == pytest.approx(.375)
    assert row["top1_accuracy"] == 1.


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("changes", [
    {"finished_at": EXACT_FINISH, "finished_by": FINISHED_BY},
    {"finished_at": None, "finished_by": None},
    {"finished_at": None},
])
def test_exactly_one_nonnull_completion_representation_is_required(kind, changes):
    truth = result(**changes)
    with pytest.raises(ValueError):
        score(kind, truth)


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_neither_completion_key_is_rejected(kind):
    truth = result()
    truth.pop("finished_at")
    with pytest.raises(ValueError):
        score(kind, truth)


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("bad_bound", [
    FINISHED_BY.replace(tzinfo=None),
    "2026-01-02T21:00:00",
    "not a datetime",
    "",
    123,
    True,
    F.kickoff,
    F.kickoff - timedelta(seconds=1),
    AVAILABLE + timedelta(microseconds=1),
])
def test_invalid_naive_or_impossible_completion_bounds_are_rejected(kind, bad_bound):
    with pytest.raises(ValueError):
        score(kind, result(bound=True, finished_by=bad_bound))


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("available", [None, "", True, AVAILABLE.replace(tzinfo=None),
                                        FINISHED_BY - timedelta(microseconds=1)])
def test_availability_must_be_aware_and_no_earlier_than_bound(kind, available):
    with pytest.raises(ValueError):
        score(kind, result(bound=True, available_at=available))


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_result_observation_can_be_both_completion_bound_and_availability_instant(kind):
    truth = result(bound=True, available_at=FINISHED_BY)
    assert score(kind, truth)["n_settled"] == 1


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_equivalent_timezone_offsets_represent_same_bound_and_available_time(kind):
    east = timezone(timedelta(hours=8))
    actual = result(bound=True, finished_by=FINISHED_BY.astimezone(east).isoformat(),
                    available_at=AVAILABLE.astimezone(east).isoformat())
    assert score(kind, actual) == score(kind, result(bound=True))


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("goals", [(-1, 0), (0, -1), (True, 0), (0, False),
                                   (1., 0), (0, 1.), (None, 0), (0, "1")])
def test_completion_bound_does_not_relax_regulation_goal_types(kind, goals):
    with pytest.raises(ValueError):
        score(kind, result(bound=True, goals=goals))


@pytest.mark.parametrize("kind", ["legacy", "multi"])
@pytest.mark.parametrize("changes", [
    {"fixture_id": "synthetic:different-fixture"},
    {"fixture_id": None},
    {"settlement": "extra_time"},
    {"settlement": "penalty_shootout"},
    {"settlement": None},
])
def test_completion_bound_does_not_relax_fixture_or_settlement_identity(kind, changes):
    with pytest.raises(ValueError):
        score(kind, result(bound=True, **changes))


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_zero_probability_truth_remains_infinite_loss_with_completion_bound(kind):
    exact = score(kind, result(goals=(0, 0)), p=(.5, 0., .5))
    actual = score(kind, result(bound=True, goals=(0, 0)), p=(.5, 0., .5))
    assert actual == exact
    row = actual["per_fixture"][0]["scores"]["statistical"]
    assert row["log_loss"] is None
    assert row["log_loss_infinite"] is True
    assert row["brier"] == 1.5
    assert row["top1_accuracy"] == 0.


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_exact_top_probability_tie_rule_is_unchanged_with_completion_bound(kind):
    actual = score(kind, result(bound=True, goals=(0, 0)), p=(.5, .5, 0.))
    assert actual == score(kind, result(goals=(0, 0)), p=(.5, .5, 0.))
    row = actual["per_fixture"][0]["scores"]["statistical"]
    assert row["top1_accuracy"] == .5
    assert row["top1_tied"] is True
    assert row["brier"] == .5


@pytest.mark.parametrize("kind", ["legacy", "multi"])
def test_unsettled_fixture_is_not_manufactured_from_a_timestamp_or_record(kind):
    scorer = score_records if kind == "legacy" else score_multi
    actual = scorer([record(kind)], {})
    assert actual["n_settled"] == 0
    assert actual["pending_fixture_ids"] == [F.match_id]
    assert actual["per_fixture"] == []
