from datetime import date, datetime, timezone
import json
from math import nan, inf

import pytest

from football_model.prematch_minutes import (
    MINUTES_BASIS,
    METHODS,
    TeamMinutesObservation,
    estimate_minutes,
)


def observation(mid="first", day=date(2016, 1, 1), minutes=None, **changes):
    return TeamMinutesObservation(
        match_id=mid, match_date=day, minutes={"a": 90.0, "b": 30.0} if minutes is None else minutes,
        **{"team_id": 1, "source": "statsbomb/exact/fixed45", **changes},
    )


def estimate(history, **changes):
    return estimate_minutes(history, **{
        "team_id": 1, "source": "statsbomb/exact/fixed45",
        "prediction_date": date(2016, 2, 1), **changes,
    })


def test_last_match_uses_only_latest_earlier_team_source_not_input_order():
    early = observation("early", date(2016, 1, 1), {"old": 90})
    late = observation("late", date(2016, 1, 31), {"new": 70, "old": 20})
    result = estimate([late, early])
    assert result["predicted_minutes"] == {"new": 70, "old": 20}
    assert result["history_match_ids"] == ["late"]
    assert result["history_dates"] == ["2016-01-31"]
    assert result["history_matches_available"] == 2
    assert result["history_matches_used"] == result["history_matches_requested"] == 1
    assert result["method"] == "last_match" and result["status"] == "estimated"


@pytest.mark.parametrize("method", METHODS)
def test_same_day_and_later_exposure_never_changes_candidate(method):
    earlier = observation("old", date(2016, 1, 31), {"old-player": 90})
    same_day = observation("target", date(2016, 2, 1), {"target-only-player": 90})
    later = observation("future", date(2016, 2, 2), {"future-only-player": 90})
    result = estimate([same_day, later, earlier], method=method)
    original = estimate([earlier], method=method)
    assert result["predicted_minutes"] == original["predicted_minutes"] == {"old-player": 90}
    assert result["history_match_ids"] == ["old"]
    assert result["excluded_records"]["same_or_later_date"] == 2
    assert "target-only-player" not in result["predicted_minutes"]
    assert "future-only-player" not in result["predicted_minutes"]


def test_recent_mean_divides_by_team_matches_not_player_appearances():
    history = [observation("first", date(2016, 1, 1), {"a": 90, "unused-bench": 0}),
               observation("second", date(2016, 1, 2), {"b": 90})]
    result = estimate(history, method="recent_mean", lookback=5)
    assert result["predicted_minutes"] == {"a": 45, "b": 45, "unused-bench": 0}
    assert result["history_matches_used"] == 2 and result["history_matches_requested"] == 5
    assert result["history_match_ids"] == ["first", "second"]
    assert result["history_team_totals"] == [90.0, 90.0]
    assert result["total_predicted_minutes"] == 90
    assert result["current_roster_known"] is False


def test_recent_mean_is_latest_five_team_matches_not_full_season_union():
    history = [observation(str(n), date(2016, 1, n), {f"p{n}": 90}) for n in range(1, 8)]
    result = estimate(history, method="recent_mean", lookback=5)
    assert result["history_match_ids"] == [str(n) for n in range(3, 8)]
    assert result["predicted_minutes"] == {f"p{n}": 18 for n in range(3, 8)}
    assert result["history_matches_available"] == 7
    assert "p1" not in result["predicted_minutes"]


def test_source_and_team_must_match_without_cross_provider_identity_merging():
    first = observation("first", date(2016, 1, 1), {"correct": 90})
    other_team = observation("other-team", date(2016, 1, 2), {"wrong-team": 90}, team_id=2)
    other_source = observation("other-source", date(2016, 1, 3), {"wrong-source": 90}, source="wyscout/early/fixed45")
    text_id = observation("text-id", date(2016, 1, 4), {"text-team": 90}, team_id="1")
    result = estimate([first, other_team, other_source, text_id], method="recent_mean")
    assert result["predicted_minutes"] == {"correct": 90}
    assert result["excluded_records"]["other_source_or_team"] == 3
    assert result["source"] == "statsbomb/exact/fixed45"


@pytest.mark.parametrize("method", METHODS)
def test_no_earlier_history_is_unknown_not_zero_or_a_fabricated_lineup(method):
    result = estimate([observation(day=date(2016, 2, 1))], method=method)
    assert result["status"] == "no_eligible_history"
    assert result["predicted_minutes"] == {}
    assert result["history_matches_used"] == result["history_matches_available"] == 0
    assert result["total_predicted_minutes"] is None
    assert result["unallocated_relative_to_990"] is None
    assert result["history_team_totals"] == []
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("method", METHODS)
def test_dismissal_shortfall_is_retained_not_rescaled_or_repaired(method):
    minutes = {f"p{n}": 90 for n in range(1, 11)} | {"dismissed": 30}
    result = estimate([observation(minutes=minutes)], method=method)
    assert result["total_predicted_minutes"] == 930
    assert result["unallocated_relative_to_990"] == 60
    assert result["full_team_reference_minutes"] == 990
    assert result["predicted_minutes"] == minutes
    assert all(value <= 90 for value in result["predicted_minutes"].values())


def test_complete_historical_matches_preserve_990_without_target_roster():
    first = {f"p{n}": 90 for n in range(11)}
    second = {f"q{n}": 90 for n in range(11)}
    result = estimate([observation("first", date(2016, 1, 1), first),
                       observation("second", date(2016, 1, 2), second)], method="recent_mean")
    assert len(result["predicted_minutes"]) == 22
    assert result["total_predicted_minutes"] == 990
    assert result["unallocated_relative_to_990"] == 0
    assert result["current_roster_known"] is False
    assert result["historical_publication_verified"] is False
    assert "first publication time unknown" in result["availability_semantics"]
    assert "transfers, injuries" in result["interpretation"]


def test_changing_input_or_output_mapping_does_not_rewrite_recorded_history():
    original = {"a": 60, "b": 30}
    record = observation(minutes=original)
    original["a"] = 90
    assert record.minutes["a"] == 60
    result = estimate([record])
    result["predicted_minutes"]["a"] = 0
    assert estimate([record])["predicted_minutes"]["a"] == 60
    with pytest.raises(TypeError):
        record.minutes["a"] = 0


@pytest.mark.parametrize("method", METHODS)
def test_duplicate_identity_is_not_double_weighted(method):
    record = observation()
    with pytest.raises(ValueError, match="Duplicate"):
        estimate([record, record], method=method)


@pytest.mark.parametrize("method", METHODS)
def test_date_only_same_team_history_tie_is_not_resolved_with_arbitrary_id(method):
    with pytest.raises(ValueError, match="Same-date"):
        estimate([observation("a"), observation("b")], method=method)


def test_duplicate_future_rows_cannot_influence_earlier_candidate():
    record = observation("future", date(2016, 2, 2))
    result = estimate([observation(), record, record])
    assert result["history_matches_used"] == 1
    assert result["excluded_records"]["same_or_later_date"] == 2


@pytest.mark.parametrize("value", ["2016-01-01", None, datetime(2016, 1, 1), datetime(2016, 1, 1, tzinfo=timezone.utc)])
def test_observation_rejects_missing_or_timestamp_calendar_date(value):
    with pytest.raises(ValueError, match="calendar date"):
        observation(day=value)


@pytest.mark.parametrize("value", ["2016-02-01", None, datetime(2016, 2, 1), datetime(2016, 2, 1, tzinfo=timezone.utc)])
def test_prediction_date_does_not_silently_drop_timestamp_semantics(value):
    with pytest.raises(ValueError, match="calendar date"):
        estimate([observation()], prediction_date=value)


@pytest.mark.parametrize("value", [True, False, None, "90", nan, inf, -inf, -0.01, 90.01])
def test_invalid_or_stoppage_inclusive_player_minutes_are_not_clipped(value):
    with pytest.raises(ValueError, match="finite in"):
        observation(minutes={"a": value})


@pytest.mark.parametrize("minutes", [{}, [], None, {"a": 0}, {"a": 0, "b": 0}])
def test_empty_or_zero_exposure_is_not_a_complete_team_record(minutes):
    with pytest.raises(ValueError, match="complete team|Complete team"):
        TeamMinutesObservation("m", 1, date(2016, 1, 1), minutes, "source")


@pytest.mark.parametrize("minutes", [{"": 90}, {" ": 90}, {1: 90}, {None: 90}])
def test_player_ids_are_not_silently_merged_or_coerced(minutes):
    with pytest.raises(ValueError, match="identities"):
        observation(minutes=minutes)


def test_over_990_team_minutes_are_not_rescaled():
    with pytest.raises(ValueError, match="no more than 990"):
        observation(minutes={str(n): 90 for n in range(12)})


@pytest.mark.parametrize("value", [None, "", " ", 0, -1, True, False, 1.5])
def test_team_identity_must_be_explicit(value):
    with pytest.raises(ValueError, match="team_id"):
        observation(team_id=value)
    with pytest.raises(ValueError, match="team_id"):
        estimate([], team_id=value)


@pytest.mark.parametrize("value", [None, "", " ", 7])
def test_source_identity_must_be_explicit(value):
    with pytest.raises(ValueError, match="source/provider/scenario"):
        observation(source=value)
    with pytest.raises(ValueError, match="source/provider/scenario"):
        estimate([], source=value)


@pytest.mark.parametrize("value", [None, "", " ", 7])
def test_match_identity_must_be_explicit(value):
    with pytest.raises(ValueError, match="match_id"):
        observation(mid=value)


@pytest.mark.parametrize("value", ["full_match", "including_stoppage", "90min", "", None])
def test_minutes_basis_cannot_silently_mix_full_match_or_stoppage(value):
    with pytest.raises(ValueError, match=MINUTES_BASIS):
        observation(basis=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, None, "5"])
def test_lookback_is_an_explicit_positive_match_count(value):
    with pytest.raises(ValueError, match="lookback"):
        estimate([], lookback=value)


@pytest.mark.parametrize("value", ["unknown", "last-five", None, 1])
def test_method_is_not_automatically_selected_or_replaced(value):
    with pytest.raises(ValueError, match="method"):
        estimate([], method=value)


def test_untyped_input_is_not_reinterpreted_as_prediction_features():
    with pytest.raises(ValueError, match="TeamMinutesObservation"):
        estimate([{"match_id": "target", "minutes": {"new": 90}}])


def test_iterable_history_is_supported_and_output_is_finite_json():
    history = (observation(str(n), date(2016, 1, n), {"a": 90 / n}) for n in range(1, 4))
    result = estimate(history, method="recent_mean")
    assert result["predicted_minutes"]["a"] == pytest.approx((90 + 45 + 30) / 3)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
