"""Small deterministic tests; all quote values below are explicitly synthetic."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone, tzinfo
import json

import pytest

from football_model.domain import Fixture, OddsSnapshot
from football_model.odds_history import SourcedOddsSnapshot, select_asof


UTC = timezone.utc
AS_OF = datetime(2026, 10, 22, 12, tzinfo=UTC)
FIXTURE = Fixture("synthetic:m1", "synthetic:H", "synthetic:A", "synthetic:league",
                  AS_OF+timedelta(hours=6))


def quote(at, prices=(2., 3.5, 4.), *, bookmaker="synthetic-book", source="synthetic-source",
          fixture_id=FIXTURE.match_id, url="https://example.test/quote/1", event="source:m1"):
    return SourcedOddsSnapshot(OddsSnapshot(*prices, at, bookmaker, fixture_id), source, url, event)


def selected(quotes, **kwargs):
    return select_asof(quotes, fixture=FIXTURE, bookmaker="synthetic-book", source="synthetic-source",
                       as_of=kwargs.pop("as_of", AS_OF), **kwargs)


def test_selects_first_and_latest_available_quotes_and_known_arithmetic():
    first = quote(AS_OF-timedelta(days=1))
    latest = quote(AS_OF, (2.2, 3.2, 3.6), url="https://example.test/quote/2")
    result = selected([latest, first])
    assert result["coverage"] == {"status": "available", "n_distinct_observation_times": 2}
    assert result["first_observed"]["age_seconds"] == 86400
    assert result["latest_observed"]["age_seconds"] == 0
    assert result["first_observed"]["decimal_odds"] == {"home": 2., "draw": 3.5, "away": 4.}
    reciprocals = [1/2, 1/3.5, 1/4]
    assert result["first_observed"]["reciprocal_sum"] == pytest.approx(sum(reciprocals))
    assert result["first_observed"]["overround"] == pytest.approx(sum(reciprocals)-1)
    assert result["first_observed"]["proportional_de_vig"]["home"] == pytest.approx(reciprocals[0]/sum(reciprocals))
    delta = result["change_first_to_latest"]
    assert delta["elapsed_seconds"] == 86400
    assert delta["decimal_odds"] == pytest.approx({"home": .2, "draw": -.3, "away": -.4})
    for field in ("reciprocal_odds", "proportional_de_vig"):
        assert delta[field]["away"] == pytest.approx(result["latest_observed"][field]["away"]-
                                                     result["first_observed"][field]["away"])
    assert result["opening_status"] == "unknown"
    assert "not_verified_opening" in result["first_observed_label"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_future_records_including_future_conflicts_do_not_change_earlier_output():
    history = [quote(AS_OF-timedelta(hours=1))]
    result = selected(history)
    future = AS_OF+timedelta(seconds=1)
    history += [quote(future), quote(future, (5., 6., 7.), event="future-source-id")]
    assert selected(history) == result


def test_asof_endpoint_is_inclusive_and_one_microsecond_after_is_excluded():
    result = selected([quote(AS_OF), quote(AS_OF+timedelta(microseconds=1), (3., 4., 5.))])
    assert result["coverage"]["n_distinct_observation_times"] == 1
    assert result["latest_observed"]["decimal_odds"]["home"] == 2


@pytest.mark.parametrize("change", [
    {"bookmaker": "different-book"}, {"source": "different-source"}, {"fixture_id": "different-fixture"},
])
def test_other_streams_do_not_change_prices_or_coverage(change):
    point = quote(AS_OF-timedelta(hours=1))
    assert selected([point, quote(AS_OF, (7., 8., 9.), **change)]) == selected([point])


def test_input_order_and_identical_duplicates_are_idempotent():
    a, b = quote(AS_OF-timedelta(days=1)), quote(AS_OF, (2.5, 3., 3.))
    assert selected([a, b]) == selected([b, a, b, a])


def test_same_instant_in_different_timezones_is_one_observation():
    utc = quote(AS_OF)
    local = replace(utc, snapshot=replace(utc.snapshot,
        observed_at=AS_OF.astimezone(timezone(timedelta(hours=8)))))
    assert selected([utc, local]) == selected([utc])
    assert selected([utc], as_of=AS_OF.astimezone(timezone(timedelta(hours=-5)))) == selected([utc])


def test_repeated_local_hour_is_ordered_by_utc_instant_and_age_not_wall_clock():
    class RepeatedHour(tzinfo):
        # Synthetic fall-back zone, avoiding an external tzdata dependency.
        def utcoffset(self, dt):
            return timedelta(hours=1 if dt.fold else 2)

        def dst(self, dt):
            return timedelta(0)

    zone = RepeatedHour()
    cutoff = datetime(2026, 10, 25, 2, 30, tzinfo=zone, fold=1)  # 01:30 UTC
    first = datetime(2026, 10, 25, 2, 45, tzinfo=zone, fold=0)  # 00:45 UTC
    future = datetime(2026, 10, 25, 2, 40, tzinfo=zone, fold=1)  # 01:40 UTC
    fixture = replace(FIXTURE, kickoff=datetime(2026, 10, 25, 4, tzinfo=UTC))
    result = select_asof([quote(first), quote(future, (3., 4., 5.))], fixture=fixture,
                        bookmaker="synthetic-book", source="synthetic-source", as_of=cutoff)
    assert result["coverage"]["n_distinct_observation_times"] == 1
    assert result["latest_observed"]["age_seconds"] == 45*60
    assert result["latest_observed"]["observed_at"] == "2026-10-25T00:45:00+00:00"


def test_identical_price_observations_keep_all_distinct_evidence_links():
    a = quote(AS_OF, url="https://example.test/quote/b")
    b = quote(AS_OF, url="https://example.test/quote/a")
    result = selected([a, b])
    assert result == selected([b, a])
    evidence = result["latest_observed"]["source_records"]
    assert [row["source_url"] for row in evidence] == ["https://example.test/quote/a", "https://example.test/quote/b"]
    assert result["coverage"]["n_distinct_observation_times"] == 1


def test_same_instant_conflicting_prices_raise_in_both_input_orders():
    a, b = quote(AS_OF), quote(AS_OF, (2.01, 3.5, 4.))
    for history in ([a, b], [b, a]):
        with pytest.raises(ValueError, match="Conflicting prices"):
            selected(history)


def test_different_source_event_links_are_not_silently_joined():
    with pytest.raises(ValueError, match="source_event_id"):
        selected([quote(AS_OF-timedelta(hours=1)), quote(AS_OF, event="source:other")])


def test_no_available_history_is_explicit_and_does_not_impute_prices():
    for points in ([], [quote(AS_OF+timedelta(seconds=1))]):
        result = selected(points)
        assert result["coverage"] == {"status": "no_available_observation", "n_distinct_observation_times": 0}
        assert result["first_observed"] is result["latest_observed"] is result["change_first_to_latest"] is None
        assert result["opening_status"] == "unknown"


def test_one_observation_has_zero_observed_change_not_a_true_opening_claim():
    result = selected([quote(AS_OF-timedelta(days=100))])
    assert result["latest_observed"]["age_seconds"] == 100*86400
    assert result["first_observed"] == result["latest_observed"]
    change = result["change_first_to_latest"]
    assert change["elapsed_seconds"] == 0
    assert all(value == 0 for value in change["decimal_odds"].values())
    assert result["opening_status"] == "unknown"


@pytest.mark.parametrize("as_of", [AS_OF.replace(tzinfo=None), date(2026, 10, 22), "2026-10-22T12:00:00Z", None])
def test_explicit_aware_prediction_time_is_required(as_of):
    with pytest.raises(ValueError, match="timezone-aware"):
        selected([], as_of=as_of)


@pytest.mark.parametrize("as_of", [FIXTURE.kickoff, FIXTURE.kickoff+timedelta(seconds=1)])
def test_pre_match_selector_does_not_accept_at_or_after_kickoff(as_of):
    with pytest.raises(ValueError, match="precede kickoff"):
        selected([], as_of=as_of)


@pytest.mark.parametrize("field", ["source", "source_url", "source_event_id"])
@pytest.mark.parametrize("value", ["", " ", None])
def test_source_provenance_fields_must_be_explicit(field, value):
    with pytest.raises(ValueError, match=field):
        replace(quote(AS_OF), **{field: value})


def test_raw_csv_rows_and_plain_unsourced_snapshots_cannot_be_used_as_history():
    raw = {"Date": "22/10/2026", "AvgH": "2", "AvgD": "3.5", "AvgA": "4"}
    with pytest.raises(ValueError, match="OddsSnapshot"):
        SourcedOddsSnapshot(raw, "source", "https://example.test", "event")
    with pytest.raises(ValueError, match="SourcedOddsSnapshot"):
        selected([quote(AS_OF).snapshot])


@pytest.mark.parametrize("selector", [{"source": ""}, {"bookmaker": " "}, {"fixture": "m1"}])
def test_explicit_source_bookmaker_and_fixture_selectors_are_required(selector):
    arguments = {"fixture": FIXTURE, "source": "synthetic-source", "bookmaker": "synthetic-book", "as_of": AS_OF}
    arguments.update(selector)
    with pytest.raises(ValueError):
        select_asof([], **arguments)


def test_underround_is_reported_and_not_rejected_as_an_arbitrary_bookmaker_margin_rule():
    result = selected([quote(AS_OF, (4., 4., 4.))])
    point = result["latest_observed"]
    assert point["overround"] == -.25
    assert point["proportional_de_vig"] == pytest.approx({"home": 1/3, "draw": 1/3, "away": 1/3})
