"""Select a sourced 1X2 quote history at an explicit pre-match information time.

Only complete OddsSnapshot objects are accepted. ``observed_at`` must represent
an actual capture/information-availability time supplied by the caller, never a
historical CSV's download time or a fabricated opening/T-24h time. This module
cannot authenticate those claims. It provides no CSV converter, fixture-name
matching, missing-price imputation, automatic market fusion or stale-age rule.

One source/bookmaker/fixture stream is summarized at a time. Its earliest
available observation is not asserted to be the bookmaker's true opening.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .domain import Fixture, OddsSnapshot, aware
from .probability import implied


_OUTCOMES = ("home", "draw", "away")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty string")
    return value


def _time(value, label):
    if not isinstance(value, datetime):
        raise ValueError(f"{label} must be an explicit timezone-aware datetime")
    return aware(value)


@dataclass(frozen=True)
class SourcedOddsSnapshot:
    """A real observation plus its explicit source-to-fixture linkage.

    The source must describe the same regulation-time full-time 1X2 market as
    the supplied OddsSnapshot. Source event IDs are not inferred from labels;
    the caller is responsible for correct home/away and fixture resolution.
    A source URL is evidence metadata, not proof that a timestamp is genuine.
    """

    snapshot: OddsSnapshot
    source: str
    source_url: str
    source_event_id: str

    def __post_init__(self):
        if not isinstance(self.snapshot, OddsSnapshot):
            raise ValueError("Supply an OddsSnapshot, not a raw or date-only CSV row")
        _time(self.snapshot.observed_at, "observed_at")
        _text(self.snapshot.fixture_id, "fixture_id")
        _text(self.snapshot.bookmaker, "bookmaker")
        for field in ("source", "source_url", "source_event_id"):
            _text(getattr(self, field), field)


def _point(snapshot, source_records, as_of):
    raw = [float(getattr(snapshot, outcome)) for outcome in _OUTCOMES]
    reciprocals = [1/price for price in raw]
    total = sum(reciprocals)
    return {
        "observed_at": snapshot.observed_at.astimezone(timezone.utc).isoformat(),
        "age_seconds": (as_of.astimezone(timezone.utc)-
                        snapshot.observed_at.astimezone(timezone.utc)).total_seconds(),
        "decimal_odds": dict(zip(_OUTCOMES, raw)),
        "reciprocal_odds": dict(zip(_OUTCOMES, reciprocals)),
        "reciprocal_sum": total,
        "overround": total-1,
        "proportional_de_vig": dict(zip(_OUTCOMES, map(float, implied(snapshot)))),
        "source_records": [{"source": source, "source_url": url, "source_event_id": event}
                           for source, url, event in sorted(source_records)],
    }


def _change(first, latest):
    changes = {
        field: {outcome: latest[field][outcome]-first[field][outcome] for outcome in _OUTCOMES}
        for field in ("decimal_odds", "reciprocal_odds", "proportional_de_vig")
    }
    changes.update({
        "elapsed_seconds": (datetime.fromisoformat(latest["observed_at"])-
                            datetime.fromisoformat(first["observed_at"])).total_seconds(),
        "reciprocal_sum": latest["reciprocal_sum"]-first["reciprocal_sum"],
        "overround": latest["overround"]-first["overround"],
        "direction": "latest_observed_minus_first_observed",
        "probability_change_unit": "probability_not_percentage_points",
    })
    return changes


def select_asof(
    quotes: Iterable[SourcedOddsSnapshot], *, fixture: Fixture,
    bookmaker: str, source: str, as_of: datetime,
) -> dict:
    """Summarize exactly one known history using only observations <= as_of.

    Other fixtures, bookmakers and sources are excluded, not averaged. Future
    observations are ignored before duplicate/conflict resolution, so appending
    future data cannot change an earlier information set. Identical observations
    are idempotent; same-instant different prices in the selected stream raise
    ValueError rather than relying on input order. Available source event IDs
    must agree, as a source-to-fixture remapping needs an explicit decision.

    Missing history returns an explicit empty result. One observation returns a
    zero first-to-latest change, not a claim that no earlier movement occurred.
    No result updates a team model or chooses a market-fusion weight.
    """
    if not isinstance(fixture, Fixture):
        raise ValueError("Supply an explicitly resolved Fixture")
    _text(fixture.match_id, "fixture.match_id")
    _text(bookmaker, "bookmaker")
    _text(source, "source")
    _time(as_of, "as_of")
    _time(fixture.kickoff, "fixture.kickoff")
    cutoff = as_of.astimezone(timezone.utc)
    if cutoff >= fixture.kickoff.astimezone(timezone.utc):
        raise ValueError("as_of must precede kickoff for this pre-match history")

    selected = {}
    event_ids = set()
    for quote in quotes:
        if not isinstance(quote, SourcedOddsSnapshot):
            raise ValueError("Every quote must be a SourcedOddsSnapshot containing an OddsSnapshot")
        snapshot = quote.snapshot
        instant = snapshot.observed_at.astimezone(timezone.utc)
        if (snapshot.fixture_id != fixture.match_id or snapshot.bookmaker != bookmaker
                or quote.source != source or instant > cutoff):
            continue
        prices = tuple(getattr(snapshot, outcome) for outcome in _OUTCOMES)
        evidence = (quote.source, quote.source_url, quote.source_event_id)
        if instant in selected:
            previous, records = selected[instant]
            if prices != tuple(getattr(previous, outcome) for outcome in _OUTCOMES):
                raise ValueError("Conflicting prices at the same observed_at in one source/bookmaker/fixture stream")
            records.add(evidence)
        else:
            selected[instant] = (snapshot, {evidence})
        event_ids.add(quote.source_event_id)
    if len(event_ids) > 1:
        raise ValueError("Selected source history maps different source_event_id values to one fixture")

    result = {
        "fixture": {"match_id": fixture.match_id, "home": fixture.home, "away": fixture.away,
                    "competition": fixture.competition,
                    "kickoff": fixture.kickoff.astimezone(timezone.utc).isoformat()},
        "as_of": cutoff.isoformat(),
        "source": source, "bookmaker": bookmaker,
        "market": "regulation_full_time_one_x_two",
        "coverage": {"status": "available" if selected else "no_available_observation",
                     "n_distinct_observation_times": len(selected)},
        "first_observed": None, "latest_observed": None,
        "change_first_to_latest": None,
        "opening_status": "unknown",
        "first_observed_label": "earliest_available_observation_not_verified_opening",
        "notes": [
            "observed_at is caller-supplied actual capture/availability time; provenance is not independently authenticated",
            "no historical CSV download time is converted into a historical quote time",
            "proportional_de_vig is a normalization of reciprocal odds, not a uniquely identified true probability",
            "price changes are descriptive, not causal evidence of lineups, injuries or motivation",
            "quote age is reported without an arbitrary maximum-age cutoff; availability does not imply executability",
        ],
    }
    if selected:
        first, last = min(selected), max(selected)
        result["first_observed"] = _point(*selected[first], as_of)
        result["latest_observed"] = _point(*selected[last], as_of)
        result["change_first_to_latest"] = _change(result["first_observed"], result["latest_observed"])
    return result
