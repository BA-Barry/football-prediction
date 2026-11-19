"""Date-only retrospective experiments must not imply precise availability."""

from datetime import date, datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pytest

from football_model.date_model import DateTeamGoalModel


def history():
    start = date(2024, 8, 1)
    return [
        SimpleNamespace(
            match_id=f"league-{i}",
            match_date=start + timedelta(days=i),
            home="A" if i % 2 else "B",
            away="B" if i % 2 else "A",
            competition="TEST_LEAGUE",
            full_time=(i % 4, (i + 1) % 3),
            goal_fit_exclusion=None,
        )
        for i in range(40)
    ]


def test_date_fit_excludes_cutoff_day_and_future_results():
    records = history()
    cutoff = records[25].match_date
    model = DateTeamGoalModel().fit(records, cutoff)

    assert model.n_matches == 25
    assert model.reference_match_ids == [r.match_id for r in records[:25]]
    assert model.training_max_date == cutoff - timedelta(days=1)
    assert all(r.match_date < cutoff for r in records if r.match_id in model.reference_match_ids)


def test_cutoff_and_later_score_changes_cannot_change_earlier_prediction():
    records = history()
    cutoff = records[25].match_date
    earlier_only = DateTeamGoalModel().fit(records[:25], cutoff)
    changed = [
        SimpleNamespace(**{**vars(r), "full_time": (19, 17)})
        if r.match_date >= cutoff else r
        for r in records
    ]
    with_later_results = DateTeamGoalModel().fit(changed, cutoff)

    expected = earlier_only.predict("A", "B", "TEST_LEAGUE", cutoff)
    actual = with_later_results.predict("A", "B", "TEST_LEAGUE", cutoff)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert all(math.isfinite(rate) and rate > 0 for rate in actual)


def test_administrative_result_is_not_used_for_goal_fitting():
    records = history()
    cutoff = records[25].match_date
    administrative = SimpleNamespace(
        match_id="administrative-result",
        match_date=cutoff - timedelta(days=1),
        home="A",
        away="B",
        competition="TEST_LEAGUE",
        full_time=(0, 2),
        goal_fit_exclusion="administrative_award_not_on_field_score",
    )
    clean = DateTeamGoalModel().fit(records, cutoff)
    with_award = DateTeamGoalModel().fit(records + [administrative], cutoff)

    assert with_award.n_matches == clean.n_matches
    assert administrative.match_id not in with_award.reference_match_ids
    assert with_award.predict("A", "B", "TEST_LEAGUE", cutoff) == pytest.approx(
        clean.predict("A", "B", "TEST_LEAGUE", cutoff), rel=1e-12, abs=1e-12
    )


def test_duplicate_eligible_match_identifier_is_rejected():
    records = history()
    with pytest.raises(ValueError, match="Duplicate"):
        DateTeamGoalModel().fit(records + [records[0]], records[25].match_date)


def test_unknown_team_or_competition_and_past_prediction_are_rejected():
    records = history()
    cutoff = records[25].match_date
    model = DateTeamGoalModel().fit(records, cutoff)

    with pytest.raises(ValueError, match="Unseen teams"):
        model.predict("UNKNOWN", "A", "TEST_LEAGUE", cutoff)
    with pytest.raises(ValueError, match="Unseen competition"):
        model.predict("A", "B", "OTHER_LEAGUE", cutoff)
    with pytest.raises(ValueError, match="precede"):
        model.predict("A", "B", "TEST_LEAGUE", cutoff - timedelta(days=1))
    with pytest.raises(ValueError, match="differ"):
        model.predict("A", "A", "TEST_LEAGUE", cutoff)


@pytest.mark.parametrize("cutoff", [datetime(2024, 8, 26), datetime(2024, 8, 26, tzinfo=timezone.utc)])
def test_datetime_is_not_silently_used_as_a_calendar_date(cutoff):
    records = history()
    with pytest.raises(ValueError, match="calendar date"):
        DateTeamGoalModel().fit(records, cutoff)

    model = DateTeamGoalModel().fit(records, date(2024, 8, 26))
    with pytest.raises(ValueError, match="cutoff date"):
        model.predict("A", "B", "TEST_LEAGUE", cutoff)


def test_model_parameters_are_json_serializable_and_preserve_basis():
    records = history()
    cutoff = records[25].match_date
    model = DateTeamGoalModel().fit(records, cutoff)
    parameters = json.loads(json.dumps(model.parameters(), allow_nan=False))

    assert parameters["basis"] == "team_marginal"
    assert parameters["rho"] == 0
    assert parameters["before_date"] == cutoff.isoformat()
    assert parameters["n_matches"] == 25
    assert parameters["availability_convention"] == (
        "strictly_earlier_calendar_date_results_assumed_available"
    )
    assert parameters["reference_match_ids"] == model.reference_match_ids
    assert all(math.isfinite(value) for value in parameters["coefficients"].values())


def test_empty_eligible_history_and_unfitted_access_are_rejected():
    records = history()
    with pytest.raises(ValueError, match="No eligible"):
        DateTeamGoalModel().fit(records, records[0].match_date)
    model = DateTeamGoalModel()
    with pytest.raises(ValueError, match="Fit earlier-date"):
        model.predict("A", "B", "TEST_LEAGUE", records[0].match_date)
    with pytest.raises(ValueError, match="not been fitted"):
        model.parameters()
