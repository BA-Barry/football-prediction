"""Calendar-source research features must not become result or odds leakage."""

from copy import deepcopy
from datetime import date, datetime, timedelta
import json
import math
from types import SimpleNamespace

import pytest

import football_model.schedule_model as schedule
from football_model.date_model import DateTeamGoalModel
from football_model.schedule_model import (
    ACTIVITY_SCOPE, DateActivityIndex, DateScheduleGoalModel, MODEL_TYPE,
    OPPONENT_COUNT_FEATURE, SCHEDULE_FEATURE_DEFINITION, SELF_COUNT_FEATURE,
    source_date_features,
)
from football_model.team_model import team_features


def record(match_id, played_date, home="A", away="B", competition="LEAGUE", score=(2, 1),
           exclusion=None):
    return SimpleNamespace(match_id=match_id, match_date=played_date, home=home, away=away,
                           competition=competition, full_time=score,
                           goal_fit_exclusion=exclusion, markets={"odds": (2., 3., 4.)})


def history():
    start = date(2024, 8, 1)
    return [record(f"m-{i}", start + timedelta(days=i),
                   home="A" if i % 2 else "B", away="B" if i % 2 else "A",
                   score=(i % 4, (i + 1) % 3)) for i in range(40)]


class DateOnlyRecord:
    """Accessing the forbidden fields makes accidental read leakage visible."""

    def __init__(self, **metadata):
        self.__dict__.update(metadata)

    @property
    def full_time(self):
        raise AssertionError("Source date index must not read score fields")

    @property
    def markets(self):
        raise AssertionError("Source date index must not read market fields")


def date_only(source):
    return DateOnlyRecord(**{key: getattr(source, key) for key in (
        "match_id", "match_date", "home", "away", "competition", "goal_fit_exclusion")})


def test_window_left_boundary_included_but_all_same_day_and_future_excluded():
    query = date(2024, 8, 20)
    records = [record("outside", query - timedelta(days=15)),
               record("left", query - timedelta(days=14)),
               record("recent-z", query - timedelta(days=1)),
               record("recent-a", query - timedelta(days=1)),
               record("today-a", query), record("today-b", query),
               record("future", query + timedelta(days=1))]
    context = DateActivityIndex(list(reversed(records))).context("A", "B", "LEAGUE", query)

    assert context["scope"] == ACTIVITY_SCOPE
    assert context["on_date"] == query.isoformat()
    assert context["window_days"] == 14
    for side in ("home", "away"):
        assert context[side]["source_fixture_count_prev14d"] == 3
        assert context[side]["previous_14d_ids"] == ["left", "recent-a", "recent-z"]
        assert context[side]["previous_date"] == "2024-08-19"
        assert context[side]["gap_days"] == 1
        assert context[side]["window_left_censored"] is False


def test_source_namespace_and_competition_are_not_combined_or_aliased():
    query = date(2024, 8, 20)
    yesterday = query - timedelta(days=1)
    records = [record("same-source", yesterday, "provider:A", "provider:B"),
               record("other-league", yesterday, "provider:A", "provider:B", "OTHER"),
               record("other-source", yesterday, "alternative:A", "alternative:B")]
    context = DateActivityIndex(records).context("provider:A", "provider:B", "LEAGUE", query)
    assert context["home"]["previous_14d_ids"] == ["same-source"]
    assert context["away"]["previous_14d_ids"] == ["same-source"]
    unknown = DateActivityIndex(records).context("A", "B", "LEAGUE", query)
    assert unknown["home"]["source_fixture_count_prev14d"] == 0
    assert unknown["home"]["has_previous_source_record"] is False


def test_index_reads_only_metadata_and_marks_special_source_records():
    query = date(2024, 8, 20)
    normal = record("normal", query - timedelta(days=3))
    special = record("special", query - timedelta(days=1), exclusion="resumed_source_date")
    index = DateActivityIndex([date_only(normal), date_only(special)])
    context = index.context("A", "B", "LEAGUE", query)
    assert context["home"]["previous_14d_ids"] == ["normal", "special"]
    assert context["home"]["special_previous_14d_ids"] == ["special"]
    assert context["home"]["source_fixture_count_prev14d"] == 2
    # Index data are a metadata snapshot, not a reference to mutable source rows.
    normal.match_date = query
    assert index.context("A", "B", "LEAGUE", query) == context


def test_zero_observed_count_preserves_no_prior_record_and_left_coverage_metadata():
    first = date(2024, 8, 1)
    index = DateActivityIndex([record("old-A", first, "A", "C")])
    context = index.context("A", "B", "LEAGUE", date(2024, 9, 1))
    assert context["home"]["source_fixture_count_prev14d"] == 0
    assert context["away"]["source_fixture_count_prev14d"] == 0
    assert context["home"]["has_previous_source_record"] is True
    assert context["home"]["previous_date"] == first.isoformat()
    assert context["home"]["gap_days"] == 31
    assert context["away"]["has_previous_source_record"] is False
    assert context["away"]["previous_date"] is None
    assert context["away"]["gap_days"] is None
    assert context["home"]["window_left_censored"] is False
    assert context["away"]["window_left_censored"] is False
    early = index.context("A", "B", "LEAGUE", date(2024, 8, 5))
    assert early["home"]["window_left_censored"] is True
    assert early["away"]["window_left_censored"] is True
    # Equality at the left boundary provides a full 14-calendar-day source span.
    boundary = index.context("A", "B", "LEAGUE", first + timedelta(days=14))
    assert boundary["home"]["window_left_censored"] is False
    missing_competition = index.context("A", "B", "OTHER", date(2024, 9, 1))
    assert missing_competition["home"]["window_left_censored"] is True


def test_only_two_count_features_are_added_and_away_encoding_is_reversed():
    query = date(2024, 8, 20)
    records = [record("A-1", query - timedelta(days=1), "A", "C"),
               record("A-2", query - timedelta(days=2), "A", "C"),
               record("B-1", query - timedelta(days=3), "B", "C")]
    context = DateActivityIndex(records).context("A", "B", "LEAGUE", query)
    before = deepcopy(context)
    features = source_date_features("A", "B", "LEAGUE", context)
    original = team_features("A", "B", "LEAGUE")
    assert features == (
        {**original[0], SELF_COUNT_FEATURE: 2, OPPONENT_COUNT_FEATURE: 1},
        {**original[1], SELF_COUNT_FEATURE: 1, OPPONENT_COUNT_FEATURE: 2},
    )
    assert context == before
    assert set(features[0]) - set(original[0]) == {SELF_COUNT_FEATURE, OPPONENT_COUNT_FEATURE}


def test_bad_index_metadata_and_duplicate_source_identifiers_are_rejected():
    good = record("m", date(2024, 8, 1))
    for field, value, message in (("match_id", "", "identifiers"),
                                  ("match_date", datetime(2024, 8, 1), "calendar dates"),
                                  ("home", "B", "differ"),
                                  ("competition", "", "identifiers"),
                                  ("goal_fit_exclusion", True, "exclusion metadata")):
        malformed = SimpleNamespace(**{**vars(good), field: value})
        with pytest.raises(ValueError, match=message):
            DateActivityIndex([malformed])
    with pytest.raises(ValueError, match="Duplicate"):
        DateActivityIndex([good, good])
    with pytest.raises(ValueError, match="calendar date"):
        DateActivityIndex([good]).context("A", "B", "LEAGUE", datetime(2024, 8, 2))


def test_mismatched_context_and_malformed_counts_are_not_silently_accepted():
    query = date(2024, 8, 20)
    context = DateActivityIndex([]).context("A", "B", "LEAGUE", query)
    with pytest.raises(ValueError, match="competition"):
        source_date_features("A", "B", "OTHER", context)
    with pytest.raises(ValueError, match="teams and sides"):
        source_date_features("B", "A", "LEAGUE", context)
    for count in (True, -1, 0.5, float("nan")):
        invalid = deepcopy(context)
        invalid["home"]["source_fixture_count_prev14d"] = count
        with pytest.raises(ValueError, match="nonnegative integers"):
            source_date_features("A", "B", "LEAGUE", invalid)
    invalid = deepcopy(context)
    invalid["window_days"] = 7
    with pytest.raises(ValueError, match="scope or window"):
        source_date_features("A", "B", "LEAGUE", invalid)


def test_fit_uses_each_training_date_not_cutoff_context_and_original_decay(monkeypatch):
    records = history()
    cutoff = records[25].match_date
    captured = {}
    original_fit = schedule.fit_weighted_poisson

    def capture(features, target, weights, regularization):
        captured.update(features=deepcopy(features), target=list(target), weights=list(weights),
                        regularization=regularization)
        return original_fit(features, target, weights, regularization)

    monkeypatch.setattr(schedule, "fit_weighted_poisson", capture)
    model = DateScheduleGoalModel().fit(records, cutoff)
    index = DateActivityIndex(records)
    assert model.n_matches == 25
    assert model.reference_match_ids == [r.match_id for r in records[:25]]
    assert model.training_max_date == cutoff - timedelta(days=1)
    for i, source in enumerate(records[:25]):
        expected = source_date_features(source.home, source.away, source.competition,
                                        index.context(source.home, source.away, source.competition,
                                                      source.match_date))
        assert captured["features"][2 * i:2 * i + 2] == list(expected)
        assert captured["target"][2 * i:2 * i + 2] == list(source.full_time)
        weight = math.exp(-math.log(2) * (cutoff - source.match_date).days / 180)
        assert captured["weights"][2 * i:2 * i + 2] == [weight, weight]
    assert captured["features"][0][SELF_COUNT_FEATURE] == 0
    assert captured["features"][48][SELF_COUNT_FEATURE] == 14
    assert captured["regularization"] == .01


def test_same_day_and_future_score_and_odds_cannot_affect_training_features_or_predictions():
    records = history()
    cutoff = records[25].match_date
    changed = []
    for source in records:
        if source.match_date >= cutoff:
            changed.append(date_only(source))
        else:
            changed.append(SimpleNamespace(**{**vars(source), "markets": object()}))
    initial = deepcopy(records)
    normal = DateScheduleGoalModel().fit(records, cutoff)
    no_later_labels = DateScheduleGoalModel().fit(changed, cutoff)
    assert normal.parameters() == no_later_labels.parameters()
    for match_date in (cutoff, cutoff + timedelta(days=5), cutoff + timedelta(days=20)):
        assert normal.activity_index.context("A", "B", "LEAGUE", match_date) == (
            no_later_labels.activity_index.context("A", "B", "LEAGUE", match_date))
        assert normal.predict("A", "B", "LEAGUE", match_date) == (
            no_later_labels.predict("A", "B", "LEAGUE", match_date))
    assert records == initial


def test_later_date_records_cannot_change_earlier_training_even_when_their_dates_change():
    records = history()
    cutoff = records[25].match_date
    altered = [SimpleNamespace(**{**vars(source),
                                 "match_date": source.match_date + timedelta(days=200)})
               if source.match_date >= cutoff else source for source in records]
    before = DateScheduleGoalModel().fit(records, cutoff)
    after = DateScheduleGoalModel().fit(altered, cutoff)
    assert before.parameters() == after.parameters()
    assert before.predict("A", "B", "LEAGUE", cutoff) == after.predict("A", "B", "LEAGUE", cutoff)


def test_special_source_date_counts_but_label_is_never_read_or_fitted(monkeypatch):
    records = history()
    cutoff = records[25].match_date
    special = date_only(record("special", records[10].match_date, exclusion="resumed_date"))
    original_fit = schedule.fit_weighted_poisson
    captured = {}

    def capture(features, target, weights, regularization):
        captured["features"] = deepcopy(features)
        return original_fit(features, target, weights, regularization)

    monkeypatch.setattr(schedule, "fit_weighted_poisson", capture)
    model = DateScheduleGoalModel().fit(records + [special], cutoff)
    assert model.n_matches == 25
    assert "special" not in model.reference_match_ids
    # Same-day eligible m-10 excludes the special record, following day includes it.
    assert captured["features"][20][SELF_COUNT_FEATURE] == 10
    assert captured["features"][22][SELF_COUNT_FEATURE] == 12
    context = model.activity_index.context("A", "B", "LEAGUE", records[11].match_date)
    assert context["home"]["special_previous_14d_ids"] == ["special"]
    # The pre-existing original model's exclusion logic is not modified.
    plain = DateTeamGoalModel().fit(records, cutoff)
    with_special = DateTeamGoalModel().fit(records + [special], cutoff)
    assert plain.parameters() == with_special.parameters()


def test_unknown_team_in_date_index_still_requires_cutoff_eligible_goal_history():
    records = history()
    cutoff = records[25].match_date
    records += [record("new-after-cutoff", cutoff + timedelta(days=1), "NEW", "A"),
                record("new-special-only", cutoff - timedelta(days=3), "SPECIAL", "A",
                       exclusion="administrative_award")]
    model = DateScheduleGoalModel().fit(records, cutoff)
    query = cutoff + timedelta(days=2)
    assert model.activity_index.context("NEW", "A", "LEAGUE", query)["home"][
        "has_previous_source_record"] is True
    for unknown in ("NEW", "SPECIAL"):
        with pytest.raises(ValueError, match="Unseen teams"):
            model.predict(unknown, "A", "LEAGUE", query)
    assert "NEW" not in model.parameters()["teams"]
    assert "SPECIAL" not in model.parameters()["teams"]


def test_date_context_updates_within_block_without_coefficient_refit():
    records = history()
    cutoff = records[25].match_date
    model = DateScheduleGoalModel().fit(records, cutoff)
    # Explicit finite coefficients isolate the refresh semantics from learned effect size.
    model.coefficients[SELF_COUNT_FEATURE] = .2
    model.coefficients[OPPONENT_COUNT_FEATURE] = -.1
    parameters = deepcopy(model.parameters())
    early_date, late_date = cutoff, cutoff + timedelta(days=27)
    early = model.predict("A", "B", "LEAGUE", early_date)
    late = model.predict("A", "B", "LEAGUE", late_date)
    for match_date, rates in ((early_date, early), (late_date, late)):
        context = model.activity_index.context("A", "B", "LEAGUE", match_date)
        features = source_date_features("A", "B", "LEAGUE", context)
        expected = tuple(math.exp(parameters["intercept"] + sum(
            parameters["coefficients"].get(key, 0) * value for key, value in row.items()))
            for row in features)
        assert rates == expected
    assert early != late
    assert model.parameters() == parameters


def test_parameters_are_serializable_original_metadata_preserved_and_definition_not_mutable():
    records = history()
    cutoff = records[25].match_date
    model = DateScheduleGoalModel().fit(records, cutoff)
    parameters = json.loads(json.dumps(model.parameters(), allow_nan=False))
    assert parameters["model_type"] == MODEL_TYPE
    assert parameters["basis"] == "team_marginal"
    assert parameters["rho"] == 0
    assert parameters["regularization"] == .01
    assert parameters["half_life_days"] == 180
    assert parameters["before_date"] == cutoff.isoformat()
    assert parameters["availability_convention"] == (
        "strictly_earlier_calendar_date_results_assumed_available")
    assert parameters["reference_match_ids"] == [source.match_id for source in records[:25]]
    assert parameters["schedule_feature_definition"] == SCHEDULE_FEATURE_DEFINITION
    assert set(SCHEDULE_FEATURE_DEFINITION["feature_names"]) == {
        SELF_COUNT_FEATURE, OPPONENT_COUNT_FEATURE}
    assert all(math.isfinite(value) for value in parameters["coefficients"].values())
    parameters["schedule_feature_definition"]["feature_names"].append("not-a-feature")
    assert model.parameters()["schedule_feature_definition"] == SCHEDULE_FEATURE_DEFINITION
    assert "not-a-feature" not in SCHEDULE_FEATURE_DEFINITION["feature_names"]


def test_bad_training_scores_and_duplicate_identifiers_do_not_fit():
    records = history()
    cutoff = records[25].match_date
    for malformed_score in ((True, 1), (-1, 2), (1.5, 2), (1,), "2-1"):
        invalid = [SimpleNamespace(**{**vars(records[0]), "full_time": malformed_score}),
                   *records[1:]]
        with pytest.raises(ValueError, match="nonnegative integer goals"):
            DateScheduleGoalModel().fit(invalid, cutoff)
    with pytest.raises(ValueError, match="Duplicate"):
        DateScheduleGoalModel().fit(records + [records[0]], cutoff)


def test_original_unfitted_chronology_and_unknown_competition_checks_are_preserved():
    records = history()
    cutoff = records[25].match_date
    with pytest.raises(ValueError, match="No eligible"):
        DateScheduleGoalModel().fit(records, records[0].match_date)
    with pytest.raises(ValueError, match="calendar date"):
        DateScheduleGoalModel().fit(records, datetime(2024, 8, 26))
    fresh = DateScheduleGoalModel()
    with pytest.raises(ValueError, match="Fit earlier-date"):
        fresh.predict("A", "B", "LEAGUE", cutoff)
    with pytest.raises(ValueError, match="not been fitted"):
        fresh.parameters()
    model = fresh.fit(records, cutoff)
    with pytest.raises(ValueError, match="precede"):
        model.predict("A", "B", "LEAGUE", cutoff - timedelta(days=1))
    with pytest.raises(ValueError, match="cutoff date"):
        model.predict("A", "B", "LEAGUE", datetime(2024, 8, 26))
    with pytest.raises(ValueError, match="differ"):
        model.predict("A", "A", "LEAGUE", cutoff)
    with pytest.raises(ValueError, match="Unseen competition"):
        model.predict("A", "B", "OTHER", cutoff)


def test_source_count_signal_is_actually_trainable_not_just_written_to_metadata():
    records = []
    start = date(2020, 1, 20)
    for i in range(64):
        played_date = start + timedelta(days=20 * i)
        count_a, count_b = i % 4, (i // 4) % 4
        for team, count in (("A", count_a), ("B", count_b)):
            for j in range(count):
                records.append(date_only(record(f"activity-{i}-{team}-{j}",
                    played_date - timedelta(days=j + 1), team, "DATE_ONLY_OPPONENT",
                    exclusion="synthetic_date_only_activity")))
        goals_a, goals_b = (round(math.exp(.3 + .5 * count)) for count in (count_a, count_b))
        home, away = ("A", "B") if i % 2 else ("B", "A")
        score = (goals_a, goals_b) if home == "A" else (goals_b, goals_a)
        records.append(record(f"eligible-{i}", played_date, home, away, score=score))
    cutoff = start + timedelta(days=20 * 64)
    model = DateScheduleGoalModel().fit(records, cutoff)
    assert model.n_matches == 64
    assert model.coefficients[SELF_COUNT_FEATURE] > .3
    assert abs(model.coefficients[OPPONENT_COUNT_FEATURE]) < .1
    assert all(math.isfinite(value) and value > 0
               for value in model.predict("A", "B", "LEAGUE", cutoff))
    assert "DATE_ONLY_OPPONENT" not in model.teams
