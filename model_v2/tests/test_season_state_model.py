from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pytest

from football_model.date_model import DateTeamGoalModel
from football_model.season_state_model import SeasonStateTeamGoalModel, season_for_date
from football_model.team_model import team_features


def match(mid, day, home="A", away="B", goals=(2, 1), season=None, excluded=None):
    return SimpleNamespace(match_id=mid, match_date=day, home=home, away=away,
                           competition="TEST", full_time=goals,
                           season=season_for_date(day) if season is None else season,
                           goal_fit_exclusion=excluded)


def history(years=(2021, 2022, 2023)):
    pairings = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rows = []
    for offset, year in enumerate(years):
        for i in range(12):
            home, away = pairings[i % len(pairings)]
            if i % 2:
                home, away = away, home
            rows.append(match(f"{year}-{i}", date(year, 8, 1)+timedelta(days=i*4), home, away,
                              ((i+offset) % 4, (2*i+offset) % 3)))
    return rows


def fitted(q=.5, years=(2021, 2022, 2023)):
    return SeasonStateTeamGoalModel(innovation_scale=q).fit(history(years), date(max(years)+1, 7, 1))


def test_q_zero_has_identical_base_features_coefficients_and_predictions():
    records, cutoff = history(), date(2024, 7, 1)
    original = DateTeamGoalModel(.01, 180).fit(records, cutoff)
    model = SeasonStateTeamGoalModel(.01, 180, 0).fit(records, cutoff)
    assert model.coefficients == original.coefficients
    assert model.intercept == original.intercept
    assert model.optimizer_iterations == original.optimizer_iterations
    assert model.reference_match_ids == original.reference_match_ids
    assert not any(k.startswith("innovation_") for k in model.coefficients)
    assert model._features("A", "B", "TEST", "2025-26") == team_features("A", "B", "TEST")
    for day in (cutoff, date(2025, 8, 1), date(2027, 6, 30)):
        assert model.predict("A", "B", "TEST", day) == original.predict("A", "B", "TEST", day)


def test_first_season_is_team_specific_and_not_a_global_year():
    rows = [match("first", date(2021, 8, 1)), match("new", date(2023, 8, 1), "C", "D")]
    model = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows, date(2023, 9, 1))
    assert model.first_season == {"A": "2021-22", "B": "2021-22", "C": "2023-24", "D": "2023-24"}
    assert not any(k.startswith("innovation_") for k in model.coefficients)
    rows = model._features("A", "C", "TEST", "2023-24")
    assert rows[0]["innovation_attack:A:2022-23"] == .5
    assert rows[0]["innovation_attack:A:2023-24"] == .5
    assert not any(k.startswith("innovation_defense:C:") for k in rows[0])


def test_gap_year_columns_retained_collinear_and_defense_sign_correct():
    model = fitted(years=(2021, 2023))
    rows = model._features("A", "B", "TEST", "2023-24")
    for year in ("2022-23", "2023-24"):
        assert rows[0][f"innovation_attack:A:{year}"] == .5
        assert rows[0][f"innovation_defense:B:{year}"] == -.5
        assert rows[1][f"innovation_attack:B:{year}"] == .5
        assert rows[1][f"innovation_defense:A:{year}"] == -.5
        assert f"innovation_attack:A:{year}" in model.coefficients
        assert f"innovation_defense:B:{year}" in model.coefficients
    # Equal columns are retained; ridge resolves this parameterization, not
    # separate empirical evidence about a year when no team fixture was seen.
    assert model.coefficients["innovation_attack:A:2022-23"] == pytest.approx(
        model.coefficients["innovation_attack:A:2023-24"], abs=1e-12)


def test_known_historical_innovations_carry_to_future_seasons_without_repeat():
    model = fitted(q=.5, years=(2021, 2023))
    model.coefficients = dict.fromkeys(model.coefficients, 0.)
    model.intercept = .1
    model.coefficients.update({"attack:A": .2, "defense:B": .3,
                               "innovation_attack:A:2022-23": .4,
                               "innovation_attack:A:2023-24": .6,
                               "innovation_defense:B:2022-23": .2,
                               "innovation_defense:B:2023-24": .1})
    # .1 + .2 - .3 + .5*(.4+.6) - .5*(.2+.1) = .35.
    expected = math.exp(.35)
    before = deepcopy(model.parameters())
    for day in (date(2024, 7, 1), date(2025, 8, 1), date(2028, 8, 1)):
        assert model.predict("A", "B", "TEST", day)[0] == pytest.approx(expected, abs=1e-14)
    assert model.parameters() == before
    assert "innovation_attack:A:2024-25" not in model.coefficients


def test_later_current_season_history_can_change_state_not_only_carry_forward():
    old_rows = history((2021, 2022))
    prior = SeasonStateTeamGoalModel(innovation_scale=1).fit(old_rows, date(2023, 8, 1))
    new_rows = [match(f"new-{i}", date(2023, 8, 2)+timedelta(days=i), goals=(5, 0)) for i in range(12)]
    current = SeasonStateTeamGoalModel(innovation_scale=1).fit(old_rows+new_rows, date(2023, 9, 1))
    assert "innovation_attack:A:2023-24" not in prior.coefficients
    assert "innovation_attack:A:2023-24" in current.coefficients
    assert abs(current.coefficients["innovation_attack:A:2023-24"]) > .01
    assert current.predict("A", "B", "TEST", date(2023, 9, 1)) != pytest.approx(
        prior.predict("A", "B", "TEST", date(2023, 9, 1)), rel=.01)


def test_cutoff_day_future_scores_and_new_teams_never_enter_history_or_first_season():
    rows, cutoff = history(), date(2023, 8, 10)
    prefix = [r for r in rows if r.match_date < cutoff]
    same_day = match("same-day", cutoff, "FUTURE", "A", (40, 0))
    future = match("future", date(2025, 8, 1), "LATER", "B", (0, 40))
    a = SeasonStateTeamGoalModel(innovation_scale=.5).fit(prefix, cutoff)
    b = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows+[same_day, future], cutoff)
    assert a.parameters() == b.parameters()
    assert "FUTURE" not in b.first_season and "LATER" not in b.first_season
    assert all(r.match_date < cutoff for r in rows if r.match_id in b.reference_match_ids)
    # The prediction method takes no score or future availability field.
    assert b.predict("A", "B", "TEST", cutoff) == a.predict("A", "B", "TEST", cutoff)


def test_excluded_scores_and_their_season_do_not_enter_the_fit():
    rows, cutoff = history(), date(2024, 7, 1)
    excluded = match("award", date(2023, 12, 1), goals=(0, 30), season="malformed", excluded="awarded")
    a = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows, cutoff)
    b = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows+[excluded], cutoff)
    assert a.parameters() == b.parameters()


@pytest.mark.parametrize("day,label", [(date(2023, 6, 30), "2022-23"), (date(2023, 7, 1), "2023-24"),
                                       (date(2024, 2, 29), "2023-24")])
def test_season_boundary_july_first_inclusive(day, label):
    assert season_for_date(day) == label
    model = SeasonStateTeamGoalModel(innovation_scale=1).fit([match("a", day)], day+timedelta(days=1))
    assert model.first_season == {"A": label, "B": label}


@pytest.mark.parametrize("label", ["2023/24", "2023-2024", "23-24", "2023-25", "0000-01", "2022-23", 2023, None])
def test_bad_season_format_or_date_disagreement_rejected(label):
    record = match("bad", date(2023, 8, 1))
    record.season = label
    with pytest.raises(ValueError, match="Season|season"):
        SeasonStateTeamGoalModel(innovation_scale=.5).fit([record], date(2023, 9, 1))


def test_duplicate_ids_empty_history_and_bad_historical_goals():
    rows = history()
    with pytest.raises(ValueError, match="Duplicate"):
        SeasonStateTeamGoalModel().fit(rows+[rows[0]], date(2024, 7, 1))
    with pytest.raises(ValueError, match="No eligible"):
        SeasonStateTeamGoalModel().fit(rows, rows[0].match_date)
    for score in ((-1, 0), (1.5, 0), (True, 0), (1,)):
        with pytest.raises(ValueError, match="full-time goals"):
            SeasonStateTeamGoalModel().fit([match("a", date(2023, 8, 1), goals=score)], date(2023, 9, 1))


@pytest.mark.parametrize("q", [-.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_innovation_scale(q):
    with pytest.raises(ValueError, match="innovation_scale"):
        SeasonStateTeamGoalModel(innovation_scale=q)


@pytest.mark.parametrize("args", [(-.01, 180), (0, 180), (.01, 0), (.01, -180), (True, 180), (.01, float("nan"))])
def test_invalid_original_numeric_settings(args):
    with pytest.raises(ValueError):
        SeasonStateTeamGoalModel(*args)


def test_unknown_team_competition_past_date_equal_sides_and_unfitted_errors():
    model = fitted()
    day = model.before_date
    with pytest.raises(ValueError, match="Unseen teams"):
        model.predict("NEW", "A", "TEST", day)
    with pytest.raises(ValueError, match="Unseen competition"):
        model.predict("A", "B", "OTHER", day)
    with pytest.raises(ValueError, match="precede"):
        model.predict("A", "B", "TEST", day-timedelta(days=1))
    with pytest.raises(ValueError, match="differ"):
        model.predict("A", "A", "TEST", day)
    with pytest.raises(ValueError, match="Fit earlier-date"):
        SeasonStateTeamGoalModel().predict("A", "B", "TEST", day)
    with pytest.raises(ValueError, match="not been fitted"):
        SeasonStateTeamGoalModel().parameters()


@pytest.mark.parametrize("day", [datetime(2024, 7, 1), datetime(2024, 7, 1, tzinfo=timezone.utc)])
def test_datetime_not_accepted_as_calendar_date(day):
    with pytest.raises(ValueError, match="calendar date"):
        SeasonStateTeamGoalModel().fit(history(), day)
    with pytest.raises(ValueError, match="cutoff date"):
        fitted().predict("A", "B", "TEST", day)


@pytest.mark.parametrize("q", [0., .5, 1.])
@pytest.mark.parametrize("years", [(2023,), (2021, 2022, 2023), (2021, 2023)])
def test_save_restore_exact_algebra_and_metadata_without_fitting(q, years, monkeypatch):
    model = fitted(q, years)
    payload = json.loads(json.dumps(model.parameters(), allow_nan=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("Restoring must not fit")

    monkeypatch.setattr("football_model.season_state_model.fit_weighted_poisson", forbidden)
    restored = SeasonStateTeamGoalModel.from_parameters(payload)
    assert restored.parameters() == payload
    for day in (model.before_date, date(2026, 8, 1), date(2029, 6, 30)):
        assert restored.predict("A", "D", "TEST", day) == model.predict("A", "D", "TEST", day)
    payload["coefficients"]["attack:A"] += 100
    payload["reference_match_ids"].append("outside")
    payload["first_season"]["A"] = "2000-01"
    assert restored.parameters() == model.parameters()


@pytest.mark.parametrize("field,value", [
    ("innovation_scale", 2), ("coefficients", {}), ("feature_names", []),
    ("first_season", {"A": "2021-22"}), ("last_season", {}),
    ("training_max_date", "2024-07-01"), ("before_date", "2021-01-01"),
    ("n_matches", 1), ("optimizer_iterations", -1), ("reference_match_ids", ["dup", "dup"]),
    ("intercept", float("nan")), ("rho", .1), ("basis", "lineup"),
    ("model_type", "other"), ("teams", ["A", "A"]),
])
def test_saved_malformed_parameters_rejected(field, value):
    payload = fitted().parameters()
    payload[field] = value
    with pytest.raises(ValueError):
        SeasonStateTeamGoalModel.from_parameters(payload)


def test_saved_missing_coefficient_repeated_feature_name_and_invalid_season_bounds_rejected():
    payload = fitted().parameters()
    del payload["coefficients"]["attack:A"]
    with pytest.raises(ValueError, match="feature names"):
        SeasonStateTeamGoalModel.from_parameters(payload)
    payload = fitted().parameters()
    payload["feature_names"].append(payload["feature_names"][0])
    with pytest.raises(ValueError, match="feature names"):
        SeasonStateTeamGoalModel.from_parameters(payload)
    payload = fitted().parameters()
    payload["last_season"]["A"] = "2020-21"
    with pytest.raises(ValueError, match="season bounds"):
        SeasonStateTeamGoalModel.from_parameters(payload)


def test_input_order_and_parameters_calls_do_not_mutate_training_records_or_model():
    rows = history()
    before = deepcopy([vars(r) for r in rows])
    a = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows, date(2024, 7, 1))
    b = SeasonStateTeamGoalModel(innovation_scale=.5).fit(rows[::-1], date(2024, 7, 1))
    assert a.parameters() == b.parameters()
    assert before == [vars(r) for r in rows]
    payload = a.parameters()
    payload["coefficients"].clear()
    payload["first_season"].clear()
    payload["reference_match_ids"].clear()
    assert a.parameters() == b.parameters()
