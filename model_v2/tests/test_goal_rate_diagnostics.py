"""Untruncated conditional moments and readable-parameter reconstruction."""

from copy import deepcopy
from datetime import date, timedelta
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import iv
from scipy.stats import poisson

from football_model.date_model import DateTeamGoalModel
import football_model.goal_rate_diagnostics as diagnostics


def row(key="one", score=(2, 1)):
    return {
        "match_id": key, "date": "2024-08-01", "competition": "E0",
        "home": "A", "away": "B", "actual_score": list(score),
        "fit_before_date": "2024-08-01", "training_max_date": "2024-07-31",
        "training_matches": 25,
        # Deliberately not lambda. The helper must never read this approximation.
        "forecast": {"expected_goals": {"home": 100, "away": 200}},
    }


def model():
    return {
        "model_type": "date_level_regularized_poisson", "basis": "team_marginal",
        "rho": 0, "regularization": .01, "half_life_days": 180,
        "before_date": "2024-08-01", "training_min_date": "2024-06-01",
        "training_max_date": "2024-07-31", "n_matches": 25,
        "teams": ["football-data:E0:A", "football-data:E0:B"], "competitions": ["E0"],
        "intercept": .12,
        "coefficients": {
            "attack:football-data:E0:A": .17, "attack:football-data:E0:B": .08,
            "defense:football-data:E0:A": .04, "defense:football-data:E0:B": .13,
            "competition:E0": -.03, "home:E0": .22,
        },
        "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
    }


def test_reconstructs_exp_linear_features_with_defense_sign_and_home_effect():
    parameters, prediction = model(), row()
    before_parameters, before_prediction = deepcopy(parameters), deepcopy(prediction)
    assert diagnostics.rates_from_parameters(parameters, prediction) == pytest.approx(
        (math.exp(.12 + .17 - .13 - .03 + .22), math.exp(.12 + .08 - .04 - .03)))
    prediction["forecast"]["expected_goals"] = {"home": float("nan"), "away": 0}
    assert diagnostics.rates_from_parameters(parameters, prediction) == pytest.approx(
        diagnostics.rates_from_parameters(before_parameters, before_prediction))
    assert parameters == before_parameters


def test_parameter_roundtrip_matches_real_fitted_date_model():
    start, cutoff = date(2024, 6, 1), date(2024, 8, 1)
    records = [SimpleNamespace(
        match_id=f"historical-{index}", match_date=start + timedelta(days=index),
        home="football-data:E0:A" if index % 2 else "football-data:E0:B",
        away="football-data:E0:B" if index % 2 else "football-data:E0:A",
        competition="E0", full_time=(index % 4, (index + 1) % 3), goal_fit_exclusion=None,
    ) for index in range(40)]
    fitted = DateTeamGoalModel().fit(records, cutoff)
    saved = json.loads(json.dumps(fitted.parameters(), allow_nan=False))
    prediction = row()
    prediction.update(training_matches=40, training_max_date=saved["training_max_date"])
    actual = diagnostics.rates_from_parameters(saved, prediction)
    assert actual == pytest.approx(fitted.predict("football-data:E0:A", "football-data:E0:B",
                                                 "E0", cutoff), rel=1e-14, abs=1e-14)


@pytest.mark.parametrize(("key", "value"), [
    ("basis", "conditional_reference"), ("rho", .1), ("rho", False),
    ("regularization", .1), ("half_life_days", 90),
    ("model_type", "some_other_poisson"), ("availability_convention", "known_at_kickoff"),
    ("training_max_date", "2024-08-01"), ("training_min_date", "2024-08-05"),
    ("intercept", float("inf")), ("intercept", 1000), ("intercept", -1000),
    ("n_matches", True), ("competitions", ["D1"]),
    ("teams", ["football-data:E0:A"]), ("before_date", "20240801"),
])
def test_reconstruction_rejects_altered_basis_configuration_and_chronology(key, value):
    parameters = model()
    parameters[key] = value
    with pytest.raises(ValueError):
        diagnostics.rates_from_parameters(parameters, row())


@pytest.mark.parametrize(("key", "value"), [
    ("date", "2024-07-31"), ("date", "2024-08-01T00:00:00"),
    ("date", date(2024, 8, 1)), ("fit_before_date", "2024-08-02"),
    ("training_max_date", "2024-07-30"), ("training_matches", 26),
    ("training_matches", 25.), ("home", "B"), ("away", "C"),
    ("competition", "D1"),
])
def test_prediction_must_match_saved_fit_metadata(key, value):
    prediction = row()
    prediction[key] = value
    with pytest.raises(ValueError):
        diagnostics.rates_from_parameters(model(), prediction)


def test_missing_coefficient_is_not_implicitly_a_zero_player_or_team_prior():
    parameters = model()
    del parameters["coefficients"]["attack:football-data:E0:B"]
    with pytest.raises(ValueError, match="reconstruct"):
        diagnostics.rates_from_parameters(parameters, row())
    parameters = model()
    parameters["coefficients"]["attack:football-data:E0:B"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostics.rates_from_parameters(parameters, row())


def test_row_statistics_analytic_event_probabilities_and_nll():
    lh, la, h, a = 1.7, .8, 2, 1
    statistics = diagnostics.row_statistics(row(score=(h, a)), (lh, la))
    assert tuple(statistics) == diagnostics.STAT_FIELDS
    assert statistics["home_rate"] == lh
    assert statistics["away_rate"] == la
    assert statistics["home_goals"] == h
    assert statistics["away_goals"] == a
    assert statistics["home_residual"] == pytest.approx(h - lh)
    assert statistics["away_residual"] == pytest.approx(a - la)
    assert statistics["home_pearson_square_minus_one"] == pytest.approx((h - lh) ** 2 / lh - 1)
    assert statistics["away_pearson_square_minus_one"] == pytest.approx((a - la) ** 2 / la - 1)
    assert statistics["pearson_cross_product"] == pytest.approx((h - lh) * (a - la) / math.sqrt(lh * la))
    probabilities = {
        "home_zero": math.exp(-lh), "away_zero": math.exp(-la),
        "total_four_plus": 1 - math.exp(-lh - la) * sum((lh + la) ** k / math.factorial(k) for k in range(4)),
        "draw": math.exp(-lh - la) * iv(0, 2 * math.sqrt(lh * la)),
        "zero_zero": math.exp(-lh - la), "one_one": lh * la * math.exp(-lh - la),
    }
    for name, probability in probabilities.items():
        assert statistics[f"{name}_probability"] == pytest.approx(probability, rel=1e-13)
        assert statistics[f"{name}_residual"] == pytest.approx(-probability)
    hnll, anll = -math.log(poisson.pmf(h, lh)), -math.log(poisson.pmf(a, la))
    assert statistics["home_poisson_nll"] == pytest.approx(hnll)
    assert statistics["away_poisson_nll"] == pytest.approx(anll)
    assert statistics["joint_score_nll"] == pytest.approx(hnll + anll)
    assert all(math.isfinite(value) for value in statistics.values())
    json.dumps(statistics, allow_nan=False)


@pytest.mark.parametrize(("score", "events"), [
    ((0, 0), {"home_zero", "away_zero", "draw", "zero_zero"}),
    ((1, 1), {"draw", "one_one"}),
    ((4, 0), {"away_zero", "total_four_plus"}),
    ((2, 2), {"draw", "total_four_plus"}),
])
def test_observed_indicators_match_inclusive_event_boundaries(score, events):
    statistics = diagnostics.row_statistics(row(score=score), (1.2, .7))
    for name in ("home_zero", "away_zero", "draw", "zero_zero", "one_one", "total_four_plus"):
        assert statistics[f"{name}_residual"] + statistics[f"{name}_probability"] == pytest.approx(int(name in events))


def test_large_score_nll_is_not_capped_by_a_clipped_probability():
    statistics = diagnostics.row_statistics(row(score=(250, 170)), (1.2, .7))
    assert statistics["joint_score_nll"] == pytest.approx(
        1.2 - 250 * math.log(1.2) + math.lgamma(251)
        + .7 - 170 * math.log(.7) + math.lgamma(171))
    assert statistics["home_poisson_nll"] > -math.log(1e-300)
    assert statistics["away_poisson_nll"] > -math.log(1e-300)
    assert statistics["joint_score_nll"] > 1000


def test_underflowing_tiny_event_probabilities_do_not_make_analytic_nll_infinite():
    # The event probability underflows, unlike the analytic loss and draw pmf.
    statistics = diagnostics.row_statistics(row(score=(0, 0)), (1000., 1000.))
    assert statistics["one_one_probability"] == 0
    assert statistics["zero_zero_probability"] == 0
    assert statistics["draw_probability"] > 0
    assert statistics["joint_score_nll"] == 2000
    assert all(math.isfinite(value) for value in statistics.values())


def test_conditional_pearson_expectation_is_zero_with_heterogeneous_rates():
    # A heterogeneous Poisson mixture has marginal variance > marginal mean.
    # The per-match residual moment must instead center at each conditional rate.
    expected_moments, expected_residuals = [], []
    for rate in (.25, 4.):
        probability = poisson.pmf(np.arange(50), rate)
        statistics = [diagnostics.row_statistics(row(score=(goal, 0)), (rate, 1.))
                      for goal in range(50)]
        expected_moments.append(sum(p * s["home_pearson_square_minus_one"]
                                    for p, s in zip(probability, statistics)))
        expected_residuals.append(sum(p * s["home_residual"]
                                      for p, s in zip(probability, statistics)))
    assert expected_moments == pytest.approx([0., 0.], abs=1e-13)
    assert expected_residuals == pytest.approx([0., 0.], abs=1e-13)
    assert np.mean((.25, 4.)) + np.var((.25, 4.)) > np.mean((.25, 4.))


@pytest.mark.parametrize("score", [(True, 0), (1., 0), (-1, 0), (0, float("nan")),
                                   (1,), (1, 0, 2), "10", None])
def test_scores_must_be_nonnegative_integers_not_coerced_values(score):
    prediction = row()
    prediction["actual_score"] = score
    with pytest.raises(ValueError, match="score"):
        diagnostics.row_statistics(prediction, (1., 1.))


@pytest.mark.parametrize("rates", [(True, 1), (0, 1), (-1, 1), (float("nan"), 1),
                                   (1, float("inf")), ("1", 1), (1,), None])
def test_rate_values_are_not_silently_repaired(rates):
    with pytest.raises(ValueError):
        diagnostics.row_statistics(row(), rates)


def test_numpy_integral_scores_are_valid_and_numpy_boolean_is_not():
    prediction = row(score=(np.int64(3), np.int32(0)))
    assert diagnostics.row_statistics(prediction, (1., 1.))["home_goals"] == 3
    prediction["actual_score"][0] = np.bool_(True)
    with pytest.raises(ValueError, match="score"):
        diagnostics.row_statistics(prediction, (1., 1.))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -.01, 1.01])
def test_numerical_failure_in_draw_probability_remains_visible(monkeypatch, invalid):
    monkeypatch.setattr(diagnostics.skellam, "pmf", lambda *args: invalid)
    with pytest.raises(ValueError):
        diagnostics.row_statistics(row(), (1., 1.))


def test_numerical_failure_in_four_plus_probability_remains_visible(monkeypatch):
    monkeypatch.setattr(diagnostics.poisson, "sf", lambda *args: float("nan"))
    with pytest.raises(ValueError, match="finite"):
        diagnostics.row_statistics(row(), (1., 1.))


def test_rate_edges_left_closed_right_open_and_top_bin_includes_three():
    assert diagnostics.RATE_EDGES == (0., .75, 1., 1.25, 1.5, 2., 3.)
    edges = np.asarray(diagnostics.RATE_EDGES[1:])
    inputs = np.column_stack((np.nextafter(edges, 0), edges))
    labels = diagnostics.rate_bin_labels(inputs)
    assert labels.shape == inputs.shape
    assert labels.tolist() == [["0-0.75", "0.75-1"], ["0.75-1", "1-1.25"],
                               ["1-1.25", "1.25-1.5"], ["1.25-1.5", "1.5-2"],
                               ["1.5-2", "2-3"], ["2-3", "3+"]]
    assert diagnostics.rate_bin_labels(float(np.nextafter(0., 1.))) == "0-0.75"
    assert diagnostics.rate_bin_labels(1000.) == "3+"


@pytest.mark.parametrize("rates", [0, -1, True, [True, 1.], [], [1, float("nan")],
                                   [float("inf")], ["1.5"], [1 + 2j]])
def test_rate_bins_reject_missing_nonpositive_and_nonreal_inputs(rates):
    with pytest.raises(ValueError):
        diagnostics.rate_bin_labels(rates)


def test_summary_uses_each_rate_and_not_a_single_unconditional_mean():
    rows = [row("low", (0, 1)), row("high", (5, 0)), row("same-rate", (1, 2))]
    rates = [(.2, 1.3), (3.8, .7), (.2, 1.3)]
    statistics = [diagnostics.row_statistics(r, value) for r, value in zip(rows, rates)]
    before = deepcopy((rows, statistics))
    result = diagnostics.summary(rows, statistics)
    assert result["n"] == 3
    assert result["home"]["observed_sum"] == 6
    assert result["home"]["expected_sum"] == pytest.approx(4.2)
    assert result["home"]["residual_per_match"] == pytest.approx(.6)
    assert result["home"]["observed_to_expected_ratio"] == pytest.approx(6 / 4.2)
    assert result["away"]["observed_sum"] == 3
    assert result["away"]["expected_sum"] == pytest.approx(3.3)
    expected_home_zero = 2 * math.exp(-.2) + math.exp(-3.8)
    assert result["home"]["goal_counts"]["0"]["expected"] == pytest.approx(expected_home_zero)
    assert not math.isclose(expected_home_zero, 3 * math.exp(-1.4))
    assert result["home"]["goal_counts"]["0"]["observed"] == 1
    assert result["home"]["goal_counts"]["4+"]["observed"] == 1
    for index, side in enumerate(("home", "away")):
        counts = result[side]["goal_counts"]
        assert sum(value["observed"] for value in counts.values()) == 3
        assert sum(value["expected"] for value in counts.values()) == pytest.approx(3)
        for k in range(4):
            assert counts[str(k)]["expected"] == pytest.approx(sum(
                math.exp(-rate[index]) * rate[index] ** k / math.factorial(k) for rate in rates))
        assert counts["4+"]["expected"] == pytest.approx(sum(poisson.sf(3, rate[index]) for rate in rates))
    for field in diagnostics.STAT_FIELDS:
        values = [stat[field] for stat in statistics]
        assert result["scalar_descriptives"][field] == pytest.approx({
            "mean": np.mean(values), "sd": np.std(values, ddof=1),
            "min": min(values), "median": np.median(values), "max": max(values)})
    assert (rows, statistics) == before
    json.dumps(result, allow_nan=False)


def test_repeated_values_are_not_deduplicated_but_repeated_match_ids_are_rejected():
    rows = [row("one"), row("two")]
    stats = [diagnostics.row_statistics(r, (1.2, .7)) for r in rows]
    result = diagnostics.summary(rows, stats)
    assert result["n"] == 2
    assert result["home"]["expected_sum"] == 2.4
    assert result["scalar_descriptives"]["home_rate"]["sd"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        diagnostics.summary([rows[0], rows[0]], stats)


def test_single_row_undefined_sample_sd_is_none_not_false_zero():
    prediction = row()
    result = diagnostics.summary([prediction], [diagnostics.row_statistics(prediction, (1.2, .7))])
    assert result["n"] == 1
    assert all(value["sd"] is None for value in result["scalar_descriptives"].values())


def test_summary_rejects_empty_misaligned_missing_nonfinite_and_inconsistent_data():
    prediction = row()
    statistic = diagnostics.row_statistics(prediction, (1.2, .7))
    for rows, stats in [([], []), ([prediction], []), ([prediction], [statistic, statistic])]:
        with pytest.raises(ValueError, match="aligned"):
            diagnostics.summary(rows, stats)
    malformed = deepcopy(statistic)
    del malformed["home_rate"]
    with pytest.raises(ValueError, match="fields"):
        diagnostics.summary([prediction], [malformed])
    malformed = {**statistic, "home_residual": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        diagnostics.summary([prediction], [malformed])
    malformed = {**statistic, "home_residual": 0}
    with pytest.raises(ValueError, match="disagree"):
        diagnostics.summary([prediction], [malformed])
    changed = row(score=(5, 1))
    with pytest.raises(ValueError, match="disagree"):
        diagnostics.summary([changed], [statistic])
