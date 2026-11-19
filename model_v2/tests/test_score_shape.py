"""Independent formulas and Poisson finite sums for mean-orthogonal scores."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest
from scipy.stats import binom, poisson

from football_model import goal_rate_diagnostics, score_shape as shape


def row(score=(2, 1)):
    return {"actual_score": list(score)}


def pmf(k, mean):
    return math.exp(k * math.log(mean) - mean - math.lgamma(k + 1))


def indicator(event, h, a):
    return {"home_zero": h == 0, "away_zero": a == 0, "draw": h == a,
            "zero_zero": h == a == 0, "one_one": h == a == 1,
            "total_four_plus": h + a >= 4}[event]


@pytest.mark.parametrize("rates", [(.35, 2.4), (1.6, .7), (4.2, 3.1)])
def test_probabilities_and_coefficients_match_independent_joint_sums(rates):
    lh, la = rates
    support = range(65)
    weights = [(h, a, pmf(h, lh) * pmf(a, la)) for h in support for a in support]
    assert math.fsum(w for _, _, w in weights) == pytest.approx(1., abs=2e-14)
    for event in shape.EVENTS:
        event_p = math.fsum(w for h, a, w in weights if indicator(event, h, a))
        ch = math.fsum(w * (h - lh) / lh for h, a, w in weights if indicator(event, h, a))
        ca = math.fsum(w * (a - la) / la for h, a, w in weights if indicator(event, h, a))
        actual = shape.event_projection(event, rates)
        assert actual == pytest.approx({"probability": event_p,
                                       "home_coefficient": ch, "away_coefficient": ca}, abs=2e-14)


@pytest.mark.parametrize("rates", [(.35, 2.4), (1.6, .7), (4.2, 3.1)])
def test_projected_events_have_zero_mean_and_both_mean_covariances(rates):
    lh, la = rates
    h, a = np.indices((65, 65))
    w = np.outer([pmf(k, lh) for k in range(65)], [pmf(k, la) for k in range(65)])
    for event in shape.EVENTS:
        p = shape.event_projection(event, rates)
        observed = np.asarray([[indicator(event, int(x), int(y)) for y in range(65)] for x in range(65)])
        z = observed - p["probability"] - p["home_coefficient"] * (h-lh) - p["away_coefficient"] * (a-la)
        assert float((z*w).sum()) == pytest.approx(0., abs=2e-14)
        assert float((z*(h-lh)*w).sum()) == pytest.approx(0., abs=5e-14)
        assert float((z*(a-la)*w).sum()) == pytest.approx(0., abs=5e-14)
        # Orthogonalization can produce residuals outside [-1,1]; it is not a
        # replacement event probability, and no such range restriction applies.
        expected_variance = p["probability"]*(1-p["probability"]) - lh*p["home_coefficient"]**2 - la*p["away_coefficient"]**2
        assert float((z*z*w).sum()) == pytest.approx(expected_variance, abs=5e-14)


@pytest.mark.parametrize("rates", [(.6, 2.3), (1.4, .8), (3.7, 1.1)])
@pytest.mark.parametrize("event", shape.EVENTS)
def test_projection_coefficient_is_probability_derivative_for_its_mean(event, rates):
    epsilon = 1e-5
    p = shape.event_projection(event, rates)
    for side, label in enumerate(("home", "away")):
        lower, upper = list(rates), list(rates)
        lower[side] -= epsilon
        upper[side] += epsilon
        derivative = (shape.event_projection(event, upper)["probability"]
                      - shape.event_projection(event, lower)["probability"]) / (2*epsilon)
        assert p[f"{label}_coefficient"] == pytest.approx(derivative, abs=2e-10)


@pytest.mark.parametrize("score", [(0, 0), (1, 1), (0, 4), (4, 0), (2, 2), (5, 3), (250, 170)])
def test_row_matches_plain_formulas_retains_raw_events_and_does_not_mutate(score):
    rates = [1.8, .65]
    prediction = row(score)
    before = deepcopy((prediction, rates))
    base = goal_rate_diagnostics.row_statistics(prediction, rates)
    actual = shape.row_shape_statistics(prediction, rates)
    assert set(actual) == set(shape.STAT_FIELDS) and len(actual) == len(shape.STAT_FIELDS) == 39
    h, a = score
    assert actual["home_dispersion_residual"] == pytest.approx(((h-rates[0])**2-h)/rates[0])
    assert actual["away_dispersion_residual"] == pytest.approx(((a-rates[1])**2-a)/rates[1])
    assert actual["mean_centered_cross_product"] == pytest.approx((h-rates[0])*(a-rates[1]))
    for key in ("home_rate", "away_rate", "home_goals", "away_goals", "home_residual", "away_residual"):
        assert actual[key] == base[key]
    for event in shape.EVENTS:
        assert actual[f"{event}_residual"] == base[f"{event}_residual"]
        assert actual[f"{event}_probability"] == base[f"{event}_probability"]
        projection = shape.event_projection(event, rates)
        assert actual[f"{event}_mean_orthogonal_residual"] == pytest.approx(
            int(indicator(event, h, a)) - projection["probability"]
            - projection["home_coefficient"]*(h-rates[0])
            - projection["away_coefficient"]*(a-rates[1]))
    assert (prediction, rates) == before
    assert all(math.isfinite(value) for value in actual.values())
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("true_mean,mu", [(1., 1.), (1., 1.5), (3., .8), (.2, 2.)])
def test_dispersion_population_mean_under_poisson_mean_error_cannot_be_negative(true_mean, mu):
    values = [shape.row_shape_statistics(row((k, 0)), (mu, 1.))["home_dispersion_residual"]
              for k in range(65)]
    expectation = math.fsum(pmf(k, true_mean)*v for k, v in enumerate(values))
    assert expectation == pytest.approx((true_mean-mu)**2/mu, abs=2e-13)
    assert expectation >= -2e-14


def test_old_pearson_can_be_negative_from_mean_error_while_new_statistic_cannot():
    rows = [goal_rate_diagnostics.row_statistics(row((k, 0)), (1.5, 1.)) for k in range(55)]
    old = math.fsum(pmf(k, 1.)*r["home_pearson_square_minus_one"] for k, r in enumerate(rows))
    new = math.fsum(pmf(k, 1.)*shape.row_shape_statistics(row((k, 0)), (1.5, 1.))[
        "home_dispersion_residual"] for k in range(55))
    assert old == pytest.approx(-1/6)
    assert new == pytest.approx(1/6)


@pytest.mark.parametrize("mu", [.6, 1.2, 2.4])
def test_general_dispersion_expectation_has_variance_mean_and_squared_mean_error(mu):
    n, probability = 6, .2
    true_mean = n*probability
    true_variance = n*probability*(1-probability)
    expected = math.fsum(float(binom.pmf(k, n, probability))*shape.row_shape_statistics(
        row((k, 0)), (mu, 1.))["home_dispersion_residual"] for k in range(n+1))
    assert expected == pytest.approx((true_variance-true_mean+(true_mean-mu)**2)/mu, abs=5e-15)
    if mu == true_mean:
        assert expected < 0


def test_dispersion_zero_mean_and_orthogonality_to_poisson_mean_residual():
    for mu in (.4, 1.7, 4.2):
        ds = [shape.row_shape_statistics(row((k, 0)), (mu, 1.))["home_dispersion_residual"] for k in range(65)]
        assert math.fsum(pmf(k, mu)*d for k, d in enumerate(ds)) == pytest.approx(0., abs=2e-13)
        assert math.fsum(pmf(k, mu)*d*(k-mu) for k, d in enumerate(ds)) == pytest.approx(0., abs=5e-13)


def test_home_away_swap_preserves_symmetric_events_and_swaps_projection_sides():
    first = shape.row_shape_statistics(row((3, 0)), (.6, 2.2))
    second = shape.row_shape_statistics(row((0, 3)), (2.2, .6))
    assert first["home_residual"] == second["away_residual"]
    assert first["home_dispersion_residual"] == second["away_dispersion_residual"]
    assert first["mean_centered_cross_product"] == second["mean_centered_cross_product"]
    pairs = [("home_zero", "away_zero"), ("away_zero", "home_zero")] + [(e, e) for e in shape.EVENTS[2:]]
    for e1, e2 in pairs:
        for suffix in ("probability", "residual", "mean_orthogonal_residual"):
            assert first[f"{e1}_{suffix}"] == pytest.approx(second[f"{e2}_{suffix}"], abs=2e-15)
        assert first[f"{e1}_home_coefficient"] == pytest.approx(second[f"{e2}_away_coefficient"], abs=2e-15)


@pytest.mark.parametrize("target", [(0, 0), (1, 1), (20, 12), (250, 170)])
def test_arbitrary_cell_has_full_support_and_exact_projection_identity(target):
    lh, la = 1.7, .9
    k, l = target
    actual = shape.event_projection("cell", (lh, la), score=target)
    p = pmf(k, lh)*pmf(l, la)
    assert actual["probability"] == pytest.approx(p, rel=2e-12, abs=1e-300)
    assert actual["home_coefficient"] == pytest.approx((k-lh)*p/lh, rel=2e-12, abs=1e-300)
    assert actual["away_coefficient"] == pytest.approx((l-la)*p/la, rel=2e-12, abs=1e-300)
    assert all(math.isfinite(x) for x in actual.values())
    if max(target) == 20:
        assert actual["probability"] > 0


def test_cross_product_is_not_implicitly_a_covariance_or_standardized_correlation():
    actual = shape.row_shape_statistics(row((2, 3)), (1., 1.))
    assert actual["mean_centered_cross_product"] == 2.
    assert actual["mean_centered_cross_product"] > 1.


def test_finite_underflow_does_not_force_analytic_or_projected_statistics_to_fail():
    actual = shape.row_shape_statistics(row((0, 0)), (1000., 1000.))
    assert actual["zero_zero_probability"] == actual["one_one_probability"] == 0.
    assert actual["home_dispersion_residual"] == pytest.approx(1000.)
    assert actual["draw_probability"] > 0
    assert all(math.isfinite(v) for v in actual.values())


@pytest.mark.parametrize("rates", [(True, 1), (0, 1), (-1, 1), (float("nan"), 1),
                                   (1, float("inf")), ("1", 1), (1,), None])
def test_rate_validation_is_shared_without_repairs(rates):
    with pytest.raises(ValueError):
        shape.row_shape_statistics(row(), rates)
    with pytest.raises(ValueError):
        shape.event_projection("draw", rates)


@pytest.mark.parametrize("score", [(True, 0), (1., 0), (-1, 0), (0, float("nan")),
                                   (1,), (1, 0, 2), "10", None])
def test_score_validation_is_shared_without_repairs(score):
    with pytest.raises(ValueError):
        shape.row_shape_statistics({"actual_score": score}, (1., 1.))
    with pytest.raises(ValueError):
        shape.event_projection("cell", (1., 1.), score=score)


@pytest.mark.parametrize("event", [None, [], True, "win", "Cell", 1])
def test_unknown_event_is_not_silently_substituted(event):
    with pytest.raises(ValueError, match="event"):
        shape.event_projection(event, (1., 1.))


def test_target_score_must_be_explicit_and_only_for_cell():
    with pytest.raises(ValueError, match="score"):
        shape.event_projection("cell", (1., 1.))
    with pytest.raises(ValueError, match="cell"):
        shape.event_projection("draw", (1., 1.), score=(1, 1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.01, 1.01])
def test_invalid_draw_shift_is_not_clipped_or_replaced(monkeypatch, value):
    original = shape.skellam.pmf
    monkeypatch.setattr(shape.skellam, "pmf", lambda k, *args: value if k == -1 else original(k, *args))
    with pytest.raises(ValueError):
        shape.row_shape_statistics(row(), (1., 1.))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.01, 1.01])
def test_invalid_total_projection_coefficient_is_not_clipped(monkeypatch, value):
    monkeypatch.setattr(shape.poisson, "pmf", lambda *args: value)
    with pytest.raises(ValueError):
        shape.row_shape_statistics(row(), (1., 1.))


def test_nonfinite_unstandardized_cross_product_is_an_explicit_failure():
    # Existing standardized diagnostics can remain finite although this new
    # unstandardized cross product cannot be represented.
    with pytest.raises(ValueError, match="finite"):
        shape.row_shape_statistics(row((0, 0)), (1e160, 1e160))


def test_numpy_integer_scores_accepted_but_numpy_bool_not_coerced():
    actual = shape.row_shape_statistics(row((np.int64(3), np.int32(0))), (1., 1.))
    assert actual["home_goals"] == 3
    with pytest.raises(ValueError):
        shape.row_shape_statistics(row((np.bool_(True), 0)), (1., 1.))
