"""Synthetic numeric checks only: no fit, real matches, downloads or sampling."""

import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson, skellam

from football_model.english_tiers_experiment import METRICS, score_forecast
from football_model.gaussian_goal_mixture import _normal_quadrature, forecast, score


ZERO = [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("rates,actual", [((1.4, 1.0), (0, 0)), ((1.4, 1.0), (2, 1)),
                                         ((0.3, 3.2), (1, 4)), ((2.0, 0.7), (24, 17))])
def test_zero_covariance_matches_existing_eight_scores(rates, actual):
    old = score_forecast(rates, actual)
    new = score(np.log(rates), ZERO, actual)
    for key in METRICS:
        assert new[key] == pytest.approx(old[key], abs=2e-9)
    for market in ("one_x_two", "total_goals", "away_minus_two_jincai"):
        assert new["forecast"][market] == pytest.approx(old["forecast"][market], abs=2e-10)


def test_zero_covariance_analytic_markets_and_moments():
    home, away = 1.1, 2.4
    result = forecast(np.log([home, away]), ZERO)
    value = result["forecast"]
    assert value["one_x_two"] == pytest.approx({"home": skellam.sf(0, home, away),
                                               "draw": skellam.pmf(0, home, away),
                                               "away": skellam.cdf(-1, home, away)})
    assert value["total_goals"] == pytest.approx({**{str(k): poisson.pmf(k, home + away) for k in range(4)},
                                                 "4+": poisson.sf(3, home + away)})
    assert value["away_minus_two_jincai"] == pytest.approx({"win": skellam.sf(2, away, home),
                                                           "draw": skellam.pmf(2, away, home),
                                                           "lose": skellam.cdf(1, away, home)})
    assert value["expected_goals"] == pytest.approx({"home": home, "away": away, "total": home + away})
    assert result["goal_variance"] == pytest.approx({"home": home, "away": away})
    assert result["goal_cross_covariance"] == 0


def test_jensen_and_mixed_distribution_not_poisson_of_mean():
    mean, cov = np.log([1.4, 1.0]), [[0.35, 0.10], [0.10, 0.2]]
    result = forecast(mean, cov)
    expected = result["forecast"]["expected_goals"]
    assert expected["home"] == pytest.approx(1.4 * math.exp(0.35 / 2))
    assert expected["away"] == pytest.approx(math.exp(0.2 / 2))
    assert expected["home"] > 1.4 and expected["away"] > 1.0
    assert result["goal_variance"]["home"] == pytest.approx(expected["home"] + expected["home"]**2 * math.expm1(.35))
    assert result["goal_cross_covariance"] == pytest.approx(expected["home"] * expected["away"] * math.expm1(.1))
    plug_in_zero = math.exp(-expected["total"])
    assert result["forecast"]["total_goals"]["0"] > plug_in_zero + .01


def test_unknown_player_variance_keeps_a_predictive_distribution_not_zero_strength():
    certain = forecast(np.log([1.3, 1.1]), ZERO)
    uncertain = forecast(np.log([1.3, 1.1]), [[.25, 0], [0, 0]])
    assert uncertain["goal_variance"]["home"] > uncertain["forecast"]["expected_goals"]["home"]
    assert uncertain["forecast"]["one_x_two"] != certain["forecast"]["one_x_two"]
    assert all(0 < p < 1 for p in uncertain["forecast"]["one_x_two"].values())
    # Integrator has no identities: assigning this variance requires a model.
    assert "unseen" not in uncertain


@pytest.mark.parametrize("cov", [[[.4, .2], [.2, .3]], [[.3, -.2], [-.2, .4]], [[.4, .4], [.4, .4]]])
def test_correlated_rates_exchange_home_and_away(cov):
    mean = np.log([1.7, .8])
    value = forecast(mean, cov)["forecast"]
    reversed_value = forecast(mean[::-1], np.asarray(cov)[::-1, ::-1])["forecast"]
    assert value["one_x_two"]["home"] == pytest.approx(reversed_value["one_x_two"]["away"], abs=2e-14)
    assert value["one_x_two"]["away"] == pytest.approx(reversed_value["one_x_two"]["home"], abs=2e-14)
    assert value["one_x_two"]["draw"] == pytest.approx(reversed_value["one_x_two"]["draw"], abs=2e-14)
    assert value["total_goals"] == pytest.approx(reversed_value["total_goals"], abs=2e-14)


def test_equal_rate_symmetry_splits_exact_top1_tie():
    scored = score(np.log([4., 4.]), [[.1, -.02], [-.02, .1]], (1, 0))
    assert scored["forecast"]["one_x_two"]["home"] == scored["forecast"]["one_x_two"]["away"]
    assert scored["top1_tied"] is True
    assert scored["top1"] == .5


def test_negative_correlation_and_singular_covariance_have_valid_moments():
    value = forecast([0., 0.], [[.2, -.2], [-.2, .2]])
    assert value["goal_cross_covariance"] < 0
    assert value["goal_variance"]["home"] > value["forecast"]["expected_goals"]["home"]
    assert not value["covariance_roundoff"]["negative_eigenvalues_zeroed_for_roundoff"]


@pytest.mark.parametrize("cov", [[[1., 1.01], [1.01, 1.]], [[-.00001, 0], [0, .2]],
                               [[.1, .01], [.02, .1]], [[0, 0], [0, float("nan")]],
                               [[.1, 0]], [0, 0]])
def test_invalid_covariance_is_not_silently_clipped(cov):
    with pytest.raises(ValueError):
        forecast([0, 0], cov)


def test_only_tiny_eigenvalue_roundoff_is_reported_and_corrected():
    raw = [[-1e-16, 0], [0, .2]]
    value = forecast([0, 0], raw)
    assert value["log_cov"][0][0] == 0
    note = value["covariance_roundoff"]
    assert note["input_log_cov"] == raw
    assert note["negative_eigenvalues_zeroed_for_roundoff"] == [-1e-16]
    assert note["roundoff_tolerance"] < 1e-12


@pytest.mark.parametrize("mean", [[0], [0, 0, 0], [0, float("inf")], [[0], [0]]])
def test_invalid_log_mean_rejected(mean):
    with pytest.raises(ValueError):
        forecast(mean, ZERO)


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError):
        forecast([0, 0], ZERO, order=order)


@pytest.mark.parametrize("actual", [(-1, 0), (True, 0), (1.5, 0), (1,), (1, 2, 3), np.asarray(1)])
def test_invalid_actual_rejected(actual):
    with pytest.raises(ValueError):
        score([0, 0], ZERO, actual)


@pytest.mark.parametrize("order", [1, 15, 25])
def test_probabilists_normal_weights_are_normalized_in_two_dimensions(order):
    _, _, log_rates, _, weights, _ = _normal_quadrature([.1, -.2], [[.3, .04], [.04, .2]], order)
    assert len(weights) == order**2
    assert np.all(weights > 0)
    assert sum(weights) == pytest.approx(1., abs=5e-15)
    assert weights @ log_rates == pytest.approx([.1, -.2], abs=1e-14)
    if order > 1:
        centered = log_rates - [.1, -.2]
        assert centered.T @ (weights[:, None] * centered) == pytest.approx(np.asarray([[.3, .04], [.04, .2]]), abs=2e-14)


def test_rare_joint_score_uses_log_sum_exp_not_underflowed_pmf():
    mean, cov, actual = np.log([1.1, .7]), [[.1, .02], [.02, .1]], (1000, 900)
    value = score(mean, cov, actual)
    _, _, _, rates, weights, _ = _normal_quadrature(mean, cov, 15)
    direct_log = poisson.logpmf(actual[0], rates[:, 0]) + poisson.logpmf(actual[1], rates[:, 1])
    assert np.all(np.exp(direct_log) == 0)
    expected = -logsumexp(np.log(weights) + direct_log)
    assert math.isfinite(value["full_score_nll"])
    assert value["full_score_nll"] == pytest.approx(expected, abs=2e-11)


def test_fixed_fifteen_and_twenty_five_orders_agree_for_bounded_synthetic_case():
    args = (np.log([1.6, 1.0]), [[.15, .03], [.03, .10]], (2, 1))
    lower, higher = score(*args, order=15), score(*args, order=25)
    assert lower["order"] == 15 and higher["order"] == 25
    for key in METRICS:
        assert lower[key] == pytest.approx(higher[key], abs=2e-7)
    assert "not MCMC or full Bayes" in lower["approximation"]


def test_conditional_actual_joint_probability_is_integrated_not_plugin_scored():
    mean, cov, actual = [.2, -.1], [[.4, .15], [.15, .3]], (0, 0)
    result = score(mean, cov, actual)
    assert result["full_score_nll"] == pytest.approx(-math.log(result["forecast"]["total_goals"]["0"]), abs=1e-13)
    assert result["full_score_nll"] != pytest.approx(result["forecast"]["expected_goals"]["total"], abs=.01)


def test_unrepresentable_rates_raise_instead_of_clipping():
    with pytest.raises(ValueError, match="floating-point domain"):
        forecast([1000, 0], ZERO)


def test_return_value_contains_only_serializable_finite_values_and_stable_keys():
    value = score([.1, -.1], [[.2, .03], [.03, .1]], [2, 1])
    assert json.loads(json.dumps(value, allow_nan=False)) == value
    assert set(value["forecast"]) == {"one_x_two", "expected_goals", "total_goals", "away_minus_two_jincai"}
    assert set(value["forecast"]["one_x_two"]) == {"home", "draw", "away"}
    assert set(value["forecast"]["total_goals"]) == {"0", "1", "2", "3", "4+"}
    assert set(value["forecast"]["away_minus_two_jincai"]) == {"win", "draw", "lose"}
