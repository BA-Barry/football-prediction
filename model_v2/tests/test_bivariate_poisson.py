"""Independent identities for the research common-shock score distribution."""

from decimal import Decimal, localcontext
import json
import math

import numpy as np
import pytest
from scipy.stats import poisson, skellam

from football_model.bivariate_poisson import (
    MAX_BATCH_TERMS, MAX_GRID_GOAL, MAX_SUM_TERMS, batch_logpmf, logpmf, score_grid,
)
from football_model.probability import score_grid as original_grid, summarize


@pytest.mark.parametrize("home,away", [(.01, .2), (.48564, 4.53747), (1.5, 1.1), (20., 25.)])
def test_zero_parameter_exactly_reproduces_original_grid(home, away):
    grid, metadata = score_grid(home, away, 0.)
    np.testing.assert_array_equal(grid.grid, original_grid(home, away, rho=0.).grid)
    assert metadata["common_mean"] == 0.
    assert metadata["pearson_correlation"] == 0.
    for h, a in ((0, 0), (1, 1), (3, 7), (100, 120)):
        assert logpmf(h, a, home, away, 0.) == pytest.approx(
            poisson.logpmf(h, home)+poisson.logpmf(a, away), abs=5e-12)


@pytest.mark.parametrize("home,away", [(.01, .2), (.48564, 4.53747), (1.5, 1.1), (20., 25.)])
@pytest.mark.parametrize("k", [0., .01, .4, .95])
def test_exact_poisson_marginals_means_covariance_and_metadata(home, away, k):
    grid, metadata = score_grid(home, away, k)
    goals = np.arange(grid.grid.shape[0])
    assert np.isfinite(grid.grid).all() and np.all(grid.grid >= 0)
    assert grid.grid.sum() == pytest.approx(1., abs=8e-15)
    assert grid.home_goal_expectation == home
    assert grid.away_goal_expectation == away
    bound = metadata["omitted_mass_bound"]
    assert bound <= 2.01e-12
    # Conditioning on the retained rectangle can change a marginal by at most
    # q/(1-q), with q <= union bound; include ordinary arithmetic tolerance.
    marginal_error_bound = bound/(1-bound)+1e-13
    np.testing.assert_allclose(grid.grid.sum(axis=1), poisson.pmf(goals, home),
                               atol=marginal_error_bound, rtol=0)
    np.testing.assert_allclose(grid.grid.sum(axis=0), poisson.pmf(goals, away),
                               atol=marginal_error_bound, rtol=0)
    for side, moment in (("home", float(grid.grid.sum(axis=1)@goals)),
                         ("away", float(grid.grid.sum(axis=0)@goals))):
        mu = home if side == "home" else away
        error = abs(moment-mu)
        assert metadata["finite_grid_mean_error"][side] == pytest.approx(error, abs=5e-15)
        assert error <= metadata["finite_grid_mean_error_upper_bounds"][side]
        assert metadata["finite_grid_means"][side] == pytest.approx(moment, abs=1e-14)
    h, a = np.indices(grid.grid.shape)
    covariance = float(np.sum(grid.grid*(h-home)*(a-away)))
    assert covariance == pytest.approx(k*min(home, away), abs=2e-8)
    assert metadata["pearson_correlation"] == pytest.approx(k*min(home, away)/math.sqrt(home*away))
    assert metadata["error_bound_scope"].endswith("not_interval_certified")
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("k", [.01, .4, .95])
def test_swap_symmetry_and_scalar_batch_agreement(k):
    first, _ = score_grid(1.8, .9, k)
    second, _ = score_grid(.9, 1.8, k)
    np.testing.assert_allclose(first.grid, second.grid.T, atol=1e-16, rtol=0)
    h, a = np.array([[0, 1], [3, 25]]), np.array([[2, 1], [0, 35]])
    home, away = np.array([[1.8, 1.], [2.5, 4.]]), np.array([[.9, 2.], [.4, 3.]])
    actual = batch_logpmf(h, a, home, away, k)
    swapped = batch_logpmf(a, h, away, home, k)
    np.testing.assert_allclose(actual, swapped, atol=3e-14, rtol=0)
    for index in np.ndindex(h.shape):
        assert actual[index] == pytest.approx(logpmf(h[index], a[index], home[index], away[index], k), abs=3e-14)
    assert batch_logpmf(1, 2, 1.8, .9, k).shape == ()


def decimal_probability(h, a, home, away, k):
    """70-digit finite convolution, independent of scipy logsumexp/PMF routines."""
    with localcontext() as ctx:
        ctx.prec = 70
        home, away, k = map(lambda x: Decimal(str(x)), (home, away, k))
        c = k*min(home, away)
        u, v = home-c, away-c
        summation = sum((u**(h-j)*v**(a-j)*c**j /
                         Decimal(math.factorial(h-j)*math.factorial(a-j)*math.factorial(j)))
                        for j in range(min(h, a)+1))
        return float((-(u+v+c)).exp()*summation), float(-(u+v+c)+summation.ln())


@pytest.mark.parametrize("h,a,home,away,k", [
    (0, 0, 1.5, 1.1, .4), (1, 1, 1.5, 1.1, .4),
    (4, 2, 4.53747, .48564, .95), (130, 110, .48564, 4.53747, .01),
])
def test_independent_decimal_full_support_log_probability(h, a, home, away, k):
    expected_probability, expected_log = decimal_probability(h, a, home, away, k)
    actual = logpmf(h, a, home, away, k)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected_log, abs=3e-12)
    if h < 10 and a < 10:
        grid, metadata = score_grid(home, away, k)
        raw_probability = grid.grid[h, a]*metadata["finite_grid_mass_before_normalization"]
        assert raw_probability == pytest.approx(expected_probability, abs=1e-15)
    else:
        grid, _ = score_grid(home, away, k)
        assert max(h, a) >= grid.grid.shape[0]


@pytest.mark.parametrize("k", [.05, .4, .95])
def test_goal_difference_skellam_and_totals_compound_poisson_identity(k):
    home, away = 1.5, 2.1
    c = k*min(home, away)
    u, v = home-c, away-c
    grid, _ = score_grid(home, away, k)
    forecast = summarize(grid)
    # A-H=V-U: shared W cancels exactly, so an independent Skellam oracle
    # checks the 1X2 and away-minus-two settlement probabilities.
    expected_outcomes = [skellam.cdf(-1, v, u), skellam.pmf(0, v, u), skellam.sf(0, v, u)]
    np.testing.assert_allclose(list(forecast["one_x_two"].values()), expected_outcomes, atol=4e-12, rtol=0)
    expected_handicap = [skellam.sf(2, v, u), skellam.pmf(2, v, u), skellam.cdf(1, v, u)]
    np.testing.assert_allclose(list(forecast["away_minus_two_jincai"].values()), expected_handicap,
                               atol=4e-12, rtol=0)
    # T=U+V+2W, not Poisson(mu_H+mu_A) once c>0.
    exact_totals = [sum(poisson.pmf(j, c)*poisson.pmf(t-2*j, u+v)
                        for j in range(t//2+1)) for t in range(4)]
    np.testing.assert_allclose(list(forecast["total_goals"].values()),
                               exact_totals+[1-sum(exact_totals)], atol=4e-12, rtol=0)
    assert math.exp(logpmf(0, 0, home, away, k)) == pytest.approx(math.exp(-home-away+c), abs=1e-15)


def test_dependence_changes_all_requested_score_markets_but_not_raw_means():
    independent, _ = score_grid(1.5, 2.1, 0.)
    dependent, _ = score_grid(1.5, 2.1, .4)
    first, second = summarize(independent), summarize(dependent)
    for field in ("one_x_two", "total_goals", "away_minus_two_jincai"):
        assert max(abs(first[field][key]-second[field][key]) for key in first[field]) > .005
    assert first["expected_goals"] == pytest.approx(second["expected_goals"], abs=1e-10)


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), float("nan"), float("inf"), "1", None])
def test_invalid_means_fail_without_coercion(value):
    with pytest.raises(ValueError):
        logpmf(1, 1, value, 1., .2)
    with pytest.raises(ValueError):
        score_grid(1., value, .2)
    with pytest.raises(ValueError):
        batch_logpmf([1], [1], [1.], [value], .2)


@pytest.mark.parametrize("value", [-.1, .950001, 1, True, np.bool_(True), float("nan"), float("inf"), "0", None])
def test_invalid_k_fail_without_clamping(value):
    with pytest.raises(ValueError):
        logpmf(1, 1, 1.5, 1.1, value)
    with pytest.raises(ValueError):
        score_grid(1.5, 1.1, value)


@pytest.mark.parametrize("value", [-1, .5, 1., True, np.bool_(True), float("nan"), float("inf"), "1", None])
def test_invalid_counts_are_not_silently_cast(value):
    with pytest.raises(ValueError):
        logpmf(value, 1, 1.5, 1.1, .1)
    with pytest.raises(ValueError):
        batch_logpmf([1], [value], [1.5], [1.1], .1)


@pytest.mark.parametrize("max_goal", [True, np.bool_(True), 0, -1, 15., "15", MAX_GRID_GOAL+1])
def test_invalid_support_resource_limits(max_goal):
    with pytest.raises(ValueError):
        score_grid(1.5, 1.1, .1, max_goal=max_goal)


def test_explicit_truncation_and_numerical_resource_failures():
    with pytest.raises(ValueError, match="material"):
        score_grid(1.5, 1.1, .1, max_goal=3)
    with pytest.raises(ValueError, match="resource"):
        score_grid(10000., 1., .1)
    with pytest.raises(ValueError, match="resource"):
        logpmf(MAX_SUM_TERMS, MAX_SUM_TERMS, 1.5, 1.1, .1)
    with pytest.raises(ValueError, match="resource"):
        logpmf(10**400, 0, 1.5, 1.1, .1)
    with pytest.raises(ValueError, match="resource"):
        logpmf(2**53+1, 0, 1.5, 1.1, .1)
    with pytest.raises(ValueError, match="resource"):
        score_grid(float.fromhex('0x0.0000000000001p-1022'), 1., .01)
    with pytest.raises(ValueError, match="resource"):
        n = MAX_BATCH_TERMS//100+1
        batch_logpmf([99]*n, [99]*n, [1.5]*n, [1.1]*n, .1)


def test_identical_shapes_nonempty_and_scalar_api():
    with pytest.raises(ValueError, match="identical shapes"):
        batch_logpmf([[1]], [1], [[1.5]], [[1.1]], .1)
    with pytest.raises(ValueError, match="nonempty"):
        batch_logpmf([], [], [], [], .1)
    with pytest.raises(ValueError, match="scalar"):
        logpmf([1], 1, 1.5, 1.1, .1)


def test_numpy_scalar_types_and_explicit_support():
    value = logpmf(np.int64(2), np.uint64(1), np.float64(1.5), np.float32(1.1), np.float64(.2))
    assert math.isfinite(value)
    grid, metadata = score_grid(1.5, 1.1, .2, max_goal=np.int64(30))
    assert grid.grid.shape == (31, 31)
    assert metadata["omitted_mass_bound"] < 1e-25
