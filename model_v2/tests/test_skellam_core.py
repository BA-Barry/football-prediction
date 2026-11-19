import math

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import skellam

from football_model.skellam_core import (
    log_probability_and_lograte_score, net_goal_probabilities,
)


def test_log_likelihood_and_analytic_scores_match_independent_calculations():
    z = np.array([-5, -1, 0, 1, 4])[:, None]
    lh = np.log([.1, .8, 2, 4, 1e-5, 100])
    la = np.log([.3, 1.2, 2, .15, 2, 90])
    logp, gh, ga = log_probability_and_lograte_score(z, lh, la)
    assert logp.shape == (5, 6)
    np.testing.assert_allclose(logp, skellam.logpmf(z, np.exp(lh), np.exp(la)), atol=3e-13)
    step = 1e-5
    dh = (log_probability_and_lograte_score(z, lh + step, la)[0]
          - log_probability_and_lograte_score(z, lh - step, la)[0]) / (2 * step)
    da = (log_probability_and_lograte_score(z, lh, la + step)[0]
          - log_probability_and_lograte_score(z, lh, la - step)[0]) / (2 * step)
    np.testing.assert_allclose(gh, dh, rtol=2e-8, atol=3e-9)
    np.testing.assert_allclose(ga, da, rtol=2e-8, atol=3e-9)


def test_home_away_exchange_flips_net_goal_and_scores():
    z = np.arange(-9, 10)
    logp, gh, ga = log_probability_and_lograte_score(z, math.log(3), math.log(.4))
    reverse, reverse_h, reverse_a = log_probability_and_lograte_score(-z, math.log(.4), math.log(3))
    np.testing.assert_allclose(logp, reverse, atol=1e-13)
    np.testing.assert_allclose(gh, reverse_a, atol=1e-13)
    np.testing.assert_allclose(ga, reverse_h, atol=1e-13)


def test_tiny_rates_use_positive_series_instead_of_probability_underflow():
    # A direct probability is below float range; its log and scores remain valid.
    lh, la = math.log(1e-100), math.log(2e-100)
    logp, gh, ga = log_probability_and_lograte_score(4, lh, la)
    expected = -math.exp(lh) - math.exp(la) + 4 * lh - gammaln(5)
    assert math.isfinite(logp)
    assert logp == pytest.approx(expected, abs=1e-12)
    assert gh == pytest.approx(4)
    assert ga == pytest.approx(-2e-100, rel=1e-12, abs=0)
    draw, _, _ = log_probability_and_lograte_score(0, lh, la)
    assert draw == pytest.approx(-3e-100, rel=1e-12, abs=0)


def test_series_recovers_conditional_moment_even_if_only_numerator_underflows():
    # I_3 is representable, I_4 underflows: simply replacing the ratio by zero
    # would lose a representable conditional common count.
    lh, la = math.log(1e100), math.log(1e-300)
    _, _, ga = log_probability_and_lograte_score(3, lh, la)
    assert ga == pytest.approx(2.5e-201 - 1e-300, rel=1e-12, abs=0)


@pytest.mark.parametrize("home,away", [(.1, .2), (1.5, 1.5), (4., .25), (25., 18.)])
def test_support_keeps_mass_and_independently_reports_both_tails(home, away):
    result = net_goal_probabilities(math.log(home), math.log(away), tail_tolerance=1e-12)
    support = result["net_goals"]
    assert support == list(range(support[0], support[-1] + 1))
    assert result["left_tail"] == pytest.approx(skellam.cdf(support[0] - 1, home, away), rel=1e-12, abs=1e-16)
    assert result["right_tail"] == pytest.approx(skellam.sf(support[-1], home, away), rel=1e-12, abs=1e-16)
    assert result["tail_mass"] == result["left_tail"] + result["right_tail"]
    assert result["tail_mass"] <= 1e-12
    assert math.fsum(result["probabilities"] + [result["left_tail"], result["right_tail"]]) == pytest.approx(1, abs=2e-13)
    np.testing.assert_allclose(result["probabilities"], skellam.pmf(support, home, away), atol=1e-14)
    np.testing.assert_allclose(result["hda"], [skellam.sf(0, home, away), skellam.pmf(0, home, away), skellam.cdf(-1, home, away)], atol=1e-14)


def test_support_probabilities_are_not_renormalized():
    result = net_goal_probabilities(math.log(1), math.log(1), tail_tolerance=.1)
    assert result["tail_mass"] > 0
    assert math.fsum(result["probabilities"]) < 1
    expected = skellam.pmf(result["net_goals"], 1, 1)
    np.testing.assert_allclose(result["probabilities"], expected, rtol=2e-14, atol=0)


def test_reflected_cdf_retains_small_survival_tails():
    symmetric = net_goal_probabilities(0., 0.)
    assert symmetric["left_tail"] > 0
    assert symmetric["right_tail"] == pytest.approx(symmetric["left_tail"], rel=1e-13, abs=0)
    tiny = net_goal_probabilities(math.log(1e-100), math.log(2e-100))
    assert tiny["hda"][0] == pytest.approx(1e-100, rel=1e-12, abs=0)
    assert tiny["hda"][2] == pytest.approx(2e-100, rel=1e-12, abs=0)


def test_invalid_inputs_and_unrepresentable_rates_raise_instead_of_clipping():
    for args in ((.1, 0., 0.), (True, 0., 0.), (0, math.nan, 0.)):
        with pytest.raises(ValueError):
            log_probability_and_lograte_score(*args)
    for args in ((0, 1000., 0.), (0, -1000., 0.)):
        with pytest.raises(FloatingPointError):
            log_probability_and_lograte_score(*args)
    for tolerance in (0, 1, math.nan, True):
        with pytest.raises(ValueError):
            net_goal_probabilities(0., 0., tolerance)
    with pytest.raises(ValueError, match="scalar"):
        net_goal_probabilities([0., 1.], 0.)


def test_scalar_inputs_produce_python_floats():
    result = log_probability_and_lograte_score(0, 0., 0.)
    assert all(type(value) is float for value in result)
