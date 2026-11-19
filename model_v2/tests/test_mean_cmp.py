"""Numerical tests independent of the fitted football experiment."""

from decimal import Decimal, localcontext
import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import i0e, i1e
from scipy.stats import poisson

from football_model.mean_cmp import batch_logpmf, cmp_score_forecast, mean_cmp
from football_model.probability import score_grid, summarize


@pytest.mark.parametrize("mu", [.48564, 1., 1.7, 4.53747])
@pytest.mark.parametrize("nu", [.25, .7, 1., 1.1048243557892572, 2., 4.])
def test_fixed_mean_tail_and_full_support(mu, nu):
    d = mean_cmp(mu, nu)
    k = np.arange(len(d.pmf))
    assert np.isfinite(d.pmf).all() and np.all(d.pmf >= 0)
    assert d.pmf.sum() == pytest.approx(1., abs=2e-14)
    assert float(d.pmf@k) == pytest.approx(mu, abs=2e-10)
    assert d.metadata["tail_mass_upper_bound"] <= 1e-14
    assert d.metadata["tail_first_moment_upper_bound"] <= 1e-13
    assert d.metadata["tail_second_moment_upper_bound"] <= 1e-11
    assert d.metadata["error_bound_scope"].endswith("not_interval_certified")
    json.dumps(d.metadata, allow_nan=False)
    assert math.isfinite(d.logpmf(len(d.pmf)+30))
    assert d.logpmf(3)-d.logpmf(2) == pytest.approx(d.eta-nu*math.log(3), abs=1e-13)


@pytest.mark.parametrize("mu", [.01, .48564, 1., 4.53747, 20.])
def test_poisson_branch_has_analytic_normalizer_and_high_score(mu):
    d = mean_cmp(mu, 1.)
    assert d.log_normalizer == mu
    assert d.eta == pytest.approx(math.log(mu))
    assert d.variance == mu
    for k in (0, 1, 15, 100):
        assert d.logpmf(k) == pytest.approx(poisson.logpmf(k, mu), abs=1e-12)


@pytest.mark.parametrize("mu", [.1, .48564, 1., 2., 4.53747, 10.])
def test_nu_two_independent_bessel_identity(mu):
    # Z(theta,2)=I0(2sqrt(theta)); no CMP-series implementation is reused.
    x = brentq(lambda x: x/2*i1e(x)/i0e(x)-mu, 1e-8, 10*mu+20, xtol=1e-13)
    eta = 2*math.log(x/2)
    logz = math.log(i0e(x))+x
    d = mean_cmp(mu, 2.)
    assert d.eta == pytest.approx(eta, abs=1e-10)
    assert d.log_normalizer == pytest.approx(logz, abs=1e-10)
    assert d.logpmf(30) == pytest.approx(30*eta-2*math.lgamma(31)-logz, abs=1e-9)


def decimal_reference(mu, nu):
    """65-digit recurrence and eta bisection; no NumPy, gammaln or CMP roots."""
    with localcontext() as ctx:
        ctx.prec = 65
        target, power = Decimal(str(mu)), Decimal(str(nu))
        def moments(eta):
            theta, w, z, first = eta.exp(), Decimal(1), Decimal(1), Decimal(0)
            for k in range(1, 1000):
                w *= theta / (Decimal(k)**power)
                z += w
                first += k*w
                if k > 60 and w < z*Decimal('1e-58'):
                    return first/z, z.ln()
            raise AssertionError("Independent decimal reference failed to converge")
        low, high = Decimal(-30), Decimal(10)
        for _ in range(190):
            eta = (low+high)/2
            m, _ = moments(eta)
            if m < target:
                low = eta
            else:
                high = eta
        eta = (low+high)/2
        return float(eta), float(moments(eta)[1])


@pytest.mark.parametrize("mu,nu", [(.48564,.25), (1.,4.), (4.53747,4.)])
def test_decimal_independent_root_and_normalizer(mu, nu):
    eta, logz = decimal_reference(mu, nu)
    d = mean_cmp(mu, nu)
    assert d.eta == pytest.approx(eta, abs=1e-10)
    assert d.log_normalizer == pytest.approx(logz, abs=1e-10)


@pytest.mark.parametrize("nu", [.25, 1., 1.1048243557892572, 2., 4.])
def test_batch_matches_scalar_without_interpolation(nu):
    means = np.array([[.48564, 1.], [2.3, 4.53747]])
    goals = np.array([[0, 1], [4, 30]])
    actual = batch_logpmf(means, goals, nu)
    expected = np.array([[mean_cmp(mu, nu).logpmf(int(k)) for mu,k in zip(mrow,grow)]
                         for mrow,grow in zip(means,goals)])
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
    assert batch_logpmf(1., 2, nu).shape == ()


def test_swap_and_poisson_forecast_identity():
    f, metadata = cmp_score_forecast(1.8, .9, 1.)
    old = summarize(score_grid(1.8, .9, rho=0))
    assert list(f["one_x_two"].values()) == pytest.approx(list(old["one_x_two"].values()), abs=1e-11)
    a, _ = cmp_score_forecast(1.8, .9, 1.2)
    b, _ = cmp_score_forecast(.9, 1.8, 1.2)
    assert a["one_x_two"]["home"] == pytest.approx(b["one_x_two"]["away"], abs=1e-12)
    assert a["one_x_two"]["draw"] == pytest.approx(b["one_x_two"]["draw"], abs=1e-12)
    assert a["total_goals"] == pytest.approx(b["total_goals"], abs=1e-12)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("value", [0, -1, True, float('nan'), float('inf'), '1'])
def test_invalid_mean_and_nu_are_rejected(value):
    with pytest.raises(ValueError):
        mean_cmp(value, 1.)
    with pytest.raises(ValueError):
        mean_cmp(1., value)


@pytest.mark.parametrize("goals", [-1, 1.5, True, '1'])
def test_invalid_counts_are_not_silently_cast(goals):
    with pytest.raises(ValueError):
        mean_cmp(1., 2.).logpmf(goals)
    with pytest.raises(ValueError):
        batch_logpmf([1.], [goals], 2.)


def test_explicit_resource_and_shape_failures():
    with pytest.raises(ValueError, match="resource"):
        mean_cmp(10000., 2.)
    with pytest.raises(ValueError, match="identical shapes"):
        batch_logpmf([[1.]], [1], 2.)
    with pytest.raises(ValueError, match="nonempty"):
        batch_logpmf([], [], 2.)
