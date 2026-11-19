"""Mean-parameterized CMP, using bounded-error adaptive series, not GPL code.

The natural parameter eta is solved rather than identified with log(mean).
Finite series normalizers are lower bounds; all omitted-mass/moment bounds are
reported relative to the retained sum. No failed evaluation falls back to a
different family. These numerical resource limits are not model-selection rules.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.special import gammaln, logsumexp

from .probability import summarize
from .vendor.penaltyblog_grid import FootballProbabilityGrid


MAX_TERMS = 4096
MAX_ITERATIONS = 100
TAIL_MASS_TOL = 1e-14
TAIL_FIRST_TOL = 1e-13
TAIL_SECOND_TOL = 1e-11
MEAN_ATOL = 2e-12
MEAN_RTOL = 2e-12


def _positive(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive real")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite positive real") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive real")
    return value


def _inputs(means, goals=None):
    raw = np.asarray(means, dtype=object)
    if not raw.size:
        raise ValueError("Need nonempty means")
    values = np.asarray([_positive(v, "mean") for v in raw.flat]).reshape(raw.shape)
    if goals is None:
        return values
    counts = np.asarray(goals, dtype=object)
    if counts.shape != raw.shape:
        raise ValueError("Means and integer goals must have identical shapes; no broadcasting")
    for k in counts.flat:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0:
            raise ValueError("Goals must be nonnegative integers")
    try:
        counts = counts.astype(float)
    except (ValueError, OverflowError) as error:
        raise ValueError("Goals exceed finite numerical resources") from error
    if not np.isfinite(counts).all():
        raise ValueError("Goals exceed finite numerical resources")
    return values, counts


def _moments(eta, nu):
    """Vectorized log series with geometric zeroth/first/second tail bounds."""
    mode_log = eta / nu
    if not np.isfinite(mode_log).all() or np.any(mode_log > math.log(MAX_TERMS - 10)):
        raise ValueError("CMP series support exceeds numerical resource limit")
    kmax = max(16, int(np.ceil(np.exp(mode_log).max())) + 8)
    while kmax < MAX_TERMS:
        k = np.arange(kmax + 1, dtype=float)
        lw = eta[:, None] * k - nu * gammaln(k + 1)
        lz = logsumexp(lw, axis=1)
        p = np.exp(lw - lz[:, None])
        mean = p @ k
        variance = np.sum(p * np.square(k[None, :] - mean[:, None]), axis=1)
        q = np.exp(eta - nu * math.log(kmax + 1))
        if np.any(q >= 1):
            kmax = min(MAX_TERMS - 1, int(kmax * 1.5) + 1)
            continue
        denominator = 1 - q
        b0 = p[:, -1] * q / denominator
        b1 = p[:, -1] * (kmax*q/denominator + q/denominator**2)
        b2 = p[:, -1] * (kmax*kmax*q/denominator + 2*kmax*q/denominator**2
                          + q*(1+q)/denominator**3)
        if not all(np.isfinite(v).all() for v in (lz, p, mean, variance, b0, b1, b2)):
            raise ValueError("CMP series produced nonfinite moments or tail bounds")
        if np.all((b0 <= TAIL_MASS_TOL) & (b1 <= TAIL_FIRST_TOL) & (b2 <= TAIL_SECOND_TOL)):
            return lz, p, mean, variance, b0, b1, b2
        if kmax == MAX_TERMS - 1:
            break
        kmax = min(MAX_TERMS - 1, int(kmax * 1.5) + 1)
    raise ValueError("CMP series tail tolerance not reached within resource limit")


def _solve(means, nu):
    means = np.asarray(means, dtype=float).reshape(-1)
    if np.any(means > MAX_TERMS - 20):
        raise ValueError("CMP mean exceeds series numerical resource limit")
    if nu == 1.:
        eta = np.log(means)
        _, retained_p, _, _, b0, b1, b2 = _moments(eta, nu)
        k = np.arange(retained_p.shape[1], dtype=float)
        # Exact Poisson log-normalizer and full-support log pmf; no root solve.
        p = np.exp(eta[:, None]*k - means[:, None] - gammaln(k+1))
        return eta, means.copy(), p, p@k, means.copy(), b0, b1, b2, np.zeros(len(means), dtype=int)
    approximation = means + (nu - 1)/(2*nu)
    eta = np.log(means) - np.log1p(means)
    positive = approximation > 0
    eta[positive] = nu * np.log(approximation[positive])
    if not np.isfinite(eta).all():
        raise ValueError("CMP starting natural parameter is nonfinite")
    lower = np.full(len(means), -np.inf)
    upper = np.full(len(means), np.inf)
    iterations = np.zeros(len(means), dtype=int)
    tolerance = MEAN_ATOL + MEAN_RTOL * means
    for iteration in range(1, MAX_ITERATIONS + 1):
        lz, p, fitted, variance, b0, b1, b2 = _moments(eta, nu)
        residual = fitted-means
        done = np.abs(residual) + b1 <= tolerance
        iterations[(iterations == 0) & done] = iteration
        if done.all():
            return eta, lz, p, fitted, variance, b0, b1, b2, iterations
        lower = np.where(residual < 0, eta, lower)
        upper = np.where(residual > 0, eta, upper)
        active = ~done
        if np.any(variance[active] <= 0):
            raise ValueError("CMP root variance is zero; cannot preserve the supplied mean")
        step = np.zeros(len(means))
        step[active] = -residual[active]/variance[active]
        step = np.clip(step, -max(1., nu), max(1., nu))
        proposed = eta + step
        both = np.isfinite(lower) & np.isfinite(upper)
        outside = both & ((proposed <= lower) | (proposed >= upper)) & active
        proposed[outside] = lower[outside] + (upper[outside]-lower[outside])/2
        eta = np.where(active, proposed, eta)
        if not np.isfinite(eta).all():
            raise ValueError("CMP mean root produced a nonfinite natural parameter")
    raise ValueError("CMP mean root did not converge within numerical resource limit")


@dataclass
class CMPDistribution:
    mean: float
    nu: float
    eta: float
    log_normalizer: float
    pmf: np.ndarray
    variance: float
    metadata: dict

    def logpmf(self, k):
        """Full-support analytic log pmf; k need not belong to the finite array."""
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0:
            raise ValueError("A goal count must be a nonnegative integer")
        try:
            result = k*self.eta-self.nu*math.lgamma(k+1)-self.log_normalizer
        except (ValueError, OverflowError) as error:
            raise ValueError("CMP log pmf exceeds numerical resources") from error
        if not math.isfinite(result):
            raise ValueError("CMP log pmf is nonfinite; no probability clipping")
        return float(result)


def mean_cmp(mu, nu):
    """Return a mean-preserving CMP with explicit numerical approximation bounds.

    For nu!=1, Z is the retained adaptive series sum and pmf sums to one on that
    support. The full infinite series' log-normalizer lies within log1p(b0) above
    this value in exact arithmetic. Metadata adds an arithmetic allowance, not
    a certified bound for all floating-point roundoff or statistical error.
    No finite array is claimed to contain the whole infinite distribution.
    """
    mu, nu = _positive(mu, "mean"), _positive(nu, "nu")
    eta, lz, p, fitted, variance, b0, b1, b2, iterations = _solve(np.array([mu]), nu)
    pmf = p[0].copy()
    k = np.arange(len(pmf), dtype=float)
    grid_mean = float(pmf@k / pmf.sum())
    residual = float(fitted[0]-mu)
    # Analytic geometric bounds control truncation, not floating-point sums.
    # Keep a separate conservative arithmetic allowance; it is not an interval
    # arithmetic proof and must not be confused with statistical uncertainty.
    roundoff = 32*np.finfo(float).eps*len(pmf)*max(1., mu)
    metadata = {
        "method": "exact_poisson" if nu == 1. else "mean_root_adaptive_log_series",
        "raw_mean": mu, "nu": nu, "support_max": len(pmf)-1,
        "poisson_exact_branch": nu == 1., "root_iterations": int(iterations[0]),
        "tail_mass_upper_bound": float(b0[0]),
        "tail_first_moment_upper_bound": float(b1[0]),
        "tail_second_moment_upper_bound": float(b2[0]),
        "log_normalizer_error_upper_bound": 0. if nu == 1. else float(np.log1p(b0[0]))+roundoff,
        "mean_root_residual": residual, "mean_error_upper_bound": abs(residual)+float(b1[0])+roundoff,
        "truncated_normalized_mean": grid_mean,
        "truncated_mean_error_upper_bound": abs(grid_mean-mu)+float(b1[0])+roundoff,
        "retained_array_probability_sum": float(pmf.sum()),
        "finite_series_is_mathematical_lower_bound": nu != 1.,
        "floating_point_roundoff_allowance": roundoff,
        "error_bound_scope": "analytic_truncation_plus_heuristic_roundoff_allowance_not_interval_certified",
        "solver_mean_tolerance": MEAN_ATOL + MEAN_RTOL*mu,
        "pmf_note": "Finite adaptive support; general CMP is normalized by retained series, not an exact infinite sum.",
        "resource_limits": {"max_terms": MAX_TERMS, "max_iterations": MAX_ITERATIONS},
    }
    if not all(math.isfinite(v) for v in (float(eta[0]), float(lz[0]), float(variance[0]), grid_mean)):
        raise ValueError("CMP distribution has nonfinite numerical values")
    return CMPDistribution(mu, nu, float(eta[0]), float(lz[0]), pmf, float(variance[0]), metadata)


def batch_logpmf(means, goals, nu):
    """Vectorized exact-mean solving for same-shaped means/goals, without caching.

    The only approximation is the controlled adaptive series/root tolerance;
    no lookup table, interpolation, outcome weighting or model fallback occurs.
    All numerical/resource failures are ValueError, not silent row deletion.
    """
    means, goals = _inputs(means, goals)
    nu = _positive(nu, "nu")
    if nu == 1.:
        result = goals*np.log(means)-means-gammaln(goals+1)
    else:
        eta, lz, *_ = _solve(means.reshape(-1), nu)
        result = goals.reshape(-1)*eta-nu*gammaln(goals.reshape(-1)+1)-lz
        result = result.reshape(means.shape)
    if not np.isfinite(result).all():
        raise ValueError("CMP batch log pmf is nonfinite; no clipping or fallback")
    return np.asarray(result)


def cmp_score_forecast(home, away, nu):
    """Summarize one coherent independent-CMP grid with explicit mean/tail bounds."""
    h, a = mean_cmp(home, nu), mean_cmp(away, nu)
    matrix = np.outer(h.pmf, a.pmf)
    retained = float(matrix.sum())
    grid = FootballProbabilityGrid(matrix, h.mean, a.mean, normalize=True)
    forecast = summarize(grid)
    metadata = {
        "distribution": "independent_mean_parameterized_cmp", "nu": h.nu,
        "raw_means": {"home": h.mean, "away": a.mean},
        "marginals": {"home": h.metadata, "away": a.metadata},
        "finite_grid_mass_before_normalization": retained,
        "joint_tail_mass_upper_bound": h.metadata["tail_mass_upper_bound"]+a.metadata["tail_mass_upper_bound"],
        "grid_mean_absolute_errors": {s: abs(forecast["expected_goals"][s]-d.mean) for s,d in (("home",h),("away",a))},
        "grid_mean_error_upper_bounds": {s: d.metadata["truncated_mean_error_upper_bound"] for s,d in (("home",h),("away",a))},
        "note": "Means remain the supplied raw means; grid moments are finite-support approximations. No fitted dependence or market fusion.",
    }
    return forecast, metadata
