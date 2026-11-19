"""Research-only common-shock bivariate Poisson with unchanged raw means.

H=U+W and A=V+W, with independent Poisson components and
c=E[W]=k*min(mu_H,mu_A).  The shared k is not a Pearson correlation.
This candidate supports nonnegative dependence only, not all possible score
dependence.  No CMP adjustment or production-default replacement occurs here.
"""

import math
from numbers import Integral, Real

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from .vendor.penaltyblog_grid import FootballProbabilityGrid


MAX_K = .95
MAX_GRID_GOAL = 512
MAX_SUM_TERMS = 100_000
MAX_BATCH_TERMS = 2_000_000
TAIL_QUANTILE = 1 - 1e-12
MAX_OMITTED_MASS = 1e-8


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} exceeds finite numerical resources") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _parameter(k):
    k = _real(k, "k")
    if not 0 <= k <= MAX_K:
        raise ValueError(f"k must lie in [0, {MAX_K}]; negative dependence is not represented")
    return k


def _goal(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Goals must be nonnegative integers")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Goals exceed finite numerical resources") from error
    # Float counts must remain exact integers; no silent conversion of huge ints.
    if not math.isfinite(result) or int(result) != value:
        raise ValueError("Goals exceed exactly representable numerical resources")
    return result


def _components(home, away, k):
    common = k * np.minimum(home, away)
    u, v = home - common, away - common
    if (not np.isfinite(common).all() or np.any(u <= 0) or np.any(v <= 0)
            or (k > 0 and np.any(common <= 0))):
        raise ValueError("Poisson component means exceed numerical resources")
    return u, v, common


def batch_logpmf(home_goals, away_goals, home_mean, away_mean, k):
    """Full-support joint log probabilities; all four inputs have the same shape.

    The finite sum runs through j=0,...,min(h,a), not through a score grid.
    A finite resource failure raises ValueError; no rows, probabilities, or
    dependence parameters are clipped or replaced by another family.
    """
    k = _parameter(k)
    raw = [np.asarray(value, dtype=object)
           for value in (home_goals, away_goals, home_mean, away_mean)]
    shape = raw[0].shape
    if not raw[0].size:
        raise ValueError("Need nonempty count/mean inputs")
    if any(value.shape != shape for value in raw):
        raise ValueError("Counts and means must have identical shapes; no broadcasting")
    h, a = [np.array([_goal(value) for value in values.flat]).reshape(shape)
            for values in raw[:2]]
    home, away = [np.array([_real(value, "mean", positive=True) for value in values.flat]).reshape(shape)
                  for values in raw[2:]]
    u, v, common = _components(home, away, k)
    if k == 0:
        result = h*np.log(home)-home-gammaln(h+1) + a*np.log(away)-away-gammaln(a+1)
    else:
        terms = int(np.minimum(h, a).max()) + 1
        if terms > MAX_SUM_TERMS or terms * h.size > MAX_BATCH_TERMS:
            raise ValueError("Bivariate Poisson finite sum exceeds numerical resource limit")
        j = np.arange(terms, dtype=float)[None, :]
        left, right = h.reshape(-1, 1)-j, a.reshape(-1, 1)-j
        allowed = (left >= 0) & (right >= 0)
        # Zero padding is only for unsupported j terms, which are -inf exactly.
        left = np.where(allowed, left, 0.)
        right = np.where(allowed, right, 0.)
        summands = (left*np.log(u.reshape(-1, 1)) - gammaln(left+1)
                    + right*np.log(v.reshape(-1, 1)) - gammaln(right+1)
                    + j*np.log(common.reshape(-1, 1)) - gammaln(j+1))
        summands = np.where(allowed, summands, -np.inf)
        result = (logsumexp(summands, axis=1) - u.reshape(-1) - v.reshape(-1)
                  - common.reshape(-1)).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Bivariate Poisson log pmf is nonfinite; no clipping or fallback")
    return np.asarray(result)


def logpmf(home_goals, away_goals, home_mean, away_mean, k):
    """Scalar full-support log pmf, including observed scores beyond grid support."""
    for value in (home_goals, away_goals, home_mean, away_mean):
        if np.asarray(value).shape != ():
            raise ValueError("logpmf requires scalar counts and means; use batch_logpmf")
    return float(batch_logpmf(home_goals, away_goals, home_mean, away_mean, k))


def score_grid(home_mean, away_mean, k, *, max_goal=None):
    """Return a normalized grid and explicit Poisson-marginal truncation metadata.

    The union bound controls omitted probability despite dependence.  Analytic
    first-moment bounds use E[W|A]=c*A/mu_A (and its symmetric counterpart).
    Reported arithmetic allowances are heuristic, not interval certificates.
    """
    home = _real(home_mean, "home_mean", positive=True)
    away = _real(away_mean, "away_mean", positive=True)
    k = _parameter(k)
    u, v, common = map(float, _components(np.asarray(home), np.asarray(away), k))
    if max_goal is None:
        quantile = float(poisson.ppf(TAIL_QUANTILE, max(home, away)))
        if not math.isfinite(quantile) or quantile > MAX_GRID_GOAL:
            raise ValueError("Poisson score support exceeds grid numerical resource limit")
        max_goal = max(15, int(quantile))
    if (isinstance(max_goal, (bool, np.bool_)) or not isinstance(max_goal, Integral)
            or not 1 <= max_goal <= MAX_GRID_GOAL):
        raise ValueError(f"max_goal must be an integer in [1, {MAX_GRID_GOAL}] within resource limits")
    max_goal = int(max_goal)
    tails = {"home": float(poisson.sf(max_goal, home)),
             "away": float(poisson.sf(max_goal, away))}
    bound = sum(tails.values())
    if not math.isfinite(bound) or bound > MAX_OMITTED_MASS:
        raise ValueError("Score truncation would omit material probability")
    goals = np.arange(max_goal+1)
    if k == 0:
        matrix = np.outer(poisson.pmf(goals, home), poisson.pmf(goals, away))
    else:
        # Convolution of independent components is nonnegative term by term.
        pu, pv, pw = [poisson.pmf(goals, mean) for mean in (u, v, common)]
        matrix = np.zeros((max_goal+1, max_goal+1))
        for j in range(max_goal+1):
            length = max_goal+1-j
            matrix[j:, j:] += pw[j]*np.outer(pu[:length], pv[:length])
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Bivariate Poisson grid contains invalid probabilities")
    retained = float(matrix.sum())
    arithmetic_mass_allowance = 32*np.finfo(float).eps*(max_goal+1)**2
    if not 1-bound-arithmetic_mass_allowance <= retained <= 1+arithmetic_mass_allowance:
        raise ValueError("Bivariate Poisson grid mass disagrees with the marginal tail bound")
    grid = FootballProbabilityGrid(matrix, home, away, normalize=True)
    finite_means = {"home": float(np.sum(grid.grid*goals[:, None])),
                    "away": float(np.sum(grid.grid*goals[None, :]))}
    # E[H; omitted] <= E[H;H>M] + E[H;A>M], each term available analytically.
    shifted_h, shifted_a = float(poisson.sf(max_goal-1, home)), float(poisson.sf(max_goal-1, away))
    omitted_first = {"home": home*shifted_h + u*tails["away"] + common*shifted_a,
                     "away": away*shifted_a + v*tails["home"] + common*shifted_h}
    raw_means = {"home": home, "away": away}
    analytic_mean_bounds = {side: (raw_means[side]*bound+omitted_first[side])/(1-bound)
                            for side in raw_means}
    mean_arithmetic_allowance = arithmetic_mass_allowance*max(1., max_goal, home, away)
    metadata = {
        "distribution": "bivariate_poisson_common_shock", "k": k,
        "raw_means": raw_means, "common_mean": common,
        "component_means": {"home_independent": u, "away_independent": v, "common": common},
        "pearson_correlation": common/math.sqrt(home)/math.sqrt(away),
        "support_max": max_goal, "poisson_marginal_tail_probabilities": tails,
        "omitted_mass_bound": bound,
        "finite_grid_mass_before_normalization": retained,
        "omitted_first_moment_upper_bounds": omitted_first,
        "finite_grid_means": finite_means,
        "finite_grid_mean_error": {side: abs(finite_means[side]-raw_means[side]) for side in raw_means},
        "analytic_truncation_mean_error_bounds": analytic_mean_bounds,
        "finite_grid_mean_error_upper_bounds": {side: analytic_mean_bounds[side]+mean_arithmetic_allowance
                                                 for side in raw_means},
        "floating_point_mass_roundoff_allowance": arithmetic_mass_allowance,
        "floating_point_mean_roundoff_allowance": mean_arithmetic_allowance,
        "error_bound_scope": "analytic_truncation_plus_heuristic_roundoff_allowance_not_interval_certified",
        "parameter_note": "k is a shared fraction of min(raw means), not Pearson correlation; nonnegative dependence only.",
        "resource_limits": {"max_grid_goal": MAX_GRID_GOAL, "max_sum_terms": MAX_SUM_TERMS,
                            "max_batch_terms": MAX_BATCH_TERMS},
    }
    return grid, metadata
