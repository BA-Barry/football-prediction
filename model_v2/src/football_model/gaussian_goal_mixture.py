"""Deterministic Poisson/log-Gaussian predictive integration, without fitting.

Conditional on two correlated Gaussian log rates, home/away goals are
independent Poisson.  Markets integrate that conditional distribution, rather
than substituting Poisson(E[rate]).  A Gaussian supplied by a Laplace fit plus
finite Gauss--Hermite integration is an approximation, not MCMC or a full
Bayesian posterior.  The caller may compare fixed orders 15 and 25; this module
does not select an order from results or automatically change it.
"""

import math
from numbers import Integral

import numpy as np
from scipy.special import gammaln, logsumexp, roots_hermitenorm
from scipy.stats import poisson, skellam


METRICS = ("full_score_nll", "one_x_two_log_loss", "one_x_two_brier", "top1",
           "total_log_loss", "total_brier", "away_minus_two_log_loss", "away_minus_two_brier")
_OUTCOMES = ("home", "draw", "away")
_TOTALS = ("0", "1", "2", "3", "4+")
_HANDICAP = ("win", "draw", "lose")


def _inputs(log_mean, log_cov, order):
    if not isinstance(order, Integral) or isinstance(order, (bool, np.bool_)) or order < 1:
        raise ValueError("A positive integer Gauss-Hermite order is required")
    try:
        mean, raw_cov = np.asarray(log_mean, dtype=float), np.asarray(log_cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Two finite log means and a finite 2 by 2 covariance are required") from exc
    if mean.shape != (2,) or raw_cov.shape != (2, 2):
        raise ValueError("Two log means and a 2 by 2 log-rate covariance are required")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(raw_cov)):
        raise ValueError("Finite log means and covariance are required")
    # Only floating-point roundoff is repaired, not a materially invalid model.
    tolerance = float(64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(raw_cov)))))
    symmetry_error = float(np.max(np.abs(raw_cov - raw_cov.T)))
    if symmetry_error > tolerance:
        raise ValueError("Log-rate covariance must be symmetric")
    cov = (raw_cov + raw_cov.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("Log-rate covariance must be positive semidefinite; no PSD clipping is permitted")
    negative = eigenvalues[eigenvalues < 0]
    nonnegative = np.maximum(eigenvalues, 0.0)
    if len(negative):
        cov = (eigenvectors * nonnegative) @ eigenvectors.T
        cov = (cov + cov.T) / 2
    # The principal symmetric square root preserves home/away exchange
    # equivariance of the finite product quadrature (unlike an arbitrary factor).
    square_root = (eigenvectors * np.sqrt(nonnegative)) @ eigenvectors.T
    square_root = (square_root + square_root.T) / 2
    diagnostic = {"input_log_cov": raw_cov.tolist(), "roundoff_tolerance": tolerance,
                  "symmetry_max_abs_error": symmetry_error,
                  "negative_eigenvalues_zeroed_for_roundoff": negative.tolist()}
    return mean, cov, square_root, int(order), diagnostic


def _normal_quadrature(log_mean, log_cov, order):
    mean, cov, square_root, order, diagnostic = _inputs(log_mean, log_cov, order)
    nodes, weights = roots_hermitenorm(order)
    weights = weights / np.sum(weights)
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    standard_nodes = np.column_stack((first.ravel(), second.ravel()))
    weights = np.outer(weights, weights).ravel()
    weights = weights / np.sum(weights)
    log_rates = mean + standard_nodes @ square_root.T
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rates = np.exp(log_rates)
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("Quadrature goal rates exceed the positive finite floating-point domain")
    return mean, cov, log_rates, rates, weights, diagnostic


def _normalized(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Conditional market probabilities are nonfinite or negative")
    mass = np.sum(values, axis=-1, keepdims=True)
    if np.any(np.abs(mass - 1) > 1e-9):
        raise ValueError("Conditional market probabilities fail numerical normalization")
    # Correct only summation/conditional special-function roundoff, not tails.
    return values / mass


def _moments(mean, cov):
    log_expected = mean + np.diag(cov) / 2
    try:
        expected = [math.exp(float(value)) for value in log_expected]
        extra_variance = [0.0 if cov[i, i] == 0 else
                          math.exp(2 * float(log_expected[i]) + math.log(math.expm1(float(cov[i, i]))))
                          for i in range(2)]
        cross = (0.0 if cov[0, 1] == 0 else
                 math.exp(float(sum(log_expected))) * math.expm1(float(cov[0, 1])))
        variance = [mu + extra for mu, extra in zip(expected, extra_variance)]
    except (OverflowError, ValueError) as exc:
        raise ValueError("Analytic predictive moments exceed the finite floating-point domain") from exc
    if any(not math.isfinite(v) or v <= 0 for v in expected + variance) or not math.isfinite(cross):
        raise ValueError("Analytic predictive moments exceed the finite floating-point domain")
    return expected, variance, cross


def _evaluate(log_mean, log_cov, order):
    mean, cov, log_rates, rates, weights, diagnostic = _normal_quadrature(log_mean, log_cov, order)
    home, away = rates.T
    # Reflection avoids computing an upper tail by subtracting a CDF from one.
    outcomes = _normalized(np.column_stack((skellam.cdf(-1, away, home),
                                            skellam.pmf(0, home, away),
                                            skellam.cdf(-1, home, away))))
    totals_rate = home + away
    totals = _normalized(np.column_stack([poisson.pmf(k, totals_rate) for k in range(4)]
                                         + [poisson.sf(3, totals_rate)]))
    handicap = _normalized(np.column_stack((skellam.cdf(-3, home, away),
                                            skellam.pmf(2, away, home),
                                            skellam.cdf(1, away, home))))
    probabilities = _normalized(weights @ outcomes)
    # These equalities follow from exchange symmetry, not an approximate-tie rule.
    if mean[0] == mean[1] and cov[0, 0] == cov[1, 1]:
        probabilities[0] = probabilities[2] = (probabilities[0] + probabilities[2]) / 2
    total_probabilities, handicap_probabilities = _normalized(weights @ totals), _normalized(weights @ handicap)
    expected, variance, cross = _moments(mean, cov)
    result = {
        "forecast": {"one_x_two": dict(zip(_OUTCOMES, map(float, probabilities))),
                     "expected_goals": {"home": expected[0], "away": expected[1], "total": sum(expected)},
                     "total_goals": dict(zip(_TOTALS, map(float, total_probabilities))),
                     "away_minus_two_jincai": dict(zip(_HANDICAP, map(float, handicap_probabilities)))},
        "log_mean": mean.tolist(), "log_cov": cov.tolist(), "order": int(order),
        "goal_variance": {"home": variance[0], "away": variance[1]},
        "goal_cross_covariance": cross,
        "quadrature": {"method": "2D standard-normal probabilists Gauss-Hermite",
                       "nodes": len(weights), "weight_sum": float(sum(weights))},
        "covariance_roundoff": diagnostic,
        "approximation": "Gaussian log-rate (e.g. Laplace) + finite Gauss-Hermite; not MCMC or full Bayes",
    }
    return result, log_rates, rates, weights


def forecast(log_mean, log_cov, order=15):
    """Integrate whole-match markets for a Gaussian pair of Poisson log rates.

    ``log_cov`` may be singular or zero.  Material asymmetry/non-PSD inputs are
    rejected; only explicitly reported floating-point roundoff is corrected.
    Away-minus-two is Jincai three-way: net away goals >=3 win, 2 draw, <=1 lose.
    """
    return _evaluate(log_mean, log_cov, order)[0]


def _categorical(probabilities, actual):
    if probabilities[actual] <= 0:
        raise ValueError("Observed categorical outcome has zero representable quadrature probability")
    return (-math.log(probabilities[actual]),
            sum((p - int(i == actual)) ** 2 for i, p in enumerate(probabilities)))


def score(log_mean, log_cov, actual, order=15):
    """Return the existing eight metrics, integrating the actual joint PMF.

    Joint-score NLL uses log-sum-exp and remains finite for rare scores whose
    joint probability underflows.  No probability floor or realized-score-based
    quadrature order selection is used.
    """
    if (not isinstance(actual, (list, tuple, np.ndarray)) or np.asarray(actual, dtype=object).shape != (2,)
            or any(not isinstance(g, Integral) or isinstance(g, (bool, np.bool_)) or g < 0 for g in actual)):
        raise ValueError("Two nonnegative integer actual goals are required")
    result, log_rates, rates, weights = _evaluate(log_mean, log_cov, order)
    actual = np.asarray(actual, dtype=float)
    conditional_log_pmf = np.sum(actual * log_rates - rates - gammaln(actual + 1), axis=1)
    with np.errstate(divide="ignore"):
        nll = -float(logsumexp(np.log(weights) + conditional_log_pmf))
    if not math.isfinite(nll):
        raise ValueError("Actual score exceeds the finite joint-log-probability domain")
    value = result["forecast"]
    p = [value["one_x_two"][key] for key in _OUTCOMES]
    outcome = 0 if actual[0] > actual[1] else 1 if actual[0] == actual[1] else 2
    loss, brier = _categorical(p, outcome)
    tied = [i for i, probability in enumerate(p) if probability == max(p)]
    total_loss, total_brier = _categorical([value["total_goals"][key] for key in _TOTALS],
                                          min(int(sum(actual)), 4))
    handicap_actual = 0 if actual[1] - actual[0] > 2 else 1 if actual[1] - actual[0] == 2 else 2
    handicap_loss, handicap_brier = _categorical([value["away_minus_two_jincai"][key] for key in _HANDICAP],
                                                handicap_actual)
    result.update(full_score_nll=nll, one_x_two_log_loss=loss, one_x_two_brier=brier,
                  top1=1 / len(tied) if outcome in tied else 0.0, top1_tied=len(tied) > 1,
                  total_log_loss=total_loss, total_brier=total_brier,
                  away_minus_two_log_loss=handicap_loss, away_minus_two_brier=handicap_brier)
    return result
