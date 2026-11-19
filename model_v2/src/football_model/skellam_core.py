"""Stable Skellam log likelihood, log-rate scores and finite support tables.

The two positive rates parameterize a NET-goal distribution. Fitting this
distribution alone does not identify them as real expected goals. Tables keep
their original probability mass and report both omitted tails separately.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.special import ive
from scipy.stats import skellam


_SERIES_RELATIVE_TOLERANCE = 2e-16
_MAX_SERIES_TERMS = 10000
_MAX_TABLE_POINTS = 1000001


def _numeric(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not booleans or strings")
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _rates(log_home, log_away):
    lh, la = np.broadcast_arrays(_numeric(log_home, "log_home"), _numeric(log_away, "log_away"))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        home, away = np.exp(lh), np.exp(la)
    if not (np.all(np.isfinite(home)) and np.all(np.isfinite(away))
            and np.all(home > 0) and np.all(away > 0)):
        raise FloatingPointError("Log rates must yield finite, strictly positive floating-point rates")
    return lh, la, home, away


def _log_positive_series(order, log_q):
    """Log of 0F1(order+1;q), with an explicit positive remaining-tail bound."""
    log_term = 0.0
    log_sum = 0.0
    log_tolerance = math.log(_SERIES_RELATIVE_TOLERANCE)
    for k in range(_MAX_SERIES_TERMS):
        # All subsequent term ratios decrease, so a geometric upper bound
        # controls the remaining relative sum once this ratio is below one.
        log_ratio = log_q - math.log(k + 1) - math.log(order + k + 1)
        if log_ratio < 0:
            ratio = math.exp(log_ratio)
            log_remaining = log_term + log_ratio - math.log1p(-ratio)
            if log_remaining - log_sum <= log_tolerance:
                return log_sum
        log_term += log_ratio
        log_sum = float(np.logaddexp(log_sum, log_term))
    raise FloatingPointError("Skellam positive series did not reach its numerical tail tolerance")


def _series_logp_and_common(z, lh, la, home, away):
    order = abs(z)
    log_q = lh + la
    log_s0 = _log_positive_series(order, log_q)
    log_s1 = _log_positive_series(order + 1, log_q)
    logp = (-home - away + max(z, 0) * lh + max(-z, 0) * la
            - math.lgamma(order + 1) + log_s0)
    # c = E[min(U,V) | U-V=z]. It is not E[V|z] when z is negative.
    log_common = log_q - math.log(order + 1) + log_s1 - log_s0
    try:
        common = math.exp(log_common)
    except OverflowError as error:
        raise FloatingPointError("Skellam conditional moment overflow") from error
    if not math.isfinite(logp) or not math.isfinite(common):
        raise FloatingPointError("Skellam positive series produced a nonfinite result")
    return logp, common


def log_probability_and_lograte_score(z, log_home, log_away):
    """Return log p(z), d log p/d log_home and d log p/d log_away.

    Inputs broadcast using NumPy rules. Scalars produce three Python floats;
    non-scalar inputs produce float arrays with the broadcast shape. No rate,
    likelihood, score or probability is clipped. Unrepresentable rates and
    failed special-function evaluations raise explicit errors.
    """
    lh, la, home, away = _rates(log_home, log_away)
    z, lh, la, home, away = np.broadcast_arrays(_numeric(z, "z"), lh, la, home, away)
    if not np.all(z == np.floor(z)) or not np.all(np.abs(z) < 2**52):
        raise ValueError("Net goals must be exactly representable integers")
    order = np.abs(z)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        s = np.exp((lh + la) / 2)
        argument = 2 * s
        denominator = ive(order, argument)
        numerator = ive(order + 1, argument)
    tiny = np.finfo(float).tiny
    # At small arguments ive(0, x) can round to one, losing the -x term
    # needed when a near-certain draw has a very small, nonzero log mass.
    # The positive series is short throughout this additional domain.
    usable = ((argument >= 0.5)
              & np.isfinite(argument) & np.isfinite(denominator) & np.isfinite(numerator)
              & (denominator >= tiny) & (numerator >= tiny))
    logp = np.empty(z.shape, dtype=float)
    common = np.empty(z.shape, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        logp[usable] = (-(np.sqrt(home[usable]) - np.sqrt(away[usable]))**2
                       + z[usable] / 2 * (lh[usable] - la[usable])
                       + np.log(denominator[usable]))
        common[usable] = s[usable] * (numerator[usable] / denominator[usable])
    for index in zip(*np.nonzero(~usable)) if z.ndim else ([()] if not usable.item() else []):
        # At high rates an unavailable Bessel evaluation cannot be rescued by
        # a short small-q series. Report that numerical domain failure instead
        # of iterating a huge sum or substituting a different distribution.
        if float(lh[index] + la[index]) > math.log(1e12):
            raise FloatingPointError("Scaled Bessel evaluation unavailable at requested Skellam rates")
        logp[index], common[index] = _series_logp_and_common(
            int(z[index]), float(lh[index]), float(la[index]), float(home[index]), float(away[index]))
    gh = np.maximum(z, 0) + common - home
    ga = np.maximum(-z, 0) + common - away
    if not all(np.all(np.isfinite(value)) for value in (logp, gh, ga)):
        raise FloatingPointError("Skellam log likelihood or log-rate score is nonfinite")
    if np.any(logp > 1e-12):
        raise FloatingPointError("Skellam log probability exceeds its valid range")
    return (float(logp), float(gh), float(ga)) if z.ndim == 0 else (logp, gh, ga)


def _probability(value, label):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise FloatingPointError(f"Skellam {label} evaluation is not a finite probability")
    return value


def net_goal_probabilities(log_home, log_away, tail_tolerance=1e-10):
    """Describe one fixture using H/D/A and an adaptive, unnormalized net table.

    left_tail is P(D < first listed net goal), right_tail is P(D > last).
    Their sum is evaluated using CDF/SF, rather than 1 minus rounded table
    masses. These are floating-point distribution evaluations, not exact-real
    arithmetic. Scalar log rates are required for this variable-width table.
    """
    lh, la, home, away = _rates(log_home, log_away)
    if lh.ndim != 0 or la.ndim != 0:
        raise ValueError("A net-goal table requires scalar log rates for one fixture")
    if (isinstance(tail_tolerance, bool) or not isinstance(tail_tolerance, Real)
            or not math.isfinite(tail_tolerance) or not 0 < tail_tolerance < 1):
        raise ValueError("tail_tolerance must lie strictly between zero and one")
    lh, la, home, away = map(float, (lh, la, home, away))
    log_draw, _, _ = log_probability_and_lograte_score(0, lh, la)
    # Reflection evaluates survival tails through a CDF directly. SciPy's
    # generic Skellam SF subtracts the CDF from one and can erase small tails.
    hda = [_probability(skellam.cdf(-1, away, home), "home-win tail"),
           _probability(math.exp(log_draw), "draw mass"),
           _probability(skellam.cdf(-1, home, away), "away-win tail")]
    if abs(math.fsum(hda) - 1) > 1e-10:
        raise FloatingPointError("Skellam H/D/A masses do not sum to one within numerical tolerance")
    mean, variance = home - away, home + away
    if not math.isfinite(variance):
        raise FloatingPointError("Skellam net-goal variance overflow")
    width = max(8.0, 6 * math.sqrt(variance))
    lower, upper = math.floor(mean - width), math.ceil(mean + width)
    for _ in range(60):
        if upper - lower + 1 > _MAX_TABLE_POINTS or max(abs(lower), abs(upper)) >= 2**52:
            raise FloatingPointError("Requested Skellam support table exceeds the supported numerical size")
        left = _probability(skellam.cdf(lower - 1, home, away), "left omitted tail")
        right = _probability(skellam.cdf(-upper - 1, away, home), "right omitted tail")
        if math.fsum((left, right)) <= tail_tolerance:
            break
        step = max(1, math.ceil(width))
        if left > tail_tolerance / 2:
            lower -= step
        if right > tail_tolerance / 2:
            upper += step
        width *= 2
    else:
        raise FloatingPointError("Skellam support search did not reach the requested tail tolerance")
    net_goals = np.arange(lower, upper + 1, dtype=np.int64)
    log_mass, _, _ = log_probability_and_lograte_score(net_goals, lh, la)
    probabilities = np.exp(log_mass)
    full_mass = math.fsum(probabilities.tolist() + [left, right])
    if abs(full_mass - 1) > 1e-10:
        raise FloatingPointError("Skellam support masses and independently evaluated tails disagree")
    return {"hda": hda, "net_goals": net_goals.tolist(),
            "probabilities": probabilities.tolist(), "tail_mass": math.fsum((left, right)),
            "left_tail": left, "right_tail": right}
