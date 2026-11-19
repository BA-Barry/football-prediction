"""Earlier-date closed-form Poisson rate scaling, separate from the base model.

The two fixed research methods have one shared scale or two side-specific
scales. They minimize equally weighted exact-score Poisson NLL; no market,
regularization, optimizer, grid search or probability clipping is involved.
This module does not verify the upstream provenance of supplied base rates.
"""

from collections.abc import Mapping
from datetime import date
import math
from numbers import Integral, Real

import numpy as np
from scipy.special import gammaln


METHODS = ("shared_scale", "side_scale")
MODEL_TYPE = "independent_poisson_rate_scale_calibrator"
ESTIMATION = {
    "algorithm": "closed_form",
    "objective": "equal_match_weight_independent_poisson_joint_score_nll",
    "optimizer": None, "regularization": None, "market_inputs": False,
}


def _method(method):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("method must be shared_scale or side_scale")
    return method


def _day(value, label):
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label} must be a calendar date or YYYY-MM-DD") from error
        if parsed.isoformat() == value:
            return parsed
    raise ValueError(f"{label} must be a calendar date or YYYY-MM-DD, not a timestamp")


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _rates(values):
    try:
        raw = np.asarray(values, dtype=object)
    except (ValueError, TypeError) as error:
        raise ValueError("rates must have shape (2,) or nonempty (n, 2)") from error
    if raw.ndim not in (1, 2) or raw.shape[-1] != 2 or (raw.ndim == 2 and not len(raw)):
        raise ValueError("rates must have shape (2,) or nonempty (n, 2)")
    for value in raw.flat:
        if _number(value, "rate") <= 0:
            raise ValueError("rates must be finite and strictly positive")
    return raw.astype(float)


def _score(values):
    try:
        raw = np.asarray(values, dtype=object)
    except (ValueError, TypeError) as error:
        raise ValueError("actual_score must contain two nonnegative integers") from error
    if raw.shape != (2,) or any(
            isinstance(goal, (bool, np.bool_)) or not isinstance(goal, Integral) or goal < 0
            for goal in raw):
        raise ValueError("actual_score must contain two nonnegative integers")
    for goal in raw:
        _number(goal, "goal")
    return tuple(int(goal) for goal in raw)


def _sum(values, label):
    try:
        result = math.fsum(values)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} sum is not finite") from error
    return _number(result, label)


def _scales(goal_sums, rate_sums, method):
    if method == "shared_scale":
        goals = _number(sum(goal_sums), "total goals")
        expected = _sum(rate_sums, "total rates")
        if goals == 0:
            raise ValueError("Zero target total goals: no strictly positive finite shared-scale MLE")
        scales = np.repeat(goals / expected, 2)
    else:
        if any(goals == 0 for goals in goal_sums):
            raise ValueError("Zero side target total goals: no strictly positive finite side-scale MLE")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scales = np.asarray(goal_sums, dtype=float) / rate_sums
    return _rates(scales)


def _apply(rates, scales):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        adjusted = rates * scales
    if not np.isfinite(adjusted).all() or np.any(adjusted <= 0):
        raise ValueError("Scaled rates overflowed, underflowed or became invalid; no repair applied")
    return adjusted


def _loss(goals, rates):
    with np.errstate(over="ignore", invalid="ignore"):
        terms = rates - goals * np.log(rates) + gammaln(goals + 1)
    if not np.isfinite(terms).all():
        raise ValueError("Analytic joint-score NLL is not finite; no clipping applied")
    total = _sum(terms.flat, "analytic joint-score NLL")
    if total < 0:
        raise ValueError("Computed analytic joint-score NLL is invalid")
    return {"sum": total, "mean": total / len(goals)}


def _residual_sums(goals, rates, scales):
    home = float(goals[0] - rates[0] * scales[0])
    away = float(goals[1] - rates[1] * scales[1])
    return {"home": _number(home, "home residual sum"),
            "away": _number(away, "away residual sum"),
            "shared": _sum((home, away), "shared residual sum")}


def fit_rate_calibrator(rows, before_date: date, method):
    """Fit only supplied strictly earlier rows; never drop invalid/future rows.

    A zero relevant observed-goal total places the optimum at boundary c=0,
    outside this strictly positive family, and is explicitly refused. Upstream
    scores from the fit date or later must be excluded by the caller, not here.
    """
    method = _method(method)
    if type(before_date) is not date:
        raise ValueError("before_date must be a datetime.date, not a timestamp or string")
    try:
        supplied = list(rows)
    except TypeError as error:
        raise ValueError("rows must be a nonempty iterable of mappings") from error
    if not supplied:
        raise ValueError("rows must be a nonempty iterable of mappings")
    parsed, seen = [], set()
    for row in supplied:
        if not isinstance(row, Mapping):
            raise ValueError("Each training row must be a mapping")
        key = row.get("match_id")
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("Missing or duplicate training match_id")
        seen.add(key)
        day = _day(row.get("date"), "row date")
        if day >= before_date:
            raise ValueError("Training row date must be strictly before before_date")
        rates = _rates(row.get("rates"))
        if rates.shape != (2,):
            raise ValueError("Each training row must contain one (2,) rates vector")
        parsed.append((day, key, _score(row.get("actual_score")), rates))
    parsed.sort(key=lambda item: (item[0], item[1]))
    goals = np.asarray([item[2] for item in parsed], dtype=float)
    rates = np.stack([item[3] for item in parsed])
    goal_sums = [sum(item[2][side] for item in parsed) for side in (0, 1)]
    for value in goal_sums:
        _number(value, "training goal sum")
    rate_sums = np.asarray([_sum(rates[:, side], "training rate") for side in (0, 1)])
    scales = _scales(goal_sums, rate_sums, method)
    adjusted = _apply(rates, scales)
    return {
        "model_type": MODEL_TYPE, "basis": "team_marginal", "rho": 0,
        "method": method, "parameters": {"scales": scales.tolist()},
        "before_date": before_date.isoformat(), "n_matches": len(parsed),
        "training_min_date": parsed[0][0].isoformat(), "training_max_date": parsed[-1][0].isoformat(),
        "training_match_ids": [item[1] for item in parsed],
        "training_goal_sums": goal_sums, "training_rate_sums": rate_sums.tolist(),
        "raw_training_analytic_joint_score_nll": _loss(goals, rates),
        "calibrated_training_analytic_joint_score_nll": _loss(goals, adjusted),
        "score_equation_residual_sums": _residual_sums(goal_sums, rate_sums, scales),
        "estimation": dict(ESTIMATION),
    }


def _validate_fit(fit):
    if not isinstance(fit, Mapping):
        raise ValueError("Saved rate calibrator must be a mapping")
    method = _method(fit.get("method"))
    try:
        if (fit["model_type"] != MODEL_TYPE or fit["basis"] != "team_marginal"
                or _number(fit["rho"], "rho") != 0):
            raise ValueError("Not an independent-Poisson team-marginal rate calibrator")
        cutoff = _day(fit["before_date"], "before_date")
        first = _day(fit["training_min_date"], "training_min_date")
        last = _day(fit["training_max_date"], "training_max_date")
        if not first <= last < cutoff:
            raise ValueError("Saved training dates must be ordered and precede before_date")
        n, ids = fit["n_matches"], fit["training_match_ids"]
        if (type(n) is not int or n < 1 or not isinstance(ids, list) or len(ids) != n
                or any(not isinstance(key, str) or not key.strip() for key in ids)
                or len(set(ids)) != n):
            raise ValueError("Saved training match count/IDs are invalid")
        estimation = fit["estimation"]
        if (not isinstance(estimation, Mapping)
                or any(key not in estimation or estimation[key] != value for key, value in ESTIMATION.items())
                or estimation["market_inputs"] is not False):
            raise ValueError("Need closed-form estimation without optimizer, regularization or market inputs")
        goal_sums = _score(fit["training_goal_sums"])
        rate_sums = _rates(fit["training_rate_sums"])
        scales = _rates(fit["parameters"]["scales"])
        if rate_sums.shape != (2,) or scales.shape != (2,):
            raise ValueError("Saved scales and rate sums must be (2,) vectors")
        expected = _scales(goal_sums, rate_sums, method)
        if not np.allclose(scales, expected, rtol=1e-12, atol=0):
            raise ValueError("Saved scales disagree with the declared closed-form MLE")
        if method == "shared_scale" and scales[0] != scales[1]:
            raise ValueError("shared_scale requires exactly equal home/away scales")
        residuals = _residual_sums(goal_sums, rate_sums, scales)
        saved_residuals = fit["score_equation_residual_sums"]
        for name, residual in residuals.items():
            if not math.isclose(_number(saved_residuals[name], name), residual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Saved score-equation residual sums disagree with scales")
        for field in ("raw_training_analytic_joint_score_nll", "calibrated_training_analytic_joint_score_nll"):
            loss = fit[field]
            total, mean = _number(loss["sum"], field), _number(loss["mean"], field)
            if total < 0 or mean < 0 or not math.isclose(mean, total / n, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Saved NLL sum and mean disagree with training count")
        return cutoff, scales
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Malformed saved rate calibrator: {error}") from error


def apply_rate_calibrator(rates, fit, prediction_date):
    """Apply outside the fitting period, with no training-loss application path.

    Date-only application cannot establish upstream historical availability;
    callers retain responsibility for base-prediction and source chronology.
    """
    values = _rates(rates)
    cutoff, scales = _validate_fit(fit)
    if _day(prediction_date, "prediction_date") < cutoff:
        raise ValueError("Application date must be on/after before_date, never within training")
    return _apply(values, scales)
