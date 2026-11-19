"""Small, auditable post-hoc calibrators for strictly earlier 1X2 forecasts.

These transformations affect only the three outcome probabilities. They do not
define a new score grid, goal distribution, or player-strength model. The caller
must separately enforce the chronology of the base forecasts and applications.
No market probabilities enter either fit.
"""

from collections.abc import Mapping
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


METHODS = ("class_bias", "temperature")
OPTIMIZER_OPTIONS = {"gtol": 1e-9, "ftol": 1e-13, "maxiter": 1000}


def _method(value):
    if not isinstance(value, str) or value not in METHODS:
        raise ValueError("method must be class_bias or temperature")
    return value


def _date(value, name):
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date YYYY-MM-DD") from exc
        if parsed.isoformat() == value:
            return parsed
    raise ValueError(f"{name} must be a date, not a timestamp, or an ISO date YYYY-MM-DD")


def _probabilities(values):
    """Validate without clipping, filling, or silently normalizing inputs."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must have shape (3,) or (n, 3)") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError("probabilities must be real numeric values")
    p = raw.astype(float)
    if p.ndim not in (1, 2) or p.shape[-1] != 3 or (p.ndim == 2 and not len(p)):
        raise ValueError("probabilities must have shape (3,) or nonempty (n, 3)")
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite, strictly positive and at most one")
    if not np.allclose(p.sum(axis=-1), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("probabilities must sum to one; inputs are not normalized")
    return p


def _numeric_parameter(values, shape, name):
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite real values of shape {shape}") from exc
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must be finite real values of shape {shape}")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _log_softmax(logits):
    """Subtract the maximum before logsumexp to avoid large-offset cancellation."""
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_q = shifted - logsumexp(shifted, axis=-1, keepdims=True)
    if not np.isfinite(log_q).all():
        raise ValueError("calibration produced non-finite log probabilities")
    return log_q


def _class_bias_objective(theta, log_p, labels):
    log_q = _log_softmax(log_p + np.array([theta[0], theta[1], 0.0]))
    q = np.exp(log_q)
    loss = float(-log_q[np.arange(len(labels)), labels].mean())
    gradient = (q.mean(axis=0) - np.bincount(labels, minlength=3) / len(labels))[:2]
    return loss, gradient


def _temperature_objective(theta, log_p, labels):
    with np.errstate(over="ignore", invalid="ignore"):
        logits = theta[0] * log_p
    log_q = _log_softmax(logits)
    q = np.exp(log_q)
    loss = float(-log_q[np.arange(len(labels)), labels].mean())
    gradient = np.array([(np.sum(q * log_p, axis=1) - log_p[np.arange(len(labels)), labels]).mean()])
    return loss, gradient


def apply_calibrator(probabilities, parameters):
    """Apply a saved fit, preserving a single-row or matrix input's shape.

    The complete fit dictionary is accepted. Only its declared ``method`` and
    ``parameters`` are used; temporal applicability is the caller's job. Extreme
    settings that cannot be represented without zero/invalid probabilities are
    reported explicitly rather than clipped or replaced with the raw forecast.
    """
    p = _probabilities(probabilities)
    if not isinstance(parameters, Mapping):
        raise ValueError("calibrator must be a mapping declaring method and parameters")
    method = _method(parameters.get("method"))
    fitted = parameters.get("parameters")
    if not isinstance(fitted, Mapping):
        raise ValueError("calibrator parameters must be a mapping")
    log_p = np.log(p)
    if method == "class_bias":
        bias = _numeric_parameter(fitted.get("bias"), (3,), "bias")
        if bias[2] != 0.0:
            raise ValueError("class_bias requires the away bias fixed at zero")
        with np.errstate(over="ignore", invalid="ignore"):
            logits = log_p + bias
    else:
        beta = float(_numeric_parameter(fitted.get("beta"), (), "beta"))
        if beta < 0:
            raise ValueError("beta must be non-negative")
        # Temperature is redundant metadata, but never accept a contradictory or
        # non-finite declaration. beta=0 represents the supported uniform limit.
        if "temperature" in fitted:
            temperature = fitted["temperature"]
            if beta == 0:
                if temperature is not None:
                    raise ValueError("temperature must be null when beta is zero")
            else:
                temperature = float(_numeric_parameter(temperature, (), "temperature"))
                if temperature <= 0 or not np.isclose(temperature * beta, 1.0, rtol=1e-10, atol=1e-10):
                    raise ValueError("temperature must be the positive reciprocal of beta")
        with np.errstate(over="ignore", invalid="ignore"):
            logits = beta * log_p
    q = np.exp(_log_softmax(logits))
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("calibration probabilities underflowed or became non-finite; no clipping was applied")
    return q


def fit_calibrator(rows, before_date: date, method: str):
    """Fit mean multiclass NLL using only explicitly supplied earlier forecasts.

    Future or duplicate rows are errors, not silently removed. All three labels
    must be present. L-BFGS-B uses analytic gradients, no regularization, tuning
    grid, outcome weighting, or market inputs. Optimizer success is recorded as
    a numerical termination status, not evidence of future calibration quality.
    """
    method = _method(method)
    if type(before_date) is not date:
        raise ValueError("before_date must be a datetime.date, not a timestamp or string")
    try:
        supplied = list(rows)
    except TypeError as exc:
        raise ValueError("rows must be a nonempty iterable of forecast mappings") from exc
    if not supplied:
        raise ValueError("rows must be a nonempty iterable of forecast mappings")
    parsed, seen = [], set()
    for row in supplied:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a forecast mapping")
        match_id = row.get("match_id")
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError("each match_id must be a nonempty string")
        if match_id in seen:
            raise ValueError(f"duplicate match_id: {match_id}")
        seen.add(match_id)
        day = _date(row.get("date"), "row date")
        if day >= before_date:
            raise ValueError(f"row date must be strictly before before_date: {match_id}")
        outcome = row.get("actual_outcome")
        if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, (int, np.integer)) or outcome not in (0, 1, 2):
            raise ValueError("actual_outcome must be an integer label 0, 1 or 2")
        probabilities = _probabilities(row.get("probabilities"))
        if probabilities.shape != (3,):
            raise ValueError("each row must contain one (3,) probability vector")
        parsed.append((day, match_id, probabilities, int(outcome)))
    parsed.sort(key=lambda item: (item[0], item[1]))
    p = np.stack([item[2] for item in parsed])
    labels = np.array([item[3] for item in parsed], dtype=int)
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("all three observed outcome classes must be present")
    log_p = np.log(p)
    objective = _class_bias_objective if method == "class_bias" else _temperature_objective
    initial = np.zeros(2) if method == "class_bias" else np.ones(1)
    bounds = None if method == "class_bias" else [(0.0, None)]
    raw_log_loss = float(-log_p[np.arange(len(labels)), labels].mean())
    try:
        fit = minimize(objective, initial, args=(log_p, labels), jac=True,
                       method="L-BFGS-B", bounds=bounds, options=dict(OPTIMIZER_OPTIONS))
    except Exception as exc:
        raise RuntimeError(f"{method} calibration optimizer failed: {exc}") from exc
    if not fit.success:
        raise RuntimeError(f"{method} calibration optimizer failed (status={fit.status}): {fit.message}")
    try:
        coefficients = _numeric_parameter(fit.x, initial.shape, "optimizer coefficients")
        _numeric_parameter(fit.fun, (), "optimizer objective")
        _numeric_parameter(fit.jac, initial.shape, "optimizer gradient")
        if method == "temperature" and coefficients[0] < 0:
            raise ValueError("optimizer beta must be non-negative")
        final_objective, gradient = objective(coefficients, log_p, labels)
        if not np.isfinite(final_objective) or not np.isfinite(gradient).all():
            raise ValueError("optimizer recomputed objective or gradient is not finite")
        if method == "class_bias":
            fitted = {"bias": [float(coefficients[0]), float(coefficients[1]), 0.0]}
        else:
            beta = float(coefficients[0])
            temperature = 1.0 / beta if beta > 0 else None
            if temperature is not None and not np.isfinite(temperature):
                raise ValueError("optimizer reciprocal temperature is not finite")
            fitted = {"beta": beta, "temperature": temperature}
        apply_calibrator(p, {"method": method, "parameters": fitted})
    except (ValueError, FloatingPointError) as exc:
        raise RuntimeError(f"{method} calibration optimizer returned an invalid fit: {exc}") from exc
    projected_gradient = gradient.copy()
    if method == "temperature" and coefficients[0] == 0:
        projected_gradient[0] = min(float(gradient[0]), 0.0)
    return {
        "method": method,
        "parameters": fitted,
        "before_date": before_date.isoformat(),
        "n_matches": len(parsed),
        "training_min_date": parsed[0][0].isoformat(),
        "training_max_date": parsed[-1][0].isoformat(),
        "training_match_ids": [item[1] for item in parsed],
        "class_counts": counts.tolist(),
        "raw_training_log_loss": raw_log_loss,
        "calibrated_training_log_loss": float(final_objective),
        "raw_training_nll": float(raw_log_loss * len(parsed)),
        "calibrated_training_nll": float(final_objective * len(parsed)),
        "optimization": {
            "algorithm": "L-BFGS-B",
            "objective": "mean_multiclass_negative_log_likelihood",
            "regularization": None,
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "function_evaluations": int(fit.nfev),
            "gradient_evaluations": int(fit.njev),
            "initial_objective": float(objective(initial, log_p, labels)[0]),
            "final_objective": float(final_objective),
            "gradient": gradient.tolist(),
            "gradient_norm_inf": float(np.max(np.abs(gradient))),
            "projected_gradient_norm_inf": float(np.max(np.abs(projected_gradient))),
            "options": dict(OPTIMIZER_OPTIONS),
        },
    }
