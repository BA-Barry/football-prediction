"""Fit one shared CMP dispersion on earlier, unchanged team-model means.

This is a distribution-only research fit: no team training, market fusion,
later-season loading or probability calibration occurs here. The caller must
replay the original 180-day/alpha=.01 source coefficients before supplying the
standard prediction rows; row-only input cannot independently prove provenance.
"""

from datetime import date
import math
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from .forecast_diagnostics import validate_prediction
from .mean_cmp import batch_logpmf
from .probability import score_grid, summarize
from .refresh_research import _same


NU_BOUNDS = (.25, 4.)
VALID_START = date(2024, 8, 1)
VALID_END = date(2025, 7, 1)
FIT_BEFORE = "2025-08-01"
XATOL = 1e-4
MAXITER = 80


class CMPFitError(RuntimeError):
    """A failed numerical fit, including its trace; never a Poisson fallback."""

    def __init__(self, message, *, history, optimizer=None):
        super().__init__(message)
        self.history = [dict(entry) for entry in history]
        self.optimizer = None if optimizer is None else dict(optimizer)
        self.details = {"success": False, "error": str(message), "history": self.history,
                        "optimizer": self.optimizer}


def _inputs(rows, before_date):
    if before_date != FIT_BEFORE or type(before_date) is not str:
        raise ValueError("CMP research fits only earlier rows before 2025-08-01")
    return _window_inputs(rows, VALID_START, VALID_END,
        "CMP parameter selection accepts only 2024/25 prediction dates")


def _window_inputs(rows, training_start, before_date, range_message):
    records = list(rows)
    if not records:
        raise ValueError("Need nonempty earlier-season prediction rows")
    identifiers, means, goals = set(), [], []
    for row in records:
        try:
            validate_prediction(row)
            played = date.fromisoformat(row["date"])
            if not training_start <= played < before_date:
                raise ValueError(range_message)
            if row["match_id"] in identifiers:
                raise ValueError("Duplicate earlier match identifier")
            identifiers.add(row["match_id"])
            raw = row["rates"]
            if (not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 2
                    or any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, float, np.integer, np.floating))
                           or not math.isfinite(float(value)) or float(value) <= 0 for value in raw)):
                raise ValueError("Need two original finite positive raw goal rates")
            rates = tuple(float(value) for value in raw)
            _same(row["forecast"], summarize(score_grid(*rates, rho=0)), "original_raw_mean_distribution")
            _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "original_one_x_two")
            means.append(rates)
            goals.append(tuple(row["actual_score"]))
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError(f"Malformed earlier CMP fitting row: {error}") from error
    order = sorted(range(len(records)), key=lambda i: (records[i]["date"], records[i]["match_id"]))
    return ([records[i] for i in order], np.asarray([means[i] for i in order], dtype=float),
            np.asarray([goals[i] for i in order], dtype=int))


def fit_cmp(rows, before_date=FIT_BEFORE):
    """Minimize equal-match joint exact-score NLL using one global shared nu.

    Only 2024/25 earlier forecasts are accepted; their supplied means are never
    changed. Endpoints are evaluated explicitly because bounded scalar search
    does not itself visit them. A failed optimizer or batch evaluation raises
    ``CMPFitError`` with all attempted values, rather than switching to nu=1.
    """
    records, means, goals = _inputs(rows, before_date)
    return _fit_cmp(records, means, goals, before_date)


def _iso_date(value, name):
    if type(value) is not str:
        raise ValueError(f"{name} must be an ISO YYYY-MM-DD date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO YYYY-MM-DD date string") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO YYYY-MM-DD date string")
    return parsed


def fit_cmp_window(rows, *, training_start, before_date):
    """Fit one shared nu on an explicit, strictly earlier calendar window.

    All supplied matches must lie in ``[training_start, before_date)``; this
    function never filters out out-of-window rows. Original saved forecasts
    retain the same chronology, raw-mean coherence and unique-ID validation as
    :func:`fit_cmp`. No weighting, decay or team refitting is introduced.
    The legacy fixed-season function retains its separate July 1 upper bound.
    """
    start = _iso_date(training_start, "training_start")
    end = _iso_date(before_date, "before_date")
    if not start < end:
        raise ValueError("training_start must be strictly earlier than before_date")
    records, means, goals = _window_inputs(rows, start, end,
        "CMP window rows must satisfy training_start <= date < before_date")
    result = _fit_cmp(records, means, goals, before_date)
    return {**result, "training_start": training_start, "training_end_exclusive": before_date}


def _fit_cmp(records, means, goals, before_date):
    """Common one-parameter objective and bounded numerical search."""
    history = []

    def objective(nu, stage="optimizer"):
        entry = {"evaluation": len(history) + 1, "nu": float(nu), "stage": stage,
                 "nll": None, "numerical_status": "pending"}
        try:
            if not math.isfinite(float(nu)) or not NU_BOUNDS[0] <= nu <= NU_BOUNDS[1]:
                raise ValueError("Nu evaluation lies outside the declared bounds")
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                values = np.asarray(batch_logpmf(means, goals, float(nu)), dtype=float)
            if values.shape != means.shape or not np.isfinite(values).all() or np.any(values > 0):
                raise ValueError("CMP batch must return finite nonpositive log probabilities for both sides")
            value = -math.fsum(float(row[0]) + float(row[1]) for row in values) / len(records)
            if not math.isfinite(value):
                raise ValueError("Nonfinite average joint-score CMP objective")
        except (ValueError, TypeError, RuntimeError, RuntimeWarning, FloatingPointError, OverflowError) as error:
            entry.update(numerical_status="error", error_type=type(error).__name__, error=str(error))
            history.append(entry)
            raise CMPFitError(f"CMP numerical evaluation failed at nu={nu}: {error}", history=history) from error
        entry.update(nll=value, numerical_status="finite")
        history.append(entry)
        return value

    poisson_objective = objective(1., stage="poisson_reference")
    objective(NU_BOUNDS[0], stage="lower_bound")
    objective(NU_BOUNDS[1], stage="upper_bound")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            optimized = minimize_scalar(objective, bounds=NU_BOUNDS, method="bounded",
                options={"xatol": XATOL, "maxiter": MAXITER})
    except CMPFitError:
        raise
    except (ValueError, RuntimeError, RuntimeWarning, FloatingPointError, OverflowError) as error:
        raise CMPFitError(f"CMP bounded optimization failed: {error}", history=history) from error
    optimizer = {"method": "scipy.optimize.minimize_scalar_bounded",
                 "success": bool(optimized.success), "status": int(optimized.status),
                 "message": str(optimized.message), "nfev": int(optimized.nfev),
                 "nit": int(optimized.nit), "x": float(optimized.x), "fun": float(optimized.fun),
                 "xatol": XATOL, "maxiter": MAXITER}
    if (not optimizer["success"] or not math.isfinite(optimizer["x"])
            or not math.isfinite(optimizer["fun"])
            or not NU_BOUNDS[0] <= optimizer["x"] <= NU_BOUNDS[1]):
        raise CMPFitError("CMP bounded optimizer did not complete a finite in-bounds fit",
                          history=history, optimizer=optimizer)
    # Include the reference and exact endpoints in the constrained search result;
    # this is not clipping an interior estimate or extending the search range.
    selected = min(history, key=lambda entry: (entry["nll"], entry["nu"]))
    nu = selected["nu"]
    at_boundary = nu in NU_BOUNDS
    return {"nu": nu, "bounds": list(NU_BOUNDS), "at_boundary": at_boundary,
            "boundary": "lower" if nu == NU_BOUNDS[0] else "upper" if nu == NU_BOUNDS[1] else None,
            "objective_opt": selected["nll"], "objective_poisson": poisson_objective,
            "objective_definition": "equal_match_mean_negative_sum_home_away_exact_score_logpmf",
            "n_matches": len(records), "n_marginals": 2 * len(records),
            "training_match_ids": [row["match_id"] for row in records],
            "training_min_date": records[0]["date"], "training_max_date": records[-1]["date"],
            "before_date": before_date, "success": True, "nfev": len(history),
            "optimizer": optimizer, "history": history,
            "shared_parameter": "one_nu_for_all_leagues_and_both_sides",
            "means_changed": False, "new_team_fits": 0, "market_blending": False,
            "source_rate_basis": "caller_replayed_original_180day_alpha_0.01_unadjusted_team_means",
            "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
            "coverage_note": "Caller supplies the complete expected earlier forecast cohort; no rows are dropped",
            "interpretation": "Earlier-score fitted objective, not an independent accuracy or profitability claim"}
