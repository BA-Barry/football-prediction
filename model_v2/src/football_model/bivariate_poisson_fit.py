"""Earlier-only, fixed-marginal bivariate-Poisson dependence research fit.

H = U + W and A = V + W, with E[W] = k * min(mu_H, mu_A).
Only the shared common-shock fraction k is learned here.  It is not Pearson's
correlation and this family cannot represent negative dependence.  Original
raw means and Poisson marginals are retained; no CMP, team refit, market blend
or later target results enter the numerical objective.
"""

import math
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from .bivariate_poisson import batch_logpmf
from .cmp_fit import _iso_date, _window_inputs


K_BOUNDS = (0., .95)
MESH = tuple(i / 20 for i in range(20))
XATOL = 1e-6
MAXITER = 100


class BivariatePoissonFitError(RuntimeError):
    """A failed research fit retaining its attempted objective trace."""

    def __init__(self, message, *, history, optimizer=None):
        super().__init__(message)
        self.history = [dict(entry) for entry in history]
        self.optimizer = None if optimizer is None else dict(optimizer)
        self.details = {"success": False, "error": str(message),
                        "history": self.history, "optimizer": self.optimizer}


def fit_window(rows, *, training_start, before_date):
    """Minimize equal-match full-support joint exact-score NLL on past rows.

    All supplied rows must lie in [training_start, before_date); out-of-window
    or malformed rows are errors, not silently dropped observations.  Reuse
    the existing original-forecast validation, not the CMP numerical model.

    A predetermined .05 mesh, the full bounded search, both boundary-adjacent
    cells, and all mesh-local-minimum brackets are searched.  The best of the
    actual finite evaluations is selected, including exactly k=0 and k=.95.
    This is a deterministic numerical candidate search, not a mathematical
    certificate of the global minimum; joint likelihood need not be concave.
    Any numerical or optimizer failure is explicit, never a Poisson fallback.
    """
    start = _iso_date(training_start, "training_start")
    end = _iso_date(before_date, "before_date")
    if not start < end:
        raise ValueError("training_start must be strictly earlier than before_date")
    records, means, goals = _window_inputs(rows, start, end,
        "Dependence fitting rows must satisfy training_start <= date < before_date")
    return _fit(records, means, goals, training_start, before_date)


def _fit(records, means, goals, training_start, before_date):
    history = []

    def objective(k, stage="optimizer"):
        entry = {"evaluation": len(history) + 1, "k": float(k), "stage": stage,
                 "nll": None, "numerical_status": "pending"}
        try:
            if not math.isfinite(float(k)) or not K_BOUNDS[0] <= k <= K_BOUNDS[1]:
                raise ValueError("k evaluation lies outside the declared bounds")
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                values = np.asarray(batch_logpmf(goals[:, 0], goals[:, 1], means[:, 0], means[:, 1], float(k)),
                                    dtype=float)
            if values.shape != (len(records),) or not np.isfinite(values).all() or np.any(values > 0):
                raise ValueError("Joint batch must return one finite nonpositive log probability per match")
            value = -math.fsum(float(item) for item in values) / len(records)
            if not math.isfinite(value):
                raise ValueError("Nonfinite average joint exact-score objective")
        except (ValueError, TypeError, RuntimeError, RuntimeWarning, FloatingPointError, OverflowError) as error:
            entry.update(numerical_status="error", error_type=type(error).__name__, error=str(error))
            history.append(entry)
            raise BivariatePoissonFitError(
                f"Bivariate Poisson numerical evaluation failed at k={k}: {error}", history=history) from error
        entry.update(nll=value, numerical_status="finite")
        history.append(entry)
        return value

    null_objective = objective(0., stage="independence_null")
    upper_objective = objective(K_BOUNDS[1], stage="upper_bound")
    mesh_values = [null_objective]
    mesh_values.extend(objective(k, stage="predetermined_mesh") for k in MESH[1:-1])
    mesh_values.append(upper_objective)
    # Do not assume global concavity or replace bounded endpoints with nearby
    # interior values. These brackets depend only on earlier-score objectives.
    brackets = [(K_BOUNDS, "full_range"), ((MESH[0], MESH[1]), "lower_adjacent"),
                ((MESH[-2], MESH[-1]), "upper_adjacent")]
    for i in range(1, len(MESH) - 1):
        if mesh_values[i] <= mesh_values[i-1] and mesh_values[i] <= mesh_values[i+1]:
            brackets.append(((MESH[i-1], MESH[i+1]), f"mesh_minimum_{i}"))
    runs = []
    for bounds, label in brackets:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                optimized = minimize_scalar(lambda k: objective(k, stage=label), bounds=bounds,
                    method="bounded", options={"xatol": XATOL, "maxiter": MAXITER})
        except BivariatePoissonFitError:
            raise
        except (ValueError, TypeError, RuntimeError, RuntimeWarning, FloatingPointError, OverflowError) as error:
            raise BivariatePoissonFitError(
                f"Bivariate Poisson bounded search failed in {label}: {error}", history=history,
                optimizer={"success": False, "completed_runs": runs, "failed_bracket": label}) from error
        try:
            x, fun = float(optimized.x), float(optimized.fun)
            run = {"bracket": label, "bounds": list(bounds), "success": bool(optimized.success),
                   "status": int(optimized.status), "message": str(optimized.message),
                   "nfev": int(optimized.nfev), "nit": int(optimized.nit),
                   "x": x if math.isfinite(x) else None, "fun": fun if math.isfinite(fun) else None}
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise BivariatePoissonFitError(
                f"Malformed bivariate Poisson optimizer result in {label}: {error}", history=history,
                optimizer={"success": False, "completed_runs": runs, "failed_bracket": label}) from error
        runs.append(run)
        if (not run["success"] or run["x"] is None or run["fun"] is None
                or not bounds[0] <= run["x"] <= bounds[1]):
            raise BivariatePoissonFitError(
                "Bivariate Poisson optimizer did not complete a finite in-bounds fit", history=history,
                optimizer={"success": False, "runs": runs, "failed_bracket": label})
    selected = min(history, key=lambda entry: (entry["nll"], entry["k"]))
    k = selected["k"]
    optimizer = {"method": "predetermined_mesh_plus_scipy_bounded_brackets",
                 "success": True, "mesh": list(MESH), "mesh_nll": mesh_values,
                 "runs": runs, "nfev": sum(run["nfev"] for run in runs),
                 "xatol": XATOL, "maxiter": MAXITER,
                 "global_optimum_certified": False,
                 "search_note": "Full range, boundary-adjacent cells and every observed mesh-local-minimum bracket"}
    return {"k": k, "bounds": list(K_BOUNDS), "at_boundary": k in K_BOUNDS,
            "boundary": "lower" if k == 0. else "upper" if k == K_BOUNDS[1] else None,
            "objective_opt": selected["nll"], "objective_poisson": null_objective,
            "objective_null": null_objective,
            "objective_definition": "equal_match_mean_negative_joint_exact_score_logpmf",
            "n_matches": len(records), "training_match_ids": [row["match_id"] for row in records],
            "training_min_date": records[0]["date"], "training_max_date": records[-1]["date"],
            "training_start": training_start, "before_date": before_date,
            "training_end_exclusive": before_date, "success": True, "nfev": len(history),
            "optimizer": optimizer, "history": history,
            "shared_parameter": "one_common_shock_fraction_k_for_all_leagues_and_both_sides",
            "common_component_definition": "c=k*min(mu_home,mu_away)",
            "pearson_correlation_definition": "k*min(mu_home,mu_away)/sqrt(mu_home*mu_away)",
            "negative_dependence_supported": False, "means_changed": False,
            "poisson_marginals_preserved": True, "cmp_stacked": False,
            "new_team_fits": 0, "market_blending": False,
            "source_rate_basis": "caller_replayed_original_180day_alpha_0.01_unadjusted_team_means",
            "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
            "coverage_note": "Caller supplies the complete expected earlier forecast cohort; no rows are dropped",
            "interpretation": "Earlier-score fitted objective, not independent accuracy or profitability evidence"}
