"""One-budget score-driven residuals inside original team-fit snapshots.

This is not a long-lived latent state-space team model. Every change of the
original competition fit resets its residual state, so the next original fit
and this filter do not deliberately learn the same earlier match twice. All
matches on one calendar day are predicted before any of their scores update
state. Only the supplied, already qualified original archive cohort updates it.
"""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import math
from numbers import Real

from scipy.optimize import minimize_scalar

from .cmp_fit import _iso_date
from .forecast_diagnostics import validate_prediction
from .probability import score_grid, summarize
from .refresh_research import _same


ETA_BOUNDS = (0., .20)
XATOL = .001
MAXITER = 12
MAX_EVALUATIONS = MAXITER + 2
TRAINING_START = "2024-08-01"
TRAINING_END_EXCLUSIVE = "2025-07-01"
FIT_BEFORE = "2025-08-01"
RESEARCH_FIELDS = (
    "dynamic_state", "cmp_nu", "cmp_numerics", "dependence_parameter",
    "dependence_rho", "dependence_k", "dependence_numerics", "sarmanov_theta",
    "calibration_method", "calibration_scales", "base_rates",
)


class DynamicFitError(RuntimeError):
    """A recorded failed fit; never a substitution of eta=0 or a wider search."""

    def __init__(self, message, *, evaluations, optimizer=None):
        super().__init__(message)
        self.evaluations = deepcopy(evaluations)
        self.history = self.evaluations
        self.optimizer = deepcopy(optimizer)
        self.details = {"success": False, "error": str(message),
                        "evaluations": self.evaluations, "optimizer": self.optimizer}


def _eta(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not ETA_BOUNDS[0] <= value <= ETA_BOUNDS[1]):
        raise ValueError("eta must be a finite real number within [0, .20]")
    return float(value)


def _prepare(rows):
    try:
        records = list(rows)
    except TypeError as error:
        raise ValueError("Need nonempty original prediction rows") from error
    if not records:
        raise ValueError("Need nonempty original prediction rows")
    seen = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("Original prediction rows must be mappings")
        if any(key in row for key in RESEARCH_FIELDS):
            raise ValueError("Research layers are not stacked on dynamic residuals")
        try:
            validate_prediction(row)
            for field in ("date", "fit_before_date", "training_max_date"):
                _iso_date(row[field], field)
            if row["match_id"] in seen:
                raise ValueError("Duplicate original match identifier")
            seen.add(row["match_id"])
            if type(row["training_matches"]) is not int or row["training_matches"] <= 0:
                raise ValueError("Original training_matches must be a positive integer")
            rates = row["rates"]
            if (not isinstance(rates, (list, tuple)) or len(rates) != 2
                    or any(isinstance(value, bool) or not isinstance(value, Real)
                           or not math.isfinite(value) or value <= 0 for value in rates)):
                raise ValueError("Need two original finite positive raw goal rates")
            _same(row["forecast"], summarize(score_grid(*rates, rho=0)),
                  "dynamic_original_raw_mean_distribution")
            _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()),
                  "dynamic_original_one_x_two")
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError(f"Malformed original dynamic-filter input: {error}") from error
    return sorted(records, key=lambda row: (row["date"], row["match_id"]))


def _run(records, eta, *, forecasts, collect):
    """Already validated records; objectives need neither copies nor score grids."""
    epochs, output, nlls = {}, [], []
    days = defaultdict(list)
    for row in records:
        days[row["date"]].append(row)
    for day, daily_rows in days.items():
        by_comp = defaultdict(list)
        for row in daily_rows:
            by_comp[row["competition"]].append(row)
        reset_today = {}
        for comp, matches in by_comp.items():
            cutoffs = {row["fit_before_date"] for row in matches}
            if len(cutoffs) != 1:
                raise ValueError("One competition/day cannot use conflicting original snapshots")
            cutoff = next(iter(cutoffs))
            previous = epochs.get(comp)
            if previous is not None and cutoff < previous["cutoff"]:
                raise ValueError("Original snapshot cutoffs must not move backwards")
            reset_today[comp] = previous is None or cutoff != previous["cutoff"]
            if reset_today[comp]:
                epochs[comp] = {"cutoff": cutoff, "teams": {}, "history": [], "last": None}

        # No score from this day is used until every daily forecast is computed.
        pending = []
        for row in daily_rows:
            comp, home, away = row["competition"], row["home"], row["away"]
            epoch = epochs[comp]
            teams = epoch["teams"]
            ah, dh = teams.get(home, (0., 0.))
            aa, da = teams.get(away, (0., 0.))
            # Multiplicative form is equivalent to adding offsets to log(mu),
            # and preserves the original float rates exactly when eta=0.
            rates = [float(row["rates"][0]) * math.exp(ah - da),
                     float(row["rates"][1]) * math.exp(aa - dh)]
            if any(not math.isfinite(value) or value <= 0 for value in rates):
                raise ValueError("Nonfinite/nonpositive dynamic rate; no clipping or fallback")
            score = row["actual_score"]
            nll = math.fsum(rate - goal * math.log(rate) + math.lgamma(goal + 1)
                            for rate, goal in zip(rates, score))
            if not math.isfinite(nll):
                raise ValueError("Nonfinite dynamic full-support exact-score NLL")
            nlls.append(nll)
            if collect:
                item = deepcopy(row)
                item.update({"base_rates": list(row["rates"]), "rates": rates,
                    "base_fit_before_date": row["fit_before_date"],
                    "base_training_max_date": row["training_max_date"],
                    "fit_before_date": day,
                    "training_max_date": max(row["training_max_date"], epoch["last"] or row["training_max_date"]),
                    "training_matches": row["training_matches"] + len(epoch["history"]),
                    "exact_score_nll": nll,
                    "dynamic_state": {"eta": eta, "home_attack": ah, "away_defense": da,
                        "away_attack": aa, "home_defense": dh,
                        "dynamic_history_ids": list(epoch["history"]),
                        "last_update_date": epoch["last"],
                        "snapshot_reset": reset_today[comp], "forecasts_rebuilt": forecasts}})
                if forecasts:
                    grid = score_grid(*rates, rho=0)
                    item["forecast"] = summarize(grid)
                    item["probabilities"] = list(item["forecast"]["one_x_two"].values())
                else:
                    # Cheap rate-only rows are not complete forecasts. Do not
                    # leave original probabilities beside adjusted rates.
                    item.pop("forecast")
                    item.pop("probabilities")
                output.append(item)
            eh, ea = (goal - rate for goal, rate in zip(score, rates))
            pending.append((comp, home, away, eta * eh / 2, eta * ea / 2, row["match_id"]))

        increments = defaultdict(lambda: [[], []])
        for comp, home, away, sh, sa, key in pending:
            increments[comp, home][0].append(sh)
            increments[comp, home][1].append(-sa)
            increments[comp, away][0].append(sa)
            increments[comp, away][1].append(-sh)
            epochs[comp]["history"].append(key)
            epochs[comp]["last"] = day
        for (comp, team), changes in increments.items():
            prior = epochs[comp]["teams"].get(team, (0., 0.))
            state = [math.fsum((prior[i], *changes[i])) for i in (0, 1)]
            if any(not math.isfinite(value) for value in state):
                raise ValueError("Nonfinite dynamic state; no clipping or fallback")
            epochs[comp]["teams"][team] = state
    return output, math.fsum(nlls) / len(records)


def filter_rows(rows, eta, *, forecasts=True):
    """Predict then update, ordered by date/ID, without changing source rows.

    ``forecasts=False`` returns rate/NLL research rows without forecast or
    probabilities keys. Only the default True result is a complete forecast.
    Initial zero residuals mean no *additional* offset on a qualified original
    team rate; they are not zero strength priors for unseen teams or players.
    """
    if type(forecasts) is not bool:
        raise ValueError("forecasts must be a bool")
    records, learning_rate = _prepare(rows), _eta(eta)
    return _run(records, learning_rate, forecasts=forecasts, collect=True)[0]


def fit_learning_rate(earlier_rows):
    """One bounded scalar fit, <=12 optimizer objectives plus 2 exact endpoints.

    The equal-match prequential full-score NLL covers only the declared earlier
    season. It is the fitting objective, not a fresh validation season. No
    wider range, restart, grid, per-league fit or failed-search fallback occurs.
    """
    records = _prepare(earlier_rows)
    if any(not TRAINING_START <= row["date"] < TRAINING_END_EXCLUSIVE for row in records):
        raise ValueError("Learning rate fits only 2024/25 rows in [2024-08-01, 2025-07-01)")
    history, cache = [], {}

    def objective(eta, stage="optimizer"):
        eta = _eta(eta)
        if eta in cache:
            return cache[eta]
        if len(history) >= MAX_EVALUATIONS:
            raise DynamicFitError("Dynamic learning-rate objective budget exhausted",
                                  evaluations=history)
        entry = {"evaluation": len(history) + 1, "eta": eta, "stage": stage,
                 "nll": None, "numerical_status": "pending"}
        try:
            _, value = _run(records, eta, forecasts=False, collect=False)
        except (ValueError, TypeError, RuntimeError, OverflowError, FloatingPointError) as error:
            entry.update({"numerical_status": "error", "error": str(error), "error_type": type(error).__name__})
            history.append(entry)
            raise DynamicFitError(f"Dynamic evaluation failed at eta={eta}: {error}",
                                  evaluations=history) from error
        if not math.isfinite(value):
            entry["numerical_status"] = "nonfinite"
            history.append(entry)
            raise DynamicFitError("Nonfinite dynamic objective", evaluations=history)
        entry.update({"nll": value, "numerical_status": "finite"})
        history.append(entry)
        cache[eta] = value
        return value

    objective(ETA_BOUNDS[0], "lower_bound")
    objective(ETA_BOUNDS[1], "upper_bound")
    try:
        optimized = minimize_scalar(objective, bounds=ETA_BOUNDS, method="bounded",
            options={"xatol": XATOL, "maxiter": MAXITER})
    except DynamicFitError:
        raise
    except (ValueError, TypeError, RuntimeError, OverflowError, FloatingPointError) as error:
        raise DynamicFitError(f"Dynamic optimizer failed: {error}", evaluations=history) from error
    optimizer = {"success": bool(optimized.success), "message": str(optimized.message),
        "x": float(optimized.x), "fun": float(optimized.fun), "nfev": int(optimized.nfev)}
    if not optimized.success or not math.isfinite(optimized.fun):
        raise DynamicFitError("Dynamic optimizer did not converge within the declared budget",
                              evaluations=history, optimizer=optimizer)
    chosen = min(history, key=lambda item: (item["nll"], item["eta"]))
    return {"eta": chosen["eta"], "objective_nll": chosen["nll"],
        "selected_stage": chosen["stage"], "selected_on_boundary": chosen["eta"] in ETA_BOUNDS,
        "evaluations": history, "n_evaluations": len(history),
        "optimizer_success": True, "optimizer_message": str(optimized.message), "optimizer": optimizer,
        "eta_bounds": list(ETA_BOUNDS), "xatol": XATOL, "max_optimizer_evaluations": MAXITER,
        "max_total_evaluations": MAX_EVALUATIONS, "before_date": FIT_BEFORE,
        "training_start": TRAINING_START, "training_end_exclusive": TRAINING_END_EXCLUSIVE,
        "training_match_ids": [row["match_id"] for row in records], "n_matches": len(records),
        "training_min_date": records[0]["date"], "training_max_date": records[-1]["date"],
        "training_objective_not_blind_validation": True, "new_team_fits": 0,
        "market_inputs": False, "restart_count": 0, "fallback": False,
        "state_scope": "competition_original_fit_snapshot", "daily_predict_before_update": True}
