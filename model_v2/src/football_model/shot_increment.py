"""Research-only goal-rate increment from strictly earlier shots on target.

Original goal rates are fixed offsets, not re-fitted to the evaluated scores.
Both scoring sides share two coefficients. Historical publication times remain
unknown: the availability assumption is earlier calendar dates, not T-24h.
"""

from dataclasses import dataclass
from datetime import date
import math

import numpy as np
from scipy.optimize import minimize


@dataclass
class ShotObservation:
    match_id: str
    match_date: date
    home: str
    away: str
    shots_on_target: tuple[int, int] | None


def history_features(observations, home, away, before_date, window=10, min_history=5):
    """Use the last league games before the original model's information date.

    Missing counts stay missing; they are not skipped to reach older games.
    A long gap after relegation is recorded, not silently called recent form.
    """
    if type(before_date) is not date or home == away:
        raise ValueError("Need a calendar cutoff and distinct teams")
    if not (type(window) is int and type(min_history) is int and 1 <= min_history <= window):
        raise ValueError("Need ordered positive history lengths")
    selected = sorted((r for r in observations if r.match_date < before_date),
                      key=lambda r: (r.match_date, r.match_id))
    if len({r.match_id for r in selected}) != len(selected):
        raise ValueError("Duplicate shot observation")
    histories = {}
    for team in (home, away):
        games = [r for r in selected if team in (r.home, r.away)][-window:]
        missing = [r.match_id for r in games if r.shots_on_target is None]
        for_values, against_values = [], []
        for record in games:
            if record.shots_on_target is None:
                continue
            if (len(record.shots_on_target) != 2 or any(type(v) is not int or v < 0
                                                       for v in record.shots_on_target)):
                raise ValueError("Shots on target must be nonnegative integers")
            own, other = (record.shots_on_target if record.home == team
                          else record.shots_on_target[::-1])
            for_values.append(own)
            against_values.append(other)
        reason = "missing_shot_counts" if missing else (
            "insufficient_league_history" if len(games) < min_history else None)
        histories[team] = {
            "match_ids": [r.match_id for r in games], "n_matches": len(games),
            "missing_match_ids": missing, "unavailable_reason": reason,
            "earliest_date": games[0].match_date.isoformat() if games else None,
            "latest_date": games[-1].match_date.isoformat() if games else None,
            "latest_gap_days": (before_date-games[-1].match_date).days if games else None,
            "for_mean": math.fsum(for_values)/len(for_values) if reason is None else None,
            "against_mean": math.fsum(against_values)/len(against_values) if reason is None else None,
        }
    h, a = histories[home], histories[away]
    features = None if any(r["unavailable_reason"] for r in (h, a)) else [
        [h["for_mean"], a["against_mean"]], [a["for_mean"], h["against_mean"]]]
    return {"before_date": before_date.isoformat(), "features": features, "history": histories}


def poisson_offset_objective(beta, x, y, rates, weights, regularization):
    """Weighted mean NLL, omitting constants independent of beta, and gradient."""
    eta = x @ beta
    mu = rates * np.exp(eta)
    value = float(np.sum(weights*(mu-y*eta)) + regularization*np.dot(beta, beta)/2)
    gradient = x.T @ (weights*(mu-y)) + regularization*beta
    return value, gradient


class OffsetShotModel:
    def __init__(self, regularization=.01, half_life_days=180):
        if not all(math.isfinite(v) and v > 0 for v in (regularization, half_life_days)):
            raise ValueError("Positive regularization and time half-life required")
        self.regularization, self.half_life_days = regularization, half_life_days

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError("Need a calendar cutoff")
        history = sorted((r for r in rows if date.fromisoformat(r["date"]) < before_date),
                         key=lambda r: (r["date"], r["match_id"]))
        if not history or len({r["match_id"] for r in history}) != len(history):
            raise ValueError("Need unique nonempty earlier out-of-time predictions")
        for row in history:
            if not date.fromisoformat(row["fit_before_date"]) <= date.fromisoformat(row["date"]):
                raise ValueError("Offset must have been fitted before its target")
            for key, shape in (("shot_features", (2, 2)), ("rates", (2,)), ("actual_score", (2,))):
                if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(row[key], dtype=object).flat):
                    raise ValueError(f"Boolean flags are not counts or rates: {key}")
                values = np.asarray(row[key])
                if values.shape != shape or values.dtype.kind not in "iuf":
                    raise ValueError(f"Wrong per-match shape or numeric type: {key}")
        raw = np.asarray([r["shot_features"] for r in history], dtype=float).reshape(-1, 2)
        y = np.asarray([r["actual_score"] for r in history], dtype=float).reshape(-1)
        rates = np.asarray([r["rates"] for r in history], dtype=float).reshape(-1)
        if (not np.isfinite(raw).all() or not np.isfinite(rates).all() or np.any(rates <= 0)
                or not np.isfinite(y).all() or np.any(y < 0) or np.any(y != np.floor(y))):
            raise ValueError("Need finite features, positive offsets, and count targets")
        weights = np.repeat([math.exp(-math.log(2)*(before_date-date.fromisoformat(r["date"])).days
                                     / self.half_life_days) for r in history], 2)
        weights /= weights.sum()
        self.mean = np.sum(weights[:, None]*raw, axis=0)
        constant = np.ptp(raw, axis=0) == 0
        self.mean[constant] = raw[0, constant]
        variance = np.sum(weights[:, None]*(raw-self.mean)**2, axis=0)
        self.scale = np.where(~constant, np.sqrt(variance), 1.)
        x = (raw-self.mean)/self.scale
        result = minimize(poisson_offset_objective, np.zeros(2),
                          args=(x, y, rates, weights, self.regularization), jac=True,
                          method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
        value, gradient = poisson_offset_objective(result.x, x, y, rates, weights, self.regularization)
        if not result.success or not np.isfinite(result.x).all() or np.max(np.abs(gradient)) > 1e-7:
            raise ValueError(f"Shot offset optimizer failed: {result.message}")
        self.beta, self.before_date = result.x, before_date
        self.metadata = {
            "before_date": before_date.isoformat(), "training_min_date": history[0]["date"],
            "training_max_date": history[-1]["date"], "n_matches": len(history),
            "reference_match_ids": [r["match_id"] for r in history],
            "regularization": self.regularization, "half_life_days": self.half_life_days,
            "coefficients": self.beta.tolist(), "feature_mean": self.mean.tolist(),
            "feature_scale": self.scale.tolist(), "optimizer_success": bool(result.success),
            "optimizer_iterations": int(result.nit), "gradient_max_abs": float(np.max(np.abs(gradient))),
            "objective_without_beta_independent_constants": value,
        }
        return self

    def predict(self, rates, features):
        if not hasattr(self, "beta"):
            raise ValueError("Fit earlier observations before predicting")
        rates, raw = np.asarray(rates, dtype=float), np.asarray(features, dtype=float)
        if (rates.shape != (2,) or raw.shape != (2, 2) or not np.isfinite(rates).all()
                or np.any(rates <= 0) or not np.isfinite(raw).all()):
            raise ValueError("Need two positive rates and two shared shot features per side")
        predicted = rates*np.exp(((raw-self.mean)/self.scale) @ self.beta)
        if not np.isfinite(predicted).all() or np.any(predicted <= 0):
            raise ValueError("Non-finite adjusted goal rates")
        return tuple(map(float, predicted))

    def parameters(self):
        if not hasattr(self, "metadata"):
            raise ValueError("Model not fitted")
        return dict(self.metadata)
