"""Research-only opponent-adjusted shot strength and one goal-rate increment.

The shot model learns count means from shots on target, not from relabelled
goals. Its expected counts are an auxiliary signal, not an independent vote.
Increment training requires historical out-of-time shot-rate predictions made
at the same information dates as the original goal-rate offsets. This module
cannot infer that provenance from two numerical rates; the research runner
must retain and check it against the saved parameter records.
"""

from datetime import date
import math

import numpy as np
from scipy.optimize import minimize

from .shot_increment import poisson_offset_objective
from .team_model import fit_weighted_poisson, team_features


def _settings(regularization, half_life_days):
    if not all(math.isfinite(v) and v > 0 for v in (regularization, half_life_days)):
        raise ValueError("Positive regularization and time half-life required")


def _calendar(value):
    if type(value) is not date:
        raise ValueError("Need a calendar date, not a fabricated timestamp")


def _pair(value, label, *, counts=False):
    original = np.asarray(value, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) for v in original.flat):
        raise ValueError(f"Boolean flags are not counts or rates: {label}")
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in "iuf":
        raise ValueError(f"Need two numeric values: {label}")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"Need finite values: {label}")
    if counts:
        if np.any(array < 0) or np.any(array != np.floor(array)):
            raise ValueError(f"Need two nonnegative counts: {label}")
    elif np.any(array <= 0):
        raise ValueError(f"Need two positive rates: {label}")
    return array


class DateTeamShotModel:
    """EPL-only Poisson attack/defense model for shots-on-target count means."""

    def __init__(self, regularization=.01, half_life_days=180):
        _settings(regularization, half_life_days)
        self.regularization, self.half_life_days = regularization, half_life_days

    def fit(self, observations, before_date):
        _calendar(before_date)
        observations = list(observations)
        for record in observations:
            _calendar(record.match_date)
        earlier = sorted((r for r in observations if r.match_date < before_date),
                         key=lambda r: (r.match_date, r.match_id))
        if len({r.match_id for r in earlier}) != len(earlier):
            raise ValueError("Duplicate earlier shot observation")
        history, missing = [], []
        for record in earlier:
            if (not isinstance(record.match_id, str) or not record.match_id
                    or not isinstance(record.home, str) or not record.home
                    or not isinstance(record.away, str) or not record.away
                    or record.home == record.away):
                raise ValueError("Need an identified fixture with two distinct teams")
            if record.shots_on_target is None:
                missing.append(record.match_id)
                continue
            _pair(record.shots_on_target, "shots_on_target", counts=True)
            history.append(record)
        if not history:
            raise ValueError("No known shots on target before cutoff")
        features, target, weights = [], [], []
        for record in history:
            features.extend(team_features(record.home, record.away, "E0"))
            target.extend(record.shots_on_target)
            weight = math.exp(-math.log(2)*(before_date-record.match_date).days
                              / self.half_life_days)
            weights.extend((weight, weight))
        vectorizer, model = fit_weighted_poisson(features, target, weights,
                                                self.regularization)
        self.coefficients = dict(zip(vectorizer.get_feature_names_out(),
                                     map(float, model.coef_)))
        self.intercept = float(model.intercept_)
        self.before_date = before_date
        self.teams = {team for r in history for team in (r.home, r.away)}
        self.metadata = {
            "model_type": "date_level_regularized_poisson_shots_on_target",
            "response": "shots_on_target", "competition": "E0",
            "regularization": self.regularization, "half_life_days": self.half_life_days,
            "before_date": before_date.isoformat(),
            "training_min_date": history[0].match_date.isoformat(),
            "training_max_date": history[-1].match_date.isoformat(),
            "n_matches": len(history), "teams": sorted(self.teams),
            "reference_match_ids": [r.match_id for r in history],
            "missing_shot_match_ids": missing, "intercept": self.intercept,
            "coefficients": dict(self.coefficients),
            "optimizer_iterations": int(model.n_iter_),
            "availability_convention": "strictly_earlier_calendar_date_assumed_available",
            "basis": "conditional_shots_on_target_means_only",
        }
        return self

    def predict(self, home, away, competition, match_date):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit earlier shot observations before predicting")
        _calendar(match_date)
        if match_date < self.before_date:
            raise ValueError("Prediction must not precede the shot fit cutoff")
        if competition != "E0":
            raise ValueError("Shot strength research supports E0 only")
        if home == away or {home, away} - self.teams:
            raise ValueError("Need distinct seen teams; unknown shot strength is not zero")
        rates = tuple(math.exp(self.intercept + math.fsum(
            self.coefficients.get(key, 0.)*value for key, value in row.items()))
            for row in team_features(home, away, "E0"))
        if not all(math.isfinite(v) and v > 0 for v in rates):
            raise ValueError("Non-finite predicted shot means")
        return rates

    def parameters(self):
        if not hasattr(self, "metadata"):
            raise ValueError("Shot model not fitted")
        return dict(self.metadata)


class ShotRateGoalIncrement:
    """Shared one-coefficient goal adjustment from log expected shot counts.

The original goal rates stay fixed. Standardization and the single coefficient
use only earlier target results, with the original .01/180 preset and no new
intercept. Shot rates must already be out-of-time and opponent-adjusted.
"""

    def __init__(self, regularization=.01, half_life_days=180):
        _settings(regularization, half_life_days)
        self.regularization, self.half_life_days = regularization, half_life_days

    def fit(self, rows, before_date):
        _calendar(before_date)
        history = sorted((r for r in rows if date.fromisoformat(r["date"]) < before_date),
                         key=lambda r: (r["date"], r["match_id"]))
        if not history or len({r["match_id"] for r in history}) != len(history):
            raise ValueError("Need unique nonempty earlier out-of-time rows")
        raw, y, rates, weights = [], [], [], []
        for row in history:
            played = date.fromisoformat(row["date"])
            if date.fromisoformat(row["fit_before_date"]) > played:
                raise ValueError("Offset and shot fit cutoff must not follow the target")
            offsets = _pair(row["rates"], "goal offsets")
            shots = _pair(row["shot_rates"], "expected shots on target")
            goals = _pair(row["actual_score"], "actual goals", counts=True)
            raw.extend(np.log(shots))
            y.extend(goals)
            rates.extend(offsets)
            weight = math.exp(-math.log(2)*(before_date-played).days / self.half_life_days)
            weights.extend((weight, weight))
        raw, y, rates, weights = map(lambda v: np.asarray(v, dtype=float),
                                     (raw, y, rates, weights))
        if not math.isfinite(float(weights.sum())) or weights.sum() <= 0:
            raise ValueError("Need positive total training weight")
        weights /= weights.sum()
        constant = np.ptp(raw) == 0
        mean = float(raw[0] if constant else np.sum(weights*raw))
        variance = float(np.sum(weights*(raw-mean)**2))
        scale = 1. if constant else math.sqrt(variance)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid training shot-rate scale")
        x = ((raw-mean)/scale)[:, None]
        result = minimize(poisson_offset_objective, np.zeros(1),
                          args=(x, y, rates, weights, self.regularization), jac=True,
                          method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
        value, gradient = poisson_offset_objective(result.x, x, y, rates, weights,
                                                   self.regularization)
        if (not result.success or not np.isfinite(result.x).all()
                or np.max(np.abs(gradient)) > 1e-7):
            raise ValueError(f"Shot-rate increment optimizer failed: {result.message}")
        self.beta, self.before_date = float(result.x[0]), before_date
        self.mean, self.scale = mean, scale
        self.metadata = {
            "model_type": "goal_poisson_offset_expected_shot_rate_increment",
            "response": "full_time_goals", "feature": "log_expected_shots_on_target",
            "before_date": before_date.isoformat(),
            "training_min_date": history[0]["date"],
            "training_max_date": history[-1]["date"], "n_matches": len(history),
            "reference_match_ids": [r["match_id"] for r in history],
            "regularization": self.regularization, "half_life_days": self.half_life_days,
            "coefficient": self.beta, "feature_mean": self.mean, "feature_scale": self.scale,
            "new_intercept": False, "optimizer_success": bool(result.success),
            "optimizer_iterations": int(result.nit),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "objective_without_beta_independent_constants": value,
            "input_convention": "out_of_time_goal_and_shot_rates_at_same_original_cutoff",
        }
        return self

    def predict(self, rates, shot_rates):
        if not hasattr(self, "beta"):
            raise ValueError("Fit earlier goal targets before predicting")
        offsets = _pair(rates, "goal offsets")
        shots = _pair(shot_rates, "expected shots on target")
        predicted = offsets*np.exp((np.log(shots)-self.mean)/self.scale*self.beta)
        if not np.isfinite(predicted).all() or np.any(predicted <= 0):
            raise ValueError("Non-finite adjusted goal rates")
        return tuple(map(float, predicted))

    def parameters(self):
        if not hasattr(self, "metadata"):
            raise ValueError("Shot-rate goal increment not fitted")
        return dict(self.metadata)
