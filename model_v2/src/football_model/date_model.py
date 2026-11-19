"""Date-level retrospective fitting, deliberately not a timed prediction API.

Assumption: eligible results from strictly earlier calendar dates are available
at the start of the fit date. This is an experimental availability convention,
not a claim about historical publication timestamps or a T-24h reconstruction.
"""

from datetime import date
import math
from typing import Sequence, TYPE_CHECKING

from .team_model import fit_weighted_poisson, team_features

if TYPE_CHECKING:
    from .football_data import HistoricalMatch


class DateTeamGoalModel:
    def __init__(self, regularization: float = 0.01, half_life_days: float = 180):
        if not all(math.isfinite(x) and x > 0 for x in (regularization, half_life_days)):
            raise ValueError("Positive shrinkage and time half-life are required")
        self.regularization = regularization
        self.half_life_days = half_life_days

    def fit(self, records: Sequence["HistoricalMatch"], before_date: date):
        if type(before_date) is not date:
            raise ValueError("Use an explicit calendar date, not a fabricated timestamp")
        history = sorted((r for r in records if r.match_date < before_date
                          and r.goal_fit_exclusion is None),
                         key=lambda r: (r.match_date, r.match_id))
        if not history:
            raise ValueError("No eligible earlier-date results")
        if len({r.match_id for r in history}) != len(history):
            raise ValueError("Duplicate match identifiers in training history")
        features, target, weights = [], [], []
        for record in history:
            features.extend(team_features(record.home, record.away, record.competition))
            target.extend(record.full_time)
            weight = math.exp(-math.log(2) * (before_date - record.match_date).days /
                              self.half_life_days)
            weights.extend((weight, weight))
        vectorizer, model = fit_weighted_poisson(features, target, weights, self.regularization)
        self.coefficients = dict(zip(vectorizer.get_feature_names_out(), map(float, model.coef_)))
        self.intercept = float(model.intercept_)
        self.before_date = before_date
        self.teams = {t for r in history for t in (r.home, r.away)}
        self.competitions = {r.competition for r in history}
        self.n_matches = len(history)
        self.training_min_date = history[0].match_date
        self.training_max_date = history[-1].match_date
        self.reference_match_ids = [r.match_id for r in history]
        self.optimizer_iterations = int(model.n_iter_)
        return self

    def predict(self, home: str, away: str, competition: str, match_date: date):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit earlier-date results before predicting")
        if type(match_date) is not date or match_date < self.before_date:
            raise ValueError("Prediction must not precede the fit cutoff date")
        if home == away:
            raise ValueError("Teams must differ")
        if {home, away} - self.teams:
            raise ValueError("Unseen teams require an explicitly estimated prior")
        if competition not in self.competitions:
            raise ValueError("Unseen competition")
        return tuple(math.exp(self.intercept + sum(self.coefficients.get(k, 0) * v
                     for k, v in row.items()))
                     for row in team_features(home, away, competition))

    def parameters(self):
        if not hasattr(self, "coefficients"):
            raise ValueError("Model has not been fitted")
        return {
            "model_type": "date_level_regularized_poisson", "regularization": self.regularization,
            "half_life_days": self.half_life_days, "before_date": self.before_date.isoformat(),
            "training_min_date": self.training_min_date.isoformat(),
            "training_max_date": self.training_max_date.isoformat(), "n_matches": self.n_matches,
            "teams": sorted(self.teams), "competitions": sorted(self.competitions),
            "intercept": self.intercept, "coefficients": self.coefficients,
            "optimizer_iterations": self.optimizer_iterations,
            "reference_match_ids": self.reference_match_ids,
            "basis": "team_marginal", "rho": 0,
            "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
            "not_for": "precise T-24h, lineup-confirmed, cross-league, or current-2026 forecasts",
        }
