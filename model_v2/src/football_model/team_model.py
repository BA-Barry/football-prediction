"""A trainable, regularized attack/defense GLM built on scikit-learn.

This is a Poisson team model, not a joint Dixon-Coles estimator. rho remains zero
until a separately validated low-score model is fitted. Timed and date-level
experiments share the same feature construction and regression implementation.
"""

from datetime import datetime
from math import log
import math
from typing import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import PoissonRegressor

from .domain import Fixture, GoalRates, MatchResult, aware


def team_features(home: str, away: str, competition: str, neutral: bool = False):
    """Known pre-match fields only; usable without inventing a kickoff time."""
    return (
        {f"attack:{home}": 1, f"defense:{away}": -1,
         f"competition:{competition}": 1, f"home:{competition}": float(not neutral)},
        {f"attack:{away}": 1, f"defense:{home}": -1,
         f"competition:{competition}": 1},
    )


def fit_weighted_poisson(features, target, weights, regularization):
    vectorizer = DictVectorizer()
    design = vectorizer.fit_transform(features)
    model = PoissonRegressor(alpha=regularization, max_iter=1000, tol=1e-8)
    model.fit(design, target, sample_weight=np.asarray(weights))
    return vectorizer, model


class TeamGoalModel:
    def __init__(self, regularization: float = 0.01, half_life_days: float = 180):
        if not all(math.isfinite(x) and x > 0 for x in (regularization, half_life_days)):
            raise ValueError("Positive shrinkage and time half-life are required")
        self.regularization = regularization
        self.half_life_days = half_life_days

    @staticmethod
    def features(fixture: Fixture):
        return team_features(fixture.home, fixture.away, fixture.competition, fixture.neutral)

    def fit(self, records: Sequence[MatchResult], as_of: datetime):
        aware(as_of)
        history = sorted((r for r in records if r.available_at <= as_of and r.fixture.kickoff < as_of),
                         key=lambda r: (r.fixture.kickoff, r.fixture.match_id))
        if not history:
            raise ValueError("No completed results were known at the fit cutoff")
        if len({r.fixture.match_id for r in history}) != len(history):
            raise ValueError("Duplicate match identifiers in training history")
        features, target, weights = [], [], []
        for record in history:
            features.extend(self.features(record.fixture))
            target.extend((record.home_goals, record.away_goals))
            days = (as_of - record.fixture.kickoff).total_seconds() / 86400
            weights.extend([np.exp(-log(2) * days / self.half_life_days)] * 2)
        self.vectorizer, self.model = fit_weighted_poisson(
            features, target, weights, self.regularization)
        self.as_of = as_of
        self.teams = {t for r in history for t in (r.fixture.home, r.fixture.away)}
        self.competitions = {r.fixture.competition for r in history}
        self.n_matches = len(history)
        self.reference_match_ids = [r.fixture.match_id for r in history]
        self.reference_weights = np.asarray(weights)[::2]
        return self

    def predict(self, fixture: Fixture, as_of: datetime):
        aware(as_of)
        if not hasattr(self, "model"):
            raise ValueError("Fit historical data before predicting")
        if self.as_of > as_of or as_of >= fixture.kickoff:
            raise ValueError("Invalid training/prediction chronology")
        if {fixture.home, fixture.away} - self.teams:
            raise ValueError("Unseen teams require an explicitly estimated league/position prior")
        if fixture.competition not in self.competitions:
            raise ValueError("Competition not represented in training")
        home, away = self.model.predict(self.vectorizer.transform(self.features(fixture)))
        return GoalRates(float(home), float(away), self.as_of, "regularized team Poisson GLM",
                         fixture.match_id, "team_marginal")
