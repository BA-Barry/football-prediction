"""Cumulative team-by-season innovations, not a fitted latent state process.

The same normalized, 180-day-weighted ridge Poisson implementation as
DateTeamGoalModel is used. A team's first observed historical season has its
base attack/defense effect. Every subsequent annual season contributes q times
another ridge-penalized coefficient. Missing annual seasons remain represented;
collinear increments are regularized, not individually identified by that fact.

q is an innovation scale relative to the base coefficient penalty, not an
identified real-world volatility. Future, unobserved increments have MAP
conditional mean zero; predictions do not integrate state uncertainty. No old
goal rate or snapshot residual is added on top of this model's fitted rates.
"""

from collections.abc import Mapping
from datetime import date
import math
from numbers import Real
import re

from .date_model import DateTeamGoalModel
from .team_model import fit_weighted_poisson, team_features


def _season_start(label):
    if not isinstance(label, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", label):
        raise ValueError("Season must use canonical YYYY-YY")
    year = int(label[:4])
    if not 1 <= year <= 9999 or int(label[5:]) != (year+1) % 100:
        raise ValueError("Season must identify consecutive annual years")
    return year


def _season_label(year):
    return f"{year:04d}-{(year+1) % 100:02d}"


def season_for_date(day):
    """July 1 inclusive through June 30 inclusive; no fabricated kickoff time."""
    if type(day) is not date:
        raise ValueError("Use an explicit calendar date")
    year = day.year-int(day.month < 7)
    if year < 1:
        raise ValueError("Calendar date precedes supported annual seasons")
    return _season_label(year)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _innovation_key(side, team, year):
    return f"innovation_{side}:{team}:{_season_label(year)}"


class SeasonStateTeamGoalModel(DateTeamGoalModel):
    """Original base features plus cumulative, ridge-regularized innovations."""

    def __init__(self, regularization=.01, half_life_days=180, innovation_scale=0.):
        regularization = _finite(regularization, "regularization")
        half_life_days = _finite(half_life_days, "half_life_days")
        q = _finite(innovation_scale, "innovation_scale")
        if not 0 <= q <= 1:
            raise ValueError("innovation_scale must be in [0, 1]")
        super().__init__(regularization, half_life_days)
        self.innovation_scale = q

    def _features(self, home, away, competition, season):
        rows = team_features(home, away, competition)
        # Exactly the original dictionaries, with no extra zero columns, at q=0.
        if self.innovation_scale == 0:
            return rows
        last = _season_start(season)
        for row, attacking, defending in ((rows[0], home, away), (rows[1], away, home)):
            for side, team, sign in (("attack", attacking, 1), ("defense", defending, -1)):
                first = _season_start(self.first_season[team])
                for year in range(first+1, last+1):
                    row[_innovation_key(side, team, year)] = sign*self.innovation_scale
        return rows

    def fit(self, records, before_date):
        if type(before_date) is not date:
            raise ValueError("Use an explicit calendar date, not a fabricated timestamp")
        history = []
        for record in records:
            if type(record.match_date) is not date:
                raise ValueError("Historical match dates must be explicit calendar dates")
            if record.match_date < before_date and record.goal_fit_exclusion is None:
                history.append(record)
        history.sort(key=lambda r: (r.match_date, r.match_id))
        if not history:
            raise ValueError("No eligible earlier-date results")
        seen, first, last = set(), {}, {}
        for record in history:
            if not isinstance(record.match_id, str) or not record.match_id:
                raise ValueError("Need nonempty historical match identifiers")
            if record.match_id in seen:
                raise ValueError("Duplicate match identifiers in training history")
            seen.add(record.match_id)
            if (any(not isinstance(s, str) or not s for s in (record.home, record.away, record.competition))
                    or record.home == record.away):
                raise ValueError("Need distinct historical teams and a competition")
            year = _season_start(record.season)
            if record.season != season_for_date(record.match_date):
                raise ValueError("Historical season label disagrees with July-June match date")
            if (not isinstance(record.full_time, (list, tuple)) or len(record.full_time) != 2
                    or any(type(goal) is not int or goal < 0 for goal in record.full_time)):
                raise ValueError("Need two nonnegative integer full-time goals")
            for team in (record.home, record.away):
                first[team] = min(first.get(team, year), year)
                last[team] = max(last.get(team, year), year)
        self.first_season = {team: _season_label(first[team]) for team in sorted(first)}
        self.last_season = {team: _season_label(last[team]) for team in sorted(last)}
        features, target, weights = [], [], []
        for record in history:
            features.extend(self._features(record.home, record.away, record.competition, record.season))
            target.extend(record.full_time)
            weight = math.exp(-math.log(2)*(before_date-record.match_date).days/self.half_life_days)
            weights.extend((weight, weight))
        # Same fitting function, sample-weight normalization, intercept, alpha,
        # solver and tolerance as the original; only these feature columns differ.
        vectorizer, model = fit_weighted_poisson(features, target, weights, self.regularization)
        self.coefficients = dict(zip(vectorizer.get_feature_names_out(), map(float, model.coef_)))
        self.intercept = float(model.intercept_)
        self.before_date = before_date
        self.teams = set(first)
        self.competitions = {r.competition for r in history}
        self.n_matches = len(history)
        self.training_min_date, self.training_max_date = history[0].match_date, history[-1].match_date
        self.reference_match_ids = [r.match_id for r in history]
        self.optimizer_iterations = int(model.n_iter_)
        return self

    def predict(self, home, away, competition, match_date):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit earlier-date results before predicting")
        if type(match_date) is not date or match_date < self.before_date:
            raise ValueError("Prediction must not precede the fit cutoff date")
        if home == away:
            raise ValueError("Teams must differ")
        if {home, away}-self.teams:
            raise ValueError("Unseen teams require an explicitly estimated prior")
        if competition not in self.competitions:
            raise ValueError("Unseen competition")
        rows = self._features(home, away, competition, season_for_date(match_date))
        return tuple(math.exp(self.intercept+sum(self.coefficients.get(k, 0)*v for k, v in row.items()))
                     for row in rows)

    def parameters(self):
        payload = super().parameters()
        payload.update({
            "model_type": "date_level_cumulative_season_poisson",
            "innovation_scale": self.innovation_scale,
            "first_season": dict(self.first_season), "last_season": dict(self.last_season),
            "coefficients": dict(self.coefficients), "feature_names": list(self.coefficients),
            "reference_match_ids": list(self.reference_match_ids),
            "season_convention": "YYYY-YY; July 1 inclusive to June 30 inclusive",
            "state_definition": "base plus q times each annual innovation after team first historical season",
            "innovation_penalty": "same alpha ridge per coefficient; q is relative feature/prior scale, not measured volatility",
            "absent_seasons": "annual increments retained even if their columns are collinear",
            "unobserved_innovation": "MAP conditional mean zero, no uncertainty integration or assertion of unchanged strength",
            "not_for": "precise T-24h, confirmed lineups, state posterior integration or unvalidated current forecasts",
        })
        return payload

    @classmethod
    def from_parameters(cls, payload):
        """Restore saved algebra, without fitting or inferring missing columns."""
        if not isinstance(payload, Mapping) or payload.get("model_type") != "date_level_cumulative_season_poisson":
            raise ValueError("Need saved cumulative season model parameters")
        try:
            instance = cls(payload["regularization"], payload["half_life_days"], payload["innovation_scale"])
            dates = {k: date.fromisoformat(payload[k]) for k in
                     ("before_date", "training_min_date", "training_max_date")}
            if any(payload[k] != value.isoformat() for k, value in dates.items()):
                raise ValueError("Saved dates must use canonical calendar dates")
            if not dates["training_min_date"] <= dates["training_max_date"] < dates["before_date"]:
                raise ValueError("Saved training dates must strictly precede cutoff")
            teams, competitions = payload["teams"], payload["competitions"]
            for label, values in (("teams", teams), ("competitions", competitions)):
                if (not isinstance(values, list) or not values
                        or any(not isinstance(v, str) or not v for v in values)
                        or values != sorted(set(values))):
                    raise ValueError(f"Saved {label} must be nonempty sorted unique labels")
            first, last = payload["first_season"], payload["last_season"]
            if not isinstance(first, Mapping) or not isinstance(last, Mapping) or set(first) != set(teams) or set(last) != set(teams):
                raise ValueError("Saved first/last seasons must cover exactly the trained teams")
            lower = _season_start(season_for_date(dates["training_min_date"]))
            upper = _season_start(season_for_date(dates["training_max_date"]))
            for team in teams:
                if not lower <= _season_start(first[team]) <= _season_start(last[team]) <= upper:
                    raise ValueError("Saved team season bounds disagree with training dates")
            if min(map(_season_start, first.values())) != lower or max(map(_season_start, last.values())) != upper:
                raise ValueError("Saved team season bounds omit a training boundary")
            names, coefficients = payload["feature_names"], payload["coefficients"]
            expected = {f"{side}:{team}" for team in teams for side in ("attack", "defense")}
            expected |= {f"{side}:{comp}" for comp in competitions for side in ("competition", "home")}
            if instance.innovation_scale != 0:
                expected |= {_innovation_key(side, team, year) for team in teams for side in ("attack", "defense")
                             for year in range(_season_start(first[team])+1, _season_start(last[team])+1)}
            if (not isinstance(coefficients, Mapping) or not isinstance(names, list)
                    or names != sorted(expected) or set(coefficients) != expected):
                raise ValueError("Saved feature names/coefficients disagree with cumulative season design")
            coefficients = {name: _finite(coefficients[name], "coefficient") for name in names}
            n, iterations, ids = payload["n_matches"], payload["optimizer_iterations"], payload["reference_match_ids"]
            if type(n) is not int or n < 1 or type(iterations) is not int or iterations < 0:
                raise ValueError("Saved training counts/iterations must be valid integers")
            if (not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n
                    or any(not isinstance(mid, str) or not mid for mid in ids)):
                raise ValueError("Saved reference IDs must uniquely cover training matches")
            if payload["basis"] != "team_marginal" or payload["rho"] != 0:
                raise ValueError("Saved model must retain the independent Poisson team basis")
            instance.coefficients, instance.intercept = coefficients, _finite(payload["intercept"], "intercept")
            instance.before_date = dates["before_date"]
            instance.training_min_date, instance.training_max_date = dates["training_min_date"], dates["training_max_date"]
            instance.teams, instance.competitions = set(teams), set(competitions)
            instance.first_season, instance.last_season = dict(first), dict(last)
            instance.n_matches, instance.optimizer_iterations = n, iterations
            instance.reference_match_ids = list(ids)
            return instance
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(f"Malformed saved cumulative season parameters: {error}") from error
