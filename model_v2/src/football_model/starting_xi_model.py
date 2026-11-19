"""Full-match goal means conditional on a known starting XI, not actual paths.

This is a separate research estimator, not an adapter for retrospective stint
coefficients. Complete regulation full-time goals (including ordinary stoppage)
are fitted directly. Starting-XI frequencies, not played minutes, define each
team's fitted reference. No observed substitutions, score/man paths, match
durations, bench lists, market probabilities or old player coefficients enter.

Historical starting-XI metadata alone does not establish that it was publicly
available at a historical pre-kickoff timestamp. This module deliberately uses
calendar dates; trusted publication-time reconstruction belongs to its caller.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


class UnknownStartingXIIdentityError(ValueError):
    """A team/player has no fitted full-match effect or explicitly fitted prior."""


def _calendar(value, name):
    if type(value) is date:
        return value
    if isinstance(value, str) and len(value) == 10:
        try:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() == value:
                return parsed
        except ValueError:
            pass
    raise ValueError(f"{name} must be a calendar date, without an invented timestamp")


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _identity_source(value, kind):
    """Read an explicit provider namespace; numeric IDs never bridge providers."""
    if isinstance(value, str):
        for source in ("impect", "statsbomb"):
            prefix = f"{source}:{kind}:"
            if value.startswith(prefix) and value[len(prefix):].strip():
                return source
    raise ValueError(f"{kind}_id requires a nonempty namespaced impect:{kind}: or statsbomb:{kind}: identity")


def _single_source(match_ids, player_ids=()):
    """Infer the sole provider from existing identities, without new payload fields."""
    sources = {_identity_source(value, "match") for value in match_ids}
    sources.update(_identity_source(value, "player") for value in player_ids)
    if len(sources) != 1:
        raise ValueError("A model must use exactly one source namespace; mixed providers have colliding integer team IDs")
    return sources.pop()


def _match_id(value):
    _identity_source(value, "match")
    return value


def _starters(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 11:
        raise ValueError(f"{name} must contain exactly 11 starting player IDs")
    for player in value:
        try:
            _identity_source(player, "player")
        except ValueError as error:
            raise ValueError(f"{name} requires namespaced impect:player: or statsbomb:player: identities") from error
    if len(set(value)) != 11:
        raise ValueError(f"{name} must contain 11 distinct starting player IDs")
    return tuple(value)


def _fixture(value):
    """Project exactly six pre-match fields; no target/path fields are read."""
    if not isinstance(value, Mapping):
        raise ValueError("A starting-XI fixture mapping is required")
    teams = tuple(value.get(key) for key in ("home_team_id", "away_team_id"))
    if any(type(team) is not int or team < 1 for team in teams) or teams[0] == teams[1]:
        raise ValueError("Need two distinct positive integer team IDs")
    home = _starters(value.get("home_starters"), "home_starters")
    away = _starters(value.get("away_starters"), "away_starters")
    if set(home) & set(away):
        raise ValueError("A starting player cannot be on both teams")
    match_id = _match_id(value.get("match_id"))
    _single_source((match_id,), home + away)
    return {"match_id": match_id,
            "date": _calendar(value.get("match_date"), "match_date"),
            "home": teams[0], "away": teams[1], "home_starters": home, "away_starters": away}


def _training_match(value):
    result = _fixture(value)
    target = value.get("full_time")
    if (not isinstance(target, (list, tuple)) or len(target) != 2
            or any(type(goal) is not int or goal < 0 for goal in target)):
        raise ValueError("full_time requires two nonnegative integer full-match goals")
    result["target"] = tuple(target)
    return result


def _penalized_poisson_objective(coefficients, design, target, penalties):
    """Summed two-sided FT NLL, without factorial constants, and its gradient."""
    eta = np.asarray(design @ coefficients).ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.exp(eta)
        objective = np.sum(means - target * eta) + .5 * np.dot(penalties * coefficients, coefficients)
        gradient = np.asarray(design.T @ (means - target)).ravel() + penalties * coefficients
    return float(objective), gradient


class FullMatchStartingXIModel:
    """Direct full-time Poisson means from team identity and starting XI.

    Ridge values multiply the SUMMED two-sided full-match loss; they are not
    sklearn's mean-loss alpha. Player ridge 100 is only an externally chosen
    earlier-research anchor, not a validated optimum for this new target/model.
    """

    def __init__(self, *, include_players=True, team_ridge=1.0, player_ridge=100.0,
                 home_ridge=1.0, max_iterations=2000):
        if type(include_players) is not bool:
            raise ValueError("include_players must be boolean")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.include_players = include_players
        self.team_ridge = _positive_number(team_ridge, "team_ridge")
        self.player_ridge = _positive_number(player_ridge, "player_ridge")
        self.home_ridge = _positive_number(home_ridge, "home_ridge")
        self.max_iterations = max_iterations

    def _expected_feature_names(self):
        names = ["intercept", "home_advantage"]
        names += [f"team_{kind}:{team}" for kind in ("attack", "defence") for team in sorted(self.teams)]
        if self.include_players:
            names += [f"player_{kind}:{player}" for kind in ("attack", "defence") for player in sorted(self.players)]
        return names

    def _expected_penalties(self):
        return np.asarray([0.0 if name == "intercept" else self.player_ridge
                           if name.startswith("player_") else self.team_ridge
                           if name.startswith("team_") else self.home_ridge
                           for name in self.feature_names])

    def _features(self, match):
        home, away = match["home"], match["away"]
        output = [
            {"intercept": 1.0, "home_advantage": 1.0,
             f"team_attack:{home}": 1.0, f"team_defence:{away}": -1.0},
            {"intercept": 1.0, f"team_attack:{away}": 1.0, f"team_defence:{home}": -1.0},
        ]
        if self.include_players:
            for team, starters, own, opponent in (
                (home, match["home_starters"], 0, 1), (away, match["away_starters"], 1, 0),
            ):
                current, reference = set(starters), self.reference_frequencies[team]
                for player in current | reference.keys():
                    deviation = float(player in current) - reference.get(player, 0.0)
                    if deviation:
                        output[own][f"player_attack:{player}"] = deviation
                        output[opponent][f"player_defence:{player}"] = -deviation
        return output

    def _design(self, matches):
        rows, columns, values = [], [], []
        for match_index, match in enumerate(matches):
            for side, features in enumerate(self._features(match)):
                row = 2 * match_index + side
                for feature, value in features.items():
                    if value:
                        rows.append(row)
                        columns.append(self.feature_index[feature])
                        values.append(value)
        return sparse.csr_matrix((values, (rows, columns)), shape=(2 * len(matches), len(self.feature_names)))

    def fit(self, rows: Sequence[Mapping], before_date: date):
        if type(before_date) is not date:
            raise ValueError("before_date must be a calendar date")
        # A failed refit must not leave old coefficients paired with a new
        # feature/reference population on this same instance.
        if hasattr(self, "coefficients"):
            del self.coefficients
        history = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Training rows must be mappings")
            # Future targets, XI identities, teams, bench and paths are not read.
            if _calendar(row.get("match_date"), "match_date") < before_date:
                history.append(_training_match(row))
        if not history:
            raise ValueError("No strictly earlier full-match starting-XI observations")
        if len({match["match_id"] for match in history}) != len(history):
            raise ValueError("Duplicate match IDs in the training window")
        self.identity_source = _single_source(match["match_id"] for match in history)
        history.sort(key=lambda match: (match["date"], match["match_id"]))
        self.before_date = before_date
        self.training_min_date, self.training_max_date = history[0]["date"], history[-1]["date"]
        self.reference_match_ids = [match["match_id"] for match in history]
        self.n_matches = len(history)
        self.teams = {team for match in history for team in (match["home"], match["away"])}
        appearances, counts = defaultdict(Counter), Counter()
        for match in history:
            for team, column in ((match["home"], "home_starters"), (match["away"], "away_starters")):
                counts[team] += 1
                appearances[team].update(match[column])
        self.team_match_counts = dict(counts)
        self.reference_frequencies = {
            team: {player: appearances[team][player] / counts[team] for player in sorted(appearances[team])}
            for team in sorted(self.teams)
        }
        self.players = {player for appearances_for_team in appearances.values() for player in appearances_for_team}
        self.feature_names = self._expected_feature_names()
        self.feature_index = {feature: index for index, feature in enumerate(self.feature_names)}
        self.penalties = self._expected_penalties()
        design = self._design(history)
        target = np.asarray([goal for match in history for goal in match["target"]], dtype=float)
        if target.sum() <= 0:
            raise ValueError("Positive total goals required for a finite unpenalized intercept")
        initial = np.zeros(len(self.feature_names))
        initial[0] = np.log(target.mean())
        result = minimize(_penalized_poisson_objective, initial,
                          args=(design, target, self.penalties), jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iterations, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Full-match starting-XI fit failed: {result.message}")
        self.coefficients = np.asarray(result.x)
        self.optimizer_diagnostics = {"success": bool(result.success), "iterations": int(result.nit),
                                      "objective": float(result.fun), "message": str(result.message)}
        return self

    def predict(self, *, match_id, match_date, home_team_id, away_team_id, home_starters, away_starters):
        """Explicit fixture-only API; no completed goal/path argument is accepted."""
        return self.predict_match({"match_id": match_id, "match_date": match_date,
                                   "home_team_id": home_team_id, "away_team_id": away_team_id,
                                   "home_starters": home_starters, "away_starters": away_starters})

    def predict_match(self, fixture):
        """Use only six fixture/XI fields; all extra target/path keys are ignored.

        Accepting a larger caller record does not grant access to its full_time,
        stints, actual minutes, bench or diagnostics. Those fields are neither
        validated nor projected into the design; scoring remains separate.
        """
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit earlier full-match observations before predicting")
        match = _fixture(fixture)
        if _identity_source(match["match_id"], "match") != self.identity_source:
            raise ValueError("Prediction source namespace must match the fitted history")
        if match["date"] < self.before_date:
            raise ValueError("Prediction date must be at or after the fitted cutoff")
        missing_teams = {match["home"], match["away"]} - self.teams
        if missing_teams:
            raise UnknownStartingXIIdentityError(f"Unknown teams without a fitted prior: {sorted(missing_teams)}")
        if self.include_players:
            missing = (set(match["home_starters"]) | set(match["away_starters"])) - self.players
            if missing:
                raise UnknownStartingXIIdentityError(f"Unknown starting players without a fitted prior: {sorted(missing)}")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            means = np.exp(np.asarray(self._design([match]) @ self.coefficients).ravel())
        if not np.all(np.isfinite(means)) or np.any(means <= 0):
            raise FloatingPointError("Nonfinite or zero full-match mean; no hidden clipping applied")
        return {"match_id": match["match_id"], "match_date": match["date"].isoformat(),
                "home_mean": float(means[0]), "away_mean": float(means[1])}

    def parameters(self):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit before exporting full-match starting-XI parameters")
        return {
            "model": "FullMatchStartingXIModel", "include_players": self.include_players,
            "team_ridge": self.team_ridge, "player_ridge": self.player_ridge,
            "home_ridge": self.home_ridge, "max_iterations": self.max_iterations,
            "before_date": self.before_date.isoformat(), "training_min_date": self.training_min_date.isoformat(),
            "training_max_date": self.training_max_date.isoformat(), "n_matches": self.n_matches,
            "reference_match_ids": list(self.reference_match_ids), "teams": sorted(self.teams),
            "players": sorted(self.players),
            "team_match_counts": {str(team): count for team, count in sorted(self.team_match_counts.items())},
            "reference_frequencies": {str(team): dict(values) for team, values in sorted(self.reference_frequencies.items())},
            "feature_names": list(self.feature_names), "coefficients": self.coefficients.tolist(),
            "penalties": self.penalties.tolist(), "optimizer": dict(self.optimizer_diagnostics),
            "response": "two-sided complete full-time goals including ordinary stoppage; no extra time or shootout",
            "loss": "summed two-sided full-match Poisson NLL up to goal-factorial constants + sum(ridge_j*coefficient_j^2)/2",
            "offset": "none; complete-match means, no artificial 90-minute exposure",
            "reference_basis": "same training-window per-team starting-XI appearance frequency; not actual minutes",
            "penalty_provenance": "player ridge 100 is an earlier-research external anchor, not an optimized full-match XI penalty",
            "limits": "conditional starting-XI association, not causal player value; date availability assumed, publication times not reconstructed",
        }

    @classmethod
    def from_parameters(cls, payload):
        if not isinstance(payload, Mapping) or payload.get("model") != "FullMatchStartingXIModel":
            raise ValueError("Not full-match starting-XI parameters")
        model = cls(**{key: payload[key] for key in ("include_players", "team_ridge", "player_ridge",
                                                  "home_ridge", "max_iterations")})
        model.before_date = _calendar(payload["before_date"], "before_date")
        model.training_min_date = _calendar(payload["training_min_date"], "training_min_date")
        model.training_max_date = _calendar(payload["training_max_date"], "training_max_date")
        if not model.training_min_date <= model.training_max_date < model.before_date:
            raise ValueError("Restored training dates violate the historical cutoff")
        teams, players = payload["teams"], payload["players"]
        if (not isinstance(teams, list) or len(teams) < 2
                or any(type(team) is not int or team < 1 for team in teams) or len(set(teams)) != len(teams)
                or not isinstance(players, list) or not players
                or any(not isinstance(player, str) for player in players)
                or len(set(players)) != len(players)):
            raise ValueError("Invalid saved team/player identities")
        model.teams, model.players = set(teams), set(players)
        model.n_matches = payload["n_matches"]
        identifiers = payload["reference_match_ids"]
        if (type(model.n_matches) is not int or model.n_matches < 1
                or not isinstance(identifiers, list) or len(identifiers) != model.n_matches
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError("Invalid saved training match count or identifiers")
        model.reference_match_ids = [_match_id(mid) for mid in identifiers]
        model.identity_source = _single_source(model.reference_match_ids, players)
        counts, references = payload["team_match_counts"], payload["reference_frequencies"]
        if not isinstance(counts, Mapping) or not isinstance(references, Mapping):
            raise ValueError("Invalid saved reference frequencies")
        model.team_match_counts = {int(team): count for team, count in counts.items()}
        model.reference_frequencies = {int(team): dict(values) for team, values in references.items()}
        if (len(model.team_match_counts) != len(counts) or len(model.reference_frequencies) != len(references)
                or set(model.team_match_counts) != model.teams or set(model.reference_frequencies) != model.teams
                or any(type(count) is not int or count < 1 for count in model.team_match_counts.values())
                or sum(model.team_match_counts.values()) != 2 * model.n_matches):
            raise ValueError("Invalid saved reference team counts")
        for team, values in model.reference_frequencies.items():
            count = model.team_match_counts[team]
            if (len(values) < 11 or any(player not in model.players or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value <= 1
                    or not isclose(value * count, round(value * count), rel_tol=0, abs_tol=1e-7)
                    for player, value in values.items())
                    or not isclose(sum(values.values()), 11.0, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("Invalid saved starting-XI reference frequencies")
        if {player for values in model.reference_frequencies.values() for player in values} != model.players:
            raise ValueError("Saved player identities disagree with reference frequencies")
        model.feature_names = list(payload["feature_names"])
        model.feature_index = {name: index for index, name in enumerate(model.feature_names)}
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        model.penalties = np.asarray(payload["penalties"], dtype=float)
        if (model.feature_names != model._expected_feature_names()
                or model.coefficients.ndim != 1 or model.penalties.ndim != 1
                or len(model.coefficients) != len(model.feature_names) or len(model.penalties) != len(model.feature_names)
                or not np.all(np.isfinite(model.coefficients)) or not np.all(np.isfinite(model.penalties))
                or not np.array_equal(model.penalties, model._expected_penalties())):
            raise ValueError("Invalid saved feature dimensions, coefficients or penalties")
        optimizer = payload["optimizer"]
        if (not isinstance(optimizer, Mapping) or optimizer.get("success") is not True
                or type(optimizer.get("iterations")) is not int or optimizer["iterations"] < 0
                or isinstance(optimizer.get("objective"), bool)
                or not isinstance(optimizer.get("objective"), (int, float)) or not isfinite(optimizer["objective"])
                or not isinstance(optimizer.get("message"), str)):
            raise ValueError("Invalid saved optimizer diagnostics")
        model.optimizer_diagnostics = dict(optimizer)
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.parameters(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_parameters(json.loads(Path(path).read_text(encoding="utf-8")))
