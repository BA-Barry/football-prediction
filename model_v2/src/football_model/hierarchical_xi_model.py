"""Fixed-variance crossed role/player full-match Poisson research model.

Each match supplies its *initial* player roles. No permanent player role, actual
minutes, future substitutions, score/man paths or existing player coefficients
are inferred. Role counts and player presence are centered on the same earlier
team history. This is fixed prior variance sharing, not learned variance
components or causal player identification.

Known coefficients use a full-Hessian Laplace approximation at their joint MAP.
Unseen individual attack/defence residuals retain their independent Gaussian
priors. The returned bivariate log-intensity normal must be integrated into
joint score probabilities; Poisson of an averaged intensity is not equivalent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import minimize

from .starting_xi_model import (
    _calendar, _fixture, _identity_source, _match_id, _penalized_poisson_objective,
    _positive_number, _single_source, _training_match,
)


ROLES = ("GK", "DEF", "MID", "FWD")


class UnknownHierarchicalTeamError(ValueError):
    """A team has no fitted full-match core or explicitly estimated team prior."""


def _role_fixture(value):
    """Read only six fixture fields plus initial-XI role mappings."""
    match = _fixture(value)
    for side in ("home", "away"):
        roles = value.get(f"{side}_roles")
        if (not isinstance(roles, Mapping) or set(roles) != set(match[f"{side}_starters"])
                or any(role not in ROLES for role in roles.values())):
            raise ValueError(f"{side}_roles must map exactly its 11 starters to GK/DEF/MID/FWD")
        if sum(role == "GK" for role in roles.values()) != 1:
            raise ValueError(f"{side}_roles must identify exactly one initial goalkeeper")
        match[f"{side}_roles"] = dict(roles)
    return match


def _role_training_match(value):
    match = _role_fixture(value)
    match["target"] = _training_match(value)["target"]
    return match


class HierarchicalStartingXIModel:
    def __init__(self, *, include_players=True, prior_sd=.10, team_ridge=1.0,
                 home_ridge=1.0, max_iterations=2000):
        if type(include_players) is not bool:
            raise ValueError("include_players must be boolean")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.include_players = include_players
        self.prior_sd = _positive_number(prior_sd, "prior_sd")
        self.team_ridge = _positive_number(team_ridge, "team_ridge")
        self.home_ridge = _positive_number(home_ridge, "home_ridge")
        self.max_iterations = max_iterations

    @property
    def component_variance(self):
        return self.prior_sd ** 2 / 2

    def _expected_feature_names(self):
        names = ["intercept", "home_advantage"]
        names += [f"team_{kind}:{team}" for kind in ("attack", "defence") for team in sorted(self.teams)]
        if self.include_players:
            names += [f"role_{kind}:{role}" for kind in ("attack", "defence") for role in ROLES]
            names += [f"player_{kind}:{player}" for kind in ("attack", "defence") for player in sorted(self.players)]
        return names

    def _expected_penalties(self):
        component_precision = 1 / self.component_variance
        return np.asarray([0.0 if name == "intercept" else component_precision
                           if name.startswith(("player_", "role_")) else self.team_ridge
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
            for side, own, opponent in (("home", 0, 1), ("away", 1, 0)):
                team = match[side]
                role_counts = Counter(match[f"{side}_roles"].values())
                for role in ROLES:
                    delta = role_counts[role] - self.reference_role_counts[team][role]
                    if delta:
                        output[own][f"role_attack:{role}"] = delta
                        output[opponent][f"role_defence:{role}"] = -delta
                current = set(match[f"{side}_starters"])
                reference = self.reference_frequencies[team]
                # Unseen individuals have a zero *prior mean*, not a known
                # zero effect. Their nonzero variance is restored separately
                # in predict_distribution; no target ID enters a fitted column.
                for player in (current & self.players) | reference.keys():
                    delta = float(player in current) - reference.get(player, 0.0)
                    if delta:
                        output[own][f"player_attack:{player}"] = delta
                        output[opponent][f"player_defence:{player}"] = -delta
        return output

    def _design(self, matches):
        rows, columns, values = [], [], []
        for index, match in enumerate(matches):
            for side, features in enumerate(self._features(match)):
                for feature, value in features.items():
                    if value:
                        rows.append(2 * index + side)
                        columns.append(self.feature_index[feature])
                        values.append(value)
        return sparse.csr_matrix((values, (rows, columns)), shape=(2 * len(matches), len(self.feature_names)))

    def fit(self, rows: Sequence[Mapping], before_date: date):
        if type(before_date) is not date:
            raise ValueError("before_date must be a calendar date")
        for attribute in ("coefficients", "precision_cholesky"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        history = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Training rows must be mappings")
            if _calendar(row.get("match_date"), "match_date") < before_date:
                history.append(_role_training_match(row))
        if not history:
            raise ValueError("No strictly earlier full-match role/XI observations")
        if len({match["match_id"] for match in history}) != len(history):
            raise ValueError("Duplicate match IDs in the training window")
        self.identity_source = _single_source(match["match_id"] for match in history)
        history.sort(key=lambda match: (match["date"], match["match_id"]))
        self.before_date = before_date
        self.training_min_date, self.training_max_date = history[0]["date"], history[-1]["date"]
        self.reference_match_ids = [match["match_id"] for match in history]
        self.n_matches = len(history)
        self.teams = {team for match in history for team in (match["home"], match["away"])}
        counts, appearances, role_totals = Counter(), defaultdict(Counter), defaultdict(Counter)
        for match in history:
            for side in ("home", "away"):
                team = match[side]
                counts[team] += 1
                appearances[team].update(match[f"{side}_starters"])
                role_totals[team].update(match[f"{side}_roles"].values())
        self.team_match_counts = dict(counts)
        self.reference_frequencies = {
            team: {player: appearances[team][player] / counts[team] for player in sorted(appearances[team])}
            for team in sorted(self.teams)
        }
        self.reference_role_counts = {
            team: {role: role_totals[team][role] / counts[team] for role in ROLES}
            for team in sorted(self.teams)
        }
        self.players = {player for values in appearances.values() for player in values}
        self.feature_names = self._expected_feature_names()
        self.feature_index = {name: index for index, name in enumerate(self.feature_names)}
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
            raise RuntimeError(f"Hierarchical starting-XI MAP fit failed: {result.message}")
        fitted_means = np.exp(np.asarray(design @ result.x).ravel())
        if not np.all(np.isfinite(fitted_means)) or np.any(fitted_means <= 0):
            raise FloatingPointError("Invalid fitted Poisson means; no hidden clipping applied")
        precision = (design.T @ design.multiply(fitted_means[:, None])).toarray()
        precision += np.diag(self.penalties)
        # A failed factorization remains an explicit failure. No diagonal
        # jitter, covariance clipping or parameter retry changes the model.
        factor = linalg.cholesky(precision, lower=True, check_finite=True)
        self.coefficients = np.asarray(result.x)
        self.precision_cholesky = factor
        self.optimizer_diagnostics = {"success": bool(result.success), "iterations": int(result.nit),
                                      "objective": float(result.fun), "message": str(result.message),
                                      "precision_cholesky_succeeded": True,
                                      "precision_dimension": len(self.feature_names)}
        return self

    def predict_distribution(self, fixture):
        """Project fixture/XI/roles only to a bivariate normal log-intensity.

        Extra completed goals, minutes, substitutions, diagnostics and paths
        are ignored. This method returns distribution inputs, not integrated
        score probabilities; its caller must integrate the conditional scores.
        """
        if not hasattr(self, "coefficients") or not hasattr(self, "precision_cholesky"):
            raise ValueError("Fit earlier full-match role/XI observations before predicting")
        match = _role_fixture(fixture)
        if _identity_source(match["match_id"], "match") != self.identity_source:
            raise ValueError("Prediction source namespace must match the fitted history")
        if match["date"] < self.before_date:
            raise ValueError("Prediction date must be at or after the fitted cutoff")
        missing_teams = {match["home"], match["away"]} - self.teams
        if missing_teams:
            raise UnknownHierarchicalTeamError(f"Unknown teams without a fitted prior: {sorted(missing_teams)}")
        unknown = sorted((set(match["home_starters"]) | set(match["away_starters"])) - self.players)
        design = self._design([match]).toarray()
        mean = design @ self.coefficients
        solved = linalg.solve_triangular(self.precision_cholesky, design.T, lower=True, check_finite=True)
        known_cov = solved.T @ solved
        unknown_cov = np.eye(2) * (len(unknown) * self.component_variance if self.include_players else 0.0)
        covariance = known_cov + unknown_cov
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            map_means = np.exp(mean)
        if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))
                or not np.all(np.isfinite(map_means)) or np.any(map_means <= 0)):
            raise FloatingPointError("Nonfinite log-intensity distribution; no hidden clipping applied")
        return {"match_id": match["match_id"], "match_date": match["date"].isoformat(),
                "log_mean": mean.tolist(), "log_cov": covariance.tolist(), "map_means": map_means.tolist(),
                "known_parameter_log_cov": known_cov.tolist(), "unknown_player_log_cov": unknown_cov.tolist(),
                "unknown_player_ids": unknown,
                "unknown_player_policy": ("independent Gaussian attack/defence residual priors integrated by caller"
                                          if self.include_players else "team-only model has no individual effect"),
                "approximation": "known coefficients: full-Hessian MAP Laplace; unseen residuals: exact Gaussian prior"}

    def parameters(self):
        if not hasattr(self, "coefficients") or not hasattr(self, "precision_cholesky"):
            raise ValueError("Fit before exporting hierarchical starting-XI parameters")
        return {"model": "HierarchicalStartingXIModel", "include_players": self.include_players,
                "prior_sd": self.prior_sd, "team_ridge": self.team_ridge, "home_ridge": self.home_ridge,
                "max_iterations": self.max_iterations, "before_date": self.before_date.isoformat(),
                "training_min_date": self.training_min_date.isoformat(), "training_max_date": self.training_max_date.isoformat(),
                "n_matches": self.n_matches, "reference_match_ids": list(self.reference_match_ids),
                "teams": sorted(self.teams), "players": sorted(self.players),
                "team_match_counts": {str(team): count for team, count in sorted(self.team_match_counts.items())},
                "reference_frequencies": {str(team): dict(values) for team, values in sorted(self.reference_frequencies.items())},
                "reference_role_counts": {str(team): dict(values) for team, values in sorted(self.reference_role_counts.items())},
                "feature_names": list(self.feature_names), "coefficients": self.coefficients.tolist(),
                "penalties": self.penalties.tolist(), "optimizer": dict(self.optimizer_diagnostics),
                "role_groups": list(ROLES), "component_prior_variance": self.component_variance,
                "prior_structure": "independent attack/defence role effects and player residuals, each variance prior_sd^2/2; fixed, not learned",
                "centering": "same earlier team history: starting-player presence frequency and per-match initial-role count mean",
                "role_semantics": "current initial position mapped to role in each match, no permanent player role assignment",
                "loss": "summed two-sided complete full-match Poisson NLL up to factorial constants + sum(penalty*coefficient^2)/2",
                "offset": "none; full-time goals including ordinary stoppage, no extra time/shootout",
                "precision_storage": "same-stem .npz, key precision_cholesky, lower factor of full posterior precision",
                "prediction": "bivariate Gaussian log-intensity; integrate joint conditional Poisson, not Poisson(mean intensity)",
                "limits": ["fixed prior sensitivity, not learned hierarchical variance or causal player value",
                           "known coefficients use local full-Hessian Laplace, not exact posterior or sampling interval",
                           "date-level information assumption; historical official-XI publication times not reconstructed",
                           "constant GK role count contributes no role-group evidence; same-role replacement cancels shared role",
                           "unseen individual residual prior variance is prior_sd^2/2, not the uncentered role-plus-player variance"]}

    @classmethod
    def from_parameters(cls, payload, precision_cholesky):
        if not isinstance(payload, Mapping) or payload.get("model") != "HierarchicalStartingXIModel":
            raise ValueError("Not hierarchical starting-XI parameters")
        model = cls(**{key: payload[key] for key in ("include_players", "prior_sd", "team_ridge", "home_ridge", "max_iterations")})
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
        counts, references, roles = (payload[key] for key in ("team_match_counts", "reference_frequencies", "reference_role_counts"))
        if any(not isinstance(value, Mapping) for value in (counts, references, roles)):
            raise ValueError("Invalid saved reference mappings")
        model.team_match_counts = {int(team): count for team, count in counts.items()}
        model.reference_frequencies = {int(team): dict(values) for team, values in references.items()}
        model.reference_role_counts = {int(team): dict(values) for team, values in roles.items()}
        if (len(model.team_match_counts) != len(counts) or len(model.reference_frequencies) != len(references)
                or len(model.reference_role_counts) != len(roles)
                or any(set(values) != model.teams for values in (model.team_match_counts, model.reference_frequencies, model.reference_role_counts))
                or any(type(count) is not int or count < 1 for count in model.team_match_counts.values())
                or sum(model.team_match_counts.values()) != 2 * model.n_matches):
            raise ValueError("Invalid saved reference team counts")
        for team in model.teams:
            values, count = model.reference_frequencies[team], model.team_match_counts[team]
            if (len(values) < 11 or any(player not in model.players or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value <= 1
                    or not isclose(value * count, round(value * count), rel_tol=0, abs_tol=1e-7)
                    for player, value in values.items())
                    or not isclose(sum(values.values()), 11.0, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("Invalid saved player reference frequencies")
            group = model.reference_role_counts[team]
            if (set(group) != set(ROLES) or any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or not 0 <= value <= 11 for value in group.values())
                    or not isclose(sum(group.values()), 11, rel_tol=0, abs_tol=1e-7)
                    or not isclose(group["GK"], 1, rel_tol=0, abs_tol=1e-7)
                    or any(not isclose(value * count, round(value * count), rel_tol=0, abs_tol=1e-7) for value in group.values())):
                raise ValueError("Invalid saved role reference counts")
        if {player for values in model.reference_frequencies.values() for player in values} != model.players:
            raise ValueError("Saved player identities disagree with references")
        model.feature_names = list(payload["feature_names"])
        model.feature_index = {name: index for index, name in enumerate(model.feature_names)}
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        model.penalties = np.asarray(payload["penalties"], dtype=float)
        dimension = len(model.feature_names)
        if (model.feature_names != model._expected_feature_names()
                or model.coefficients.shape != (dimension,) or model.penalties.shape != (dimension,)
                or not np.all(np.isfinite(model.coefficients)) or not np.all(np.isfinite(model.penalties))
                or not np.array_equal(model.penalties, model._expected_penalties())):
            raise ValueError("Invalid saved feature dimensions, coefficients or penalties")
        factor = np.asarray(precision_cholesky, dtype=float)
        if (factor.shape != (dimension, dimension) or not np.all(np.isfinite(factor))
                or np.any(np.diag(factor) <= 0) or np.any(np.triu(factor, k=1) != 0)):
            raise ValueError("Invalid lower posterior precision Cholesky factor")
        model.precision_cholesky = factor.copy()
        optimizer = payload["optimizer"]
        if (not isinstance(optimizer, Mapping) or optimizer.get("success") is not True
                or type(optimizer.get("iterations")) is not int or optimizer["iterations"] < 0
                or isinstance(optimizer.get("objective"), bool)
                or not isinstance(optimizer.get("objective"), (int, float)) or not isfinite(optimizer["objective"])
                or not isinstance(optimizer.get("message"), str)
                or optimizer.get("precision_cholesky_succeeded") is not True
                or optimizer.get("precision_dimension") != dimension):
            raise ValueError("Invalid saved optimizer or precision diagnostics")
        model.optimizer_diagnostics = dict(optimizer)
        if payload.get("role_groups") != list(ROLES) or payload.get("component_prior_variance") != model.component_variance:
            raise ValueError("Saved role groups or component variance disagree with settings")
        return model

    def save(self, path):
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("Save hierarchical parameters to a .json path with a same-stem .npz factor")
        parameters = self.parameters()
        np.savez_compressed(path.with_suffix(".npz"), precision_cholesky=self.precision_cholesky)
        path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        parameters = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as stored:
            if set(stored.files) != {"precision_cholesky"}:
                raise ValueError("Expected one posterior precision Cholesky factor")
            factor = stored["precision_cholesky"]
        return cls.from_parameters(parameters, factor)
