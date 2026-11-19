"""Date-level joint team/player conditional-intensity research model.

This model conditions on an actual historical on-pitch/state path. Its results
are NOT pre-match 1X2 probabilities. Period clocks include recorded stoppage
time; no UTC kickoff or historical availability timestamp is invented.

Only log(duration_seconds / 5400) is fixed. Teams, home advantage, player
deviations and start-of-stint state effects are estimated jointly. Player
references use only the fitted window's exposure. All coefficients except the
intercept have explicit ridge penalties; this also resolves redundant team
parameter shifts. Coefficients are conditional associations, not causal values.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


class UnknownStintIdentityError(ValueError):
    """A model has no fitted identity effect or explicitly estimated prior."""


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
    raise ValueError(f"{name} must be a calendar date, without an invented timezone")


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _players(value, name):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} needs nonempty actual on-pitch IDs")
    if any(not isinstance(p, str) or not p for p in value) or len(set(value)) != len(value):
        raise ValueError(f"{name} needs distinct nonempty player IDs")
    return tuple(value)


def _match(value, response_kind):
    mid = value.get("match_id")
    if not isinstance(mid, str) or not mid:
        raise ValueError("match_id must be a nonempty stable string")
    day = _calendar(value.get("match_date"), "match_date")
    teams = tuple(value.get(k) for k in ("home_team_id", "away_team_id"))
    if any(type(t) is not int or t < 1 for t in teams) or teams[0] == teams[1]:
        raise ValueError("Need two distinct positive integer team IDs")
    values = value.get("stints")
    if not isinstance(values, list) or not values:
        raise ValueError("Nonempty parsed stints are required")
    normalized, seen, previous = [], set(), {}
    suffix = "goals" if response_kind == "goals" else "xg"
    for row in values:
        sid = row.get("stint_id")
        period = row.get("period")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("Duplicate or missing stint_id")
        if type(period) is not int or period not in {1, 2}:
            raise ValueError("Only ordinary two-half period IDs are supported")
        seen.add(sid)
        start = _number(row.get("start_second"), "start_second", nonnegative=True)
        end = _number(row.get("end_second"), "end_second", positive=True)
        duration = _number(row.get("duration_seconds"), "duration_seconds", positive=True)
        if end <= start or not isclose(end - start, duration, abs_tol=1e-7):
            raise ValueError("Stint duration must equal its positive period-clock interval")
        if period in previous and start < previous[period] - 1e-7:
            raise ValueError("Stints overlap or are not in period-clock order")
        previous[period] = end
        home = _players(row.get("home_players"), "home_players")
        away = _players(row.get("away_players"), "away_players")
        if set(home) & set(away):
            raise ValueError("A player cannot be on both teams in the same stint")
        target = tuple(_number(row.get(f"{side}_{suffix}"), f"{side}_{suffix}", nonnegative=True)
                       for side in ("home", "away"))
        if response_kind == "goals" and any(y != int(y) for y in target):
            raise ValueError("Fractional goals require a separate xg quasi-likelihood model")
        score = _number(row.get("home_score_difference"), "home_score_difference")
        man = _number(row.get("home_man_difference"), "home_man_difference")
        normalized.append({"stint_id": sid, "period": period, "start_second": start,
                           "end_second": end, "duration_seconds": duration,
                           "home_players": home, "away_players": away,
                           "target": target, "score": score, "man": man})
    return {"match_id": mid, "date": day, "home": teams[0], "away": teams[1],
            "stints": normalized}


class JointStintIntensityModel:
    def __init__(self, *, include_players=True, response_kind="goals", team_ridge=1.0,
                 player_ridge=10.0, context_ridge=1.0, max_iterations=2000):
        if type(include_players) is not bool:
            raise ValueError("include_players must be boolean")
        if response_kind not in {"goals", "xg"}:
            raise ValueError("response_kind must be goals or xg")
        for name, value in (("team_ridge", team_ridge), ("player_ridge", player_ridge),
                            ("context_ridge", context_ridge)):
            _number(value, name, positive=True)
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.include_players, self.response_kind = include_players, response_kind
        self.team_ridge, self.player_ridge = float(team_ridge), float(player_ridge)
        self.context_ridge, self.max_iterations = float(context_ridge), max_iterations

    def _features(self, match, stint):
        h, a = match["home"], match["away"]
        output = [
            {"intercept": 1.0, "home_advantage": 1.0,
             f"team_attack:{h}": 1.0, f"team_defence:{a}": -1.0,
             "own_score_difference": stint["score"], "own_man_difference": stint["man"]},
            {"intercept": 1.0, f"team_attack:{a}": 1.0, f"team_defence:{h}": -1.0,
             "own_score_difference": -stint["score"], "own_man_difference": -stint["man"]},
        ]
        if self.include_players:
            for team, current, own, opponent in (
                (h, stint["home_players"], 0, 1), (a, stint["away_players"], 1, 0)):
                current = set(current)
                reference = self.reference_fractions[team]
                for player in current | reference.keys():
                    delta = float(player in current) - reference.get(player, 0.0)
                    if delta:
                        output[own][f"player_attack:{player}"] = delta
                        output[opponent][f"player_defence:{player}"] = -delta
        return output

    def _design(self, matches):
        ri, ci, values, exposures, targets, metadata = [], [], [], [], [], []
        for match in matches:
            for stint in match["stints"]:
                for side, features in enumerate(self._features(match, stint)):
                    row = len(exposures)
                    for feature, value in features.items():
                        if value:
                            ri.append(row); ci.append(self.feature_index[feature]); values.append(value)
                    exposures.append(stint["duration_seconds"] / 5400.0)
                    targets.append(stint["target"][side])
                    metadata.append((match["match_id"], stint["stint_id"], side))
        matrix = sparse.csr_matrix((values, (ri, ci)), shape=(len(exposures), len(self.feature_names)))
        return matrix, np.asarray(exposures), np.asarray(targets), metadata

    def fit(self, matches: Sequence[dict], before_date: date):
        if type(before_date) is not date:
            raise ValueError("before_date must be a calendar date")
        # Inspect only dates before selecting. Future player identities, targets
        # and lineup fractions never enter the fitted window.
        history = [_match(m, self.response_kind) for m in matches
                   if _calendar(m.get("match_date"), "match_date") < before_date]
        if not history:
            raise ValueError("No earlier historical matches")
        if len({m["match_id"] for m in history}) != len(history):
            raise ValueError("Duplicate match IDs in training window")
        history.sort(key=lambda m: (m["date"], m["match_id"]))
        self.before_date = before_date
        self.training_min_date, self.training_max_date = history[0]["date"], history[-1]["date"]
        self.reference_match_ids = [m["match_id"] for m in history]
        self.teams = {t for m in history for t in (m["home"], m["away"])}
        seconds, totals = defaultdict(lambda: defaultdict(float)), defaultdict(float)
        self.players = set()
        for match in history:
            for stint in match["stints"]:
                for team, column in ((match["home"], "home_players"), (match["away"], "away_players")):
                    totals[team] += stint["duration_seconds"]
                    for player in stint[column]:
                        seconds[team][player] += stint["duration_seconds"]
                        self.players.add(player)
        self.reference_fractions = {t: {p: s / totals[t] for p, s in seconds[t].items()}
                                    for t in self.teams}
        self.feature_names = ["intercept", "home_advantage", "own_score_difference", "own_man_difference"]
        self.feature_names += [f"team_{kind}:{t}" for kind in ("attack", "defence") for t in sorted(self.teams)]
        if self.include_players:
            self.feature_names += [f"player_{kind}:{p}" for kind in ("attack", "defence") for p in sorted(self.players)]
        self.feature_index = {name: index for index, name in enumerate(self.feature_names)}
        penalties = np.asarray([0.0 if name == "intercept" else self.player_ridge
                               if name.startswith("player_") else self.team_ridge
                               if name.startswith("team_") else self.context_ridge
                               for name in self.feature_names])
        design, exposure, target, _ = self._design(history)
        if target.sum() <= 0:
            raise ValueError("Positive total response required for a finite unpenalized intercept")

        def objective(coefficients):
            eta = np.asarray(design @ coefficients).ravel()
            with np.errstate(over="ignore", invalid="ignore"):
                mean = exposure * np.exp(eta)
                loss = np.sum(mean - target * eta) + .5 * np.dot(penalties * coefficients, coefficients)
                gradient = np.asarray(design.T @ (mean - target)).ravel() + penalties * coefficients
            return float(loss), gradient

        initial = np.zeros(len(self.feature_names))
        initial[0] = np.log(target.sum() / exposure.sum())
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iterations, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Joint conditional-intensity fit failed: {result.message}")
        self.coefficients = np.asarray(result.x)
        self.penalties = penalties
        self.n_matches, self.n_stints = len(history), len(target) // 2
        self.optimizer_diagnostics = {"success": bool(result.success), "iterations": int(result.nit),
                                      "objective": float(result.fun), "message": str(result.message)}
        return self

    def predict_match(self, parsed_match):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit a historical window before predicting")
        match = _match(parsed_match, self.response_kind)
        if match["date"] < self.before_date:
            raise ValueError("Prediction date must be at or after the fitted cutoff")
        missing_teams = {match["home"], match["away"]} - self.teams
        if missing_teams:
            raise UnknownStintIdentityError(f"Unknown teams without a fitted prior: {sorted(missing_teams)}")
        if self.include_players:
            missing = {p for s in match["stints"] for column in ("home_players", "away_players")
                       for p in s[column]} - self.players
            if missing:
                raise UnknownStintIdentityError(f"Unknown players without a fitted prior: {sorted(missing)}")
        design, exposure, target, _ = self._design([match])
        eta = np.asarray(design @ self.coefficients).ravel()
        rate = np.exp(eta)
        means = exposure * rate
        if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(means)) or np.any(rate <= 0):
            raise FloatingPointError("Nonfinite or zero intensity; no hidden clipping applied")
        loss = means - target * eta
        rows = []
        for index, stint in enumerate(match["stints"]):
            h, a = 2 * index, 2 * index + 1
            rows.append({"stint_id": stint["stint_id"], "period": stint["period"],
                         "duration_seconds": stint["duration_seconds"],
                         "home_intensity_per90": float(rate[h]), "away_intensity_per90": float(rate[a]),
                         "home_expected_target": float(means[h]), "away_expected_target": float(means[a]),
                         "home_actual_target": float(target[h]), "away_actual_target": float(target[a]),
                         "point_process_loss": float(loss[h] + loss[a])})
        return {"match_id": match["match_id"], "match_date": match["date"].isoformat(),
                "response_kind": self.response_kind, "include_players": self.include_players,
                "n_stints": len(rows), "stints": rows,
                "expected_home_target": float(means[::2].sum()),
                "expected_away_target": float(means[1::2].sum()),
                "actual_home_target": float(target[::2].sum()), "actual_away_target": float(target[1::2].sum()),
                "point_process_loss": float(loss.sum()),
                "loss_kind": "point_process_nll_up_to_model_independent_constants" if self.response_kind == "goals"
                             else "xg_point_process_style_quasi_loss_not_probability_likelihood",
                "usage": "Retrospective actual-state conditional intensity, not pre-match 1X2"}

    def parameters(self):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit before exporting parameters")
        return {"model": "JointStintIntensityModel", "include_players": self.include_players,
                "response_kind": self.response_kind, "team_ridge": self.team_ridge,
                "player_ridge": self.player_ridge, "context_ridge": self.context_ridge,
                "max_iterations": self.max_iterations, "before_date": self.before_date.isoformat(),
                "training_min_date": self.training_min_date.isoformat(),
                "training_max_date": self.training_max_date.isoformat(),
                "n_matches": self.n_matches, "n_stints": self.n_stints,
                "reference_match_ids": list(self.reference_match_ids), "teams": sorted(self.teams),
                "players": sorted(self.players),
                "reference_fractions": {str(t): dict(values) for t, values in self.reference_fractions.items()},
                "feature_names": list(self.feature_names), "coefficients": self.coefficients.tolist(),
                "penalties": self.penalties.tolist(), "optimizer": dict(self.optimizer_diagnostics),
                "penalty_scaling": "summed two-sided point-process loss + sum(ridge_j*coefficient_j^2)/2; intercept unpenalized",
                "reference_basis": "same training-window per-team exposure-weighted player fractions",
                "fixed_offset": "log(actual duration_seconds/5400), no marginal team-rate offset",
                "limits": "retrospective state paths only; xg quasi-loss is not a goal distribution; no causal effects or uncertainty claims"}

    @classmethod
    def from_parameters(cls, payload):
        if payload.get("model") != "JointStintIntensityModel":
            raise ValueError("Not joint conditional-intensity parameters")
        model = cls(**{key: payload[key] for key in ("include_players", "response_kind", "team_ridge",
                    "player_ridge", "context_ridge", "max_iterations")})
        model.before_date = _calendar(payload["before_date"], "before_date")
        model.training_min_date = _calendar(payload["training_min_date"], "training_min_date")
        model.training_max_date = _calendar(payload["training_max_date"], "training_max_date")
        if not model.training_min_date <= model.training_max_date < model.before_date:
            raise ValueError("Restored training dates violate the cutoff")
        teams, players = payload["teams"], payload["players"]
        if (not isinstance(teams, list) or len(teams) < 2
                or any(type(t) is not int or t < 1 for t in teams)
                or len(set(teams)) != len(teams)
                or not isinstance(players, list) or not players
                or any(not isinstance(p, str) or not p for p in players)
                or len(set(players)) != len(players)):
            raise ValueError("Invalid saved team/player identities")
        model.teams = set(teams)
        model.players = set(players)
        model.reference_fractions = {int(t): dict(values) for t, values in payload["reference_fractions"].items()}
        model.feature_names = list(payload["feature_names"])
        expected_names = ["intercept", "home_advantage", "own_score_difference", "own_man_difference"]
        expected_names += [f"team_{kind}:{t}" for kind in ("attack", "defence") for t in sorted(model.teams)]
        if model.include_players:
            expected_names += [f"player_{kind}:{p}" for kind in ("attack", "defence") for p in sorted(model.players)]
        model.feature_index = {name: index for index, name in enumerate(model.feature_names)}
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        model.penalties = np.asarray(payload["penalties"], dtype=float)
        if (model.feature_names != expected_names
                or model.coefficients.ndim != 1 or model.penalties.ndim != 1
                or len(model.feature_index) != len(model.feature_names)
                or len(model.coefficients) != len(model.feature_names)
                or len(model.penalties) != len(model.feature_names)
                or not np.all(np.isfinite(model.coefficients)) or not np.all(np.isfinite(model.penalties))
                or np.any(model.penalties < 0) or set(model.reference_fractions) != model.teams):
            raise ValueError("Invalid saved parameter dimensions or values")
        if any(p not in model.players or not isfinite(v) or not 0 <= v <= 1
               for values in model.reference_fractions.values() for p, v in values.items()):
            raise ValueError("Invalid saved reference fractions")
        model.n_matches, model.n_stints = payload["n_matches"], payload["n_stints"]
        model.reference_match_ids = list(payload["reference_match_ids"])
        model.optimizer_diagnostics = dict(payload["optimizer"])
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.parameters(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_parameters(json.loads(Path(path).read_text(encoding="utf-8")))
