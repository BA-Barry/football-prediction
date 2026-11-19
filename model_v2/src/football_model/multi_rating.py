"""Time-safe adaptation of Spoho's multi-rating feature representation.

Rating equations and parameter choices are adapted from
spoho-datascience/sparsity-sports-forecasting, models/rating_models/
RatingModels.py and models/evaluation/ELOXGBoostModel.py (Apache-2.0).
Source: https://github.com/spoho-datascience/sparsity-sports-forecasting
Upstream author labels in the source files: Dell-PC. Changes dated 2026-10-26:
pure in-memory API, league/team identities, strict forecast-day cutoffs,
explicit unknown history and chronological shot-adjustment learning.
The upstream license is distributed in licenses/sparsity-sports-forecasting.txt.

Shot adjustment follows the source's two-count XGBoost classifier, not a
handwritten conversion of shots to goals. Unlike its random two-fold fitting,
each July-to-June period uses a classifier trained before that July 1. This
prevents later outcomes from modifying an earlier historical rating feature.
No target odds, lineup values, promotion backfills, files or downloads are used.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import math


UPSTREAM = "https://github.com/spoho-datascience/sparsity-sports-forecasting"
SEED = 20261026
RATING_PARAMETERS = {
    "results": {"c": 10.0, "d": 400.0, "k": 20.0},
    "goals": {"c": 10.0, "d": 400.0, "k0": 10.0, "lambda": 1.25},
    "shots": {"c": 10.0, "d": 400.0, "k": 25.0},
    "shots_on_target": {"c": 10.0, "d": 400.0, "k": 30.0},
    "pi": {"gamma": 0.7, "lambda": 0.035},
}
SHOT_COLUMNS = {
    "shots": ("home_shots", "away_shots"),
    "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
}


def _day(value):
    if isinstance(value, str):
        return date.fromisoformat(value)
    if type(value) is date:
        return value
    raise ValueError("Expected a calendar date or YYYY-MM-DD string")


def _count(value, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a match count")
    try:
        number = float(value)
    except (ValueError, TypeError):
        if optional:
            return None
        raise ValueError("A nonnegative integer score is required") from None
    if not math.isfinite(number):
        if optional:
            return None
        raise ValueError("A finite score is required")
    if number < 0 or not number.is_integer():
        raise ValueError("Match counts must be nonnegative integers")
    return int(number)


def _key(row, side):
    league, team = row["league"], row[f"{side}_team"]
    if not isinstance(league, str) or not league or not isinstance(team, str) or not team:
        raise ValueError("Need nonempty league and team identifiers")
    return league, team


def _period_start(day):
    return date(day.year if day.month >= 7 else day.year - 1, 7, 1)


def _label(row):
    # The upstream class order is away=0, draw=1, home=2.
    return 2 if row["home_goals"] > row["away_goals"] else (
        0 if row["home_goals"] < row["away_goals"] else 1)


def _shot_models(observations):
    """Return chronological adjustment probabilities and fitting provenance."""
    import numpy as np
    from xgboost import XGBClassifier

    periods = sorted({_period_start(r["_date"]) for r in observations})
    probabilities = {}
    splits = []
    for start in periods:
        current = [(i, r) for i, r in enumerate(observations)
                   if _period_start(r["_date"]) == start]
        for metric, columns in SHOT_COLUMNS.items():
            history = [r for r in observations if r["_date"] < start
                       and all(r[c] is not None for c in columns)]
            record = {
                "metric": metric, "prediction_period_start": start.isoformat(),
                "training_count": len(history),
                "training_min_date": min((r["_date"] for r in history), default=None),
                "training_max_date": max((r["_date"] for r in history), default=None),
                "training_classes": sorted({_label(r) for r in history}),
                "fitted": False,
            }
            for field in ("training_min_date", "training_max_date"):
                record[field] = record[field].isoformat() if record[field] else None
            available = [(i, r) for i, r in current
                         if all(r[c] is not None for c in columns)]
            if record["training_classes"] != [0, 1, 2] or not available:
                record["reason"] = ("all_three_training_classes_required"
                                    if record["training_classes"] != [0, 1, 2]
                                    else "no_current_observed_counts")
                splits.append(record)
                continue
            model = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.3,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                random_state=SEED, n_jobs=1, tree_method="hist",
            )
            model.fit(np.asarray([[r[c] for c in columns] for r in history], dtype=float),
                      np.asarray([_label(r) for r in history], dtype=int))
            predicted = model.predict_proba(
                np.asarray([[r[c] for c in columns] for _, r in available], dtype=float))
            for (i, _), p in zip(available, predicted):
                probabilities[i, metric] = float(p[2] + 0.5 * p[1])
            record["fitted"] = True
            record["predicted_observations"] = len(available)
            splits.append(record)
    return probabilities, splits


def _expected(home, away):
    return 1.0 / (1.0 + 10.0 ** ((away - home) / 400.0))


def _pi_goals(rating):
    return math.copysign(10.0 ** (abs(rating) / 3.0) - 1.0, rating)


def _state():
    return {"results": 0.0, "goals": 0.0, "shots": 0.0,
            "shots_on_target": 0.0, "pi_home": 0.0, "pi_away": 0.0,
            "n": 0, "shots_n": 0, "shots_on_target_n": 0,
            "shots_observed": 0, "shots_on_target_observed": 0,
            "last_date": None, "shots_last_date": None,
            "shots_on_target_last_date": None}


def _update(states, row, adjustment):
    h, a = states[_key(row, "home")], states[_key(row, "away")]
    outcome = _label(row) / 2.0
    goal_diff = row["home_goals"] - row["away_goals"]
    for metric, factor in (("results", 20.0),
                           ("goals", 10.0 * (1.0 + abs(goal_diff)) ** 1.25)):
        difference = factor * (outcome - _expected(h[metric], a[metric]))
        h[metric] += difference
        a[metric] -= difference
    expected_diff = (_pi_goals(h["pi_home"]) - _pi_goals(a["pi_away"])) / 2.0
    error = 3.0 * math.log10(abs(goal_diff - expected_diff) + 1.0)
    home_change = 0.035 * (error if expected_diff < goal_diff else -error)
    away_change = 0.035 * (error if expected_diff > goal_diff else -error)
    h["pi_home"] += home_change
    h["pi_away"] += 0.7 * home_change
    a["pi_away"] += away_change
    a["pi_home"] += 0.7 * away_change
    for metric, factor in (("shots", 25.0), ("shots_on_target", 30.0)):
        for side, team in (("home", h), ("away", a)):
            if row[SHOT_COLUMNS[metric][0 if side == "home" else 1]] is not None:
                team[f"{metric}_observed"] += 1
        score = adjustment.get(metric)
        if score is not None:
            difference = factor * (score - _expected(h[metric], a[metric]))
            h[metric] += difference
            a[metric] -= difference
            for team in (h, a):
                team[f"{metric}_n"] += 1
                team[f"{metric}_last_date"] = row["_date"]
    for team in (h, a):
        team["n"] += 1
        team["last_date"] = row["_date"]


def _extract(states, target):
    h, a = states[_key(target, "home")], states[_key(target, "away")]
    features = {}
    for metric in ("results", "goals", "shots", "shots_on_target"):
        count = "n" if metric in ("results", "goals") else f"{metric}_n"
        hr, ar = (h[metric] if h[count] else None), (a[metric] if a[count] else None)
        features[f"mr_{metric}_home"] = hr
        features[f"mr_{metric}_away"] = ar
        features[f"mr_{metric}_diff"] = hr - ar if hr is not None and ar is not None else None
    for side, state in (("home", h), ("away", a)):
        features[f"mr_{side}_history_count"] = float(state["n"])
        features[f"mr_{side}_days_since_history"] = (
            float((target["_cutoff"] - state["last_date"]).days) if state["last_date"] else None)
        for metric in SHOT_COLUMNS:
            features[f"mr_{side}_{metric}_update_count"] = float(state[f"{metric}_n"])
            features[f"mr_{side}_{metric}_observed_count"] = float(state[f"{metric}_observed"])
            last = state[f"{metric}_last_date"]
            features[f"mr_{side}_{metric}_days_since_update"] = (
                float((target["_cutoff"] - last).days) if last else None)
        for venue in ("home", "away"):
            features[f"mr_pi_{side}_{venue}"] = state[f"pi_{venue}"] if state["n"] else None
        features[f"mr_pi_{side}_overall"] = (
            (state["pi_home"] + state["pi_away"]) / 2.0 if state["n"] else None)
    features["mr_pi_expected_goal_diff"] = (
        (_pi_goals(h["pi_home"]) - _pi_goals(a["pi_away"])) / 2.0
        if h["n"] and a["n"] else None)
    return features


def build_rating_features(observations, targets, *, include_adjusted_shots=True):
    """Build features with strict ``observation.date < target.forecast_date``.

    Records use dates, league and stable home_team/away_team identifiers.
    Observations require home_goals/away_goals. Optional home_shots/away_shots
    and home_shots_on_target/away_shots_on_target train shot adjustments. Targets
    require match_id and forecast_date. Input order is preserved in output.
    The complete histories are single chronological streams per league/team;
    one match must occur once in observations, even if it has multiple sources.
    """
    targets = [dict(t, _cutoff=_day(t["forecast_date"])) for t in targets]
    if not targets:
        return {"feature_names": [], "rows": [], "metadata": {"upstream": UPSTREAM,
                "new_shot_adjustment_fits": 0, "shot_adjustment_splits": []}}
    if len({t["match_id"] for t in targets}) != len(targets):
        raise ValueError("Duplicate target match_id")
    for target in targets:
        if _key(target, "home") == _key(target, "away"):
            raise ValueError("Home and away identifiers must differ")
    max_cutoff = max(t["_cutoff"] for t in targets)
    history = []
    seen = set()
    for source in observations:
        day = _day(source["date"])
        if day >= max_cutoff:
            continue
        row = dict(source, _date=day)
        hk, ak = _key(row, "home"), _key(row, "away")
        identity = day, hk, ak
        if hk == ak or identity in seen:
            raise ValueError("Duplicate observation or identical opponents")
        seen.add(identity)
        for field in ("home_goals", "away_goals"):
            row[field] = _count(row[field])
        for columns in SHOT_COLUMNS.values():
            for field in columns:
                row[field] = _count(row.get(field), optional=True)
        history.append(row)
    history.sort(key=lambda r: (r["_date"], r["league"], r["home_team"], r["away_team"]))
    adjusted, splits = _shot_models(history) if include_adjusted_shots else ({}, [])
    states, pointer, rows = defaultdict(_state), 0, {}
    for target in sorted(targets, key=lambda t: (t["_cutoff"], str(t["match_id"]))):
        while pointer < len(history) and history[pointer]["_date"] < target["_cutoff"]:
            _update(states, history[pointer], {
                metric: adjusted.get((pointer, metric)) for metric in SHOT_COLUMNS})
            pointer += 1
        features = _extract(states, target)
        rows[target["match_id"]] = {
            "match_id": target["match_id"], "forecast_date": target["_cutoff"].isoformat(),
            "features": features,
            "history_max_date": history[pointer - 1]["_date"].isoformat() if pointer else None,
        }
    return {
        "feature_names": list(next(iter(rows.values()))["features"]),
        "rows": [rows[t["match_id"]] for t in targets],
        "metadata": {
            "upstream": UPSTREAM, "license": "Apache-2.0",
            "parameter_source": "models/evaluation/ELOXGBoostModel.py",
            "rating_parameters": RATING_PARAMETERS,
            "history_matches_before_last_cutoff": len(history),
            "history_first_date": history[0]["_date"].isoformat() if history else None,
            "history_last_date": history[-1]["_date"].isoformat() if history else None,
            "strict_cutoff": "observation.date < forecast_date; no same-day observations",
            "identity": "league + stable team identifier",
            "new_shot_adjustment_fits": sum(r["fitted"] for r in splits),
            "shot_adjustment_splits": splits,
            "shot_adjustment_parameters": {"n_estimators": 100, "max_depth": 6,
                "learning_rate": 0.3, "n_jobs": 1, "tree_method": "hist", "seed": SEED},
            "source_parameter_differences": [
                "No tuning; upstream rating constants retained",
                "Shot-adjustment random two-fold replaced by pre-July-1 fitting and annual freeze",
                "XGBoost random seed and CPU hist / one-thread execution are explicit",
                "Target period ratings update only after each strictly earlier observed match",
                "Internal zero prior retained, entirely unobserved output is None plus count=0",
                "No promotion/relegation future backfill; no shared league identity",
            ],
            "omitted_source_inputs": [
                "odds rating: historical quote availability time not verified",
                "marketvalue rating: target starting-lineup market values unavailable",
            ],
            "missing_policy": "Carry forward after observation; expose update counts and age; unknown=None",
            "limitations": [
                "Calendar dates are not proof of exact publication timestamps",
                "Current shot-adjustment version is frozen per season, unlike random-fold source",
                "No learned cross-league or promoted-team prior; no reset at season boundary",
            ],
        },
    }
