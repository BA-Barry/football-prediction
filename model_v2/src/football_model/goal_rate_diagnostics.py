"""Conditional goal-rate diagnostics of saved original-team predictions.

No model is fitted or corrected here. Rates are reconstructed from readable
coefficients, never from a truncated score-grid expectation. Pearson moments
refer to conditional Poisson rates; unconditional goal variance is not their
dispersion estimate. The caller validates complete archived trials against
source history before using these per-row helpers.
"""

from datetime import date
import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson, skellam

from .team_model import team_features


RATE_EDGES = (0., .75, 1., 1.25, 1.5, 2., 3.)
STAT_FIELDS = (
    "home_rate", "away_rate", "home_goals", "away_goals",
    "home_residual", "away_residual", "home_pearson_square_minus_one",
    "away_pearson_square_minus_one", "pearson_cross_product",
    "home_zero_probability", "away_zero_probability", "total_four_plus_probability",
    "draw_probability", "zero_zero_probability", "one_one_probability",
    "home_zero_residual", "away_zero_residual", "total_four_plus_residual",
    "draw_residual", "zero_zero_residual", "one_one_residual",
    "home_poisson_nll", "away_poisson_nll", "joint_score_nll",
)


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"Need a finite real number: {label}")
    try:
        numeric = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Need a finite real number: {label}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"Need a finite real number: {label}")
    return numeric


def _day(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Use an ISO calendar date: {label}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Use an ISO calendar date: {label}") from error
    if parsed.isoformat() != value:
        raise ValueError(f"Use an ISO calendar date: {label}")
    return parsed


def _score(row):
    try:
        score = row["actual_score"]
        if (not isinstance(score, (tuple, list, np.ndarray)) or len(score) != 2
                or any(isinstance(g, (bool, np.bool_)) or not isinstance(g, Integral)
                       or g < 0 for g in score)):
            raise ValueError("Actual score must contain two nonnegative integers")
    except (KeyError, TypeError) as error:
        raise ValueError("Actual score must contain two nonnegative integers") from error
    # Very large Python integers must not become Infinity in the diagnostics.
    for goal in score:
        _number(goal, "goal")
    return tuple(int(g) for g in score)


def _rates(rates):
    try:
        if len(rates) != 2:
            raise ValueError("Need two finite positive conditional goal rates")
        values = tuple(_number(r, "goal rate") for r in rates)
    except TypeError as error:
        raise ValueError("Need two finite positive conditional goal rates") from error
    if any(r <= 0 for r in values):
        raise ValueError("Need two finite positive conditional goal rates")
    return values


def rates_from_parameters(model, row):
    """Recover untruncated lambda_h/lambda_a from one already validated fit.

    Saved prediction rows have human-readable team labels, whereas the original
    Football-Data adapter's model keys include the provider and competition.
    Missing required coefficients and unknown teams are errors, not zero priors.
    Whole-history membership/exclusion checks belong to ``validate_trial``.
    """
    try:
        if not isinstance(model, Mapping) or not isinstance(row, Mapping):
            raise ValueError("Saved parameters and prediction must be mappings")
        if (model["model_type"] != "date_level_regularized_poisson"
                or model["basis"] != "team_marginal"
                or model["availability_convention"] !=
                "strictly_earlier_calendar_date_results_assumed_available"):
            raise ValueError("Need original date-level team-marginal parameters")
        for key, value in (("regularization", .01), ("half_life_days", 180), ("rho", 0)):
            if _number(model[key], key) != value:
                raise ValueError(f"Original-model parameter differs: {key}")
        played = _day(row["date"], "prediction date")
        cutoff = _day(model["before_date"], "fit cutoff")
        first = _day(model["training_min_date"], "first training date")
        last = _day(model["training_max_date"], "last training date")
        if not first <= last < cutoff <= played:
            raise ValueError("Training must precede fit cutoff and prediction date")
        if (row["fit_before_date"] != cutoff.isoformat()
                or row["training_max_date"] != last.isoformat()):
            raise ValueError("Prediction training dates disagree with its fit")
        if (type(model["n_matches"]) is not int or model["n_matches"] < 1
                or type(row["training_matches"]) is not int
                or row["training_matches"] != model["n_matches"]):
            raise ValueError("Prediction training count disagrees with its fit")
        home, away, competition = row["home"], row["away"], row["competition"]
        if (any(not isinstance(value, str) or not value for value in (home, away, competition))
                or home == away):
            raise ValueError("Need distinct team labels and a competition")
        if competition not in model["competitions"]:
            raise ValueError("Unknown competition in saved fit")
        teams = tuple(f"football-data:{competition}:{label}" for label in (home, away))
        if any(team not in model["teams"] for team in teams):
            raise ValueError("Unknown team: no estimated prior")
        coefficients = model["coefficients"]
        if not isinstance(coefficients, Mapping):
            raise ValueError("Saved coefficients must be a mapping")
        intercept = _number(model["intercept"], "intercept")
        rates = []
        for features in team_features(*teams, competition):
            linear = intercept + math.fsum(
                _number(coefficients[key], key) * value for key, value in features.items())
            rates.append(math.exp(linear))
        return _rates(rates)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Cannot reconstruct saved original-team rates: {error}") from error


def _probability(value, label):
    value = _number(value, label)
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid computed probability: {label}")
    return value


def row_statistics(row, rates):
    """Return finite scalars from exact untruncated independent-Poisson formulas.

    Event residuals are observed indicators minus conditional probabilities.
    NLL uses lambda - y log(lambda) + gammaln(y+1) without probability clipping.
    A nonfinite Skellam/draw evaluation is an explicit error, not a fallback.
    """
    h, a = _score(row)
    lh, la = _rates(rates)
    total = _number(lh + la, "total goal rate")
    rh, ra = h - lh, a - la
    ph, pa = rh / math.sqrt(lh), ra / math.sqrt(la)
    home_zero = math.exp(-lh)
    away_zero = math.exp(-la)
    four_plus = _probability(poisson.sf(3, total), "total four plus")
    draw = _probability(skellam.pmf(0, lh, la), "draw")
    zero_zero = math.exp(-total)
    # Log form avoids overflowing lh * la before multiplication by exp(-sum).
    one_one = math.exp(math.log(lh) + math.log(la) - total)
    hnll = lh - h * math.log(lh) + float(gammaln(float(h) + 1))
    anll = la - a * math.log(la) + float(gammaln(float(a) + 1))
    output = {
        "home_rate": lh, "away_rate": la, "home_goals": h, "away_goals": a,
        "home_residual": rh, "away_residual": ra,
        "home_pearson_square_minus_one": ph * ph - 1,
        "away_pearson_square_minus_one": pa * pa - 1,
        "pearson_cross_product": ph * pa,
        "home_zero_probability": home_zero, "away_zero_probability": away_zero,
        "total_four_plus_probability": four_plus, "draw_probability": draw,
        "zero_zero_probability": zero_zero, "one_one_probability": one_one,
        "home_zero_residual": int(h == 0) - home_zero,
        "away_zero_residual": int(a == 0) - away_zero,
        "total_four_plus_residual": int(h + a >= 4) - four_plus,
        "draw_residual": int(h == a) - draw,
        "zero_zero_residual": int(h == a == 0) - zero_zero,
        "one_one_residual": int(h == a == 1) - one_one,
        "home_poisson_nll": hnll, "away_poisson_nll": anll,
        "joint_score_nll": hnll + anll,
    }
    for key, value in output.items():
        _number(value, key)
    return output


def rate_bin_labels(rates):
    """Left-closed/right-open rate bins; the final bin is [3, infinity).

    Zero is an edge, not an allowed Poisson rate. Scalars return a string;
    nonempty arrays/sequences retain their shape in a string ndarray.
    """
    try:
        raw = np.asarray(rates)
    except (ValueError, TypeError) as error:
        raise ValueError("Rate bins require finite positive real rates") from error
    if raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("Rate bins require finite positive real rates")
    if any(isinstance(value, (bool, np.bool_)) for value in
           np.asarray(rates, dtype=object).flat):
        raise ValueError("Rate bins require finite positive real rates")
    values = raw.astype(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Rate bins require finite positive real rates")
    names = np.asarray([f"{lower:g}-{upper:g}" for lower, upper in
                        zip(RATE_EDGES, RATE_EDGES[1:])] + ["3+"])
    labels = names[np.searchsorted(RATE_EDGES, values, side="right") - 1]
    return str(labels) if raw.ndim == 0 else labels


def summary(rows, stats):
    """Summarize a unique aligned cohort; never deduplicate or pool mean rates.

    Sample SD is undefined for n=1 and therefore recorded as None. Expected goal
    counts sum each individual conditional Poisson pmf/sf, not a pmf at mean rate.
    Scalar point summaries alone do not establish a Poisson dispersion change.
    """
    rows, stats = list(rows), list(stats)
    if not rows or len(rows) != len(stats):
        raise ValueError("Need nonempty aligned rows and statistics")
    seen = set()
    for row, statistic in zip(rows, stats):
        if not isinstance(row, Mapping) or not isinstance(statistic, Mapping):
            raise ValueError("Rows and statistics must be mappings")
        key = row.get("match_id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate match identifier")
        seen.add(key)
        if set(statistic) != set(STAT_FIELDS):
            raise ValueError("Statistics do not contain the diagnostic scalar fields")
        replay = row_statistics(row, (statistic["home_rate"], statistic["away_rate"]))
        for field, value in replay.items():
            if not math.isclose(_number(statistic[field], field), value,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Statistics disagree with row/rates: {field}")
    n = len(rows)
    matrix = np.asarray([[row[key] for key in STAT_FIELDS] for row in stats], dtype=float)
    descriptions = {}
    for index, key in enumerate(STAT_FIELDS):
        column = matrix[:, index]
        descriptions[key] = {
            "mean": float(column.mean()), "sd": float(column.std(ddof=1)) if n > 1 else None,
            "min": float(column.min()), "median": float(np.median(column)), "max": float(column.max()),
        }
    result = {"n": n, "scalar_descriptives": descriptions}
    for side in ("home", "away"):
        goals = np.asarray([row[f"{side}_goals"] for row in stats])
        rates = np.asarray([row[f"{side}_rate"] for row in stats])
        expected = float(rates.sum())
        observed = int(goals.sum())
        probability_counts = poisson.pmf(np.arange(4)[None, :], rates[:, None]).sum(axis=0)
        expected_counts = [*map(float, probability_counts), float(poisson.sf(3, rates).sum())]
        if any(not math.isfinite(value) or value < 0 for value in expected_counts):
            raise ValueError("Nonfinite conditional expected goal counts")
        result[side] = {
            "observed_sum": observed, "expected_sum": expected,
            "residual_per_match": (observed - expected) / n,
            "observed_to_expected_ratio": observed / expected,
            "goal_counts": {
                label: {"observed": int((goals == index).sum()) if index < 4 else int((goals >= 4).sum()),
                        "expected": expected_counts[index]}
                for index, label in enumerate(("0", "1", "2", "3", "4+"))
            },
        }
    return result
