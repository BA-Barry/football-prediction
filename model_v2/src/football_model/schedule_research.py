"""Research-only chronological trials of recorded fixture-date count features.

All coefficients are jointly refitted at the original 28-day August anchor.
Source dates update between fits, but same-date results and future labels never
enter training. Archives replay readable parameters without fitting. This is a
date-level retrospective information convention, not verified fixture/news
publication availability or a complete player workload reconstruction.
"""

from datetime import timedelta
import json
import math
from pathlib import Path
import warnings

from sklearn.exceptions import ConvergenceWarning

from . import native_experiment
from .forecast_diagnostics import validate_prediction
from .probability import score_grid, summarize
from .real_backtest import de_vig, evaluate
from .refresh_research import (_blocks, _invalid_constant, _number, _same,
                               _settings, _source_records, _unique_object)
from .schedule_model import (DateActivityIndex, DateScheduleGoalModel, MODEL_TYPE,
                             OPPONENT_COUNT_FEATURE, SELF_COUNT_FEATURE,
                             SCHEDULE_FEATURE_DEFINITION, source_date_features)


INTERVAL_DAYS = 28
CONFIG = {"id": "schedule_count14_ridge_poisson_h180", "family": "schedule_count14",
          "half_life_days": 180, "regularization": 0.01, "window_days": 14}


def _exact(saved, expected, label):
    """Compare source identity/count metadata, retaining integer/bool types."""
    if type(saved) is not type(expected):
        raise ValueError(f"Saved metadata type differs from source: {label}")
    if isinstance(expected, dict):
        if set(saved) != set(expected):
            raise ValueError(f"Saved metadata keys differ from source: {label}")
        for key, value in expected.items():
            _exact(saved[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(saved) != len(expected):
            raise ValueError(f"Saved metadata sequence differs from source: {label}")
        for index, (value, other) in enumerate(zip(saved, expected)):
            _exact(value, other, f"{label}[{index}]")
    elif saved != expected:
        raise ValueError(f"Saved metadata differs from source: {label}")


def _fit(parameters, cutoff, history, competition, index):
    if not isinstance(parameters, dict):
        raise ValueError("Fit parameters must be readable mappings")
    fixed = {
        "model_type": MODEL_TYPE, "before_date": cutoff,
        "training_min_date": history[0].match_date.isoformat(),
        "training_max_date": history[-1].match_date.isoformat(),
        "teams": sorted({team for record in history for team in (record.home, record.away)}),
        "competitions": [competition], "n_matches": len(history),
        "reference_match_ids": [record.match_id for record in history],
        "basis": "team_marginal",
        "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
        "not_for": "precise T-24h, lineup-confirmed, cross-league, or current-2026 forecasts",
        "schedule_feature_definition": SCHEDULE_FEATURE_DEFINITION,
    }
    expected_keys = set(fixed) | {"regularization", "half_life_days", "rho", "intercept",
                                  "coefficients", "optimizer_iterations"}
    if set(parameters) != expected_keys:
        raise ValueError("Saved parameters do not contain exactly the research model fields")
    for key, value in fixed.items():
        _exact(parameters[key], value, key)
    for key, value in (("regularization", .01), ("half_life_days", 180), ("rho", 0)):
        if _number(parameters[key], key) != value:
            raise ValueError(f"Research-model setting changed: {key}")
    iterations = parameters["optimizer_iterations"]
    if type(iterations) is not int or not 0 <= iterations < 1000:
        raise ValueError("Invalid or iteration-exhausted optimizer result")
    _number(parameters["intercept"], "intercept")
    names = {f"{kind}:{team}" for team in fixed["teams"] for kind in ("attack", "defense")}
    names |= {f"competition:{competition}", f"home:{competition}",
              SELF_COUNT_FEATURE, OPPONENT_COUNT_FEATURE}
    coefficients = parameters["coefficients"]
    if not isinstance(coefficients, dict) or set(coefficients) != names:
        raise ValueError("Saved coefficients do not cover exactly the trained source-date features")
    for key, value in coefficients.items():
        _number(value, key)
    # Reconstruct every historical feature row from source dates. This checks
    # chronology/feature availability, not whether saved coefficients are optimal.
    for record in history:
        context = index.context(record.home, record.away, competition, record.match_date)
        if any(set(row) - names for row in source_date_features(
                record.home, record.away, competition, context)):
            raise ValueError("Historical source-date features absent from saved parameters")


def replay_prediction(parameters, record, index):
    """Rebuild rates, complete score summary and date metadata; perform no fit."""
    context = index.context(record.home, record.away, record.competition, record.match_date)
    features = source_date_features(record.home, record.away, record.competition, context)
    rates = [math.exp(parameters["intercept"] + sum(
        parameters["coefficients"][key] * value for key, value in row.items()))
        for row in features]
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Replayed goal rates must be finite and positive")
    return rates, summarize(score_grid(*rates, rho=0)), context


def _validate(trial, records, start, end, competition):
    if not isinstance(trial, dict):
        raise ValueError("A complete saved run must be a mapping")
    _exact(trial["config"], CONFIG, "config")
    if (trial["start"] != start.isoformat() or trial["end_exclusive"] != end.isoformat()
            or type(trial["update_interval_days"]) is not int
            or trial["update_interval_days"] != INTERVAL_DAYS):
        raise ValueError("Saved run settings differ from the fixed 28-day research arm")
    if trial["failures"] != []:
        raise ValueError("Incomplete run: recorded fit/prediction failures")
    if not all(isinstance(trial[key], list) for key in ("predictions", "fits", "skipped")):
        raise ValueError("Complete run must include predictions, fits and skipped lists")
    targets, expected_skips, histories = _blocks(records, start, end, INTERVAL_DAYS)
    expected = {record.match_id for rows in targets.values() for record in rows}
    if expected != native_experiment.expected_target_ids(records, start, end, INTERVAL_DAYS):
        raise ValueError("Internal expected coverage disagreement")
    index = DateActivityIndex(records)
    fits = {}
    for parameters in trial["fits"]:
        cutoff = parameters["before_date"]
        if cutoff in fits or cutoff not in targets:
            raise ValueError("Duplicate, empty-block or non-anchored fit cutoff")
        _fit(parameters, cutoff, histories[cutoff], competition, index)
        fits[cutoff] = parameters
    if set(fits) != set(targets):
        raise ValueError("Missing fit for an eligible target block")
    skips = {}
    for row in trial["skipped"]:
        key = row["match_id"]
        if key in skips or set(row) != {"match_id", "reason"}:
            raise ValueError("Duplicate or malformed skipped-match record")
        skips[key] = row["reason"]
    if skips != expected_skips:
        raise ValueError("Skipped coverage/reasons disagree with the original exclusion/unknown policy")
    source = {record.match_id: record for record in records}
    seen = set()
    for row in trial["predictions"]:
        validate_prediction(row)
        key = row["match_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected saved prediction")
        seen.add(key)
        record = source[key]
        cutoff = start + timedelta(days=((record.match_date - start).days // INTERVAL_DAYS)
                                   * INTERVAL_DAYS)
        cutoff_text = cutoff.isoformat()
        if row["fit_before_date"] != cutoff_text:
            raise ValueError("Prediction cutoff does not follow the fixed August 1 anchor")
        fit = fits[cutoff_text]
        identity = {"competition": competition, "date": record.match_date.isoformat(),
                    "home": record.home_label, "away": record.away_label,
                    "actual_score": list(record.full_time), "source_url": record.source_url,
                    "training_matches": fit["n_matches"],
                    "training_max_date": fit["training_max_date"],
                    "context_before_date": record.match_date.isoformat()}
        for field, value in identity.items():
            _exact(row[field], value, field)
        _same(row["market_prices"], record.markets, "market_prices")
        _same(row["market_probabilities"], {key: list(map(float, de_vig(prices)))
              for key, prices in record.markets.items()}, "market_probabilities")
        rates, forecast, context = replay_prediction(fit, record, index)
        _exact(row["schedule_context"], context, "schedule_context")
        _same(row["rates"], rates, "rates")
        _same(row["forecast"], forecast, "forecast")
        _same(row["probabilities"], [forecast["one_x_two"][side]
                                    for side in ("home", "draw", "away")], "probabilities")
    if seen != expected:
        raise ValueError("Run did not complete the full expected prediction coverage")
    return trial


def validate_trial(trial, records, start, end, competition=None):
    """Replay all source-bound results without refitting or changing the cohort.

    This checks the supplied archive's complete parameter/feature metadata and
    its consequences, not an independent proof that its optimizer found a fit.
    """
    _settings(start, end, INTERVAL_DAYS)
    try:
        records, competition = _source_records(records, competition)
        return _validate(trial, records, start, end, competition)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Malformed source-date research data: {error}") from error


def load_trial(path, records, start, end, competition=None):
    """Read a complete JSON research trial; load no pickle and perform no fit."""
    trial = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return validate_trial(trial, records, start, end, competition)


def run_trial(records, start, end, competition=None):
    """Fit each nonempty fixed-anchor block; fail explicitly without fallback."""
    _settings(start, end, INTERVAL_DAYS)
    try:
        records, competition = _source_records(records, competition)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed source-date research records: {error}") from error
    targets, skipped, _ = _blocks(records, start, end, INTERVAL_DAYS)
    predictions, fits = [], []
    for cutoff_text, matches in targets.items():
        cutoff = start.fromisoformat(cutoff_text)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                # Keep the full identity/date index, including special dates and
                # within-block earlier fixtures. The model filters labels <cutoff.
                model = DateScheduleGoalModel(.01, 180).fit(records, cutoff)
        except ConvergenceWarning as error:
            raise ValueError(f"Source-date Poisson optimizer did not converge at {cutoff_text}; "
                             "no fallback run") from error
        except (ValueError, FloatingPointError) as error:
            raise ValueError(f"Source-date model fit failed at {cutoff_text}; no fallback: {error}") from error
        parameters = model.parameters()
        fits.append(parameters)
        for record in matches:
            try:
                rates = list(map(float, model.predict(record.home, record.away,
                                                      competition, record.match_date)))
                forecast = summarize(score_grid(*rates, rho=0))
                context = model.activity_index.context(record.home, record.away,
                                                       competition, record.match_date)
            except (ValueError, FloatingPointError, OverflowError) as error:
                raise ValueError(f"Source-date prediction failed for {record.match_id}; "
                                 f"no fallback: {error}") from error
            home_goals, away_goals = record.full_time
            predictions.append({
                "match_id": record.match_id, "competition": competition,
                "date": record.match_date.isoformat(), "home": record.home_label,
                "away": record.away_label, "fit_before_date": cutoff_text,
                "training_max_date": parameters["training_max_date"],
                "training_matches": parameters["n_matches"],
                "actual_score": list(record.full_time),
                "actual_outcome": 0 if home_goals > away_goals else 2 if home_goals < away_goals else 1,
                "probabilities": [forecast["one_x_two"][side] for side in ("home", "draw", "away")],
                "forecast": forecast, "rates": rates, "schedule_context": context,
                "context_before_date": record.match_date.isoformat(),
                "market_probabilities": {key: list(map(float, de_vig(prices)))
                                         for key, prices in record.markets.items()},
                "market_prices": record.markets, "source_url": record.source_url,
            })
    trial = {"config": dict(CONFIG), "start": start.isoformat(), "end_exclusive": end.isoformat(),
             "update_interval_days": INTERVAL_DAYS, "predictions": predictions, "fits": fits,
             "skipped": [{"match_id": key, "reason": reason} for key, reason in skipped.items()],
             "failures": [], "metrics": evaluate(predictions)}
    return validate_trial(trial, records, start, end, competition)
