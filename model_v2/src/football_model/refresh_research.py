"""Fixed-August-anchor refresh-frequency research for the original team model.

This module neither selects nor promotes a refresh interval. Saved 28-day runs
are read as JSON, checked against the supplied source records and replayed from
their readable parameters; no pickle is loaded and no archived fit is rerun.
New 7/14-day runs retain the same model, regularization, decay and unknown-team
policy. All results remain date-level retrospective experiments.
"""

from datetime import date, timedelta
import json
import math
from numbers import Real
from pathlib import Path
import warnings

from sklearn.exceptions import ConvergenceWarning

from . import native_experiment
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import validate_prediction
from .probability import score_grid, summarize
from .real_backtest import de_vig
from .team_model import team_features


INTERVALS = (7, 14, 28)
PERIODS = {
    "2024-25": (native_experiment.VALID_START, native_experiment.VALID_END),
    "2025-26": (native_experiment.TEST_START, native_experiment.TEST_END),
}


def _settings(start, end, interval):
    if (type(start) is not date or type(end) is not date or start >= end
            or (start.month, start.day) != (8, 1)):
        raise ValueError("Use ordered calendar dates with a fixed August 1 anchor")
    if type(interval) is not int or interval not in INTERVALS:
        raise ValueError("Research refresh interval must be 7, 14 or 28 days")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Need a finite real number: {label}")
    return float(value)


def _same(saved, expected, label):
    """Recursive ordinary data comparison, not a new serialization contract."""
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or set(saved) != set(expected):
            raise ValueError(f"Saved keys disagree with replay/source: {label}")
        for key, value in expected.items():
            _same(saved[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(saved, (list, tuple)) or len(saved) != len(expected):
            raise ValueError(f"Saved sequence disagrees with replay/source: {label}")
        for index, (value, other) in enumerate(zip(saved, expected)):
            _same(value, other, f"{label}[{index}]")
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        actual = _number(saved, label)
        if not math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Saved number disagrees with replay/source: {label}")
    elif saved != expected:
        raise ValueError(f"Saved value disagrees with replay/source: {label}")


def _source_records(records, competition):
    records = list(records)
    if not records:
        raise ValueError("Need source records from one league")
    competitions = {r.competition for r in records}
    if competition is None and len(competitions) == 1:
        competition = next(iter(competitions))
    if competition not in LEAGUES or competitions != {competition}:
        raise ValueError("Use exactly one declared league, including training records")
    seen = set()
    for row in records:
        if not isinstance(row.match_id, str) or not row.match_id or row.match_id in seen:
            raise ValueError("Missing or duplicate source match identifier")
        seen.add(row.match_id)
        if type(row.match_date) is not date:
            raise ValueError("Source records must use calendar dates")
        if (not isinstance(row.home, str) or not row.home or not isinstance(row.away, str)
                or not row.away or row.home == row.away):
            raise ValueError("Source teams must be distinct and nonempty")
        if (len(row.full_time) != 2
                or any(type(goal) is not int or goal < 0 for goal in row.full_time)):
            raise ValueError("Source scores must contain two nonnegative integers")
        if (not isinstance(row.source_url, str) or not row.source_url
                or not isinstance(row.home_label, str) or not row.home_label
                or not isinstance(row.away_label, str) or not row.away_label
                or row.home_label == row.away_label):
            raise ValueError("Missing source identity or URL")
        if row.goal_fit_exclusion is not None and (
                not isinstance(row.goal_fit_exclusion, str) or not row.goal_fit_exclusion):
            raise ValueError("Source exclusion reason must be explicit or None")
        if not isinstance(row.markets, dict) or not set(row.markets) <= set(MARKET_COLUMNS):
            raise ValueError("Unknown source markets")
        for key, prices in row.markets.items():
            if (not isinstance(prices, (tuple, list)) or len(prices) != 3
                    or any(_number(p, key) <= 1 for p in prices)):
                raise ValueError("Source decimal prices must be finite and greater than one")
    return sorted(records, key=lambda r: (r.match_date, r.match_id)), competition


def _blocks(records, start, end, interval):
    """Include empty/skipped-only blocks without shifting the fixed anchor."""
    targets, skipped, histories = {}, {}, {}
    cutoff = start
    while cutoff < end:
        stop = min(end, cutoff + timedelta(days=interval))
        history = [r for r in records if r.match_date < cutoff and r.goal_fit_exclusion is None]
        known = {team for r in history for team in (r.home, r.away)}
        usable = []
        for row in records:
            if not cutoff <= row.match_date < stop:
                continue
            if row.goal_fit_exclusion:
                skipped[row.match_id] = row.goal_fit_exclusion
            elif not {row.home, row.away} <= known:
                skipped[row.match_id] = "unseen_team_no_estimated_prior"
            else:
                usable.append(row)
        if usable:
            targets[cutoff.isoformat()] = usable
            histories[cutoff.isoformat()] = history
        cutoff = stop
    return targets, skipped, histories


def _fit(fit, cutoff, history, competition):
    if not isinstance(fit, dict):
        raise ValueError("Fit parameters must be readable mappings")
    fixed = {
        "model_type": "date_level_regularized_poisson", "before_date": cutoff,
        "training_min_date": history[0].match_date.isoformat(),
        "training_max_date": history[-1].match_date.isoformat(),
        "teams": sorted({t for row in history for t in (row.home, row.away)}),
        "competitions": [competition],
        "reference_match_ids": [row.match_id for row in history],
        "basis": "team_marginal",
        "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
    }
    for key, value in fixed.items():
        if fit[key] != value:
            raise ValueError(f"Saved fit metadata disagrees with earlier history: {key}")
    if type(fit["n_matches"]) is not int or fit["n_matches"] != len(history):
        raise ValueError("Saved training count disagrees with earlier history")
    for key, value in (("regularization", 0.01), ("half_life_days", 180), ("rho", 0)):
        if _number(fit[key], key) != value:
            raise ValueError(f"Original-model parameter changed: {key}")
    iterations = fit["optimizer_iterations"]
    if type(iterations) is not int or not 0 <= iterations < 1000:
        raise ValueError("Invalid or iteration-exhausted optimizer result")
    _number(fit["intercept"], "intercept")
    coefficients = fit["coefficients"]
    names = {f"{kind}:{team}" for team in fixed["teams"] for kind in ("attack", "defense")}
    names |= {f"competition:{competition}", f"home:{competition}"}
    if not isinstance(coefficients, dict) or set(coefficients) != names:
        raise ValueError("Saved coefficients do not cover exactly the trained features")
    for key, value in coefficients.items():
        _number(value, key)


def _forecast(parameters, record):
    rates = [math.exp(parameters["intercept"] + sum(
        parameters["coefficients"][key] * value for key, value in features.items()))
        for features in team_features(record.home, record.away, record.competition)]
    return summarize(score_grid(*rates, rho=0))


def _validate(trial, records, start, end, competition, interval):
    if not isinstance(trial, dict):
        raise ValueError("A complete saved run must be a mapping")
    if (trial["config"] != native_experiment.ORIGINAL or trial["start"] != start.isoformat()
            or trial["end_exclusive"] != end.isoformat()
            or type(trial["update_interval_days"]) is not int
            or trial["update_interval_days"] != interval):
        raise ValueError("Saved run settings differ from the declared original-model research arm")
    if trial["failures"] != []:
        raise ValueError("Incomplete run: recorded fit/prediction failures")
    if not all(isinstance(trial[key], list) for key in ("predictions", "fits", "skipped")):
        raise ValueError("Complete run must include predictions, fits and skipped lists")
    targets, expected_skips, histories = _blocks(records, start, end, interval)
    expected = {row.match_id for rows in targets.values() for row in rows}
    if expected != native_experiment.expected_target_ids(records, start, end, interval):
        raise ValueError("Internal expected coverage disagreement")
    fits = {}
    for fit in trial["fits"]:
        cutoff = fit["before_date"]
        if cutoff in fits or cutoff not in targets:
            raise ValueError("Duplicate, empty-block or non-anchored fit cutoff")
        _fit(fit, cutoff, histories[cutoff], competition)
        fits[cutoff] = fit
    if set(fits) != set(targets):
        raise ValueError("Missing fit for an eligible target block")
    skipped = {}
    for row in trial["skipped"]:
        key = row["match_id"]
        if key in skipped or set(row) != {"match_id", "reason"}:
            raise ValueError("Duplicate or malformed skipped-match record")
        skipped[key] = row["reason"]
    if skipped != expected_skips:
        raise ValueError("Skipped coverage/reasons disagree with unknown-team and exclusion policy")
    source = {row.match_id: row for row in records}
    seen = set()
    for row in trial["predictions"]:
        validate_prediction(row)
        key = row["match_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected saved prediction")
        seen.add(key)
        record = source[key]
        cutoff = start + timedelta(days=((record.match_date - start).days // interval) * interval)
        cutoff_text = cutoff.isoformat()
        if row["fit_before_date"] != cutoff_text:
            raise ValueError("Prediction cutoff does not follow the fixed August 1 anchor")
        fit = fits[cutoff_text]
        identity = {"competition": competition, "date": record.match_date.isoformat(),
                    "home": record.home_label, "away": record.away_label,
                    "actual_score": list(record.full_time), "source_url": record.source_url}
        for field, value in identity.items():
            if row[field] != value:
                raise ValueError(f"Saved prediction identity differs from source: {field}")
        if (type(row["training_matches"]) is not int
                or row["training_matches"] != fit["n_matches"]
                or row["training_max_date"] != fit["training_max_date"]):
            raise ValueError("Prediction training metadata differs from its fit")
        _same(row["market_prices"], record.markets, "market_prices")
        _same(row["market_probabilities"], {key: list(map(float, de_vig(value)))
              for key, value in record.markets.items()}, "market_probabilities")
        replay = _forecast(fit, record)
        _same(row["forecast"], replay, "forecast")
        _same(row["probabilities"], [replay["one_x_two"][key] for key in ("home", "draw", "away")],
              "probabilities")
    if seen != expected:
        raise ValueError("Run did not complete the full expected prediction coverage")
    return trial


def validate_trial(trial, records, start, end, competition=None, interval=28):
    """Validate an entire arm without changing it or reducing its match cohort.

    Saved coefficients and forecasts are replayed, not refitted. Iteration and
    finite-number checks do not constitute an independent optimum proof. The
    supplied records remain the authority for historical identities and scores.
    """
    _settings(start, end, interval)
    try:
        records, competition = _source_records(records, competition)
        return _validate(trial, records, start, end, competition, interval)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Malformed refresh research data: {error}") from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def load_reference(path, records, start, end, competition):
    """Read/replay a complete 28-day JSON archive; perform zero model fits."""
    trial = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return validate_trial(trial, records, start, end, competition, interval=28)


def run_frequency(records, start, end, interval):
    """Run one fixed-frequency arm; never select, repair or promote a model."""
    _settings(start, end, interval)
    try:
        records, competition = _source_records(records, None)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed refresh research source records: {error}") from error
    if interval == 28:
        raise ValueError("Read the saved 28-day reference with load_reference; do not refit it")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            trial = native_experiment.run_config(records, dict(native_experiment.ORIGINAL),
                                                 start, end, interval_days=interval)
    except ConvergenceWarning as error:
        raise ValueError("Original Poisson optimizer did not converge; no fallback run") from error
    return validate_trial(trial, records, start, end, competition, interval)
