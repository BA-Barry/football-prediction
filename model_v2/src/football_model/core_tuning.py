"""Small chronological search of the existing regularized team Poisson model.

Only its time half-life and L2 strength vary. This does not change team features,
the independent score distribution, unknown-team treatment or market blending.
One shared hyperparameter policy is selected across five separately fitted
leagues; later-season results are never arguments to selection.
"""

from datetime import date, timedelta
import json
import math
import warnings

from sklearn.exceptions import ConvergenceWarning

from .date_model import DateTeamGoalModel
from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import align_tracks
from .probability import score_grid, summarize
from .real_backtest import de_vig, evaluate


VALID_START = date(2024, 8, 1)
VALID_END = date(2025, 7, 1)
UPDATE_INTERVAL_DAYS = 28


def candidate_configs():
    """Return fresh copies of the nine explicitly specified parameter settings."""
    return [
        {"id": f"regularized_poisson_h{half}_a{format(alpha, 'g').replace('.', 'p')}",
         "family": "original_regularized", "half_life_days": half,
         "regularization": alpha}
        for half in (90, 180, 365) for alpha in (.001, .01, .1)
    ]


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("Candidate configuration must be a mapping")
    if (type(config.get("half_life_days")) not in (int, float)
            or type(config.get("regularization")) not in (int, float)):
        raise ValueError("Candidate half-life and regularization must be numeric, not booleans")
    canonical = {c["id"]: c for c in candidate_configs()}
    if config.get("id") not in canonical or config != canonical[config["id"]]:
        raise ValueError("Use one of the nine declared candidate configurations")
    return dict(canonical[config["id"]])


def _records(records):
    records = list(records)
    if not records or len({r.competition for r in records}) != 1:
        raise ValueError("Run each nonempty league separately")
    if records[0].competition not in LEAGUES:
        raise ValueError("Unknown domestic league")
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate match identifiers must be resolved before fitting")
    for row in records:
        if type(row.match_date) is not date:
            raise ValueError("Record dates must be calendar dates")
        if any(not isinstance(s, str) or not s for s in
               (row.match_id, row.home, row.away, row.home_label, row.away_label, row.source_url)):
            raise ValueError("Need nonempty match, team and source identities")
        if row.home == row.away:
            raise ValueError("Teams must differ")
        if (not isinstance(row.full_time, (tuple, list)) or len(row.full_time) != 2
                or any(type(g) is not int or g < 0 for g in row.full_time)):
            raise ValueError("Goals must be non-negative integers")
        if not isinstance(row.markets, dict) or not set(row.markets) <= set(MARKET_COLUMNS):
            raise ValueError("Unknown market quotation keys")
        for prices in row.markets.values():
            de_vig(prices)
    return sorted(records, key=lambda r: (r.match_date, r.match_id))


def run_candidate(records, config, start, end):
    """Fit one league every 28 days from the fixed August 1 calendar anchor.

    A convergence warning is a recorded failed fit, not a successful estimate or
    a reason to switch solver/model. Other numerical fit/prediction failures are
    also retained. No-result blocks advance the same calendar anchor.
    """
    config = _configuration(config)
    if (type(start) is not date or type(end) is not date or start >= end
            or (start.month, start.day) != (8, 1)
            or end > date(start.year + 1, 8, 1)):
        raise ValueError("Use ordered calendar dates within a season starting August 1")
    records = _records(records)
    predictions, fits, failures, skipped = [], [], [], []
    cutoff = start
    while cutoff < end:
        stop = min(end, cutoff + timedelta(days=UPDATE_INTERVAL_DAYS))
        targets = [r for r in records if cutoff <= r.match_date < stop]
        usable = []
        for match in targets:
            if match.goal_fit_exclusion:
                skipped.append({"match_id": match.match_id, "reason": match.goal_fit_exclusion})
            else:
                usable.append(match)
        earlier = [r for r in records if r.match_date < cutoff and not r.goal_fit_exclusion]
        known = {team for r in earlier for team in (r.home, r.away)}
        targets = []
        for match in usable:
            if not {match.home, match.away} <= known:
                skipped.append({"match_id": match.match_id, "reason": "unseen_team_no_estimated_prior"})
            else:
                targets.append(match)
        if targets:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model = DateTeamGoalModel(config["regularization"], config["half_life_days"]).fit(
                        earlier, cutoff)
                parameters = model.parameters()
            except (ConvergenceWarning, ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                failures.append({"cutoff": cutoff.isoformat(), "stage": "fit",
                    "matches": [r.match_id for r in targets], "error_type": type(error).__name__,
                    "error": str(error)})
                cutoff = stop
                continue
            fits.append(parameters)
            for match in targets:
                try:
                    rates = model.predict(match.home, match.away, match.competition, match.match_date)
                    forecast = summarize(score_grid(*rates, rho=0))
                except (ValueError, FloatingPointError, OverflowError) as error:
                    failures.append({"cutoff": cutoff.isoformat(), "stage": "predict",
                        "match_id": match.match_id, "error_type": type(error).__name__, "error": str(error)})
                    continue
                home, away = match.full_time
                predictions.append({
                    "match_id": match.match_id, "competition": match.competition,
                    "date": match.match_date.isoformat(), "home": match.home_label,
                    "away": match.away_label, "fit_before_date": cutoff.isoformat(),
                    "training_max_date": model.training_max_date.isoformat(),
                    "training_matches": model.n_matches,
                    "actual_score": [home, away],
                    "actual_outcome": 0 if home > away else 2 if home < away else 1,
                    "rates": list(map(float, rates)),
                    "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
                    "market_probabilities": {k: list(map(float, de_vig(v))) for k, v in match.markets.items()},
                    "market_prices": {k: tuple(v) for k, v in match.markets.items()},
                    "source_url": match.source_url,
                })
        cutoff = stop
    return {"config": config, "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "update_interval_days": UPDATE_INTERVAL_DAYS, "predictions": predictions,
            "fits": fits, "failures": failures, "skipped": skipped, "metrics": evaluate(predictions)}


def select_policy(validation_runs, expected_ids):
    """Select one policy from all nine complete 2024/25 global forecast tracks.

    Callers aggregate the five single-league runs for each configuration first.
    Scores are recomputed from matching rows rather than trusting a saved rank.
    All settings must complete the prescribed identical cohort without failures;
    incomplete or failed results remain reports, not eligible replacements.
    """
    runs = list(validation_runs)
    expected = set(expected_ids)
    if not expected or any(not isinstance(key, str) or not key for key in expected):
        raise ValueError("Need nonempty expected validation match identifiers")
    if len(runs) != len(candidate_configs()):
        raise ValueError("Selection requires all nine candidate runs")
    tracks, by_id = {}, {}
    for run in runs:
        config = _configuration(run["config"])
        identifier = config["id"]
        if identifier in by_id:
            raise ValueError("Duplicate candidate configuration")
        if (run["start"] != VALID_START.isoformat()
                or run["end_exclusive"] != VALID_END.isoformat()
                or run.get("update_interval_days") != UPDATE_INTERVAL_DAYS):
            raise ValueError("Selection accepts only the designated 2024/25 validation period")
        if run["failures"]:
            raise ValueError("All nine candidates must complete validation without failures")
        rows = run["predictions"]
        if (len(rows) != len(expected) or {p["match_id"] for p in rows} != expected):
            raise ValueError("All nine candidates must complete the identical expected match set")
        if {p["competition"] for p in rows} != set(LEAGUES):
            raise ValueError("Aggregate all five domestic leagues before selecting one global policy")
        for row in rows:
            played = date.fromisoformat(row["date"])
            cutoff = date.fromisoformat(row["fit_before_date"])
            if not VALID_START <= played < VALID_END:
                raise ValueError("Later-season or out-of-period predictions cannot select parameters")
            expected_cutoff = VALID_START + timedelta(days=((played - VALID_START).days // 28) * 28)
            if cutoff != expected_cutoff:
                raise ValueError("Forecast cutoff must follow the fixed August 1 28-day anchor")
        tracks[identifier] = rows
        by_id[identifier] = run
    # Fresh records contain tuple quotations; archived JSON contains lists.
    # Compare the same JSON-native representation, without changing any values.
    aligned, coverage = align_tracks(json.loads(json.dumps(tracks, allow_nan=False)))
    if set(coverage["common_match_ids"]) != expected:
        raise ValueError("Validation tracks do not share the expected cohort")
    scores = {}
    for identifier, rows in aligned.items():
        entries = list(rows.values())
        values = probability_scores([r["probabilities"] for r in entries],
                                   [r["actual_outcome"] for r in entries])
        # No clipping of a zero probability for the actually observed outcome.
        observed = [r["probabilities"][r["actual_outcome"]] for r in entries]
        values["log_loss"] = (math.inf if any(p == 0 for p in observed)
                              else -math.fsum(math.log(p) for p in observed) / len(observed))
        scores[identifier] = values
    chosen = min(scores, key=lambda identifier: (scores[identifier]["log_loss"], identifier))
    if not math.isfinite(scores[chosen]["log_loss"]):
        raise ValueError("No candidate has finite observed-outcome validation log loss")
    return {"config": dict(by_id[chosen]["config"]), "validation_scores": scores[chosen],
            "selection_metric": "2024/25 pooled identical-match chronological validation 1X2 log loss",
            "eligible_candidates": len(runs), "total_candidates": len(runs),
            "expected_matches": len(expected), "candidate_scores": scores,
            "policy": "shared_half_life_and_regularization_separate_league_coefficients"}
