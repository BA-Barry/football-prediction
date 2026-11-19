"""Same-target, same-cohort summaries for player regularization experiments.

Clock alternatives are repeated descriptions of each match, not independent
observations. No model fitting, unknown-player imputation or significance test
is performed here. The caller must supply chronologically held-out validation
scores to ``select_regularization``; these containers cannot prove timing.
"""

from datetime import date
from math import fsum, isfinite

from .full_season_experiment import summarize


KINDS = ("team", "fixed100", "selected")
DELTAS = ("selected_minus_fixed100", "fixed100_minus_team", "selected_minus_team")
AVERAGING_NOTE = (
    "Equal clock average of mean match loss on the same intersection of match IDs; "
    "each match contributes once per clock, and clocks are not independent observations."
)


def _index(scores):
    result = {}
    for prediction in scores["predictions"]:
        mid = prediction["match_id"]
        if not isinstance(mid, str) or not mid or mid in result:
            raise ValueError("Predictions require distinct nonempty match IDs")
        day = prediction["match_date"]
        try:
            valid_day = isinstance(day, str) and date.fromisoformat(day).isoformat() == day
        except ValueError:
            valid_day = False
        if not valid_day:
            raise ValueError("Prediction match_date must be an ISO calendar date")
        for field in ("actual_home_target", "actual_away_target", "point_process_loss",
                      "expected_home_target", "expected_away_target"):
            value = prediction[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"Prediction {field} must be finite numeric data")
        if any(prediction[f"{prefix}_{side}_target"] < 0
               for prefix in ("actual", "expected") for side in ("home", "away")):
            raise ValueError("Observed and expected targets must not be negative")
        n_stints = prediction["n_stints"]
        if isinstance(n_stints, bool) or not isinstance(n_stints, int) or n_stints <= 0:
            raise ValueError("Prediction n_stints must be a positive integer")
        if not isinstance(prediction["loss_kind"], str) or not prediction["loss_kind"]:
            raise ValueError("Prediction loss_kind must be a nonempty string")
        result[mid] = prediction
    return result


def _clocks(scores):
    if not isinstance(scores, dict) or not scores:
        raise ValueError("A nonempty mapping of declared clocks is required")
    if any(not isinstance(clock, str) or not clock for clock in scores):
        raise ValueError("Clock labels must be nonempty strings")
    return sorted(scores)


def _ridge(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Player ridge keys must be positive finite numeric values")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError("Player ridge keys must be positive finite numeric values") from None
    if not isfinite(result) or result <= 0:
        raise ValueError("Player ridge keys must be positive finite numeric values")
    return result


def _identities(indices):
    """Check every observed overlap, even when outside the shared cohort."""
    target_references, path_references = {}, {}
    loss_kinds = set()
    for clock, tracks in indices.items():
        path_references[clock] = {}
        for predictions in tracks.values():
            for mid, prediction in predictions.items():
                target = (prediction["match_date"], prediction["actual_home_target"],
                          prediction["actual_away_target"], prediction["loss_kind"])
                previous = target_references.setdefault(mid, target)
                if target != previous:
                    raise ValueError("All clocks and tracks must retain the same match date, target and loss kind")
                previous_path = path_references[clock].setdefault(mid, prediction["n_stints"])
                if prediction["n_stints"] != previous_path:
                    raise ValueError("Within each clock, tracks must score the same actual state path")
                loss_kinds.add(prediction["loss_kind"])
    if len(loss_kinds) > 1:
        raise ValueError("Aggregated losses must use one loss definition")


def _common(indices):
    groups = [set(predictions) for tracks in indices.values() for predictions in tracks.values()]
    return sorted(set.intersection(*groups))


def select_regularization(scores_by_clock_and_ridge):
    """Select a ridge using ``scores[clock][ridge]`` on common validation IDs.

    Ridge keys may be positive finite numbers or numeric strings. Equivalent
    numeric values are normalized, and every clock must retain the same set.
    Candidate metrics use round-trip numeric strings as keys (e.g. ``"100"``).
    A loss tie within 1e-12 selects the larger ridge. No common observations is
    an error: it cannot be converted into a zero-loss winning candidate.
    """
    clocks = _clocks(scores_by_clock_and_ridge)
    indices, expected_candidates = {}, None
    for clock in clocks:
        scores = scores_by_clock_and_ridge[clock]
        if not isinstance(scores, dict) or not scores:
            raise ValueError("Every clock requires a nonempty set of ridge candidates")
        normalized = {}
        for key, values in scores.items():
            ridge = _ridge(key)
            if ridge in normalized:
                raise ValueError("Ridge keys must be distinct after numeric normalization")
            normalized[ridge] = _index(values)
        candidates = set(normalized)
        if expected_candidates is None:
            expected_candidates = candidates
        elif candidates != expected_candidates:
            raise ValueError("Every clock must retain the same numeric ridge candidate set")
        indices[clock] = normalized
    _identities(indices)
    common = _common(indices)
    if not common:
        raise ValueError("Regularization selection needs a nonempty common validation cohort")
    metrics = {}
    for ridge in sorted(expected_candidates):
        per_clock = {clock: summarize([indices[clock][ridge][mid] for mid in common])
                     for clock in clocks}
        metrics[format(ridge, ".17g")] = {
            "player_ridge": ridge, "per_clock": per_clock,
            "mean_clock_match_loss": fsum(row["mean_match_loss"] for row in per_clock.values()) / len(clocks)}
    best_loss = min(row["mean_clock_match_loss"] for row in metrics.values())
    selected = max(row["player_ridge"] for row in metrics.values()
                   if row["mean_clock_match_loss"] <= best_loss + 1e-12)
    return {"selected_player_ridge": selected, "common_match_ids": common,
            "clocks": clocks, "candidate_metrics": metrics,
            "lower_is_better": True, "tie_tolerance": 1e-12,
            "averaging_note": AVERAGING_NOTE,
            "sampling_uncertainty": "not estimated; descriptive validation choice, not a significance claim"}


def _comparison(indices, common):
    summaries = {kind: summarize([indices[kind][mid] for mid in common]) if common else None
                 for kind in KINDS}
    means = {kind: summary["mean_match_loss"] if summary is not None else None
             for kind, summary in summaries.items()}
    rows = []
    for mid in common:
        losses = {kind: indices[kind][mid]["point_process_loss"] for kind in KINDS}
        rows.append({"match_id": mid, "date": indices["team"][mid]["match_date"],
                     **{f"{kind}_loss": value for kind, value in losses.items()},
                     "selected_minus_fixed100": losses["selected"] - losses["fixed100"],
                     "fixed100_minus_team": losses["fixed100"] - losses["team"],
                     "selected_minus_team": losses["selected"] - losses["team"]})
    deltas = {name: fsum(row[name] for row in rows) / len(rows) if rows else None for name in DELTAS}
    return {"common_match_ids": list(common),
            **{f"{kind}_on_common": summary for kind, summary in summaries.items()},
            "mean_match_loss": means, **deltas, "paired_matches": rows}


def compare_regularization(scores):
    """Compare ``scores[clock][team|fixed100|selected]`` on one shared cohort.

    All per-clock summaries use the intersection across every clock and track.
    Partitions may differ across clocks but not between models of one clock.
    An empty common cohort produces None summaries/deltas and empty rows, not
    a fabricated improvement. The fixed100 label refers to the caller's fixed
    ridge-100 comparator; prediction containers do not prove fitted settings.
    """
    clocks = _clocks(scores)
    indices = {}
    for clock in clocks:
        if set(scores[clock]) != set(KINDS):
            raise ValueError("Every clock requires team, fixed100 and selected score tracks")
        indices[clock] = {kind: _index(scores[clock][kind]) for kind in KINDS}
    _identities(indices)
    common = _common(indices)
    per_clock = {clock: _comparison(indices[clock], common) for clock in clocks}
    clock_averaged = {"matches": len(common), "clock_count": len(clocks),
                      "mean_match_loss": {
                          kind: fsum(row["mean_match_loss"][kind] for row in per_clock.values()) / len(clocks)
                          if common else None for kind in KINDS},
                      **{name: fsum(row[name] for row in per_clock.values()) / len(clocks)
                         if common else None for name in DELTAS}}
    return {"clocks": clocks, "common_match_ids": common,
            "per_clock": per_clock, "clock_averaged": clock_averaged,
            "lower_is_better": True, "averaging_note": AVERAGING_NOTE,
            "sampling_uncertainty": "not estimated; clocks and weekly fits are not independent samples",
            "usage": "retrospective conditional intensity comparison, not pre-match probability or causal player value"}
