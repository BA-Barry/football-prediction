"""Same-cohort description of fixed and training-size-scaled player penalties.

The six caller-declared score tracks separate a change in the selected base
penalty from a change in its scaling policy. These are loss differences on an
observed historical state path, not causal effects or pre-match probabilities.
Clock alternatives describe the same matches and are never counted as extra
independent observations. No model fitting or parameter choice occurs here.
"""

from math import fsum
from statistics import median, stdev

from .full_season_experiment import summarize
from .regularization_comparison import AVERAGING_NOTE, _clocks, _common, _identities, _index


TRACKS = ("team", "fixed100", "fixed_old", "fixed_new", "scaled_old", "scaled_new")
CONTRAST_PAIRS = {
    "scaled_new_minus_fixed_old": ("scaled_new", "fixed_old"),
    "scaled_new_minus_fixed_new": ("scaled_new", "fixed_new"),
    "scaled_old_minus_fixed_old": ("scaled_old", "fixed_old"),
    "fixed_new_minus_fixed_old": ("fixed_new", "fixed_old"),
    "scaled_new_minus_scaled_old": ("scaled_new", "scaled_old"),
    "fixed_old_minus_team": ("fixed_old", "team"),
    "scaled_new_minus_team": ("scaled_new", "team"),
    "scaled_new_minus_fixed100": ("scaled_new", "fixed100"),
}
DECOMPOSITION_NOTE = (
    "Arithmetic decomposition, not causal attribution: total = policy at new base + "
    "base change at fixed policy = policy at old base + base change at scaled policy."
)


def _decomposition(contrasts):
    total = contrasts["scaled_new_minus_fixed_old"]
    policy_new = contrasts["scaled_new_minus_fixed_new"]
    policy_old = contrasts["scaled_old_minus_fixed_old"]
    base_fixed = contrasts["fixed_new_minus_fixed_old"]
    base_scaled = contrasts["scaled_new_minus_scaled_old"]
    new_path = fsum((policy_new, base_fixed)) if total is not None else None
    old_path = fsum((policy_old, base_scaled)) if total is not None else None
    return {"total": total,
            "policy_at_new": policy_new, "base_at_fixed": base_fixed,
            "policy_at_old": policy_old, "base_at_scaled": base_scaled,
            "policy_at_new_plus_base_at_fixed": new_path,
            "policy_at_old_plus_base_at_scaled": old_path,
            "interaction_policy_by_base": policy_new - policy_old if total is not None else None,
            "total_minus_new_path_sum": total - new_path if total is not None else None,
            "total_minus_old_path_sum": total - old_path if total is not None else None}


def _distribution(values):
    """Describe paired match deltas without an iid or sampling-error claim."""
    if not values:
        return None
    return {"matches": len(values), "mean": fsum(values) / len(values),
            "median": median(values),
            "sample_standard_deviation": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values),
            "wins": sum(value < 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value > 0 for value in values)}


def _aggregate_rows(rows):
    means = {track: fsum(row[f"{track}_loss"] for row in rows) / len(rows) if rows else None
             for track in TRACKS}
    contrasts = {name: fsum(row[name] for row in rows) / len(rows) if rows else None
                 for name in CONTRAST_PAIRS}
    return {"mean_match_loss": means, **contrasts,
            "decomposition": _decomposition(contrasts),
            "contrast_distributions": {
                name: _distribution([row[name] for row in rows]) for name in CONTRAST_PAIRS},
            "paired_matches": rows}


def _comparison(indices, common):
    summaries = {f"{track}_on_common": summarize([indices[track][mid] for mid in common])
                 if common else None for track in TRACKS}
    rows = []
    for mid in common:
        losses = {track: indices[track][mid]["point_process_loss"] for track in TRACKS}
        contrasts = {name: losses[first] - losses[second]
                     for name, (first, second) in CONTRAST_PAIRS.items()}
        rows.append({"match_id": mid, "date": indices["team"][mid]["match_date"],
                     **{f"{track}_loss": loss for track, loss in losses.items()},
                     **contrasts, "decomposition": _decomposition(contrasts)})
    return {"common_match_ids": list(common), **summaries, **_aggregate_rows(rows)}


def compare_scaling(scores):
    """Compare ``scores[clock][track]`` on one all-clock, all-track intersection.

    Every declared clock must have exactly ``TRACKS``. Loss definitions and
    target/date identities agree across all observed overlaps, even those
    outside the intersection; state-path partitions agree within each clock.
    An empty common cohort yields None summaries and empty paired rows, never
    a fabricated zero loss. Track labels describe settings asserted by the
    caller; scores alone cannot establish which parameters were fitted.

    Clock-averaged distributions first average each match's loss difference
    over the declared clocks, retaining one row per shared match. Their sample
    standard deviation is descriptive spread, not a standard error or CI.
    """
    clocks = _clocks(scores)
    indices = {}
    for clock in clocks:
        if not isinstance(scores[clock], dict) or set(scores[clock]) != set(TRACKS):
            raise ValueError("Every clock requires exactly team, fixed100, fixed_old, fixed_new, "
                             "scaled_old and scaled_new score tracks")
        indices[clock] = {track: _index(scores[clock][track]) for track in TRACKS}
    _identities(indices)
    common = _common(indices)
    per_clock = {clock: _comparison(indices[clock], common) for clock in clocks}

    averaged_rows = []
    for index, mid in enumerate(common):
        source_rows = [per_clock[clock]["paired_matches"][index] for clock in clocks]
        losses = {f"{track}_loss": fsum(row[f"{track}_loss"] for row in source_rows) / len(clocks)
                  for track in TRACKS}
        contrasts = {name: fsum(row[name] for row in source_rows) / len(clocks)
                     for name in CONTRAST_PAIRS}
        averaged_rows.append({"match_id": mid, "date": source_rows[0]["date"],
                              **losses, **contrasts,
                              "decomposition": _decomposition(contrasts)})
    averaged = {"matches": len(common), "clock_count": len(clocks),
                **_aggregate_rows(averaged_rows)}
    coverage = {clock: {track: {"supported_matches": len(indices[clock][track]),
                               "outside_common_matches": len(indices[clock][track]) - len(common)}
                        for track in TRACKS} for clock in clocks}
    return {"clocks": clocks, "common_match_ids": common,
            "per_clock": per_clock, "clock_averaged": averaged,
            "coverage": coverage,
            "coverage_note": (
                "One intersection across all clocks and all six tracks; supported counts "
                "are not requested-match coverage, and excluded matches have no imputed loss."),
            "lower_is_better": True, "averaging_note": AVERAGING_NOTE,
            "decomposition_note": DECOMPOSITION_NOTE,
            "distribution_note": (
                "Wins/ties/losses count negative/exact-zero/positive paired loss differences; "
                "sample standard deviation describes spread, not a standard error or confidence interval."),
            "sampling_uncertainty": "not estimated; clocks, weekly fits and matches are not assumed independent",
            "usage": "retrospective conditional intensity comparison, not pre-match probability or causal player value"}
