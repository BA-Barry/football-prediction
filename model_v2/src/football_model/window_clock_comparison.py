"""Pure same-cohort summaries for target-window and clock sensitivity.

Within a target/window/clock, team and player losses use the same actual path.
Across windows the responses differ, so only target-specific player increments
are described; raw losses are never subtracted as an accuracy comparison.
"""

from math import isfinite

from .rolling_player_experiment import compare_supported_week


WINDOWS = ("full", "fixed45")
CLOCKS = ("exact", "floor", "nearest_half_up", "ceil")
KINDS = ("team", "players")


def _index(values):
    result = {}
    for prediction in values["predictions"]:
        mid = prediction["match_id"]
        if not isinstance(mid, str) or not mid or mid in result:
            raise ValueError("Predictions require distinct nonempty match IDs")
        for field in ("actual_home_target", "actual_away_target", "point_process_loss",
                      "expected_home_target", "expected_away_target"):
            value = prediction[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"Prediction {field} must be finite numeric data")
        if any(prediction[f"actual_{side}_target"] < 0 for side in ("home", "away")):
            raise ValueError("Goal targets must not be negative")
        result[mid] = prediction
    return result


def _target_identity(prediction):
    return (prediction["match_date"], prediction["actual_home_target"],
            prediction["actual_away_target"], prediction["loss_kind"])


def _intersection(groups):
    return sorted(set.intersection(*(set(group) for group in groups)))


def _pair(indices, common):
    return compare_supported_week(
        {"predictions": [indices["team"][mid] for mid in common]},
        {"predictions": [indices["players"][mid] for mid in common]})


def _intensity_spread(indices, clocks, common):
    rows = []
    for mid in common:
        values = {clock: indices[clock]["players"][mid]["expected_home_target"]
                  + indices[clock]["players"][mid]["expected_away_target"] for clock in clocks}
        rows.append({"match_id": mid, "conditional_integrated_intensity_by_clock": values,
                     "range_width": max(values.values()) - min(values.values())})
    return {"clock_count": len(clocks), "matches": len(rows), "rows": rows,
            "mean_range_width": sum(r["range_width"] for r in rows) / len(rows) if rows else None,
            "max_range_width": max((r["range_width"] for r in rows), default=None),
            "interpretation": "same target only; conditional integrated goals, not 1X2 error, confidence interval or clock ranking"}


def compare_tracks(scores):
    """Summarize nested ``scores[window][clock][team|players]`` without fitting.

    Both windows must retain the same declared clock set; an exact-only set is
    supported. A match's date/goal target must agree across clocks of one
    window. Fixed-window goals may differ from full goals but cannot exceed
    them. Checks cover every observed overlap, not only the final intersection.
    Unknown-player skips stay absent; empty support is represented by None/[]
    rather than a fabricated zero loss or coefficient.
    """
    if set(scores) != set(WINDOWS):
        raise ValueError("Both full and fixed45 target windows are required")
    clock_set = set(scores["full"])
    if not clock_set or "exact" not in clock_set or not clock_set <= set(CLOCKS):
        raise ValueError("A nonempty declared clock set including exact is required")
    if set(scores["fixed45"]) != clock_set:
        raise ValueError("Full and fixed45 windows must retain the same clock set")
    clocks = [clock for clock in CLOCKS if clock in clock_set]
    indices, target_references = {}, {}
    for window in WINDOWS:
        indices[window], target_references[window] = {}, {}
        for clock in clocks:
            if set(scores[window][clock]) != set(KINDS):
                raise ValueError("Every clock requires team and players score tracks")
            indices[window][clock] = {kind: _index(scores[window][clock][kind]) for kind in KINDS}
            for kind in KINDS:
                for mid, prediction in indices[window][clock][kind].items():
                    previous = target_references[window].setdefault(mid, prediction)
                    if _target_identity(previous) != _target_identity(prediction):
                        raise ValueError("Within one window, clocks must retain the same match date, goal target and loss kind")
    for mid in target_references["full"].keys() & target_references["fixed45"].keys():
        full, fixed = target_references["full"][mid], target_references["fixed45"][mid]
        if full["match_date"] != fixed["match_date"]:
            raise ValueError("Cross-window records must refer to the same match date")
        if full["loss_kind"] != fixed["loss_kind"]:
            raise ValueError("Cross-window increments need compatible loss definitions")
        if any(fixed[f"actual_{side}_target"] > full[f"actual_{side}_target"] for side in ("home", "away")):
            raise ValueError("Fixed-window goal target cannot exceed full target for either team")

    windows = {}
    for window in WINDOWS:
        per_clock = {}
        for clock in clocks:
            common = _intersection(indices[window][clock].values())
            per_clock[clock] = _pair(indices[window][clock], common)
        shared = _intersection(indices[window][clock][kind] for clock in clocks for kind in KINDS)
        windows[window] = {
            "per_clock": per_clock,
            "all_clocks_common": {
                "common_match_ids": shared,
                "comparisons": {clock: _pair(indices[window][clock], shared) for clock in clocks},
                "player_integrated_intensity_spread": _intensity_spread(indices[window], clocks, shared),
                "interpretation": "within-window same goal target; each clock's team/player comparison uses its own identical path"}}

    common_all = _intersection(indices[window][clock][kind]
                               for window in WINDOWS for clock in clocks for kind in KINDS)
    by_clock = {}
    for clock in clocks:
        full = _pair(indices["full"][clock], common_all)
        fixed = _pair(indices["fixed45"][clock], common_all)
        full_delta, fixed_delta = full["mean_player_minus_team_loss"], fixed["mean_player_minus_team_loss"]
        paired_rows = []
        if common_all:
            full_rows = {r["match_id"]: r for r in full["paired_matches"]}
            fixed_rows = {r["match_id"]: r for r in fixed["paired_matches"]}
            for mid in common_all:
                a, b = full_rows[mid], fixed_rows[mid]
                paired_rows.append({"match_id": mid, "date": a["date"],
                    "full_player_minus_team_loss": a["player_minus_team_loss"],
                    "fixed_player_minus_team_loss": b["player_minus_team_loss"],
                    "fixed_minus_full_increment_difference": b["player_minus_team_loss"] - a["player_minus_team_loss"]})
        by_clock[clock] = {
            "full": full, "fixed45": fixed,
            "full_player_minus_team_loss": full_delta,
            "fixed_player_minus_team_loss": fixed_delta,
            "fixed_minus_full_increment_difference": None if full_delta is None or fixed_delta is None else fixed_delta - full_delta,
            "paired_target_specific_increment_differences": paired_rows}
    return {"clocks": clocks, "windows": windows,
        "all_windows_clocks_common": {
            "common_match_ids": common_all, "by_clock": by_clock,
            "interpretation": "same matches but different full/fixed goal targets; difference of target-specific player increments is descriptive, NOT a same-target raw-loss accuracy comparison or causal effect"},
        "sampling_uncertainty": "not estimated; intersections may be selected and updates share historical teams",
        "usage": "retrospective window/clock sensitivity only; no pre-match probabilities, zero-prior imputation or clock selection"}
