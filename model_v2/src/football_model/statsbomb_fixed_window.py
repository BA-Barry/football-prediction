"""Crop a goal-split timing replay to fixed [0,2700] half-clock windows.

The input is a complete goal-only path from timing_precision_experiment.
This is not another event parser: clocks, lineups and score states are never
reconstructed or repaired. In particular, the second half keeps the complete
source first-half score and personnel state, including stoppage-time changes.
"""

from collections import defaultdict
from copy import deepcopy
from math import isclose, isfinite


WINDOW_SECONDS = 2700.0


class FixedWindowCropError(ValueError):
    """A complete goal/right-boundary path is not available for safe cropping."""


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0 or positive and value <= 0):
        raise FixedWindowCropError(f"{name} must be finite nonnegative clock data")
    return float(value)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise FixedWindowCropError(f"{name} must be a nonnegative integer")
    return value


def _pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise FixedWindowCropError(f"{name} must contain home and away goal counts")
    return [_count(v, name) for v in value]


def _clocked_records(values, name, period_ends):
    if not isinstance(values, list):
        raise FixedWindowCropError(f"{name} requires explicit period/second records")
    for value in values:
        if not isinstance(value, dict) or type(value.get("period")) is not int or value["period"] not in (1, 2):
            raise FixedWindowCropError(f"{name} requires a source period")
        second = _number(value.get("second"), f"{name} second")
        if second > period_ends[value["period"]]:
            raise FixedWindowCropError(f"{name} exceeds the source half end")
    return values


def crop_fixed_window(match):
    """Return a new JointStintIntensityModel-compatible goal-only match.

    Goals at exactly 2700 remain. A crossing interval retains only its exposure
    before 2700 and no right-boundary goal. Explicit goal attribution must prove
    every source goal was individually split at its interval's right endpoint;
    counts alone cannot establish a goal clock. Source objects are untouched.
    """
    if not isinstance(match, dict):
        raise FixedWindowCropError("A complete timing replay match is required")
    diagnostics = match.get("diagnostics")
    if not isinstance(diagnostics, dict) or "fixed_window_seconds_by_half" in diagnostics:
        raise FixedWindowCropError("Expected full timing replay diagnostics, not a cropped path")
    teams = (match.get("home_team_id"), match.get("away_team_id"))
    if any(type(t) is not int or t < 1 for t in teams) or teams[0] == teams[1]:
        raise FixedWindowCropError("Two distinct source team IDs are required")
    if not isinstance(match.get("match_id"), str) or not match["match_id"].startswith("statsbomb:match:"):
        raise FixedWindowCropError("Keep the namespaced StatsBomb match ID")
    lengths = diagnostics.get("period_seconds")
    if not isinstance(lengths, dict) or set(lengths) != {"1", "2"}:
        raise FixedWindowCropError("Both true source half durations are required")
    ends = {p: _number(lengths[str(p)], "source half duration", positive=True) for p in (1, 2)}
    for period, duration in ends.items():
        if duration < WINDOW_SECONDS:
            raise FixedWindowCropError(
                f"Source half {period} duration {duration} is shorter than 2700; "
                f"never extend exposure; source period_seconds={lengths!r}")
    full_goals = _pair(diagnostics.get("event_goals"), "source event_goals")
    attribution = _clocked_records(diagnostics.get("goal_attribution"), "goal_attribution", ends)
    conflicts = _clocked_records(diagnostics.get("actor_conflicts"), "actor_conflicts", ends)
    goal_index = {}
    for goal in attribution:
        eid = goal.get("event_id")
        if not isinstance(eid, str) or not eid or eid in goal_index:
            raise FixedWindowCropError("Goal attribution needs distinct source goal IDs")
        if goal.get("scoring_team_id") not in teams:
            raise FixedWindowCropError("Goal attribution needs the source scoring team")
        _pair(goal.get("score_before"), "goal score_before")
        goal_index[eid] = goal

    roster, player_rows = {}, match.get("player_minutes")
    if not isinstance(player_rows, list) or not player_rows:
        raise FixedWindowCropError("Source player_minutes roster is required, including zero exposure")
    for player in player_rows:
        if not isinstance(player, dict):
            raise FixedWindowCropError("Invalid source player_minutes row")
        pid, team = player.get("player_id"), player.get("team_id")
        if (not isinstance(pid, str) or not pid.startswith("statsbomb:player:")
                or pid in roster or team not in teams):
            raise FixedWindowCropError("Distinct source player/team identities are required")
        roster[pid] = team

    source_stints = match.get("stints")
    if not isinstance(source_stints, list) or not source_stints:
        raise FixedWindowCropError("Nonempty goal-split source stints are required")
    previous, last_period, seen_stints = {1: 0.0, 2: 0.0}, 1, set()
    score, assigned, cropped, kept_goals = [0, 0], set(), [], [0, 0]
    for row in source_stints:
        if not isinstance(row, dict):
            raise FixedWindowCropError("Invalid source stint")
        sid, period = row.get("stint_id"), row.get("period")
        if not isinstance(sid, str) or not sid or sid in seen_stints:
            raise FixedWindowCropError("Distinct source stint IDs are required")
        seen_stints.add(sid)
        if type(period) is not int or period not in (1, 2) or period < last_period:
            raise FixedWindowCropError("Source stints must retain two-half chronological order")
        last_period = period
        start = _number(row.get("start_second"), "stint start")
        end = _number(row.get("end_second"), "stint end", positive=True)
        duration = _number(row.get("duration_seconds"), "stint duration", positive=True)
        if (end <= start or end > ends[period] or not isclose(start, previous[period], rel_tol=0, abs_tol=1e-7)
                or not isclose(end - start, duration, rel_tol=0, abs_tol=1e-7)):
            raise FixedWindowCropError("Source stints must be contiguous positive full-half intervals")
        previous[period] = end
        if "home_xg" in row or "away_xg" in row:
            raise FixedWindowCropError("Use the goal-only timing replay, not uncut xg targets")
        for team, column in zip(teams, ("home_players", "away_players")):
            players = row.get(column)
            if (not isinstance(players, list) or not 7 <= len(players) <= 11
                    or any(not isinstance(p, str) for p in players) or len(set(players)) != len(players)
                    or any(roster.get(p) != team for p in players)):
                raise FixedWindowCropError("Stint players must be distinct IDs from the source team roster")
        if row.get("home_man_difference") != len(row["home_players"]) - len(row["away_players"]):
            raise FixedWindowCropError("Source man difference disagrees with source lineups")
        start_score = [_count(row.get(f"start_{s}_goals"), "start score") for s in ("home", "away")]
        if start_score != score or row.get("home_score_difference") != score[0] - score[1]:
            raise FixedWindowCropError("Full source score flow must include first-half stoppage goals")
        targets = [_count(row.get(f"{s}_goals"), "stint goal target") for s in ("home", "away")]
        ids = row.get("goal_event_ids")
        if (not isinstance(ids, list) or any(not isinstance(eid, str) or not eid for eid in ids)
                or len(ids) != sum(targets) or sum(targets) > 1):
            raise FixedWindowCropError("Goals must be individually split and match goal_event_ids")
        counted = [0, 0]
        for eid in ids:
            goal = goal_index.get(eid)
            if goal is None or eid in assigned:
                raise FixedWindowCropError("Every goal_event_id needs one explicit source goal attribution")
            if (goal["period"] != period or not isclose(goal["second"], end, rel_tol=0, abs_tol=1e-7)
                    or _pair(goal["score_before"], "goal score_before") != start_score):
                raise FixedWindowCropError("Every source goal must be split at its known right boundary")
            if any(goal.get(k) != row[k] for k in ("home_players", "away_players")):
                raise FixedWindowCropError("Goal attribution and its right-boundary stint lineup disagree")
            counted[teams.index(goal["scoring_team_id"])] += 1
            assigned.add(eid)
        if counted != targets:
            raise FixedWindowCropError("Goal target sides disagree with source goal_event_ids")
        if start < WINDOW_SECONDS:
            kept = deepcopy(row)
            kept["source_end_second"], kept["source_duration_seconds"] = end, duration
            if end > WINDOW_SECONDS:
                kept.update(end_second=WINDOW_SECONDS, duration_seconds=WINDOW_SECONDS - start,
                            home_goals=0, away_goals=0, goal_event_ids=[])
            for i, side in enumerate(("home", "away")):
                kept_goals[i] += kept[f"{side}_goals"]
            cropped.append(kept)
        score = [score[i] + targets[i] for i in (0, 1)]
    if (any(not isclose(previous[p], ends[p], rel_tol=0, abs_tol=1e-7) for p in (1, 2))
            or score != full_goals or assigned != set(goal_index)):
        raise FixedWindowCropError("Complete source half coverage and all source goals must reconcile")

    result = deepcopy(match)
    result["stints"] = cropped
    exposure = defaultdict(float)
    for row in cropped:
        for column in ("home_players", "away_players"):
            for pid in row[column]:
                exposure[pid] += row["duration_seconds"]
    for player in result["player_minutes"]:
        player["on_pitch_seconds"] = exposure[player["player_id"]]
        player["minutes"] = exposure[player["player_id"]] / 60
        player["minutes_basis"] = "retained fixed half windows under source timing scenario, not full-match played minutes"
    retained = [g for g in attribution if g["second"] <= WINDOW_SECONDS]
    excluded = [g for g in attribution if g["second"] > WINDOW_SECONDS]
    updated = result["diagnostics"]
    updated.update(source_diagnostics=deepcopy(diagnostics), source_period_seconds=deepcopy(lengths),
        fixed_window_seconds_by_half={"1": WINDOW_SECONDS, "2": WINDOW_SECONDS},
        period_seconds={"1": WINDOW_SECONDS, "2": WINDOW_SECONDS},
        full_goals=list(full_goals), window_goals=kept_goals, event_goals=list(kept_goals),
        window_excluded_goals=[full_goals[i] - kept_goals[i] for i in (0, 1)],
        excluded_goal_event_ids=[g["event_id"] for g in excluded], excluded_goal_attribution=deepcopy(excluded),
        goal_attribution=deepcopy(retained), actor_conflicts=deepcopy([c for c in conflicts if c["second"] <= WINDOW_SECONDS]),
        total_duration_seconds=sum(s["duration_seconds"] for s in cropped),
        all_window_goals_preserved=True, all_source_goals_preserved=not excluded,
        player_minutes_basis="fixed [0,2700] each half; scenario exposure, not actual full-match minutes",
        usage="retrospective fixed-window goal-only sensitivity, not full-FT or pre-match probabilities",
        crop_rule="source goal clocks verified at stint right endpoints; boundary 2700 included; no actor-derived masks",
        score_state="source full score/personnel path retained, including first-half stoppage effects on second-half state")
    return result
