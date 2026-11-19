"""Research-only score-state descriptions for saved Understat shot records.

There is no period or second-level clock in these records. Only minute < 45 is
interpreted, conditional on the source's whole-match minute convention. Equal
minutes are unordered; neither IDs nor array order establish event precedence.
Whole-match goal accounting may be checked against an independent score, but
that score and shot h_goals/a_goals never supply a pre-shot state.
This is not a full-match timeline, an exposure model, or a production feature.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping


RESULTS = frozenset(("MissedShots", "SavedShot", "BlockedShot", "Goal",
                     "ShotOnPost", "OwnGoal"))
STATE_ORDER = ("trailing", "drawing", "leading")


def _minute(value):
    if isinstance(value, bool):
        raise ValueError("Shot minute must be a nonnegative integer")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError("Shot minute must be a nonnegative integer")


def _score(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError("Independent full-time score must contain two integers")
    return tuple(value)


def _own_state(home, away, side):
    difference = (home-away) * (1 if side == "h" else -1)
    return "leading" if difference > 0 else "trailing" if difference < 0 else "drawing"


def reconstruct_shot_states(shots, *, expected_full_time=None, match_id=None):
    """Return conservative pre-shot score sets and source-quality measurements.

    A Goal credits h_a; an OwnGoal credits the opposite side. For a shot before
    minute 45, strictly smaller minutes establish a prefix. Other goals at the
    same minute may be before or after it, so all resulting score pairs remain.
    The current Goal/OwnGoal is excluded from its own pre-shot score.
    If independent accounting conflicts, derived states are withheld rather
    than repaired. Raw xG is passed through without summing or changing NPxG.
    """
    if not isinstance(shots, Mapping) or set(shots) != {"h", "a"}:
        raise ValueError("Need explicit h/a shot arrays")
    expected = _score(expected_full_time) if expected_full_time is not None else None
    rows, seen, goals, regressions = [], set(), [], []
    for side in ("h", "a"):
        if not isinstance(shots[side], list):
            raise ValueError("Need shot arrays")
        previous = None
        for index, event in enumerate(shots[side]):
            if not isinstance(event, Mapping):
                raise ValueError("Need shot objects")
            source_id = event.get("id")
            uid = str(source_id) if source_id is not None else ""
            if not uid or isinstance(source_id, bool) or uid in seen:
                raise ValueError("Missing or duplicate shot id within a match")
            seen.add(uid)
            minute = _minute(event.get("minute"))
            result = event.get("result")
            if event.get("h_a") != side or result not in RESULTS:
                raise ValueError("Invalid shot side or unknown result semantics")
            if match_id is not None and str(event.get("match_id")) != str(match_id):
                raise ValueError("Shot match identity differs from its source match")
            if previous is not None and minute < previous:
                regressions.append({"side": side, "source_index": index,
                                    "previous_minute": previous, "minute": minute})
            previous = minute
            credit = ("a" if side == "h" else "h") if result == "OwnGoal" else (
                side if result == "Goal" else None)
            row = {
                "shot_id": uid, "minute": minute, "side": side, "result": result,
                "source_side_index": index, "goal_credit": credit,
                "source_xG": event.get("xG"), "situation": event.get("situation"),
                "shotType": event.get("shotType"), "lastAction": event.get("lastAction"),
            }
            rows.append(row)
            if credit is not None:
                goals.append(row)

    counted = tuple(sum(goal["goal_credit"] == side for goal in goals) for side in ("h", "a"))
    consistent = counted == expected if expected is not None else None
    goal_minutes = Counter(goal["minute"] for goal in goals)
    same_goals = defaultdict(list)
    for goal in goals:
        same_goals[goal["minute"]].append(goal)
    counts = Counter()
    for row in rows:
        minute = row["minute"]
        others = [goal for goal in same_goals[minute] if goal["shot_id"] != row["shot_id"]]
        counts["shots"] += 1
        counts["shots_sharing_minute_with_other_goal"] += bool(others)
        row.update(possible_scores=None, unique_score=None, possible_own_states=None,
                   unique_own_state=None, score_is_unique=False, own_state_is_unique=False)
        if minute >= 45:
            row["status"] = "unidentified_period_or_halftime_overlap"
            counts["minute_ge45_unknown"] += 1
            continue
        counts["minute_lt45"] += 1
        if consistent is False:
            row["status"] = "unusable_independent_goal_accounting_conflict"
            counts["early_accounting_conflict"] += 1
            continue
        earlier = [goal for goal in goals if goal["minute"] < minute]
        base_h, base_a = (sum(goal["goal_credit"] == side for goal in earlier)
                          for side in ("h", "a"))
        other_h, other_a = (sum(goal["goal_credit"] == side for goal in others)
                            for side in ("h", "a"))
        possible = [[base_h+h, base_a+a] for h in range(other_h+1) for a in range(other_a+1)]
        states = {_own_state(h, a, row["side"]) for h, a in possible}
        row["possible_scores"] = possible
        row["score_is_unique"] = len(possible) == 1
        row["unique_score"] = possible[0] if row["score_is_unique"] else None
        row["possible_own_states"] = [state for state in STATE_ORDER if state in states]
        row["own_state_is_unique"] = len(states) == 1
        row["unique_own_state"] = next(iter(states)) if len(states) == 1 else None
        row["status"] = "early_unique_score" if len(possible) == 1 else "early_same_minute_score_set"
        counts["early_unique_score" if len(possible) == 1 else "early_ambiguous_score"] += 1
        counts["early_unique_own_state" if len(states) == 1 else "early_ambiguous_own_state"] += 1
    return {
        "research_only": True,
        "clock_limit": "Only minute<45; relies on source whole-match minute counting; no period/seconds.",
        "order_limit": "Equal minutes unordered; source indices and IDs are provenance only.",
        "xg_limit": "Raw per-shot xG preserved, never aggregated or substituted for source NPxG.",
        "independent_full_time": list(expected) if expected is not None else None,
        "counted_goal_credit": list(counted),
        "goal_accounting_consistent": consistent,
        "own_goal_count": sum(goal["result"] == "OwnGoal" for goal in goals),
        "same_minute_multiple_goal_groups": {
            str(minute): count for minute, count in goal_minutes.items() if count > 1},
        "source_minute_regressions": regressions,
        "counts": dict(counts),
        "shots": rows,
    }
