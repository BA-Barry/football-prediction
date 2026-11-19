"""Conditional historical context bounds, never probabilities or live features.

Normalized minute rows contain event COUNTS, not durations. Source score/card
differences are deliberately unused: their pre/post-event semantics differ.
The caller declares clock offsets; author half labels/ranges remain assumptions.
Unknown branches survive even if all remaining candidates happen to agree.
"""
from collections import Counter, defaultdict
from itertools import product


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _score(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected independent two-sided score")
    return tuple(_integer(v, "score") for v in value)


def _credit(shot):
    if shot["result"] == "OwnGoal":
        return "a" if shot["side"] == "h" else "h"
    return shot["side"] if shot["result"] == "Goal" else None


def goal_window_matching(shots, events, offsets):
    """Maximum bipartite matching; no ordering inferred from IDs or arrays.

    Only count feasibility is returned, not a selected event association. Both
    halves remain eligible where numeric clocks overlap. This cannot validate
    source period labels or a particular non-goal shot's time.
    """
    goals = [s for s in shots if _credit(s) is not None]
    units = [(half, minute, side) for (half, minute), counts in events.items()
             for side, index in (("h", 0), ("a", 1)) for _ in range(counts[index])]
    adjacency = [[j for j, (_, minute, side) in enumerate(units)
                  if side == _credit(s) and minute - s["minute"] in offsets]
                 for s in goals]
    assigned = {}

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    matched = sum(augment(i, set()) for i in range(len(goals)))
    return {"shot_goals": len(goals), "minute_goals": len(units),
            "maximum_matched": matched,
            "all_goals_have_one_to_one_window_match": matched == len(goals) == len(units)}


def reconstruct_archive_context(shots, minute_rows, *, expected_full_time,
                                expected_half_time=None, clock_offsets=(0, 1, 2)):
    """Enumerate pre-shot (score, recorded red cards) bounds under clock assumptions.

    Shots: id, side h/a, minute int, result; xG is retained as supplied. Rows:
    side h/a, half_id 1/2, minute int, goals int, red_cards int. Repeated rows
    are state splits: their events are summed, not deduplicated or timed.
    Each candidate coordinate retains its joint states. Arbitrary within-minute
    event prefixes give an outer bound, not equally likely or verified states.
    A goal shot must find its own credited event at that coordinate; otherwise
    that branch stays unknown. Current goals never enter their own pre-score.
    Coverage/unknown flags depend on the historical shot result. They MUST NOT
    be inputs to a model predicting that same shot's goal outcome.
    """
    expected = _score(expected_full_time)
    expected_half = _score(expected_half_time) if expected_half_time is not None else None
    offsets = tuple(clock_offsets)
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("Need distinct declared clock offsets")
    for offset in offsets:
        _integer(offset, "clock offset")
    seen = set()
    for shot in shots:
        if not shot.get("id") or str(shot["id"]) in seen:
            raise ValueError("Missing or duplicate shot ID")
        seen.add(str(shot["id"]))
        _integer(shot["minute"], "shot minute")
        if shot["side"] not in ("h", "a") or shot["result"] not in (
                "Goal", "OwnGoal", "BlockedShot", "MissedShots", "SavedShot", "ShotOnPost"):
            raise ValueError("Invalid shot side or result")
    events, coordinate_sides, repeated = defaultdict(lambda: [0, 0, 0, 0]), defaultdict(set), Counter()
    for row in minute_rows:
        if row["side"] not in ("h", "a") or row["half_id"] not in (1, 2):
            raise ValueError("Invalid minute side or author half")
        minute = _integer(row["minute"], "source minute")
        key = (row["half_id"], minute)
        index = 0 if row["side"] == "h" else 1
        events[key][index] += _integer(row["goals"], "goal count")
        events[key][index + 2] += _integer(row["red_cards"], "card count")
        coordinate_sides[key].add(row["side"])
        repeated[(row["side"], *key)] += 1
    if not events or not any(k[0] == 1 for k in events) or not any(k[0] == 2 for k in events):
        raise ValueError("Need source rows for both author halves")
    counted = tuple(sum(v[i] for v in events.values()) for i in (0, 1))
    counted_half = tuple(sum(v[i] for k, v in events.items() if k[0] == 1) for i in (0, 1))
    shot_counted = tuple(sum(_credit(s) == side for s in shots) for side in ("h", "a"))
    matching = goal_window_matching(shots, events, offsets)
    reasons = []
    if counted != expected:
        reasons.append("minute_full_time_goal_accounting_conflict")
    if expected_half is not None and counted_half != expected_half:
        reasons.append("minute_half_time_goal_accounting_conflict")
    if shot_counted != expected:
        reasons.append("shot_full_time_goal_accounting_conflict")
    if not matching["all_goals_have_one_to_one_window_match"]:
        reasons.append("goal_landmarks_incompatible_with_declared_clock_window")
    max_first = max(k[1] for k in events if k[0] == 1)
    # Explicit research assumption: author half-1 observed range bounds its
    # candidate coordinates; half 2 starts at 46. Missing half-2 end rows are
    # UNKNOWN, never silently extrapolated from the final score.
    output = []
    for shot in shots:
        item = {"shot_id": str(shot["id"]), "side": shot["side"], "minute": shot["minute"],
                "result": shot["result"], "source_xG": shot.get("xG"), "branches": [],
                "joint_candidate_bounds": [], "all_branches_covered": False,
                "window_unique_joint_state": None, "window_unique_score": None,
                "window_unique_own_score_difference": None,
                "window_unique_own_red_card_record_difference": None}
        if reasons:
            item["status"] = "withheld_source_inconsistency"
            item["unknown_reasons"] = list(reasons)
            output.append(item)
            continue
        union = set()
        for offset in offsets:
            minute = shot["minute"] + offset
            halves = ([1] if minute <= max_first else []) + ([2] if minute >= 46 else [])
            if not halves:
                item["branches"].append({"offset": offset, "half_id": None, "minute": minute,
                                         "unknown_reason": "no_author_half_candidate"})
            for half in halves:
                key = (half, minute)
                branch = {"offset": offset, "half_id": half, "minute": minute}
                item["branches"].append(branch)
                if coordinate_sides.get(key) != {"h", "a"}:
                    branch["unknown_reason"] = "missing_two_sided_minute_coordinate"
                    continue
                current = list(events[key])
                credit = _credit(shot)
                if credit is not None:
                    index = 0 if credit == "h" else 1
                    if current[index] == 0:
                        branch["unknown_reason"] = "current_goal_not_identified_at_coordinate"
                        continue
                    current[index] -= 1
                earlier = [sum(v[i] for k, v in events.items() if k < key) for i in range(4)]
                bounds = sorted({tuple(base + addition for base, addition in zip(earlier, additions))
                                 for additions in product(*(range(v + 1) for v in current))})
                branch["joint_candidate_bounds"] = [list(v) for v in bounds]
                union.update(bounds)
        item["joint_candidate_bounds"] = [list(v) for v in sorted(union)]
        unknown = sorted({b["unknown_reason"] for b in item["branches"] if "unknown_reason" in b})
        item["unknown_reasons"] = unknown
        item["all_branches_covered"] = bool(item["branches"]) and not unknown
        if item["all_branches_covered"]:
            scores = {(v[0], v[1]) for v in union}
            score_diff = {(v[0]-v[1])*(1 if shot["side"] == "h" else -1) for v in union}
            red_diff = {(v[2]-v[3])*(1 if shot["side"] == "h" else -1) for v in union}
            item["window_unique_joint_state"] = list(next(iter(union))) if len(union) == 1 else None
            item["window_unique_score"] = list(next(iter(scores))) if len(scores) == 1 else None
            item["window_unique_own_score_difference"] = next(iter(score_diff)) if len(score_diff) == 1 else None
            item["window_unique_own_red_card_record_difference"] = next(iter(red_diff)) if len(red_diff) == 1 else None
        item["status"] = ("unknown_clock_or_association_branch" if unknown else
                          "conditional_unique_joint" if len(union) == 1 else "conditional_state_set")
        output.append(item)
    return {"research_only": True, "clock_offsets": list(offsets),
            "clock_assumption": "Source minute = shot minute + a declared offset; empirical sensitivity hypotheses, not official clock alignment.",
            "period_assumption": "Author half1 observed maximum bounds candidate half1; half2 begins at 46. Author labels/ranges are not official period or end markers.",
            "bound_semantics": "Outer joint state bounds; no probabilities. Global matching need not extend every local branch. Unknown branches prevent uniqueness.",
            "usage_limit": "Coverage/unknown uses the historical shot result; never a feature for predicting that same shot outcome. Do not select only unique-state shots for training.",
            "red_card_semantics": "Recorded sanctions only; own minus opponent is positive for more own sanctions. Player identities and actual on-pitch numbers unverified.",
            "independent_full_time": list(expected), "counted_minute_goals": list(counted),
            "counted_shot_goals": list(shot_counted), "counted_author_half1_goals": list(counted_half),
            "withhold_reasons": reasons, "goal_window_matching": matching,
            "author_first_half_max_minute": max_first,
            "extra_repeated_team_half_minute_rows": sum(v-1 for v in repeated.values()),
            "joint_state_columns": ["home_goals", "away_goals", "home_recorded_red_cards", "away_recorded_red_cards"],
            "shots": output}
