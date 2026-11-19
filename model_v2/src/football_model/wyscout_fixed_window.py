"""Goal-only Wyscout fixed-90-minute exploratory lineup paths.

Each half's target/exposure window is exactly [0, 2700], not a last-action
estimate of the whistle. All first-half events still update the next half's
initial state. Substitution clocks are three declared approximations using
ONLY reported minutes. Actor actions never move a substitution or censor a
goal. The result is neither actual player seconds nor full-FT score modelling.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite

from .wyscout_probe import goal_team, substitutions, tags


SCENARIOS = ("early", "middle", "late")
WINDOW_SECONDS = 2700.0


class FixedWindowParseError(ValueError):
    """The declared approximation cannot form a path without repairing data."""


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise FixedWindowParseError(f"{label} must be an integer >= {minimum}")
    return value


def substitution_clock(minute, scenario):
    """A declared hypothesis; not a fitted or verified Wyscout rounding rule."""
    _integer(minute, "substitution minute")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown fixed substitution scenario")
    period = 1 if minute <= 45 else 2
    nominal = minute * 60 if period == 1 else (minute - 45) * 60
    shift = {"early": 60, "middle": 30, "late": 0}[scenario]
    return period, float(max(0, nominal - shift))


def _red_minute(value):
    if value in (None, "null", "", "None"):
        return None
    if type(value) is int:
        minute = value
    elif isinstance(value, str) and value.isdecimal():
        minute = int(value)
    else:
        raise FixedWindowParseError("Unusable metadata redCards minute")
    if minute < 0:
        raise FixedWindowParseError("Negative metadata redCards minute")
    return None if minute == 0 else minute


def _prepare(meta, events):
    mid = _integer(meta.get("wyId"), "match wyId", 1)
    if meta.get("duration", "Regular") != "Regular":
        raise FixedWindowParseError("Only regular two-half matches are supported")
    try:
        kickoff = datetime.strptime(meta["dateutc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError) as error:
        raise FixedWindowParseError("Explicit source UTC dateutc required") from error
    raw_teams = meta.get("teamsData")
    if not isinstance(raw_teams, dict) or len(raw_teams) != 2:
        raise FixedWindowParseError("Exactly two team metadata records required")
    by_side = {}
    roster, starters, subs, metadata_reds = {}, {}, [], {}
    all_roster_ids = set()
    for raw in raw_teams.values():
        side = raw.get("side")
        team = _integer(raw.get("teamId"), "teamId", 1)
        if side not in {"home", "away"} or side in by_side or team in roster:
            raise FixedWindowParseError("Distinct home and away team records required")
        by_side[side] = raw
        formation = raw.get("formation", {})
        lineup, bench = formation.get("lineup"), formation.get("bench", [])
        if not isinstance(lineup, list) or len(lineup) != 11 or not isinstance(bench, list):
            raise FixedWindowParseError("Each starting XI must contain exactly 11 players")
        roster[team], starters[team] = set(), set()
        for entry in lineup + bench:
            pid = _integer(entry.get("playerId"), "roster playerId", 1)
            if pid in all_roster_ids:
                raise FixedWindowParseError("Duplicate roster player or player on both teams")
            all_roster_ids.add(pid)
            roster[team].add(pid)
            minute = _red_minute(entry.get("redCards", 0))
            if minute is not None:
                metadata_reds[(team, pid)] = minute
        starters[team] = {entry["playerId"] for entry in lineup}
        try:
            reported_subs = substitutions(formation)
        except ValueError as error:
            raise FixedWindowParseError(str(error)) from error
        for sub in reported_subs:
            subs.append({"team": team, "out": _integer(sub.get("playerOut"), "playerOut", 1),
                         "in": _integer(sub.get("playerIn"), "playerIn", 1),
                         "minute": _integer(sub.get("minute"), "substitution minute"), "source_order": len(subs)})
    if set(by_side) != {"home", "away"}:
        raise FixedWindowParseError("Both metadata sides required")
    teams = (by_side["home"]["teamId"], by_side["away"]["teamId"])
    metadata_ft = [_integer(by_side[s].get("score"), "metadata FT score") for s in ("home", "away")]
    metadata_ht = [by_side[s].get("scoreHT") for s in ("home", "away")]
    if not isinstance(events, list) or not events:
        raise FixedWindowParseError("A nonempty single-match event list is required")
    rows, seen_ids, previous, tails = [], set(), {}, {}
    full, halftime, window = [0, 0], [0, 0], [0, 0]
    raw_reds = defaultdict(list)
    for index, event in enumerate(events):
        if event.get("matchId") != mid:
            raise FixedWindowParseError("Foreign matchId in single-match events")
        eid = _integer(event.get("id"), "event id", 1)
        if eid in seen_ids:
            raise FixedWindowParseError("Duplicate source event ID")
        seen_ids.add(eid)
        period = {"1H": 1, "2H": 2}.get(event.get("matchPeriod"))
        second = event.get("eventSec")
        if period is None or isinstance(second, bool) or not isinstance(second, (int, float)) or not isfinite(second) or second < 0:
            raise FixedWindowParseError("Only finite nonnegative 1H/2H period clocks are supported")
        second = float(second)
        if period in previous and second < previous[period]:
            raise FixedWindowParseError("Source period clocks go backwards; no event reordering repair")
        previous[period], tails[period] = second, max(second, tails.get(period, 0.0))
        team = _integer(event.get("teamId"), "event teamId", 1)
        if team not in teams:
            raise FixedWindowParseError("Event team is not a match team")
        pid = _integer(event.get("playerId"), "event playerId")
        try:
            marked = tags(event)
            awarded = goal_team(event, set(teams))
        except (ValueError, KeyError, TypeError) as error:
            raise FixedWindowParseError("Unusable original goal/card tags") from error
        goal_side = None if awarded is None else teams.index(awarded)
        if goal_side is not None:
            full[goal_side] += 1
            halftime[goal_side] += period == 1
            window[goal_side] += second <= WINDOW_SECONDS
        red = bool(marked & {1701, 1703})
        row = {"kind": "raw", "index": index, "period": period, "second": second, "event_id": eid,
               "team": team, "player": pid, "goal_side": goal_side, "red": red,
               "is_shot": event.get("eventId") == 10 or event.get("eventName") == "Shot"
                  or event.get("eventName") == "Free Kick" and event.get("subEventId") in {33, 35}}
        rows.append(row)
        if red:
            raw_reds[(team, pid)].append(row)
    shared_rejects, post_red_issues = [], []
    if full != metadata_ft:
        shared_rejects.append(f"Primary goal events FT {full} disagree with metadata FT {metadata_ft}")
    for period in (1, 2):
        if tails.get(period, -1) < WINDOW_SECONDS:
            shared_rejects.append(f"Half {period} observed tail {tails.get(period)} is shorter than fixed 2700 seconds")
    for identity, reds in raw_reds.items():
        affecting = [r for r in reds if r["period"] == 1 or r["second"] <= WINDOW_SECONDS]
        if identity not in metadata_reds or identity[1] not in roster[identity[0]]:
            for red in reds:
                issue = {"event_id": red["event_id"], "team_id": identity[0], "player_id": identity[1],
                         "period": red["period"], "second": red["second"], "issue": "raw_red_not_supported_by_player_metadata"}
                if red in affecting:
                    shared_rejects.append(f"Impacting red event {red['event_id']} unsupported by player metadata")
                else:
                    post_red_issues.append(issue)
        if len(affecting) > 1:
            shared_rejects.append(f"Repeated impacting dismissal events for player {identity[1]}")
        elif len(reds) > 1:
            post_red_issues.extend({"event_id": r["event_id"], "issue": "repeated_post_second_half_red"}
                                   for r in reds if r not in affecting)
    for identity, minute in metadata_reds.items():
        if identity not in raw_reds:
            if minute <= 90:
                shared_rejects.append(f"Metadata dismissed player {identity[1]} has no raw dismissal clock")
            else:
                post_red_issues.append({"team_id": identity[0], "player_id": identity[1], "reported_minute": minute,
                                       "issue": "post90_metadata_red_without_raw_clock_not_applied"})
    return {"mid": mid, "date": kickoff.date().isoformat(), "kickoff": kickoff.isoformat(), "teams": teams,
            "roster": roster, "starters": starters, "subs": subs, "rows": rows, "metadata_ft": metadata_ft,
            "metadata_ht": metadata_ht, "full_goals": full, "halftime_goals": halftime, "window_goals": window,
            "tails": tails, "shared_rejects": shared_rejects, "post_red_issues": post_red_issues}


def _replay(prepared, scenario, player_names):
    h, a = prepared["teams"]
    active = {t: set(prepared["starters"][t]) for t in (h, a)}
    entered = {t: set(active[t]) for t in (h, a)}
    removed, dismissed = ({t: set() for t in (h, a)} for _ in range(2))
    score, targets_total, stints = [0, 0], [0, 0], []
    conflicts, goal_attribution, off_pitch_reds, post_records = [], [], [], []
    sub_clocks = []
    rows = list(prepared["rows"])
    for sub in prepared["subs"]:
        period, second = substitution_clock(sub["minute"], scenario)
        row = {"kind": "sub", "period": period, "second": second, "index": sub["source_order"], **sub}
        rows.append(row)
        sub_clocks.append({"team_id": sub["team"], "player_out": sub["out"], "player_in": sub["in"],
                           "reported_minute": sub["minute"], "period": period, "assumed_second": second,
                           "source_order": sub["source_order"]})
    for period in (1, 2):
        within = sorted((r for r in rows if r["period"] == period),
                        key=lambda r: (r["second"], 0 if r["kind"] == "raw" else 1, r["index"]))
        start, start_score, goals, goal_ids = 0.0, tuple(score), [0, 0], []

        def close(second):
            nonlocal start, start_score, goals, goal_ids
            clock = min(second, WINDOW_SECONDS)
            elapsed = clock - start
            if elapsed < 0 or elapsed == 0 and any(goals):
                raise FixedWindowParseError("Goal has no positive exposure under fixed-window replay")
            if elapsed:
                if not all(7 <= len(active[t]) <= 11 for t in (h, a)):
                    raise FixedWindowParseError("Assumed on-pitch count outside 7..11")
                stints.append({"stint_id": f"wyscout:match:{prepared['mid']}:fixed90:{scenario}:stint:{len(stints)+1}",
                    "period": period, "start_second": start, "end_second": clock, "duration_seconds": elapsed,
                    "home_players": [f"wyscout:player:{p}" for p in sorted(active[h])],
                    "away_players": [f"wyscout:player:{p}" for p in sorted(active[a])],
                    "start_home_goals": start_score[0], "start_away_goals": start_score[1],
                    "home_score_difference": start_score[0] - start_score[1],
                    "home_man_difference": len(active[h]) - len(active[a]),
                    "home_goals": goals[0], "away_goals": goals[1], "goal_event_ids": list(goal_ids)})
            start, start_score, goals, goal_ids = clock, tuple(score), [0, 0], []

        for row in within:
            second, team = row["second"], row["team"]
            if period == 2 and second > WINDOW_SECONDS:
                post_records.append({"kind": row["kind"], "second": second, "event_id": row.get("event_id"),
                                     "is_goal": row.get("goal_side") is not None, "is_red": row.get("red", False)})
                continue
            if second > WINDOW_SECONDS:
                close(WINDOW_SECONDS)
            if row["kind"] == "sub":
                outgoing, incoming = row["out"], row["in"]
                if (outgoing not in active[team] or incoming not in prepared["roster"][team]
                        or incoming in entered[team] | dismissed[team] or outgoing == incoming):
                    raise FixedWindowParseError(f"Illegal {scenario} substitution chain/state at minute {row['minute']} for team {team}")
                close(second)
                active[team].remove(outgoing)
                removed[team].add(outgoing)
                active[team].add(incoming)
                entered[team].add(incoming)
            else:
                pid = row["player"]
                retained = second <= WINDOW_SECONDS
                if (row["is_shot"] or row["goal_side"] is not None) and pid not in active[team]:
                    conflicts.append({"event_id": row["event_id"], "period": period, "second": second,
                        "team_id": team, "player_id": f"wyscout:player:{pid}" if pid else None,
                        "is_goal": row["goal_side"] is not None, "in_target_window": retained,
                        "issue": "actor_outside_assumed_lineup_not_repaired"})
                if row["goal_side"] is not None:
                    side = row["goal_side"]
                    goal_attribution.append({"event_id": row["event_id"], "period": period, "second": second,
                        "in_target_window": retained, "score_before": list(score), "scoring_team_id": (h, a)[side],
                        "home_players": [f"wyscout:player:{p}" for p in sorted(active[h])],
                        "away_players": [f"wyscout:player:{p}" for p in sorted(active[a])]})
                    if retained:
                        goals[side] += 1
                        targets_total[side] += 1
                        goal_ids.append(row["event_id"])
                        close(second)
                    score[side] += 1
                    start_score = tuple(score)
                if row["red"]:
                    if pid in dismissed[team]:
                        raise FixedWindowParseError(f"Repeated impacting red for player {pid}")
                    if pid in active[team]:
                        close(second)
                        active[team].remove(pid)
                    else:
                        off_pitch_reds.append({"event_id": row["event_id"], "player_id": f"wyscout:player:{pid}",
                                              "period": period, "second": second, "man_count_changed": False})
                    dismissed[team].add(pid)
        close(WINDOW_SECONDS)
    if targets_total != prepared["window_goals"]:
        raise FixedWindowParseError("Replay did not preserve every goal in the fixed target windows")
    seconds = defaultdict(float)
    for stint in stints:
        for team, column in ((h, "home_players"), (a, "away_players")):
            for pid in stint[column]:
                seconds[(team, pid)] += stint["duration_seconds"]
    def name(pid):
        return player_names.get(pid, player_names.get(f"wyscout:player:{pid}", str(pid)))
    return {"match_id": f"wyscout:match:{prepared['mid']}", "source_match_id": prepared["mid"],
        "match_date": prepared["date"], "source_kickoff_utc": prepared["kickoff"],
        "home_team_id": h, "away_team_id": a, "stints": stints,
        "player_minutes": [{"team_id": team, "player_id": f"wyscout:player:{pid}", "player_name": name(pid),
            "started": pid in prepared["starters"][team], "on_pitch_seconds": seconds[(team, f"wyscout:player:{pid}")],
            "minutes": seconds[(team, f"wyscout:player:{pid}")] / 60}
            for team in (h, a) for pid in sorted(entered[team])],
        "diagnostics": {"scenario": scenario, "fixed_window_seconds_by_half": {"1": WINDOW_SECONDS, "2": WINDOW_SECONDS},
            "total_duration_seconds": sum(s["duration_seconds"] for s in stints), "window_goals": targets_total,
            "full_source_goals": list(prepared["full_goals"]), "source_metadata_ft": list(prepared["metadata_ft"]),
            "full_source_halftime_goals": list(prepared["halftime_goals"]), "source_metadata_ht": prepared["metadata_ht"],
            "observed_last_event_seconds_not_whistle": {str(k): v for k, v in prepared["tails"].items()},
            "substitution_clocks": sub_clocks, "actor_conflicts": conflicts, "goal_attribution": goal_attribution,
            "off_pitch_reds_no_man_change": off_pitch_reds, "post_second_half_records_not_applied": post_records,
            "post_second_half_red_issues": prepared["post_red_issues"],
            "identity_namespace": "wyscout; native team IDs, namespaced match/player IDs; not StatsBomb",
            "state_path_description": "raw goals/cards before same-second substitutions; raw source order and substitution source order retained; all 1H states carried into 2H",
            "player_minutes_basis": "only two fixed 45-minute windows under declared approximate lineup clocks; not actual played minutes",
            "usage": "exploratory fixed-window goals and approximate player states; not full-FT score or pre-match probabilities; no xg"}}


def parse_match(meta, events, player_names=None):
    """Parse ONE streamed match into scenarios and explicit rejection reasons.

    ``events`` is a single group from ``wyscout_data.iter_match_events``. This
    adapter does not open a whole-season event file or fit any model. Only
    ``common_supported`` matches belong in three-scenario paired comparisons.
    """
    try:
        prepared = _prepare(meta, events)
    except FixedWindowParseError as error:
        return {"match_id": f"wyscout:match:{meta.get('wyId')}", "scenarios": {}, "common_supported": False,
                "rejects": {s: [str(error)] for s in SCENARIOS}, "diagnostics": {"shared_reject_reasons": [str(error)]}}
    diagnostics = {"full_source_goals": prepared["full_goals"], "source_metadata_ft": prepared["metadata_ft"],
        "window_goals": prepared["window_goals"], "full_source_halftime_goals": prepared["halftime_goals"],
        "source_metadata_ht": prepared["metadata_ht"], "source_halftime_agrees": prepared["metadata_ht"] == prepared["halftime_goals"],
        "shared_reject_reasons": prepared["shared_rejects"], "post_second_half_red_issues": prepared["post_red_issues"],
        "observed_last_event_seconds_not_whistle": {str(k): v for k, v in prepared["tails"].items()}}
    paths, rejects = {}, {}
    for scenario in SCENARIOS:
        if prepared["shared_rejects"]:
            rejects[scenario] = list(prepared["shared_rejects"])
            continue
        try:
            paths[scenario] = _replay(prepared, scenario, player_names or {})
        except FixedWindowParseError as error:
            rejects[scenario] = [str(error)]
    return {"match_id": f"wyscout:match:{prepared['mid']}", "match_date": prepared["date"], "scenarios": paths,
            "common_supported": not rejects, "rejects": rejects, "diagnostics": diagnostics}
