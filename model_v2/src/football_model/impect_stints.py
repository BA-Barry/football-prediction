"""Goals-only retrospective on-pitch reconstruction for IMPECT V3 open data.

V3 clocks use 0 and 10000 second period offsets, not a continuous match clock.
The first KICK_OFF supplies the actual opening clock, which need not be zero;
FINAL_WHISTLE supplies each half's actual end. Event ``duration`` is deliberately
unused: at half-time it can include the gap to the next encoded period. These
actual paths and revised source records are not historically available pre-match
features, player ratings, xG values, or causal estimates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite


class ImpectParseError(ValueError):
    """An on-pitch path cannot be identified without additional assumptions."""


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ImpectParseError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ImpectParseError(f"{name} must be finite and nonnegative")
    return float(value)


def _clock(value, period=None):
    if not isinstance(value, dict):
        raise ImpectParseError("gameTime object is required")
    encoded = _number(value.get("gameTimeInSec"), "gameTimeInSec")
    inferred = 1 if encoded < 10000 else 2 if encoded < 20000 else None
    if inferred is None or period is not None and inferred != period:
        raise ImpectParseError("V3 clock and ordinary periodId disagree")
    return inferred, encoded - (inferred - 1) * 10000


def _timestamp(value):
    if not isinstance(value, str):
        raise ImpectParseError("scheduledDate must include a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ImpectParseError("invalid scheduledDate") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ImpectParseError("scheduledDate must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pid(value):
    return f"impect:player:{value}"


def _position(value):
    if not isinstance(value, str) or not value:
        raise ImpectParseError("nonempty source position required")
    return value


def parse_match(metadata, events, lineups, player_kpis=None, *, expected_score=None):
    """Reconstruct actual-state stints accepted by the goals intensity model.

    Team IDs remain provider-specific integers; player/match IDs are namespaced.
    Same-clock lineup rows are applied atomically without invented replacement
    pairs. Goal/membership ties across the two source lists are rejected. Red
    cards on the pitch require coherent same-clock BANK records; unsupported
    send-off paths are not silently repaired. Optional score comparison must
    come from an independent source. KPI playDuration is an audit, never a target
    or a timing correction; different position rows are summed per player.
    """
    if not isinstance(metadata, dict) or not isinstance(lineups, dict):
        raise ImpectParseError("metadata and lineups must be objects")
    mid = _integer(metadata.get("id"), "match id")
    iteration = _integer(metadata.get("iterationId"), "iterationId")
    home = _integer(metadata.get("homeSquadId"), "homeSquadId")
    away = _integer(metadata.get("awaySquadId"), "awaySquadId")
    if home == away:
        raise ImpectParseError("two distinct squads are required")
    kickoff = _timestamp(metadata.get("scheduledDate"))
    if (_integer(lineups.get("id"), "lineup match id") != mid
            or _integer(lineups.get("iterationId"), "lineup iterationId") != iteration
            or _timestamp(lineups.get("dateTime")) != kickoff):
        raise ImpectParseError("lineup match identity or scheduled datetime disagrees")
    expected = None
    if expected_score is not None:
        if not isinstance(expected_score, (list, tuple)) or len(expected_score) != 2:
            raise ImpectParseError("expected_score must contain home and away goals")
        expected = [_integer(y, "expected goal count", 0) for y in expected_score]

    roster, active, starting, changes = {}, {}, {}, defaultdict(list)
    positions, all_players = {}, set()
    for key, team in (("squadHome", home), ("squadAway", away)):
        squad = lineups.get(key)
        if not isinstance(squad, dict) or _integer(squad.get("id"), "lineup squad id") != team:
            raise ImpectParseError("lineup squad identity disagrees")
        players, starts, substitutions = (squad.get(k) for k in
                                            ("players", "startingPositions", "substitutions"))
        if not all(isinstance(x, list) for x in (players, starts, substitutions)):
            raise ImpectParseError("roster, starting positions and substitutions lists required")
        roster[team], positions[team] = {}, {}
        for row in players:
            if not isinstance(row, dict):
                raise ImpectParseError("roster player must be an object")
            pid = _integer(row.get("id"), "roster player id")
            if pid in all_players:
                raise ImpectParseError("duplicate roster identity or player on both teams")
            all_players.add(pid)
            roster[team][pid] = row.get("name") or str(pid)
        for row in starts:
            if not isinstance(row, dict):
                raise ImpectParseError("starting position must be an object")
            pid = _integer(row.get("playerId"), "starting player id")
            pos = _position(row.get("position"))
            if pid not in roster[team] or pid in positions[team] or pos == "BANK":
                raise ImpectParseError("invalid or duplicate starting position")
            positions[team][pid] = pos
        if len(positions[team]) != 11:
            raise ImpectParseError("each team needs exactly 11 distinct starting players")
        active[team] = set(positions[team])
        starting[team] = set(active[team])
        seen_changes = set()
        for row in substitutions:
            if not isinstance(row, dict):
                raise ImpectParseError("lineup change must be an object")
            pid = _integer(row.get("playerId"), "lineup change player id")
            period, clock = _clock(row.get("gameTime"))
            before, after = _position(row.get("fromPosition")), _position(row.get("toPosition"))
            if pid not in roster[team] or (period, clock, pid) in seen_changes:
                raise ImpectParseError("unknown or duplicate same-clock lineup change player")
            if before == after == "BANK":
                raise ImpectParseError("BANK to BANK is not an identifiable change")
            if row.get("exchangedPlayerId") is not None or row.get("substitutionType") is not None:
                raise ImpectParseError("V4 substitution semantics are not supported by this V3 adapter")
            seen_changes.add((period, clock, pid))
            changes[(period, clock)].append((team, pid, before, after))

    if not isinstance(events, list) or not events:
        raise ImpectParseError("nonempty source events required")
    ordered, ids, indices = [], set(), set()
    for event in events:
        if not isinstance(event, dict):
            raise ImpectParseError("event must be an object")
        eid = _integer(event.get("id"), "event id")
        index = _integer(event.get("index"), "event index", 0)
        if eid in ids or index in indices:
            raise ImpectParseError("duplicate event id/index")
        ids.add(eid)
        indices.add(index)
        period = _integer(event.get("periodId"), "periodId")
        if period not in (1, 2):
            raise ImpectParseError("only ordinary periods 1 and 2 are supported")
        _, clock = _clock(event.get("gameTime"), period)
        kind = _position(event.get("actionType"))
        critical = {"GOAL", "OWN_GOAL", "RED_CARD", "FINAL_WHISTLE"}
        if (kind in critical or event.get("action") in critical) and event.get("action") != kind:
            raise ImpectParseError("critical action/actionType labels must agree")
        if ("RED" in kind and kind != "RED_CARD") or kind in {"SECOND_YELLOW", "SEND_OFF", "DISMISSAL"}:
            raise ImpectParseError("unknown send-off action requires explicit source semantics")
        ordered.append((index, period, clock, kind, event))
    ordered.sort(key=lambda row: row[0])
    model_rows = [row for row in ordered if row[3] in {"GOAL", "OWN_GOAL", "RED_CARD", "FINAL_WHISTLE"}]
    for previous, current in zip(model_rows, model_rows[1:]):
        if (current[1], current[2]) < (previous[1], previous[2]):
            raise ImpectParseError("model event index and period clock ordering disagree")
    anomalies = [{"previous_event_id": a[4]["id"], "event_id": b[4]["id"],
                  "issue": "all_event_index_clock_inversion"}
                 for a, b in zip(ordered, ordered[1:]) if (b[1], b[2]) < (a[1], a[2])]
    starts, ends, opening_events = {}, {}, {}
    for period in (1, 2):
        period_rows = [row for row in ordered if row[1] == period]
        # Do not search past an earlier action for a convenient later kick-off:
        # that could silently discard already recorded match exposure/events.
        opening = period_rows[0] if period_rows else None
        if (opening is None or opening[3] != "KICK_OFF"
                or opening[4].get("action") not in {"KICKOFF_WHISTLE", "KICK_OFF"}):
            raise ImpectParseError("each period must begin with an identifiable KICK_OFF opening whistle")
        starts[period] = opening[2]
        opening_events[str(period)] = {
            "event_id": opening[4]["id"], "event_index": opening[0],
            "source_action": opening[4]["action"], "source_action_type": opening[3],
            "source_game_time": dict(opening[4]["gameTime"]),
            "period_start_second": opening[2],
            "opening_label": "KICKOFF_WHISTLE" if opening[4]["action"] == "KICKOFF_WHISTLE"
                             else "legacy_KICK_OFF_action_label",
        }
        if any(row[2] < starts[period] for row in period_rows):
            raise ImpectParseError("event occurs before recorded period kickoff")
        whistles = [row for row in model_rows if row[1] == period and row[3] == "FINAL_WHISTLE"]
        if len(whistles) != 1 or whistles[0][2] <= 0:
            raise ImpectParseError("each period requires exactly one positive FINAL_WHISTLE")
        ends[period] = whistles[0][2]
        if ends[period] <= starts[period]:
            raise ImpectParseError("FINAL_WHISTLE must follow the recorded period kickoff")
        if any(row[2] > ends[period] for row in model_rows if row[1] == period):
            raise ImpectParseError("model event occurs beyond recorded period end")
        anomalies.extend({"event_id": row[4]["id"], "issue": "nonmodel_event_beyond_period_end"}
                         for row in ordered if row[1] == period and row[2] > ends[period])
    if any(clock > ends[period] for period, clock in changes):
        raise ImpectParseError("lineup change occurs beyond recorded period end")
    if any(clock < starts[period] for period, clock in changes):
        raise ImpectParseError("lineup change occurs before recorded period kickoff")

    def actor(event):
        team = _integer(event.get("squadId"), "event squadId")
        player = event.get("player")
        pid = _integer(player.get("id") if isinstance(player, dict) else None, "event player id")
        if team not in roster or pid not in roster[team]:
            raise ImpectParseError("goal/card actor is not in the squad roster")
        return team, pid

    grouped = defaultdict(list)
    for row in model_rows:
        grouped[(row[1], row[2])].append(row)
    score, stints = [0, 0], []
    removed, dismissed = {home: set(), away: set()}, {home: set(), away: set()}
    status = Counter()
    for period in (1, 2):
        start, start_score = starts[period], tuple(score)

        def close(clock, goal_rows=()):
            nonlocal start, start_score
            duration = clock - start
            goals = [0, 0]
            for row in goal_rows:
                team, pid = actor(row[4])
                if pid not in active[team]:
                    raise ImpectParseError("goal actor is recorded off the pitch")
                side = 0 if team == home else 1
                if row[3] == "OWN_GOAL":
                    side = 1 - side
                goals[side] += 1
            if duration < 0 or duration == 0 and any(goals):
                raise ImpectParseError("goal has no positive exposure or state path goes backwards")
            if duration > 0:
                if not all(7 <= len(active[t]) <= 11 for t in (home, away)):
                    raise ImpectParseError("on-pitch counts must remain within 7..11")
                stints.append({
                    "stint_id": f"impect:match:{mid}:period:{period}:stint:{len(stints) + 1}",
                    "period": period, "start_second": start, "end_second": clock,
                    "duration_seconds": duration,
                    "home_players": [_pid(p) for p in sorted(active[home])],
                    "away_players": [_pid(p) for p in sorted(active[away])],
                    "start_home_goals": start_score[0], "start_away_goals": start_score[1],
                    "home_score_difference": start_score[0] - start_score[1],
                    "home_man_difference": len(active[home]) - len(active[away]),
                    "home_goals": goals[0], "away_goals": goals[1],
                    "goal_event_ids": [row[4]["id"] for row in goal_rows],
                })
            start = clock
            score[0] += goals[0]
            score[1] += goals[1]
            start_score = tuple(score)

        keys = sorted({key for key in grouped if key[0] == period}
                      | {key for key in changes if key[0] == period})
        for key in keys:
            clock = key[1]
            rows, updates = grouped[key], changes[key]
            goals = [row for row in rows if row[3] in {"GOAL", "OWN_GOAL"}]
            reds = [row for row in rows if row[3] == "RED_CARD"]
            membership = [row for row in updates if "BANK" in row[2:]]
            if goals and membership:
                raise ImpectParseError("same-clock goal and membership change have no cross-list order")
            if len(goals) > 1:
                raise ImpectParseError("multiple same-clock goals have no identifiable state exposure")
            if goals:
                close(clock, goals)

            red_players = {home: set(), away: set()}
            for row in reds:
                team, pid = actor(row[4])
                if pid in dismissed[team] or pid in red_players[team]:
                    raise ImpectParseError("duplicate dismissal identity")
                red_players[team].add(pid)
            if membership or any(red_players[t] & active[t] for t in (home, away)):
                close(clock)
            for team in (home, away):
                own = [row for row in updates if row[0] == team]
                outgoing = {pid for _, pid, before, after in own if after == "BANK"}
                incoming = {pid for _, pid, before, after in own if before == "BANK"}
                on_red = red_players[team] & active[team]
                if not on_red <= outgoing:
                    raise ImpectParseError("on-pitch red card needs a coherent same-clock BANK departure")
                if len(outgoing) - len(incoming) != len(on_red):
                    raise ImpectParseError("unpaired lineup membership change is not explained by a red card")
                for _, pid, before, after in own:
                    current = positions[team].get(pid, "BANK")
                    if current != before:
                        raise ImpectParseError("lineup fromPosition disagrees with preceding state")
                    if before == "BANK" and pid in removed[team] | dismissed[team] | red_players[team]:
                        raise ImpectParseError("removed or dismissed player cannot enter again")
                for _, pid, before, after in own:
                    if after == "BANK":
                        active[team].remove(pid)
                        positions[team].pop(pid)
                        removed[team].add(pid)
                        status["bank_departures"] += 1
                    else:
                        positions[team][pid] = after
                        active[team].add(pid)
                        status["bank_entries" if before == "BANK" else "tactical_position_changes"] += 1
                status["on_pitch_dismissals"] += len(on_red)
                status["off_pitch_dismissals"] += len(red_players[team] - on_red)
                dismissed[team].update(red_players[team])
                if not 7 <= len(active[team]) <= 11:
                    raise ImpectParseError("on-pitch counts must remain within 7..11")
        close(ends[period])

    if expected is not None and score != expected:
        raise ImpectParseError(f"event goals {score} disagree with independent expected score {expected}")
    if [sum(s[f"{side}_goals"] for s in stints) for side in ("home", "away")] != score:
        raise ImpectParseError("stint goals do not conserve source goals")
    seconds = defaultdict(float)
    for stint in stints:
        for team, key in ((home, "home_players"), (away, "away_players")):
            for pid in stint[key]:
                seconds[(team, pid)] += stint["duration_seconds"]
    player_minutes = [{"team_id": team, "player_id": _pid(pid), "player_name": name,
                       "started": pid in starting[team], "on_pitch_seconds": seconds[team, _pid(pid)],
                       "minutes": seconds[team, _pid(pid)] / 60}
                      for team in (home, away) for pid, name in sorted(roster[team].items())]
    timing = {"provided": player_kpis is not None,
              "usage": "same-provider cross-file duration consistency audit; not independent clock truth; no auto-correction"}
    if player_kpis is not None:
        if not isinstance(player_kpis, dict) or _integer(player_kpis.get("matchId"), "KPI match id") != mid:
            raise ImpectParseError("player KPI match identity disagrees")
        durations, position_rows, observed = defaultdict(float), Counter(), set()
        for key, team in (("squadHome", home), ("squadAway", away)):
            squad = player_kpis.get(key)
            if (not isinstance(squad, dict) or _integer(squad.get("id"), "KPI squad id") != team
                    or not isinstance(squad.get("players"), list)):
                raise ImpectParseError("player KPI squad identity/players disagree")
            for row in squad["players"]:
                if not isinstance(row, dict):
                    raise ImpectParseError("player duration KPI must be an object")
                pid = _integer(row.get("id"), "KPI player id")
                pos = _position(row.get("position"))
                if pid not in roster[team] or (team, pid, pos) in observed:
                    raise ImpectParseError("unknown or duplicate player-position duration KPI")
                observed.add((team, pid, pos))
                position_rows[team, pid] += 1
                durations[team, pid] += _number(row.get("playDuration"), "playDuration")
        comparisons = []
        missing, unused_without_duration = [], []
        for row in player_minutes:
            team, pid = row["team_id"], int(row["player_id"].split(":")[-1])
            if (team, pid) not in durations:
                (missing if row["on_pitch_seconds"] > 0 else unused_without_duration).append(row["player_id"])
                continue
            provider = durations[team, pid]
            comparisons.append({"team_id": team, "player_id": row["player_id"],
                                "source_position_rows": position_rows[team, pid],
                                "provider_duration_observed": position_rows[team, pid] > 0,
                                "reconstructed_seconds": row["on_pitch_seconds"],
                                "provider_play_duration_seconds": provider,
                                "difference_seconds": row["on_pitch_seconds"] - provider})
        timing.update({"player_comparisons": comparisons, "missing_active_player_ids": missing,
                       "unused_roster_without_duration_ids": unused_without_duration,
                       "compared_players": len(comparisons), "source_position_rows": len(observed),
                       "max_absolute_difference_seconds": max((abs(x["difference_seconds"]) for x in comparisons), default=None),
                       "all_active_players_have_duration": not missing})
    return {
        "match_id": f"impect:match:{mid}", "source_match_id": mid,
        "match_date": kickoff.date().isoformat(), "scheduled_kickoff_utc": kickoff.isoformat(),
        "source_scheduled_datetime_text": metadata["scheduledDate"],
        "home_team_id": home, "away_team_id": away, "team_identity_namespace": "impect:squad",
        "stints": stints, "player_minutes": player_minutes,
        "diagnostics": {"provider": "Impect", "source_data_version": "V3", "iteration_id": iteration,
                        "n_events": len(events), "n_stints": len(stints),
                        "event_types": dict(Counter(row[3] for row in ordered)),
                        "status_events": dict(status), "period_seconds": {str(k): v for k, v in ends.items()},
                        "period_start_seconds": {str(k): v for k, v in starts.items()},
                        "period_duration_seconds": {str(k): ends[k] - starts[k] for k in ends},
                        "opening_whistle_events": opening_events,
                        "total_duration_seconds": sum(ends[k] - starts[k] for k in ends), "event_goals": score,
                        "independent_expected_score": expected,
                        "score_matches_independent_source": True if expected is not None else None,
                        "player_duration_audit": timing, "all_source_event_clock_anomalies": anomalies,
                        "source_last_calculation_datetime_text": lineups.get("lastCalculationDate"),
                        "time_semantics": "V3 period offsets 0/10000; first recorded KICK_OFF actual half start and FINAL_WHISTLE actual half end; period_seconds retains end clock; event.duration unused",
                        "usage": "retrospective goals-only actual-state paths; not pre-match features, ratings or xG",
                        "historical_availability": "unknown; scheduled kickoff and later revision times are not publication times",
                        "red_card_policy": "on-pitch cards require same-clock BANK exit; bench cards do not remove on-pitch players"},
    }
