"""Reconstruct retrospective on-pitch intervals from StatsBomb open event data.

This is an event-data adapter, not a pre-match feature generator. Source clocks
are period-relative and have no asserted timezone; no historical publication,
lineup-announcement or conditional team-intensity timestamps are invented.
Intervals include each half's recorded stoppage time, but not half-time. Goals
at an interval's right boundary belong to the lineups immediately before that
goal. Only ordinary two-half matches are supported.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
import re


class StatsBombParseError(ValueError):
    """Source states cannot be reconciled without inventing an on-pitch path."""


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise StatsBombParseError(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(text: str) -> float:
    if not isinstance(text, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", text):
        raise StatsBombParseError("timestamp must be HH:MM:SS[.fraction] period clock")
    hours, minutes, seconds = text.split(":")
    if int(minutes) >= 60 or not 0 <= float(seconds) < 60:
        raise StatsBombParseError("invalid period clock")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _player_id(value: int) -> str:
    return f"statsbomb:player:{value}"


def parse_match(metadata: dict, events: list[dict], lineups: list[dict]) -> dict:
    """Return validated retrospective stints and actual player minutes.

    ``home_players``/``away_players`` use stable namespaced player IDs; team IDs
    remain StatsBomb integers and are not silently equated to other providers'
    IDs. The ``player_minutes`` list includes unused roster players at zero.
    Own goals use ``Own Goal Against`` as the single scoring perspective; a
    supplied ``Own Goal For`` must link to that perspective. Unknown states,
    incomplete scores, extra periods and ambiguous clocks raise, not impute.
    """
    match_id = _integer(metadata.get("match_id"), "match_id", minimum=1)
    try:
        home = _integer(metadata["home_team"]["home_team_id"], "home_team_id", minimum=1)
        away = _integer(metadata["away_team"]["away_team_id"], "away_team_id", minimum=1)
    except (KeyError, TypeError) as exc:
        raise StatsBombParseError("missing match team identity") from exc
    if home == away:
        raise StatsBombParseError("match requires two distinct teams")
    expected = [_integer(metadata.get("home_score"), "home_score"),
                _integer(metadata.get("away_score"), "away_score")]
    if not isinstance(events, list) or not events or not isinstance(lineups, list):
        raise StatsBombParseError("nonempty events and two roster records required")

    roster: dict[int, dict[int, str]] = {}
    all_players: set[int] = set()
    for entry in lineups:
        team = _integer(entry.get("team_id"), "roster team_id", minimum=1)
        if team not in {home, away} or team in roster:
            raise StatsBombParseError("unexpected or duplicate roster team")
        roster[team] = {}
        for player in entry.get("lineup", []):
            pid = _integer(player.get("player_id"), "roster player_id", minimum=1)
            if pid in all_players:
                raise StatsBombParseError("duplicate roster player or player on both teams")
            all_players.add(pid)
            roster[team][pid] = player.get("player_name") or str(pid)
    if set(roster) != {home, away}:
        raise StatsBombParseError("both team rosters required")

    ids, indices = set(), set()
    ordered = []
    for event in events:
        eid = event.get("id")
        index = _integer(event.get("index"), "event index", minimum=1)
        if not isinstance(eid, str) or not eid or eid in ids or index in indices:
            raise StatsBombParseError("missing or duplicate event ID/index")
        ids.add(eid)
        indices.add(index)
        period = _integer(event.get("period"), "period", minimum=1)
        if period not in {1, 2}:
            raise StatsBombParseError("extra periods are not supported by this regulation adapter")
        clock = _seconds(event.get("timestamp"))
        kind = event.get("type", {}).get("name")
        if not isinstance(kind, str) or not kind:
            raise StatsBombParseError("event type is required")
        ordered.append((index, period, clock, kind, event))
    ordered.sort(key=lambda item: item[0])
    # Old provider exports occasionally reset a trailing Ball Receipt clock at
    # half-time, or place it beyond Half End. Those untouched records are not
    # targets or state changes for this adapter. Preserve their anomalies while
    # retaining strict chronology for every event that can affect a stint.
    def affects_stint(row):
        kind, event = row[3], row[4]
        if kind in {"Foul Committed", "Bad Behaviour"}:
            field = "foul_committed" if kind == "Foul Committed" else "bad_behaviour"
            return event.get(field, {}).get("card", {}).get("name") in {"Red Card", "Second Yellow"}
        return kind in {"Starting XI", "Half Start", "Half End", "Shot", "Own Goal Against",
                        "Own Goal For", "Substitution", "Player Off", "Player On"}

    clock_anomalies = [{"previous_event_id": previous[4]["id"],
                        "event_id": current[4]["id"], "event_type": current[3],
                        "previous_event_type": previous[3], "period": current[1],
                        "source_timestamp": current[4]["timestamp"],
                        "issue": "all_event_index_clock_inversion"}
                       for previous, current in zip(ordered, ordered[1:])
                       if current[1] < previous[1]
                       or (current[1] == previous[1] and current[2] < previous[2])]
    model_ordered = [row for row in ordered if affects_stint(row)]
    for previous, current in zip(model_ordered, model_ordered[1:]):
        if current[1] < previous[1] or (current[1] == previous[1] and current[2] < previous[2]):
            raise StatsBombParseError("model-relevant event index and period clock ordering disagree")

    own_against = [row[4] for row in ordered if row[3] == "Own Goal Against"]
    for _, period, clock, kind, event in ordered:
        if kind != "Own Goal For":
            continue
        peers = [other for other in own_against
                 if other["id"] in event.get("related_events", [])
                 or event["id"] in other.get("related_events", [])]
        if len(peers) != 1:
            raise StatsBombParseError("Own Goal For requires one linked Own Goal Against")
        other = peers[0]
        if (other["period"] != period or _seconds(other["timestamp"]) != clock
                or other.get("team", {}).get("id") == event.get("team", {}).get("id")):
            raise StatsBombParseError("own-goal perspectives disagree")

    active: dict[int, set[int]] = {}
    starting: dict[int, set[int]] = {}
    for _, period, clock, kind, event in ordered:
        if kind != "Starting XI":
            continue
        team = event.get("team", {}).get("id")
        if team not in roster or team in active or period != 1 or clock != 0:
            raise StatsBombParseError("Starting XI must occur once per team at first-half zero")
        players = [_integer(p.get("player", {}).get("id"), "starting player", minimum=1)
                   for p in event.get("tactics", {}).get("lineup", [])]
        if len(players) != 11 or len(set(players)) != 11 or set(players) - roster[team].keys():
            raise StatsBombParseError("Starting XI must identify 11 distinct roster players")
        active[team] = set(players)
        starting[team] = set(players)
    if set(active) != {home, away}:
        raise StatsBombParseError("both Starting XI events required")

    score = [0, 0]
    temporary = {home: set(), away: set()}
    removed = {home: set(), away: set()}
    dismissed = {home: set(), away: set()}
    counts = Counter(row[3] for row in ordered)
    status_counts = Counter()
    stints = []
    period_lengths = {}

    def actor(event: dict) -> tuple[int, int]:
        team = _integer(event.get("team", {}).get("id"), "actor team_id", minimum=1)
        pid = _integer(event.get("player", {}).get("id"), "actor player_id", minimum=1)
        if team not in roster or pid not in roster[team]:
            raise StatsBombParseError(f"event {event['id']}: actor not in the team's roster")
        return team, pid

    for period in (1, 2):
        rows = [row for row in model_ordered if row[1] == period]
        ends = {row[2] for row in rows if row[3] == "Half End"}
        starts = [row for row in rows if row[3] == "Half Start"]
        if len(ends) != 1 or not starts or any(row[2] != 0 for row in starts):
            raise StatsBombParseError("each half needs consistent Half End and zero Half Start")
        end = next(iter(ends))
        if end <= 0 or any(row[2] > end for row in rows):
            raise StatsBombParseError("events exceed or lack the recorded half duration")
        clock_anomalies.extend({"event_id": row[4]["id"], "event_type": row[3],
                                "period": period, "source_timestamp": row[4]["timestamp"],
                                "issue": "nonmodel_event_beyond_half_end"}
                               for row in ordered if row[1] == period and row[2] > end)
        period_lengths[str(period)] = end
        start = 0.0
        boundary_index = rows[0][0]
        start_score = tuple(score)
        goals = [0, 0]
        xg = [0.0, 0.0]
        goal_ids: list[str] = []
        shot_ids: list[str] = []

        def close(clock: float, index: int) -> None:
            nonlocal start, boundary_index, start_score, goals, xg, goal_ids, shot_ids
            elapsed = clock - start
            if elapsed < 0:
                raise StatsBombParseError("state transition precedes the current interval")
            if elapsed == 0 and (any(goals) or any(xg)):
                raise StatsBombParseError("scoring at zero exposure cannot be allocated without inference")
            if elapsed > 0:
                if not all(7 <= len(active[team]) <= 11 for team in (home, away)):
                    raise StatsBombParseError("unexplained on-pitch player count outside 7..11")
                stints.append({
                    "stint_id": f"statsbomb:match:{match_id}:period:{period}:stint:{len(stints)+1}",
                    "period": period, "start_second": start, "end_second": clock,
                    "duration_seconds": elapsed,
                    "home_players": [_player_id(pid) for pid in sorted(active[home])],
                    "away_players": [_player_id(pid) for pid in sorted(active[away])],
                    "start_home_goals": start_score[0], "start_away_goals": start_score[1],
                    "home_score_difference": start_score[0] - start_score[1],
                    "home_man_difference": len(active[home]) - len(active[away]),
                    "home_goals": goals[0], "away_goals": goals[1],
                    "home_xg": xg[0], "away_xg": xg[1],
                    "goal_event_ids": list(goal_ids), "shot_event_ids": list(shot_ids),
                    "start_boundary_event_index": boundary_index,
                    "end_boundary_event_index": index,
                })
            start, boundary_index, start_score = clock, index, tuple(score)
            goals, xg, goal_ids, shot_ids = [0, 0], [0.0, 0.0], [], []

        for index, _, clock, kind, event in rows:
            if kind == "Shot":
                team, pid = actor(event)
                if pid not in active[team]:
                    raise StatsBombParseError("shot taken by a player recorded off the pitch")
                value = event.get("shot", {}).get("statsbomb_xg")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                    raise StatsBombParseError("every shot needs finite statsbomb_xg in [0, 1]")
                side = 0 if team == home else 1
                xg[side] += float(value)
                shot_ids.append(event["id"])
                if event.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                    goals[side] += 1
                    goal_ids.append(event["id"])
                    close(clock, index)
                    score[side] += 1
                    start_score = tuple(score)
            elif kind == "Own Goal Against":
                team, pid = actor(event)
                if pid not in active[team]:
                    raise StatsBombParseError("own goal by a player recorded off the pitch")
                side = 1 if team == home else 0
                goals[side] += 1
                goal_ids.append(event["id"])
                close(clock, index)
                score[side] += 1
                start_score = tuple(score)
            elif kind == "Substitution":
                team, outgoing = actor(event)
                incoming = event.get("substitution", {}).get("replacement", {}).get("id")
                if (outgoing not in active[team] | temporary[team]
                        or incoming not in roster[team] or incoming in active[team]
                        or incoming in temporary[team] | removed[team] | dismissed[team]):
                    raise StatsBombParseError("illegal or unreconciled substitution")
                close(clock, index)
                active[team].discard(outgoing)
                temporary[team].discard(outgoing)
                removed[team].add(outgoing)
                active[team].add(incoming)
            elif kind == "Player Off":
                team, pid = actor(event)
                if pid in dismissed[team]:
                    status_counts["off_after_dismissal"] += 1
                    continue
                if pid not in active[team]:
                    raise StatsBombParseError("Player Off actor is not currently on the pitch")
                close(clock, index)
                active[team].remove(pid)
                if event.get("player_off", {}).get("permanent", False):
                    removed[team].add(pid)
                else:
                    temporary[team].add(pid)
            elif kind == "Player On":
                team, pid = actor(event)
                if pid not in temporary[team] or pid in dismissed[team] | removed[team]:
                    raise StatsBombParseError("Player On requires a matching temporary absence")
                close(clock, index)
                temporary[team].remove(pid)
                active[team].add(pid)
            elif kind in {"Foul Committed", "Bad Behaviour"}:
                field = "foul_committed" if kind == "Foul Committed" else "bad_behaviour"
                card = event.get(field, {}).get("card", {}).get("name")
                if card not in {"Red Card", "Second Yellow"}:
                    continue
                team, pid = actor(event)
                if pid in dismissed[team]:
                    status_counts["duplicate_dismissal_events"] += 1
                    continue
                if pid in active[team] | temporary[team]:
                    close(clock, index)
                    active[team].discard(pid)
                    temporary[team].discard(pid)
                    status_counts["on_pitch_dismissals"] += 1
                else:
                    status_counts["off_pitch_dismissals"] += 1
                dismissed[team].add(pid)
        close(end, rows[-1][0])

    if score != expected:
        raise StatsBombParseError(f"event goals {score} disagree with metadata {expected}")
    minutes = defaultdict(float)
    for stint in stints:
        for team, column in ((home, "home_players"), (away, "away_players")):
            for pid in stint[column]:
                minutes[(team, pid)] += stint["duration_seconds"]
    player_minutes = [
        {"team_id": team, "player_id": _player_id(pid), "player_name": name,
         "started": pid in starting[team], "on_pitch_seconds": minutes[(team, _player_id(pid))],
         "minutes": minutes[(team, _player_id(pid))] / 60.0}
        for team in (home, away) for pid, name in sorted(roster[team].items())
    ]
    return {
        "match_id": f"statsbomb:match:{match_id}", "source_match_id": match_id,
        "match_date": metadata.get("match_date"), "source_kickoff_clock_text": metadata.get("kick_off"),
        "home_team_id": home, "away_team_id": away,
        "stints": stints, "player_minutes": player_minutes,
        "diagnostics": {
            "n_events": len(events), "n_stints": len(stints), "event_types": dict(counts),
            "status_events": dict(status_counts), "period_seconds": period_lengths,
            "total_duration_seconds": sum(period_lengths.values()),
            "event_goals": score, "score_matches_metadata": True,
            "home_xg": sum(row["home_xg"] for row in stints),
            "away_xg": sum(row["away_xg"] for row in stints),
            "own_goal_scoring_perspective": "Own Goal Against only; linked Own Goal For not counted again",
            "time_semantics": "retrospective period clocks; timezone and historical availability unknown",
            "usage": "actual on-pitch path, not pre-match projected minutes or a fitted player rating",
            "all_source_event_clock_anomalies": clock_anomalies,
            "clock_policy": "source index preserved; strict chronology for stint-affecting events; other anomalies recorded, not repaired",
        },
    }
