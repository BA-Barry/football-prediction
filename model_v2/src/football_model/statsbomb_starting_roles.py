"""Complete-catalogue StatsBomb opening-XI/role adapter; no fitting or downloads.

This is a retrospective XI-and-opening-role-known condition, not a claim that
positions were publicly available before kickoff. Match targets are independently
joined to Football-Data by explicit team aliases, calendar date and home/away.
Only Starting XI tactics and zero-time first-period lineup positions form the
features; later substitutions, actual minutes, cards and final roles do not.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

from .full_season_experiment import TEAM_NAMES
from .statsbomb_data import season_profile


EXPECTED_MATCHES = 380
TEAM_ALIASES = {
    "Arsenal": "Arsenal", "Leicester City": "Leicester", "Watford": "Watford",
    "Liverpool": "Liverpool", "Southampton": "Southampton", "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom", "AFC Bournemouth": "Bournemouth",
    "Everton": "Everton", "Stoke City": "Stoke", "Crystal Palace": "Crystal Palace",
    "Chelsea": "Chelsea", "Manchester City": "Man City", "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham", "Manchester United": "Man United",
    "West Ham United": "West Ham", "Sunderland": "Sunderland", "Norwich City": "Norwich",
    "Aston Villa": "Aston Villa",
}
# Exact IDs AND names are checked, rather than inferring roles from substrings.
POSITION_MAPPING = {
    1: ("Goalkeeper", "GK"), 2: ("Right Back", "DEF"),
    3: ("Right Center Back", "DEF"), 4: ("Center Back", "DEF"),
    5: ("Left Center Back", "DEF"), 6: ("Left Back", "DEF"),
    7: ("Right Wing Back", "DEF"), 8: ("Left Wing Back", "DEF"),
    9: ("Right Defensive Midfield", "MID"), 10: ("Center Defensive Midfield", "MID"),
    11: ("Left Defensive Midfield", "MID"), 12: ("Right Midfield", "MID"),
    13: ("Right Center Midfield", "MID"), 14: ("Center Midfield", "MID"),
    15: ("Left Center Midfield", "MID"), 16: ("Left Midfield", "MID"),
    17: ("Right Wing", "FWD"), 18: ("Right Attacking Midfield", "MID"),
    19: ("Center Attacking Midfield", "MID"), 20: ("Left Attacking Midfield", "MID"),
    21: ("Left Wing", "FWD"), 22: ("Right Center Forward", "FWD"),
    23: ("Center Forward", "FWD"), 24: ("Left Center Forward", "FWD"),
    25: ("Secondary Striker", "FWD"),
}
BASE_FIELDS = {"match_id", "match_date", "home_team_id", "away_team_id",
               "home_starters", "away_starters", "full_time"}


class StartingRolesSourceError(ValueError):
    """The supplied observations cannot form an unambiguous opening XI."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _resolved(path):
    return Path(path).expanduser().resolve()


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise StartingRolesSourceError(f"{label} must be an integer >= {minimum}")
    return value


def _identity(metadata):
    mid = _integer(metadata.get("match_id"), "match_id")
    when = metadata.get("match_date")
    if not isinstance(when, str) or date.fromisoformat(when).isoformat() != when:
        raise StartingRolesSourceError("match_date must be an ISO calendar date")
    teams = [_integer(metadata[side + "_team"][side + "_team_id"], side + " team")
             for side in ("home", "away")]
    if teams[0] == teams[1]:
        raise StartingRolesSourceError("Two distinct match teams required")
    return mid, when, teams


def role(position_id, label):
    mapping = POSITION_MAPPING.get(position_id) if type(position_id) is int else None
    if mapping is None or mapping[0] != label:
        raise StartingRolesSourceError(f"Unknown or conflicting opening position: {position_id!r}, {label!r}")
    return mapping[1]


def extract_roles(metadata, events, lineups, full_time):
    """Return only the nine model fields; source roster names stay in provenance."""
    mid, when, teams = _identity(metadata)
    if (not isinstance(full_time, (list, tuple)) or len(full_time) != 2
            or any(type(x) is not int or x < 0 for x in full_time)):
        raise StartingRolesSourceError("full_time needs two nonnegative integer goals")
    if not isinstance(events, list) or not isinstance(lineups, list):
        raise StartingRolesSourceError("Events and lineup lists required")
    openings = {}
    for event in events:
        if event.get("type", {}).get("name") != "Starting XI":
            continue
        team = event.get("team", {}).get("id")
        if (type(team) is not int or team not in teams or team in openings
                or type(event.get("period")) is not int or event.get("period") != 1
                or event.get("timestamp") not in {"00:00:00", "00:00:00.000"}
                or type(event.get("minute")) is not int or event.get("minute") != 0
                or type(event.get("second")) is not int or event.get("second") != 0):
            raise StartingRolesSourceError("Starting XI must occur once per team at first-period zero")
        rows = event.get("tactics", {}).get("lineup")
        if not isinstance(rows, list) or len(rows) != 11:
            raise StartingRolesSourceError("Starting XI requires eleven players")
        initial = {}
        for row in rows:
            pid = _integer(row.get("player", {}).get("id"), "starting player")
            if pid in initial:
                raise StartingRolesSourceError("Duplicate starting player")
            position = row.get("position", {})
            initial[pid] = (position.get("id"), position.get("name"))
            role(*initial[pid])
        openings[team] = initial
    if set(openings) != set(teams):
        raise StartingRolesSourceError("Both teams' Starting XI events required")
    if set(openings[teams[0]]) & set(openings[teams[1]]):
        raise StartingRolesSourceError("A starting player cannot be on both teams")
    rosters, all_players = {}, set()
    for entry in lineups:
        team = entry.get("team_id")
        if type(team) is not int or team not in teams or team in rosters:
            raise StartingRolesSourceError("Unexpected or duplicate roster team")
        rows = entry.get("lineup")
        if not isinstance(rows, list):
            raise StartingRolesSourceError("Roster player list required")
        initial = {}
        for row in rows:
            pid = _integer(row.get("player_id"), "roster player")
            if pid in all_players:
                raise StartingRolesSourceError("Duplicate roster player or identity on both teams")
            all_players.add(pid)
            positions = row.get("positions")
            if not isinstance(positions, list):
                raise StartingRolesSourceError("Roster position list required")
            opening = [p for p in positions if p.get("from_period") == 1 and p.get("from") == "00:00"]
            if len(opening) > 1:
                raise StartingRolesSourceError("Ambiguous zero-time opening positions")
            if opening:
                p = opening[0]
                if p.get("start_reason") != "Starting XI":
                    raise StartingRolesSourceError("Zero-time opening position lacks Starting XI reason")
                initial[pid] = (p.get("position_id"), p.get("position"))
                role(*initial[pid])
        rosters[team] = initial
    if set(rosters) != set(teams) or any(rosters[t] != openings[t] for t in teams):
        raise StartingRolesSourceError("Starting XI and independent zero-time lineup positions disagree")
    result = {"match_id": f"statsbomb:match:{mid}", "match_date": when,
              "home_team_id": teams[0], "away_team_id": teams[1], "full_time": list(full_time)}
    for side, team in zip(("home", "away"), teams):
        result[side + "_starters"] = [f"statsbomb:player:{pid}" for pid in openings[team]]
        result[side + "_roles"] = {f"statsbomb:player:{pid}": role(*position)
                                  for pid, position in openings[team].items()}
    return result


def load_scores(path):
    """Keep duplicate exact keys for explicit rejection, never silently pick one."""
    indexed = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if not row.get("Date"):
                continue
            when = datetime.strptime(row["Date"], "%d/%m/%Y").date().isoformat()
            indexed[(when, row["HomeTeam"], row["AwayTeam"])].append((line, row))
    return indexed


def independent_score(metadata, scores):
    _, when, teams = _identity(metadata)
    labels = [metadata[side + "_team"][side + "_team_name"] for side in ("home", "away")]
    if any(label not in TEAM_ALIASES for label in labels):
        raise StartingRolesSourceError(f"No explicit Football-Data team alias: {labels!r}")
    if any(TEAM_NAMES.get(team) != TEAM_ALIASES[label] for team, label in zip(teams, labels)):
        raise StartingRolesSourceError("Source team ID/name conflicts with existing explicit team-ID mapping")
    key = (when, *(TEAM_ALIASES[label] for label in labels))
    rows = scores.get(key, [])
    if len(rows) != 1:
        raise StartingRolesSourceError(f"Exact date/home/away join requires one score row, found {len(rows)}: {key!r}")
    line, row = rows[0]
    if row["Div"] != "E0":
        raise StartingRolesSourceError("Independent score row must belong to E0")
    def goals(fields):
        values = [row[field] for field in fields]
        if any(not isinstance(x, str) or not x.isdigit() for x in values):
            raise StartingRolesSourceError("Independent goal counts must be nonnegative integers")
        return [int(x) for x in values]
    ft, ht = goals(("FTHG", "FTAG")), goals(("HTHG", "HTAG"))
    if any(h > f for h, f in zip(ht, ft)):
        raise StartingRolesSourceError("Independent half-time goals exceed full-time goals")
    if ft != [_integer(metadata.get(side + "_score"), side + " score", 0) for side in ("home", "away")]:
        raise StartingRolesSourceError("Independent full-time score disagrees with StatsBomb metadata")
    return {"full_time": ft, "half_time": ht, "csv_line": line, "exact_join_key": list(key)}


def event_scores(metadata, events):
    """Score-only audit, independent of player minutes and opening features."""
    _, _, teams = _identity(metadata)
    ft, ht, goals, own_against, own_for = [0, 0], [0, 0], [], {}, []
    for event in events:
        kind = event.get("type", {}).get("name")
        is_goal = kind == "Shot" and event.get("shot", {}).get("outcome", {}).get("name") == "Goal"
        if kind == "Own Goal For":
            own_for.append(event)
        if not is_goal and kind != "Own Goal Against":
            continue
        eid = event.get("id")
        if not isinstance(eid, str) or not eid or eid in goals:
            raise StartingRolesSourceError("Goal event identities must be distinct")
        goals.append(eid)
        team, period = event.get("team", {}).get("id"), event.get("period")
        if type(team) is not int or team not in teams or type(period) is not int or period not in (1, 2):
            raise StartingRolesSourceError("Goal must have a match team and regulation period")
        scoring = 1 - teams.index(team) if kind == "Own Goal Against" else teams.index(team)
        ft[scoring] += 1
        if period == 1:
            ht[scoring] += 1
        if kind == "Own Goal Against":
            own_against[eid] = event
    for event in own_for:
        peers = [other for eid, other in own_against.items()
                 if eid in event.get("related_events", []) or event.get("id") in other.get("related_events", [])]
        if (len(peers) != 1 or peers[0].get("period") != event.get("period")
                or peers[0].get("timestamp") != event.get("timestamp")
                or peers[0].get("team", {}).get("id") == event.get("team", {}).get("id")):
            raise StartingRolesSourceError("Own Goal For requires one consistent linked Own Goal Against")
    return {"full_time": ft, "half_time": ht, "goal_event_ids": goals}


def opening_position_anomalies(lineups):
    """Retain every zero-time ambiguity, not just the first raised error."""
    anomalies = []
    for team in lineups:
        for player in team.get("lineup", []):
            positions = player.get("positions")
            if not isinstance(positions, list):
                continue
            opening = [p for p in positions if p.get("from_period") == 1 and p.get("from") == "00:00"]
            issues = []
            if len(opening) > 1:
                issues.append("multiple_zero_time_opening_positions")
            if any(p.get("start_reason") != "Starting XI" for p in opening):
                issues.append("zero_time_position_not_labelled_starting_xi")
            if issues:
                anomalies.append({"team_id": team.get("team_id"), "player_id": player.get("player_id"),
                                  "issues": issues, "zero_time_positions": [
                                      {k: p.get(k) for k in ("position_id", "position", "from", "from_period", "start_reason")}
                                      for p in opening]})
    return anomalies


def inspect_match(metadata, source, scores):
    mid = metadata.get("match_id")
    observation = {"source_match_id": mid, "match_date": metadata.get("match_date"),
                   "included": False, "status": "source_conflict", "errors": [],
                   "source_metadata": metadata, "source_files": {}}
    loaded = {}
    for kind in ("events", "lineups"):
        path = source / f"{kind}_{mid}.json"
        observation["source_files"][kind] = str(path)
        try:
            loaded[kind] = read(path)
        except (OSError, ValueError) as error:
            observation["errors"].append({"stage": kind, "error": str(error)})
    score, target = None, None
    try:
        score = independent_score(metadata, scores)
        observation["independent_score"] = score
    except (StartingRolesSourceError, KeyError, TypeError, ValueError) as error:
        observation["errors"].append({"stage": "score_join", "error": str(error)})
    if "events" in loaded:
        try:
            event_target = event_scores(metadata, loaded["events"])
            observation["event_score_reconciliation"] = event_target
            if score and any(score[k] != event_target[k] for k in ("full_time", "half_time")):
                raise StartingRolesSourceError("Independent FT/HT disagrees with event goal counts")
        except (StartingRolesSourceError, KeyError, TypeError, ValueError) as error:
            observation["errors"].append({"stage": "score_events", "error": str(error)})
    # Still inspect opening observations if another independent check failed.
    if set(loaded) == {"events", "lineups"}:
        try:
            observation["opening_position_anomalies"] = opening_position_anomalies(loaded["lineups"])
            raw_target = [metadata.get("home_score"), metadata.get("away_score")]
            target = extract_roles(metadata, loaded["events"], loaded["lineups"],
                                   score["full_time"] if score else raw_target)
        except (StartingRolesSourceError, KeyError, TypeError, ValueError) as error:
            observation["errors"].append({"stage": "opening_xi_roles", "error": str(error)})
    if not observation["errors"] and target is not None:
        observation.update(included=True, status="included")
        return observation, target
    return observation, None


def run(root, source, catalogue, score_file):
    root, source, catalogue, score_file = map(
        _resolved, (root, source, catalogue, score_file)
    )
    matches, scores = read(catalogue), load_scores(score_file)
    if not isinstance(matches, list) or not matches:
        raise ValueError("Explicit nonempty match catalogue required")
    counts = Counter(row.get("match_id") for row in matches)
    observations, converted = [], []
    for metadata in sorted(matches, key=lambda row: (str(row.get("match_date")), str(row.get("match_id")))):
        observation, target = inspect_match(metadata, source, scores)
        if counts[metadata.get("match_id")] > 1:
            observation["errors"].append({"stage": "catalogue", "error": "Duplicate source match identity"})
            observation.update(included=False, status="source_conflict")
            target = None
        observations.append(observation)
        if target is not None:
            converted.append(target)
    created = datetime.now(timezone.utc)
    output = root / "outputs" / created.strftime("StatsBomb首发角色接通_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    receipts = sorted(str(p) for p in source.glob("下载来源_*.json"))
    role_counts = Counter(r for match in converted for side in ("home", "away") for r in match[side + "_roles"].values())
    inspection = {
        "created_at_utc": created.isoformat(), "source_directory": str(source),
        "catalogue_file": str(catalogue), "source_score_file": str(score_file),
        "source_receipt_files": receipts, "source_readme": str(source / "SOURCE_README.md"),
        "source_license": str(source / "LICENSE.pdf"),
        "expected_catalogue_matches": len(matches), "season_expected_matches": EXPECTED_MATCHES,
        "schedule_profile": season_profile(matches), "included_matches": len(converted),
        "statuses": dict(Counter(row["status"] for row in observations)),
        "team_aliases": TEAM_ALIASES, "position_mapping": POSITION_MAPPING,
        "role_counts": dict(role_counts), "matches": observations,
        "historical_lineup_announcement_at": None, "historical_opening_position_announcement_at": None,
        "historical_first_available_at": None, "kickoff_timezone": None,
        "interpretation": "Retrospective XI-and-opening-roles-known conditional research. Historical publication times are unknown; not blind, causal, or evidence of pre-kickoff role availability.",
        "role_semantics": "Per-match roles only, no whole-season modal roles. Wing backs->DEF; wide midfielders and attacking midfielders->MID; wings and forwards->FWD. This explicit grouping matches the prior coarse research semantics, not tactical truth.",
        "source_terms": "StatsBomb / Hudl Open Data and Football-Data original terms remain applicable. Personal noncommercial local research; no raw or derived data redistribution. StatsBomb attribution and logo required for published insights as specified in retained SOURCE_README.md and LICENSE.pdf.",
        "source_url": "https://github.com/hudl/open-data", "not_used": [
            "old stint subsets", "actual player minutes", "substitutions", "red-card filtering",
            "final positions", "whole-season modal player roles", "future role backfill"],
        "fits": 0, "downloads": 0, "forecasts": 0,
    }
    _write(output / "matches.json", converted)
    _write(output / "inspection.json", inspection)
    return {"output_directory": str(output), "matches_file": str(output / "matches.json"),
            "inspection_file": str(output / "inspection.json"), "expected_matches": len(matches),
            "included_matches": len(converted), "statuses": inspection["statuses"],
            "role_counts": dict(role_counts), "fits": 0, "downloads": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "catalogue", "score-file"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.source, args.catalogue, args.score_file), ensure_ascii=False))


if __name__ == "__main__":
    main()
