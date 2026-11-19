"""Append per-match opening-position roles to the saved IMPECT XI dataset.

Roles belong to the current match observation, not permanently to a player.
Only startingPositions are consulted. Revised opening positions may be coded
post hoc: this is an XI-and-opening-roles-known retrospective condition, not
a reconstruction of historically public pre-match position information.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from . import impect_starting_xi as xi


ROLE_MAPPING = {
    "GOALKEEPER": "GK",
    "CENTRAL_DEFENDER": "DEF",
    "LEFT_WINGBACK_DEFENDER": "DEF",
    "RIGHT_WINGBACK_DEFENDER": "DEF",
    "DEFENSE_MIDFIELD": "MID",
    "CENTRAL_MIDFIELD": "MID",
    "ATTACKING_MIDFIELD": "MID",
    "CENTER_FORWARD": "FWD",
    "LEFT_WINGER": "FWD",
    "RIGHT_WINGER": "FWD",
}
BASE_FIELDS = {"match_id", "match_date", "home_team_id", "away_team_id",
               "home_starters", "away_starters", "full_time"}
EXPECTED_MATCHES = 306


class StartingRolesSourceError(ValueError):
    """A role observation cannot be matched to the explicit saved XI input."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _on_g(path):
    path = Path(path).resolve()
    if path.drive and path.drive.upper() != "G:":
        raise ValueError("Project, saved source inputs and new outputs must remain on G:")
    return path


def extract_roles(metadata, lineups, expected):
    """Reuse XI identity/roster validation and add current-match roles only."""
    if not isinstance(expected, dict) or set(expected) != BASE_FIELDS:
        raise StartingRolesSourceError("Expected the original seven-field XI match schema")
    base = xi.extract_match(metadata, lineups, expected["full_time"])
    for field in BASE_FIELDS - {"home_starters", "away_starters"}:
        if base[field] != expected[field]:
            raise StartingRolesSourceError(f"Saved XI {field} disagrees with current source identity")
    for side in ("home", "away"):
        players = expected[side + "_starters"]
        if not isinstance(players, list) or len(players) != 11 or len(set(players)) != 11:
            raise StartingRolesSourceError("Saved XI requires eleven distinct starter IDs per side")
        if set(base[side + "_starters"]) != set(players):
            raise StartingRolesSourceError("Saved XI starter set disagrees with current source")
    result = {field: list(value) if isinstance(value, list) else value for field, value in expected.items()}
    for source_side, side in (("squadHome", "home"), ("squadAway", "away")):
        roles = {}
        for row in lineups[source_side]["startingPositions"]:
            label = row["position"]
            if label not in ROLE_MAPPING:
                raise StartingRolesSourceError(f"Unmapped starting position label: {label!r}")
            roles[f"impect:player:{row['playerId']}"] = ROLE_MAPPING[label]
        result[side + "_roles"] = {player: roles[player] for player in expected[side + "_starters"]}
    return result


def _metadata(item):
    return {"id": item["source_match_id"], "iterationId": xi.ITERATION_ID,
            "homeSquadId": item["home_team_id"], "awaySquadId": item["away_team_id"],
            "scheduledDate": item["source_scheduled_date_text"]}


def inspect_match(item, expected, score_path):
    """Keep the original independent FT/HT join and exact receipt provenance."""
    metadata = _metadata(item)
    receipt = item.get("lineups_receipt")
    observation = {"source_match_id": metadata["id"], "included": False,
                   "target_period": xi.TARGET_FROM.isoformat() <= expected["match_date"] <= xi.TARGET_THROUGH.isoformat(),
                   "status": "source_not_included", "original_source_observation": item}
    if item.get("included") is not True:
        return observation, None
    if not isinstance(receipt, dict):
        observation.update(status="lineups_receipt_missing")
        return observation, None
    _on_g(receipt["file"])
    # Reuse the old reader/join/validation without changing it. Other file
    # kinds, substitutions and current squad attributes never enter this call.
    record = {"match_id": metadata["id"], "files": [receipt]}
    source_observation, base = xi.inspect_match(metadata, record, score_path)
    observation["rechecked_xi_source"] = source_observation
    if base is None:
        observation.update(status=source_observation["status"], error=source_observation.get("error"))
        return observation, None
    if base["full_time"] != expected["full_time"]:
        observation.update(status="saved_full_time_conflict", error="Independent full-time join changed versus saved XI input")
        return observation, None
    if item.get("scheduled_kickoff_utc") != source_observation["scheduled_kickoff_utc"]:
        observation.update(status="saved_kickoff_conflict", error="Saved source kickoff disagrees with rechecked source instant")
        return observation, None
    try:
        result = extract_roles(metadata, read(receipt["file"]), expected)
    except (StartingRolesSourceError, xi.StartingXiSourceError) as error:
        observation.update(status="roles_or_saved_xi_conflict", error=str(error))
        return observation, None
    observation.update(status="included", included=True)
    return observation, result


def run(root, source_inspection, matches_source):
    root, source_inspection, matches_source = map(_on_g, (root, source_inspection, matches_source))
    source, matches = read(source_inspection), read(matches_source)
    if not isinstance(source, dict) or not isinstance(source.get("matches"), list):
        raise ValueError("Explicit saved XI source inspection required")
    if not isinstance(matches, list) or len(matches) != EXPECTED_MATCHES:
        raise ValueError(f"Expected the existing complete {EXPECTED_MATCHES}-match XI input")
    source_rows = source["matches"]
    if len(source_rows) != EXPECTED_MATCHES:
        raise ValueError("Source inspection must retain the full original catalogue denominator")
    ids = [row.get("match_id") for row in matches if isinstance(row, dict)]
    source_ids = [f"impect:match:{row['source_match_id']}" for row in source_rows]
    if len(ids) != len(matches) or len(set(ids)) != len(ids) or len(set(source_ids)) != len(source_ids) or set(ids) != set(source_ids):
        raise ValueError("Saved XI and source inspection must have identical distinct match IDs")
    score_path = _on_g(source["source_score_file"])
    by_id = dict(zip(ids, matches))
    observations, converted = [], []
    for item in source_rows:
        expected = by_id[f"impect:match:{item['source_match_id']}"]
        observation, match = inspect_match(item, expected, score_path)
        observations.append(observation)
        if match is not None:
            converted.append(match)
    created = datetime.now(timezone.utc)
    output = root / "outputs" / created.strftime("IMPECT首发角色接通_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    role_counts = Counter(role for match in converted for side in ("home", "away") for role in match[side + "_roles"].values())
    inspection = {
        "created_at_utc": created.isoformat(), "source_inspection_file": str(source_inspection),
        "matches_source_file": str(matches_source), "source_score_file": str(score_path),
        "expected_catalogue_matches": len(matches), "included_matches": len(converted),
        "expected_target_matches": sum(row["target_period"] for row in observations),
        "included_target_matches": sum(row["target_period"] and row["included"] for row in observations),
        "statuses": dict(Counter(row["status"] for row in observations)),
        "role_mapping": ROLE_MAPPING, "role_counts": dict(role_counts), "matches": observations,
        "historical_lineup_announcement_at": None, "historical_opening_position_announcement_at": None,
        "historical_first_available_at": None,
        "interpretation": "Condition on XI plus opening roles known; opening positions may be post-hoc coded, not verified historically public pre-match role information",
        "role_semantics": "Roles are per-match observations. Preserve cross-match role changes; never whole-season modal roles or future role backfill. WINGER->FWD and WINGBACK_DEFENDER->DEF are explicit research groupings, not tactical truth.",
        "source_terms": source.get("source_terms", "IMPECT custom terms; local personal noncommercial research; no observed or derived data redistribution"),
        "not_used": ["substitutions", "events", "player_kpis", "players.currentSquadId", "whole-season modal player roles", "red-card filtering"],
        "fits": 0, "downloads": 0, "forecasts": 0,
    }
    _write(output / "matches.json", converted)
    _write(output / "inspection.json", inspection)
    return {"output_directory": str(output), "matches_file": str(output / "matches.json"),
            "included_matches": len(converted), "expected_matches": len(matches),
            "included_targets": inspection["included_target_matches"], "expected_targets": inspection["expected_target_matches"],
            "statuses": inspection["statuses"], "role_counts": dict(role_counts), "fits": 0, "downloads": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-inspection", required=True, type=Path)
    parser.add_argument("--matches-source", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.source_inspection, args.matches_source), ensure_ascii=False))


if __name__ == "__main__":
    main()
