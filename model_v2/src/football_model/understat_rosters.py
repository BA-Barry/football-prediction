"""Extract identities from saved Understat rosters, never match performance.

The roster block has no fixture date or home/away club names. Other response
blocks, such as shots, can provide identity evidence but are not read here. The
caller must supply an independently joined fixture identity; this parser cannot
verify that a response belongs to that fixture. Source positions distinguish starters
from Sub only, and are not returned as tactical or prematch role features.
"""

from datetime import date
import re


STARTING_POSITIONS = frozenset({
    "GK", "DR", "DC", "DL", "DMR", "DMC", "DML", "MR", "MC", "ML",
    "AMR", "AMC", "AML", "FWR", "FW", "FWL",
})


class UnderstatRosterSourceError(ValueError):
    """A saved roster cannot supply an unambiguous starting XI."""


def _identifier(value, label):
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return value
    raise UnderstatRosterSourceError(f"{label} requires a positive decimal ID")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise UnderstatRosterSourceError(f"{label} requires a nonempty string")
    return value


def extract_starting_xi(payload, identity):
    """Return source-namespaced XI identities and explicit availability limits.

    ``identity`` supplies match_id, date, home, away and understat_id from an
    external exact fixture join. Neither observed minutes nor positionOrder,
    substitutions, player performance or whole-season metadata are inspected.
    Unknown source position codes fail explicitly instead of guessing starters.
    """
    if not isinstance(identity, dict):
        raise UnderstatRosterSourceError("External fixture identity must be an object")
    match_id = _text(identity.get("match_id"), "match_id")
    home, away = (_text(identity.get(k), k) for k in ("home", "away"))
    if home == away:
        raise UnderstatRosterSourceError("Distinct external home and away identities required")
    played = identity.get("date")
    if not isinstance(played, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", played):
        raise UnderstatRosterSourceError("External date requires YYYY-MM-DD")
    try:
        date.fromisoformat(played)
    except ValueError as error:
        raise UnderstatRosterSourceError("Invalid external fixture date") from error
    source_match = _identifier(identity.get("understat_id"), "understat_id")
    if not isinstance(payload, dict) or not isinstance(payload.get("rosters"), dict):
        raise UnderstatRosterSourceError("Payload requires a rosters object")
    rosters = payload["rosters"]
    if set(rosters) != {"h", "a"}:
        raise UnderstatRosterSourceError("Rosters require exactly h and a sides")

    players, starters, teams = set(), {}, {}
    for side in ("h", "a"):
        roster = rosters[side]
        if not isinstance(roster, dict) or not roster:
            raise UnderstatRosterSourceError("Each side requires a nonempty roster object")
        current_starters, current_teams = set(), set()
        for key, row in roster.items():
            if not isinstance(row, dict):
                raise UnderstatRosterSourceError("Roster row must be an object")
            row_id = _identifier(row.get("id"), "roster row id")
            if not isinstance(key, str) or key != row_id:
                raise UnderstatRosterSourceError("Roster dictionary key must equal its row id")
            if row.get("h_a") != side:
                raise UnderstatRosterSourceError("Roster h_a disagrees with its side")
            current_teams.add(_identifier(row.get("team_id"), "team_id"))
            player = _identifier(row.get("player_id"), "player_id")
            if player in players:
                raise UnderstatRosterSourceError("Duplicate player_id in this match, including Subs")
            players.add(player)
            position = row.get("position")
            if not isinstance(position, str) or position not in STARTING_POSITIONS | {"Sub"}:
                raise UnderstatRosterSourceError("Unsupported source position; starters are not inferred")
            if position != "Sub":
                current_starters.add(player)
        if len(current_teams) != 1:
            raise UnderstatRosterSourceError("team_id must be consistent within each side")
        teams[side] = next(iter(current_teams))
        if len(current_starters) != 11:
            raise UnderstatRosterSourceError("Exactly 11 non-Sub starters required per side")
        starters[side] = [f"understat:player:{p}" for p in sorted(current_starters, key=int)]
    if teams["h"] == teams["a"]:
        raise UnderstatRosterSourceError("Distinct home and away source team_id values required")
    return {
        "match_id": match_id, "match_date": played, "home": home, "away": away,
        "understat_id": source_match,
        "home_starters": starters["h"], "away_starters": starters["a"],
        "source_url": f"https://understat.com/match/{source_match}",
        "identity_basis": "caller_supplied_exact_fixture_join_not_verified_by_roster_payload",
        "historical_lineup_announcement_at": None, "historical_first_available_at": None,
        "availability": "retrospective_starting_xi_known; original_publication_time_unknown",
    }
