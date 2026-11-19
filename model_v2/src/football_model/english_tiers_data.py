"""Narrow CC0 English-tier intake, not a model or an historical as-of feed.

Source text is retained verbatim. Regular league and play-off observations stay
separate; extra-time/penalty scores are never promoted to regulation-time goals.
Unrecognized lines and team labels are reported rather than silently dropped.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen


BASE = "https://raw.githubusercontent.com/openfootball/england/master"
SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
DIVISIONS = {"en.1": "1-premierleague.txt", "en.2": "2-championship.txt"}
MONTHS = {name: n for n, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Explicit aliases, not suffix stripping/fuzzy matching. Other labels remain
# visible with an unresolved canonical ID until deliberately added here.
TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa", "Barnsley FC": "Barnsley",
    "Birmingham City FC": "Birmingham", "Blackburn Rovers FC": "Blackburn",
    "Blackpool FC": "Blackpool", "Bolton Wanderers FC": "Bolton",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton",
    "Bristol City FC": "Bristol City", "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff", "Charlton Athletic FC": "Charlton",
    "Chelsea FC": "Chelsea", "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace", "Derby County FC": "Derby",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Hull City AFC": "Hull",
    "Huddersfield Town AFC": "Huddersfield", "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds", "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool", "Luton Town FC": "Luton",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Middlesbrough FC": "Middlesbrough", "Millwall FC": "Millwall",
    "Newcastle United FC": "Newcastle", "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest", "Oxford United FC": "Oxford",
    "Peterborough United FC": "Peterborough", "Plymouth Argyle FC": "Plymouth",
    "Portsmouth FC": "Portsmouth", "Preston North End FC": "Preston",
    "Queens Park Rangers FC": "QPR", "Reading FC": "Reading",
    "Rotherham United FC": "Rotherham", "Sheffield United FC": "Sheffield United",
    "Sheffield Wednesday FC": "Sheffield Wednesday", "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke", "Sunderland AFC": "Sunderland",
    "Swansea City AFC": "Swansea", "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford", "West Bromwich Albion FC": "West Brom",
    "West Ham United FC": "West Ham", "Wigan Athletic FC": "Wigan",
    "Wolverhampton Wanderers FC": "Wolves", "Wrexham AFC": "Wrexham",
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton & Hove Albion": "Brighton", "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton", "Fulham": "Fulham",
    "Leeds United": "Leeds", "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def _team(label):
    canonical = TEAM_ALIASES.get(label)
    return {"source_label": label, "canonical_label": canonical,
            "team_id": None if canonical is None else "england:" + canonical}


def _season_years(season):
    if not re.fullmatch(r"20\d{2}-\d{2}", season):
        raise ValueError("Season must be YYYY-YY")
    start = int(season[:4])
    if (start + 1) % 100 != int(season[-2:]):
        raise ValueError("Season years must be consecutive")
    return start, start + 1


def parse_text(text, *, season, competition, source_url="", source_path=""):
    """Parse two documented Football.TXT match layouts; report other lines.

    Date-year completion uses only the explicitly supplied July-to-June season.
    No publication time, UTC kickoff, lineups, odds or missing goals are invented.
    """
    first_year, last_year = _season_years(season)
    if competition not in DIVISIONS:
        raise ValueError("Known English tier required")
    rows, issues, metadata = [], [], []
    day, clock, heading, stage = None, None, None, "regular"
    annotation_depth = 0
    date_line = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2})(?: (\d{4}))?$")
    for number, original in enumerate(text.splitlines(), 1):
        value = original.strip()
        if not value:
            continue
        if value.startswith(("#", "=")):
            metadata.append({"line": number, "text": value})
            continue
        if value.startswith("▪"):
            heading = value.lstrip("▪ ")
            if re.search(r"playoff|play-off|final", heading, re.I):
                stage = "playoff"
            elif re.search(r"matchday|regular", heading, re.I):
                stage = "regular"
            else:
                stage = "unknown"
                issues.append({"line": number, "kind": "unrecognized_stage", "raw": original})
            continue
        indentation = len(original) - len(original.lstrip())
        if rows and indentation >= 18 and (value.startswith("(") or annotation_depth > 0):
            rows[-1]["source_annotations"].append({"line": number, "raw": original})
            annotation_depth += value.count("(") - value.count(")")
            if annotation_depth < 0:
                issues.append({"line": number, "kind": "unbalanced_annotation", "raw": original})
                annotation_depth = 0
            continue
        if annotation_depth:
            issues.append({"line": number, "kind": "unterminated_annotation_before_line", "raw": original})
            annotation_depth = 0
        found_date = date_line.fullmatch(value)
        if found_date:
            weekday, month_name, day_text, year_text = found_date.groups()
            month = MONTHS[month_name]
            year = int(year_text) if year_text else first_year if month >= 7 else last_year
            try:
                candidate = date(year, month, int(day_text))
                if not (date(first_year, 7, 1) <= candidate < date(last_year, 7, 1)):
                    raise ValueError("Date lies outside stated season")
                if WEEKDAYS[candidate.weekday()] != weekday:
                    raise ValueError("Date weekday disagrees with calendar")
                day, clock = candidate, None
            except ValueError as error:
                day, clock = None, None
                issues.append({"line": number, "kind": "invalid_date", "raw": original, "error": str(error)})
            continue
        # Do not let a malformed date-looking line inherit an earlier date.
        if re.match(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", value):
            day, clock = None, None
            issues.append({"line": number, "kind": "unparsed_date", "raw": original})
            continue
        has_clock = re.match(r"^(\d{2}:\d{2})\s+(.+)$", value)
        if has_clock:
            clock, value = has_clock.groups()
            hour, minute = map(int, clock.split(":"))
            if hour > 23 or minute > 59:
                issues.append({"line": number, "kind": "invalid_clock", "raw": original})
                clock = None
                continue
        parts = re.split(r"\s{2,}", value)
        home = away = score_text = None
        if len(parts) == 3 and parts[1].startswith("v "):
            home, away, score_text = parts[0], parts[1][2:].strip(), parts[2]
        elif len(parts) == 3:
            home, score_text, away = parts
        elif len(parts) == 2 and " v " in parts[0]:
            home, away = parts[0].split(" v ", 1)
            score_text = parts[1]
        if home is None or away is None or not home or not away or home == away:
            issues.append({"line": number, "kind": "unparsed_match_or_status", "raw": original})
            continue
        pairs = re.findall(r"(?<![\d-])(\d+)-(\d+)(?![\d-])", score_text)
        pairs = [[int(a), int(b)] for a, b in pairs]
        aet, penalties = "a.e.t." in score_text, "pen." in score_text
        # Accept only understood numeric-score annotation shapes. Arbitrary
        # suffixes/statuses cannot masquerade as ordinary completed results.
        remainder = re.sub(r"\d+-\d+|a\.e\.t\.|pen\.|[(),\s]", "", score_text)
        if not pairs or remainder or (penalties and len(pairs) < 2):
            issues.append({"line": number, "kind": "unparsed_score_or_status", "raw": original})
            continue
        if not (aet or penalties) and not re.fullmatch(r"\d+-\d+(?:\s+\(\d+-\d+\))?", score_text):
            issues.append({"line": number, "kind": "ambiguous_ordinary_score", "raw": original})
            continue
        if day is None:
            issues.append({"line": number, "kind": "match_without_valid_date", "raw": original})
            continue
        reported_score = pairs[1] if penalties else pairs[0]
        home_info, away_info = _team(home), _team(away)
        unknown = [x["source_label"] for x in (home_info, away_info) if x["team_id"] is None]
        if unknown:
            issues.append({"line": number, "kind": "unknown_team_alias", "labels": unknown, "raw": original})
        regular_score = reported_score if not (aet or penalties) else None
        invalid_half = not (aet or penalties) and len(pairs) == 2 and any(h > f for h, f in zip(pairs[1], reported_score))
        if invalid_half:
            issues.append({"line": number, "kind": "half_time_exceeds_full_time", "raw": original})
        if stage == "regular" and (aet or penalties):
            issues.append({"line": number, "kind": "extra_time_in_regular_stage", "raw": original})
        key = f"openfootball:{competition}:{season}:{day.isoformat()}:{home}|{away}"
        rows.append({"match_id": key, "competition": competition, "season": season,
                     "match_date": day.isoformat(), "stage": stage, "round_text": heading,
                     "home": home_info, "away": away_info,
                     "source_clock_text": clock, "kickoff_utc": None,
                     "neutral_status": "unknown",
                     "reported_score": reported_score, "regulation_time_score": regular_score,
                     "extra_time": aet, "penalties": penalties, "score_pairs": pairs,
                     "source_annotations": [],
                     "raw_score_text": score_text, "raw_line": original, "source_line": number,
                     "eligible_regular_result": stage == "regular" and regular_score is not None and not unknown and not invalid_half,
                     "source_url": source_url, "source_path": source_path})
    if annotation_depth:
        issues.append({"line": len(text.splitlines()), "kind": "unterminated_annotation_at_eof"})
    counts = Counter(row["match_id"] for row in rows)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    regular = [row for row in rows if row["stage"] == "regular"]
    teams = sorted({team["source_label"] for row in regular for team in (row["home"], row["away"])})
    appearances = Counter(team["source_label"] for row in regular for team in (row["home"], row["away"]))
    dates = [row["match_date"] for row in regular]
    return {"competition": competition, "season": season, "records": rows,
            "metadata_lines": metadata, "issues": issues, "duplicate_keys": duplicates,
            "coverage": {"parsed_records": len(rows), "regular_records": len(regular),
                         "playoff_records": sum(r["stage"] == "playoff" for r in rows),
                         "eligible_regular_records": sum(r["eligible_regular_result"] for r in rows),
                         "regular_team_count": len(teams), "regular_team_appearances": dict(sorted(appearances.items())),
                         "regular_date_min": min(dates) if dates else None,
                         "regular_date_max": max(dates) if dates else None,
                         "extra_time_records": sum(r["extra_time"] for r in rows),
                         "penalty_records": sum(r["penalties"] for r in rows)},
            "time_semantics": {"match_date": "source calendar date only",
                               "clock": "source local text; timezone not verified",
                               "historical_first_availability": None,
                               "download_time_is_historical_availability": False}}


def reinspect_saved(directory):
    """Reparse retained raw files offline, preserving previous inspection output."""
    directory = Path(directory).resolve()
    if os.name == "nt" and directory.drive.upper() != "G:":
        raise ValueError("All project intake files must stay on G:")
    manifest = json.loads((directory / "intake_summary.json").read_text(encoding="utf-8"))
    created = datetime.now(timezone.utc)
    output = directory / ("inspection_" + created.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    inspections = []
    for item in manifest["files"]:
        if item["competition"] is None:
            continue
        path = Path(item["path"]).resolve()
        if path.parent != directory:
            raise ValueError("Raw input path is outside its intake directory")
        value = parse_text(path.read_text(encoding="utf-8-sig"), season=item["season"],
                           competition=item["competition"], source_url=item["source_url"], source_path=str(path))
        inspections.append(value)
        (output / f"{item['season']}_{item['competition']}.inspection.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result = {"created_at": created.isoformat(), "raw_directory": str(directory),
              "inspection_directory": str(output), "new_network_requests": 0,
              "coverage": [{"season": x["season"], "competition": x["competition"], **x["coverage"],
                            "issues": len(x["issues"]), "duplicate_keys": x["duplicate_keys"]} for x in inspections],
              "transition_summary": transition_summary(inspections), "trained": False,
              "annotations_policy": "Raw scorer/event annotations retained as unparsed source annotations, not player-event or on-field data."}
    (output / "coverage_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result, output


def _fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "personal-football-research/0.1"}), timeout=30) as response:
        return response.read(), {"resolved_url": response.url, "status": response.status,
                                 "http_date": response.headers.get("Date"),
                                 "last_modified": response.headers.get("Last-Modified"),
                                 "content_type": response.headers.get("Content-Type")}


def transition_summary(inspections):
    """Describe shared team identities, not a fitted cross-division adjustment."""
    membership = {}
    for inspection in inspections:
        for row in inspection["records"]:
            if not row["eligible_regular_result"]:
                continue
            for side in ("home", "away"):
                team_id = row[side]["team_id"]
                membership.setdefault(team_id, {}).setdefault(row["season"], set()).add(row["competition"])
    transitions = []
    for team_id, seasons in sorted(membership.items()):
        ordered = sorted(seasons)
        for earlier, later in zip(ordered, ordered[1:]):
            if int(later[:4]) == int(earlier[:4]) + 1 and seasons[earlier] != seasons[later]:
                transitions.append({"team_id": team_id, "earlier_season": earlier,
                                    "earlier_divisions": sorted(seasons[earlier]),
                                    "later_season": later, "later_divisions": sorted(seasons[later])})
    focus = {"england:" + name: {season: sorted(divisions) for season, divisions in sorted(membership.get("england:" + name, {}).items())}
             for name in ("Ipswich", "Hull", "Coventry")}
    return {"adjacent_season_division_changes": transitions, "focus_team_membership": focus,
            "identifiability_note": "Shared identities across seasons permit studying transitions only with explicit temporal/strength assumptions. Disconnected contemporaneous divisions alone do not identify a division-strength offset; a different league goal average is not that offset.",
            "model_fit_performed": False, "model_changes": False}


def run_intake(root, *, fetch=_fetch):
    root = Path(root).resolve()
    if os.name == "nt" and root.drive.upper() != "G:":
        raise ValueError("All project intake files must stay on G:")
    started = datetime.now(timezone.utc)
    directory = root / "data" / "english-tiers" / started.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    files, inspections = [], []
    # Exactly eight narrow result texts and one repository license; no full clone.
    requests = [(season, code, f"{BASE}/{season}/{filename}")
                for season in SEASONS for code, filename in DIVISIONS.items()]
    requests.append((None, None, f"{BASE}/LICENSE.md"))
    for season, code, url in requests:
        body, response = fetch(url)
        received = datetime.now(timezone.utc)
        name = "LICENSE.md" if season is None else f"{season}_{code}.txt"
        path = directory / name
        with path.open("xb") as stream:
            stream.write(body)
        files.append({"season": season, "competition": code, "source_url": url,
                      "repository": "https://github.com/openfootball/england", "git_ref": "master",
                      "path": str(path), "bytes": len(body), "received_at": received.isoformat(), **response})
        if season is not None:
            inspection = parse_text(body.decode("utf-8-sig"), season=season, competition=code,
                                    source_url=url, source_path=str(path))
            inspections.append(inspection)
            (directory / f"{season}_{code}.inspection.json").write_text(
                json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result = {"started_at": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat(),
              "directory": str(directory), "source_license": "Repository declares CC0-1.0; original per-record rights/provenance not independently guaranteed",
              "files": files, "coverage": [{"season": x["season"], "competition": x["competition"],
                                             **x["coverage"], "issues": len(x["issues"]), "duplicate_keys": x["duplicate_keys"]} for x in inspections],
              "transition_summary": transition_summary(inspections), "trained": False,
              "predictions_generated": False, "existing_forecasts_modified": False,
              "limitations": ["Mutable master source observations, not a historical as-of reconstruction or a pinned snapshot of upstream rights.",
                              "Play-offs remain separate and do not supply regular league training rows.",
                              "Extra-time/penalty markers preserve source text; reported scores are not inferred 90-minute scores.",
                              "Present source may contain corrections or administrative outcomes; special matches need source-specific verification.",
                              "Dates and unverified local clock strings cannot reconstruct UTC kickoff or first publication."]}
    (directory / "intake_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result, directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reinspect", type=Path)
    args = parser.parse_args(argv)
    result, directory = reinspect_saved(args.reinspect) if args.reinspect else run_intake(args.root)
    print(json.dumps({"directory": str(directory), "files": len(result.get("files", [])),
                      "coverage": result["coverage"], "trained": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
