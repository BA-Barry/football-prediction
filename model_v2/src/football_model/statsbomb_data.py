"""Bounded official open-data retrieval and local full-season parsing.

Raw data and derived observations stay separate from distributable software.
Download clocks are not historical feature-availability timestamps.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .statsbomb_stints import StatsBombParseError, parse_match


BASE = "https://raw.githubusercontent.com/hudl/open-data/master/"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def season_profile(matches):
    """Measured schedule coverage, not an assumption from the catalog label."""
    if not matches:
        raise ValueError("No match metadata")
    ids = Counter(m["match_id"] for m in matches)
    pairs = Counter((m["home_team"]["home_team_id"], m["away_team"]["away_team_id"]) for m in matches)
    appearances = Counter(team for pair, count in pairs.items() for team in pair for _ in range(count))
    teams = set(appearances)
    expected = {(h, a) for h in teams for a in teams if h != a}
    return {"matches": len(matches), "teams": len(teams),
            "date_min": min(m["match_date"] for m in matches),
            "date_max": max(m["match_date"] for m in matches),
            "duplicate_match_ids": [mid for mid, n in ids.items() if n > 1],
            "duplicate_home_away_pairs": [list(pair) for pair, n in pairs.items() if n > 1],
            "missing_home_away_pairs": [list(pair) for pair in sorted(expected - set(pairs))],
            "team_match_counts": {str(k): v for k, v in sorted(appearances.items())},
            "complete_double_round_robin": set(pairs) == expected
                and all(n == 1 for n in pairs.values()) and all(n == 1 for n in ids.values())}


def _fetch(item, destination):
    relative, filename = item
    path = destination / filename
    reused, retrieved = path.exists(), None
    if reused:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    else:
        with urlopen(Request(BASE + relative, headers={"User-Agent": "personal-football-statistics/0.7"}),
                     timeout=60) as response:
            payload = response.read()
        if path.suffix == ".json":
            json.loads(payload)
        with path.open("xb") as stream:
            stream.write(payload)
        retrieved = datetime.now(timezone.utc).isoformat()
    return {"source_url": BASE + relative, "local_path": str(path), "bytes": path.stat().st_size,
            "reused_local_file": reused, "retrieved_at_utc": retrieved,
            "observed_local_at_utc": datetime.now(timezone.utc).isoformat(),
            "historical_first_available_at": None}


def download_season(destination, competition=2, season=27):
    """One competition-season, two concurrent downloads; reuse local originals."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(type(x) is not int or x < 1 for x in (competition, season)):
        raise ValueError("Positive integer competition and season IDs required")
    records = [_fetch(item, destination) for item in [
        ("LICENSE.pdf", "LICENSE.pdf"), ("README.md", "SOURCE_README.md"),
        (f"data/matches/{competition}/{season}.json", f"matches_{competition}_{season}.json")]]
    matches = json.loads((destination / f"matches_{competition}_{season}.json").read_text(encoding="utf-8"))
    profile = season_profile(matches)
    if profile["duplicate_match_ids"]:
        raise ValueError("Duplicated source match IDs would duplicate event files")
    items = [(f"data/{kind}/{m['match_id']}.json", f"{kind}_{m['match_id']}.json")
             for m in sorted(matches, key=lambda m: (m["match_date"], m["match_id"]))
             for kind in ("events", "lineups")]
    failures = []

    def attempt(item):
        try:
            return _fetch(item, destination), None
        except Exception as error:
            return None, {"source_url": BASE + item[0], "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (record, failure) in enumerate(pool.map(attempt, items), 1):
            if record:
                records.append(record)
            if failure:
                failures.append(failure)
            if index % 40 == 0 or index == len(items):
                print(f"Official event/lineup files inspected: {index}/{len(items)}; failures {len(failures)}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "StatsBomb / Hudl Open Data", "competition_id": competition, "season_id": season,
        "schedule_profile": profile, "files": records, "failures": failures,
        "purpose": "personal noncommercial retrospective football analytics",
        "terms_url": BASE + "LICENSE.pdf", "registration_submitted": False,
        "raw_data_redistribution": False,
        "historical_lineup_announcement_time": None, "kickoff_timezone": None}
    path = destination / f"下载来源_{stamp}.json"
    write_json(path, manifest)
    print(f"Download record: {path}", flush=True)
    return manifest


def parse_season(destination, competition=2, season=27):
    """Return successful retrospective paths and explicit failures; no training."""
    destination = Path(destination)
    matches = json.loads((destination / f"matches_{competition}_{season}.json").read_text(encoding="utf-8"))
    profile = season_profile(matches)
    if profile["duplicate_match_ids"]:
        raise ValueError("Resolve duplicate metadata before parsing")
    parsed, failures, reconciled = [], [], []
    for meta in sorted(matches, key=lambda m: (m["match_date"], m["match_id"])):
        mid = meta["match_id"]
        try:
            events = json.loads((destination / f"events_{mid}.json").read_text(encoding="utf-8"))
            lineups = json.loads((destination / f"lineups_{mid}.json").read_text(encoding="utf-8"))
            path = parse_match(meta, events, lineups)
            raw_xg = sum(e["shot"]["statsbomb_xg"] for e in events if e["type"]["name"] == "Shot")
            parsed_xg = path["diagnostics"]["home_xg"] + path["diagnostics"]["away_xg"]
            if abs(raw_xg - parsed_xg) > 1e-8:
                raise StatsBombParseError("Raw shot xG disagrees with interval totals")
            parsed.append(path)
            reconciled.append({"source_match_id": mid, "raw_xg": raw_xg, "parsed_xg": parsed_xg,
                               "goals": path["diagnostics"]["event_goals"], "source_events": len(events)})
        except (StatsBombParseError, OSError, json.JSONDecodeError) as error:
            failures.append({"source_match_id": mid, "date": meta["match_date"],
                             "home": meta["home_team"]["home_team_name"],
                             "away": meta["away_team"]["away_team_name"], "error": str(error)})
    return parsed, {"schedule_profile": profile, "parsed_matches": len(parsed),
                    "parse_failures": failures, "source_reconciliation": reconciled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    download_season(args.root / "data/statsbomb/premier-league-2015-16")
