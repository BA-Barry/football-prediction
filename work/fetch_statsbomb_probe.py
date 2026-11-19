"""Bounded official open-data retrieval for an internal 34-match parser study.

No credentials, third-party mirrors, scraping or future availability claims.
Untouched source files stay on G and are not included in the code archive.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("G:/Codex/football-prediction-2026-10-22")
DEST = ROOT / "data/statsbomb/20261022_player_probe"
BASE = "https://raw.githubusercontent.com/hudl/open-data/master/"


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(item):
    relative, local_name = item
    path = DEST / local_name
    url = BASE + relative
    retrieved = None
    reused = path.exists()
    if not reused:
        with urlopen(Request(url, headers={"User-Agent": "personal-football-statistics/0.6"}),
                     timeout=60) as response:
            payload = response.read()
        if path.suffix == ".json":
            json.loads(payload)  # Do not save an HTTP error body as source JSON.
        with path.open("xb") as stream:
            stream.write(payload)
        retrieved = now()
    return {"source_url": url, "local_path": str(path), "bytes": path.stat().st_size,
            "retrieved_at_utc": retrieved, "observed_local_at_utc": now(),
            "reused_local_file": reused, "historical_first_available_at": None}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = [fetch(item) for item in [
        ("LICENSE.pdf", "LICENSE.pdf"), ("README.md", "SOURCE_README.md"),
        ("data/competitions.json", "competitions.json"),
        ("data/matches/9/281.json", "matches_9_281.json"),
    ]]
    matches = json.loads((DEST / "matches_9_281.json").read_text(encoding="utf-8"))
    matches = [m for m in matches if 904 in (m["home_team"]["home_team_id"],
                                            m["away_team"]["away_team_id"])]
    if not matches:
        raise ValueError("No published Leverkusen matches in the selected season")
    items = [(f"data/{kind}/{m['match_id']}.json", f"{kind}_{m['match_id']}.json")
             for m in sorted(matches, key=lambda m: (m["match_date"], m["match_id"]))
             for kind in ("events", "lineups")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, record in enumerate(pool.map(fetch, items), 1):
            records.append(record)
            if index % 10 == 0 or index == len(items):
                print(f"Official event/lineup files available: {index}/{len(items)}", flush=True)
    manifest = {"created_at_utc": now(), "source": "StatsBomb / Hudl Open Data",
                "purpose": "personal noncommercial retrospective football analytics",
                "scope": f"9/281: {len(matches)} published Leverkusen Bundesliga 2023/24 matches",
                "not_complete_league": True, "records": records,
                "terms_url": BASE + "LICENSE.pdf",
                "limits": ["No redistribution of raw data; software license is separate",
                           "No historical lineup announcement or first-availability timestamps",
                           "No verified kickoff timezone; no current injury evidence",
                           "No registration or personal contact details submitted"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = DEST / f"下载来源_{stamp}.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Source record: {destination}", flush=True)


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
