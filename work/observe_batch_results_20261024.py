"""Read official FPL snapshots without changing any forecast or model record."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = REPOSITORY_ROOT
BASE = ROOT / "data" / "prematch-intelligence" / "20261024_batch"
UTC = timezone.utc


def stamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def download(item: tuple[str, str]) -> tuple[str, bytes, dict]:
    label, url = item
    started_at = stamp()
    try:
        request = Request(url, headers={"User-Agent": "FootballResearchPersonal/0.30"})
        with urlopen(request, timeout=30) as response:
            body = response.read()
            received_at = stamp()
            metadata = {
                "source_url": url,
                "request_started_at": started_at,
                "received_at": received_at,
                "status": response.status,
                "response_url": response.url,
                "response_date": response.headers.get("Date"),
                "age_seconds": response.headers.get("Age"),
                "cache_control": response.headers.get("Cache-Control"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type"),
                "bytes": len(body),
                "raw_path": f"{label}_raw.json",
            }
    except Exception as error:
        body = error.read() if isinstance(error, HTTPError) else b""
        metadata = {
            "source_url": url,
            "request_started_at": started_at,
            "received_at": stamp(),
            "status": error.code if isinstance(error, HTTPError) else None,
            "error": f"{type(error).__name__}: {error}",
            "bytes": len(body),
            "raw_path": f"{label}_error_body.bin" if body else None,
        }
    return label, body, metadata


def write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main() -> None:
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE / f"result_observation_{run_stamp}"
    output_dir.mkdir()
    sources = [
        ("fixtures", "https://fantasy.premierleague.com/api/fixtures/"),
        ("bootstrap", "https://fantasy.premierleague.com/api/bootstrap-static/"),
    ]
    fetched = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for label, body, metadata in executor.map(download, sources):
            if body and metadata.get("raw_path"):
                with (output_dir / metadata["raw_path"]).open("xb") as output:
                    output.write(body)
            fetched[label] = metadata
    write_json(output_dir / "source_metadata.json", {
        "recorded_at": stamp(), "sources": fetched,
        "note": "Fresh receipt is not proof of fresh upstream content; HTTP cache metadata is preserved. No retry or access workaround.",
    })
    report = {
        "recorded_at": stamp(),
        "output_dir": str(output_dir),
        "source_metadata": fetched,
        "fixture_observations": [],
        "settlement_policy": "Only a unique exact official fixture ID, team IDs and kickoff match with finished=true and integer non-negative scores is settlement-ready; no result is inferred from elapsed time.",
        "forecast_changes": 0,
        "model_fits": 0,
    }
    if all(item.get("status") == 200 for item in fetched.values()):
        current = json.loads((output_dir / "fixtures_raw.json").read_bytes())
        bootstrap = json.loads((output_dir / "bootstrap_raw.json").read_bytes())
        old = json.loads((BASE / "fpl_fixtures_raw.json").read_bytes())
        old_metadata = json.loads((BASE / "fpl_raw_source_metadata.json").read_text(encoding="utf-8-sig"))
        batch = json.loads((BASE / "fixtures.json").read_text(encoding="utf-8-sig"))["fixtures"]
        old_names = {team["name"]: team["id"] for team in old_metadata["teams"]}
        current_teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
        aliases = {
            "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
            "Leeds United": "Leeds", "Manchester United": "Man Utd",
            "Manchester City": "Man City", "Newcastle United": "Newcastle",
            "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Spurs",
        }
        report["current_fixture_count"] = len(current)
        report["current_team_mapping"] = current_teams
        for fixture in batch:
            home_name = aliases.get(fixture["home"], fixture["home"])
            away_name = aliases.get(fixture["away"], fixture["away"])
            observation = {"match_id": fixture["match_id"], "home": fixture["home"], "away": fixture["away"], "expected_kickoff": fixture["kickoff"]}
            home_id, away_id = old_names.get(home_name), old_names.get(away_name)
            old_matches = [item for item in old if item.get("team_h") == home_id and item.get("team_a") == away_id and item.get("kickoff_time") == fixture["kickoff"]]
            if len(old_matches) != 1:
                observation.update({"verified": False, "reason": "No unique original official fixture identity", "original_matches": len(old_matches)})
            else:
                source_id = old_matches[0]["id"]
                matches = [item for item in current if item.get("id") == source_id and item.get("team_h") == home_id and item.get("team_a") == away_id and item.get("kickoff_time") == fixture["kickoff"]]
                mapping_ok = current_teams.get(home_id) == home_name and current_teams.get(away_id) == away_name
                observation.update({"official_fixture_id": source_id, "home_team_id": home_id, "away_team_id": away_id, "team_mapping_matches": mapping_ok})
                if len(matches) != 1 or not mapping_ok:
                    observation.update({"verified": False, "reason": "Current source identity or team mapping differs", "current_matches": len(matches)})
                else:
                    match = matches[0]
                    scores_ok = all(type(match.get(key)) is int and match[key] >= 0 for key in ("team_h_score", "team_a_score"))
                    observation.update({"verified": True, "kickoff_time": match.get("kickoff_time"), "event": match.get("event"), "started": match.get("started"), "finished": match.get("finished"), "finished_provisional": match.get("finished_provisional"), "minutes": match.get("minutes"), "home_goals": match.get("team_h_score"), "away_goals": match.get("team_a_score"), "settlement_ready": match.get("finished") is True and scores_ok})
            report["fixture_observations"].append(observation)
    else:
        report["unavailable_reason"] = "At least one official snapshot was unavailable; no settlement inferred."
    write_json(output_dir / "result_observations.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
