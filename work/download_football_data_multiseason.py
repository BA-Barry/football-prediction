"""Download official historical CSVs once for a personal football statistics model.

This task only retrieves and checks files. It does not fit a model or inspect
prediction performance, particularly for the later 2025/26 evaluation season.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = REPOSITORY_ROOT
sys.path.insert(0, str(ROOT / "model_v2" / "src"))
from football_model.football_data import LEAGUES, load_directory


SEASONS = (("2022/23", "2223"), ("2023/24", "2324"), ("2025/26", "2526"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-years", nargs="+", type=int,
                        choices=range(2020, 2026), default=None,
                        help="Season start years; default remains 2022 2023 2025")
    args = parser.parse_args()
    seasons = SEASONS if args.start_years is None else tuple(
        (f"{year}/{str(year+1)[-2:]}", f"{str(year)[-2:]}{str(year+1)[-2:]}")
        for year in sorted(set(args.start_years)))
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    summaries = []
    refused = False
    for season, code in seasons:
        if refused:
            break
        destination = ROOT / "data" / "football-data" / season.replace("/", "-") / run_id
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "purpose": "Personal noncommercial football statistical model and entertainment; not general-purpose AI training",
            "download_task": "Official historical files and coverage checks only; no parameter fitting or test-season performance inspection",
            "season": season, "started_at": started.isoformat(), "directory": str(destination),
            "source_information": "https://football-data.co.uk/data.php", "files": [], "error": None,
            "availability_note": "Download timestamps are present observations, not historical result or odds publication times",
        }
        for league, label in LEAGUES.items():
            url = f"https://football-data.co.uk/mmz4281/{code}/{league}.csv"
            try:
                request = Request(url, headers={"User-Agent": "Codex-personal-file-download/1.0"})
                with urlopen(request, timeout=30) as response:
                    body = response.read()
                    resolved = response.url
                    content_type = response.headers.get("Content-Type", "")
                first = body.splitlines()[0] if body else b""
                if not first.startswith((b"Div,", b"\xef\xbb\xbfDiv,")):
                    raise ValueError("Response does not have the expected official CSV header")
                path = destination / f"{league}.csv"
                path.write_bytes(body)
                manifest["files"].append({
                    "league": label, "code": league, "path": str(path), "source": url,
                    "resolved_url": resolved, "content_type": content_type, "bytes": len(body),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"Downloaded {season} {league} {len(body)} bytes", flush=True)
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                manifest["error"] = {
                    "source": url, "message": str(error),
                    "retry_after": error.headers.get("Retry-After") if isinstance(error, HTTPError) else None,
                }
                # One request per file. Any failure stops, especially server
                # refusal/rate limiting. No alternate host, mirror, or retry.
                refused = True
                break
        manifest_path = destination / "下载来源.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if manifest["files"]:
            records, report = load_directory(destination, season=season)
            for league, entry in report["leagues"].items():
                file_path = destination / f"{league}.csv"
                with file_path.open(encoding="utf-8-sig", newline="") as stream:
                    raw = [r for r in csv.DictReader(stream) if (r.get("HomeTeam") or "").strip()]
                entry["missing_full_time_raw_rows"] = sum(
                    not all((r.get(k) or "").strip() for k in ("FTHG", "FTAG", "FTR")) for r in raw)
                entry["missing_half_time_raw_rows"] = sum(
                    not all((r.get(k) or "").strip() for k in ("HTHG", "HTAG", "HTR")) for r in raw)
                expected_teams = 20 if league in {"E0", "SP1", "I1"} or (league == "F1" and int(season[:4]) < 2023) else 18
                expected_matches = expected_teams * (expected_teams - 1)
                entry["expected_league_teams"] = expected_teams
                entry["expected_regular_season_matches"] = expected_matches
                entry["regular_season_count_complete"] = entry["input_rows"] == expected_matches
                entry["team_count_matches_expected"] = entry["teams"] == expected_teams
                entry["coverage_note"] = "Expected count checks are not a per-match fact, status, or historical-availability audit"
            report["special_status_audit"] = "Not comprehensive. Existing adapter only has an explicit exclusion for the 2024/25 Union Berlin-Bochum administrative award. No original scores or dates changed."
            report_path = destination / "字段覆盖检查.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {
                "season": season, "directory": str(destination), "manifest": str(manifest_path),
                "inspection": str(report_path), "downloaded_files": len(manifest["files"]),
                "records": len(records), "date_min": report["date_min"], "date_max": report["date_max"],
                "rejected_rows": report["rejected_rows"], "duplicate_keys": report["duplicate_keys"],
                "field_issues": report["field_issues"],
                "leagues": {k: {field: item[field] for field in (
                    "input_rows", "parsed_rows", "teams", "date_min", "date_max", "half_time_rows",
                    "missing_full_time_raw_rows", "missing_half_time_raw_rows", "regular_season_count_complete",
                    "team_count_matches_expected", "market_rows")}
                    for k, item in report["leagues"].items()}, "error": manifest["error"],
            }
        else:
            summary = {"season": season, "directory": str(destination), "manifest": str(manifest_path),
                       "downloaded_files": 0, "error": manifest["error"]}
        summaries.append(summary)
    overall = ROOT / "data" / "football-data" / f"多赛季接入摘要_{run_id}.json"
    overall.write_text(json.dumps({"run_id": run_id, "seasons": summaries,
        "stopped_after_error": refused}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(overall), "seasons": summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
