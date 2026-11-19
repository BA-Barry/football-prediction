"""One direct official openfootball file; inspection and description only."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from urllib.request import Request, urlopen

from football_model.openfootball import inspect_payload
from football_model.openfootball_snapshot import _calendar_coverage

ROOT = REPOSITORY_ROOT
URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/en.1.json"


def main():
    started = datetime.now(timezone.utc)
    target = ROOT / "data" / "prospective_context" / started.strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True, exist_ok=False)
    request = Request(URL, headers={"User-Agent": "football-personal-research-context/0.28"})
    with urlopen(request, timeout=30) as response:
        body = response.read()
        headers = dict(response.headers)
    observed_at = datetime.now(timezone.utc)
    (target / "en.1.json").write_bytes(body)
    payload = json.loads(body.decode("utf-8-sig"))
    inspected = inspect_payload(payload, competition="en.1", season="2026-27", source_url=URL,
                                observed_at=observed_at)
    coverage = _calendar_coverage(inspected, observed_at.date())
    teams = {}
    for label in ("Manchester United FC", "AFC Bournemouth"):
        found = [row for row in inspected["records"] if label in (row["home_label"], row["away_label"])
                 and row["full_time"] is not None and row["match_date"] < observed_at.date().isoformat()]
        if not found:
            # Report source spelling instead of silently creating a cross-source alias.
            found = [row for row in inspected["records"]
                     if any(part in (row["home_label"] + "|" + row["away_label"]).lower()
                            for part in (["manchester united"] if "United" in label else ["bournemouth"]))
                     and row["full_time"] is not None and row["match_date"] < observed_at.date().isoformat()]
        teams[label] = sorted(found, key=lambda row: row["match_date"])[-5:]
    metadata = {
        "url": URL, "observed_at": observed_at.isoformat(), "downloaded_bytes": len(body),
        "source_response_headers": headers,
        "timestamp_semantics": "observed_at is this response receipt, not match publication or source availability",
        "source_calendar_date": observed_at.date().isoformat(), "coverage": coverage,
        "inspection_without_records": {key: value for key, value in inspected.items() if key != "records"},
        "recent_reported_results": teams,
        "trained": False, "predictions_generated": False,
    }
    (target / "inspection.json").write_text(json.dumps(inspected, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "context.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(target), "observed_at": observed_at.isoformat(), "coverage": coverage,
                      "inspection": metadata["inspection_without_records"], "recent_reported_results": teams},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
