"""Save the away club's explicit completed report beside a fresh FPL receipt."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from observe_batch_results_20261024 import download, stamp, write_json


def main():
    root = REPOSITORY_ROOT
    directory = root / "data/prematch-intelligence/20261024_batch/result_observation_20261024T150244904813Z"
    url = "https://www.mancity.com/news/mens/aston-villa-v-city-premier-league-match-report-63928435"
    _, body, metadata = download(("villa_official_away_club_report", url))
    metadata["raw_path"] = "villa_official_away_club_report_raw.html"
    if metadata.get("status") != 200:
        raise RuntimeError(f"Official completed report unavailable: {metadata}")
    raw_path = directory / metadata["raw_path"]
    with raw_path.open("xb") as output:
        output.write(body)
    text = body.decode("utf-8", errors="replace")
    # Identity and completion evidence remain explicit human-reviewed source assertions.
    if "City slip to first Premier League loss of the season at Villa" not in text:
        raise ValueError("Unexpected official report title")
    write_json(directory / "villa_official_away_club_report_metadata.json", metadata)
    observed = json.loads((directory / "result_observations.json").read_text(encoding="utf-8"))
    row = next(item for item in observed["fixture_observations"] if item["official_fixture_id"] == 72)
    if (row["home_goals"], row["away_goals"]) != (2, 1):
        raise ValueError("Report and saved provider score differ")
    evidence = {
        "recorded_at": stamp(), "observation_kind": "saved_official_away_club_completed_match_report",
        "match_id": row["match_id"], "official_fixture_id": 72,
        "home": row["home"], "away": row["away"], "kickoff": row["kickoff_time"],
        "score": {"home": 2, "away": 1}, "source_url": url,
        "source_kind": "official_away_club_match_report", "original_raw_path": str(raw_path),
        "original_received_at": metadata["received_at"],
        "page_title": "City slip to first Premier League loss of the season at Villa",
        "page_category": "Match Report", "published_at": None,
        "body_completion_evidence": [
            {"paraphrase": "The opening paragraph reports Manchester City's completed away defeat at Villa, with score 2–1.",
             "short_excerpt": "went down to a 2-1 defeat away at Aston Villa"},
            {"paraphrase": "The report states City retain the league lead after the loss and next face Brighton in the cup, confirming it is a completed-match account."}
        ],
        "completion_supported": True, "exact_finished_at": None,
        "finished_by_observation": metadata["received_at"],
        "fpl_crosscheck": {"observation_path": str(directory / "result_observations.json"),
                           "finished": row["finished"], "finished_provisional": row["finished_provisional"],
                           "home_goals": row["home_goals"], "away_goals": row["away_goals"]},
        "time_limitations": ["Receipt is a confirmed completion upper bound, never the exact final whistle.",
                             "FPL provisional flag remains false/true as observed; it is not overwritten."],
        "forecast_changes": 0, "model_fits": 0, "parameter_changes": 0
    }
    path = directory / "villa_official_club_completion_evidence.json"
    write_json(path, evidence)
    print(json.dumps({"evidence": str(path), "received_at": metadata["received_at"], "bytes": len(body)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
