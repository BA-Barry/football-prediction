"""Add a saved official away-club report to original and revised forecasts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from bs4 import BeautifulSoup
from football_model.batch_scoring import run

ROOT = REPOSITORY_ROOT
OBS = ROOT / "data/prematch-intelligence/20261024_batch/result_observation_20261024T160147143875Z"

def write(path, data):
    with path.open("x", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")

def main():
    raw = OBS / "everton_completed_report_news_raw.html"
    meta_path = OBS / "everton_completed_report_news_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    text = BeautifulSoup(raw.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)
    # The text and score were inspected, and these are literal source checks.
    assert metadata["status"] == 200
    assert "Everton Defeated At Arsenal" in text
    assert "consigned Everton to a 2-0 defeat" in text
    assert "The result moves Arsenal joint top of the league table." in text
    snapshot = json.loads((OBS / "result_observations.json").read_text(encoding="utf-8"))
    row = next(x for x in snapshot["fixture_observations"] if x["home"] == "Arsenal")
    evidence = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "observation_kind": "saved_official_away_club_completed_match_report",
        "match_id": row["match_id"], "official_fixture_id": row["official_fixture_id"],
        "home": row["home"], "away": row["away"], "kickoff": row["expected_kickoff"],
        "score": {"home": 2, "away": 0}, "source_url": metadata["source_url"],
        "source_kind": "official_away_club_match_report", "original_raw_path": str(raw),
        "original_metadata_path": str(meta_path), "original_received_at": metadata["received_at"],
        "page_title": "Everton Defeated At Arsenal", "page_category": "Men",
        "published_at": None,
        "body_completion_evidence": [
            {"paraphrase": "Opening paragraph reports Everton's completed 0–2 away defeat at Arsenal.",
             "short_excerpt": "consigned Everton to a 2-0 defeat"},
            {"paraphrase": "The body gives the resulting league-table change and Everton's next Carabao Cup fixture."}
        ],
        "completion_supported": True, "exact_finished_at": None,
        "finished_by_observation": metadata["received_at"],
        "fpl_crosscheck": {"observation_path": str(OBS / "result_observations.json"),
            "finished": row["finished"], "finished_provisional": row["finished_provisional"],
            "home_goals": row["home_goals"], "away_goals": row["away_goals"]},
        "forecast_changes": 0, "model_fits": 0, "parameter_changes": 0,
        "time_limitations": ["Receipt confirms completion by that time; it is not the actual whistle time.",
                             "FPL administrative completion fields remain as originally observed."]
    }
    evidence_path = OBS / "arsenal_official_away_club_completion_evidence.json"
    write(evidence_path, evidence)
    cases = {
        "original10": ROOT / "outputs/五线赛果评分_20261024T150438997728Z/原赛前记录及新增赛果.json",
        "player_revision9": ROOT / "outputs/五线赛果评分_20261024T150441350712Z/原赛前记录及新增赛果.json"
    }
    output = {}
    for label, case in cases.items():
        # FPL snapshot contains Ipswich, which is not part of revision9, so use
        # the explicit club evidence here and keep the full FPL snapshot separate.
        output[label] = run(ROOT, case, [evidence_path])
    write(OBS / "verified_scoring_outputs.json", output)
    print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
