"""Retain observed official schedule evidence separately from original cohort."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(r"G:\Codex\football-prediction-2026-10-22")
DEST = ROOT / "data/prematch-intelligence/NEW20261024_cohort_20261024T150344311377Z"
sys.path.insert(0, str(ROOT / "work"))
from observe_batch_results_20261024 import write_json, stamp


def main():
    observed = stamp()
    fixtures = json.loads((DEST / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    news = json.loads((DEST / "news_evidence.json").read_text(encoding="utf-8"))
    efl = "https://www.efl.com/competitions/carabao-cup/"
    cup = [
        ("Bradford City", "Peterborough United", "2026-10-27T19:45:00Z"),
        ("Fleetwood Town", "Arsenal", "2026-10-27T20:00:00Z"),
        ("Manchester City", "Brighton & Hove Albion", "2026-10-28T19:30:00Z"),
        ("AFC Bournemouth", "Aston Villa", "2026-10-28T19:45:00Z"),
        ("Fulham", "Crystal Palace", "2026-10-28T19:45:00Z"),
        ("Sunderland", "Brentford", "2026-10-28T19:45:00Z"),
        ("Liverpool", "Chelsea", "2026-10-28T20:00:00Z"),
        ("Everton", "Newcastle United", "2026-10-29T19:45:00Z"),
    ]
    cup_rows = [{"home": h, "away": a, "kickoff": k, "competition": "Carabao Cup",
                 "stage": "Round Four", "source_url": efl, "observed_at": observed,
                 "source_kind": "official_league_search_cache", "published_at": None,
                 "retrieval_note": "Official EFL search-index body showed dated fixtures; direct page body exposed an empty Fixtures component. Observation is not upstream publication or last-update time."}
                for h, a, k in cup]
    crosscheck = []
    for f in fixtures:
        crosscheck.append({"fixture_id": f["match_id"], "home": f["home"], "away": f["away"],
            "kickoff": f["kickoff"], "fpl_source": f["source_url"],
            "league_page_source": "https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-202627-premier-league-season",
            "observed_at": observed, "source_kind": "official_league_page_body_via_web_tool",
            "match_and_time_agree": True, "timezone_source": "https://www.gov.uk/when-do-the-clocks-change",
            "timezone_note": "GOV.UK confirms UK returns to GMT on25Oct2026; Oct31-Nov2 UK displayed times equal UTC. Literal FPL UTC fields independently agree."})
        for team in (f["home"], f["away"]):
            matches = [r for r in cup_rows if team in (r["home"], r["away"])]
            for r in matches:
                kickoff = datetime.fromisoformat(f["kickoff"].replace("Z", "+00:00"))
                previous = datetime.fromisoformat(r["kickoff"].replace("Z", "+00:00"))
                gap = (kickoff-previous).total_seconds()/3600
                fact = {"id": f"cup-schedule:{r['kickoff'][:10]}:{team.lower().replace(' ', '-')}",
                    "url": efl, "published_at": None, "observed_at": observed,
                    "claim_type": "fact", "team": team, "category": "schedule",
                    "status": "official_source_assertion", "source_kind": "official_league_search_cache",
                    "claim": f"Official EFL indexed schedule lists {r['home']} hosting {r['away']} in Carabao Cup Round Four at {r['kickoff']}; this EPL match is {gap:g} scheduled kickoff hours later. Rotation, added time/penalties, actual participant minutes and competition priority are unknown.",
                    "scheduled_cup_fixture": r, "scheduled_kickoff_gap_hours": gap,
                    "retrieval_note": r["retrieval_note"],
                    "limits": ["Scheduled kickoff gap is not actual recovery time or an estimated strength effect.",
                        "Do not infer that a club takes either competition more seriously without sourced statements and later validation.",
                        "This is not a complete all-competition calendar; European next-match pressure and other fixtures may still be missing."]}
                if {r["home"], r["away"]} == {"Liverpool", "Chelsea"}:
                    fact["independent_club_corroboration"] = [
                        "https://www.liverpoolfc.com/tickets/ticket-info/2026-27/league-cup-home/chelsea-home-lc-2627",
                        "https://www.chelseafc.com/en/news/article/carabao-cup-ticket-news-liverpool-away-2026-27"]
                if {r["home"], r["away"]} == {"Everton", "Newcastle United"}:
                    fact["independent_club_corroboration"] = ["https://www.newcastleunited.com/en/matches/mens-team/fixtures"]
                if {r["home"], r["away"]} == {"Sunderland", "Brentford"}:
                    fact["independent_club_corroboration"] = ["https://www.brentfordfc.com/en/matches/mens/fixtures"]
                news["evidence_by_fixture"][f["match_id"]].append(fact)
    news["recorded_at"] = observed
    news["source_limits"] = ["Core fixture and FPL availability source-only packet remains unchanged in news_evidence.json.",
        "This enriched packet adds observed cup scheduling facts, without probabilities, manual fatigue penalties, or inferred motivation."]
    write_json(DEST / "news_with_cup_context.json", news)
    write_json(DEST / "official_schedule_crosscheck.json", {"recorded_at": observed, "crosscheck": crosscheck,
        "cup_fixtures": cup_rows,
        "additional_club_fixture_sources": [
            {"url": "https://www.manutd.com/en/matches/mens-team/fixtures?competition=premier-league", "observed_at": observed,
             "claim": "Club page lists Chelsea away31Oct12:30 at Stamford Bridge, agreeing with FPL and Premier League."},
            {"url": "https://www.liverpoolfc.com/tickets/match/liverpool-v-arsenal-english-premier-league-20261101", "observed_at": observed,
             "claim": "Liverpool club ticket page lists Arsenal home1Nov16:30, agreeing with FPL and Premier League."}],
        "market_search_note": "Exploratory search found aggregator pages but no verified complete single-bookmaker current1X2 snapshot for allten. No best-price composite, stale search price, or absent bookmaker attribution is promoted as a primary quote. Opening-to-current change remains unverified."})
    print(json.dumps({"directory": str(DEST), "news_path": str(DEST / "news_with_cup_context.json"),
        "fixture_count": len(fixtures), "cup_facts": sum(len([x for x in v if x.get('scheduled_cup_fixture')]) for v in news['evidence_by_fixture'].values())}, ensure_ascii=False))

if __name__ == "__main__":
    main()
