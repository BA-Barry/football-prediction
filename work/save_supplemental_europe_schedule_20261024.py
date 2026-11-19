"""Save newly observed European scheduling gaps without modifying forecasts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
DEST = ROOT / "data/prematch-intelligence/NEW20261024_cohort_20261024T150344311377Z"


def main():
    observed = datetime.now(timezone.utc).isoformat()
    ucl = "https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-phase-fixtures/"
    uel = "https://www.uefa.com/uefaeuropaleague/news/02a8-2174cafa5bb6-82bbc20c9b92-1000--2026-27-europa-league-all-the-league-phase-fixtures-and-re/"
    uecl = "https://www.uefa.com/uefaconferenceleague/news/02aa-21e40e257e01-64251de1d99e-1000--uefa-conference-league-matchday-2-highlights-and-round-up-tr/"
    specs = [
        ("Aston Villa", "Barcelona", "Aston Villa", "2026-11-03T20:00:00Z", "UEFA Champions League", ucl),
        ("Manchester United", "Manchester United", "Roma", "2026-11-03T20:00:00Z", "UEFA Champions League", ucl),
        ("Liverpool", "Fenerbahçe", "Liverpool", "2026-11-04T17:45:00Z", "UEFA Champions League", ucl),
        ("Manchester City", "Leipzig", "Manchester City", "2026-11-04T20:00:00Z", "UEFA Champions League", ucl),
        ("Arsenal", "Slavia Praha", "Arsenal", "2026-11-04T20:00:00Z", "UEFA Champions League", ucl),
        ("AFC Bournemouth", "Sparta Praha", "AFC Bournemouth", "2026-11-05T17:45:00Z", "UEFA Europa League", uel),
        ("Crystal Palace", "Crystal Palace", "Hoffenheim", "2026-11-05T17:45:00Z", "UEFA Europa League", uel),
        ("Sunderland", "Sunderland", "GNK Dinamo", "2026-11-05T20:00:00Z", "UEFA Europa League", uel),
        ("Brighton & Hove Albion", "Getafe", "Brighton & Hove Albion", "2026-11-05T20:00:00Z", "UEFA Conference League", uecl),
    ]
    cohort = json.loads((DEST / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    rows = []
    for team, home, away, kickoff, competition, url in specs:
        f, = [f for f in cohort if team in (f["home"], f["away"])]
        gap = (datetime.fromisoformat(kickoff.replace("Z", "+00:00"))-
               datetime.fromisoformat(f["kickoff"].replace("Z", "+00:00"))).total_seconds()/3600
        corroboration = []
        if team == "Aston Villa":
            corroboration = ["https://www.premierleague.com/en/news/4699703/champions-league-league-phase-schedule-confirmed",
                "https://next.gc.avfcservices.co.uk/news/2026/september/30/ticketing-barcelona-ticket-details/"]
        if team == "Manchester United":
            corroboration = ["https://www.manutd.com/en/news/champions-league-dates-confirmed-2026-27",
                "https://www.manutd.com/en/matches/mens-team/fixtures?competition=uefa-champions-league&month=2026-11",
                "https://www.asroma.com/en/news/75819/uefa-champions-league-schedule"]
        if team == "Brighton & Hove Albion":
            corroboration = ["https://www.brightonandhovealbion.com/uefa-conference-supporter-info"]
        rows.append({"team": team, "home": home, "away": away, "kickoff": kickoff,
            "competition": competition, "observed_at": observed, "published_at": None,
            "source_url": url, "source_kind": "official_uefa_page_body_or_search_index_body_via_web_tool",
            "corroborating_primary_sources": corroboration,
            "affected_prospective_fixture_id": f["match_id"], "scheduled_kickoff_gap_hours_after_epl": gap,
            "claim": f"Official European schedule lists {home} hosting {away} at {kickoff}; it follows this EPL fixture by {gap:g} scheduled kickoff hours.",
            "limits": ["Future scheduled fixture and arithmetic kickoff gap only, not actual player recovery, travel duration, fatigue effect or competition priority.",
                "This observation was recorded after the earlier cup-enriched packet; it does not retroactively establish that any original LLM forecast used it."]})
    payload = {"recorded_at": observed, "scope": "New source-only schedule supplement; no prediction probabilities, fitting or prior-record changes.",
        "trigger": {"source_url": "https://www.premierleague.com/en/news/4688862/fixture-amendments-for-premier-league-matches-in-october-and-november",
            "root_observation": "Parent identified explicit Aston Villa November3 UCL involvement on the amendment page.",
            "audit_note": "This agent's later web text did not expose the same sentence, so exact opponent/time were verified against the separate official UEFA schedule and league/club ticket sources rather than inferred from the amendment."},
        "fixtures": rows,
        "participation_source": {"url": "https://www.premierleague.com/ar/news/4675720/how-do-other-competitions-fixtures-affect-the-premier-league-and-its-clubs",
            "observed_at": observed, "source_kind": "official_league_search_index_body",
            "claim": "Premier League source identifies nine English European participants: UCL Arsenal/ManchesterCity/ManchesterUnited/AstonVilla/Liverpool; UEL Bournemouth/Sunderland/CrystalPalace; ConferenceBrighton. All nine have a sourced Nov3-5 fixture above."},
        "uncertainties": ["Not a certified complete all-competition calendar or exhaustive last-minute rescheduling audit.",
            "Starting lineups, actual European minutes, travel plans, tactical priorities and importance by competition are not known.",
            "Scheduled involvement permits an explicit rotation/priority uncertainty statement; it does not justify hand-written numerical penalties.",
            "UEFA pages specify CET, normally21:00 unless otherwise shown; in November CET=UTC+1. Club GMT/UK statements corroborate Villa and ManchesterUnited and Brighton."],
        "future_revision_only": {"do_not_overwrite": ["news_evidence.json", "news_with_cup_context.json", "existing packets", "existing LLM responses", "existing predictions"],
            "next_action": "Provide these facts to a separately dated pre-kickoff revision with the earlier version retained; compare paired fixture versions without counting new matches.",
            "known_data_gap": "Earlier source-only cup packet omitted the verified Nov3-5 European next-match commitments. This is an input-completeness gap; no settled loss is causally attributed to it."}}
    path = DEST / "supplemental_europe_schedule_observation.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"path": str(path), "european_teams": len(rows), "recorded_at": observed,
        "villa": rows[0]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
