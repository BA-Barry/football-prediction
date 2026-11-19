"""Save explicitly attributed named-bookmaker page observations, not live offers."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
DEST = ROOT / "data/prematch-intelligence/NEW20261024_cohort_20261024T150344311377Z"


def main():
    recorded = datetime.now(timezone.utc).isoformat()
    fixtures = json.loads((DEST / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    # Each triplet is one complete Bet365 row actually shown in a retrieved page.
    # The fixed book was selected for availability, not maximum odds or outcomes.
    data = {
        ("Chelsea", "Manchester United"): ("12071042", "chelsea-vs-man-utd", [1.80, 3.80, 4.00]),
        ("Coventry City", "Sunderland"): ("12071041", "coventry-vs-sunderland", [2.70, 3.30, 2.50]),
        ("AFC Bournemouth", "Leeds United"): ("12070904", "bournemouth-vs-leeds", [2.20, 3.40, 3.25]),
        ("Brentford", "Nottingham Forest"): ("12070903", "brentford-vs-nottm-forest", [1.95, 3.60, 3.60]),
        ("Manchester City", "Brighton & Hove Albion"): ("12070901", "man-city-vs-brighton", [1.67, 4.10, 4.33]),
        ("Tottenham Hotspur", "Crystal Palace"): ("12070900", "tottenham-vs-crystal-palace", [1.61, 3.90, 5.00]),
        ("Aston Villa", "Fulham"): ("12071043", "aston-villa-vs-fulham", [2.05, 3.40, 3.50]),
        ("Liverpool", "Arsenal"): ("12071040", "liverpool-vs-arsenal", [3.25, 3.50, 2.10]),
        ("Newcastle United", "Everton"): ("12071039", "newcastle-vs-everton", [2.20, 3.50, 3.00]),
        ("Hull City", "Ipswich Town"): (None, None, [2.55, 3.40, 2.60]),
    }
    observations = []
    for fixture in fixtures:
        source_id, slug, prices = data[(fixture["home"], fixture["away"])]
        url = (f"https://oddsrun.com/match/football/gb/94/{source_id}/england-premier-league/{slug}/" if source_id else
               "https://www.footymetrics.com/fixtures/38300233-premier-league-hull-city-ipswich-town")
        source_kind = "third_party_page_body_via_web_tool"
        source = "OddsRun named Bet365 1X2 table via public web tool"
        extra = []
        if fixture["home"] == "Brentford":
            source_kind = "third_party_search_index_body_via_web_tool"
            extra = ["First complete named Bet365 row was shown in the search-index body. Later direct page body showed an empty odds component; retained as a cached observation, not a verified currently offered price or chronological price move."]
        if fixture["home"] == "Hull City":
            source_kind = "third_party_search_index_body_via_web_tool"
            source = "FootyMetrics explicitly named Bet365 Full-time 1X2 block via public web search-index body"
            extra = ["Direct FootyMetrics body was unavailable. Named Full-time Bet365 row was in search body and independently matched SportScore and OddsSafari indexed named Bet365 rows; upstream quote freshness remains unknown."]
        observations.append({"fixture_id": fixture["match_id"], "market": "regulation_full_time_one_x_two",
            "observed_at": recorded, "decimal_prices": dict(zip(("home", "draw", "away"), prices)),
            "bookmaker": "Bet365", "source": source, "source_url": url,
            "source_event_id": None, "aggregator_event_id": source_id,
            "source_kind": source_kind, "source_quote_excerpt": "Bet365 | " + " | ".join(f"{p:.2f}" for p in prices),
            "fixture_kickoff": fixture["kickoff"], "bookmaker_updated_at": None,
            "opening_prices": None, "cache_age_seconds": None,
            "observation_clock_note": "Record saved after reading the public web response in this active turn; exact per-response HTTP receipt clocks, cache ages and upstream publication/update times are unavailable. This recorded observation is not an originating bookmaker timestamp.",
            "selection_note": "Fixed named Bet365 complete standard 1X2 row; no cross-bookmaker best-price selection and no inferred opening-to-current movement.",
            "limits": ["Third-party representation attributed to Bet365; originating bookmaker quote not independently retrieved.",
                "Public web/search content may be cached, including several-day-old source crawls. Do not call this an executable live quote or a verified latest market consensus.",
                "Regulation full-time three-outcome row only; no early-payout promotion or qualification market.",
                "No historical opening snapshots, guaranteed availability, bookmaker publication time, or stake-specific price is known.", *extra]})
    payload = {"recorded_at": recorded, "fixture_source_files": ["fixtures.json"],
        "market_observations": observations, "coverage": {"fixtures": 10, "complete_named_bookmaker_rows": len(observations),
            "originating_bookmaker_direct_verified": 0, "verified_latest_executable_quotes": 0},
        "scope": "Descriptive captured named-bookmaker research comparison; lower freshness/source certainty than a dated originating-bookmaker snapshot. Kept separate from source-only LLM packets.",
        "retrieval_failures": [
            {"source_url": "https://sports.ladbrokes.com/sport/football/england/premier-league", "outcome": "tool inaccessible"},
            {"source_url": "https://sports.williamhill.com/betting/en-gb/football/competitions/OB_TY295/English-Premier-League/matches", "outcome": "502"},
            {"source_url": "https://www.betfair.com/sport/football/english-premier-league/10932509", "outcome": "Direct bookmaker page only showed original round, no new-cohort quotes"},
            {"source_url": "https://www.paddypower.com/football/english-premier-league", "outcome": "Explicit country restriction; no retry or workaround"},
            {"source_url": "https://sports.bwin.com/en/sports/football-4/betting/england-14/premier-league-102841", "outcome": "Body exposed no actual match quote table"},
            {"source_url": "https://www.bet365.com/hub/en-gb/football", "outcome": "Direct body fetch error; search cache only showed older fixtures"}],
        "earlier_record_note": "market_unavailable.json records the earlier acquisition status and remains preserved. This later record supplies cached third-party named-bookmaker observations, not an upgrade to verified live prices."}
    with (DEST / "actual_market_observations.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"path": str(DEST / "actual_market_observations.json"), "coverage": payload["coverage"], "recorded_at": recorded}, ensure_ascii=False))

if __name__ == "__main__":
    main()
