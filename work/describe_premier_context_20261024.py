"""Check observed current supplemental results and describe each team's rows."""
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
CONTEXT = ROOT / "data" / "prospective_context" / "20261023T172314505200Z"


def describe(rows, team):
    def summary(sample):
        won = drawn = lost = gf = ga = 0
        for row in sample:
            home, away = row["full_time"]
            own, opp = (home, away) if row["home_label"] == team else (away, home)
            gf += own
            ga += opp
            won += own > opp
            drawn += own == opp
            lost += own < opp
        return {"played": len(sample), "won": won, "drawn": drawn, "lost": lost,
                "goals_for": gf, "goals_against": ga, "points": won * 3 + drawn}
    ordered = sorted(rows, key=lambda row: row["match_date"])
    return {"team": team, "all_seven": summary(ordered), "last_five": summary(ordered[-5:]),
            "home": summary([row for row in ordered if row["home_label"] == team]),
            "away": summary([row for row in ordered if row["away_label"] == team]), "results": ordered}


def main():
    inspected = json.loads((CONTEXT / "inspection.json").read_text(encoding="utf-8"))
    supplements = json.loads((CONTEXT / "supplements.json").read_text(encoding="utf-8"))
    official = json.loads((CONTEXT / "official_context.json").read_text(encoding="utf-8"))
    original = [row for row in inspected["records"] if row["full_time"] is not None]
    keys = {(row["match_date"], row["home_label"], row["away_label"]) for row in original}
    assert len(supplements["records"]) == 4
    for row in supplements["records"]:
        key = row["match_date"], row["home_label"], row["away_label"]
        assert key not in keys
        keys.add(key)
    combined = original + supplements["records"]
    descriptions = []
    for expected in official["league_aggregates_after_seven"]["teams"]:
        team = expected["team"]
        rows = [row for row in combined if team in (row["home_label"], row["away_label"])]
        described = describe(rows, team)
        for field, actual in described["all_seven"].items():
            assert actual == expected[field], (team, field, actual, expected[field])
        descriptions.append(described)
    output = {"source_inspection": str(CONTEXT / "inspection.json"),
              "source_supplements": str(CONTEXT / "supplements.json"),
              "scope": "Premier League matchday 1-7 descriptive results for two teams only",
              "original_league_scored_rows": len(original), "added_official_rows": len(supplements["records"]),
              "combined_league_scored_rows": len(combined), "other_matchday_6_7_results_missing": 16,
              "trained": False, "predictions_generated": False, "teams": descriptions,
              "limits": ["Not opponent-adjusted", "Small seven-game season sample", "Other teams have incomplete later results",
                         "No injury, fatigue or competition-priority weights estimated"]}
    (CONTEXT / "descriptive_context.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(CONTEXT / "descriptive_context.json"),
                      "combined_league_scored_rows": len(combined),
                      "teams": [{key: value for key, value in row.items() if key != "results"}
                                for row in descriptions]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
