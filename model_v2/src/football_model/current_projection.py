"""One existing-parameter current research projection, not a live deployment.

All selected results have been observed before this run. Date-only match dates
remain date-only; no kickoff/publication times are manufactured. Partial current
league coverage is retained and reported rather than completed by imputation.
"""

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path

from .date_model import DateTeamGoalModel
from .football_data import HistoricalMatch, load_directory
from .native_experiment import season_directories
from .probability import score_grid, summarize
from .prospective import _time


ALIASES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry", "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool", "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest", "Sunderland AFC": "Sunderland", "Tottenham Hotspur FC": "Tottenham",
}


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def convert_current(row, source_path, as_of):
    """Explicit source aliases, no suffix stripping or fuzzy team matching."""
    if _time(row["observed_at"], "source observed_at") > as_of:
        raise ValueError("Current source was not observed by the fit cutoff")
    day = date.fromisoformat(row["match_date"])
    if day >= as_of.date():
        raise ValueError("Use strictly earlier calendar-date results in this research projection")
    h, a = ALIASES[row["home_label"]], ALIASES[row["away_label"]]
    score = row["full_time"]
    if (not isinstance(score, (list, tuple)) or len(score) != 2 or
            any(type(x) is not int or x < 0 for x in score)):
        raise ValueError("Reported full-time scores require two nonnegative integer goals")
    return HistoricalMatch(
        match_id=f"football-data:E0:2026/27:{h}|{a}", competition="E0", season="2026/27",
        match_date=day, source_time_text=None, home=f"football-data:E0:{h}", away=f"football-data:E0:{a}",
        home_label=h, away_label=a, full_time=tuple(score), half_time=None, markets={},
        source_path=str(source_path), source_url=row["source_url"], goal_fit_exclusion=None)


def load_current_context(context_dir, as_of):
    context_dir = Path(context_dir)
    inspection = json.loads((context_dir / "inspection.json").read_text(encoding="utf-8"))
    supplements = json.loads((context_dir / "supplements.json").read_text(encoding="utf-8"))
    if inspection["rejected_rows"] or inspection["duplicate_keys"]:
        raise ValueError("Current source contains unresolved invalid or duplicate rows")
    rows = [convert_current(row, context_dir / "en.1.json", as_of)
            for row in inspection["records"] if row["full_time"] is not None]
    rows += [convert_current(row, context_dir / "supplements.json", as_of) for row in supplements["records"]]
    if len({row.match_id for row in rows}) != len(rows):
        raise ValueError("Overlapping source matches need explicit resolution, not double counting")
    return rows, inspection, supplements


def team_descriptive(records, team, season):
    rows = sorted([r for r in records if r.season == season and team in (r.home_label, r.away_label)],
                  key=lambda r: (r.match_date, r.match_id))
    wins = draws = gf = ga = 0
    games = []
    for row in rows:
        home = row.home_label == team
        f, a = row.full_time if home else tuple(reversed(row.full_time))
        wins += f > a
        draws += f == a
        gf += f
        ga += a
        games.append({"date": row.match_date.isoformat(), "home": row.home_label,
                      "away": row.away_label, "score": list(row.full_time), "source": row.source_url})
    return {"team": team, "season": season, "played": len(rows), "won": wins, "drawn": draws,
            "lost": len(rows)-wins-draws, "goals_for": gf, "goals_against": ga,
            "points": 3*wins+draws, "results": games}


def run_projection(root, context_dir):
    root, context_dir = Path(root).resolve(), Path(context_dir).resolve()
    if os.name == "nt" and (root.drive.upper() != "G:" or context_dir.drive.upper() != "G:"):
        raise ValueError("Projection inputs and outputs must stay on G:")
    cutoff = datetime.now(timezone.utc)
    if cutoff >= datetime(2026, 10, 25, 14, tzinfo=timezone.utc):
        raise ValueError("Cannot generate this prematch research projection after kickoff")
    records, sources = [], []
    for season, directory in sorted(season_directories(root).items()):
        manifest = json.loads((directory / "下载来源.json").read_text(encoding="utf-8"))
        source = next(item for item in manifest["files"] if Path(item["path"]).name == "E0.csv")
        observed = _time(source["downloaded_at"], "historical file downloaded_at")
        if observed > cutoff:
            raise ValueError("Historical source file was downloaded after the current cutoff")
        matches, report = load_directory(directory, season=season)
        if report["rejected_rows"] or report["duplicate_keys"]:
            raise ValueError(f"Unresolved historical source rows in {season}")
        selected = [r for r in matches if r.competition == "E0" and r.match_date < cutoff.date()]
        records.extend(selected)
        sources.append({"season": season, "path": str(directory / "E0.csv"),
                        "observed_at": observed.isoformat(), "n": len(selected)})
    current, inspection, supplements = load_current_context(context_dir, cutoff)
    records.extend(current)
    # Exactly one fit, using the already established .01/180-day settings.
    model = DateTeamGoalModel(.01, 180).fit(records, cutoff.date())
    rates = model.predict("football-data:E0:Man United", "football-data:E0:Bournemouth", "E0", date(2026, 10, 25))
    forecast = summarize(score_grid(*rates, rho=0))
    generated = datetime.now(timezone.utc)
    if generated >= datetime(2026, 10, 25, 14, tzinfo=timezone.utc):
        raise ValueError("Cannot generate this prematch research projection after kickoff")
    params = model.parameters()
    descriptive = [team_descriptive(records, team, season)
                   for team in ("Man United", "Bournemouth") for season in ("2025/26", "2026/27")]
    output = root / "outputs" / f"当前球队研究投影_{generated.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    track = {"id": f"statistical:{generated.isoformat()}",
             "fixture_id": "prematch:premier-league:2026-10-25:manchester-united:bournemouth",
             "generated_at": generated.isoformat(), "model": "existing_regularized_poisson_alpha.01_half_life180",
             "probabilities": list(forecast["one_x_two"].values()),
             "description": f"current date-level research extension; means={list(rates)}; league coverage incomplete; no lineup or market inputs"}
    missing_current = sum(row["full_time"] is None and date.fromisoformat(row["match_date"]) < cutoff.date()
                          for row in inspection["records"]) - len(supplements["records"])
    result = {"fit_count": 1, "parameter_searches": 0, "fit_input_cutoff": cutoff.isoformat(),
              "generated_at": generated.isoformat(), "training_matches": model.n_matches,
              "current_season_matches": len(current), "current_missing_other_league_results": missing_current,
              "source_inspection": str(context_dir / "inspection.json"), "supplements": str(context_dir / "supplements.json"),
              "sources": sources, "alias_map": ALIASES, "model_parameters": params, "forecast": forecast,
              "track": track, "descriptive_team_results": descriptive,
              "limits": ["This is a new current date-level research adapter, not promotion of the underlying retrospective estimator to timed/lineup-calibrated deployment.",
                         "The underlying estimator's historical scope warning is preserved in model_parameters; no historical first-publication times are reconstructed.",
                         "54 known current league results do not reconstruct all 70 first-seven-round results; missing matches are not zeros.",
                         "Both target teams have their own seven league results; opponents' recent strength can still be stale.",
                         "Injuries, player-specific impacts, rotation, European priorities and tactical changes are not quantified in this statistical line.",
                         "No retrospective future target result, market probability, LLM output or xG enters this fit; statistical calibration on current season not established."]}
    _write(output / "球队投影与来源.json", result)
    _write(output / "统计线路.json", track)
    _write(output / "LLM描述性球队资料.json", descriptive)
    return result, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args(argv)
    result, output = run_projection(args.root, args.context)
    print(json.dumps({"output_directory": str(output), "fit_count": result["fit_count"],
                      "training_matches": result["training_matches"], "current_matches": result["current_season_matches"],
                      "probabilities": result["track"]["probabilities"], "expected_goals": result["forecast"]["expected_goals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
