"""One fixed-parameter fit for an explicitly sourced batch of future fixtures.

This research adapter records real observation/generation times. Date-only
results stay date-only; fixture kickoffs must already have been resolved with
time zones by the caller. No downloads, searches, LLM calls or tuning occur.
"""

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .current_projection import ALIASES, load_current_context, team_descriptive
from .date_model import DateTeamGoalModel
from .domain import Fixture
from .football_data import load_directory
from .native_experiment import season_directories
from .probability import score_grid, summarize
from .prospective import _text, _time


COMPETITIONS = {"E0": "E0", "premier-league": "E0", "Premier League": "E0"}
FIXTURE_FIELDS = ("match_id", "home", "away", "competition", "kickoff", "neutral")
MODEL_LABEL = "existing_regularized_poisson_alpha.01_half_life180"


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def _resolved_paths(root, context_dir, fixture_path):
    paths = tuple(Path(value).resolve() for value in (root, context_dir, fixture_path))
    if os.name == "nt" and any(path.drive.upper() != "G:" for path in paths):
        raise ValueError("Batch inputs, outputs and project root must stay on G:")
    return paths


def read_fixtures(path, as_of):
    """Read explicit identities/times; never interpret an unsourced local clock."""
    cutoff = _time(as_of, "fit cutoff")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("fixtures"), list) or not payload["fixtures"]:
        raise ValueError("Input requires a nonempty fixtures list")
    policy = payload.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("Batch policy must be an object")
    output, ids, games = [], set(), set()
    for row in payload["fixtures"]:
        if not isinstance(row, dict):
            raise ValueError("Each fixture must be an object")
        values = {field: _text(row.get(field), field)
                  for field in ("match_id", "home", "away", "competition")}
        fixture = Fixture(**values, kickoff=_time(row.get("kickoff"), "kickoff"),
                          neutral=row.get("neutral", False))
        if fixture.kickoff <= cutoff:
            raise ValueError("Batch contains a fixture that is not future at the fit cutoff")
        observed = _time(row.get("observed_at"), "fixture source observed_at")
        if observed > cutoff:
            raise ValueError("Fixture source was not observed by the fit cutoff")
        source_url = _text(row.get("source_url"), "fixture source_url")
        parsed_url = urlsplit(source_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError("Fixture source_url requires an HTTP(S) source")
        home_label = _text(row.get("home_source_label"), "home_source_label")
        away_label = _text(row.get("away_source_label"), "away_source_label")
        if home_label == away_label:
            raise ValueError("Fixture source teams must differ")
        if row.get("published_at") is not None and _time(row["published_at"], "fixture published_at") > cutoff:
            raise ValueError("Fixture source publication is after the fit cutoff")
        source = {key: value for key, value in row.items() if key not in FIXTURE_FIELDS}
        source["observed_at"] = observed.isoformat()
        # Both source labels and resolved labels are retained; no fuzzy matching.
        resolved_home, resolved_away = ALIASES.get(home_label), ALIASES.get(away_label)
        game_key = (COMPETITIONS.get(fixture.competition, fixture.competition), fixture.kickoff,
                    resolved_home or home_label, resolved_away or away_label)
        if fixture.match_id in ids or game_key in games:
            raise ValueError("Duplicate batch fixture identity or game")
        ids.add(fixture.match_id)
        games.add(game_key)
        output.append((fixture, source, resolved_home, resolved_away))
    return output, policy


def _history(root, cutoff):
    records, sources = [], []
    for season, directory in sorted(season_directories(root).items()):
        manifest = json.loads((directory / "下载来源.json").read_text(encoding="utf-8"))
        source = next(item for item in manifest["files"] if Path(item["path"]).name == "E0.csv")
        observed = _time(source["downloaded_at"], "historical file downloaded_at")
        if observed > cutoff:
            raise ValueError("Historical source file was downloaded after the fit cutoff")
        matches, report = load_directory(directory, season=season)
        if report["rejected_rows"] or report["duplicate_keys"]:
            raise ValueError(f"Unresolved historical source rows in {season}")
        selected = [row for row in matches if row.competition == "E0" and row.match_date < cutoff.date()]
        records.extend(selected)
        sources.append({"season": season, "path": str(directory / "E0.csv"),
                        "observed_at": observed.isoformat(), "n": len(selected)})
    return records, sources


def _missing_earlier_results(inspection, current, cutoff):
    """Count source fixtures still unfilled after explicit source supplements."""
    known = {(row.match_date, row.home_label, row.away_label) for row in current}
    pending = {
        (date.fromisoformat(row["match_date"]), ALIASES.get(row["home_label"], row["home_label"]),
         ALIASES.get(row["away_label"], row["away_label"]))
        for row in inspection["records"]
        if row["full_time"] is None and date.fromisoformat(row["match_date"]) < cutoff.date()
    }
    return len(pending - known)


def _fixture_data(fixture):
    return {field: getattr(fixture, field) for field in FIXTURE_FIELDS if field != "kickoff"} | {
        "kickoff": fixture.kickoff.isoformat()}


def run_batch_projection(root, context_dir, fixture_path):
    """Fit .01/180 once, keeping unsupported fixtures as explicit missing rows.

    Returns ``(result, output_directory)``. ``result['fixtures']`` contains the
    original Fixture identity, source, optional statistical ``track``, goal
    ``rates``, ``forecast``, full numerical ``score_distribution`` and raw team
    descriptions. It can be used directly to prepare distinct LLM input packets.
    """
    root, context_dir, fixture_path = _resolved_paths(root, context_dir, fixture_path)
    cutoff = datetime.now(timezone.utc)
    fixtures, policy = read_fixtures(fixture_path, cutoff)
    records, sources = _history(root, cutoff)
    current, inspection, supplements = load_current_context(context_dir, cutoff)
    records.extend(current)
    # This is a data refresh, not a search: all fixtures share exactly this fit.
    model = DateTeamGoalModel(.01, 180).fit(records, cutoff.date())
    cases = []
    for fixture, source, home, away in fixtures:
        team_labels = [team for team in (home, away) if team is not None]
        descriptions = [team_descriptive(records, team, season)
                        for team in team_labels for season in ("2025/26", "2026/27")]
        case = {"fixture": _fixture_data(fixture), "fixture_source": source,
                "resolved_team_labels": {"home": home, "away": away},
                "status": "missing", "rates": None, "forecast": None,
                "score_distribution": None, "track": None, "missing_reasons": {},
                "descriptive_team_results": descriptions}
        reason = None
        competition = COMPETITIONS.get(fixture.competition)
        if home is None or away is None:
            unresolved = [source[f"{side}_source_label"] for side, label in (("home", home), ("away", away))
                          if label is None]
            reason = "Unresolved explicit team aliases: " + ", ".join(unresolved)
        elif competition is None:
            reason = "Competition is outside this E0-only fitted research adapter"
        elif fixture.neutral:
            reason = "Neutral venue is unsupported by this fitted home-advantage estimator"
        elif {f"football-data:E0:{home}", f"football-data:E0:{away}"} - model.teams:
            reason = "Unseen team requires an explicitly estimated prior; no zero-effect substitution"
        elif fixture.kickoff <= datetime.now(timezone.utc):
            reason = "Kickoff elapsed during batch fit; no prematch forecast generated"
        if reason is None:
            rates = model.predict(f"football-data:E0:{home}", f"football-data:E0:{away}",
                                  competition, fixture.kickoff.date())
            grid = score_grid(*rates, rho=0)
            forecast = summarize(grid)
            generated = datetime.now(timezone.utc)
            if generated >= fixture.kickoff:
                reason = "Kickoff elapsed during projection; no prematch forecast retained"
            else:
                case.update(status="predicted", rates=dict(zip(("home", "away"), map(float, rates))),
                            forecast=forecast,
                            score_distribution={
                                "home_goal_axis": list(range(grid.grid.shape[0])),
                                "away_goal_axis": list(range(grid.grid.shape[1])),
                                "probabilities": grid.grid.tolist(),
                                "note": "Normalized numerical truncation; score_grid rejects omitted tails above 1e-8."},
                            track={"id": f"statistical:{fixture.match_id}:{generated.isoformat()}",
                                   "fixture_id": fixture.match_id, "generated_at": generated.isoformat(),
                                   "model": MODEL_LABEL,
                                   "probabilities": [forecast["one_x_two"][key] for key in ("home", "draw", "away")],
                                   "description": "Current date-level E0 research projection; one shared fixed-parameter fit; no lineup, market or LLM inputs"})
        if reason is not None:
            case["missing_reasons"]["statistical"] = reason
        cases.append(case)
    generated = datetime.now(timezone.utc)
    predicted = sum(case["status"] == "predicted" for case in cases)
    missing = _missing_earlier_results(inspection, current, cutoff)
    result = {
        "fit_count": 1, "parameter_searches": 0, "fit_input_cutoff": cutoff.isoformat(),
        "generated_at": generated.isoformat(), "training_matches": model.n_matches,
        "current_season_matches": len(current), "current_missing_reported_results": missing,
        "fixture_count": len(cases), "predicted_fixtures": predicted, "missing_fixtures": len(cases) - predicted,
        "fixture_input": str(fixture_path), "source_inspection": str(context_dir / "inspection.json"),
        "supplements": str(context_dir / "supplements.json"),
        "sources": sources, "current_source_rows": len(inspection["records"]),
        "supplement_rows": len(supplements["records"]), "alias_map": ALIASES,
        "model_parameters": model.parameters(), "policy": policy, "fixtures": cases,
        "limits": [
            "Current date-level research adapter, not a timed, lineup-confirmed or calibrated live deployment.",
            "The underlying historical estimator's scope warning is retained in model_parameters, not silently removed.",
            "Only actually observed results from strictly earlier UTC calendar dates enter this fit; no reconstructed publication or kickoff times.",
            "Missing result counts describe this source's earlier dated fixtures, not proof of globally complete coverage.",
            "Unknown aliases, unseen teams, unsupported competitions and neutral venues remain missing, not zero effects.",
            "Injuries, minutes, rotation, competition priorities, tactical changes and parameter uncertainty are not quantified here.",
            "No xG, market price, LLM output or future target result is used to fit; one shared .01/180 fit and no parameter search.",
            "A full-time score distribution does not identify a fitted joint half-time/full-time process.",
        ],
    }
    output = root / "outputs" / f"批量球队投影_{generated.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "批量球队投影与来源.json", result)
    _write(output / "逐场统计线路.json", [{"fixture": case["fixture"], "track": case["track"],
                                      "missing_reasons": case["missing_reasons"]} for case in cases])
    _write(output / "LLM描述性球队资料.json", [{"fixture": case["fixture"], "fixture_source": case["fixture_source"],
                                          "descriptive_team_results": case["descriptive_team_results"]}
                                         for case in cases])
    return result, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args(argv)
    result, output = run_batch_projection(args.root, args.context, args.fixtures)
    print(json.dumps({"output_directory": str(output), "fit_count": result["fit_count"],
                      "training_matches": result["training_matches"], "current_matches": result["current_season_matches"],
                      "fixture_count": result["fixture_count"], "predicted_fixtures": result["predicted_fixtures"],
                      "missing_fixtures": result["missing_fixtures"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
