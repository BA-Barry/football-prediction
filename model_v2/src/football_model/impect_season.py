"""Inspect saved full-season IMPECT files without downloading or fitting."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .impect_pilot import independent_score
from .impect_stints import ImpectParseError, parse_match


TEAM_NAMES = {27: "FC Koln", 29: "Dortmund", 30: "Wolfsburg", 31: "Hoffenheim",
    32: "M'gladbach", 33: "Bayern Munich", 34: "Freiburg", 35: "Ein Frankfurt",
    36: "Darmstadt", 37: "RB Leipzig", 38: "Werder Bremen", 39: "Augsburg",
    41: "Leverkusen", 42: "Mainz", 46: "Stuttgart", 416: "Bochum",
    432: "Heidenheim", 446: "Union Berlin"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def inspect_match(metadata, record, score_path):
    """Keep source-quality reasons separate from any later model performance."""
    item = {"source_match_id": metadata["id"], "scheduled_date": metadata["scheduledDate"],
            "home_team_id": metadata["homeSquadId"], "away_team_id": metadata["awaySquadId"],
            "target_period": metadata["scheduledDate"][:10] >= "2024-01-01",
            "status": "download_incomplete", "training_eligible": False}
    if record is None or record.get("status") != "completed":
        return item, None
    files = {row["kind"]: row for row in record["files"]}
    if (record["match_id"] != metadata["id"] or set(files) != {"events", "lineups", "player_kpis"}
            or len(record["files"]) != 3):
        raise ValueError("Downloaded match record has conflicting identity or file kinds")
    events, lineups, kpis = (read(files[kind]["file"]) for kind in ("events", "lineups", "player_kpis"))
    item.update(events=len(events), source_files=files,
        raw_goal_events=sum(e.get("actionType") in {"GOAL", "OWN_GOAL"} for e in events),
        raw_red_events=sum(e.get("actionType") == "RED_CARD" for e in events),
        raw_own_goal_events=sum(e.get("actionType") == "OWN_GOAL" for e in events))
    try:
        score = independent_score(metadata, score_path, team_names=TEAM_NAMES)
    except (KeyError, ValueError) as error:
        item.update(status="independent_score_join_failure", error=str(error))
        return item, None
    item["independent_score"] = score
    try:
        match = parse_match(metadata, events, lineups, kpis, expected_score=score["full_time"])
    except ImpectParseError as error:
        item.update(status="parse_failure", error=str(error))
        return item, None
    halftime = [sum(s[f"{side}_goals"] for s in match["stints"] if s["period"] == 1)
                for side in ("home", "away")]
    if halftime != score["half_time"]:
        item.update(status="half_time_conflict", event_half_time=halftime)
        return item, None
    audit = match["diagnostics"]["player_duration_audit"]
    item.update(stints=len(match["stints"]),
        total_seconds=match["diagnostics"]["total_duration_seconds"],
        goals=match["diagnostics"]["event_goals"], event_half_time=halftime,
        status_events=match["diagnostics"]["status_events"],
        clock_anomalies=match["diagnostics"]["all_source_event_clock_anomalies"],
        duration_compared_players=audit["compared_players"],
        duration_missing_active=audit["missing_active_player_ids"],
        max_duration_difference_seconds=audit["max_absolute_difference_seconds"])
    if audit["missing_active_player_ids"]:
        item.update(status="active_duration_source_missing")
        return item, None
    if audit["max_absolute_difference_seconds"] is None or audit["max_absolute_difference_seconds"] > 1e-6:
        item.update(status="duration_source_conflict", duration_comparisons=audit["player_comparisons"])
        return item, None
    item.update(status="eligible", training_eligible=True)
    return item, match


def run(root, source):
    root, source = Path(root), Path(source)
    now, timer = datetime.now(timezone.utc), perf_counter()
    selection = read(source / "selection.json")
    catalogue = read(root / "data/impect/feasibility_20261023/matches_743.json")
    selected = selection["selected_matches"]
    catalogue_by_id = {r["id"]: r for r in catalogue}
    if (len(catalogue_by_id) != len(catalogue) or len(selected) != len(catalogue)
            or {r["id"] for r in selected} != set(catalogue_by_id)):
        raise ValueError("Season selection must cover the distinct existing full catalogue")
    if any(row != catalogue_by_id[row["id"]] for row in selected):
        raise ValueError("Saved season selection and source catalogue disagree")
    score_path = root / "data/football-data/2023-24/20261022T113804786593Z/D1.csv"
    measurements, parsed = [], []
    for index, metadata in enumerate(sorted(catalogue, key=lambda r: (r["scheduledDate"], r["id"])), 1):
        record_path = source / "matches" / f"{metadata['id']}.json"
        record = read(record_path) if record_path.exists() else None
        item, match = inspect_match(metadata, record, score_path)
        measurements.append(item)
        if match is not None:
            parsed.append(match)
        if index % 25 == 0 or index == len(catalogue):
            print(f"IMPECT inspect {index}/{len(catalogue)}: eligible {len(parsed)}, "
                  f"status {dict(Counter(r['status'] for r in measurements))}", flush=True)
    output = root / "outputs" / now.strftime("IMPECT整季接通_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    problems = [r for r in measurements if not r["training_eligible"]]
    result = {"created_at_utc": now.isoformat(), "source_directory": str(source),
        "selection": selection, "expected_catalogue_matches": len(catalogue),
        "expected_target_matches": sum(r["target_period"] for r in measurements),
        "eligible_matches": len(parsed),
        "eligible_target_matches": sum(r["training_eligible"] and r["target_period"] for r in measurements),
        "statuses": dict(Counter(r["status"] for r in measurements)), "matches": measurements,
        "problem_count_by_month": dict(Counter(r["scheduled_date"][:7] for r in problems)),
        "problem_count_by_team": dict(Counter(str(r[k]) for r in problems for k in ("home_team_id", "away_team_id"))),
        "raw_red_card_matches": sum(r.get("raw_red_events", 0) > 0 for r in measurements),
        "eligible_red_card_matches": sum(r["training_eligible"] and r.get("raw_red_events", 0) > 0 for r in measurements),
        "raw_own_goal_matches": sum(r.get("raw_own_goal_events", 0) > 0 for r in measurements),
        "eligible_own_goal_matches": sum(r["training_eligible"] and r.get("raw_own_goal_events", 0) > 0 for r in measurements),
        "stints": sum(len(r["stints"]) for r in parsed),
        "goals_in_eligible": sum(sum(r["diagnostics"]["event_goals"]) for r in parsed),
        "duration_consistency": "same-provider cross-file consistency, not independent clock truth; no source repair",
        "eligibility": "parseable actual paths + independent FT/HT scores + all active duration sources agree within 1e-6 seconds floating noise",
        "source_score_file": str(score_path), "explicit_team_mapping": TEAM_NAMES,
        "elapsed_seconds": perf_counter() - timer,
        "limits": ["Revised historical data; first publication and historical pre-match availability unknown",
            "Excluded matches may be nonrandom; retain full catalogue and target denominator",
            "Retrospective actual-state paths only; no pre-match 1X2, xG, causal injury values or modern player mapping",
            "Local noncommercial research under Impect terms; no raw redistribution"],
        "fits": 0, "parameter_searches": 0, "new_forecasts": 0}
    write(output / "inspection.json", result)
    write(output / "parsed_matches.json", parsed)
    write(output / "unresolved_matches.json", problems)
    lines = ["# IMPECT整季真实数据接通", "", f"目录{len(catalogue)}场，可信{len(parsed)}场；目标原始{result['expected_target_matches']}场，可信{result['eligible_target_matches']}场。",
        f"逐项状态：{result['statuses']}。", "不要求零异常才执行可信子集研究；排除可能非随机，不能冒称完整赛季训练。",
        f"红牌比赛覆盖{result['eligible_red_card_matches']}/{result['raw_red_card_matches']}；乌龙比赛覆盖{result['eligible_own_goal_matches']}/{result['raw_own_goal_matches']}。",
        "同源时长仅一致性旁证；全场/半场比分来自独立Football-Data。原始资料不修改，不依后续损失选去留。", "",
        "| 比赛ID | 日期 | 状态 | 原因 |", "|---|---|---|---|"]
    lines += [f"| {r['source_match_id']} | {r['scheduled_date'][:10]} | {r['status']} | {r.get('error', '')} |" for r in problems]
    with (output / "整季接通说明.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"output_directory": str(output), "eligible": len(parsed), "expected": len(catalogue),
            "eligible_targets": result["eligible_target_matches"], "statuses": result["statuses"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.source), ensure_ascii=False))


if __name__ == "__main__":
    main()
