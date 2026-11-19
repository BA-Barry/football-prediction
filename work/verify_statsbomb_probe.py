"""Independent source totals and one real temporary-off/substitution example."""
import json
from math import isclose
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
SOURCE = ROOT / "data/statsbomb/20261022_player_probe"
OUTPUT = ROOT / "outputs/球员在场数据接入_20261022T125043786815Z"


def seconds(event):
    hh, mm, ss = event["timestamp"].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def main():
    summary = json.loads((OUTPUT / "接入检查.json").read_text(encoding="utf-8"))
    matches = json.loads((SOURCE / "matches_9_281.json").read_text(encoding="utf-8"))
    raw_xg, raw_events = 0.0, 0
    raw_goals = sum(m["home_score"] + m["away_score"] for m in matches)
    for match in matches:
        path = SOURCE / f"events_{match['match_id']}.json"
        events = json.loads(path.read_text(encoding="utf-8"))
        raw_events += len(events)
        raw_xg += sum(e["shot"]["statsbomb_xg"] for e in events if e["type"]["name"] == "Shot")
    assert isclose(raw_xg, summary["total_xg"], abs_tol=1e-9)
    assert raw_goals == summary["total_goals"]
    assert raw_events == summary["source_event_count"]
    stints = json.loads((OUTPUT / "真实在场时段.json").read_text(encoding="utf-8"))
    first = [s for s in stints if s["statsbomb_match_id"] == 3895052]
    exposure = sum(s["duration_seconds"] for s in first)
    actual_player = sum(s["duration_seconds"] for s in first
                        if "statsbomb:player:32712" in s["home_players"])
    raw_first = json.loads((SOURCE / "events_3895052.json").read_text(encoding="utf-8"))
    half1 = max(seconds(e) for e in raw_first if e["period"] == 1 and e["type"]["name"] == "Half End")
    player_events = {e["type"]["name"]: e for e in raw_first
                     if e.get("player", {}).get("id") == 32712
                     and e["type"]["name"] in {"Player Off", "Player On", "Substitution"}}
    absence = seconds(player_events["Player On"]) - seconds(player_events["Player Off"])
    # The player returned, then was substituted 1.7 seconds later. Counting all
    # remaining match time as temporary absence would be incorrect.
    expected_player = half1 + seconds(player_events["Substitution"]) - absence
    assert isclose(exposure, 5931.906, abs_tol=1e-6)
    assert isclose(absence, 126.807, abs_tol=1e-6)
    assert isclose(actual_player, expected_player, abs_tol=1e-6)
    result = {"raw_event_reconciliation": raw_events, "raw_goal_reconciliation": raw_goals,
              "raw_xg_reconciliation": raw_xg, "first_match_clock_seconds": exposure,
              "frimpong_temporary_absence_seconds": absence,
              "frimpong_actual_on_pitch_seconds": actual_player,
              "frimpong_source_reconstructed_seconds": expected_player,
              "temporary_return_followed_by_substitution_distinguished": True,
              "model_fitted_or_modified": False}
    (OUTPUT / "独立总量与真实案例复核.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
