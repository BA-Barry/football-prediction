"""Local, reproducible player-event integration; not a fitted player model.

Official StatsBomb raw JSON stays separate from the software distribution.
This command performs actual parsing and exact cross-provider reconciliation.
It never calls the training API with invented conditional offsets/timestamps.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median

from .football_data import LEAGUES
from .native_experiment import (VALID_START, VALID_END, TEST_START, TEST_END,
                                expected_target_ids, load_seasons, season_directories)
from .statsbomb_stints import StatsBombParseError, parse_match


# Explicit provider identity links for this specific Bundesliga study.
# No name similarity, date-only matching, or implicit integer-ID equivalence.
SB_TO_FD_D1 = {
    169: "Bayern Munich", 172: "Augsburg", 174: "Stuttgart", 175: "Hoffenheim",
    176: "Werder Bremen", 177: "Mainz", 179: "Wolfsburg", 180: "Dortmund",
    182: "RB Leipzig", 183: "Freiburg", 184: "Ein Frankfurt", 185: "M'gladbach",
    186: "FC Koln", 190: "Union Berlin", 201: "Heidenheim", 868: "Bochum",
    872: "Darmstadt", 904: "Leverkusen",
}


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_minutes_csv(path, rows):
    """Player-exposure schema, distinct from the forecast-specific CSV writer."""
    if not rows:
        raise ValueError("No parsed player rows to export")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconcile_match(metadata, records):
    """Exactly one same-date/home/away result, with independently equal score."""
    try:
        home = SB_TO_FD_D1[metadata["home_team"]["home_team_id"]]
        away = SB_TO_FD_D1[metadata["away_team"]["away_team_id"]]
    except KeyError as error:
        raise ValueError("No explicit StatsBomb to Football-Data D1 identity mapping") from error
    candidates = [r for r in records if r.competition == "D1" and r.season == "2023/24"
                  and r.match_date.isoformat() == metadata["match_date"]
                  and r.home_label == home and r.away_label == away]
    if len(candidates) != 1:
        raise ValueError(f"Expected one exact cross-provider match, found {len(candidates)}")
    found = candidates[0]
    if found.full_time != (metadata["home_score"], metadata["away_score"]):
        raise ValueError("Cross-provider FT score conflict; do not overwrite either source")
    return found


def coverage_simulation(records):
    """Identity/date coverage only: no goal fit, scoring or chosen update rule."""
    output = {}
    for season, start, end in (("2024/25", VALID_START, VALID_END),
                               ("2025/26", TEST_START, TEST_END)):
        detail, totals = {}, Counter()
        for competition in LEAGUES:
            league = [r for r in records if r.competition == competition]
            eligible = {r.match_id for r in league if start <= r.match_date < end
                        and not r.goal_fit_exclusion}
            ids28 = expected_target_ids(league, start, end, interval_days=28)
            ids7 = expected_target_ids(league, start, end, interval_days=7)
            detail[competition] = {
                "eligible_target_matches": len(eligible), "known_team_targets_28d": len(ids28),
                "known_team_targets_7d": len(ids7),
                "unknown_team_matches_28d": sorted(eligible - ids28),
                "unknown_team_matches_7d": sorted(eligible - ids7),
            }
            totals.update({key: detail[competition][key] for key in (
                "eligible_target_matches", "known_team_targets_28d", "known_team_targets_7d")})
        output[season] = {"totals": dict(totals), "leagues": detail}
    return {"not_model_performance": True, "model_or_prior_changed": False,
            "note": "Only past team identity availability; no fit/convergence/accuracy claim",
            "seasons": output}


def run_probe(root, data_directory=None):
    root = Path(root)
    data = Path(data_directory) if data_directory else root / "data/statsbomb/20261022_player_probe"
    metadata = json.loads((data / "matches_9_281.json").read_text(encoding="utf-8"))
    if not metadata or len({m["match_id"] for m in metadata}) != len(metadata):
        raise ValueError("Empty/duplicated metadata; do not multiply matches")
    records, _ = load_seasons(season_directories(root))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"球员在场数据接入_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    results, failures, joins, join_failures = [], [], [], []
    all_stints, all_minutes, source_files = [], [], []
    for meta in sorted(metadata, key=lambda m: (m["match_date"], m["match_id"])):
        mid = meta["match_id"]
        try:
            events_path, lineups_path = data / f"events_{mid}.json", data / f"lineups_{mid}.json"
            events = json.loads(events_path.read_text(encoding="utf-8"))
            lineups = json.loads(lineups_path.read_text(encoding="utf-8"))
            result = parse_match(meta, events, lineups)
        except (StatsBombParseError, OSError, json.JSONDecodeError) as error:
            failures.append({"source_match_id": mid, "error": str(error)})
            continue
        fd_id = None
        try:
            fd = reconcile_match(meta, records)
            fd_id = fd.match_id
            joins.append({"statsbomb_match_id": mid, "football_data_match_id": fd_id,
                          "date": meta["match_date"], "home": fd.home_label,
                          "away": fd.away_label, "score": list(fd.full_time)})
        except ValueError as error:
            join_failures.append({"source_match_id": mid, "error": str(error)})
        results.append(result)
        common = {"statsbomb_match_id": mid, "football_data_match_id": fd_id,
                  "match_date": meta["match_date"],
                  "home_team_id": result["home_team_id"], "away_team_id": result["away_team_id"]}
        all_stints.extend({**common, **row} for row in result["stints"])
        all_minutes.extend({**common, **row} for row in result["player_minutes"])
        source_files.extend(str(p) for p in (events_path, lineups_path))

    team_matches = Counter(team for m in metadata for team in (
        m["home_team"]["home_team_id"], m["away_team"]["away_team_id"]))
    players = {p["player_id"] for p in all_minutes if p["on_pitch_seconds"] > 0}
    seen_player_matches = Counter(p["player_id"] for p in all_minutes if p["on_pitch_seconds"] > 0)
    event_types = Counter()
    for r in results:
        event_types.update(r["diagnostics"]["event_types"])
    exposures = [r["diagnostics"]["total_duration_seconds"] / 60 for r in results]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_purpose": "retrospective parser integration, no player fit or probability changes",
        "source": "StatsBomb / Hudl Open Data", "source_directory": str(data),
        "source_metadata": str(data / "matches_9_281.json"),
        "source_terms": str(data / "LICENSE.pdf"), "source_files": source_files,
        "requested_matches": len(metadata), "parsed_matches": len(results),
        "parse_failures": failures, "exact_fd_matches": len(joins), "join_failures": join_failures,
        "date_min": min(m["match_date"] for m in metadata),
        "date_max": max(m["match_date"] for m in metadata),
        "source_event_count": sum(r["diagnostics"]["n_events"] for r in results),
        "stint_count": len(all_stints), "roster_match_rows": len(all_minutes),
        "unique_on_pitch_players": len(players),
        "players_with_only_one_observed_match": sum(n == 1 for n in seen_player_matches.values()),
        "published_match_share_of_306": len(metadata) / 306,
        "team_match_counts": {str(k): v for k, v in sorted(team_matches.items())},
        "event_types": dict(event_types),
        "match_clock_minutes_including_stoppage": {
            "n": len(exposures), "mean": mean(exposures) if exposures else None,
            "median": median(exposures) if exposures else None,
            "min": min(exposures) if exposures else None, "max": max(exposures) if exposures else None,
            "not_ball_in_play_time": True, "half_time_excluded": True},
        "total_goals": sum(s["home_goals"] + s["away_goals"] for s in all_stints),
        "total_xg": sum(s["home_xg"] + s["away_xg"] for s in all_stints),
        "source_first_available_at": None, "historical_lineup_published_at": None,
        "kickoff_timezone": None, "injury_status": "not inferred from absence or stoppages",
        "current_prediction_model_modified": False, "player_parameters_fitted": False,
        "limitations": ["All published games involve Leverkusen: selected star-shaped schedule",
            "Each other team has only two games; no complete-league player evaluation",
            "Actual target-match substitutions are outcomes, not pre-match covariates",
            "No verified historical official lineup announcement time",
            "No calibrated conditional team offsets: do not relabel marginal rates",
            "No causal or uncertainty claim from on/off totals",
            "Internal data/derived output only; no raw-data distribution in source archive"]}
    simulation = coverage_simulation(records)
    _json(output / "真实在场时段.json", all_stints)
    write_minutes_csv(output / "实际出场分钟.csv", all_minutes)
    _json(output / "跨来源比赛对应.json", joins)
    _json(output / "逐场解析诊断.json", [{k: r[k] for k in (
        "match_id", "source_match_id", "match_date", "diagnostics")} for r in results])
    _json(output / "接入检查.json", summary)
    _json(output / "更新频率覆盖模拟.json", simulation)
    lines = ["# 免费历史球员数据：真实接入进度", "",
        f"StatsBomb官方德甲2023/24公开表共{len(metadata)}场；本轮成功解析{len(results)}场，"
        f"与Football-Data日期、主客队及全场比分严格对应{len(joins)}场。",
        f"共{summary['source_event_count']:,}条事件，{len(all_stints)}个阵容／比分恒定时段，"
        f"{len(players)}名实际在场球员；解析失败{len(failures)}场，对应失败{len(join_failures)}场。", "",
        "这些全部是勒沃库森的34场比赛，仅覆盖完整德甲306场中的11.1%；其他球队各出现2场。"
        "这能验证数据链路，不能据此发布可靠的全联赛球员评分。", "",
        "已处理首发与未使用替补、换人、临时离返场、红牌／两黄红、比分状态及逐次射门xG。"
        "分半场保留补时、不计中场休息；分钟不是球在运动的净时间，也不硬压为90分钟。", "",
        "## 未更改的边界", "",
        "没有训练真实球员系数、没有把小样本套进正式概率，也未修改旧球队模型、赔率双轨或验证选择。"
        "当前StintObservation需要可靠的条件球队强度与历史信息时点；不能伪造UTC、首发公告或offset来满足接口。", "",
        "首发／换人来自赛后事件，不证明在赛前24小时已知。下载时间是今天，历史首次公开时间、"
        "首发公告时间和开球时区保留未知；未出场不等于伤停。", "",
        "## 升班球队：更新频率覆盖模拟", "",
        "这里只比较已有球队身份的覆盖，不拟合、不评分，不代表7天更新一定更准或优化已完成。", "",
        "|赛季|28天更新的可预测目标|7天更新的可预测目标|", "|---|---:|---:|"]
    for season, entry in simulation["seasons"].items():
        counts = entry["totals"]
        lines.append(f"|{season}|{counts['known_team_targets_28d']}|{counts['known_team_targets_7d']}|")
    lines += ["", "仍不认识的新球队不填默认实力。下一轮可独立验证更新频率；真正首次出现的球队，"
        "需从更早升班队估计明确先验或补次级联赛并校准强度。", "",
        "## 下一步", "",
        "继续免费历史路线，优先扩大到经核实完整的联赛事件赛季，再建立按历史窗口联合拟合的球队／球员模型。"
        "先与用户讨论目标响应使用进球还是xG，并保留两者对照；不能将xG拟合直接声称是经过校准的赛果概率。", "",
        "数据来源：StatsBomb / Hudl Open Data。原始数据和本地派生表仅作个人非商业研究，不随源码对外分发；"
        "若公开分析，须遵守来源署名及品牌标识要求。官方条款还请求使用者先表达研究兴趣，"
        "本轮未代用户注册或提交姓名邮箱。", "",
        "来源与用途记录见原始文件目录；可运行检查代码为项目player_probe与statsbomb_stints模块。"]
    (output / "球员数据接入说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "parsed": len(results), "stints": len(all_stints),
                      "exact_joins": len(joins), "failures": failures + join_failures}, ensure_ascii=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-directory", type=Path)
    args = parser.parse_args()
    run_probe(args.root, args.data_directory)
