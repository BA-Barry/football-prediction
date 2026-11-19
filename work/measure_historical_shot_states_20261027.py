"""Measure saved 760-match historical shot score states; no downloads or fits."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_v2/src"))
from football_model.football_data import load_directory
from football_model.understat_shot_states import reconstruct_shot_states
from football_model.xg_history import join_understat


ARCHIVE = ROOT / "data/understat/roster_seasons_EPL_2023_2024_20261025"
HISTORY = ROOT / "data/understat/history_20261024T170717354684Z"
OUT = ROOT / "outputs/平局条件信息定位_20261027"
DIRECTORIES = {
    2023: ROOT / "data/football-data/2023-24/20261022T113804786593Z",
    2024: ROOT / "outputs/Football-Data_2024-25_个人使用_20261022T111235Z",
}


def read(path):
    return json.loads(path.read_bytes())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")


def main():
    selection = {row["understat_id"]: row for row in read(ARCHIVE / "比赛选择.json")["selected"]}
    counts, by_season, per_match, detailed = Counter(), defaultdict(Counter), [], []
    seen_global, invalid, examples = {}, [], defaultdict(list)
    for year, directory in DIRECTORIES.items():
        season = f"{year}/{str(year+1)[-2:]}"
        independent, parsed = load_directory(directory, season=season)
        independent_by_id = {row.match_id: row for row in independent}
        joined, join_report = join_understat(read(HISTORY / f"EPL_{year}.json")["dates"],
                                            independent, season, "E0")
        if len(joined) != 380 or parsed["rejected_rows"] or parsed["duplicate_keys"]:
            raise ValueError("Historical independent fixture source needs inspection")
        for match in sorted(joined, key=lambda row: (row.match_date, row.match_id)):
            uid = match.understat_id
            selected = selection[uid]
            if selected["match_id"] != match.match_id:
                raise ValueError("Saved selection and independent fixture identity differ")
            fd = independent_by_id[match.match_id]
            payload = read(ARCHIVE / "raw" / f"{uid}.json")
            identity = {"match_id": match.match_id, "understat_id": uid, "season": season,
                        "date": match.match_date.isoformat(), "home": match.home, "away": match.away}
            try:
                result = reconstruct_shot_states(payload["shots"],
                                                expected_full_time=fd.full_time, match_id=uid)
            except (ValueError, KeyError, TypeError) as error:
                invalid.append({**identity, "error": str(error)})
                counts["invalid_matches"] += 1
                continue
            local = Counter(result["counts"])
            local["matches"] = 1
            local["own_goal_events"] = result["own_goal_count"]
            local["goal_events_including_own_goals"] = sum(result["counted_goal_credit"])
            local["goal_accounting_conflict_matches"] = int(
                result["goal_accounting_consistent"] is False)
            local["minute_regression_matches"] = int(bool(result["source_minute_regressions"]))
            local["minute_regressions"] = len(result["source_minute_regressions"])
            local["same_minute_multiple_goal_matches"] = int(
                bool(result["same_minute_multiple_goal_groups"]))
            local["same_minute_multiple_goal_groups"] = len(result["same_minute_multiple_goal_groups"])
            for side in ("h", "a"):
                for event in payload["shots"][side]:
                    sid = str(event["id"])
                    local["cross_match_duplicate_shot_ids"] += int(
                        sid in seen_global and seen_global[sid] != uid)
                    seen_global[sid] = uid
                    local["source_final_score_field_mismatches"] += int(
                        (int(event["h_goals"]), int(event["a_goals"])) != fd.full_time)
                    local["minute_gt90_shots"] += int(int(event["minute"]) > 90)
            counts.update(local)
            by_season[season].update(local)
            summary = {**identity, "counts": dict(local),
                       "independent_full_time": list(fd.full_time),
                       "counted_goal_credit": result["counted_goal_credit"],
                       "goal_accounting_consistent": result["goal_accounting_consistent"],
                       "source_minute_regressions": result["source_minute_regressions"],
                       "same_minute_multiple_goal_groups": result["same_minute_multiple_goal_groups"]}
            per_match.append(summary)
            detailed.append({**identity, **result})
            for label, condition in (
                ("minute_regressions", result["source_minute_regressions"]),
                ("same_minute_goals", result["same_minute_multiple_goal_groups"]),
                ("own_goals", result["own_goal_count"])):
                if condition and len(examples[label]) < 3:
                    examples[label].append(summary)
            for row in result["shots"]:
                if (row["status"] == "early_same_minute_score_set"
                        and len(examples["early_ambiguous_score"]) < 3):
                    examples["early_ambiguous_score"].append({**identity, "shot": row})
                if (not row["score_is_unique"] and row["own_state_is_unique"]
                        and len(examples["ambiguous_score_known_own_state"]) < 3):
                    examples["ambiguous_score_known_own_state"].append({**identity, "shot": row})
    source_info = {
        "shots": str(ARCHIVE / "raw"), "selection": str(ARCHIVE / "比赛选择.json"),
        "league_indices": str(HISTORY), "independent_scores": {
            str(year): str(path) for year, path in DIRECTORIES.items()},
    }
    meaning = {
        "scope": "760 saved EPL matches in 2023/24 and 2024/25; research-only retrospective measurement.",
        "rule": "Only minute<45; smaller minutes before shot, equal minutes unordered; self goal excluded.",
        "clock_assumption": "Source minutes count across the whole match; no period or seconds. Early availability is conditional on this convention.",
        "unknown": "Every minute>=45 remains unknown, including late second-half shots; no ID/array-order timeline inference.",
        "own_goal": "OwnGoal credits opposite h_a; full-time score checks accounting only, never fills states.",
        "state_set": "Ambiguous score pairs may still imply one own leading/drawing/trailing state.",
        "xg": "Source raw per-shot xG preserved; no summing, chain reconstruction or NPxG substitution.",
        "availability": "Retrospective archive; original publication/revision timestamps unknown.",
        "not_measured": "No full-match exposure time, target-match features, state correction coefficients, training or current results.",
        "source_paths": source_info,
    }
    coverage = {**meaning, "counts": dict(counts),
                "by_season": {key: dict(value) for key, value in by_season.items()},
                "invalid_matches": invalid, "examples": dict(examples), "per_match": per_match}
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "射门比分状态覆盖.json", coverage)
    with gzip.open(OUT / "历史逐射门比分状态.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({**meaning, "counts": dict(counts), "matches": detailed}, handle,
                  ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    total = counts["shots"]
    available = counts["minute_lt45"]
    report = f"""# 历史射门比分状态重建测量

本轮只测量已有原始记录可支持什么，没有下载、训练或拟合状态修正系数，也没有查询当前比赛赛果。独立整场比分只核对进球记账，未填入任何射门前状态。

共处理 {counts['matches']} 场、{total:,} 条射门，解析失败 {len(invalid)} 场。2023/24 与 2024/25 英超各 380 场。逐场覆盖见[射门比分状态覆盖]({(OUT/'射门比分状态覆盖.json').as_posix()})，逐射门记录见[历史逐射门比分状态]({(OUT/'历史逐射门比分状态.json.gz').as_posix()})。

## 原字段语义与实际冲突

- h_goals/a_goals 在全部 {total:,} 条记录中都与独立终场比分相同，不能用作射门前比分。
- {counts['own_goal_events']} 条 OwnGoal 必须记给 h_a 的对侧。按该规则统计 Goal/OwnGoal，{counts['matches']-counts['goal_accounting_conflict_matches']}/{counts['matches']} 场与独立终场比分一致；这只能支持本批次记账完整性，不证明每条事件时刻正确。
- {counts['minute_regression_matches']} 场在原同侧数组中出现分钟回退，共 {counts['minute_regressions']} 次，提示半场补时与下半场分钟重叠的实际风险。原字段没有 period 或秒，不能按 minute、id 或源数组顺序构造全场时间线。
- {counts['same_minute_multiple_goal_matches']} 场出现同一 minute 标签的多个进球；{counts['shots_sharing_minute_with_other_goal']} 条射门与另一个进球共享 minute 标签。共享标签不一定发生在同一个实际分钟，尤其跨半场时；不能用 id 的大小确定先后。
- 同场重复 id 会由解析器明确报错；本次实际解析失败为 {len(invalid)}，跨场重复 id 为 {counts['cross_match_duplicate_shot_ids']}。

## 保守恢复范围

只解释 minute<45 的 {available:,} 条射门，占全部 {available/total:.2%}。这一可用性仍依赖来源采用全场累计分钟、下半场不会重新从 0 起计的语义。它不是官方 period 的替代，不能当作完整全场或真实时间暴露复现。

在该范围内，{counts['early_unique_score']:,} 条具有唯一射前比分，{counts['early_ambiguous_score']} 条保留同分钟可能比分集合。集合由更早分钟已发生的 Goal/OwnGoal 加上同分钟其他进球的所有可能前缀计数构成，当前 Goal/OwnGoal 自身不计入射前。

从本方角度看，{counts['early_unique_own_state']:,} 条可唯一判定领先／持平／落后，另有 {counts['early_ambiguous_own_state']} 条状态仍有歧义。比分不唯一但所有可能比分都处于同一领先或落后状态时，保留“比分未知、状态已知”的区别。

其余 {counts['minute_ge45_unknown']:,} 条射门均标为半场未识别／时钟重叠未知，即使分钟很大也不凭经验恢复。{counts['minute_gt90_shots']} 条 minute>90 的记录不用于估计实际补时长度或终场暴露。未知字段为 null，绝不当作 0:0 或持平。

## 学到什么与仍缺什么

已得到一个可复用的研究组件，能安全处理乌龙归属、自身进球排除、同分钟状态集合以及“比分不唯一但本方状态唯一”。这确认了历史源可以提供一部分条件信息，也量化了缺失 period 带来的覆盖限制。

原始每射门 xG 原样保留，没有相加、补射链聚合或替代来源 NPxG；不能把本测量的射门覆盖当成 xG 覆盖或整场攻防强度。历史文件的最初发布时间和修订仍未知。要恢复完整比分状态，需要独立可靠的半场／阶段标识或更精确事件时钟；本轮没有用 ID、数组经验顺序或最终比分代替这些信息。

本轮结果是数据可用性与语义测量，不是平局预测提升的证据，没有据此产生新的模型系数或更新现有预测。
"""
    (OUT / "射门比分状态测量说明.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": dict(counts),
                      "invalid_matches": len(invalid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
