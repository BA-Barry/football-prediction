"""Weekly goal-only sensitivity fits on explicit fixed Wyscout windows.

The user's chosen research target is each half's [0,2700] clock interval.
These are retrospective conditional intensities, never full-match forecasts.
Three minute-clock assumptions are all retained, without test-driven selection.
"""

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path

from .full_season_experiment import paired_comparison, score_matches, summarize
from .joint_stint_model import JointStintIntensityModel
from .rolling_player_experiment import compare_supported_week, rolling_windows
from .statsbomb_data import write_json
from .wyscout_data import iter_match_events
from .wyscout_fixed_window import parse_match


SCENARIOS = ("early", "middle", "late")
TARGET_FROM = date(2018, 2, 1)
PLAYER_RIDGE = 100.0


def common_scenario_comparison(scores):
    """Same matches across all assumptions; not a clock selection function."""
    indices = {s: {k: {p["match_id"]: p for p in scores[s][k]["predictions"]}
                   for k in ("team", "players")} for s in SCENARIOS}
    common = sorted(set.intersection(*(set(indices[s][k]) for s in SCENARIOS for k in ("team", "players"))))
    if not common:
        return {"common_match_ids": [], "scenarios": {}, "intensity_ranges": [],
                "note": "no common supported matches; no zero coefficients or scores imputed"}
    compared, ranges = {}, []
    for s in SCENARIOS:
        matched = {k: {"predictions": [indices[s][k][mid] for mid in common]} for k in ("team", "players")}
        compared[s] = paired_comparison(matched["team"], matched["players"])
    for mid in common:
        predictions = [indices[s]["players"][mid] for s in SCENARIOS]
        actual = {(p["match_date"], p["actual_home_target"], p["actual_away_target"]) for p in predictions}
        if len(actual) != 1:
            raise ValueError("Clock scenarios must retain the same fixed-window goal target")
        totals = {s: indices[s]["players"][mid]["expected_home_target"] + indices[s]["players"][mid]["expected_away_target"]
                  for s in SCENARIOS}
        ranges.append({"match_id": mid, "conditional_integrated_intensity_by_clock": totals,
                       "range_width": max(totals.values()) - min(totals.values())})
    return {"common_match_ids": common, "scenarios": compared, "intensity_ranges": ranges,
        "mean_conditional_intensity_range": sum(r["range_width"] for r in ranges) / len(ranges),
        "max_conditional_intensity_range": max(r["range_width"] for r in ranges),
        "note": "Within each clock, same-path team/player increment; cross-clock spread is sensitivity, NOT a confidence interval or clock ranking"}


def quality_summary(paths, diagnostics):
    """Describe retained targets and lineup contradictions without changing them."""
    source_goals = sum(sum(r["diagnostics"].get("full_source_goals", [])) for r in diagnostics)
    source_window_goals = sum(sum(r["diagnostics"].get("window_goals", [])) for r in diagnostics)
    scenarios = {}
    for scenario, values in paths.items():
        conflicts = [c for m in values for c in m["diagnostics"]["actor_conflicts"]
                     if c["in_target_window"]]
        scenarios[scenario] = {
            "matches": len(values), "stints": sum(len(m["stints"]) for m in values),
            "window_goals": sum(r["home_goals"] + r["away_goals"] for m in values for r in m["stints"]),
            "window_seconds": sum(r["duration_seconds"] for m in values for r in m["stints"]),
            "retained_full_source_goals": sum(sum(m["diagnostics"]["full_source_goals"]) for m in values),
            "actor_conflicts_in_target_window": len(conflicts),
            "goal_actor_conflicts_in_target_window": sum(c["is_goal"] for c in conflicts),
            "matches_with_target_actor_conflict": sum(any(c["in_target_window"] for c in m["diagnostics"]["actor_conflicts"])
                                                      for m in values),
        }
    return {"source_matches": len(diagnostics), "source_primary_goals": source_goals,
        "source_window_goals": source_window_goals, "source_stoppage_goals_outside_target": source_goals - source_window_goals,
        "common_parse_matches": len(paths["early"]),
        "excluded_matches": [r for r in diagnostics if not r["common_supported"]],
        "scenario_summary": scenarios,
        "usage": "whole-match source exclusions and common-scenario cohort; first/last actor actions never define a censoring window"}


def parse_source(root, output):
    source = Path(root) / "data/wyscout/premier-league-2017-18"
    matches = json.loads((source / "matches_England.json").read_text(encoding="utf-8"))
    metadata = {m["wyId"]: m for m in matches}
    if len(metadata) != len(matches):
        raise ValueError("Duplicate original match IDs")
    players = json.loads((source / "players.json").read_text(encoding="utf-8"))
    names = {p["wyId"]: p.get("shortName") or str(p["wyId"]) for p in players}
    paths, diagnostics, seen = {s: [] for s in SCENARIOS}, [], set()
    for mid, events in iter_match_events(source / "events_England.json"):
        if mid not in metadata or mid in seen:
            raise ValueError("Unknown or duplicate source match")
        seen.add(mid)
        result = parse_match(metadata[mid], events, player_names=names)
        diagnostics.append({"source_match_id": mid, "match_date": metadata[mid]["dateutc"][:10],
            "common_supported": result["common_supported"], "rejects": result["rejects"],
            "diagnostics": result["diagnostics"]})
        if result["common_supported"]:
            for s in SCENARIOS:
                paths[s].append(result["scenarios"][s])
        if len(seen) % 50 == 0:
            print(f"Fixed-window parse {len(seen)}/{len(matches)}; common={len(paths['early'])}", flush=True)
    if seen != set(metadata):
        raise ValueError("Original match metadata lacks corresponding event groups")
    write_json(output / "固定窗口逐场准入与诊断.json", diagnostics)
    write_json(output / "固定90分钟派生时段.json", paths)
    quality = quality_summary(paths, diagnostics)
    write_json(output / "固定窗口数据质量汇总.json", quality)
    return paths, quality


def run(root):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"第二赛季固定窗口球员探索_{stamp}"
    output.mkdir(parents=True)
    model_dir = output / "models"
    model_dir.mkdir()
    write_json(output / "实验设置.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "user_chosen_target": "fixed half [0,2700], NOT full-match goals",
        "target_from": TARGET_FROM.isoformat(), "weekly_cutoff": "Monday, match_date strictly earlier; no within-week updates",
        "clock_scenarios": {"period": "minute<=45: 1H; else 2H, an assumption not verified source semantics",
            "early": "max(0,nominal-60)", "middle": "max(0,nominal-30)", "late": "nominal",
            "nominal": "minute*60 for 1H; (minute-45)*60 for 2H; no actor-derived relocation or window clipping"},
        "player_ridge": PLAYER_RIDGE, "penalty_origin": "fixed prior 2015/16 goal-validation choice; no 2017/18 tuning",
        "team_ridge": 1.0, "context_ridge": 1.0,
        "score_state": "full first-half goal flow including stoppage determines second-half score; only target/exposure masked",
        "unknown_players": "entire match skipped until observed in strictly earlier retained window; no zero prior",
        "identity": "Wyscout team/player IDs separate from StatsBomb; no cross-provider player coefficient transfer",
        "formal_pre_match_probabilities_changed": False, "xg_generated": False,
        "not_for": "full-match 1X2, held-out clock selection, confidence intervals, causal player values or current betting"})
    fits, weeks = 0, []
    try:
        paths, quality = parse_source(root, output)
        by_id = {s: {m["match_id"]: m for m in paths[s]} for s in SCENARIOS}
        totals = {s: {k: {"predictions": [], "skipped": [], "requested_matches": 0}
                      for k in ("team", "players")} for s in SCENARIOS}
        for cutoff, early_history, early_target in rolling_windows(paths["early"], from_date=TARGET_FROM):
            history_ids, target_ids = [m["match_id"] for m in early_history], [m["match_id"] for m in early_target]
            week = {"cutoff": cutoff.isoformat(), "history_matches": len(history_ids), "target_matches": len(target_ids), "scenarios": {}}
            print(f"Weekly fixed-window fit {cutoff}: history={len(history_ids)}, target={len(target_ids)}", flush=True)
            for scenario in SCENARIOS:
                history, target = [by_id[scenario][mid] for mid in history_ids], [by_id[scenario][mid] for mid in target_ids]
                scores, models = {}, {}
                for kind, players in (("team", False), ("players", True)):
                    model = JointStintIntensityModel(include_players=players, response_kind="goals", player_ridge=PLAYER_RIDGE).fit(history, cutoff)
                    fits += 1
                    path = model_dir / f"{cutoff}_{scenario}_{kind}.json"
                    model.save(path)
                    scores[kind], models[kind] = score_matches(model, target), model
                    for key in ("predictions", "skipped"):
                        totals[scenario][kind][key].extend(scores[kind][key])
                    totals[scenario][kind]["requested_matches"] += len(target)
                restored = JointStintIntensityModel.load(model_dir / f"{cutoff}_{scenario}_players.json")
                if restored.parameters() != models["players"].parameters():
                    raise ValueError("Saved model parameters do not roundtrip")
                if scores["players"]["predictions"]:
                    witness = next(m for m in target if m["match_id"] == scores["players"]["predictions"][0]["match_id"])
                    if restored.predict_match(witness) != models["players"].predict_match(witness):
                        raise ValueError("Saved model prediction does not roundtrip")
                week["scenarios"][scenario] = {"comparison": compare_supported_week(scores["team"], scores["players"]),
                    "known_players": len(models["players"].players), "player_skips": scores["players"]["skipped"],
                    "team_optimizer": models["team"].optimizer_diagnostics, "player_optimizer": models["players"].optimizer_diagnostics,
                    "parameters_and_prediction_roundtrip": True}
            weeks.append(week)
            write_json(output / "逐周拟合与覆盖.json", weeks)
        comparisons = {}
        for scenario in SCENARIOS:
            for kind in ("team", "players"):
                write_json(output / f"{scenario}_{kind}_predictions.json", totals[scenario][kind])
            comparisons[scenario] = compare_supported_week(totals[scenario]["team"], totals[scenario]["players"])
        result = {"actual_model_fits": fits, "active_weeks": len(weeks), "data_quality": quality,
            "same_path_comparison_per_scenario": comparisons,
            "all_scenarios_common": common_scenario_comparison(totals),
            "coverage": {s: {k: {"requested": totals[s][k]["requested_matches"], "supported": len(totals[s][k]["predictions"]),
                "skipped": len(totals[s][k]["skipped"])} for k in ("team", "players")} for s in SCENARIOS},
            "limits": "exploratory provider/clock/target shift; no untouched-test, significance, full-time probability, or motivation-factor claim"}
        write_json(output / "固定窗口探索结果.json", result)
        (output / "第二赛季球员探索说明.md").write_text(report(result), encoding="utf-8")
        print(f"Fixed-window exploration complete: {output}; actual fits={fits}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"error": f"{type(error).__name__}: {error}", "completed_model_fits": fits,
            "status": "incomplete; partial output is not a completed experiment"})
        raise


def report(result):
    q, c = result["data_quality"], result["all_scenarios_common"]
    lines = ["# 2017/18固定90分钟窗口：球员层探索", "",
        "用户选择：完整比分球队验证先行，球员层用每半场固定45分钟作探索对照。"
        "这不是完整比赛进球目标，也不是赛前1X2。补时只退出窗口目标，不从第一半场状态流删除。", "",
        f"原始{q['source_matches']}场；三套近似时钟共同可解析{q['common_parse_matches']}场，"
        f"排除{len(q['excluded_matches'])}场，逐场原因保留。实际完成{result['active_weeks']}个比赛周、"
        f"{result['actual_model_fits']}次拟合，每周一只用更早日期资料。", "",
        f"原始主进球事件{q['source_primary_goals']}球，固定窗口剩{q['source_window_goals']}球，"
        f"{q['source_stoppage_goals_outside_target']}个补时进球不属于本研究目标。"
        f"再排除{len(q['excluded_matches'])}场源问题比赛后，训练／评分来源共{q['scenario_summary']['early']['window_goals']}个窗口进球。"
        "独立比分表全季1018球，源事件少1球所在比赛整场排除，未补造。", "",
        "三套时钟只使用原整数分钟：<=45分视为上半场，其余视为下半场；"
        "名义秒点分别减60、减30、不减，最早为0。没有用后续球员动作反推秒点，也没有把换人裁到窗口终点。"
        "同秒先处理真实进球／红牌，后按名单源顺序换人；这是研究假设，不是已验证真实时钟。", "",
        "| 假设 | 较晚请求场数 | 严格已知球员可评分 | 同路径球员−球队平均损失 |",
        "|---|---:|---:|---:|"]
    for scenario in SCENARIOS:
        coverage, comp = result["coverage"][scenario]["players"], result["same_path_comparison_per_scenario"][scenario]
        delta = comp["mean_player_minus_team_loss"]
        lines.append(f"| {scenario} | {coverage['requested']} | {coverage['supported']} | {'—' if delta is None else f'{delta:+.6f}'} |")
    lines += ["", "损失为固定研究窗口的条件点过程负对数似然（相同常数省略），越低越好。"
        "每行仅比较同一假定路径、相同进球与相同比赛；不能拿完整比赛1X2的Log loss直接相比。"
        "未知球员整场跳过，不设零效果；球队仅比较模型也使用同路径与同场集合。", "",
        f"三套时钟及两类模型共同可评分{len(c['common_match_ids'])}场。"]
    if c["common_match_ids"]:
        lines += [f"同场球员模型的窗口积分强度平均极差{c['mean_conditional_intensity_range']:.6f}、"
            f"最大极差{c['max_conditional_intensity_range']:.6f}（进球单位）。这是情景敏感性，不是置信区间。", "",
            f"在三套时钟共同{len(c['common_match_ids'])}场上，球员−球队平均损失：" + "、".join(
                f"{s} {c['scenarios'][s]['mean_player_minus_team_loss']:+.6f}" for s in SCENARIOS) +
            "。正值表示加入球员后在本次样本上更差，不据此修改正则或挑选时钟。"]
    lines += ["", "## 假定阵容与真实动作的冲突", "",
        "下表为整季共同可解析集合的窗口内射门／进球者不在假定阵容中的次数。"
        "它们不是删除条件，也不等于真实时钟误差的完整数量；全部进球保留。", "",
        "| 时钟假设 | 有冲突比赛 | 射门／进球动作冲突 | 其中进球冲突 |",
        "|---|---:|---:|---:|"]
    for scenario in SCENARIOS:
        d = q["scenario_summary"][scenario]
        lines.append(f"| {scenario} | {d['matches_with_target_actor_conflict']} | {d['actor_conflicts_in_target_window']} | {d['goal_actor_conflicts_in_target_window']} |")
    lines += ["", "球员正则100沿用更早2015/16进球验证的选择，本赛季没有调参或选择表现最好的时钟。"
        "Wyscout与StatsBomb身份空间分开；跨赛季／跨提供商／固定窗口目标同时变化，不能把差异单独归因球员或年份。"
        "未计算抽样不确定性，不能宣称稳定提高。", "",
        "源缺失进球、窗口内矛盾红牌与观测尾不足的比赛明确排除；这不证明排除样本随机。"
        "射门／进球者与假定阵容的冲突保留在逐场诊断，不自动补入球员或删除其进球。"
        "第一半场补时进球仍改变下半场初始比分；红牌状态同理。", "",
        "本轮不生成xG，不调整市场权重、不新增新球员先验或战意系数，不改正式赛前概率。"
        "详细设置、全部共同派生时段、逐周参数／覆盖／未知身份与每场结果均留本目录。", "",
        "数据由Pappalardo/Massucco发布，源存档声明CC BY4.0："
        "[作者数据集合](https://figshare.com/collections/Soccer_match_event_dataset/4415000)。"
        "[socceraction历史时钟修复讨论](https://github.com/ML-KULeuven/socceraction/issues/228)"
        "用于理解转换约定，不作为公开分钟字段已具备秒级精度的证明；本适配器为原创代码。", ""]
    return "\n".join(lines)


def summarize_existing(output):
    """Refresh only descriptive quality/report fields; never refit or rescore."""
    output = Path(output)
    result = json.loads((output / "固定窗口探索结果.json").read_text(encoding="utf-8"))
    paths = json.loads((output / "固定90分钟派生时段.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "固定窗口逐场准入与诊断.json").read_text(encoding="utf-8"))
    result["data_quality"] = quality_summary(paths, diagnostics)
    write_json(output / "固定窗口数据质量汇总.json", result["data_quality"])
    write_json(output / "固定窗口探索结果.json", result)
    (output / "第二赛季球员探索说明.md").write_text(report(result), encoding="utf-8")
    return {"output_directory": str(output), "report_only": True, "new_model_fits": 0,
            "original_model_fits": result["actual_model_fits"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--summarize-existing", type=Path, help="Descriptive summaries only; no new fitting")
    arguments = parser.parse_args()
    if arguments.summarize_existing:
        print(json.dumps(summarize_existing(arguments.summarize_existing), ensure_ascii=False))
    else:
        run(arguments.root)
