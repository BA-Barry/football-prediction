"""Same-provider weekly window/clock sensitivity, with no parameter search.

Source goals and state clocks stay exact; only substitutions are rounded.
Fixed targets keep H1 stoppage state carried into H2, but not stoppage goals.
This is viewed-season conditional-intensity research, not pre-match forecasting.
"""

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path

from .full_season_experiment import reconcile_football_data, score_matches
from .joint_stint_model import JointStintIntensityModel
from .rolling_player_experiment import compare_supported_week, rolling_windows
from .statsbomb_data import write_json
from .statsbomb_fixed_window import FixedWindowCropError, crop_fixed_window
from .timing_precision_experiment import SCENARIOS, TimingReplayError, measure_match
from .window_clock_comparison import compare_tracks


WINDOWS = ("full", "fixed45")
KINDS = ("team", "players")
TARGET_FROM = date(2016, 2, 1)
PLAYER_RIDGE = 100.0


def empty_score():
    return {"predictions": [], "skipped": [], "requested_matches": 0}


def append_score(total, score):
    for key in ("predictions", "skipped"):
        total[key].extend(score[key])
    total["requested_matches"] += score["requested_matches"]


def summarize_paths(paths):
    return {window: {clock: {
        "matches": len(matches),
        "stints": sum(len(m["stints"]) for m in matches),
        "goals": sum(sum(m["diagnostics"]["event_goals"]) for m in matches),
        "exposure_seconds": sum(s["duration_seconds"] for m in matches for s in m["stints"]),
        "actor_conflicts": sum(len(m["diagnostics"]["actor_conflicts"]) for m in matches),
        "goal_actor_conflicts": sum(c["is_goal"] for m in matches for c in m["diagnostics"]["actor_conflicts"]),
        "positive_exposure_players": len({p for m in matches for s in m["stints"]
            for key in ("home_players", "away_players") for p in s[key]}),
    } for clock, matches in by_clock.items()} for window, by_clock in paths.items()}


def parse_source(root, output):
    """One source match in memory at a time; preserve each legal replay."""
    source = root / "data/statsbomb/premier-league-2015-16"
    metadata = json.loads((source / "matches_2_27.json").read_text(encoding="utf-8"))
    if len({m["match_id"] for m in metadata}) != len(metadata):
        raise ValueError("Distinct original source match IDs required")
    paths = {window: {clock: [] for clock in SCENARIOS} for window in WINDOWS}
    records, source_events = [], 0
    for meta in sorted(metadata, key=lambda m: (m["match_date"], m["match_id"])):
        mid = meta["match_id"]
        events = json.loads((source / f"events_{mid}.json").read_text(encoding="utf-8"))
        source_events += len(events)
        lineups = json.loads((source / f"lineups_{mid}.json").read_text(encoding="utf-8"))
        record = {"source_match_id": mid, "match_date": meta["match_date"],
                  "source_failure": None, "clock_failures": {}, "fixed_window_failures": {}}
        try:
            measured = measure_match(meta, events, lineups)
        except TimingReplayError as error:
            record["source_failure"] = str(error)
            records.append(record)
            continue
        record["clock_failures"] = measured["failures"]
        for clock, match in measured["scenarios"].items():
            paths["full"][clock].append(match)
            try:
                paths["fixed45"][clock].append(crop_fixed_window(match))
            except FixedWindowCropError as error:
                record["fixed_window_failures"][clock] = str(error)
        records.append(record)
        if len(records) % 50 == 0:
            print(f"Window-clock source parse {len(records)}/{len(metadata)}", flush=True)
    by_id = {w: {c: {m["match_id"]: m for m in values} for c, values in clocks.items()}
             for w, clocks in paths.items()}
    exact_ids = sorted(set(by_id["full"]["exact"]) & set(by_id["fixed45"]["exact"]))
    common_ids = sorted(set.intersection(*(set(by_id[w][c]) for w in WINDOWS for c in SCENARIOS)))
    quality = {"source_matches": len(metadata), "source_events": source_events,
        "source_failures": [r for r in records if r["source_failure"]],
        "exact_window_eligible_match_ids": exact_ids, "clock_common_match_ids": common_ids,
        "exact_fixed_window_exclusions": [r for r in records if r["fixed_window_failures"].get("exact")],
        "clock_replay_exclusions": [r for r in records if r["clock_failures"]],
        "all_legal_paths": summarize_paths(paths),
        "cohorts": {"exact_window_eligible": {"clocks": ["exact"], "matches": len(exact_ids)},
                    "clock_common": {"clocks": list(SCENARIOS), "matches": len(common_ids)}},
        "limits": "shared cohorts may be selected; no actor-dependent censoring or repairs; full380 diagnostics not mixed with paired subset scores"}
    write_json(output / "逐场源与重放排除.json", records)
    write_json(output / "按窗口与时钟派生时段.json", paths)
    write_json(output / "数据与共同样本汇总.json", quality)
    independent = reconcile_football_data(paths["full"]["exact"],
        root / "data/football-data/2015-16/20261022_full_player/E0.csv")
    write_json(output / "独立全场比分核对.json", independent)
    return by_id, quality


def fixed_target_training_comparison(full_trained_scores, fixed_trained_scores):
    """Both training strategies scored on identical fixed goal/state paths."""
    clocks, result = list(full_trained_scores), {}
    indices = {"full_training": {c: {k: {p["match_id"]: p for p in full_trained_scores[c][k]["predictions"]}
        for k in KINDS} for c in clocks},
        "fixed_training": {c: {k: {p["match_id"]: p for p in fixed_trained_scores[c][k]["predictions"]}
        for k in KINDS} for c in clocks}}
    common_all = sorted(set.intersection(*(set(indices[t][c][k])
        for t in indices for c in clocks for k in KINDS)))
    for clock in clocks:
        paired = {}
        for training in indices:
            pair = {k: {"predictions": [indices[training][clock][k][mid] for mid in common_all]} for k in KINDS}
            paired[training] = compare_supported_week(pair["team"], pair["players"])
        per_kind = {}
        for kind in KINDS:
            first = {"predictions": [indices["full_training"][clock][kind][mid] for mid in common_all]}
            second = {"predictions": [indices["fixed_training"][clock][kind][mid] for mid in common_all]}
            comparison = compare_supported_week(first, second)
            per_kind[kind] = {"mean_fixed_training_minus_full_training_loss": comparison["mean_player_minus_team_loss"],
                "common_match_ids": common_all,
                "full_training_on_fixed_target": comparison.get("team_on_common"),
                "fixed_training_on_fixed_target": comparison.get("player_on_common")}
        first = paired["full_training"]["mean_player_minus_team_loss"]
        second = paired["fixed_training"]["mean_player_minus_team_loss"]
        result[clock] = {**paired, "per_kind_same_fixed_target": per_kind,
            "fixed_training_minus_full_training_player_increment": None if first is None or second is None else second - first}
    return {"common_match_ids": common_all, "by_clock": result,
        "interpretation": "both strategies scored on exactly the same fixed goal/state paths; isolates training-window sensitivity conditional on chosen cohort, not causal effect or pre-match accuracy"}


def fit_cohort(name, ids, clocks, by_id, output, progress):
    destination = output / name
    destination.mkdir()
    model_dir = destination / "models"
    model_dir.mkdir()
    paths = {w: {c: [by_id[w][c][mid] for mid in ids] for c in clocks} for w in WINDOWS}
    totals = {w: {c: {k: empty_score() for k in KINDS} for c in clocks} for w in WINDOWS}
    full_on_fixed = {c: {k: empty_score() for k in KINDS} for c in clocks}
    weeks, initial_fits = [], progress["actual_model_fits"]
    for cutoff, history, target in rolling_windows(paths["full"]["exact"], from_date=TARGET_FROM):
        history_ids, target_ids = [m["match_id"] for m in history], [m["match_id"] for m in target]
        print(f"{name} weekly fit {cutoff}: history={len(history_ids)}, target={len(target_ids)}", flush=True)
        week = {"cutoff": cutoff.isoformat(), "history_matches": len(history_ids),
            "target_matches": len(target_ids), "variants": {}}
        for clock in clocks:
            for window in WINDOWS:
                history_path = [by_id[window][clock][mid] for mid in history_ids]
                target_path = [by_id[window][clock][mid] for mid in target_ids]
                scores, diagnostics = {}, {}
                for kind in KINDS:
                    model = JointStintIntensityModel(include_players=kind == "players", response_kind="goals",
                        player_ridge=PLAYER_RIDGE, team_ridge=1.0, context_ridge=1.0).fit(history_path, cutoff)
                    progress["actual_model_fits"] += 1
                    filename = model_dir / f"{cutoff}_{window}_{clock}_{kind}.json"
                    model.save(filename)
                    restored = JointStintIntensityModel.load(filename)
                    if restored.parameters() != model.parameters():
                        raise ValueError("Saved parameters did not roundtrip")
                    score = score_matches(model, target_path)
                    if score["predictions"]:
                        witness_id = score["predictions"][0]["match_id"]
                        witness = next(m for m in target_path if m["match_id"] == witness_id)
                        if restored.predict_match(witness) != model.predict_match(witness):
                            raise ValueError("Saved model prediction did not roundtrip")
                    append_score(totals[window][clock][kind], score)
                    scores[kind] = score
                    diagnostics[kind] = {"optimizer": model.optimizer_diagnostics,
                        "known_players": len(model.players), "supported": len(score["predictions"]),
                        "skipped": score["skipped"], "saved_parameter_and_witness_prediction_roundtrip": True}
                    if window == "full":
                        fixed_target = [by_id["fixed45"][clock][mid] for mid in target_ids]
                        append_score(full_on_fixed[clock][kind], score_matches(model, fixed_target))
                week["variants"][f"{window}/{clock}"] = {"diagnostics": diagnostics,
                    "comparison": compare_supported_week(scores["team"], scores["players"])}
        weeks.append(week)
        progress["completed_weeks"].append({"cohort": name, "cutoff": cutoff.isoformat()})
        write_json(destination / "逐周拟合与覆盖.json", weeks)
    for window in WINDOWS:
        for clock in clocks:
            for kind in KINDS:
                write_json(destination / f"{window}_{clock}_{kind}_predictions.json", totals[window][clock][kind])
    for clock in clocks:
        for kind in KINDS:
            write_json(destination / f"full_training_fixed_target_{clock}_{kind}_predictions.json", full_on_fixed[clock][kind])
    result = {"actual_model_fits": progress["actual_model_fits"] - initial_fits,
        "active_weeks": len(weeks), "source_matches": len(ids), "path_summary": summarize_paths(paths),
        "coverage": {w: {c: {k: {"requested": totals[w][c][k]["requested_matches"],
            "supported": len(totals[w][c][k]["predictions"]), "skipped": len(totals[w][c][k]["skipped"])}
            for k in KINDS} for c in clocks} for w in WINDOWS},
        "target_and_clock_comparison": compare_tracks(totals),
        "same_fixed_target_training_window_comparison": fixed_target_training_comparison(full_on_fixed, totals["fixed45"]),
        "additional_cross_scoring_fits": 0}
    write_json(destination / "共同样本比较结果.json", result)
    return result


def report(result):
    quality = result["data_quality"]
    primary = result["cohorts"]["exact_window_eligible"]["target_and_clock_comparison"]["all_windows_clocks_common"]
    primary_row = primary["by_clock"]["exact"]
    lines = ["# 同源目标窗口与换人时间精度排查", "",
        "已按用户继续要求执行；固定参数，不新增新球员先验、市场融合、战意系数或xG。"
        "此处是已查看2015/16赛季的事后条件强度研究，不是赛前概率或新盲测。", "",
        f"本轮原秒级共同{len(primary['common_match_ids'])}场，球员−球队平均损失："
        f"完整目标{primary_row['full_player_minus_team_loss']:+.6f}，"
        f"固定窗口目标{primary_row['fixed_player_minus_team_loss']:+.6f}；两者均为小幅改善。"
        "另一个四时钟共同集合中，两种目标下四套假设的球员增量也都为负。"
        "这不能证明稳定提升，也不能排除Wyscout更复杂的时钟误差；"
        "目前不能仅用目标截断或本次取整机制解释2017/18的负面结果。", "",
        f"原始{quality['source_matches']}场、{quality['source_events']:,}条事件；完整路径进球"
        f"{quality['all_legal_paths']['full']['exact']['goals']}，独立全场比分核对另存。"
        f"本轮实际完成{result['actual_model_fits']}次拟合。", "",
        "## 数据与样本", "",
        f"原秒级两半可覆盖45分钟的共同集合{quality['cohorts']['exact_window_eligible']['matches']}场；"
        "完整时长和固定窗口双方都只用该集合训练和评分。"
        "原380场完整路径仍保存，不把其均分混入子集比较。", "",
        "真实Half End早于2700秒的比赛不能补暴露，逐场时长在排除明细。"
        "再要求exact／floor／nearest-half-up／ceil都能合法重放且覆盖窗口，形成单独时钟共同集合。"
        "某些换人取整跨过临时离返场或改变红牌归属时拒绝整场，不修补状态。"
        "两个集合分别训练、分别解释；选择并非随机，不能拿它们的不同均分直接评价进步。", "",
        "固定窗口是各半场[0,2700]秒，边界进球保留。按原进球ID和秒数裁目标，不按时长比例分球。"
        "H1补时比分／红牌与人员状态仍进入H2初态，不删后续状态。"
        "取整只改换人；其他事件、真实半场结束保持原秒数。同秒保持源事件索引约定。", "",
        "## 球员增量：同场、同路径内比较", "",
        "下表每行均是球员模型−球队模型的条件点过程损失，负数较好。"
        "每个集合内部再取两窗口、全部时钟及两模型总交集。完整与固定目标不同，"
        "其原始损失不能直接作准确率比较；增量变化也只是联合窗口敏感性。", "",
        "| 训练集合 | 源场数 | 共同较晚场数 | 时钟 | 完整时长球员增量 | 固定窗口球员增量 |",
        "|---|---:|---:|---|---:|---:|"]
    def value(number):
        return "—" if number is None else f"{number:+.6f}"
    for name, cohort in result["cohorts"].items():
        common = cohort["target_and_clock_comparison"]["all_windows_clocks_common"]
        for clock, row in common["by_clock"].items():
            lines.append(f"| {name} | {cohort['source_matches']} | {len(common['common_match_ids'])} | {clock} | "
                f"{value(row['full_player_minus_team_loss'])} | {value(row['fixed_player_minus_team_loss'])} |")
    lines += ["", "## 相同固定目标：分开检查训练窗口", "",
        "额外把完整时长训练的模型在固定窗口路径评分，不新增拟合。"
        "下表两种训练方案使用相同比赛、固定目标及路径，再比较球员增量，"
        "避免把训练与评分目标的改变混为一件事。", "",
        "| 训练集合 | 共同较晚场数 | 时钟 | 完整训练→固定目标球员增量 | 固定训练→固定目标球员增量 |",
        "|---|---:|---|---:|---:|"]
    for name, cohort in result["cohorts"].items():
        comparison = cohort["same_fixed_target_training_window_comparison"]
        for clock, row in comparison["by_clock"].items():
            lines.append(f"| {name} | {len(comparison['common_match_ids'])} | {clock} | "
                f"{value(row['full_training']['mean_player_minus_team_loss'])} | "
                f"{value(row['fixed_training']['mean_player_minus_team_loss'])} |")
    lines += ["", "## 覆盖与时钟限制", "",
        "请求／支持是较晚评分场数；来源进球和动作冲突是整季共同来源统计，"
        "包括较晚评分比赛，不是某一次训练集或94／89场共同评分集的统计。", "",
        "| 训练集合 | 窗口 | 时钟 | 较晚请求场数 | 已知球员支持 | 全季来源进球数 | 全季动作冲突 | 其中进球 |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, cohort in result["cohorts"].items():
        for window in WINDOWS:
            for clock, coverage in cohort["coverage"][window].items():
                q = cohort["path_summary"][window][clock]
                lines.append(f"| {name} | {window} | {clock} | {coverage['players']['requested']} | "
                    f"{coverage['players']['supported']} | {q['goals']} | {q['actor_conflicts']} | {q['goal_actor_conflicts']} |")
    lines += ["", "未知球员整场跳过，不能默认0；固定窗口首次获得正曝光的日期可能改变。"
        "动作冲突保留并量化，取整路径不是真实阵容，不删除射门或进球来改善结果。"
        "时钟假设的强度极差见结果JSON，是情景敏感性而非置信区间或误差上界。", "",
        "## 固定设置与解释边界", "",
        "2016-02-01起每周一扩展训练，训练日期严格早于周一，同周不更新；"
        "进球为唯一目标，player ridge100沿用既有较早验证选择，team/context ridge均1。"
        "未搜索超参数或按本轮结果选时钟；日期级次日可用是假设，历史首次公开时间未知。", "",
        "未计算抽样不确定性，拟合次数不等于独立样本数。"
        "优化器success仅代表其停止条件成功，不等于所有未缩放梯度均满足最严格阈值。"
        "本轮不能单独解释2017/18表现变差的原因，因为球员、年份、提供商、采集差异等仍在。"
        "球员项是正则化的条件关联，不是因果贡献；未接入正式赛前接口、市场融合或半全场。", "",
        "所有源与派生路径、排除、逐周参数、严格未知身份、每场预测和比较均在本目录。"
        "本轮新代码为原创适配与统计编排，复用既有SciPy联合模型；没有新下载、依赖安装或原数据改写。", "",
        "来源：[Hudl/StatsBomb Open Data](https://github.com/hudl/open-data)，"
        "[独立Football-Data 2015/16比分](https://www.football-data.co.uk/mmz4281/1516/E0.csv)。"
        "原始数据条款、下载记录留项目目录data，不随源码分发。", ""]
    return "\n".join(lines)


def run(root):
    root = Path(root).resolve()
    output = root / "outputs" / ("同源窗口与时钟对照_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    progress = {"actual_model_fits": 0, "completed_weeks": []}
    write_json(output / "实验设置.json", {
        "target_from": TARGET_FROM.isoformat(), "player_ridge": PLAYER_RIDGE, "team_ridge": 1.0, "context_ridge": 1.0,
        "selection_using_this_run": False, "clocks": list(SCENARIOS), "windows": list(WINDOWS),
        "fixed_target": "both true half ends >=2700 required; original goals <=2700 only; source full H1 state carried to H2",
        "cohorts": "exact fixed-window-eligible; separately all clocks legal and fixed-window-eligible",
        "rounding": "period-local substitutions only; floor, half-up, ceil; clip only at true Half End; original index breaks ties",
        "fit_cutoff": "Monday; strictly earlier calendar dates; no within-week results",
        "full_training_on_fixed_target": "extra scoring only, zero extra fits",
        "unknown_players": "whole-match reject, no zero prior", "status": "viewed-season exploratory diagnostics, not untouched test",
        "formal_prematch_probabilities_changed": False, "market_changed": False, "xg_generated": False})
    try:
        by_id, quality = parse_source(root, output)
        cohorts = {}
        for name, clocks, ids in (
            ("exact_window_eligible", ("exact",), quality["exact_window_eligible_match_ids"]),
            ("clock_common", SCENARIOS, quality["clock_common_match_ids"])):
            cohorts[name] = fit_cohort(name, ids, clocks, by_id, output, progress)
        result = {"actual_model_fits": progress["actual_model_fits"], "data_quality": quality, "cohorts": cohorts,
            "additional_cross_scoring_fits": 0, "formal_prematch_probabilities_changed": False}
        write_json(output / "同源窗口与时钟结果.json", result)
        (output / "同源窗口与时钟说明.md").write_text(report(result), encoding="utf-8")
        print(f"Window-clock experiment complete: {output}; actual fits={progress['actual_model_fits']}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {**progress, "error": f"{type(error).__name__}: {error}",
            "status": "incomplete; partial models are not a completed experiment"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--report-only", type=Path, help="Refresh an existing descriptive report; no fits or scoring")
    arguments = parser.parse_args()
    if arguments.report_only:
        result = json.loads((arguments.report_only / "同源窗口与时钟结果.json").read_text(encoding="utf-8"))
        (arguments.report_only / "同源窗口与时钟说明.md").write_text(report(result), encoding="utf-8")
        print(json.dumps({"report_directory": str(arguments.report_only), "new_fits": 0, "new_scoring": False}))
    else:
        run(arguments.root)
