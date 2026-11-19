"""Forward refitting of the two already implemented mean-scale calibrators.

No new score distribution, team features, parameter search or default change.
Both methods remain reported; targets never select a winning method.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .cmp_experiment import loss_vectors as checked_losses
from .cmp_forward_experiment import (TRAINING_START, BOUNDARIES, WINDOWS,
                                     window_rows, write_histogram)
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, write_json
from .rate_calibration import METHODS
from .rate_calibration_forward import fit_window, research_tracks
from .rate_calibration_experiment import ARMS, DOMAINS, analyze_later, load_period
from .refresh_experiment import describe


RESULT_FILE = "进球率尺度多窗口前推结果.json"
REPORT_FILE = "进球率尺度多窗口前推说明.md"
PARAMETER_FILE = "进球率尺度参数.json"
PRIMARY_METHOD = "side_scale"
PRIMARY_DOMAIN = "exact_score"


def analyze(tracks, statistics, *, repeats=REPEATS):
    # Reject zero observed probabilities before legacy metrics can apply their
    # generic numerical log-loss clipping. No rows or losses are repaired here.
    for arm in ARMS:
        checked_losses(tracks[arm])
    result, aligned, aligned_stats = analyze_later(tracks, statistics, repeats=repeats)
    vectors = {arm: checked_losses(aligned[arm]) for arm in ARMS}
    result["paired_loss_descriptives"] = {}
    for method in METHODS:
        result["paired_loss_descriptives"][method] = {}
        for domain in DOMAINS:
            delta = vectors[method][domain] - vectors["original"][domain]
            result["paired_loss_descriptives"][method][domain] = {**describe(delta),
                "n_negative": int(np.sum(delta < 0)), "n_positive": int(np.sum(delta > 0)),
                "n_zero": int(np.sum(delta == 0))}
    return result, aligned, aligned_stats


def write_csv(path, tracks):
    vectors = {arm: checked_losses(tracks[arm]) for arm in ARMS}
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals",
              "base_fit_before_date", "calibration_before_date", "base_home_rate", "base_away_rate"]
    fields += [f"{arm}_{key}" for arm in ARMS for key in
               ("home_rate", "away_rate", "home_scale", "away_scale", "p_home", "p_draw", "p_away",
                *(f"{domain}_nll" for domain in DOMAINS))]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(tracks["original"]):
            item = {key: row[key] for key in fields[:5]}
            item.update(zip(("home_goals", "away_goals"), row["actual_score"]))
            item.update(base_fit_before_date=row["fit_before_date"],
                        calibration_before_date=row["calibration_before_date"],
                        base_home_rate=row["base_rates"][0], base_away_rate=row["base_rates"][1])
            for arm in ARMS:
                entry = tracks[arm][i]
                item.update({f"{arm}_{side}_rate": entry["rates"][j] for j, side in enumerate(("home", "away"))})
                item.update({f"{arm}_{side}_scale": entry["calibration_scales"][j] for j, side in enumerate(("home", "away"))})
                item.update({f"{arm}_p_{side}": entry["probabilities"][j] for j, side in enumerate(("home", "draw", "away"))})
                item.update({f"{arm}_{domain}_nll": float(vectors[arm][domain][i]) for domain in DOMAINS})
            writer.writerow(item)


def _interval(analysis, method, domain, weeks=4):
    return analysis["bootstrap"][str(weeks)]["features"][f"{method}_minus_original_{domain}"]["interval_95"]


def report(result):
    pooled = result["pooled"]
    ordered = ("exact_score", *[d for d in DOMAINS if d != "exact_score"])
    labels = {"exact_score": "完整比分NLL（主目标）", "one_x_two": "胜平负LL",
              "total_goals_0_1_2_3_4plus": "总进球0/1/2/3/4+ LL", "away_minus_two_three_way": "客让两球竞彩LL"}
    lines = ["# 既有进球率尺度：固定窗口前推重估", "",
        f"完成6个固定窗口、{result['counts']['new_calibrator_fits']}次闭式校准拟合、"
        f"{pooled['n']}场唯一比赛、{result['counts']['new_research_forecasts']}条调整预测。"
        "每窗共享方案估计1个尺度、分侧方案估计2个尺度，共18个逐期标量估计，不是18个同时生效参数。"
        "0次新增球队拟合，正式默认未改，文件与临时缓存均在项目目录。", "",
        "## 为什么继续检验这一项", "",
        "v0.17只检验过较早全季学习后在下一赛季固定尺度，主客偏差随后反向，分侧修正反而恶化。"
        "本轮不发明新模型，而是检验同样两种既有修正定期重估能否改善；不是重复年度固定应用，也不因失败扩展结构。", "",
        "共享：c=Σ两侧实际进球/Σ两侧原μ；分侧：c主=Σ主进球/Σ主μ，c客同理。"
        "用训练窗内原始时间外预测与真实进球等权拟合，不继承球队180天权重，不做额外时间衰减或按联赛拟合。"
        "这是带logμ固定offset、仅学习截距的Poisson完整比分NLL闭式解；目标进球和为零时明确失败，不加epsilon。"
        "调整后的μ′=cμ会真正改变均值和Poisson方差；不是上一轮保持均值的分布实验，也不是单独修改胜平负概率。"
        "所有玩法从同一新比分分布派生，没有CMP、相关性或市场融合。", "",
        "数学表达参考[GLM offset定义](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)"
        "和[Poisson概率函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)。"
        "闭式比例由本项目的等权似然直接推导，未声称文档证明足球预测有效，也未引入statsmodels依赖。", "",
        "## 固定设计及信息边界", "",
        "沿用原180天/.01、28天更新球队存档与既有资格规则。训练从2024-08-01扩展，"
        "在2025年2/5/8/11月、2026年2/5月固定重估，参数只使用严格较早日期，保存后生成随后窗口预测。"
        "主比较预先为分侧方案−原样的完整比分NLL，共享方案为简单对照；两种方法全部报告，不按后段选赢家。"
        "目标截至2026-07-01之前，每场评分一次，不跳过失败窗口、不按表现改变训练长度。", "",
        "整季来源文件核验包含未来窗口赛果，不声称从未打开未来文件；拟合接口仅接收严格更早行。"
        "历史首次公开时点未知，以更早日期赛果次日可用为研究假设。两季已反复查看，仍是探索研究，不是新盲测。"
        "扩展训练窗口重叠，所以6窗口不是6次独立实验；队列为原模型可预测存档，不代表全部赛程。", "",
        "| 训练截止／目标窗口 | 训练场数 | 评分场数 | 共享尺度 | 分侧主尺度 | 分侧客尺度 |",
        "|---|---:|---:|---:|---:|---:|"]
    for fold in result["folds"]:
        p = fold["parameters"]
        shared = p["shared_scale"]["parameters"]["scales"]
        side = p["side_scale"]["parameters"]["scales"]
        lines.append(f"| [{fold['cutoff']}, {fold['end_exclusive']}) | {p['shared_scale']['n_matches']} | "
            f"{fold['analysis']['n']} | {shared[0]:.6f} | {side[0]:.6f} | {side[1]:.6f} |")
    lines += ["", "## 全体同场比较（逐场等权，损失越低越好）", "",
        "| 目标 | 原样 | 方法 | 调整后 | 新减旧 | 四周条件探索95%区间 |", "|---|---:|---|---:|---:|---|"]
    for domain in ordered:
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        old = pooled["scores"]["original"][domain][field]
        for method in METHODS:
            new = pooled["scores"][method][domain][field]
            ci = _interval(pooled, method, domain)
            interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
            lines.append(f"| {labels[domain]} | {old:.6f} | {method} | {new:.6f} | {new-old:+.6f} | {interval} |")
    lines += ["", "## 各窗与赛季目标（新减旧）", "",
        "| 目标窗口 | 方法 | 比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |", "|---|---|---:|---:|---:|---:|"]
    for fold in result["folds"]:
        for method in METHODS:
            d = fold["analysis"]["mean_differences"][f"{method}_minus_original"]
            lines.append(f"| {fold['cutoff']} | {method} | " + " | ".join(f"{d[k]:+.6f}" for k in ordered) + " |")
    lines += ["", "| 赛季目标部分 | 场数 | 方法 | 比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |", "|---|---:|---|---:|---:|---:|---:|"]
    for season, analysis in result["by_season"].items():
        for method in METHODS:
            d = analysis["mean_differences"][f"{method}_minus_original"]
            lines.append(f"| {season} | {analysis['n']} | {method} | " + " | ".join(f"{d[k]:+.6f}" for k in ordered) + " |")
    lines += ["", f"2024/25只含2月后的{result['by_season']['2024-25_partial']['n']}场原存档目标，"
        f"2025/26为原完整{result['by_season']['2025-26']['n']}场可预测队列；不把不同队列成绩直接横比。", "",
        "## 市场独立对照", "",
        "| 市场 | 同场数 | 市场胜平负LL | 原LL | 共享LL | 分侧LL |", "|---|---:|---:|---:|---:|---:|"]
    for market, item in pooled["markets"].items():
        if item["n"]:
            scores = item["arm_scores_on_same_matches"]
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                + " | ".join(f"{scores[arm]['log_loss']:.6f}" for arm in ARMS) + " |")
    lines += ["", "## 完整比分区间的区块长度敏感性", "",
        "| 方法 | 日历块长度 | 条件探索95%区间（新减旧） |", "|---|---:|---|"]
    for method in METHODS:
        for weeks in BLOCKS:
            interval = _interval(pooled, method, "exact_score", weeks)
            label = "不可估计" if interval is None else f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
            lines.append(f"| {method} | {weeks}周 | {label} |")
    ci = _interval(pooled, PRIMARY_METHOD, PRIMARY_DOMAIN)
    if ci is None:
        conclusion = "主比较完整比分的四周区间不可估计，不能据此确认提升。"
    elif ci[0] > 0:
        conclusion = "主比较完整比分NLL上升，四周条件区间全为正，本队列内的方向是恶化而非提升。"
    elif ci[1] < 0:
        conclusion = "主比较完整比分NLL下降，四周条件区间全为负，但这仍不能确认未来提升。"
    else:
        conclusion = "主比较完整比分的四周区间跨零，未确认稳定提升。"
    lines += ["", "## 结论及限制", "",
        conclusion
        + "训练目标下降只是拟合性质，不是预测成功；定期重估也未被证明优于其他频率。"
        "不把多方法、多目标的局部改善当自动采纳理由，正式默认保持原样。", "",
        "共享五联赛日历2/4/8周移动区块，4周为主，空周保留；"
        f"{result['settings']['bootstrap_repeats']}重复，seed={SEED}。"
        "区间条件于已拟合尺度路径和球队预测，不含尺度重估、结构选择、球队重训及全部长期依赖；"
        "跨赛季非平稳性仍限制外推。所有方法、目标和区块长度保留，不作多重检验后的确认性显著宣称。", "",
        "JSON保留联赛、月份、主客残差、事件计数、校准、Brier/RPS/准确率、市场同子集及均值/SD/中位数/分位数/极值。"
        "两张逐场损失直方图保留所有观察，没有裁掉高比分或不利尾部。完整比分NLL用原始调整μ的全支持log公式，"
        "有限网格遗漏质量另列，不把截断后的均值当模型μ。正负场数仅描述，不是独立二项检验。", "",
        "尺度只代表给定原μ下的历史整体偏差，不能替代球队攻防变化、球员影响、伤停或赛事重视程度建模。"
        "当前阵容、欧战跨联赛连接及半全场过程仍未完成，不能据此直接发布当前串关。"
        "竞彩客让两球让胜须净胜至少3球，净胜2球为让平，不是亚洲盘退款。"
        "历史报价不等于已核实开盘/T−24h可交易报价，低赔率不等于高价值。", "",
        "实验设计技能使本轮固定窗口、同场验证既有方法；统计分析技能使所有目标及条件不确定性完整保留。"
        "没有新增神经网络、强化学习、球队先验、依赖、hash、冻结contract或发布门禁。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"training_start": TRAINING_START, "windows": [list(pair) for pair in WINDOWS],
        "methods": list(METHODS), "primary_method": PRIMARY_METHOD, "primary_domain": PRIMARY_DOMAIN,
        "training_policy": "expanding_strict_past_equal_match_exact_score_NLL_closed_form",
        "active_parameters_per_method": {"shared_scale": 1, "side_scale": 2},
        "base_half_life_days": 180, "base_regularization": .01, "base_update_days": 28,
        "bootstrap_repeats": repeats, "bootstrap_blocks_weeks": list(BLOCKS), "seed": SEED,
        "source_reference": str(parent), "previously_viewed_seasons": True,
        "production_api_changed": False, "new_model_structure": False,
        "means_adjusted": True, "stacked_with_cmp_or_dependence": False,
        "source_loading": "whole_seasons_validated_before_fit; calibrator_receives_strict_past_rows_only"}
    write_json(output / "实验设置.json", settings)
    active_cutoff = None
    try:
        rows, sources = [], {}
        for season in ("2024-25", "2025-26"):
            entries, sources[season] = load_period(root, parent, season)
            rows.extend(entries)
        expected = {r["match_id"] for r in rows if BOUNDARIES[0] <= r["date"] < BOUNDARIES[-1]}
        tracks, statistics = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
        folds, scored = [], set()
        for cutoff, end in WINDOWS:
            active_cutoff = cutoff
            training, target = window_rows(rows, cutoff, end)
            parameters = fit_window(training, training_start=TRAINING_START, before_date=cutoff)
            folder = output / cutoff
            folder.mkdir()
            write_json(folder / PARAMETER_FILE, parameters)
            generated, stats, exported = research_tracks(target, parameters, before_date=cutoff)
            for arm in ARMS:
                for row in generated[arm]:
                    row["calibration_target_end_exclusive"] = end
            for row in generated["original"]:
                if row["match_id"] in scored or not cutoff <= row["date"] < end:
                    raise ValueError("Each target must be scored exactly once in its declared window")
                scored.add(row["match_id"])
            write_json(folder / "完整研究预测.json", generated)
            write_json(folder / "比分网格与来源.json", exported)
            analysis, aligned, aligned_stats = analyze(generated, stats, repeats=repeats)
            write_json(folder / "同场分析.json", analysis)
            for arm in ARMS:
                tracks[arm].extend(aligned[arm])
                statistics[arm].extend(aligned_stats[arm])
            folds.append({"cutoff": cutoff, "end_exclusive": end, "parameters": parameters, "analysis": analysis})
            print(f"fold {cutoff}: training={len(training)} target={len(target)} scales="
                f"{[parameters[m]['parameters']['scales'] for m in METHODS]} deltas={analysis['mean_differences']}", flush=True)
        if scored != expected:
            raise ValueError("Require the complete declared archived target cohort, no silent row loss")
        pooled, aligned, aligned_stats = analyze(tracks, statistics, repeats=repeats)
        by_season = {}
        for season, start, end in (("2024-25_partial", "2024-08-01", "2025-07-01"),
                                   ("2025-26", "2025-08-01", "2026-07-01")):
            indices = [i for i, r in enumerate(aligned["original"]) if start <= r["date"] < end]
            subset = {arm: [aligned[arm][i] for i in indices] for arm in ARMS}
            subset_stats = {arm: [aligned_stats[arm][i] for i in indices] for arm in ARMS}
            by_season[season], _, _ = analyze(subset, subset_stats, repeats=repeats)
        result = {"settings": settings, "folds": folds, "pooled": pooled, "by_season": by_season, "sources": sources,
            "counts": {"new_calibrator_fits": len(folds)*len(METHODS), "new_scalar_estimates": len(folds)*3,
                "new_team_fits": 0, "unique_scored_matches": len(scored),
                "new_research_forecasts": pooled["n"]*len(METHODS), "reused_original_forecasts": pooled["n"],
                "warmup_only_forecast_ids": [r["match_id"] for r in rows if r["date"] < BOUNDARIES[0]],
                "reused_team_parameter_snapshots": sum(s["parameter_snapshots"] for s in sources.values())},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "elapsed_seconds": time.perf_counter()-started}}
        write_json(output / "完整前推预测.json", aligned)
        write_csv(output / "逐场前推概率.csv", aligned)
        for method in METHODS:
            write_histogram(output / f"{method}_逐场损失分布.svg",
                            {arm: aligned[arm] for arm in ("original", method)}, candidate=method)
        write_json(output / RESULT_FILE, result)
        (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
        return result
    except Exception as exc:
        write_json(output / "研究失败.json", {"cutoff": active_cutoff, "type": type(exc).__name__,
            "message": str(exc), "history": getattr(exc, "history", None), "failed_no_fallback": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = args.parent or args.root / "outputs" / "更新频率研究_20261023T073047979805Z"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "outputs" / f"进球率尺度多窗口前推_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "n": result["pooled"]["n"],
        "fits": result["counts"]["new_calibrator_fits"], "differences": result["pooled"]["mean_differences"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
