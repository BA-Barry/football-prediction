"""Fixed-calendar forward validation of the existing one-parameter CMP policy.

Expanding earlier history, six non-overlapping target windows, unchanged raw
means. Seasons have already been viewed; this is not a new blind evaluation.
"""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .cmp_fit import fit_cmp_window
from .cmp_experiment import ARMS, DOMAINS, analyze, loss_vectors, research_tracks, write_csv
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, validate_prediction, write_json
from .rate_calibration_experiment import load_period
from .refresh_experiment import describe


TRAINING_START = "2024-08-01"
BOUNDARIES = ("2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01",
              "2026-02-01", "2026-05-01", "2026-07-01")
WINDOWS = tuple(zip(BOUNDARIES[:-1], BOUNDARIES[1:]))
RESULT_FILE = "CMP多窗口前推结果.json"
REPORT_FILE = "CMP多窗口前推说明.md"


def window_rows(rows, cutoff, end):
    """Partition by calendar only; future scores do not determine membership."""
    if (cutoff, end) not in WINDOWS:
        raise ValueError("Use the declared fixed calendar windows")
    ordered = sorted(rows, key=lambda r: (r["date"], r["match_id"]))
    if len({r["match_id"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate archived match identifier")
    for row in ordered:
        validate_prediction(row)
    training = [r for r in ordered if TRAINING_START <= r["date"] < cutoff]
    target = [r for r in ordered if cutoff <= r["date"] < end]
    if not training or not target:
        raise ValueError("Declared fold requires nonempty training and target; no skipped folds")
    return training, target


def loss_descriptives(tracks):
    per_loss = {arm: loss_vectors(tracks[arm]) for arm in ARMS}
    return {domain: {**describe(per_loss["cmp"][domain]-per_loss["original"][domain]),
        "n_negative": int(np.sum(per_loss["cmp"][domain] < per_loss["original"][domain])),
        "n_positive": int(np.sum(per_loss["cmp"][domain] > per_loss["original"][domain]))}
        for domain in DOMAINS}


def analyze_forward(tracks, *, repeats=REPEATS):
    result, aligned = analyze(tracks, repeats=repeats)
    result["paired_loss_descriptives"] = loss_descriptives(aligned)
    return result, aligned


def write_histogram(path, tracks, *, candidate="cmp"):
    """Raw paired-loss distributions, all outcomes retained; not a test."""
    ARMS = ("original", candidate)
    per_loss = {arm: loss_vectors(tracks[arm]) for arm in ARMS}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" viewBox="0 0 1000 680">',
             '<rect width="1000" height="680" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="30" y="30" font-size="18">Paired loss differences: {escape(candidate.upper())} minus original (negative is better)</text>']
    for index, domain in enumerate(DOMAINS):
        v = per_loss[candidate][domain]-per_loss["original"][domain]
        counts, edges = np.histogram(v, bins=30)
        x, y = 45+(index%2)*495, 90+(index//2)*305
        peak = max(int(counts.max()), 1)
        width, height = 420, 190
        for i, count in enumerate(counts):
            bar = height*count/peak
            color = "#2171a5" if (edges[i]+edges[i+1])/2 < 0 else "#d98031"
            parts.append(f'<rect x="{x+i*width/len(counts):.3f}" y="{y+height-bar:.3f}" width="{width/len(counts)-1:.3f}" height="{bar:.3f}" fill="{color}"/>')
        parts += [f'<text x="{x}" y="{y-20}" font-size="14">{escape(domain)}</text>',
                  f'<path d="M{x} {y} V{y+height} H{x+width}" fill="none" stroke="#777"/>',
                  f'<text x="{x}" y="{y+height+19}" font-size="12">{edges[0]:+.4f}</text>',
                  f'<text x="{x+width-65}" y="{y+height+19}" font-size="12">{edges[-1]:+.4f}</text>',
                  f'<text x="{x}" y="{y+height+42}" font-size="12">n={len(v)}, mean={np.mean(v):+.6f}, SD={np.std(v,ddof=1):.6f}, peak count={peak}</text>']
        if edges[0] <= 0 <= edges[-1]:
            zero = x+width*(0-edges[0])/(edges[-1]-edges[0])
            parts.append(f'<path d="M{zero:.3f} {y} V{y+height}" stroke="#111" stroke-dasharray="3 3"/>')
    parts.append('</g></svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def report(result):
    pooled = result["pooled"]
    labels = {"exact_score": "完整比分NLL（主目标）", "one_x_two": "胜平负LL",
              "total_goals_0_1_2_3_4plus": "总进球0/1/2/3/4+ LL", "away_minus_two_three_way": "客让两球竞彩LL"}
    lines = ["# CMP多时间窗口严格前推研究", "",
        f"实际完成{len(result['folds'])}次共享ν重估、{pooled['n']}场不重叠评分、0次新增球队拟合。"
        "每个窗口只有一个活跃ν，不是总共只估计一个标量；原始主客μ不变，正式模型未替换。", "",
        "## 固定设计与信息边界", "",
        "初始历史从2024-08-01开始，首6个月用于ν初始学习；2025年2/5/8/11月、2026年2/5月固定重估。"
        "用[2024-08-01,截止日)内全部原始时间外预测和真实比分等权拟合，ν在随后目标窗口内固定。"
        "最终目标到2026-07-01之前。没有按表现选择窗口、丢弃失败窗口、改变权重或扩大ν范围。"
        "旧球队模型仍180天／alpha=.01／28天更新，原数据资格规则保持；目标队列为既有可预测存档，并非全部赛程。", "",
        "来源CSV与存档按整季加载作核验，包含窗口之后的赛果；不声称从未打开未来文件。"
        "真正的训练接口只接收严格早于截止日的行，且保存ν后才生成该窗口研究预测。"
        "同日赛果不入ν训练。历史首次公开时点未知，严格更早日历日期可用仍是研究假设。"
        "这些赛季均已查看，本轮是探索性前推，不是新盲测。", "",
        "| 训练截止／目标窗口 | 训练场数 | 评分场数 | 目标实际日期 | ν | 完整比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for fold in result["folds"]:
        a, fit = fold["analysis"], fold["fit"]
        d = a["cmp_minus_original"]
        lines.append(f"| [{fold['cutoff']}, {fold['end_exclusive']}) | {fit['n_matches']} | {a['n']} | {a['first_date']}—{a['last_date']} | {fit['nu']:.6f} | "
            f"{d['exact_score']:+.6f} | {d['one_x_two']:+.6f} | {d['total_goals_0_1_2_3_4plus']:+.6f} | {d['away_minus_two_three_way']:+.6f} |")
    lines += ["", "Δ为CMP减原模型，负值更好。窗口包含休赛期，日历范围不等于实际有比赛日期；空周不伪造比赛。"
        "训练集相互重叠，所以六窗口不是六次独立实验，也不把胜出窗口数量当显著性。", "",
        "## 全体同场结果（逐场等权，不取六窗口等权）", "",
        "| 目标 | 原模型 | 前推CMP | Δ | 四周条件探索95%区间 |", "|---|---:|---:|---:|---|"]
    for domain in DOMAINS:
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        old, new = (pooled["scores"][arm][domain][field] for arm in ARMS)
        ci = pooled["bootstrap"]["4"]["differences"][domain]["interval_95"]
        interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {labels[domain]} | {old:.6f} | {new:.6f} | {new-old:+.6f} | {interval} |")
    lines += ["", "| 赛季目标部分 | 场次 | 完整比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |", "|---|---:|---:|---:|---:|---:|"]
    for season, item in result["by_season"].items():
        d = item["cmp_minus_original"]
        lines.append(f"| {season} | {item['n']} | {d['exact_score']:+.6f} | {d['one_x_two']:+.6f} | {d['total_goals_0_1_2_3_4plus']:+.6f} | {d['away_minus_two_three_way']:+.6f} |")
    primary_ci = pooled["bootstrap"]["4"]["differences"]["exact_score"]["interval_95"]
    primary_note = ("主目标完整比分的四周区间跨零，不能确认稳定改善。"
                    if primary_ci and primary_ci[0] <= 0 <= primary_ci[1]
                    else "主目标仍只提供条件探索证据，不能据此确认未来表现。")
    short_ci = pooled["bootstrap"]["2"]["differences"]["one_x_two"]["interval_95"]
    short_note = ("" if short_ci is None else
                  f"胜平负两周区间为[{short_ci[0]:+.6g}, {short_ci[1]:+.6g}]，"
                  + ("略跨零，不能只引用更有利的区块长度。" if short_ci[0] <= 0 <= short_ci[1]
                     else "应与4/8周结果共同解释。"))
    old_acc, new_acc = (pooled["scores"][arm]["one_x_two"]["accuracy"] for arm in ARMS)
    lines += ["", "## 本轮判断", "", primary_note + short_note,
        f"胜平负最高概率命中率为{old_acc:.4%}→{new_acc:.4%}；概率评分的点改善并不等于命中更多比赛。"
        "ν路径的数值稳定不等于收益稳定；季度重估频率本身也没有被比较或证明最优。"
        "全部结果作为研究候选保留，不替换正式模型，也不据此开出当前投注建议。"]
    lines += ["", "2024/25只含2月之后目标，2025/26为原1,742场完整可预测队列。不能把本轮合并成绩直接与上一轮1,742场年度成绩比较。"
        "结果JSON保留所有窗口、赛季、联赛、月份、市场同子集、Brier/RPS/准确率、均值/SD/中位数/分位数/极值及2/8周敏感性。"
        "逐场损失直方图保留所有观察，没有裁掉高比分或不利尾部；正负场数只是描述，不是独立二项检验。", "",
        "| 市场独立对照 | 同场数 | 市场胜平负LL | 原模型LL | CMP LL |", "|---|---:|---:|---:|---:|"]
    for market, item in pooled["markets"].items():
        if item["n"]:
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['original']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['cmp']['log_loss']:.6f} |")
    lines += ["",
        "## 不确定性与解释", "",
        "共享五联赛日历连续2/4/8周移动区块，4周为主；"
        f"{result['settings']['bootstrap_repeats']}重复，seed={SEED}。"
        "区间条件于这条已拟合ν路径和球队预测，不包含每次重新估计ν、方向选择或球队重训的不确定性；"
        "重叠训练与长期球队关联不能靠窗口数消除。相同预测下，跨窗口的非平稳性也限制区间的未来外推。"
        "这些是多目标探索区间，无同时覆盖保证，不作确认性显著或盈利宣称。", "",
        "ν变化只描述在给定μ和历史下更合适的分布集中程度，不直接代表球队实力或战意变化；也可能吸收均值错配。"
        "独立主客边际仍未学习进球相关性。竞彩客让两球让胜须净胜至少3球，净胜2球为让平，不是亚洲盘退款。", "",
        "市场继续独立对照，不融合。历史报价缺真实可交易开盘/T−24h信息，不据此计算收益或宣称低赔率有价值。"
        "球员可用名单、伤停、赛事重视程度、跨联赛连接与半全场过程仍未在此研究中解决。"
        "没有深度学习、强化学习或生产默认变更。", "",
        "实验设计技能使目标窗口固定不重叠、训练严格前推；统计分析技能使全部目标、尾部与条件区间保留。"
        "方法沿用[均值参数化CMP论文](https://arxiv.org/html/1606.03214)，未复制GPL项目源码。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"training_start": TRAINING_START, "windows": [list(pair) for pair in WINDOWS],
        "training_policy": "expanding_all_strictly_earlier_archived_forecasts_equal_match_weight",
        "nu_bounds": [.25, 4.], "active_nu_per_window": 1, "n_refits_planned": len(WINDOWS),
        "base_half_life_days": 180, "base_regularization": .01, "base_update_days": 28,
        "primary_domain": "exact_score", "bootstrap_repeats": repeats,
        "bootstrap_blocks_weeks": list(BLOCKS), "seed": SEED,
        "source_reference": str(parent), "previously_viewed_seasons": True,
        "production_api_changed": False, "raw_means_changed": False,
        "source_loading": "whole_seasons_validated_before_fit; fit_receives_strict_past_rows_only"}
    write_json(output / "实验设置.json", settings)
    active_cutoff = None
    try:
        sources, rows = {}, []
        for season in ("2024-25", "2025-26"):
            entries, sources[season] = load_period(root, parent, season)
            rows.extend(entries)
        tracks = {arm: [] for arm in ARMS}
        folds, expected, scored = [], set(), set()
        for row in rows:
            if BOUNDARIES[0] <= row["date"] < BOUNDARIES[-1]:
                expected.add(row["match_id"])
        for cutoff, end in WINDOWS:
            active_cutoff = cutoff
            training, target = window_rows(rows, cutoff, end)
            fit = fit_cmp_window(training, training_start=TRAINING_START, before_date=cutoff)
            folder = output / cutoff
            folder.mkdir()
            write_json(folder / "CMP参数.json", fit)
            # The fit never sees these target outcomes; nu is saved before use.
            generated = research_tracks(target, fit["nu"])
            for arm in ARMS:
                for row in generated[arm]:
                    row.update(cmp_training_start=TRAINING_START, cmp_fit_before_date=cutoff,
                               cmp_target_end_exclusive=end)
            for row in generated["cmp"]:
                if row["match_id"] in scored or not cutoff <= row["date"] < end:
                    raise ValueError("Each target identity must be scored exactly once in its own window")
                scored.add(row["match_id"])
            write_json(folder / "完整研究预测.json", generated)
            analysis, aligned = analyze_forward(generated, repeats=repeats)
            write_json(folder / "同场分析.json", analysis)
            for arm in ARMS:
                tracks[arm].extend(aligned[arm])
            folds.append({"cutoff": cutoff, "end_exclusive": end, "fit": fit, "analysis": analysis})
            print(f"fold {cutoff}: training={len(training)} target={len(target)} nu={fit['nu']:.9f} delta={analysis['cmp_minus_original']}", flush=True)
        if scored != expected:
            raise ValueError("Forward cohort must contain every declared archived target exactly once")
        pooled, aligned = analyze_forward(tracks, repeats=repeats)
        by_season = {}
        for season, start, end in (("2024-25_partial", "2024-08-01", "2025-07-01"),
                                   ("2025-26", "2025-08-01", "2026-07-01")):
            subset = {arm: [r for r in aligned[arm] if start <= r["date"] < end] for arm in ARMS}
            by_season[season], _ = analyze_forward(subset, repeats=repeats)
        nus = [f["fit"]["nu"] for f in folds]
        result = {"settings": settings, "folds": folds, "pooled": pooled, "by_season": by_season,
            "nu_path_descriptives": describe(nus), "sources": sources,
            "counts": {"new_dispersion_fits": len(folds), "new_team_fits": 0,
                "unique_scored_matches": len(scored), "research_predictions": pooled["n"],
                "warmup_only_forecast_ids": [r["match_id"] for r in rows if r["date"] < BOUNDARIES[0]],
                "reused_team_parameter_snapshots": sum(s["parameter_snapshots"] for s in sources.values())},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "elapsed_seconds": time.perf_counter()-started}}
        write_json(output / "完整前推预测.json", aligned)
        write_csv(output / "逐场前推概率.csv", aligned)
        write_histogram(output / "逐场损失分布.svg", aligned)
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
    output = args.output or args.root / "outputs" / f"CMP多窗口前推_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "n": result["pooled"]["n"],
        "nu_path": [f["fit"]["nu"] for f in result["folds"]],
        "pooled_differences": result["pooled"]["cmp_minus_original"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
