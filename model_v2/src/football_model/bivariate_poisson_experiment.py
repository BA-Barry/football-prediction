"""Fixed-mean common-shock dependence, separate from CMP and production."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .bivariate_poisson import logpmf, score_grid as joint_grid
from .bivariate_poisson_fit import fit_window
from .cmp_experiment import DOMAINS, analyze as compare, loss_vectors, write_csv
from .cmp_forward_experiment import (TRAINING_START, BOUNDARIES, WINDOWS,
                                     window_rows, write_histogram)
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, validate_prediction, write_json
from .goal_rate_diagnostics import row_statistics
from .probability import score_grid, summarize
from .rate_calibration_experiment import load_period
from .refresh_experiment import describe
from .refresh_research import _same


CANDIDATE = "bivariate_poisson"
ARMS = ("original", CANDIDATE)
DELTA = CANDIDATE + "_minus_original"
RESULT_FILE = "主客进球相关性前推结果.json"
REPORT_FILE = "主客进球相关性前推说明.md"
PARAMETER_FILE = "相关参数.json"
METHOD_URL = "https://doi.org/10.1111/1467-9884.00366"


def research_tracks(rows, k):
    """Original archived means in, a new joint PMF out; never refit teams."""
    tracks = {arm: [] for arm in ARMS}
    for row in rows:
        validate_prediction(row)
        if "cmp_nu" in row or "dependence_k" in row:
            raise ValueError("Require original forecasts, no stacked research distributions")
        stats = row_statistics(row, row["rates"])
        _same(row["forecast"], summarize(score_grid(*row["rates"], rho=0)), "original_forecast")
        _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "original_probabilities")
        grid, metadata = joint_grid(*row["rates"], k)
        forecast = summarize(grid)
        nll = -logpmf(*row["actual_score"], *row["rates"], k)
        if not np.isfinite(nll):
            raise ValueError(f"Nonfinite full-support dependence NLL: {row['match_id']}")
        tracks["original"].append({**row, "exact_score_nll": stats["joint_score_nll"]})
        tracks[CANDIDATE].append({**row, "forecast": forecast,
            "probabilities": [forecast["one_x_two"][side] for side in ("home", "draw", "away")],
            "exact_score_nll": float(nll), "dependence_k": float(k), "dependence_numerics": metadata})
    return tracks


def event_summary(rows):
    """Use joint probabilities for 0:0 and 1:1, Poisson *marginal* zeros."""
    keys = ("home_zero", "away_zero", "draw", "zero_zero", "one_one", "total_four_plus")
    expected, actual = {key: [] for key in keys}, {key: [] for key in keys}
    covariances, correlations, residual_products = [], [], []
    for row in rows:
        h, a = row["actual_score"]
        mh, ma = row["rates"]
        k = row.get("dependence_k", 0.)
        c = k * min(mh, ma)
        ps = (np.exp(-mh), np.exp(-ma), row["probabilities"][1],
              np.exp(logpmf(0, 0, mh, ma, k)), np.exp(logpmf(1, 1, mh, ma, k)),
              row["forecast"]["total_goals"]["4+"])
        ys = (h == 0, a == 0, h == a, h == a == 0, h == a == 1, h+a >= 4)
        for key, p, y in zip(keys, ps, ys):
            expected[key].append(float(p))
            actual[key].append(bool(y))
        covariances.append(c)
        correlations.append(c / np.sqrt(mh*ma))
        residual_products.append((h-mh)*(a-ma))
    return {"events": {key: {"actual_count": int(sum(actual[key])),
        "expected_count": float(sum(expected[key])),
        "actual_minus_expected_rate": float(np.mean(actual[key])-np.mean(expected[key]))} for key in keys},
        "raw_goal_means": np.mean([r["rates"] for r in rows], axis=0).tolist(),
        "observed_goal_means": np.mean([r["actual_score"] for r in rows], axis=0).tolist(),
        "conditional_covariance": describe(covariances), "conditional_pearson_correlation": describe(correlations),
        "observed_residual_product": describe(residual_products),
        "residual_product_caveat": "May absorb conditional-mean misspecification; not raw-score Pearson correlation",
        "maximum_finite_grid_mean_error": float(max(abs(r["forecast"]["expected_goals"][side]-r["rates"][i])
            for r in rows for i, side in enumerate(("home", "away"))))}


def analyze(tracks, *, repeats=REPEATS):
    result, aligned = compare(tracks, repeats=repeats, candidate=CANDIDATE, event_summary_fn=event_summary)
    vectors = {arm: loss_vectors(aligned[arm]) for arm in ARMS}
    result["paired_loss_descriptives"] = {}
    for domain in DOMAINS:
        values = vectors[CANDIDATE][domain] - vectors["original"][domain]
        result["paired_loss_descriptives"][domain] = {**describe(values),
            "n_negative": int(np.sum(values < 0)), "n_positive": int(np.sum(values > 0)),
            "n_zero": int(np.sum(values == 0))}
    return result, aligned


def report(result):
    pooled = result["pooled"]
    labels = ("完整比分NLL（主目标）", "胜平负LL", "总进球0/1/2/3/4+ LL", "客让两球竞彩LL")
    lines = ["# 主客进球相关性：双变量泊松独立研究", "",
        f"实际完成{len(result['folds'])}次共享比例k重估、{pooled['n']}场不重叠评分、0次新增球队拟合。"
        "每个窗口仅一个活跃k，不是全程只拟合一个标量。没有叠加CMP，正式模型未替换。", "",
        "## 这次改变的是预测分布本身", "",
        "H=U+W，A=V+W，三个潜在变量独立。W均值c=k·min(μH,μA)，U和V均值分别为μH−c、μA−c。"
        "主客原始预期进球与各自Poisson边际保持不变，仅改变联合分布。k范围预先定为[0,0.95]；0就是原独立模型。"
        "共同分量是统计构造，不表示双方会同时进球，也不是比赛战意的因果解释。", "",
        "k不是Pearson相关系数：每场相关系数为c/√(μHμA)。本轮只检验非负相关的共同成分族，未检验负相关、"
        "Dixon–Coles、copula或全部相关性结构。若k为零或不改善，不能据此证明主客独立。"
        "总进球方差由μH+μA变为μH+μA+2c，净胜球方差变为μH+μA−2c；边际方差不变。"
        "所以它能改变平局、总进球及让两球概率，但不保证所有强弱组合的平局概率都增加。", "",
        f"方法参考[Karlis & Ntzoufras (2003)]({METHOD_URL})。固定原均值、共享比例k与本轮更新规则是本项目的研究设定，"
        "不是声称复现论文全部球队拟合或对角膨胀模型。未复制外部项目源码、未新增依赖。", "",
        "## 时间与拟合设计", "",
        "沿用原180天半衰期／alpha=.01／28天球队更新存档；不叠加前轮均值调参。"
        "历史从2024-08-01开始，固定2025年2/5/8/11月、2026年2/5月重估k，目标截至2026-07-01之前。"
        "每次只把严格更早日期的原始时间外预测及真实比分交给训练接口，逐场等权最小化全支持完整比分NLL；"
        "先保存参数再生成目标预测，不按目标成绩改窗口、范围或权重，不丢失败窗口。", "",
        "先固定扫描k=0,.05,…,.95，再细化局部候选及边界邻域，并比较确切两端点。固定均值后的似然不保证凹；"
        "这里只报告数值候选最小值，不宣称全局最优的数学认证。", "",
        "整季源CSV与存档加载含后续赛果，但训练接口只收到过去子集；不声称从未打开未来文件。"
        "首次公开时点未知，以严格较早日历日可用为研究假设。两赛季均已查看，这是探索性前推而非新盲测。"
        "原始可预测队列的资格规则保持，未把它说成全部五大联赛赛程。", "",
        "| 参数截止／目标窗口 | 训练场数 | 评分场数 | k | 边界最优 | 比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for fold in result["folds"]:
        fit, analysis = fold["fit"], fold["analysis"]
        d = analysis[DELTA]
        lines.append(f"| [{fold['cutoff']}, {fold['end_exclusive']}) | {fit['n_matches']} | {analysis['n']} | "
            f"{fit['k']:.6f} | {fit['at_boundary']} | " + " | ".join(f"{d[key]:+.6f}" for key in DOMAINS) + " |")
    lines += ["", "Δ为相关候选减原模型，负值较好。训练集相互重叠，六个窗口不是六次独立实验。", "",
        "## 全体同场成绩（逐场等权）", "",
        "| 目标 | 原模型 | 相关候选 | Δ | 四周条件探索95%区间 |", "|---|---:|---:|---:|---|"]
    for domain, label in zip(DOMAINS, labels):
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        old, new = (pooled["scores"][arm][domain][field] for arm in ARMS)
        ci = pooled["bootstrap"]["4"]["differences"][domain]["interval_95"]
        interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {label} | {old:.6f} | {new:.6f} | {new-old:+.6f} | {interval} |")
    lines += ["", "| 赛季目标部分 | 场数 | 比分Δ | 胜平负Δ | 总进球Δ | 让两球Δ |", "|---|---:|---:|---:|---:|---:|"]
    for season, item in result["by_season"].items():
        lines.append(f"| {season} | {item['n']} | " + " | ".join(f"{item[DELTA][d]:+.6f}" for d in DOMAINS) + " |")
    lines += ["", "2024/25只包含2月后目标；2025/26包含原完整可预测队列。合并成绩不与不同场数的年度成绩直接比较。", "",
        "| 事件 | 实际次数 | 原期望次数 | 相关候选期望次数 |", "|---|---:|---:|---:|"]
    for key, item in pooled["event_diagnostics"]["original"]["events"].items():
        candidate = pooled["event_diagnostics"][CANDIDATE]["events"][key]
        lines.append(f"| {key} | {item['actual_count']} | {item['expected_count']:.3f} | {candidate['expected_count']:.3f} |")
    old_acc, new_acc = (pooled["scores"][arm]["one_x_two"]["accuracy"] for arm in ARMS)
    lines += ["", f"胜平负最高概率命中率：{old_acc:.4%}→{new_acc:.4%}。概率评分与命中率不是同一目标。"
        "主客零进球的边际概率理论上不变，0:0和1:1使用真正联合概率，不再相乘冒充相关模型。", "",
        "## 市场继续单列，不融合", "",
        "| 市场 | 同场数 | 市场胜平负LL | 原LL | 相关候选LL |", "|---|---:|---:|---:|---:|"]
    for market, item in pooled["markets"].items():
        if item["n"]:
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['original']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches'][CANDIDATE]['log_loss']:.6f} |")
    primary = pooled["bootstrap"]["4"]["differences"]["exact_score"]["interval_95"]
    conclusion = ("主目标完整比分的四周区间跨零，未能确认稳定改善。" if primary and primary[0] <= 0 <= primary[1]
        else "主目标区间只提供条件探索证据，不能当作未来表现保证。")
    zero_folds = sum(f["fit"]["k"] == 0 for f in result["folds"])
    all_cross_zero = all(pooled["bootstrap"][str(w)]["differences"][d]["interval_95"] is not None
        and pooled["bootstrap"][str(w)]["differences"][d]["interval_95"][0] <= 0
        <= pooled["bootstrap"][str(w)]["differences"][d]["interval_95"][1] for w in BLOCKS for d in DOMAINS)
    lines += ["", "## 判断与局限", "",
        f"{zero_folds}个窗口选择确切k=0，这是显式比较独立零点后的拟合选择，不是求解失败回退；"
        "零参数窗口完整比分损失差约1e−17仅是两条等价公式的浮点求和差，不解释为改善或恶化。"
        + ("四个目标的2/4/8周条件探索区间全部跨零。" if all_cross_zero else "所有区块敏感性需一并解释。"), "",
        conclusion + "不因某一个次目标改善自动替换正式模型。"
        "k也可能吸收原始均值错配，不能解释为球队实力、赛事重视程度或足球固有规律。", "",
        "共享五联赛日历连续2/4/8周移动区块，4周为主，空周保留；"
        f"{result['settings']['bootstrap_repeats']}次，seed={SEED}。"
        "区间条件于已拟合k路径和存档球队预测，不包含k重估、模型方向选择和球队重训不确定性；"
        "长期球队相关、重叠训练与非平稳性限制外推。这是多目标探索区间，没有同时覆盖保证，不做确认性显著宣称。", "",
        "结果JSON保留所有联赛、月份、市场同子集、2/8周敏感性、Brier/RPS/准确率和逐场损失的均值/SD/中位数/分位数/极值。"
        "直方图不裁掉高比分及不利尾部，正负场数仅描述，不作独立二项检验。", "",
        "有限网格用Poisson边际并集尾界及矩界检查截断误差；截断后归一化均值仅数值近似原μ，"
        "每场实际误差与界单列。完整比分损失使用全支持有限和logpmf，真实高比分不因网格截断而删掉或填零。"
        "解析尾界不是浮点区间算术认证，也不是预测的统计置信区间。", "",
        "竞彩客让两球让胜须净胜至少3球，净胜2球为让平，不是亚洲盘退款。"
        "历史报价缺可靠逐时点开盘/T−24h信息，不能据此宣称可交易收益；低赔率不等于高价值。"
        "当前名单、伤停、战意、欧战跨联赛强度连接与半全场过程仍未解决，不据此推荐当前串关。"
        "没有神经网络、强化学习、新增生产门禁或生产默认变更。", "",
        "实验设计技能使本轮固定窗口、同场同均值比较；统计分析技能使全部目标和条件区间完整保留。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"training_start": TRAINING_START, "windows": [list(pair) for pair in WINDOWS],
        "family": "fixed_mean_bivariate_poisson_common_shock",
        "common_mean": "k * min(raw_home_mean, raw_away_mean)", "k_bounds": [0., .95],
        "objective_scan": [i/20 for i in range(20)], "active_k_per_window": 1,
        "n_refits_planned": len(WINDOWS), "training_policy": "strict_past_expanding_equal_match_exact_score_NLL",
        "base_half_life_days": 180, "base_regularization": .01, "base_update_days": 28,
        "primary_domain": "exact_score", "secondary_domains": list(DOMAINS[1:]),
        "bootstrap_repeats": repeats, "bootstrap_blocks_weeks": list(BLOCKS), "seed": SEED,
        "source_reference": str(parent), "method_reference": METHOD_URL, "previously_viewed_seasons": True,
        "production_api_changed": False, "raw_means_changed": False, "poisson_marginals_changed": False,
        "stacked_with_cmp": False, "positive_dependence_only": True,
        "source_loading": "whole_seasons_validated; fitter_receives_strict_past_rows_only"}
    write_json(output / "实验设置.json", settings)
    active_cutoff = None
    try:
        rows, sources = [], {}
        for season in ("2024-25", "2025-26"):
            entries, sources[season] = load_period(root, parent, season)
            rows.extend(entries)
        expected = {r["match_id"] for r in rows if BOUNDARIES[0] <= r["date"] < BOUNDARIES[-1]}
        tracks, folds, scored = {arm: [] for arm in ARMS}, [], set()
        for cutoff, end in WINDOWS:
            active_cutoff = cutoff
            training, target = window_rows(rows, cutoff, end)
            fit = fit_window(training, training_start=TRAINING_START, before_date=cutoff)
            folder = output / cutoff
            folder.mkdir()
            write_json(folder / PARAMETER_FILE, fit)
            generated = research_tracks(target, fit["k"])
            for arm in ARMS:
                for row in generated[arm]:
                    row.update(dependence_training_start=TRAINING_START, dependence_fit_before_date=cutoff,
                               dependence_target_end_exclusive=end)
            for row in generated[CANDIDATE]:
                if row["match_id"] in scored or not cutoff <= row["date"] < end:
                    raise ValueError("Each target must be scored exactly once in its declared window")
                scored.add(row["match_id"])
            write_json(folder / "完整研究预测.json", generated)
            analysis, aligned = analyze(generated, repeats=repeats)
            write_json(folder / "同场分析.json", analysis)
            for arm in ARMS:
                tracks[arm].extend(aligned[arm])
            folds.append({"cutoff": cutoff, "end_exclusive": end, "fit": fit, "analysis": analysis})
            print(f"fold {cutoff}: training={len(training)} target={len(target)} k={fit['k']:.9f} delta={analysis[DELTA]}", flush=True)
        if scored != expected:
            raise ValueError("Require the entire declared archived target cohort without silent row loss")
        pooled, aligned = analyze(tracks, repeats=repeats)
        by_season = {}
        for season, start, end in (("2024-25_partial", "2024-08-01", "2025-07-01"),
                                   ("2025-26", "2025-08-01", "2026-07-01")):
            subset = {arm: [r for r in aligned[arm] if start <= r["date"] < end] for arm in ARMS}
            by_season[season], _ = analyze(subset, repeats=repeats)
        result = {"settings": settings, "folds": folds, "pooled": pooled, "by_season": by_season,
            "k_path_descriptives": describe([f["fit"]["k"] for f in folds]), "sources": sources,
            "counts": {"new_dependence_fits": len(folds), "new_team_fits": 0,
                "unique_scored_matches": len(scored), "research_predictions": pooled["n"],
                "warmup_only_forecast_ids": [r["match_id"] for r in rows if r["date"] < BOUNDARIES[0]],
                "reused_team_parameter_snapshots": sum(s["parameter_snapshots"] for s in sources.values())},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "elapsed_seconds": time.perf_counter()-started}}
        write_json(output / "完整前推预测.json", aligned)
        write_csv(output / "逐场前推概率.csv", aligned, candidate=CANDIDATE)
        write_histogram(output / "逐场损失分布.svg", aligned, candidate=CANDIDATE)
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
    output = args.output or args.root / "outputs" / f"主客进球相关性前推_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "n": result["pooled"]["n"],
        "k_path": [f["fit"]["k"] for f in result["folds"]],
        "pooled_differences": result["pooled"][DELTA]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
