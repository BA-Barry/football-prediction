"""One earlier-fitted mean-CMP dispersion; unchanged raw team means and defaults."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .cmp_fit import fit_cmp
from .evaluation import probability_scores
from .football_data import MARKET_COLUMNS
from .forecast_diagnostics import (BLOCKS, REPEATS, SEED, align_tracks,
    calibration_summary, losses, moving_block_means, validate_prediction, write_json)
from .goal_rate_diagnostics import row_statistics
from .mean_cmp import cmp_score_forecast, mean_cmp
from .probability import score_grid, summarize
from .rate_calibration_experiment import load_period
from .refresh_experiment import prediction_arrays, scores_for
from .refresh_research import _same


ARMS = ("original", "cmp")
DOMAINS = ("exact_score", "one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way")
RESULT_FILE = "CMP单参数研究结果.json"
REPORT_FILE = "CMP单参数研究说明.md"
PARAMETER_FILE = "较早赛季CMP参数.json"


def research_tracks(rows, nu):
    """Copy archives; generate only the research score distributions, no refits."""
    tracks = {arm: [] for arm in ARMS}
    for row in rows:
        validate_prediction(row)
        stats = row_statistics(row, row["rates"])
        _same(row["forecast"], summarize(score_grid(*row["rates"], rho=0)), "original_forecast")
        _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "original_probabilities")
        try:
            forecast, metadata = cmp_score_forecast(*row["rates"], nu)
            h, a = (mean_cmp(mu, nu) for mu in row["rates"])
            nll = float(-h.logpmf(row["actual_score"][0])-a.logpmf(row["actual_score"][1]))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            raise RuntimeError(f"CMP forecast failed for {row['match_id']}: {exc}") from exc
        if not np.isfinite(nll):
            raise ValueError(f"Nonfinite full-support CMP NLL: {row['match_id']}")
        tracks["original"].append({**row, "exact_score_nll": stats["joint_score_nll"]})
        tracks["cmp"].append({**row, "forecast": forecast,
            "probabilities": [forecast["one_x_two"][k] for k in ("home", "draw", "away")],
            "exact_score_nll": nll, "cmp_nu": nu, "cmp_numerics": metadata})
    return tracks


def loss_vectors(rows):
    values = {"exact_score": np.asarray([r["exact_score_nll"] for r in rows])}
    if not np.isfinite(values["exact_score"]).all():
        raise ValueError("Full-support score losses must be finite")
    for domain, (p, y) in prediction_arrays(rows).items():
        observed = p[np.arange(len(p)), y]
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise ValueError("Observed probabilities must be positive; no loss clipping")
        values[domain] = -np.log(observed)
    return values


def score_rows(rows):
    return {**scores_for(rows), "exact_score": {"n": len(rows),
        "mean_nll": float(loss_vectors(rows)["exact_score"].mean())}}


def event_summary(rows):
    """Distribution-specific events; never use Poisson formulas for CMP events."""
    expected = {k: [] for k in ("home_zero", "away_zero", "draw", "zero_zero", "one_one", "total_four_plus")}
    actual = {k: [] for k in expected}
    for r in rows:
        h, a = r["actual_score"]
        if "cmp_nu" in r:
            hd, ad = (mean_cmp(mu, r["cmp_nu"]) for mu in r["rates"])
            p0h, p0a = float(np.exp(hd.logpmf(0))), float(np.exp(ad.logpmf(0)))
            p11 = float(np.exp(hd.logpmf(1)+ad.logpmf(1)))
        else:
            p0h, p0a = (float(np.exp(-mu)) for mu in r["rates"])
            p11 = float(np.exp(-sum(r["rates"]))*np.prod(r["rates"]))
        ps = (p0h, p0a, r["probabilities"][1], p0h*p0a, p11, r["forecast"]["total_goals"]["4+"])
        ys = (h == 0, a == 0, h == a, h == a == 0, h == a == 1, h+a >= 4)
        for key, p, y in zip(expected, ps, ys):
            expected[key].append(p)
            actual[key].append(y)
    return {"events": {k: {"actual_count": int(sum(actual[k])), "expected_count": float(sum(expected[k])),
                          "actual_minus_expected_rate": float(np.mean(actual[k])-np.mean(expected[k]))} for k in expected},
        "raw_goal_means": np.mean([r["rates"] for r in rows], axis=0).tolist(),
        "observed_goal_means": np.mean([r["actual_score"] for r in rows], axis=0).tolist(),
        "maximum_finite_grid_mean_error": float(max(abs(r["forecast"]["expected_goals"][side]-r["rates"][i])
                                                   for r in rows for i, side in enumerate(("home", "away"))))}


def analyze(raw_tracks, *, repeats=REPEATS, candidate="cmp", event_summary_fn=event_summary):
    # Shared same-match scoring; distribution-specific diagnostics are supplied
    # by each research caller. Defaults retain the original CMP interface.
    ARMS = ("original", candidate)
    if set(raw_tracks) != set(ARMS):
        raise ValueError("Require original and declared candidate tracks")
    tracks, coverage = align_tracks(json.loads(json.dumps(raw_tracks, allow_nan=False)))
    if any(coverage["outside_common"][arm] for arm in ARMS):
        raise ValueError("No silent coverage shrinkage")
    rows = {arm: list(tracks[arm].values()) for arm in ARMS}
    reference = rows["original"]
    for a, b in zip(reference, rows[candidate]):
        if a["rates"] != b["rates"]:
            raise ValueError("Score-distribution research must preserve raw means exactly")
    per_loss = {arm: loss_vectors(rows[arm]) for arm in ARMS}
    differences = {d: per_loss[candidate][d]-per_loss["original"][d] for d in DOMAINS}
    result = {"n": len(reference), "coverage": coverage, "first_date": reference[0]["date"],
        "last_date": reference[-1]["date"], "scores": {arm: score_rows(rows[arm]) for arm in ARMS},
        f"{candidate}_minus_original": {d: float(v.mean()) for d, v in differences.items()},
        "calibration": {arm: calibration_summary(*prediction_arrays(rows[arm])["one_x_two"]) for arm in ARMS},
        "event_diagnostics": {arm: event_summary_fn(rows[arm]) for arm in ARMS}, "bootstrap": {}}
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in reference], np.column_stack(list(differences.values())),
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["differences"] = {d: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                               for i, d in enumerate(DOMAINS)}
        result["bootstrap"][str(weeks)] = boot
    for name, labels in (("by_league", [r["competition"] for r in reference]),
                         ("by_month", [r["date"][:7] for r in reference])):
        result[name] = {}
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            result[name][label] = {"n": len(indices),
                "scores": {arm: score_rows([rows[arm][i] for i in indices]) for arm in ARMS},
                f"{candidate}_minus_original": {d: float(v[indices].mean()) for d, v in differences.items()}}
    result["markets"] = {}
    y = np.asarray([r["actual_outcome"] for r in reference])
    for market in MARKET_COLUMNS:
        indices = [i for i, r in enumerate(reference) if market in r["market_probabilities"]]
        item = {"n": len(indices), "missing_ids": [r["match_id"] for r in reference if market not in r["market_probabilities"]]}
        if indices:
            p = np.asarray([reference[i]["market_probabilities"][market] for i in indices])
            diff = np.column_stack([per_loss[arm]["one_x_two"][indices]-losses(p, y[indices]) for arm in ARMS])
            boot = moving_block_means([reference[i]["date"] for i in indices], diff,
                                      block_weeks=4, repeats=repeats, seed=SEED)
            means, intervals = boot.pop("mean"), boot.pop("interval_95")
            item.update({"match_ids": [reference[i]["match_id"] for i in indices],
                "market_scores": probability_scores(p, y[indices]),
                "model_scores_on_same_matches": {arm: probability_scores(
                    prediction_arrays(rows[arm])["one_x_two"][0][indices], y[indices]) for arm in ARMS},
                "model_minus_market": {arm: {"mean": means[j], "interval_95": None if intervals is None else intervals[j]}
                                       for j, arm in enumerate(ARMS)}, "bootstrap_metadata": boot})
        result["markets"][market] = item
    return result, rows


def write_csv(path, tracks, *, candidate="cmp"):
    ARMS = ("original", candidate)
    per_loss = {arm: loss_vectors(tracks[arm]) for arm in ARMS}
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals", "home_mean", "away_mean"]
    fields += [f"{arm}_{key}" for arm in ARMS for key in
               ("p_home", "p_draw", "p_away", *(f"{d}_nll" for d in DOMAINS))]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(tracks["original"]):
            item = {key: row[key] for key in fields[:5]}
            item.update(zip(("home_goals", "away_goals"), row["actual_score"]))
            item.update(zip(("home_mean", "away_mean"), row["rates"]))
            for arm in ARMS:
                item.update({f"{arm}_p_{label}": tracks[arm][i]["probabilities"][j]
                             for j, label in enumerate(("home", "draw", "away"))})
                item.update({f"{arm}_{d}_nll": float(per_loss[arm][d][i]) for d in DOMAINS})
            writer.writerow(item)


def report(result):
    fit, later = result["fit"], result["later"]
    lines = ["# 单参数CMP比分分布研究", "",
        f"仅用2024/25的{result['earlier']['n']}场训练一个共享离散参数ν={fit['nu']:.9f}；"
        f"保存后固定应用于2025/26的{later['n']}场。正式默认模型未改，所有产物与临时文件均在G盘。", "",
        "## 本轮真正改变了什么", "",
        "球队攻击、防守、主场、联赛参数及每场原始主客预期进球μ全部保留。只把独立Poisson边际替换为独立均值参数化CMP；"
        "ν=1回到Poisson，ν>1使相同均值下的分布更集中，ν<1更分散。不是人为给平局加分，也没有引入主客进球相关性。"
        "按论文定义反解内部η=logθ，使E[Y]=μ；不能把θ直接设为原μ。"
        "[均值参数化CMP论文，§2公式1–2](https://arxiv.org/html/1606.03214)。", "",
        "仅基于已验证原180天半衰期、alpha=0.01、28天更新的存档预测；没有叠加前轮365天/.001调参结果。"
        "逐场等权最小化主客完整比分联合负对数似然，单一ν跨主客和五联赛共用。预定范围[0.25,4]，"
        "检查ν=1及两端点，记录每次目标计算、优化状态和边界状态；这不是保证找到了任意范围全局最优。", "",
        "旧球队拟合只取严格更早日期赛果，以次日可用作为研究假设；源文件没有提供可靠历史首次公开时间，"
        "因此不能宣称已还原真实逐时点信息集。", "",
        f"边界最优：{fit['at_boundary']}。较早训练表现是拟合内成绩，有乐观偏差；两季已在此前研究看过，"
        "较晚成绩仍为探索性跨季验证，不是新盲测。ν也可能吸收原均值预测误差，不能解释为足球固有离散规律。", "",
        "## 较晚同场结果（损失越低越好）", "",
        "| 目标 | 原Poisson | 单参数CMP | 新减旧 | 四周区块95%区间 |", "|---|---:|---:|---:|---|"]
    labels = {"exact_score": "完整比分NLL（主目标）", "one_x_two": "胜平负LL", "total_goals_0_1_2_3_4plus": "总进球0/1/2/3/4+ LL", "away_minus_two_three_way": "客让两球竞彩LL"}
    for d in DOMAINS:
        field = "mean_nll" if d == "exact_score" else "log_loss"
        a, b = (later["scores"][arm][d][field] for arm in ARMS)
        ci = later["bootstrap"]["4"]["differences"][d]["interval_95"]
        interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {labels[d]} | {a:.6f} | {b:.6f} | {b-a:+.6f} | {interval} |")
    primary_ci = later["bootstrap"]["4"]["differences"]["exact_score"]["interval_95"]
    primary_text = ("主目标完整比分的四周区间跨零，不能确认稳定改善。" if primary_ci and primary_ci[0] <= 0 <= primary_ci[1]
                    else "主目标区间仅为条件探索结果，不能据此确认未来改善。")
    one_x_two_cis = [later["bootstrap"][str(w)]["differences"]["one_x_two"]["interval_95"] for w in BLOCKS]
    secondary_text = ("胜平负的2/4/8周条件探索区间均偏向改善，是保留候选的信号，但不是确认性证据。"
                      if all(ci and ci[1] < 0 for ci in one_x_two_cis) else "胜平负的不同区块结果需一并解释，不能只挑点改善。")
    lines += ["", "全部目标、联赛/月度分组、胜平负Brier/RPS/准确率及2/8周敏感性均保存在结果JSON，未只挑改善项。", "",
        primary_text + secondary_text + "各目标可能有取舍，正式默认模型保持不变。", "",
        "| 联赛 | 场次 | 完整比分新减旧 | 胜平负新减旧 | 总进球新减旧 |", "|---|---:|---:|---:|---:|"]
    for league, item in later["by_league"].items():
        delta = item["cmp_minus_original"]
        lines.append(f"| {league} | {item['n']} | {delta['exact_score']:+.6f} | {delta['one_x_two']:+.6f} | {delta['total_goals_0_1_2_3_4plus']:+.6f} |")
    lines += ["", "各联赛表现并不一致；这些分组为事后描述，不据此另拟合联赛参数或挑选适用联赛。", "",
        "| 事件 | 实际次数 | 原期望次数 | CMP期望次数 |", "|---|---:|---:|---:|"]
    for key, item in later["event_diagnostics"]["original"]["events"].items():
        b = later["event_diagnostics"]["cmp"]["events"][key]
        lines.append(f"| {key} | {item['actual_count']} | {item['expected_count']:.3f} | {b['expected_count']:.3f} |")
    lines += ["", "这些事件只是描述分布变化，不是按事后偏差逐项补丁。原始μ逐场完全相等；"
        f"CMP有限网格均值与原μ最大差{later['event_diagnostics']['cmp']['maximum_finite_grid_mean_error']:.3g}。"
        "数值实现使用log域、自适应支持、概率/一二阶矩尾界和均值求根；每场数值元数据随预测保存。"
        "尾界为解析截断界，另列浮点误差余量但不是区间算术认证的总误差保证，更不是概率的统计置信区间。"
        "完整比分NLL直接使用完整支持logpmf，不将有限比分网格外的真实比分删掉或设零。", "",
        "## 市场对照：不融合、不当作开盘回测", "",
        "| 市场 | 同场数 | 市场胜平负LL | 原LL | CMP LL |", "|---|---:|---:|---:|---:|"]
    for market, item in later["markets"].items():
        if item["n"]:
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['original']['log_loss']:.6f} | {item['model_scores_on_same_matches']['cmp']['log_loss']:.6f} |")
    lines += ["", "## 解释边界与下一步", "",
        "共享五联赛日历、同场成对损失差，移动连续2/4/8周区块，4周为主；"
        f"{result['settings']['bootstrap_repeats']}次，seed={SEED}。区间条件于已拟合ν及球队预测，"
        "不包含ν估计、结构选择或球队重训不确定性，也不是盈利区间。所有比较均探索性，不作多重检验后的显著性承诺。", "",
        "本轮不自动采纳新模型。完整比分、胜平负、总进球与让球目标可能有取舍；即使历史点估计改善，也不能据此保证未来或盈利。"
        "客让两球竞彩让胜须客队净胜至少3球，净胜2球为让平，不是亚洲盘走水。"
        "历史赔率采集时间不能自动解释为已知开盘/T−24h可交易报价，低赔率不等于高价值。", "",
        "球员首发/伤停、比赛重视程度、跨赛事强度和半全场状态仍未在此实验中得到解决；不据此直接生成当前欧战串关。"
        "没有加入神经网络或市场融合。是否进一步引入比分相关结构属于下一项实质变更，另行讨论。", "",
        "实验设计技能约束为同场、同μ、仅一个结构参数；统计分析技能用于配对损失与赛程相关性的条件探索区间。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"base_half_life_days": 180, "base_regularization": 0.01, "base_rho": 0,
        "update_interval_days": 28, "nu_bounds": [0.25, 4.0], "parameters_learned": 1,
        "means_changed": False, "dependence_changed": False, "selection": "earlier equal-match joint exact-score NLL",
        "source_reference": str(parent), "previously_viewed_seasons": True, "production_api_changed": False,
        "primary_domain": "exact_score", "secondary_domains": list(DOMAINS[1:]),
        "interval_scope": "conditional_on_fitted_nu_and_archived_team_predictions_exploratory",
        "bootstrap_blocks_weeks": list(BLOCKS), "bootstrap_repeats": repeats, "seed": SEED}
    write_json(output / "实验设置.json", settings)
    try:
        earlier_rows, earlier_source = load_period(root, parent, "2024-25")
        fit = fit_cmp(earlier_rows)
        write_json(output / PARAMETER_FILE, fit)
        print(f"earlier CMP parameter saved: nu={fit['nu']:.9f} n={len(earlier_rows)}", flush=True)
        # The later source CSVs/archives are opened only after the parameter file.
        later_rows, later_source = load_period(root, parent, "2025-26")
        results, aligned = {}, {}
        for period, rows in (("earlier", earlier_rows), ("later", later_rows)):
            tracks = research_tracks(rows, fit["nu"])
            write_json(output / f"{period}_完整研究预测.json", tracks)
            results[period], aligned[period] = analyze(tracks, repeats=repeats)
            write_csv(output / f"{period}_同场概率.csv", aligned[period])
            print(f"{period}: n={len(rows)} differences={results[period]['cmp_minus_original']}", flush=True)
        result = {"settings": settings, "fit": fit, **results,
            "sources": {"earlier": earlier_source, "later": later_source},
            "counts": {"new_team_fits": 0, "new_dispersion_fits": 1,
                       "reused_parameter_snapshots": earlier_source["parameter_snapshots"]+later_source["parameter_snapshots"],
                       "new_research_forecasts": len(earlier_rows)+len(later_rows)},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "elapsed_seconds": time.perf_counter()-started}}
        write_json(output / RESULT_FILE, result)
        (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
        return result
    except Exception as exc:
        write_json(output / "研究失败.json", {"type": type(exc).__name__, "message": str(exc),
            "history": getattr(exc, "history", None), "failed_no_fallback": True})
        raise


def main():
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = args.parent or args.root / "outputs" / "更新频率研究_20261023T073047979805Z"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "outputs" / f"CMP单参数研究_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "nu": result["fit"]["nu"],
        "later_differences": result["later"]["cmp_minus_original"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
