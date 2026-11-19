"""First real-data, fixed-parameter, date-level walk-forward experiment.

No hyperparameter search, player adjustment, market blending, or betting rules.
Markets are retrospective comparators with unknown individual quote timestamps.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np

from .date_model import DateTeamGoalModel
from .evaluation import calibration_bins, probability_scores
from .football_data import HistoricalMatch, LEAGUES, MARKET_COLUMNS, load_directory
from .probability import score_grid, summarize


def de_vig(prices):
    raw = 1 / np.asarray(prices, dtype=float)
    if raw.shape != (3,) or not np.isfinite(raw).all() or np.any(raw <= 0) or np.any(raw >= 1):
        raise ValueError("Need three valid decimal prices greater than one")
    return raw / raw.sum()


def descriptive(records):
    output = {}
    for competition in sorted({r.competition for r in records}):
        subset = [r for r in records if r.competition == competition and not r.goal_fit_exclusion]
        values = np.asarray([r.full_time for r in subset], dtype=float)
        output[competition] = {"n_matches": len(subset), "goal_counts": {}, "statistics": {}}
        for index, side in enumerate(("home", "away")):
            x = values[:, index]
            output[competition]["statistics"][side] = {
                "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                "median": float(np.median(x)), "variance": float(x.var(ddof=1)),
                "zero_fraction": float((x == 0).mean()), "max": int(x.max()),
            }
            output[competition]["goal_counts"][side] = dict(sorted(Counter(map(int, x)).items()))
    return output


def walk_forward(records, *, start_date=date(2025, 1, 1), interval_days=7,
                 regularization=0.01, half_life_days=180):
    if type(start_date) is not date or type(interval_days) is not int or interval_days < 1:
        raise ValueError("Use a calendar start date and positive integer update interval")
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate source match IDs must be resolved before fitting")
    predictions, skips, fits = [], [], []
    for competition in sorted({r.competition for r in records}):
        league = sorted((r for r in records if r.competition == competition),
                        key=lambda r: (r.match_date, r.match_id))
        cutoff = start_date
        while cutoff <= league[-1].match_date:
            end = cutoff + timedelta(days=interval_days)
            targets = [r for r in league if cutoff <= r.match_date < end]
            eligible_targets = []
            for match in targets:
                if match.goal_fit_exclusion:
                    skips.append({"match_id": match.match_id, "date": match.match_date.isoformat(),
                                  "competition": competition, "reason": match.goal_fit_exclusion})
                else:
                    eligible_targets.append(match)
            if eligible_targets:
                earlier = [r for r in league if r.match_date < cutoff and not r.goal_fit_exclusion]
                if not earlier:
                    skips.extend({"match_id": r.match_id, "date": r.match_date.isoformat(),
                                  "competition": competition, "reason": "no_earlier_history"}
                                 for r in eligible_targets)
                else:
                    model = DateTeamGoalModel(regularization, half_life_days).fit(earlier, cutoff)
                    fits.append({"competition": competition, **model.parameters()})
                    for match in eligible_targets:
                        if {match.home, match.away} - model.teams:
                            skips.append({"match_id": match.match_id,
                                "date": match.match_date.isoformat(), "competition": competition,
                                "reason": "unseen_team_no_estimated_prior"})
                            continue
                        home, away = model.predict(match.home, match.away, competition, match.match_date)
                        forecast = summarize(score_grid(home, away, rho=0))
                        p = list(forecast["one_x_two"].values())
                        h, a = match.full_time
                        label = 0 if h > a else 2 if h < a else 1
                        predictions.append({
                            "match_id": match.match_id, "competition": competition,
                            "date": match.match_date.isoformat(), "home": match.home_label,
                            "away": match.away_label, "fit_before_date": cutoff.isoformat(),
                            "training_matches": model.n_matches,
                            "training_max_date": model.training_max_date.isoformat(),
                            "actual_score": [h, a], "actual_outcome": label,
                            "probabilities": p, "forecast": forecast,
                            "market_probabilities": {key: list(map(float, de_vig(prices)))
                                                     for key, prices in match.markets.items()},
                            "market_prices": match.markets,
                            "source_url": match.source_url,
                        })
            cutoff = end
        print(f"{competition}: {sum(p['competition'] == competition for p in predictions)} historical predictions",
              flush=True)
    predictions.sort(key=lambda p: (p["date"], p["competition"], p["match_id"]))
    return {"predictions": predictions, "skipped": skips, "fits": fits}


def paired_log_loss_interval(predictions, market, repeats=1000, seed=20261022):
    """Paired fit-cutoff cluster bootstrap; descriptive, not a parameter interval."""
    paired = [p for p in predictions if market in p["market_probabilities"]]
    if not paired:
        return None
    blocks = {}
    for p in paired:
        y = p["actual_outcome"]
        delta = -math.log(max(p["probabilities"][y], 1e-300)) + math.log(
            max(p["market_probabilities"][market][y], 1e-300))
        blocks.setdefault(p["fit_before_date"], []).append(delta)
    groups = list(blocks.values())
    sizes = np.asarray([len(g) for g in groups])
    sums = np.asarray([sum(g) for g in groups])
    result = {"n_matches": len(paired), "n_date_blocks": len(groups),
              "mean_model_minus_market_log_loss": float(sums.sum() / sizes.sum()),
              "interval_type": "paired_fit_cutoff_cluster_bootstrap_percentile_95",
              "repeats": repeats, "seed": seed,
              "note": "Exploratory; single season; fit-cutoff blocks do not remove all team/time dependence"}
    if len(groups) < 2:
        result["interval"] = None
        return result
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(groups), size=(repeats, len(groups)))
    means = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    result["interval"] = list(map(float, np.quantile(means, (0.025, 0.975))))
    return result


def evaluate(predictions):
    if not predictions:
        return {"model": None, "markets": {}}
    p = np.asarray([r["probabilities"] for r in predictions])
    y = np.asarray([r["actual_outcome"] for r in predictions])
    observed = np.asarray([r["actual_score"] for r in predictions], dtype=float)
    rates = np.asarray([[r["forecast"]["expected_goals"][s] for s in ("home", "away")]
                        for r in predictions])
    residual = observed - rates
    total_bucket = np.minimum(observed.sum(axis=1).astype(int), 4)
    total_p = np.asarray([list(r["forecast"]["total_goals"].values()) for r in predictions])
    total_truth = np.eye(5)[total_bucket]
    over = np.asarray([r["forecast"]["over_2_5"] for r in predictions])
    over_truth = (observed.sum(axis=1) >= 3).astype(int)
    corr = float(np.corrcoef(residual.T)[0, 1]) if len(predictions) > 2 else None
    out = {"model": probability_scores(p, y), "calibration": calibration_bins(p, y, bins=5),
           "goals": {
               "total_bucket_log_loss": float(-np.log(np.maximum(total_p[np.arange(len(y)), total_bucket], 1e-300)).mean()),
               "total_bucket_brier_sum": float(np.square(total_p - total_truth).sum(axis=1).mean()),
               "over_2_5_brier": float(np.square(over - over_truth).mean()),
               "over_2_5_predicted_mean": float(over.mean()),
               "over_2_5_observed_frequency": float(over_truth.mean()),
           }, "count_diagnostics": {
               "observed_mean": dict(zip(("home", "away"), map(float, observed.mean(axis=0)))),
               "predicted_mean": dict(zip(("home", "away"), map(float, rates.mean(axis=0)))),
               "mean_squared_pearson_residual": dict(zip(("home", "away"),
                   map(float, (np.square(residual) / rates).mean(axis=0)))),
               "home_away_residual_correlation": corr,
               "predicted_draw_fraction": float(p[:, 1].mean()),
               "observed_draw_fraction": float((y == 1).mean()),
               "note": "Descriptive held-out diagnostics, not a formal dispersion/independence test; no Gaussian residual assumption",
           }, "markets": {}}
    for market in MARKET_COLUMNS:
        paired = [r for r in predictions if market in r["market_probabilities"]]
        if not paired:
            continue
        labels = [r["actual_outcome"] for r in paired]
        q = [r["market_probabilities"][market] for r in paired]
        out["markets"][market] = {
            "market": probability_scores(q, labels),
            "model_on_identical_matches": probability_scores([r["probabilities"] for r in paired], labels),
            "paired_log_loss": paired_log_loss_interval(paired, market),
        }
    return out


def _bar_svg(descriptives, title="2024/25 observed goals — excluded administrative result"):
    """Raw goal-count histogram, before fitting; no new graphics dependencies."""
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="620" viewBox="0 0 960 620">',
             '<rect width="960" height="620" fill="white"/>',
             f'<text x="20" y="25" font-size="18">{title}</text>']
    colors = ("#246e9c", "#d58435")
    for i, (comp, value) in enumerate(sorted(descriptives.items())):
        x0, y0 = 35 + (i % 2) * 480, 65 + (i // 2) * 185
        counts = value["goal_counts"]
        max_count = max(n for side in counts.values() for n in side.values())
        lines.append(f'<text x="{x0}" y="{y0}" font-size="16">{comp} (n={value["n_matches"]}); blue=home, orange=away</text>')
        for g in range(10):
            x = x0 + g * 40
            for s, side in enumerate(("home", "away")):
                count = counts[side].get(g, 0)
                height = count / max_count * 100
                lines.append(f'<rect x="{x+s*16}" y="{y0+120-height:.2f}" width="15" height="{height:.2f}" fill="{colors[s]}"/>')
            lines.append(f'<text x="{x+8}" y="{y0+138}" font-size="12">{g}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def write_csv(path, predictions):
    fields = ["competition", "date", "home", "away", "fit_before_date", "training_matches",
              "actual_home_goals", "actual_away_goals", "p_home", "p_draw", "p_away",
              "expected_home_goals", "expected_away_goals", "p_goals_0", "p_goals_1", "p_goals_2",
              "p_goals_3", "p_goals_4_plus", "p_over_2_5", "p_both_score",
              "fair_odds_home", "fair_odds_draw", "fair_odds_away"]
    fields += [f"{m}_{s}" for m in MARKET_COLUMNS for s in ("home", "draw", "away")]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in predictions:
            f, p = r["forecast"], r["probabilities"]
            row = {key: r[key] for key in ("competition", "date", "home", "away", "fit_before_date", "training_matches")}
            row.update({"actual_home_goals": r["actual_score"][0], "actual_away_goals": r["actual_score"][1],
                        "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                        "expected_home_goals": f["expected_goals"]["home"],
                        "expected_away_goals": f["expected_goals"]["away"],
                        "p_over_2_5": f["over_2_5"], "p_both_score": f["both_score"],
                        "fair_odds_home": 1/p[0], "fair_odds_draw": 1/p[1], "fair_odds_away": 1/p[2]})
            for k in ("0", "1", "2", "3", "4+"):
                row[f"p_goals_{'4_plus' if k == '4+' else k}"] = f["total_goals"][k]
            for market, q in r["market_probabilities"].items():
                row.update({f"{market}_{s}": q[i] for i, s in enumerate(("home", "draw", "away"))})
            writer.writerow(row)


def write_report(path, result):
    protocol, overall = result["protocol"], result["overall"]
    lines = ["# 首次真实数据拟合与时间前推检验", "",
        "本报告使用真实赛果，不是合成演示。它是单赛季、固定参数、日期级回顾性实验，不是当前比赛推荐。", "",
        "## 本轮做法", "",
        f"- 来源：本地 Football-Data 2024/25 五大联赛 CSV，共 {result['data']['records']} 场；进球拟合可用 {result['data']['goal_fit_rows']} 场。",
        f"- 从 {protocol['evaluation_start']} 起，每 {protocol['update_interval_days']} 天分别更新各联赛的攻防、赛事和主场参数。",
        "- 仅用更新日期之前的赛果；同一天比赛互不使用结果。假设更早日期结果次日已知，来源并未提供历史首次公开时间。",
        "- 延用正则强度 0.01、时间半衰期 180 天；rho=0。没有调参、校准、球员修正、战意加减分或市场融合。",
        "- 1X2、总进球、大小球、双方进球来自同一个 Poisson 比分分布。模型期望进球不是射门事件型 xG。",
        "- 后续更新可使用此前已结束的检验期比赛；这是在线时间前推检验，不是整个检验期只拟合一次，也不是独立外部赛季。",
        "- 赔率只是历史 pre-closing / closing 对照，非已核实的开盘或精确赛前24小时报价；各市场都与同一批比赛上的模型比较。", "",
        "## 胜平负概率表现", "",
        "Log loss、Brier、RPS 都越低越好；Brier 为三类误差平方和，RPS 除以2。命中率仅作辅助。", "",
        "| 范围 | 场数 | 模型 Log loss | Avg pre-closing | Avg closing | 模型 Brier | 模型命中率 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for comp, metrics in [("合计", overall), *[(LEAGUES[k], v) for k, v in result["by_league"].items()]]:
        model = metrics["model"]
        if not model:
            continue
        markets = metrics["markets"]
        pre = markets.get("avg_preclosing", {}).get("market", {}).get("log_loss")
        close = markets.get("avg_closing", {}).get("market", {}).get("log_loss")
        fmt = lambda x: "—" if x is None else f"{x:.4f}"
        lines.append(f"| {comp} | {model['n']} | {model['log_loss']:.4f} | {fmt(pre)} | {fmt(close)} | {model['brier_sum']:.4f} | {model['accuracy']:.1%} |")
    lines += ["", "## 与市场的配对差异", "",
        "差值为模型 Log loss 减市场 Log loss：正数表示模型较差。以下为按更新周重采样的95%区间，种子固定；属于探索性不确定性描述，不是模型参数区间，不能消除全部球队及时间依赖。", "",
        "| 市场 | 同场数量 | 平均差值 | 95%周分组重采样区间 |", "|---|---:|---:|---|"]
    for market, entry in overall["markets"].items():
        d = entry["paired_log_loss"]
        interval = d["interval"]
        value = "—" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        lines.append(f"| {market} | {d['n_matches']} | {d['mean_model_minus_market_log_loss']:.4f} | {value} |")
    diag = overall["count_diagnostics"]
    goals = overall["goals"]
    lines += ["", "## 分布与假设检查", "",
        "![原始进球分布](原始进球分布.svg)", "",
        f"- 全场总进球分桶（0、1、2、3、4+）Log loss：{goals['total_bucket_log_loss']:.4f}；Brier：{goals['total_bucket_brier_sum']:.4f}。",
        f"- 大于2.5球：平均预测 {goals['over_2_5_predicted_mean']:.1%}，实际发生 {goals['over_2_5_observed_frequency']:.1%}；Brier {goals['over_2_5_brier']:.4f}。",
        f"- 平局：平均预测 {diag['predicted_draw_fraction']:.1%}，实际发生 {diag['observed_draw_fraction']:.1%}。",
        f"- 主/客场均方 Pearson 残差：{diag['mean_squared_pearson_residual']['home']:.3f} / {diag['mean_squared_pearson_residual']['away']:.3f}；双方进球残差相关 {diag['home_away_residual_correlation']:.3f}。",
        "- 上述是检验期描述性诊断，未作独立性/离散程度显著性断言。进球是计数，不套用 OLS 的正态残差或等方差检查；高比分有效比赛不按离群值删除。",
        "- 每类校准分箱、各联赛完整指标、缺失项和逐场概率见配套 JSON/CSV。当前尚无拟合后的校准映射或参数置信区间。", "",
        "## 排除、覆盖与边界", "",
        f"- 检验日期范围内比赛 {result['coverage']['target_rows']} 场，实际预测 {len(result['predictions'])} 场，跳过 {len(result['skipped'])} 场。跳过原因：{json.dumps(result['coverage']['skipped_reasons'], ensure_ascii=False)}。",
        "- 2024-12-14 柏林联合—波鸿原文件的0–2为裁决赛果，未当作场上进球学习，未改写原CSV。[德甲官方解释](https://www.bundesliga.com/de/bundesliga/news/vfl-bochum-union-berlin-schiedsgericht-urteil-drei-punkte-feuerzeug-wurf-31928)",
        "- 这是日期级实验，不声称重建历史赛前24小时或首发后信息集。下载时点仍只是今天的观测时间。",
        "- 球员、伤停、轮换、赛事重视程度、跨联赛强度桥梁及半全场尚未纳入；不得直接外推到欧战。",
        "- 公平赔率仅为本模型概率倒数，未加误差缓冲，不是价值已被证实或保证盈利。未选择投注策略、串关或进行收益优化。",
        "- 本轮真实检验成绩会影响后续研发，因此未来调参需另留新时间段/赛季作独立检验，不能反复调到这段成绩变好再称独立测试。", "",
        "## 可复用文件", "",
        "- `逐场历史预测.csv`：可用表格软件查看的每场模型概率、市场概率、总进球和公平赔率。",
        "- `回测完整结果.json`：全部预测、校准分箱、指标、来源和实验设置。",
        "- `逐周拟合参数.json`：每次更新的参数、训练日期和比赛ID。",
        "- `赛季末模型参数.json`：所有合格赛果拟合的研究模型，训练止于2025年，不是当前2026年球队实力。",
        "- `数据接入检查.json`、`原始进球分布.svg`：覆盖、排除与原始分布。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_dir: Path, output_dir: Path, start_date=date(2025, 1, 1)):
    records, data_report = load_directory(input_dir)
    if data_report["rejected_rows"] or data_report["duplicate_keys"]:
        raise ValueError("Resolve rejected rows or duplicate match IDs before fitting")
    output_dir.mkdir(parents=True, exist_ok=False)
    desc = descriptive(records)
    (output_dir / "原始进球分布.svg").write_text(_bar_svg(desc), encoding="utf-8")
    print(f"Loaded {len(records)} results; descriptive distributions saved before fitting", flush=True)
    experiment = walk_forward(records, start_date=start_date)
    predictions = experiment["predictions"]
    if not predictions:
        raise ValueError("No out-of-time predictions produced")
    final_models = {}
    for comp in sorted({r.competition for r in records}):
        league = [r for r in records if r.competition == comp]
        before = max(r.match_date for r in league) + timedelta(days=1)
        final_models[comp] = DateTeamGoalModel().fit(league, before).parameters()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"analysis_type": "exploratory_fixed_parameter_date_level_walk_forward",
                     "evaluation_start": start_date.isoformat(), "update_interval_days": 7,
                     "regularization": 0.01, "half_life_days": 180, "rho": 0,
                     "hyperparameter_search": False, "market_blending": False,
                     "player_adjustment": False, "precise_quote_timestamps": False},
        "runtime": {"python": sys.version, **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")}},
        "data": data_report, "descriptives": desc,
        "coverage": {"target_rows": sum(r.match_date >= start_date for r in records),
                     "predicted_rows": len(predictions),
                     "skipped_reasons": dict(Counter(r["reason"] for r in experiment["skipped"]))},
        "overall": evaluate(predictions),
        "by_league": {c: evaluate([p for p in predictions if p["competition"] == c])
                      for c in sorted({p["competition"] for p in predictions})},
        "predictions": predictions, "skipped": experiment["skipped"],
    }
    for name, payload in (("数据接入检查.json", data_report), ("回测完整结果.json", result),
                          ("逐周拟合参数.json", experiment["fits"]), ("赛季末模型参数.json", final_models)):
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output_dir / "逐场历史预测.csv", predictions)
    write_report(output_dir / "首次真实数据回测报告.md", result)
    print(json.dumps({"output_directory": str(output_dir), "coverage": result["coverage"],
                      "model": result["overall"]["model"],
                      "avg_preclosing": result["overall"]["markets"].get("avg_preclosing", {}).get("market")},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "outputs" /
                        "Football-Data_2024-25_个人使用_20261022T111235Z")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2025, 1, 1))
    args = parser.parse_args()
    output = args.output_dir or root / "outputs" / ("真实数据首次回测_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run(args.input_dir, output, args.start_date)


if __name__ == "__main__":
    main()
