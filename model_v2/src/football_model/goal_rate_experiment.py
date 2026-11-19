"""Read-only goal-rate diagnostics of the completed refresh experiment.

Reconstruct conditional rates, not truncated score-grid expectations. Retain all
three arms and both already-viewed seasons; fit/select/alter no model.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .football_data import LEAGUES
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, align_tracks, moving_block_means, write_json
from .goal_rate_diagnostics import RATE_EDGES, STAT_FIELDS, rate_bin_labels, rates_from_parameters, row_statistics, summary
from .native_experiment import load_seasons, season_directories
from .refresh_experiment import ARMS, CONTRASTS
from .refresh_research import INTERVALS, PERIODS, validate_trial


RESULT_FILE = "进球率诊断结果.json"
REPORT_FILE = "进球率诊断说明.md"
NLL_FIELDS = ("home_poisson_nll", "away_poisson_nll", "joint_score_nll")
MOMENT_FIELDS = ("home_residual", "away_residual", "home_pearson_square_minus_one",
    "away_pearson_square_minus_one", "pearson_cross_product", "home_zero_residual",
    "away_zero_residual", "total_four_plus_residual", "draw_residual",
    "zero_zero_residual", "one_one_residual")
BIN_NAMES = tuple(f"{lower:g}-{upper:g}" for lower, upper in zip(RATE_EDGES, RATE_EDGES[1:])) + ("3+",)
SOURCES = {
    "poisson": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html",
    "scaled_bessel": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html",
    "moving_blocks": "https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html",
}


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=invalid)


def _cohort(entries, statistics, indices):
    if not indices:
        return {"n": 0, "summaries": None, "mean_nll_differences": None}
    summaries = {arm: summary([entries[arm][i] for i in indices],
                              [statistics[arm][i] for i in indices]) for arm in ARMS}
    differences = {f"{arm}_minus_days_28": {field: float(np.mean([
        statistics[arm][i][field] - statistics["days_28"][i][field] for i in indices]))
        for field in NLL_FIELDS} for arm in CONTRASTS}
    return {"n": len(indices), "summaries": summaries, "mean_nll_differences": differences}


def analyze_period(raw_tracks, raw_statistics, *, repeats=REPEATS):
    """One row per match and shared weights across arms/sides; no new fit."""
    if set(raw_tracks) != set(ARMS) or set(raw_statistics) != set(ARMS):
        raise ValueError("Need exactly the three declared refresh arms")
    # JSON containers unify tuples/lists; no numeric rounding or correction.
    raw_tracks = json.loads(json.dumps(raw_tracks, allow_nan=False))
    tracks, coverage = align_tracks(raw_tracks)
    entries = {arm: list(track.values()) for arm, track in tracks.items()}
    statistics = {}
    for arm in ARMS:
        available = raw_statistics[arm]
        all_ids = {row["match_id"] for row in raw_tracks[arm]}
        if not isinstance(available, dict) or set(available) != all_ids:
            raise ValueError("Diagnostic statistics and complete arm IDs differ")
        # Check every row including the extra-coverage subset, not just common.
        summary(raw_tracks[arm], [available[row["match_id"]] for row in raw_tracks[arm]])
        statistics[arm] = [available[key] for key in coverage["common_match_ids"]]
    reference = entries["days_28"]
    result = _cohort(entries, statistics, list(range(len(reference))))
    result.update({"coverage": coverage, "first_date": reference[0]["date"],
                   "last_date": reference[-1]["date"]})
    features = {f"{arm}_{field}": np.asarray([row[field] for row in statistics[arm]])
                for arm in ARMS for field in (*MOMENT_FIELDS, *NLL_FIELDS)}
    features.update({f"{arm}_minus_days_28_{field}":
        features[f"{arm}_{field}"] - features[f"days_28_{field}"]
        for arm in CONTRASTS for field in NLL_FIELDS})
    result["bootstrap"] = {}
    for weeks in BLOCKS:
        boot = moving_block_means([row["date"] for row in reference],
            np.column_stack(list(features.values())), block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["features"] = {name: {"mean": means[i],
            "interval_95": None if intervals is None else intervals[i]}
            for i, name in enumerate(features)}
        result["bootstrap"][str(weeks)] = boot
    for name, labels in (("by_league", [row["competition"] for row in reference]),
                         ("by_month", [row["date"][:7] for row in reference])):
        result[name] = {label: _cohort(entries, statistics,
            [i for i, group in enumerate(labels) if group == label]) for label in sorted(set(labels))}
    result["fixed_28_rate_bins"] = {}
    for side in ("home", "away"):
        labels = rate_bin_labels([row[f"{side}_rate"] for row in statistics["days_28"]])
        result["fixed_28_rate_bins"][side] = {label: _cohort(entries, statistics,
            [i for i, group in enumerate(labels) if group == label]) for label in BIN_NAMES}
    result["extra_coverage"] = {}
    for arm in ARMS:
        ids = set(coverage["outside_common"][arm])
        extra = [row for row in raw_tracks[arm] if row["match_id"] in ids]
        result["extra_coverage"][arm] = {"match_ids": sorted(ids),
            "summary_separate_not_comparable_to_common": summary(extra,
                [raw_statistics[arm][row["match_id"]] for row in extra]) if extra else None}
    return result, entries, statistics


def write_csv(path, entries, statistics):
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals",
              "fixed_28_home_rate_bin", "fixed_28_away_rate_bin"]
    fields += [f"{arm}_{field}" for arm in ARMS for field in
               ("fit_before_date", "training_matches", *STAT_FIELDS)]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(entries["days_28"]):
            item = {field: row[field] for field in fields[:5]}
            item.update({f"{side}_goals": row["actual_score"][j] for j, side in enumerate(("home", "away"))})
            item.update({f"fixed_28_{side}_rate_bin": rate_bin_labels(statistics["days_28"][i][f"{side}_rate"])
                         for side in ("home", "away")})
            for arm in ARMS:
                item.update({f"{arm}_{key}": entries[arm][i][key]
                             for key in ("fit_before_date", "training_matches")})
                item.update({f"{arm}_{key}": value for key, value in statistics[arm][i].items()})
            writer.writerow(item)


def _ci(value):
    return "不可估计" if value is None else f"[{value[0]:+.5f}, {value[1]:+.5f}]"


def report(result):
    counts = result["counts"]
    lines = ["# 赛前球队模型：条件进球率诊断", "",
        f"本轮0次拟合、0条新增模型预测、0次概率修正。读取{counts['parameter_snapshots_read']}份已有参数，"
        f"对{counts['saved_prediction_rows_validated']}条存档预测核验日期、来源、覆盖及完整比分摘要。"
        f"共同队列共{counts['unique_common_matches']}场；三臂合计{counts['common_arm_diagnostic_rows']}条诊断记录，"
        "不是三倍独立样本。所有新文件与临时目录均在项目目录。", "",
        "## 本轮问什么", "",
        "检查每场赛前条件进球率λ，而非射门xG。28天旧轨为诊断参照，7／14天完整保留作敏感性对照；"
        "只重放原系数，不从有限比分网格的expected_goals倒推λ。模型仍是ridge=0.01、半衰期180天、rho=0的独立Poisson。"
        "不把两个主客侧当两场，也不按本轮诊断挑选新分布或更新频率。", "",
        "2024/25和2025/26均已查看，是探索性回顾而非新增盲测。"
        "历史可用性仅假定严格更早日期赛果已知，同日不使用；训练cutoff不等于实际发布／T−24h时点。", "",
        "## 当前读数", "",
        "以下主读数只总结28天参照轨；全部三臂、全部预定诊断及块长均在后文／JSON保留，不按区间筛选。", ""]
    for season, data in result["periods"].items():
        d = data["summaries"]["days_28"]["scalar_descriptives"]
        lines.append(f"{season}：主每场进球残差{d['home_residual']['mean']:+.5f}、客{d['away_residual']['mean']:+.5f}；"
            f"平局发生率−预测{100*d['draw_residual']['mean']:+.3f}个百分点，总进球≥4差"
            f"{100*d['total_four_plus_residual']['mean']:+.3f}个百分点。")
    lines += ["", "两季均值方向不同，不能采用统一的主／客进球平移。平局点估计同向不等于稳定校准结论；"
        "零进球与1:1并非同步错配，不能只将‘所有低比分’统一放大。"
        "主客残差乘积需结合均值及分组看，不能直接当相关系数估计。", "",
        "## 覆盖与完整比分损失", "",
        "精确比分负对数似然NLL=λ主−y主logλ主+log(y主!)+λ客−y客logλ客+log(y客!)；"
        "在log域直接计算，合法大比分不截断、不设概率下限。它与上轮0／1／2／3／4+总进球分箱LL不是同一目标，不能横比数值。", "",
        "| 赛季 | 更新 | 共同场次 | 主进球NLL | 客进球NLL | 联合精确比分NLL |", "|---|---|---:|---:|---:|---:|"]
    for season, data in result["periods"].items():
        for arm in ARMS:
            d = data["summaries"][arm]["scalar_descriptives"]
            lines.append(f"| {season} | {arm} | {data['n']} | {d['home_poisson_nll']['mean']:.6f} | "
                         f"{d['away_poisson_nll']['mean']:.6f} | {d['joint_score_nll']['mean']:.6f} |")
    lines += ["", "| 赛季 | 对照 | 精确比分平均NLL差 | 2周95%区间 | 4周 | 8周 |", "|---|---|---:|---|---|---|"]
    for season, data in result["periods"].items():
        for arm in CONTRASTS:
            name = f"{arm}_minus_days_28"
            key = f"{name}_joint_score_nll"
            lines.append(f"| {season} | {name} | {data['mean_nll_differences'][name]['joint_score_nll']:+.6f} | "
                + " | ".join(_ci(data["bootstrap"][str(w)]["features"][key]["interval_95"]) for w in BLOCKS) + " |")
    lines += ["", "负差表示该精确比分目标更好，不能用一个目标覆盖胜平负／让球风险。"
        "区间跨零不证明等效，不跨零也不提供本轮反复探索下的确认性结论。额外7天覆盖单列，不混入共同场次。", "",
        "## 条件均值：实际−预测", "",
        "所有残差方向均为实际减预测。正数表示模型低估，负数表示高估；与旧报告的‘预测概率−发生率’方向不同。"
        "O:E是实际进球总和／逐场λ总和，仅为汇总描述，不直接当修正系数。", "",
        "| 赛季 | 更新 | 侧 | 实际进球 | λ总和 | 每场实际−预测 | 4周95%区间 | O:E |",
        "|---|---|---|---:|---:|---:|---|---:|"]
    for season, data in result["periods"].items():
        for arm in ARMS:
            for side, label in (("home", "主"), ("away", "客")):
                s = data["summaries"][arm][side]
                interval = data["bootstrap"]["4"]["features"][f"{arm}_{side}_residual"]["interval_95"]
                lines.append(f"| {season} | {arm} | {label} | {s['observed_sum']} | {s['expected_sum']:.3f} | "
                    f"{s['residual_per_match']:+.5f} | {_ci(interval)} | {s['observed_to_expected_ratio']:.4f} |")
    lines += ["", "两季不是同一比赛队列，球队、对阵、联赛权重均变化。赛际均值变化仅描述，不足以归因实力漂移或主场系数。", "",
        "## 分布形态与低比分（28天轨）", "",
        "Pearson平方减一=((y−λ)²/λ)−1；主客标准化残差乘积=(y主−λ主)(y客−λ客)/√(λ主λ客)。"
        "前者同时受均值错配影响，不是纯过度离散估计；后者含均值偏差乘积，不是相关系数。", "",
        "| 赛季 | 指标 | 平均值 | 2周95%区间 | 4周 | 8周 |", "|---|---|---:|---|---|---|"]
    labels = {"home_pearson_square_minus_one": "主Pearson²−1", "away_pearson_square_minus_one": "客Pearson²−1",
        "pearson_cross_product": "主客标准化残差乘积", "home_zero_residual": "主零进球残差",
        "away_zero_residual": "客零进球残差", "total_four_plus_residual": "总进球≥4残差",
        "draw_residual": "平局残差", "zero_zero_residual": "0:0残差", "one_one_residual": "1:1残差"}
    for season, data in result["periods"].items():
        for field, label in labels.items():
            key = f"days_28_{field}"
            value = data["summaries"]["days_28"]["scalar_descriptives"][field]["mean"]
            lines.append(f"| {season} | {label} | {value:+.5f} | "
                + " | ".join(_ci(data["bootstrap"][str(w)]["features"][key]["interval_95"]) for w in BLOCKS) + " |")
    lines += ["", "事件残差单位是每场概率／发生率差，不是进球数。解析概率使用每场独立Poisson的零进球、总进球尾部和Skellam平局；"
        "0:0、1:1是平局子集，不能把重叠残差相加当独立证据。", "",
        "## 原始进球分布（28天轨）", "",
        "期望场数=逐场P(Y=k|λ)求和，不用平均λ代替异质条件分布。4+保留全部合法大比分，"
        "原始最大值另存完整描述统计。以下为数表，不新增仪表盘或图形依赖。", "",
        "| 赛季 | 侧 | 0球 实际/期望 | 1球 | 2球 | 3球 | 4+ |", "|---|---|---|---|---|---|---|"]
    for season, data in result["periods"].items():
        for side in ("home", "away"):
            c = data["summaries"]["days_28"][side]["goal_counts"]
            lines.append(f"| {season} | {side} | " + " | ".join(
                f"{c[k]['observed']}/{c[k]['expected']:.2f}" for k in ("0", "1", "2", "3", "4+")) + " |")
    lines += ["", "## 分组与定位边界", "",
        "完整JSON保留全部臂的联赛、日历月、固定28天参照λ分箱；分箱(0,.75)、[.75,1)、[1,1.25)、"
        "[1.25,1.5)、[1.5,2)、[2,3)、[3,∞)，各臂按同一28天成员比较，空箱n=0且summary=null。"
        "这些组只作描述，无子组区间或多重比较覆盖保证，不按误差最大组调整参数。", "",
        "| 赛季 | 联赛 | 场次 | 主每场残差 | 客每场残差 | 平局残差 | 总≥4残差 |", "|---|---|---:|---:|---:|---:|---:|"]
    for season, data in result["periods"].items():
        for league, group in data["by_league"].items():
            d = group["summaries"]["days_28"]["scalar_descriptives"]
            lines.append(f"| {season} | {LEAGUES[league]} | {group['n']} | "
                + " | ".join(f"{d[f]['mean']:+.5f}" for f in
                    ("home_residual", "away_residual", "draw_residual", "total_four_plus_residual")) + " |")
    lines += ["", "## 时间相关与可复现性", "",
        f"各季以比赛为抽样单位，同一连续日历周权重共同作用于五联赛、三臂和主客全部指标；"
        f"2／4／8周非环形移动块各{result['settings']['repeats']}次，种子{SEED}，保留空周并按比赛数加权。"
        "主参考4周，其他为预定敏感性。区间条件于已拟合预测，不含重训／选参、模型方向选择或跨季未知性；"
        "端点权重、局部平稳近似、少量赛季和长期球队关联均有限制。逐场CSV保留三臂λ、比分、全部残差与NLL。", "",
        "统计分析技能使本轮采用逐场条件残差、同场差值和时间块，而不以总进球无条件方差／均值比"
        "或IID正态检验决定分布。独立算术复核另存；参数重放不等于重新求解优化器或证明数据商真实无误。", "",
        "方法依据："
        f"[SciPy Poisson概率与尾部]({SOURCES['poisson']})、"
        f"[数值稳定的缩放Bessel定义]({SOURCES['scaled_bessel']})、"
        f"[非环形移动块定义]({SOURCES['moving_blocks']})。只参考公式／方法，未复制上游源码、无新包／数据下载。", "",
        "## 下一步与未改变", "",
        "先依据本轮和较早历史定位均值／低比分／尾部误差，再设计只用更早数据学习的一致比分分布研究；"
        "不能按已查看后期直接拟合修正或据残差自动换负二项／Dixon–Coles。正式结构、球员先验或市场融合改变再与用户讨论。"
        "现有市场双轨、球员实验、未知整场跳过、特殊赛排除均未改；"
        "实时伤停／首发、欧战跨联赛桥梁、半全场联合过程和赛事投入差异仍未完成。"
        "这些历史诊断不构成2026/27即时预测、未核实12场赛事方向或串关获利证明。", ""]
    return "\n".join(lines)


def run(root, parent=None, *, repeats=REPEATS):
    root = Path(root).resolve()
    if type(repeats) is not int or repeats < 2:
        raise ValueError("Need at least two integer bootstrap repeats")
    if parent is None:
        candidates = sorted(p for p in (root / "outputs").glob("更新频率研究_*")
                            if (p / "更新频率研究结果.json").exists())
        if not candidates:
            raise ValueError("Need completed refresh research; no fallback fitting")
        parent = candidates[-1]
    parent = Path(parent).resolve()
    prior = _read_json(parent / "更新频率研究结果.json")
    if prior["settings"]["intervals_days"] != list(INTERVALS):
        raise ValueError("Parent intervals differ from the declared arms")
    output = root / "outputs" / ("进球率诊断_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    settings = {"parent_directory": str(parent), "intervals_days": list(INTERVALS),
        "main_arm": "days_28", "sensitivity_arms": list(CONTRASTS), "rate_bin_edges": list(RATE_EDGES),
        "rate_bin_convention": "fixed_days_28_membership_left_closed_right_open_zero_not_allowed_last_unbounded",
        "block_weeks": list(BLOCKS), "primary_block_weeks": 4, "repeats": repeats, "seed": SEED,
        "new_model_fits": 0, "new_model_predictions": 0, "model_selection": False,
        "probabilities_changed": False, "previously_viewed_seasons": list(PERIODS)}
    write_json(output / "分析设置.json", settings)
    directories = season_directories(root)
    periods, reports = {}, {}
    counts = {"parameter_snapshots_read": 0, "saved_prediction_rows_validated": 0,
              "common_arm_diagnostic_rows": 0, "unique_common_matches": 0, "saved_trials_read": 0}
    for period, (start, end) in PERIODS.items():
        selected = {season: path for season, path in directories.items() if int(season[:4]) <= start.year}
        records, reports[period] = load_seasons(selected)
        tracks, statistics = {arm: [] for arm in ARMS}, {arm: {} for arm in ARMS}
        for competition in sorted(LEAGUES):
            league = [row for row in records if row.competition == competition]
            for days in INTERVALS:
                arm = f"days_{days}"
                trial = _read_json(parent / period / f"{competition}_{arm}.json")
                validate_trial(trial, league, start, end, competition, interval=days)
                fits = {fit["before_date"]: fit for fit in trial["fits"]}
                for row in trial["predictions"]:
                    key = row["match_id"]
                    if key in statistics[arm]:
                        raise ValueError("Duplicate identity across saved league trials")
                    rates = rates_from_parameters(fits[row["fit_before_date"]], row)
                    statistics[arm][key] = row_statistics(row, rates)
                tracks[arm].extend(trial["predictions"])
                counts["parameter_snapshots_read"] += len(fits)
                counts["saved_prediction_rows_validated"] += len(trial["predictions"])
                counts["saved_trials_read"] += 1
                print(f"{period} {competition} {days}d: replayed {len(trial['predictions'])} saved rows", flush=True)
        data, entries, matched_statistics = analyze_period(tracks, statistics, repeats=repeats)
        if data["coverage"] != prior["periods"][period]["coverage"]:
            raise ValueError("Source common cohort changed since refresh research")
        periods[period] = data
        counts["unique_common_matches"] += data["n"]
        counts["common_arm_diagnostic_rows"] += len(ARMS) * data["n"]
        destination = output / period
        destination.mkdir()
        write_csv(destination / "逐场进球率与残差.csv", entries, matched_statistics)
        write_json(output / "阶段进度.json", {"completed_periods": list(periods), "counts": counts})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "parent_directory": str(parent), "counts": counts, "periods": periods, "data_reports": reports,
        "new_model_fits": 0, "new_model_predictions": 0, "probabilities_changed": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "method_sources": SOURCES}
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        saved = _read_json(args.report_only / RESULT_FILE)
        (args.report_only / REPORT_FILE).write_text(report(saved), encoding="utf-8")
    else:
        run(args.root, args.parent)
