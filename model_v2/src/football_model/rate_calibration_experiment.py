"""Earlier-only rate scaling with coherent full-score research predictions.

Two closed-form fits, no base refits and no production replacement. The later
season was already viewed; this is exploratory chronological application.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import poisson

from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, align_tracks, calibration_summary, losses, moving_block_means, validate_prediction, write_json
from .goal_rate_diagnostics import STAT_FIELDS, rates_from_parameters, row_statistics, summary as goal_summary
from .goal_rate_experiment import MOMENT_FIELDS, _read_json
from .native_experiment import TEST_START, load_seasons, season_directories
from .probability import score_grid, summarize
from .rate_calibration import METHODS, _day, apply_rate_calibrator, fit_rate_calibrator
from .refresh_experiment import prediction_arrays, scores_for
from .refresh_research import PERIODS, _same, validate_trial


ARMS = ("original", *METHODS)
RESULT_FILE = "进球率调整研究结果.json"
REPORT_FILE = "进球率调整研究说明.md"
FORECAST_FILE = "较晚研究预测.json"
DOMAINS = ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way", "exact_score")
SOURCES = {
    "offset": "https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html",
    "poisson": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html",
    "moving_blocks": "https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html",
}


def load_period(root, parent, period):
    """Validate the complete original 28-day trials against local CSV history."""
    start, end = PERIODS[period]
    directories = {season: path for season, path in season_directories(root).items()
                   if int(season[:4]) <= start.year}
    records, reports = load_seasons(directories)
    rows, snapshots = [], 0
    for comp in sorted(LEAGUES):
        trial = _read_json(Path(parent) / period / f"{comp}_days_28.json")
        validate_trial(trial, [r for r in records if r.competition == comp], start, end, comp, interval=28)
        fits = {fit["before_date"]: fit for fit in trial["fits"]}
        snapshots += len(fits)
        for row in trial["predictions"]:
            rates = rates_from_parameters(fits[row["fit_before_date"]], row)
            rows.append({**row, "rates": list(rates)})
    if len({r["match_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source identity across league trials")
    rows.sort(key=lambda row: (row["date"], row["match_id"]))
    return rows, {"source_reports": reports, "parameter_snapshots": snapshots,
                  "saved_prediction_rows": len(rows), "saved_trials": len(LEAGUES)}


def fit_earlier(rows):
    start, end = PERIODS["2024-25"]
    if not rows or any(not start <= date.fromisoformat(r["date"]) < end for r in rows):
        raise ValueError("Use exactly the declared earlier season for rate learning")
    return {method: fit_rate_calibrator(rows, TEST_START, method) for method in METHODS}


def _grid_metadata(rates):
    grid = score_grid(*rates, rho=0)
    maximum = grid.grid.shape[0] - 1
    h, a = (float(poisson.sf(maximum, value)) for value in rates)
    return grid, {"max_goal_inclusive": maximum,
        "omitted_tail_mass_before_normalization": h + a - h*a,
        "normalized_finite_grid": True, "rho": 0}


def research_tracks(rows, parameters, *, before_date=TEST_START):
    if set(parameters) != set(METHODS) or not rows:
        raise ValueError("Require both declared rate calibrators and source rows")
    cutoff = _day(before_date, "before_date").isoformat()
    tracks = {arm: [] for arm in ARMS}
    statistics = {arm: [] for arm in ARMS}
    exported = []
    for row in rows:
        validate_prediction(row)
        # Validate before NumPy conversion so bools/strings cannot become rates.
        base_stats = row_statistics(row, row["rates"])
        rates = np.asarray([base_stats["home_rate"], base_stats["away_rate"]])
        # Raw source is copied, never updated in place or replaced on disk.
        row_export = {key: row[key] for key in ("match_id", "date", "competition", "home", "away",
            "actual_score", "actual_outcome", "source_url")}
        row_export.update({"base_fit_before_date": row["fit_before_date"],
            "calibration_before_date": cutoff, "rates": {},
            "research_forecasts": {}, "grid_metadata": {}, "production_api_changed": False})
        for arm in ARMS:
            if arm == "original":
                adjusted = rates.copy()
            else:
                model = parameters[arm]
                if model["method"] != arm or model["before_date"] != cutoff:
                    raise ValueError("Rate calibrator method/cutoff differs from declared experiment")
                adjusted = apply_rate_calibrator(rates, model, row["date"])
            grid, metadata = _grid_metadata(adjusted)
            if arm == "original":
                # A mismatched standalone input must not combine old markets
                # with a different lambda/NLL; retain the archive on agreement.
                _same(row["forecast"], summarize(grid), "research_original_forecast")
                _same(row["probabilities"], [row["forecast"]["one_x_two"][key]
                    for key in ("home", "draw", "away")], "research_original_probabilities")
            forecast = row["forecast"] if arm == "original" else summarize(grid)
            research = {**row, "rates": adjusted.tolist(), "forecast": forecast,
                        "probabilities": [forecast["one_x_two"][key] for key in ("home", "draw", "away")]}
            tracks[arm].append(research)
            statistics[arm].append(row_statistics(row, adjusted))
            row_export["rates"][arm] = adjusted.tolist()
            row_export["research_forecasts"][arm] = forecast
            row_export["grid_metadata"][arm] = metadata
        exported.append(row_export)
    return tracks, statistics, exported


def _cohort(tracks, statistics, indices):
    entries = {arm: [tracks[arm][i] for i in indices] for arm in ARMS}
    stats = {arm: [statistics[arm][i] for i in indices] for arm in ARMS}
    scores = {arm: {**scores_for(entries[arm]), "exact_score": {"n": len(indices),
        "mean_nll": float(np.mean([r["joint_score_nll"] for r in stats[arm]]))}} for arm in ARMS}
    def value(arm, domain):
        return scores[arm][domain]["mean_nll" if domain == "exact_score" else "log_loss"]
    return {"n": len(indices), "scores": scores,
        "goal_summaries": {arm: goal_summary(entries[arm], stats[arm]) for arm in ARMS},
        "mean_differences": {f"{method}_minus_original": {
            domain: value(method, domain) - value("original", domain) for domain in DOMAINS} for method in METHODS}}


def loss_vectors(tracks, statistics):
    result = {}
    for arm in ARMS:
        result[arm] = {domain: losses(p, y) for domain, (p, y) in prediction_arrays(tracks[arm]).items()}
        result[arm]["exact_score"] = np.asarray([s["joint_score_nll"] for s in statistics[arm]])
    return result


def analyze_later(raw_tracks, raw_statistics, *, repeats=REPEATS):
    if set(raw_tracks) != set(ARMS) or set(raw_statistics) != set(ARMS):
        raise ValueError("Need all three declared rate research arms")
    normalized = json.loads(json.dumps(raw_tracks, allow_nan=False))
    matched, coverage = align_tracks(normalized)
    if any(coverage["outside_common"][arm] for arm in ARMS):
        raise ValueError("Rate research must preserve full source coverage for every arm")
    # Statistics are supplied in original input order; explicitly map IDs before sorting.
    statistics = {}
    for arm in ARMS:
        goal_summary(normalized[arm], raw_statistics[arm])
        mapped = {row["match_id"]: s for row, s in zip(normalized[arm], raw_statistics[arm])}
        statistics[arm] = [mapped[key] for key in coverage["common_match_ids"]]
    tracks = {arm: list(track.values()) for arm, track in matched.items()}
    rows = tracks["original"]
    result = _cohort(tracks, statistics, list(range(len(rows))))
    result.update({"match_ids": coverage["common_match_ids"], "coverage": coverage,
        "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "calibration": {arm: calibration_summary(*prediction_arrays(tracks[arm])["one_x_two"]) for arm in ARMS}})
    per_loss = loss_vectors(tracks, statistics)
    features = {f"{method}_minus_original_{domain}": per_loss[method][domain] - per_loss["original"][domain]
                for method in METHODS for domain in DOMAINS}
    for arm in ARMS:
        for field in MOMENT_FIELDS:
            features[f"{arm}_{field}"] = np.asarray([s[field] for s in statistics[arm]])
        p, y = prediction_arrays(tracks[arm])["one_x_two"]
        for c, label in enumerate(("home", "draw", "away")):
            features[f"{arm}_{label}_probability_minus_frequency"] = p[:, c] - (y == c)
    result["bootstrap"] = {}
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in rows], np.column_stack(list(features.values())),
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["features"] = {key: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, key in enumerate(features)}
        result["bootstrap"][str(weeks)] = boot
    for name, labels in (("by_league", [r["competition"] for r in rows]),
                         ("by_month", [r["date"][:7] for r in rows])):
        result[name] = {label: _cohort(tracks, statistics, [i for i, value in enumerate(labels) if value == label])
                        for label in sorted(set(labels))}
    result["markets"] = {}
    for market in MARKET_COLUMNS:
        indices = [i for i, row in enumerate(rows) if market in row["market_probabilities"]]
        missing = [r["match_id"] for r in rows if market not in r["market_probabilities"]]
        if not indices:
            result["markets"][market] = {"n": 0, "missing_ids": missing}
            continue
        p = np.asarray([rows[i]["market_probabilities"][market] for i in indices])
        y = np.asarray([rows[i]["actual_outcome"] for i in indices])
        market_loss = losses(p, y)
        differences = {arm: per_loss[arm]["one_x_two"][indices] - market_loss for arm in ARMS}
        boot = moving_block_means([rows[i]["date"] for i in indices], np.column_stack(list(differences.values())),
                                  block_weeks=4, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        result["markets"][market] = {"n": len(indices), "match_ids": [rows[i]["match_id"] for i in indices],
            "missing_ids": missing, "market_scores": probability_scores(p, y),
            "arm_scores_on_same_matches": {arm: probability_scores(
                prediction_arrays(tracks[arm])["one_x_two"][0][indices], y) for arm in ARMS},
            "differences": {arm: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, arm in enumerate(ARMS)}, "bootstrap_metadata": boot}
    return result, tracks, statistics


def write_csv(path, tracks, statistics):
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals", "base_fit_before_date", "calibration_before_date"]
    fields += [f"{arm}_{key}" for arm in ARMS for key in
               (*STAT_FIELDS, "p_home", "p_draw", "p_away", *(f"{d}_nll" for d in DOMAINS))]
    per_loss = loss_vectors(tracks, statistics)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(tracks["original"]):
            item = {key: row[key] for key in fields[:5]}
            item.update({"home_goals": row["actual_score"][0], "away_goals": row["actual_score"][1],
                "base_fit_before_date": row["fit_before_date"],
                "calibration_before_date": row.get("calibration_before_date", TEST_START.isoformat())})
            for arm in ARMS:
                item.update({f"{arm}_{key}": value for key, value in statistics[arm][i].items()})
                item.update({f"{arm}_p_{label}": tracks[arm][i]["probabilities"][j]
                             for j, label in enumerate(("home", "draw", "away"))})
                item.update({f"{arm}_{domain}_nll": float(per_loss[arm][domain][i]) for domain in DOMAINS})
            writer.writerow(item)


def _ci(value):
    return "不可估计" if value is None else f"[{value[0]:+.6g}, {value[1]:+.6g}]"


def report(result):
    later, counts = result["later"], result["counts"]
    lines = ["# 较早学习、跨季固定应用的进球率调整研究", "",
        f"本轮实际完成2次闭式进球率校准拟合，共3个标量参数。读取{counts['source_parameter_snapshots_read']}份已有球队参数，"
        "0次基础模型重训。较早学习样本" + str(result["earlier"]["n"]) + "场，较晚同场检验" + str(later["n"]) + "场；"
        f"生成{counts['new_research_forecasts']}条独立研究forecast，原{counts['reused_later_source_forecasts']}条原样复用。"
        "正式模型／原始存档不改，全部新文件和临时文件在项目目录。", "",
        "## 设计与时间边界", "",
        "只用2024/25旧28天轨的时间外基础λ及真实比分，跨五联赛逐场等权学习；"
        "校准截止2025-08-01，拟合函数执行后才加载2025/26原始CSV和预测。"
        "校准器不继承或二次叠加基础模型的180天时间权重。基础球队参数仍沿用原存档的28天严格更早日期滚动更新，"
        "不是将整季球队实力固定。固定的是两个校准方案的尺度系数。", "",
        "原样、共享尺度、分侧尺度全部保留。主对照分侧−原样，共享为预定简单对照；"
        "不按后期选择赢家，不扩参数网格、不新增联赛专属系数、市场融合、球员先验或战意猜测。"
        "两季此前均已查看，这是探索性按时间外推应用，不是新盲测；截点是回顾数据边界，不代表当年实际发布时刻。", "",
        "## 模型与学到的参数", "",
        "共享：λ′主=cλ主、λ′客=cλ客，c=Σ两侧真实进球／Σ两侧基础λ。"
        "分侧：c主=Σ主进球／Σλ主，c客同理。相当于把logλ作为固定offset学习截距；"
        "等权独立Poisson精确比分NLL的闭式解，无优化器、正则或市场输入。新Poisson方差为新λ，不是c²λ。"
        "如目标进球和为零，严格正有限尺度MLE不存在，明确报错而不加epsilon。", "",
        "| 方法 | 主尺度 | 客尺度 | 学习场次 | 原训练比分NLL | 调后训练比分NLL |", "|---|---:|---:|---:|---:|---:|"]
    for method, p in result["parameters"].items():
        lines.append(f"| {method} | {p['parameters']['scales'][0]:.8f} | {p['parameters']['scales'][1]:.8f} | {p['n_matches']} | "
            f"{p['raw_training_analytic_joint_score_nll']['mean']:.6f} | {p['calibrated_training_analytic_joint_score_nll']['mean']:.6f} |")
    lines += ["", "训练内NLL下降及共享总均值／分侧各侧均值平衡是拟合代数性质，不是验证成功。"
        "尺度保持同一侧所有比赛λ的相对排序，不能补回真实首发、伤停、赛事投入或球队特有变化。", "",
        "## 当前研究读数", ""]
    for method in METHODS:
        diff = later["mean_differences"][method+"_minus_original"]
        lines.append(f"{method}相对原样：胜平负LL差{diff['one_x_two']:+.6g}，"
                     f"精确比分NLL差{diff['exact_score']:+.6g}。负差较好；完整目标与区间见下，不据此自动选模型。")
    lines += ["", "先看跨季方向与多目标代价，再判断尺度调整是否有用途；"
        "不把原均值偏差逆向拟合后训练内归零当成稳定规律。所有数值为本批历史研究，不代表即时概率。", "",
        "## 较晚赛季同场表现", "",
        "| 方案 | 胜平负LL | Brier sum | RPS | 总进球五箱LL | 客让二球三项LL | 精确比分NLL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, scores in later["scores"].items():
        s = scores["one_x_two"]
        lines.append(f"| {arm} | {s['log_loss']:.6f} | {s['brier_sum']:.6f} | {s['rps']:.6f} | "
            f"{scores[DOMAINS[1]]['log_loss']:.6f} | {scores[DOMAINS[2]]['log_loss']:.6f} | {scores['exact_score']['mean_nll']:.6f} |")
    lines += ["", "同一新λ重新生成独立Poisson完整比分网格，胜平负、总进球、双方进球、让球均来自同一分布；"
        "不混入上一轮独立1X2校准向量。网格为归一化有限近似，每条记录保存最大球数与归一化前遗漏尾部质量。"
        "真实λ另存，精确比分NLL在log域计算，不从网格截断／裁剪得出。不同评分目标尺度不同，不能横比LL大小。"
        "总进球五类为0／1／2／3／4+；客让二球让胜需净胜至少3，净胜2为三项让平，不是亚洲盘走盘。", "",
        "## 调整−原样：全部目标与区间", "",
        "| 对照 | 目标 | 平均NLL差 | 2周95%区间 | 4周 | 8周 |", "|---|---|---:|---|---|---|"]
    for method in METHODS:
        for domain in DOMAINS:
            key = f"{method}_minus_original_{domain}"
            lines.append(f"| {method}−原样 | {domain} | {later['mean_differences'][method+'_minus_original'][domain]:+.6g} | "
                + " | ".join(_ci(later["bootstrap"][str(w)]["features"][key]["interval_95"]) for w in BLOCKS) + " |")
    lines += ["", "负差较好。所有目标／块长均保留，不按结果筛选；区间跨零不证明等效，不跨零也不是反复探索后的确认性结论。", "",
        "## 均值、平局与尾部是否同时改善", "",
        "残差一律实际−预测，正值表示低估；事件以百分点展示，进球残差是球／场。"
        "均值平衡不是条件分布充分校准，零进球、平局和尾部不能按需要单独改写。", "",
        "| 方案 | 主进球残差 | 客进球残差 | 平局残差百分点 | 0:0 | 1:1 | 总≥4 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        d = later["goal_summaries"][arm]["scalar_descriptives"]
        lines.append(f"| {arm} | {d['home_residual']['mean']:+.5f} | {d['away_residual']['mean']:+.5f} | "
            + " | ".join(f"{100*d[key]['mean']:+.3f}" for key in
                ("draw_residual", "zero_zero_residual", "one_one_residual", "total_four_plus_residual")) + " |")
    lines += ["", "完整JSON保留24项逐场条件统计的n、均值／SD／极值／中位数，实际与逐场期望0／1／2／3／4+计数、"
        "全部三臂主客残差／低比分／尾部时间块区间、1X2校准箱及联赛／月份描述。"
        "Pearson²−1不是纯离散度估计，主客残差乘积不是相关系数；分组不作因果解释或用于后期再调参。", "",
        "## 市场双轨（同场1X2）", "",
        "| 市场字段 | 场次 | 市场LL | 原样LL | 共享LL | 分侧LL |", "|---|---:|---:|---:|---:|---:|"]
    for market, data in later["markets"].items():
        if data["n"]:
            lines.append(f"| {market} | {data['n']} | {data['market_scores']['log_loss']:.6f} | "
                + " | ".join(f"{data['arm_scores_on_same_matches'][arm]['log_loss']:.6f}" for arm in ARMS) + " |")
    lines += ["", "赔率倒数归一化只作去水对照，不作真概率或训练目标。pre-closing不等于核实开盘／T−24h，"
        "closing可能含更晚信息，均无独立采集时间；聚合均价未必可成交。缺失分别列出，不能据历史LL推导可执行串关收益。", "",
        "## 不确定性与复现", "",
        f"较晚{later['n']}场是抽样单位，三臂／五联赛／所有指标共同使用2／4／8周非环形日历块，各"
        f"{result['settings']['repeats']}次，种子{SEED}；保留空周并按比赛加权。主参考4周，其余为敏感性。"
        "区间条件于已有基础参数和校准系数，不含校准估计／重训／模型方向选择、跨季未知性或单场参数误差。"
        "端点抽样、短程依赖／局部平稳近似和长期球队关联有限制，无同时覆盖或确认性p值。", "",
        "实验设计技能使方案保持小规模预定、严格较早拟合和同场完整对照；统计分析技能使报告同时保留多目标差值、"
        "原始分布描述及条件时间相关区间，不以训练内均值平衡、IID检验或单个显著子组宣布成功。", "",
        "方法依据："
        f"[GLM固定offset含义]({SOURCES['offset']})、[Poisson概率]({SOURCES['poisson']})、"
        f"[非环形移动块]({SOURCES['moving_blocks']})。闭式解由本报告Poisson目标直接求导，未复制上游源码或安装statsmodels。", "",
        "## 未改变与下一步", "",
        "两个研究方案不自动进入正式API；重大结构、球员先验或市场融合改变再讨论。"
        "先核对本轮跨季稳定性和逐目标代价，再决定下一项只用更早资料的研究，不拿后期结果直接回调系数。"
        "实时首发伤停、球员赛前衔接、欧战跨联赛桥梁、赛事投入不确定性和半全场联合过程仍未完成。"
        "本轮是历史概率研究，不是2026/27即时12场预测或获利保证。", ""]
    return "\n".join(lines)


def run(root, parent=None, *, repeats=REPEATS):
    root = Path(root).resolve()
    if type(repeats) is not int or repeats < 2:
        raise ValueError("Need at least two integer repeats")
    if parent is None:
        candidates = sorted(p for p in (root / "outputs").glob("更新频率研究_*")
                            if (p / "更新频率研究结果.json").exists())
        if not candidates:
            raise ValueError("Need completed refresh archives, no fallback training")
        parent = candidates[-1]
    parent = Path(parent).resolve()
    output = root / "outputs" / ("进球率调整研究_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    settings = {"parent_directory": str(parent), "base_interval_days": 28, "methods": list(METHODS),
        "calibration_before_date": TEST_START.isoformat(), "training_period": "2024-25", "application_period": "2025-26",
        "primary_contrast": "side_scale_minus_original", "secondary_contrast": "shared_scale_minus_original",
        "training_weighting": "equal_match_no_extra_decay", "new_base_fits": 0,
        "model_selection": False, "market_fusion": False, "production_api_changed": False,
        "block_weeks": list(BLOCKS), "primary_block_weeks": 4, "repeats": repeats, "seed": SEED,
        "previously_viewed_seasons": list(PERIODS)}
    write_json(output / "实验设置.json", settings)
    earlier_rows, earlier_source = load_period(root, parent, "2024-25")
    parameters = fit_earlier(earlier_rows)
    write_json(output / "进球率校准参数.json", parameters)
    print(f"Completed two closed-form rate calibrators on {len(earlier_rows)} earlier matches", flush=True)
    # Do not even load later CSV results until the two earlier fits are finished.
    later_rows, later_source = load_period(root, parent, "2025-26")
    if set(r["match_id"] for r in earlier_rows) & set(r["match_id"] for r in later_rows):
        raise ValueError("Earlier and later match identities overlap")
    raw_tracks, statistics, exported = research_tracks(later_rows, parameters)
    later, tracks, statistics = analyze_later(raw_tracks, statistics, repeats=repeats)
    earlier_stats = [row_statistics(r, r["rates"]) for r in earlier_rows]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "parent_directory": str(parent), "parameters": parameters,
        "counts": {"new_rate_calibrator_fits": 2, "new_scalar_rate_parameters": 3, "new_base_model_fits": 0,
            "source_parameter_snapshots_read": earlier_source["parameter_snapshots"] + later_source["parameter_snapshots"],
            "source_prediction_rows": len(earlier_rows) + len(later_rows),
            "new_research_forecasts": len(METHODS) * len(later_rows), "reused_later_source_forecasts": len(later_rows)},
        "earlier": {"n": len(earlier_rows), "match_ids": [r["match_id"] for r in earlier_rows],
            "first_date": earlier_rows[0]["date"], "last_date": earlier_rows[-1]["date"],
            "raw_goal_summary": goal_summary(earlier_rows, earlier_stats),
            "calibrated_training_scores_not_temporal_validation": True},
        "later": later, "source_loading": {"earlier": earlier_source, "later": later_source},
        "grid_approximation": {arm: {"max_omitted_tail_mass_before_normalization": max(
            r["grid_metadata"][arm]["omitted_tail_mass_before_normalization"] for r in exported)} for arm in ARMS},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "method_sources": SOURCES, "production_api_changed": False}
    write_json(output / FORECAST_FILE, exported)
    write_csv(output / "较晚逐场概率与进球残差.csv", tracks, statistics)
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": result["counts"],
                      "mean_differences": later["mean_differences"]}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        value = _read_json(args.report_only / RESULT_FILE)
        (args.report_only / REPORT_FILE).write_text(report(value), encoding="utf-8")
    else:
        run(args.root, args.parent)
