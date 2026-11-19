"""Earlier-only calibration research on the fixed original pre-match model.

Three planned arms, no winner selection, market blending or production mutation.
Calibrated 1X2 vectors are separate research outputs, NOT coherent score grids.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import (BLOCKS, CLASS_NAMES, CLASS_LABELS, REPEATS, SEED,
    calibration_summary, load_saved_tracks, losses, moving_block_means, validate_prediction, write_json)
from .native_experiment import (ORIGINAL, VALID_START, VALID_END, TEST_START, TEST_END,
    expected_target_ids, load_seasons, run_config, season_directories)
from .prematch_calibration import apply_calibrator, fit_calibrator


METHODS = ("class_bias", "temperature")
ARM_NAMES = {"original": "原样", "class_bias": "类别偏置（2参数）", "temperature": "单温度（1参数）"}
RESULT_FILE = "较早校准研究结果.json"
REPORT_FILE = "较早校准研究说明.md"


def ordered_rows(rows):
    if not rows or len({r["match_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty unique prediction rows")
    for row in rows:
        validate_prediction(row)
    return sorted(rows, key=lambda r: (r["date"], r["match_id"]))


def make_arms(rows, parameters):
    """Do not copy/overwrite forecast.one_x_two or any score-derived fields."""
    if set(parameters) != set(METHODS):
        raise ValueError("Require both predeclared research methods")
    raw = np.asarray([r["probabilities"] for r in rows], dtype=float)
    arms = {"original": raw.copy()}
    for method in METHODS:
        model = parameters[method]
        if model["method"] != method:
            raise ValueError("Calibrator label mismatch")
        cutoff = date.fromisoformat(model["before_date"])
        if date.fromisoformat(model["training_max_date"]) >= cutoff:
            raise ValueError("Calibrator training must precede its cutoff")
        if any(date.fromisoformat(r["date"]) < cutoff for r in rows):
            raise ValueError("Cannot apply later-fitted calibrator to earlier forecast dates")
        arms[method] = apply_calibrator(raw, model)
    return arms


def analyze_arms(rows, arms, *, repeats=REPEATS, seed=SEED):
    """Same matches and shared temporal resampling weights across all arms."""
    if not rows or set(arms) != set(ARM_NAMES):
        raise ValueError("Need matches and three declared arms")
    if len({r["match_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs in analysis")
    for row in rows:
        validate_prediction(row)
    arms = {name: np.asarray(p, dtype=float) for name, p in arms.items()}
    y = np.asarray([r["actual_outcome"] for r in rows], dtype=int)
    dates = [r["date"] for r in rows]
    n = len(rows)
    # Public score validation also rejects invalid calibrated probability arrays.
    scores = {name: probability_scores(p, y) for name, p in arms.items()}
    per_loss = {name: losses(p, y) for name, p in arms.items()}
    delta = {f"{method}_minus_original": per_loss[method] - per_loss["original"] for method in METHODS}
    features = dict(delta)
    truth = np.eye(3)[y]
    for name, p in arms.items():
        for c, outcome in enumerate(CLASS_NAMES):
            features[f"{name}_{outcome}_bias"] = p[:, c] - truth[:, c]
    names = list(features)
    values = np.column_stack(list(features.values()))
    bootstrap = {}
    for length in BLOCKS:
        boot = moving_block_means(dates, values, block_weeks=length, repeats=repeats, seed=seed)
        intervals, means = boot.pop("interval_95"), boot.pop("mean")
        boot["features"] = {name: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, name in enumerate(names)}
        bootstrap[str(length)] = boot
    groups = {}
    for grouping, labels in (("by_league", [r["competition"] for r in rows]),
                             ("by_month", [r["date"][:7] for r in rows])):
        a = np.asarray(labels)
        groups[grouping] = {label: {"n": int((a == label).sum()),
            "scores": {name: probability_scores(p[a == label], y[a == label]) for name, p in arms.items()},
            "mean_differences": {name: float(v[a == label].mean()) for name, v in delta.items()}}
            for label in sorted(set(labels))}
    markets = {}
    for market in MARKET_COLUMNS:
        indices = [i for i, row in enumerate(rows) if market in row["market_probabilities"]]
        missing = [r["match_id"] for r in rows if market not in r["market_probabilities"]]
        if not indices:
            markets[market] = {"n": 0, "missing_ids": missing}
            continue
        p = np.asarray([rows[i]["market_probabilities"][market] for i in indices])
        market_loss = losses(p, y[indices])
        diffs = {name: per_loss[name][indices] - market_loss for name in arms}
        boot = moving_block_means([dates[i] for i in indices], np.column_stack(list(diffs.values())),
                                   block_weeks=4, repeats=repeats, seed=seed)
        intervals = boot.pop("interval_95")
        means = boot.pop("mean")
        markets[market] = {"n": len(indices), "match_ids": [rows[i]["match_id"] for i in indices],
            "missing_ids": missing, "market_scores": probability_scores(p, y[indices]),
            "arm_scores_on_identical_matches": {name: probability_scores(q[indices], y[indices]) for name, q in arms.items()},
            "differences": {name: {"mean": means[j], "interval_95": None if intervals is None else intervals[j]}
                            for j, name in enumerate(diffs)}, "bootstrap_metadata": boot}
    return {"n": n, "match_ids": [r["match_id"] for r in rows], "first_date": min(dates), "last_date": max(dates),
        "scores": scores, "calibration": {name: calibration_summary(p, y) for name, p in arms.items()},
        "contrasts": {name: {"mean": float(v.mean()), "by_actual_outcome": {outcome: {
            "n": int((y == c).sum()), "weighted_contribution": float(np.where(y == c, v, 0).mean())}
            for c, outcome in enumerate(CLASS_NAMES)}} for name, v in delta.items()},
        "bootstrap": bootstrap, "markets": markets, **groups}


def raw_summary(rows):
    y = np.asarray([r["actual_outcome"] for r in rows], dtype=int)
    p = np.asarray([r["probabilities"] for r in rows])
    return {"n": len(rows), "first_date": min(r["date"] for r in rows), "last_date": max(r["date"] for r in rows),
        "scores": probability_scores(p, y), "calibration": calibration_summary(p, y),
        "by_league": {comp: {"n": sum(r["competition"] == comp for r in rows),
            "calibration": calibration_summary(p[[r["competition"] == comp for r in rows]],
                                               y[[r["competition"] == comp for r in rows]])}
            for comp in sorted({r["competition"] for r in rows})}}


def probability_histogram(rows):
    """Raw earlier out-of-time probability histogram, before calibrator fitting."""
    p = np.asarray([r["probabilities"] for r in rows])
    counts = [np.histogram(p[:, c], bins=np.linspace(0, 1, 11))[0] for c in range(3)]
    maximum = max(int(x.max()) for x in counts)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="300" viewBox="0 0 960 300">',
             '<rect width="960" height="300" fill="white"/>',
             f'<text x="25" y="24" font-size="16">Earlier original 1X2 probabilities; n={len(rows)} (not independent class samples)</text>']
    for c, name in enumerate(CLASS_NAMES):
        left = 30 + 315 * c
        lines.append(f'<text x="{left}" y="50" font-size="16">{name}; common y maximum {maximum}</text>')
        for b, count in enumerate(counts[c]):
            height = 160 * int(count) / maximum
            lines.append(f'<rect x="{left+27*b}" y="{225-height:.2f}" width="22" height="{height:.2f}" fill="#356b8c"/>')
        for b, label in ((0, "0"), (5, "0.5"), (10, "1")):
            lines.append(f'<text x="{left+27*b}" y="245" font-size="12">{label}</text>')
    lines += ['<text x="25" y="280" font-size="13">x=predicted probability, equal-width 0.1 bins; height=count; raw observations retained</text>', '</svg>']
    return "\n".join(lines)


def fit_earlier_original(root, output):
    # Read only 2022/23–2024/25 CSVs; later outcomes are not calibrator inputs.
    directories = {s: p for s, p in season_directories(root).items() if s != "2025/26"}
    records, source_reports = load_seasons(directories)
    all_rows, coverage, fits = [], {}, 0
    for comp in sorted(LEAGUES):
        league = [r for r in records if r.competition == comp]
        trial = run_config(league, ORIGINAL, VALID_START, VALID_END)
        write_json(output / "较早旧模型预测" / f"{comp}_original.json", trial)
        expected = expected_target_ids(league, VALID_START, VALID_END)
        actual = {r["match_id"] for r in trial["predictions"]}
        if trial["failures"] or len(actual) != len(trial["predictions"]) or actual != expected:
            raise ValueError(f"Incomplete earlier original forecasts for {comp}; saved details, no silent cohort reduction")
        fits += len(trial["fits"])
        all_rows.extend(trial["predictions"])
        coverage[comp] = {"predicted": len(actual), "fits": len(trial["fits"]),
            "skipped": trial["skipped"], "failures": trial["failures"]}
        print(f"earlier original {comp}: {len(actual)} predictions, {len(trial['fits'])} fits", flush=True)
    return ordered_rows(all_rows), {"sources": source_reports, "leagues": coverage, "new_base_model_fits": fits}


def fit_research_calibrators(earlier):
    rows = ordered_rows(earlier)
    if any(not VALID_START <= date.fromisoformat(r["date"]) < VALID_END for r in rows):
        raise ValueError("Calibration training must use the declared earlier forecast season only")
    return {method: fit_calibrator(rows, TEST_START, method) for method in METHODS}


def research_predictions(rows, arms):
    """Standalone vectors, with explicit refusal of a score-grid interpretation."""
    return [{**{key: r[key] for key in ("match_id", "date", "competition", "home", "away", "actual_score", "actual_outcome")},
        "actual_score": list(r["actual_score"]), "base_fit_before_date": r["fit_before_date"],
        "research_one_x_two": {name: p[i].tolist() for name, p in arms.items()},
        "not_a_calibrated_score_grid": True} for i, r in enumerate(rows)]


def write_csv(path, rows, arms):
    fields = ["match_id", "date", "competition", "actual_outcome", "base_fit_before_date"]
    fields += [f"{name}_p_{c}" for name in arms for c in CLASS_NAMES]
    fields += [f"{name}_log_loss" for name in arms]
    y = np.asarray([r["actual_outcome"] for r in rows], dtype=int)
    per_loss = {name: losses(p, y) for name, p in arms.items()}
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            row = {k: r[k] for k in fields[:4]}
            row["base_fit_before_date"] = r["fit_before_date"]
            for name, p in arms.items():
                row.update({f"{name}_p_{c}": float(p[i, j]) for j, c in enumerate(CLASS_NAMES)})
                row[f"{name}_log_loss"] = float(per_loss[name][i])
            writer.writerow(row)


def report(result):
    later, earlier = result["later"], result["earlier_raw"]
    fmt = lambda a: "不可估计" if a is None else f"[{a[0]:+.6f}, {a[1]:+.6f}]"
    lines = ["# 只用较早数据学习的1X2校准研究", "",
        f"新增旧模型基础拟合{result['counts']['new_base_model_fits']}次、校准拟合2次。"
        f"较早预测{earlier['n']}场；较晚{later['n']}场原始概率读取既有存档，0次重训较晚基础模型。"
        "所有新文件、模型参数、测试临时文件在项目目录。", "",
        "## 设计与时间顺序", "",
        "固定旧自建模型ridge=0.01、半衰期180天，每28天仅使用严格更早日期赛果更新。"
        "补齐2024/25时间外基础预测，再以其真实赛果拟合跨五联赛、逐场等权共享校准参数。"
        "两校准器均截止2025-08-01；2025/26全季不再更新校准器，基础模型仍沿用既有逐时更新。", "",
        "2025-08-01是回顾研究的数据截点，不是声称当年实际运行或发布过这些校准预测；本轮真实执行时间另存UTC字段。", "",
        "原样、两参数类别偏置、单温度三臂全部保留。主对照为类别偏置−原样，温度为预定描述对照；"
        "不按已查看的2025/26选赢家，不扫描正则／网格，不用市场或球员路径拟合校准。"
        "2024/25全季选定的native配置不是当年逐时已选模型，本轮不拿它训练校准器。", "",
        "基础预测的每个日期都晚于其训练结果；校准器只读取较早赛季的时间外预测与标签，"
        "不会把基础模型训练内概率当校准训练样本。无随机打乱的交叉验证。", "",
        "## 早期与后期原始偏差是否同向", "",
        "偏差=平均预测概率−实际频率（百分点）。相同方向不等于稳定规律；后期已查看，仍是探索性研究。", "",
        "| 赛季／阶段 | 场次 | 主胜偏差 | 平局偏差 | 客胜偏差 |", "|---|---:|---:|---:|---:|"]
    for label, n, calibration in (("2024/25校准学习样本", earlier["n"], earlier["calibration"]),
                                   ("2025/26较晚原样", later["n"], later["calibration"]["original"])):
        gaps = [100*calibration["classes"][c]["mean_probability_minus_frequency"] for c in CLASS_NAMES]
        lines.append(f"| {label} | {n} | " + " | ".join(f"{g:+.3f}" for g in gaps) + " |")
    lines += ["", "## 学到的参数与训练内评分", "",
        "类别偏置：q=softmax(log(p)+[b主,b平,0])；温度：q=softmax(β·log(p))，β=1/T≥0。"
        "β=0时为均匀分布极限，不伪造有限T；本轮无联合温度＋偏置、无联赛专属参数。", "",
        "| 方法 | 参数 | 学习场次 | 原始学习LL | 校准后学习LL |", "|---|---|---:|---:|---:|"]
    for method, p in result["parameters"].items():
        lines.append(f"| {ARM_NAMES[method]} | {json.dumps(p['parameters'], ensure_ascii=False)} | {p['n_matches']} | "
                     f"{p['raw_training_log_loss']:.6f} | {p['calibrated_training_log_loss']:.6f} |")
    lines += ["", "训练内LL下降与类别偏置下的训练均值平衡是拟合结果，不是时间外验证成功。"
        "数值优化状态／梯度与全部训练IDs保存，不因优化器成功就宣称统计最优或未来有效。", "",
        "## 较晚同场成绩", "", "| 研究臂 | 场次 | Log loss | Brier sum | RPS | 最高概率结果命中率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, s in later["scores"].items():
        lines.append(f"| {ARM_NAMES[name]} | {s['n']} | {s['log_loss']:.6f} | {s['brier_sum']:.6f} | "
                     f"{s['rps']:.6f} | {s['accuracy']:.2%} |")
    lines += ["", "差值负数较好；下列为已拟合基础预测和校准器条件下的探索性95%区间。", "",
        "| 对照 | 平均LL差 | 2周块 | 4周块 | 8周块 |", "|---|---:|---|---|---|"]
    for method in METHODS:
        name = f"{method}_minus_original"
        intervals = [fmt(later["bootstrap"][str(w)]["features"][name]["interval_95"]) for w in BLOCKS]
        lines.append(f"| {method}−原样 | {later['contrasts'][name]['mean']:+.6f} | " + " | ".join(intervals) + " |")
    bias_difference = later["contrasts"]["class_bias_minus_original"]["mean"]
    temperature_difference = later["contrasts"]["temperature_minus_original"]["mean"]
    lines += ["", f"本批样本类别偏置的LL差{bias_difference:+.6f}，温度的LL差{temperature_difference:+.6f}。"
        "不能把微小点估计变化当作稳定收益；尤其要同时看区间、Brier、RPS及各类别偏差。"
        "不据此选赢家，也没有证明两臂与原样等效。"]
    lines += ["", "## 较晚各类别偏差", "", "| 研究臂 | 类别 | 平均概率 | 实际频率 | 偏差（百分点） | 4周区间（概率单位） |",
        "|---|---|---:|---:|---:|---|"]
    for name, cal in later["calibration"].items():
        for c, label in zip(CLASS_NAMES, CLASS_LABELS):
            gap = cal["classes"][c]
            ci = later["bootstrap"]["4"]["features"][f"{name}_{c}_bias"]["interval_95"]
            lines.append(f"| {ARM_NAMES[name]} | {label} | {gap['mean_probability']:.2%} | {gap['observed_frequency']:.2%} | "
                         f"{100*gap['mean_probability_minus_frequency']:+.3f} | {fmt(ci)} |")
    lines += ["", "总体偏差变小不等于充分校准，也不等于Log loss／排名更好。完整5箱人数、分联赛／月份与赛果贡献均保留。", "",
        "| 联赛 | 场次 | 类别偏置−原样 | 温度−原样 |", "|---|---:|---:|---:|"]
    for comp, row in later["by_league"].items():
        lines.append(f"| {LEAGUES[comp]} | {row['n']} | {row['mean_differences']['class_bias_minus_original']:+.6f} | "
                     f"{row['mean_differences']['temperature_minus_original']:+.6f} |")
    lines += ["", "## 市场双轨", "",
        "校准学习不依赖市场是否缺失；下列市场比较才按其各自可用的共同比赛对齐。", "",
        "| 市场 | 场次 | 市场LL | 原样LL | 类别偏置LL | 温度LL |", "|---|---:|---:|---:|---:|---:|"]
    for market, m in later["markets"].items():
        if not m["n"]:
            lines.append(f"| {market} | 0 | — | — | — | — |")
            continue
        values = [m["arm_scores_on_identical_matches"][a]["log_loss"] for a in ARM_NAMES]
        lines.append(f"| {market} | {m['n']} | {m['market_scores']['log_loss']:.6f} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## 解释边界与尚未采用的变更", "",
        "2025/26此前已查看。本轮研究方向由之前的误差诊断启发，即使校准只用更早数据，"
        "也不能把同一后期赛季重新称为新盲测；全部三臂均报告，不因排名改正式模型。"
        "五联赛合并参数只是这项低维研究假设，不代表各联赛可互换或欧战已验证。", "",
        f"时间块设置沿用2／4／8个连续日历周，每种{result['settings']['repeats']}次，种子{SEED}。"
        "同周五联赛与三臂共用权重，空周保留，按比赛数加权；非环形拼接截到原周跨度。"
        "端点抽样偏少，短赛季与球队长期关联仍限制解释。区间不含校准参数估计、基础模型重训／选择、"
        "跨赛季漂移的不确定性，更不是单场概率区间；各区间没有同时覆盖保证，不作确认性显著声明。", "",
        "输出`research_one_x_two`是独立研究向量：未覆盖原`forecast.one_x_two`、总进球、让球、半全场或比分网格。"
        "不能把校准1X2与旧比分分布混用计算投注。若未来采用，需要另行讨论一致联合分布与真正新时期检验。", "",
        "赔率无精确采集时间，pre-closing不等于开盘／T−24h，均价不保证可成交；市场不是标签或真实概率。"
        "无市场融合、未知球队／球员默认先验、伤停猜测或战意系数。原排除和严格未知球队跳过规则保留。", "",
        "方法来源：[Guo等2017温度校准](https://proceedings.mlr.press/v70/guo17a.html)；"
        "[scikit-learn校准说明](https://scikit-learn.org/stable/modules/calibration.html)。"
        "论文研究图像／文本分类，不预设对足球有效。只参考方法，自写NumPy／SciPy小优化器，"
        "没有复制上游代码或新增依赖、下载、hash、冻结接口或发布门禁。", ""]
    return "\n".join(lines)


def run(root, parent=None, *, repeats=REPEATS):
    root = Path(root).resolve()
    if parent is None:
        parents = sorted(p for p in (root / "outputs").glob("整包模型训练_*") if (p / "整包训练完整结果.json").exists())
        if not parents:
            raise ValueError("Need a completed original/native experiment")
        parent = parents[-1]
    parent = Path(parent).resolve()
    output = root / "outputs" / ("较早校准研究_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    (output / "较早旧模型预测").mkdir()
    settings = {"base_config": ORIGINAL, "base_update_days": 28,
        "calibration_training_season": "2024/25", "calibrator_before_date": TEST_START.isoformat(),
        "later_start": TEST_START.isoformat(), "later_end_exclusive": TEST_END.isoformat(),
        "methods": list(METHODS), "fit_per_league": False, "match_equal_weight": True,
        "primary_contrast": "class_bias_minus_original", "block_weeks": list(BLOCKS),
        "repeats": repeats, "seed": SEED, "calibrator_updated_in_later_season": False,
        "winner_selected_on_later_results": False, "previously_viewed_later_season": True,
        "formal_probability_api_changed": False, "score_grid_reweighted": False,
        "market_fusion": False, "parent_directory": str(parent)}
    write_json(output / "实验设置.json", settings)
    earlier, coverage = fit_earlier_original(root, output)
    write_json(output / "较早预测覆盖与数据.json", coverage)
    early_summary = raw_summary(earlier)
    write_json(output / "较早原始偏差.json", early_summary)
    (output / "较早原始概率直方图.svg").write_text(probability_histogram(earlier), encoding="utf-8")
    parameters = fit_research_calibrators(earlier)
    for method, model in parameters.items():
        write_json(output / f"校准参数_{method}.json", model)
    # Later saved forecasts and labels are loaded only after both fits complete.
    saved, later_coverage = load_saved_tracks(parent)
    later_rows = ordered_rows(list(saved["original"].values()))
    if set(r["match_id"] for r in earlier) & set(r["match_id"] for r in later_rows):
        raise ValueError("Calibration training and evaluation identities overlap")
    arms = make_arms(later_rows, parameters)
    later = analyze_arms(later_rows, arms, repeats=repeats)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "parent_directory": str(parent), "earlier_raw": early_summary, "parameters": parameters,
        "later": later, "later_source_coverage": later_coverage,
        "counts": {"new_base_model_fits": coverage["new_base_model_fits"], "new_calibrator_fits": 2,
                   "new_earlier_base_predictions": len(earlier), "reused_later_base_predictions": len(later_rows),
                   "new_later_base_model_fits": 0, "new_research_calibrated_vectors": len(later_rows)*len(METHODS)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    write_json(output / "较晚研究概率.json", research_predictions(later_rows, arms))
    write_csv(output / "较晚研究概率与损失.csv", later_rows, arms)
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": result["counts"], "later_scores": later["scores"]}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        result = json.loads((args.report_only / RESULT_FILE).read_text(encoding="utf-8"))
        (args.report_only / REPORT_FILE).write_text(report(result), encoding="utf-8")
    else:
        run(args.root, args.parent)
