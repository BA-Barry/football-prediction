"""Same-match diagnostics of saved pre-match forecasts; no fitting or calibration.

Calendar moving blocks are an exploratory dependence sensitivity, conditional on
already saved forecasts. They do not create another independent test season.
"""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from .evaluation import calibration_bins, probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .native_experiment import CANDIDATES, ORIGINAL, TEST_START, TEST_END


MODEL_IDS = {
    "original": ORIGINAL["id"],
    "native_poisson_180": "poisson_h180",
    "native_dc_180": "dixon_coles_h180",
}
MODEL_NAMES = {"original": "旧自建", "native_poisson_180": "原生Poisson180",
               "native_dc_180": "原生DC180", "validation_selected": "较早验证选定"}
CLASS_NAMES = ("home", "draw", "away")
CLASS_LABELS = ("主胜", "平局", "客胜")
RESULT_FILE = "赛前概率诊断结果.json"
REPORT_FILE = "赛前概率诊断说明.md"
SOURCE_RESULT = "整包训练完整结果.json"
SEED = 20261023
REPEATS = 2000
BLOCKS = (2, 4, 8)
PRIMARY_MARKET = "avg_preclosing"
SOURCES = {
    "moving_blocks": "https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html",
    "calibration": "https://scikit-learn.org/stable/modules/calibration.html",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def probabilities(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Need three finite probabilities in [0, 1]")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Probabilities must sum to one; no silent renormalization")
    return p


def validate_prediction(row):
    """Ordinary saved-data validation, including chronology and score identity."""
    for key in ("match_id", "home", "away", "source_url"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"Missing prediction {key}")
    if row["home"] == row["away"] or row["competition"] not in LEAGUES:
        raise ValueError("Invalid fixture identity")
    played, cutoff, last = (date.fromisoformat(row[k]) for k in
                            ("date", "fit_before_date", "training_max_date"))
    if not last < cutoff <= played:
        raise ValueError("Training must be strictly earlier than fit cutoff and prediction date")
    score = row["actual_score"]
    if len(score) != 2 or any(type(g) is not int or g < 0 for g in score):
        raise ValueError("Invalid actual score")
    y = row["actual_outcome"]
    expected = 0 if score[0] > score[1] else 2 if score[0] < score[1] else 1
    if type(y) is not int or y != expected:
        raise ValueError("Outcome disagrees with score")
    probabilities(row["probabilities"])
    markets, prices = row["market_probabilities"], row["market_prices"]
    if set(markets) != set(prices) or not set(markets) <= set(MARKET_COLUMNS):
        raise ValueError("Market probability/price keys disagree")
    for key, values in markets.items():
        p = probabilities(values)
        odds = np.asarray(prices[key], dtype=float)
        if odds.shape != (3,) or not np.isfinite(odds).all() or np.any(odds <= 1):
            raise ValueError("Invalid market decimal odds")
        inverse = 1 / odds
        if not np.allclose(p, inverse / inverse.sum(), rtol=0, atol=1e-10):
            raise ValueError("Saved market probabilities disagree with proportional de-vig")


def align_tracks(tracks):
    """Check every overlap, not just final intersection; never overwrite duplicates."""
    if not tracks or any(not rows for rows in tracks.values()):
        raise ValueError("Need nonempty forecast tracks")
    mapped, identities = {}, {}
    fields = ("competition", "date", "home", "away", "actual_score", "actual_outcome",
              "source_url", "market_probabilities", "market_prices")
    for name, rows in tracks.items():
        local = {}
        for row in rows:
            validate_prediction(row)
            key = row["match_id"]
            if key in local:
                raise ValueError(f"Duplicate match ID in {name}: {key}")
            identity = {k: row[k] for k in fields}
            if key in identities and identities[key] != identity:
                raise ValueError(f"Conflicting match identity or market: {key}")
            identities[key] = identity
            local[key] = row
        mapped[name] = local
    common = set.intersection(*(set(rows) for rows in mapped.values()))
    if not common:
        raise ValueError("No common forecast matches")
    first = next(iter(mapped.values()))
    ids = sorted(common, key=lambda k: (first[k]["date"], k))
    return {name: {k: rows[k] for k in ids} for name, rows in mapped.items()}, {
        "common_match_ids": ids,
        "counts": {k: len(v) for k, v in mapped.items()},
        "outside_common": {k: sorted(set(v) - common) for k, v in mapped.items()},
    }


def load_saved_tracks(parent):
    parent = Path(parent)
    prior = json.loads((parent / SOURCE_RESULT).read_text(encoding="utf-8"))
    selection = json.loads((parent / "验证期选择.json").read_text(encoding="utf-8"))
    if selection != prior["selection"] or set(selection) != set(LEAGUES):
        raise ValueError("Saved validation selection disagrees with result")
    configurations = {c["id"]: c for c in [*CANDIDATES, ORIGINAL]}
    tracks = {k: [] for k in (*MODEL_IDS, "validation_selected")}
    cache = {}
    for comp in sorted(LEAGUES):
        selected = selection[comp]["config"]
        if selected != configurations.get(selected["id"]):
            raise ValueError("Saved selection configuration mismatch")
        choices = {**MODEL_IDS, "validation_selected": selection[comp]["config"]["id"]}
        for label, config in choices.items():
            if config not in configurations:
                raise ValueError("Unknown saved model configuration")
            key = (comp, config)
            if key not in cache:
                path = parent / "较晚赛季" / f"{comp}_{config}.json"
                trial = json.loads(path.read_text(encoding="utf-8"))
                if (trial["config"] != configurations[config] or trial["start"] != TEST_START.isoformat()
                        or trial["end_exclusive"] != TEST_END.isoformat() or trial["update_interval_days"] != 28):
                    raise ValueError("Saved run settings mismatch")
                for row in trial["predictions"]:
                    if row["competition"] != comp or not TEST_START <= date.fromisoformat(row["date"]) < TEST_END:
                        raise ValueError("Saved forecast outside declared league/period")
                cache[key] = trial["predictions"]
            tracks[label].extend(cache[key])
    matched, coverage = align_tracks(tracks)
    if set(coverage["common_match_ids"]) != set(prior["common_test_match_ids"]):
        raise ValueError("Loaded common cohort differs from original experiment")
    return matched, coverage


def moving_block_means(dates, values, *, block_weeks=4, repeats=REPEATS, seed=SEED):
    """Joint non-circular moving calendar-week blocks, ratio of sums to counts.

    Preserve empty weeks and sample all leagues/columns with the same weights.
    Draw ceil(W/L) starts in [0, W-L], concatenate, truncate to W weeks.
    Non-circular blocks underweight endpoints; return metadata, not p-values.
    """
    if type(block_weeks) is not int or block_weeks < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("Positive integer block length and at least two repeats required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    days = [date.fromisoformat(d) if isinstance(d, str) else d for d in dates]
    v = np.asarray(values, dtype=float)
    if (not days or any(type(d) is not date for d in days) or v.ndim != 2
            or v.shape[0] != len(days) or v.shape[1] < 1 or not np.isfinite(v).all()):
        raise ValueError("Need finite n x k values and n calendar dates")
    first = min(days)
    monday = first - timedelta(days=first.weekday())
    week_ids = np.asarray([(d - monday).days // 7 for d in days])
    weeks = int(week_ids.max()) + 1
    counts = np.bincount(week_ids, minlength=weeks)
    sums = np.zeros((weeks, v.shape[1]))
    np.add.at(sums, week_ids, v)
    result = {"block_weeks": block_weeks, "repeats": repeats, "seed": seed,
        "n_matches": len(days), "n_calendar_weeks": weeks,
        "n_nonempty_weeks": int(np.count_nonzero(counts)),
        "calendar_week_starts": [(monday + timedelta(weeks=w)).isoformat() for w in range(weeks)],
        "weekly_match_counts": counts.tolist(), "mean": v.mean(axis=0).tolist(),
        "method": "noncircular_moving_calendar_week_blocks_ratio_of_sums_percentile_95",
        "interval_95": None, "valid_repeats": 0, "zero_count_repeats": 0}
    if block_weeks >= weeks:
        result["unavailable_reason"] = "block_length_not_smaller_than_calendar_span"
        return result
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, weeks - block_weeks + 1, size=(repeats, (weeks + block_weeks - 1) // block_weeks))
    sampled = (starts[:, :, None] + np.arange(block_weeks)).reshape(repeats, -1)[:, :weeks]
    weights = np.zeros((repeats, weeks), dtype=np.int64)
    np.add.at(weights, (np.arange(repeats)[:, None], sampled), 1)
    denominators = weights @ counts
    valid = denominators > 0
    result["valid_repeats"] = int(valid.sum())
    result["zero_count_repeats"] = int((~valid).sum())
    if valid.sum() < 2:
        result["unavailable_reason"] = "fewer_than_two_nonempty_resamples"
        return result
    means = (weights[valid] @ sums) / denominators[valid, None]
    result["interval_95"] = np.quantile(means, [0.025, 0.975], axis=0).T.tolist()
    return result


def losses(p, y):
    return -np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny))


def calibration_summary(p, y):
    truth = np.eye(3)[y]
    bins = calibration_bins(p, y, bins=5)
    return {"classes": {name: {
        "mean_probability": float(p[:, c].mean()), "observed_frequency": float(truth[:, c].mean()),
        "mean_probability_minus_frequency": float((p[:, c] - truth[:, c]).mean()),
        "binary_brier": float(np.square(p[:, c] - truth[:, c]).mean()),
    } for c, name in enumerate(CLASS_NAMES)}, "equal_width_5_bins": bins,
        "note": "Marginal bias near zero is not sufficient calibration; bins are descriptive, not used for selection."}


def analyze_cohort(tracks, ids, market, *, repeats=REPEATS, seed=SEED):
    """All comparisons within this call use exactly the same ordered matches."""
    reference = tracks["original"]
    rows = [reference[k] for k in ids]
    y = np.asarray([r["actual_outcome"] for r in rows], dtype=int)
    dates = [r["date"] for r in rows]
    p = {name: np.asarray([track[k]["probabilities"] for k in ids]) for name, track in tracks.items()}
    p[market] = np.asarray([r["market_probabilities"][market] for r in rows])
    per_loss = {name: losses(value, y) for name, value in p.items()}
    deltas = {f"{name}_minus_{market}": per_loss[name] - per_loss[market] for name in tracks}
    deltas["validation_selected_minus_original"] = per_loss["validation_selected"] - per_loss["original"]
    features = dict(deltas)
    truth = np.eye(3)[y]
    for name, value in p.items():
        for c, outcome in enumerate(CLASS_NAMES):
            features[f"{name}_{outcome}_bias"] = value[:, c] - truth[:, c]
    names = list(features)
    matrix = np.column_stack(list(features.values()))
    bootstrap = {}
    for length in BLOCKS:
        boot = moving_block_means(dates, matrix, block_weeks=length, repeats=repeats, seed=seed)
        intervals = boot.pop("interval_95")
        means = boot.pop("mean")
        boot["features"] = {name: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, name in enumerate(names)}
        bootstrap[str(length)] = boot
    contrasts = {}
    for name, delta in deltas.items():
        contrasts[name] = {"mean_log_loss_difference": float(delta.mean()),
            "by_actual_outcome": {outcome: {
                "n": int((y == c).sum()),
                "within_outcome_mean_difference": float(delta[y == c].mean()) if (y == c).any() else None,
                "weighted_contribution_to_overall": float(np.where(y == c, delta, 0).mean()),
            } for c, outcome in enumerate(CLASS_NAMES)}}
    grouped = {}
    for group_name, labels in (
            ("by_league", [r["competition"] for r in rows]),
            ("by_calendar_month", [r["date"][:7] for r in rows])):
        a = np.asarray(labels)
        grouped[group_name] = {label: {"n": int((a == label).sum()),
            "scores": {name: probability_scores(value[a == label], y[a == label]) for name, value in p.items()},
            "contrasts": {name: float(delta[a == label].mean()) for name, delta in deltas.items()}}
            for label in sorted(set(labels))}
    return {"n": len(ids), "match_ids": ids, "first_date": min(dates), "last_date": max(dates),
        "scores": {name: probability_scores(value, y) for name, value in p.items()},
        "calibration": {name: calibration_summary(value, y) for name, value in p.items()},
        "contrasts": contrasts, "bootstrap": bootstrap, **grouped}


def diagnose(tracks, *, repeats=REPEATS, seed=SEED):
    matched, coverage = align_tracks(tracks)
    if set(matched) != {*MODEL_IDS, "validation_selected"}:
        raise ValueError("Require all four declared saved model tracks")
    ids = coverage["common_match_ids"]
    ref = matched["original"]
    markets, missing = {}, {}
    for market in MARKET_COLUMNS:
        usable = [k for k in ids if market in ref[k]["market_probabilities"]]
        usable_set = set(usable)
        missing[market] = [k for k in ids if k not in usable_set]
        markets[market] = analyze_cohort(matched, usable, market, repeats=repeats, seed=seed) if usable else None
    if markets[PRIMARY_MARKET] is None:
        raise ValueError("No primary market observations; do not substitute closing prices")
    return {"coverage": {**coverage, "missing_market_ids": missing}, "markets": markets}


def report(result):
    primary = result["markets"][PRIMARY_MARKET]
    boot = primary["bootstrap"]["4"]
    fmt = lambda x: "不可估计" if x is None else f"[{x[0]:+.6f}, {x[1]:+.6f}]"
    lines = ["# 已存赛前概率：同场校准与时间关联敏感性", "",
        f"只读来源：`{result['parent_directory']}`。0次拟合、0条新模型预测、0次概率修改；本轮重新计算评分和诊断。", "",
        f"主队列 {primary['n']} 场，{primary['first_date']} 至 {primary['last_date']}，"
        f"{boot['n_calendar_weeks']} 个连续日历周，{boot['n_nonempty_weeks']} 个非空周；保留空周。", "",
        "原实验按28天更新，严格使用更早日期赛果。不是球员事后路径强度实验，也不是2026/27即时预测。"
        "该赛季成绩此前已查看，本轮为探索性诊断，不是新增盲测。", "",
        "## 同场整体评分", "",
        "Log loss、Brier三类平方误差之和、RPS均越低越好；命中率单列，不代替概率质量。", "",
        "| 方案 | 场次 | Log loss | Brier sum | RPS | 最高概率结果命中率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, score in primary["scores"].items():
        lines.append(f"| {MODEL_NAMES.get(name, name)} | {score['n']} | {score['log_loss']:.6f} | "
                     f"{score['brier_sum']:.6f} | {score['rps']:.6f} | {score['accuracy']:.2%} |")
    main_key = f"original_minus_{PRIMARY_MARKET}"
    gap = primary["contrasts"][main_key]["mean_log_loss_difference"]
    away_bias = primary["calibration"]["original"]["classes"]["away"]["mean_probability_minus_frequency"]
    lines += ["", f"本批数据旧自建相对市场的平均损失差为{gap:+.6f}，4周区间"
        f"{fmt(boot['features'][main_key]['interval_95'])}；正数表示旧自建误差更大。"
        f"其平均客胜概率相对实际频率偏差{100*away_bias:+.3f}个百分点。"
        "这是误差定位，不是把未来每场客胜概率统一扣除该值的依据；也不能仅凭此归因主场参数或球员影响。"]
    lines += ["", "## 同场损失差及块长敏感性", "",
        "主诊断指定旧自建−Avg pre-closing，其他为描述性对照；差值负数更好。"
        "区间是以已存预测为条件的探索性95%百分位区间，不是单场胜率区间。", "",
        "| 对照 | 平均差 | 2周块区间 | 4周块区间 | 8周块区间 |", "|---|---:|---|---|---|"]
    for name, contrast in primary["contrasts"].items():
        cells = [fmt(primary["bootstrap"][str(w)]["features"][name]["interval_95"]) for w in BLOCKS]
        lines.append(f"| {name} | {contrast['mean_log_loss_difference']:+.6f} | " + " | ".join(cells) + " |")
    lines += ["", "## 三类概率偏差", "",
        "偏差=平均预测概率−实际频率，正数偏高。区间使用同一4周联合抽样。"
        "总体偏差接近零仍可能在不同概率区间或联赛内失准；完整5箱表和箱内人数在JSON中。", "",
        "| 方案 | 结果 | 平均概率 | 实际频率 | 偏差（百分点） | 4周区间（概率单位） |",
        "|---|---|---:|---:|---:|---|"]
    for name, calibration in primary["calibration"].items():
        for outcome, label in zip(CLASS_NAMES, CLASS_LABELS):
            c = calibration["classes"][outcome]
            interval = boot["features"][f"{name}_{outcome}_bias"]["interval_95"]
            lines.append(f"| {MODEL_NAMES.get(name, name)} | {label} | {c['mean_probability']:.2%} | "
                f"{c['observed_frequency']:.2%} | {100*c['mean_probability_minus_frequency']:+.3f} | {fmt(interval)} |")
    highest_away = [b for b in primary["calibration"]["original"]["equal_width_5_bins"]
                    if b["outcome"] == "away" and b["bin"] == 4]
    if highest_away:
        lines += ["", f"旧自建客胜概率[0.8,1]箱仅{highest_away[0]['n']}场，"
                  "高概率端样本稀少，不用单个箱的频率直接拟合修正系数或推断高赔率价值。"]
    lines += ["", "## 误差来自哪些赛果／联赛", "",
        "以下只作误差定位，不按这些结果挑模型、挑月份或挑投注。赛果分项是总体差的加权贡献，三项相加等于总体差。", "",
        "| 方案−Avg pre-closing | 主胜贡献 | 平局贡献 | 客胜贡献 |", "|---|---:|---:|---:|"]
    for name in MODEL_NAMES:
        c = primary["contrasts"][f"{name}_minus_{PRIMARY_MARKET}"]["by_actual_outcome"]
        lines.append(f"| {MODEL_NAMES[name]} | " + " | ".join(f"{c[k]['weighted_contribution_to_overall']:+.6f}" for k in CLASS_NAMES) + " |")
    lines += ["", "| 联赛 | 场次 | 旧自建−市场 | 较早选定−市场 |", "|---|---:|---:|---:|"]
    for comp, row in primary["by_league"].items():
        contrasts = row["contrasts"]
        lines.append(f"| {LEAGUES[comp]} | {row['n']} | {contrasts[f'original_minus_{PRIMARY_MARKET}']:+.6f} | "
                     f"{contrasts[f'validation_selected_minus_{PRIMARY_MARKET}']:+.6f} |")
    lines += ["", "## 市场来源敏感性", "",
        "每行均重新对齐该市场与四模型的共同队列；缺失名单明确保存，不补造报价或概率。", "",
        "| 市场字段 | 场次 | 市场LL | 旧自建−市场 | 4周区间 |", "|---|---:|---:|---:|---|"]
    for market, row in result["markets"].items():
        if row is None:
            lines.append(f"| {market} | 0 | — | — | — |")
            continue
        c = f"original_minus_{market}"
        interval = row["bootstrap"]["4"]["features"][c]["interval_95"]
        lines.append(f"| {market} | {row['n']} | {row['scores'][market]['log_loss']:.6f} | "
                     f"{row['contrasts'][c]['mean_log_loss_difference']:+.6f} | {fmt(interval)} |")
    lines += ["", "## 方法与边界", "",
        f"每种块长均{result['settings']['repeats']}次、种子{result['settings']['seed']}。周从星期一开始；"
        "从所有合法连续L周起点有放回抽取ceil(W/L)块，拼接后截到W周，不把赛季末与赛季初环接。"
        "每个块包含所有联赛和所有方案，使用同一抽样权重；每次以逐场差之和÷比赛数计算，不能等权平均周分数。"
        "空分母样本不用于分位数，其次数和有效次数均保存。", "",
        "非环形块对端点覆盖较少；块长是事先固定的敏感性设置，不按区间好看与否挑选。"
        "周跨度有限，尤其8周只有少量有效时间块。此法依赖局部弱平稳／短程相关近似，不能保证消除球队长期关联。"
        "未重训或重选参数，区间不含拟合／选择不确定性、跨赛季漂移或新球员不确定性。"
        "各区间为边际探索性区间，不提供多重比较的同时覆盖或确认性显著结论。", "",
        "AvgH/D/A、B365H/D/A是pre-closing字段，不等于已核实开盘或T−24h报价。"
        "closing字段可能包含更晚信息，两者都没有独立采集时间；市场聚合均价不一定可成交。"
        "去水为赔率倒数归一化，不把市场当真实概率。模型与市场的信息时点不同，"
        "误差差不能仅归因于结构，更不是可交易收益或串关价值证明。", "",
        "校准箱只是观察预测概率与发生频率的对应。没有把原始连续概率Brier写成分箱分解，"
        "也没有用本轮后段拟合校准器、融合市场、增加未知球员先验或战意系数。"
        "主客场、伤停、赛事投入等新因子仍需真实赛前时间数据与另行验证。", "",
        "方法参考："
        f"[arch非环形移动块定义]({SOURCES['moving_blocks']})；"
        f"[scikit-learn校准与评分边界]({SOURCES['calibration']})。"
        "只参考方法；本轮NumPy实现，无复制上游代码、无安装arch、无新增依赖或数据下载。", "",
        "所有新增文件与运行临时目录在项目目录。逐场CSV、共享IDs、全部箱表、按月／联赛评分和区间元数据随结果保留；"
        "原始数据、已有模型和预测文件不改写。", ""]
    return "\n".join(lines)


def write_paired_csv(path, matched):
    fields = ["match_id", "date", "competition", "home", "away", "actual_outcome"]
    fields += [f"{name}_log_loss" for name in matched]
    fields += [f"{market}_log_loss" for market in MARKET_COLUMNS]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, r in matched["original"].items():
            y = r["actual_outcome"]
            row = {field: r[field] for field in fields[:6]}
            for name, track in matched.items():
                row[f"{name}_log_loss"] = float(losses(np.asarray([track[key]["probabilities"]]), np.asarray([y]))[0])
            for market, p in r["market_probabilities"].items():
                row[f"{market}_log_loss"] = float(losses(np.asarray([p]), np.asarray([y]))[0])
            writer.writerow(row)


def run(root, parent=None, *, repeats=REPEATS):
    root = Path(root).resolve()
    if parent is None:
        parents = sorted(p for p in (root / "outputs").glob("整包模型训练_*") if (p / SOURCE_RESULT).exists())
        if not parents:
            raise ValueError("No saved native experiment; run it explicitly first")
        parent = parents[-1]
    parent = Path(parent).resolve()
    output = root / "outputs" / ("赛前概率诊断_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    settings = {"repeats": repeats, "seed": SEED, "calendar_block_weeks": list(BLOCKS),
        "primary_block_weeks": 4, "primary_contrast": f"original_minus_{PRIMARY_MARKET}",
        "calibration_bins": 5, "parent_directory": str(parent),
        "exploratory_previously_viewed_season": True, "refit_models": False,
        "probability_recalibration": False, "market_blending": False}
    # Record choices before computing the new diagnostic quantities, not a frozen contract.
    write_json(output / "分析设置.json", settings)
    matched, original_coverage = load_saved_tracks(parent)
    result = diagnose({name: list(rows.values()) for name, rows in matched.items()}, repeats=repeats)
    result.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "parent_directory": str(parent), "source_loading_coverage": original_coverage,
        "new_model_fits": 0, "new_model_predictions": 0, "probabilities_changed": False,
        "runtime": {"numpy": np.__version__}, "method_sources": SOURCES})
    write_json(output / RESULT_FILE, result)
    write_paired_csv(output / "同场逐场损失.csv", matched)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "primary_n": result["markets"][PRIMARY_MARKET]["n"],
        "new_model_fits": 0, "new_model_predictions": 0}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        data = json.loads((args.report_only / RESULT_FILE).read_text(encoding="utf-8"))
        (args.report_only / REPORT_FILE).write_text(report(data), encoding="utf-8")
    else:
        run(args.root, args.parent)
