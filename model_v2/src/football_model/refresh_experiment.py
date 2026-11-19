"""Exploratory 7/14/28-day refresh comparison, leaving production unchanged.

Only the refresh interval changes. Both already-viewed seasons and all arms are
reported; no winner selection, calibration, market blending or player priors.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn

from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import (BLOCKS, CLASS_NAMES, REPEATS, SEED, align_tracks,
    calibration_summary, losses, moving_block_means, write_json)
from .native_experiment import ORIGINAL, load_seasons, season_directories
from .refresh_research import INTERVALS, PERIODS, load_reference, run_frequency, validate_trial


RESULT_FILE = "更新频率研究结果.json"
REPORT_FILE = "更新频率研究说明.md"
ARMS = tuple(f"days_{days}" for days in INTERVALS)
CONTRASTS = ("days_7", "days_14")


def categorical_scores(p, y):
    """Ordered 0/1/2/3/4+ goals; RPS divided by K-1, not 1X2 RPS."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    if (p.ndim != 2 or p.shape[1] != 5 or y.shape != (len(p),) or not len(p)
            or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= 5))
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("Need finite five-category goal probabilities and integer labels")
    truth = np.eye(5)[y]
    return {"n": len(p), "log_loss": float(losses(p, y).mean()),
        "brier_sum": float(np.square(p - truth).sum(axis=1).mean()),
        "rps": float(np.square(np.cumsum(p, axis=1)[:, :-1] -
                              np.cumsum(truth, axis=1)[:, :-1]).sum(axis=1).mean()/4),
        "accuracy": float((p.argmax(axis=1) == y).mean())}


def prediction_arrays(rows):
    y = np.asarray([r["actual_outcome"] for r in rows], dtype=int)
    total_y = np.asarray([min(sum(r["actual_score"]), 4) for r in rows], dtype=int)
    margin = np.asarray([r["actual_score"][1]-r["actual_score"][0]-2 for r in rows])
    handicap_y = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))
    p = np.asarray([r["probabilities"] for r in rows], dtype=float)
    totals = np.asarray([[r["forecast"]["total_goals"][key] for key in ("0", "1", "2", "3", "4+")]
                         for r in rows], dtype=float)
    handicap = np.asarray([[r["forecast"]["away_minus_two_jincai"][key] for key in ("win", "draw", "lose")]
                           for r in rows], dtype=float)
    return {"one_x_two": (p, y), "total_goals_0_1_2_3_4plus": (totals, total_y),
            "away_minus_two_three_way": (handicap, handicap_y)}


def scores_for(rows):
    arrays = prediction_arrays(rows)
    return {name: (categorical_scores if name.startswith("total_goals") else probability_scores)(p, y)
            for name, (p, y) in arrays.items()}


def _paired_summary(tracks, ids):
    entries = {arm: [tracks[arm][key] for key in ids] for arm in ARMS}
    scores = {arm: scores_for(rows) for arm, rows in entries.items()}
    return {"n": len(ids), "scores": scores,
        "mean_differences": {f"{arm}_minus_days_28": {
            domain: scores[arm][domain]["log_loss"]-scores["days_28"][domain]["log_loss"]
            for domain in scores[arm]} for arm in CONTRASTS}}


def freshness(rows, records):
    """Calendar-date information age, not publication/odds timestamp precision."""
    source = {r.match_id: r for r in records}
    if len(source) != len(records):
        raise ValueError("Duplicate source match IDs")
    result = []
    eligible = [r for r in records if not r.goal_fit_exclusion]
    for row in rows:
        match = source[row["match_id"]]
        played, cutoff, last = (date.fromisoformat(row[k]) for k in
                               ("date", "fit_before_date", "training_max_date"))
        if played != match.match_date or not last < cutoff <= played:
            raise ValueError("Invalid freshness chronology")
        omitted = [r for r in eligible if r.competition == row["competition"]
                   and cutoff <= r.match_date < played]
        teams = {match.home, match.away}
        result.append({"match_id": row["match_id"], "snapshot_age_days": (played-cutoff).days,
            "last_included_result_age_days": (played-last).days,
            "eligible_results_since_cutoff_before_match_date": len(omitted),
            "involving_either_team_since_cutoff_before_match_date": sum(
                bool({r.home, r.away} & teams) for r in omitted)})
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Need nonempty finite descriptive values")
    return {"n": len(a), "mean": float(a.mean()), "sd": None if len(a) < 2 else float(a.std(ddof=1)),
        "min": float(a.min()), "q25": float(np.quantile(a, .25)), "median": float(np.median(a)),
        "q75": float(np.quantile(a, .75)), "max": float(a.max())}


def analyze_period(raw_tracks, records, *, repeats=REPEATS):
    if set(raw_tracks) != set(ARMS):
        raise ValueError("Need all three predeclared refresh arms")
    # The original in-memory source has tuples; JSON archives have lists. Use
    # JSON-native containers for equality, without changing or rounding numbers.
    raw_tracks = json.loads(json.dumps(raw_tracks, allow_nan=False))
    tracks, coverage = align_tracks(raw_tracks)
    ids = coverage["common_match_ids"]
    entries = {arm: list(tracks[arm].values()) for arm in ARMS}
    reference = entries["days_28"]
    arrays = {arm: prediction_arrays(rows) for arm, rows in entries.items()}
    result = _paired_summary(tracks, ids)
    result.update({"coverage": coverage, "first_date": reference[0]["date"], "last_date": reference[-1]["date"],
        "calibration": {arm: calibration_summary(*arrays[arm]["one_x_two"]) for arm in ARMS}})
    per_loss = {arm: {domain: losses(p, y) for domain, (p, y) in domains.items()}
                for arm, domains in arrays.items()}
    features = {f"{arm}_minus_days_28_{domain}": per_loss[arm][domain]-per_loss["days_28"][domain]
                for arm in CONTRASTS for domain in per_loss[arm]}
    truth = np.eye(3)[arrays["days_28"]["one_x_two"][1]]
    for arm in ARMS:
        for c, label in enumerate(CLASS_NAMES):
            features[f"{arm}_{label}_bias"] = arrays[arm]["one_x_two"][0][:, c]-truth[:, c]
    bootstrap = {}
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in reference], np.column_stack(list(features.values())),
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["features"] = {name: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, name in enumerate(features)}
        bootstrap[str(weeks)] = boot
    result["bootstrap"] = bootstrap
    result["loss_difference_descriptives"] = {name: describe(v) for name, v in features.items() if "minus" in name}
    y = arrays["days_28"]["one_x_two"][1]
    result["one_x_two_outcome_contributions"] = {arm: {label: {
        "n": int((y == c).sum()), "weighted_contribution": float(np.where(y == c,
            per_loss[arm]["one_x_two"]-per_loss["days_28"]["one_x_two"], 0).mean())}
        for c, label in enumerate(CLASS_NAMES)} for arm in CONTRASTS}
    fresh = {arm: freshness(rows, records) for arm, rows in entries.items()}
    result["freshness"] = {arm: {field: describe([r[field] for r in rows])
        for field in rows[0] if field != "match_id"} for arm, rows in fresh.items()}
    result["probability_change_from_28"] = {arm: {
        "total_variation": describe(np.abs(arrays[arm]["one_x_two"][0]-arrays["days_28"]["one_x_two"][0]).sum(axis=1)/2),
        "added_training_matches": describe([row["training_matches"]-old["training_matches"]
                                             for row, old in zip(entries[arm], reference)])}
        for arm in CONTRASTS}
    groups = (("by_league", [r["competition"] for r in reference]),
              ("by_month", [r["date"][:7] for r in reference]),
              ("by_28_day_snapshot_age", ["0-6" if r["snapshot_age_days"] < 7 else
                  "7-13" if r["snapshot_age_days"] < 14 else "14-27" for r in fresh["days_28"]]))
    for name, labels in groups:
        result[name] = {label: _paired_summary(tracks, [key for key, group in zip(ids, labels) if group == label])
                        for label in sorted(set(labels))}
    markets = {}
    for name in MARKET_COLUMNS:
        available = [i for i, row in enumerate(reference) if name in row["market_probabilities"]]
        missing = [r["match_id"] for r in reference if name not in r["market_probabilities"]]
        if not available:
            markets[name] = {"n": 0, "missing_ids": missing}
            continue
        p = np.asarray([reference[i]["market_probabilities"][name] for i in available])
        market_loss = losses(p, y[available])
        diffs = {arm: per_loss[arm]["one_x_two"][available]-market_loss for arm in ARMS}
        boot = moving_block_means([reference[i]["date"] for i in available], np.column_stack(list(diffs.values())),
                                  block_weeks=4, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        markets[name] = {"n": len(available), "match_ids": [ids[i] for i in available], "missing_ids": missing,
            "market_scores": probability_scores(p, y[available]),
            "arm_scores_on_same_matches": {arm: probability_scores(arrays[arm]["one_x_two"][0][available], y[available])
                                           for arm in ARMS},
            "differences": {arm: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, arm in enumerate(ARMS)}, "bootstrap_metadata": boot}
    result["markets"] = markets
    result["extra_coverage"] = {arm: {"match_ids": coverage["outside_common"][arm],
        "scores_separate_not_comparable_to_common": scores_for([r for r in raw_tracks[arm]
            if r["match_id"] in set(coverage["outside_common"][arm])]) if coverage["outside_common"][arm] else None}
        for arm in ARMS}
    return result, entries, fresh


def write_csv(path, entries, fresh):
    reference = entries["days_28"]
    fields = ["match_id", "date", "competition", "actual_outcome", "actual_total_goal_bucket"]
    fields += [f"{arm}_{field}" for arm in ARMS for field in
        ("fit_before_date", "training_matches", "snapshot_age_days", "last_included_result_age_days",
         "eligible_results_since_cutoff_before_match_date", "involving_either_team_since_cutoff_before_match_date",
         "p_home", "p_draw", "p_away", "one_x_two_log_loss", "total_goal_bucket_log_loss", "away_minus_two_log_loss")]
    arrays = {arm: prediction_arrays(rows) for arm, rows in entries.items()}
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(reference):
            item = {key: row[key] for key in fields[:4]}
            item["actual_total_goal_bucket"] = min(sum(row["actual_score"]), 4)
            for arm in ARMS:
                local = entries[arm][i]
                item.update({f"{arm}_{key}": local[key] for key in ("fit_before_date", "training_matches")})
                item.update({f"{arm}_{key}": value for key, value in fresh[arm][i].items() if key != "match_id"})
                item.update({f"{arm}_p_{label}": local["probabilities"][c] for c, label in enumerate(CLASS_NAMES)})
                for domain, field in (("one_x_two", "one_x_two_log_loss"),
                    ("total_goals_0_1_2_3_4plus", "total_goal_bucket_log_loss"),
                    ("away_minus_two_three_way", "away_minus_two_log_loss")):
                    p, y = arrays[arm][domain]
                    item[f"{arm}_{field}"] = float(-np.log(max(p[i, y[i]], np.finfo(float).tiny)))
            writer.writerow(item)


def difference_histogram(entries):
    arrays = {arm: prediction_arrays(rows)["one_x_two"] for arm, rows in entries.items()}
    values = {arm: losses(*arrays[arm])-losses(*arrays["days_28"]) for arm in CONTRASTS}
    limit = max(float(np.abs(v).max()) for v in values.values()) or 1e-6
    edges = np.linspace(-limit, limit, 31)
    counts = {arm: np.histogram(v, bins=edges)[0] for arm, v in values.items()}
    peak = max(int(v.max()) for v in counts.values())
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="320" viewBox="0 0 960 320">',
             '<rect width="960" height="320" fill="white"/>',
             f'<text x="20" y="25" font-size="16">Paired match 1X2 log-loss changes; common n={len(entries["days_28"])}; negative better</text>']
    for index, arm in enumerate(CONTRASTS):
        left = 35 + index*475
        lines.append(f'<text x="{left}" y="55" font-size="15">{arm} minus days_28; count maximum {peak}</text>')
        for b, count in enumerate(counts[arm]):
            height = 185*count/peak
            lines.append(f'<rect x="{left+b*14}" y="{255-height:.3f}" width="12" height="{height:.3f}" fill="#357596"/>')
        lines.append(f'<text x="{left}" y="282" font-size="13">{-limit:.4f}</text>')
        lines.append(f'<text x="{left+190}" y="282" font-size="13">0</text>')
        lines.append(f'<text x="{left+375}" y="282" font-size="13">{limit:.4f}</text>')
    return "\n".join(lines+["</svg>"])


def report(result):
    counts = result["counts"]
    lines = ["# 球队模型更新频率研究", "",
        f"新增基础拟合{counts['new_base_model_fits']}次；复用28天轨基础参数{counts['reused_base_parameter_snapshots']}份。"
        "7／14天重新生成完整比分摘要，28天直接读取既有完整存档；没有拟合校准器。所有新增文件、临时文件与模型均在G盘。", "",
        "## 只改变更新速度", "",
        "仍为旧自建Poisson：ridge=0.01、半衰期180天、rho=0，每联赛单独训练。"
        "各季均从8月1日固定锚定7／14／28个日历日块；空块正常推进，不按首场或周一重新对齐。"
        "训练只用cutoff严格之前的合格赛果，同日结果不用；两种新频率不按后期成绩选赢家。", "",
        "日期只是历史可用性假设，不是精确发布时间或T−24h复原。2024/25和2025/26均已查看，本研究是探索性回顾，"
        "不是新盲测；实际运行时间另记UTC。未知球队整场跳过、三场特殊赛排除保持不变。", "",
        "## 覆盖与同场比较", "",
        "| 赛季 | 7天完整覆盖 | 14天完整覆盖 | 28天完整覆盖 | 三轨共同 | 7天额外 |", "|---|---:|---:|---:|---:|---:|"]
    for season, data in result["periods"].items():
        cov = data["coverage"]
        lines.append(f"| {season} | {cov['counts']['days_7']} | {cov['counts']['days_14']} | {cov['counts']['days_28']} | "
                     f"{data['n']} | {len(cov['outside_common']['days_7'])} |")
    lines += ["", "额外可预测比赛单独保存／评分，不把覆盖增加混成原共同队列的准确率改善。", "",
        "| 赛季 | 更新 | 共同场次 | 1X2 LL | Brier sum | RPS | 总进球分箱LL | 客让二球三项LL |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for season, data in result["periods"].items():
        for arm, scores in data["scores"].items():
            s = scores["one_x_two"]
            lines.append(f"| {season} | {arm} | {s['n']} | {s['log_loss']:.6f} | {s['brier_sum']:.6f} | {s['rps']:.6f} | "
                f"{scores['total_goals_0_1_2_3_4plus']['log_loss']:.6f} | {scores['away_minus_two_three_way']['log_loss']:.6f} |")
    lines += ["", "总进球目标为0／1／2／3／4+五类，不是精确比分或全部整数进球分布评分。"
        "客让二球三项：客队净胜≥3为让胜、净胜2为让平，其余让负；不是亚洲盘走盘。"
        "这两项与1X2来自相同Poisson比分分布，各自尺度不同，不能横向比较LL大小。", "",
        "## 配对变化与不确定性", "",
        "预定主比较7−28天，次比较14−28天；负差较好。所有臂和次目标完整保留，不作正式晋升。", "",
        "| 赛季 | 比较 | 1X2平均LL差 | 2周块95%区间 | 4周块 | 8周块 |", "|---|---|---:|---|---|---|"]
    for season, data in result["periods"].items():
        for arm in CONTRASTS:
            key = f"{arm}_minus_days_28_one_x_two"
            intervals = [data["bootstrap"][str(w)]["features"][key]["interval_95"] for w in BLOCKS]
            fmt = lambda ci: "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
            lines.append(f"| {season} | {arm}−28天 | {data['mean_differences'][f'{arm}_minus_days_28']['one_x_two']:+.6f} | "
                         + " | ".join(fmt(ci) for ci in intervals) + " |")
    lines += ["", "不能仅按微小点估计或某项指标排名决定频率；应同时查看其他目标与跨季一致性。"
        "区间跨零不是等效证明，某个探索性区间不跨零也不是正式优越／劣效结论。", "",
        "| 赛季 | 更新 | 客胜平均偏差（百分点） | 平均快照年龄（日） | 平均最近入模结果年龄（日） |",
        "|---|---|---:|---:|---:|"]
    for season, data in result["periods"].items():
        for arm in ARMS:
            gap = 100*data["calibration"][arm]["classes"]["away"]["mean_probability_minus_frequency"]
            age = data["freshness"][arm]
            lines.append(f"| {season} | {arm} | {gap:+.3f} | {age['snapshot_age_days']['mean']:.2f} | "
                         f"{age['last_included_result_age_days']['mean']:.2f} |")
    lines += ["", "区间以已拟合预测为条件，不包含重训、模型方向选择、跨赛季变化或单场参数不确定性。"
        f"各季五联赛／全部臂共用同一连续日历周抽样，空周保留、按比赛数加权；2／4／8周非环形块各{result['settings']['repeats']}次。"
        "端点抽样偏少，球队长期关联与少量赛季仍有限制；多比较不具同时覆盖保证，不计算确认性显著性。", "",
        "## 新鲜度与时间漂移", "",
        "同时保存快照年龄（比赛日−cutoff）、最近入模结果年龄（比赛日−training_max_date），"
        "以及cutoff至比赛前一日漏掉的合格结果数／涉及两队结果数。按旧28天快照年龄0–6／7–13／14–27日同场分组，"
        "只作描述，不是按结果挑子组，也不能将年龄效果解释为因果。", "",
        "缩短更新间隔不会直接改变180天半衰期；无新历史时，推进cutoff只是将全部样本权重乘相同常数，"
        "加权平均Poisson目标理论最优解不变。变化主要来自新增已结束赛果，另有数值误差；不能说时间过去就自动降低实力。", "",
        "分联赛／月份评分、原始逐场损失差直方图、全部三类偏差／可靠性箱、总进球／让球配对区间、"
        "逐场CSV与完整模型参数均保留。不删除合法大比分或表现差的联赛。", "",
        "## 市场双轨", "", "| 赛季 | 市场 | 共同场次 | 市场LL | 7天LL | 14天LL | 28天LL |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for season, data in result["periods"].items():
        for name, market in data["markets"].items():
            if not market["n"]:
                lines.append(f"| {season} | {name} | 0 | — | — | — | — |")
                continue
            values = [market["arm_scores_on_same_matches"][arm]["log_loss"] for arm in ARMS]
            lines.append(f"| {season} | {name} | {market['n']} | {market['market_scores']['log_loss']:.6f} | "
                         + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "各市场仅在其有报价的同场集合比较；缺赔率不阻止球队训练或主要评分。"
        "pre-closing无精确时间，不是开盘或T−24h；closing可能含更多临场信息，均价不保证可成交。", "",
        "## 未改变与下一步", "",
        "没有市场融合、校准叠加、未知球队／球员先验、伤停／赛事重视程度猜测或正式模型替换。"
        "本轮完整forecast来自各自模型同一比分分布，不使用上一轮独立校准1X2。半全场仍需单独联合时间过程验证。"
        "结果不用于当前2026/27实力、12场未核实赛事或中奖保证。", "",
        "实验设计技能使比较保持单因素、同场配对和所有候选；统计分析技能使报告保留原始分布、效应差及时间依赖限制。"
        "没有新下载／依赖／hash／冻结contract／发布门禁；数据来源与软件复用说明沿用THIRD_PARTY.md。", ""]
    return "\n".join(lines)


def run(root, earlier_parent=None, later_parent=None, *, repeats=REPEATS, resume=None):
    root = Path(root).resolve()
    if type(repeats) is not int or repeats < 2:
        raise ValueError("Need at least two integer bootstrap repeats")
    saved_settings = None
    if resume is not None:
        output = Path(resume).resolve()
        if output.parent != root/"outputs" or not output.name.startswith("更新频率研究_"):
            raise ValueError("Resume only the explicitly named refresh output under this root")
        saved_settings = json.loads((output/"实验设置.json").read_text(encoding="utf-8"))
        earlier_parent = earlier_parent or saved_settings["earlier_parent"]
        later_parent = later_parent or saved_settings["later_parent"]
    if earlier_parent is None:
        available = sorted(p for p in (root/"outputs").glob("较早校准研究_*")
                           if (p/"较早校准研究结果.json").exists())
        if not available:
            raise ValueError("Need completed earlier original-model forecasts")
        earlier_parent = available[-1]
    earlier_parent = Path(earlier_parent).resolve()
    if later_parent is None:
        prior = json.loads((earlier_parent/"较早校准研究结果.json").read_text(encoding="utf-8"))
        later_parent = prior["parent_directory"]
    later_parent = Path(later_parent).resolve()
    if resume is None:
        output = root/"outputs"/("更新频率研究_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        output.mkdir(parents=True, exist_ok=False)
    settings = {"intervals_days": list(INTERVALS), "base_config": ORIGINAL, "regularization": .01,
        "periods": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in PERIODS.items()},
        "anchor": "August_01_fixed_calendar_blocks_empty_blocks_advance", "primary_contrast": "days_7_minus_days_28",
        "secondary_contrast": "days_14_minus_days_28", "model_selection": False, "calibration": False,
        "formal_probability_api_changed": False, "market_fusion": False, "previously_viewed_seasons": list(PERIODS),
        "block_weeks": list(BLOCKS), "repeats": repeats, "seed": SEED,
        "earlier_parent": str(earlier_parent), "later_parent": str(later_parent)}
    if saved_settings is not None:
        if saved_settings != settings:
            raise ValueError("Resume settings differ; do not combine different experiments")
    else:
        write_json(output/"实验设置.json", settings)
    directories = season_directories(root)
    results, counts, data_reports = {}, {"new_base_model_fits": 0, "reused_base_parameter_snapshots": 0,
        "new_prediction_rows": 0, "reused_prediction_rows": 0}, {}
    execution = {"resumed": resume is not None, "new_fits_in_this_invocation": 0,
                 "new_frequency_fits_reloaded_without_refitting": 0}
    for period, (start, end) in PERIODS.items():
        # Do not read later-season CSV results while generating the earlier arm.
        selected = {season: path for season, path in directories.items() if int(season[:4]) <= start.year}
        records, reports = load_seasons(selected)
        data_reports[period] = reports
        destination = output/period
        destination.mkdir(exist_ok=resume is not None)
        tracks = {arm: [] for arm in ARMS}
        for comp in sorted(LEAGUES):
            league = [r for r in records if r.competition == comp]
            path = (earlier_parent/"较早旧模型预测"/f"{comp}_original.json" if period == "2024-25" else
                    later_parent/"较晚赛季"/f"{comp}_{ORIGINAL['id']}.json")
            original = load_reference(path, league, start, end, comp)
            for days in INTERVALS:
                trial_path = destination/f"{comp}_days_{days}.json"
                if resume is not None and trial_path.exists():
                    trial = json.loads(trial_path.read_text(encoding="utf-8"))
                    validate_trial(trial, league, start, end, comp, interval=days)
                    if days == 28 and trial != original:
                        raise ValueError("Resumed 28-day copy differs from its parent archive")
                    if days != 28:
                        execution["new_frequency_fits_reloaded_without_refitting"] += len(trial["fits"])
                    reloaded = True
                else:
                    trial = original if days == 28 else run_frequency(league, start, end, days)
                    if days != 28:
                        execution["new_fits_in_this_invocation"] += len(trial["fits"])
                    write_json(trial_path, trial)
                    reloaded = False
                tracks[f"days_{days}"].extend(trial["predictions"])
                count_field = "reused_base_parameter_snapshots" if days == 28 else "new_base_model_fits"
                counts[count_field] += len(trial["fits"])
                counts["reused_prediction_rows" if days == 28 else "new_prediction_rows"] += len(trial["predictions"])
                print(f"{period} {comp} {days}d: n={len(trial['predictions'])}, fits={len(trial['fits'])}, reused={days == 28}, reloaded={reloaded}", flush=True)
        result, entries, fresh = analyze_period(tracks, records, repeats=repeats)
        write_csv(destination/"同场概率与损失.csv", entries, fresh)
        (destination/"原始逐场损失差.svg").write_text(difference_histogram(entries), encoding="utf-8")
        results[period] = result
        write_json(output/"阶段进度.json", {"completed_periods": list(results), "counts": counts, "execution": execution})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "counts": counts, "periods": results, "data_reports": data_reports, "execution": execution,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__}}
    write_json(output/RESULT_FILE, result)
    (output/REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts,
        "contrasts": {p: r["mean_differences"] for p, r in results.items()}}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--earlier-parent", type=Path)
    parser.add_argument("--later-parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue saved valid arms without rerunning completed fits")
    args = parser.parse_args()
    if args.report_only:
        result = json.loads((args.report_only/RESULT_FILE).read_text(encoding="utf-8"))
        (args.report_only/REPORT_FILE).write_text(report(result), encoding="utf-8")
    else:
        run(args.root, args.earlier_parent, args.later_parent, resume=args.resume)
