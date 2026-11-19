"""Two-arm chronological source-date-feature research; no production adoption.

The 14-day window, original GLM settings and both already-viewed seasons are
fixed research choices, not a search over the later season. No new data, market
fusion, player prior, injury estimate or neural-network model is introduced.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn

from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import (BLOCKS, REPEATS, SEED, align_tracks,
    calibration_summary, losses, moving_block_means, write_json)
from .goal_rate_diagnostics import STAT_FIELDS, rates_from_parameters, row_statistics, summary as goal_summary
from .goal_rate_experiment import MOMENT_FIELDS, _read_json
from .native_experiment import load_seasons, season_directories
from .probability import score_grid, summarize
from .refresh_experiment import describe, prediction_arrays, scores_for
from .refresh_research import PERIODS, _same, validate_trial as validate_original
from .schedule_model import DateActivityIndex, SCHEDULE_FEATURE_DEFINITION
from .schedule_research import CONFIG, load_trial, run_trial


ARMS = ("original", "schedule")
DOMAINS = ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way", "exact_score")
RESULT_FILE = "赛程日期特征研究结果.json"
REPORT_FILE = "赛程日期特征研究说明.md"


def original_rows(trial, records):
    """Decorate copies for research, without changing the saved original arm."""
    fits = {fit["before_date"]: fit for fit in trial["fits"]}
    source = {r.match_id: r for r in records}
    index = DateActivityIndex(records)
    rows = []
    for row in trial["predictions"]:
        record = source[row["match_id"]]
        rows.append({**row, "rates": list(rates_from_parameters(fits[row["fit_before_date"]], row)),
            "schedule_context": index.context(record.home, record.away, record.competition, record.match_date),
            "context_before_date": row["date"]})
    return rows


def cohort(entries, stats, indices):
    local = {arm: [entries[arm][i] for i in indices] for arm in ARMS}
    ss = {arm: [stats[arm][i] for i in indices] for arm in ARMS}
    scores = {arm: {**scores_for(local[arm]), "exact_score": {"n": len(indices),
        "mean_nll": float(np.mean([s["joint_score_nll"] for s in ss[arm]]))}} for arm in ARMS}
    return {"n": len(indices), "scores": scores,
        "goal_summaries": {arm: goal_summary(local[arm], ss[arm]) for arm in ARMS},
        "mean_differences": {domain: scores["schedule"][domain]["mean_nll" if domain == "exact_score" else "log_loss"]
            - scores["original"][domain]["mean_nll" if domain == "exact_score" else "log_loss"] for domain in DOMAINS}}


def feature_vectors(entries, stats):
    """Shared resampling vector: four paired losses, all moments and H/D/A bias."""
    per_loss = {arm: {domain: losses(p, y) for domain, (p, y) in prediction_arrays(entries[arm]).items()}
                for arm in ARMS}
    for arm in ARMS:
        per_loss[arm]["exact_score"] = np.asarray([s["joint_score_nll"] for s in stats[arm]])
    features = {f"schedule_minus_original_{domain}": per_loss["schedule"][domain]-per_loss["original"][domain]
                for domain in DOMAINS}
    for arm in ARMS:
        for field in MOMENT_FIELDS:
            features[f"{arm}_{field}"] = np.asarray([s[field] for s in stats[arm]])
        p, y = prediction_arrays(entries[arm])["one_x_two"]
        for c, name in enumerate(("home", "draw", "away")):
            features[f"{arm}_{name}_probability_minus_frequency"] = p[:, c]-(y == c)
    return per_loss, features


def analyze_period(raw_tracks, *, repeats=REPEATS):
    if set(raw_tracks) != set(ARMS):
        raise ValueError("Need exactly the two declared source-date research arms")
    tracks, coverage = align_tracks(json.loads(json.dumps(raw_tracks, allow_nan=False)))
    if any(coverage["outside_common"][arm] for arm in ARMS):
        raise ValueError("Both arms must retain the complete original prediction cohort")
    entries = {arm: list(tracks[arm].values()) for arm in ARMS}
    rows = entries["original"]
    stats = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for row in entries[arm]:
            ss = row_statistics(row, row["rates"])
            _same(row["forecast"], summarize(score_grid(*row["rates"], rho=0)), "same_rate_forecast")
            _same(row["probabilities"], [row["forecast"]["one_x_two"][k] for k in ("home", "draw", "away")],
                  "same_rate_one_x_two")
            if row["context_before_date"] != row["date"]:
                raise ValueError("Date context must be evaluated at each target date")
            stats[arm].append(ss)
    for original, schedule in zip(rows, entries["schedule"]):
        _same(original["schedule_context"], schedule["schedule_context"], "same_date_context")
        for field in ("fit_before_date", "training_max_date", "training_matches"):
            if original[field] != schedule[field]:
                raise ValueError("Arms must use identical labels, cutoff and training coverage")
    result = cohort(entries, stats, list(range(len(rows))))
    per_loss, features = feature_vectors(entries, stats)
    result.update({"coverage": coverage, "match_ids": coverage["common_match_ids"],
        "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "calibration": {arm: calibration_summary(*prediction_arrays(entries[arm])["one_x_two"]) for arm in ARMS},
        "loss_difference_descriptives": {domain: describe(features[f"schedule_minus_original_{domain}"])
                                         for domain in DOMAINS}, "bootstrap": {}})
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in rows], np.column_stack(list(features.values())),
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["features"] = {key: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, key in enumerate(features)}
        result["bootstrap"][str(weeks)] = boot
    contexts = [r["schedule_context"] for r in rows]
    result["date_feature_descriptives"] = {side: {
        "source_fixture_count_prev14d": describe([c[side]["source_fixture_count_prev14d"] for c in contexts]),
        "left_censored_windows": sum(c[side]["window_left_censored"] for c in contexts),
        "without_previous_source_record": sum(not c[side]["has_previous_source_record"] for c in contexts),
        "windows_containing_special_source_records": sum(bool(c[side]["special_previous_14d_ids"]) for c in contexts),
        "count_frequencies": {str(k): sum(c[side]["source_fixture_count_prev14d"] == k for c in contexts)
                              for k in sorted({c[side]["source_fixture_count_prev14d"] for c in contexts})}}
        for side in ("home", "away")}
    grouping = (("by_league", [r["competition"] for r in rows]),
                ("by_month", [r["date"][:7] for r in rows]),
                ("by_observed_count_comparison", ["equal" if c["home"]["source_fixture_count_prev14d"] == c["away"]["source_fixture_count_prev14d"]
                    else "home_more" if c["home"]["source_fixture_count_prev14d"] > c["away"]["source_fixture_count_prev14d"]
                    else "away_more" for c in contexts]))
    for name, labels in grouping:
        result[name] = {label: cohort(entries, stats, [i for i, value in enumerate(labels) if value == label])
                        for label in sorted(set(labels))}
    result["markets"] = {}
    y = prediction_arrays(rows)["one_x_two"][1]
    for market in MARKET_COLUMNS:
        ids = [i for i, row in enumerate(rows) if market in row["market_probabilities"]]
        missing = [row["match_id"] for row in rows if market not in row["market_probabilities"]]
        if not ids:
            result["markets"][market] = {"n": 0, "missing_ids": missing}
            continue
        p = np.asarray([rows[i]["market_probabilities"][market] for i in ids])
        market_loss = losses(p, y[ids])
        diffs = {arm: per_loss[arm]["one_x_two"][ids]-market_loss for arm in ARMS}
        boot = moving_block_means([rows[i]["date"] for i in ids], np.column_stack(list(diffs.values())),
                                  block_weeks=4, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        result["markets"][market] = {"n": len(ids), "match_ids": [rows[i]["match_id"] for i in ids], "missing_ids": missing,
            "market_scores": probability_scores(p, y[ids]),
            "arm_scores_on_same_matches": {arm: probability_scores(prediction_arrays(entries[arm])["one_x_two"][0][ids], y[ids])
                                           for arm in ARMS},
            "differences": {arm: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, arm in enumerate(ARMS)}, "bootstrap_metadata": boot}
    return result, entries, stats


def write_csv(path, entries, stats):
    fields = ["match_id", "date", "competition", "actual_home_goals", "actual_away_goals", "fit_before_date", "training_matches"]
    fields += [f"{side}_{field}" for side in ("home", "away") for field in
        ("source_fixture_count_prev14d", "previous_date", "gap_days", "window_left_censored", "has_previous_source_record")]
    fields += [f"{arm}_p_{side}" for arm in ARMS for side in ("home", "draw", "away")]
    fields += [f"{arm}_{field}" for arm in ARMS for field in STAT_FIELDS]
    fields += [f"{arm}_{domain}_loss" for arm in ARMS for domain in DOMAINS]
    per_loss, _ = feature_vectors(entries, stats)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(entries["original"]):
            values = {key: row[key] for key in ("match_id", "date", "competition", "fit_before_date", "training_matches")}
            values.update({"actual_home_goals": row["actual_score"][0], "actual_away_goals": row["actual_score"][1]})
            values.update({f"{side}_{field}": row["schedule_context"][side][field] for side in ("home", "away") for field in
                ("source_fixture_count_prev14d", "previous_date", "gap_days", "window_left_censored", "has_previous_source_record")})
            for arm in ARMS:
                values.update({f"{arm}_p_{side}": entries[arm][i]["probabilities"][c] for c, side in enumerate(("home", "draw", "away"))})
                values.update({f"{arm}_{field}": stats[arm][i][field] for field in STAT_FIELDS})
                values.update({f"{arm}_{domain}_loss": per_loss[arm][domain][i] for domain in DOMAINS})
            writer.writerow(values)


def report(result):
    lines = ["# 来源联赛赛程日期特征研究", "", "## 范围与实际执行", "",
        "这是未正式采用的两臂研究：原28天Poisson对照，同类模型只增加自己／对手两个前14日来源赛程记录数系数，原攻防等系数也联合重估。窗口固定[d−14,d)，不按较晚成绩改窗口、正则或模型。",
        "记录数不是完整赛程负荷、实际休息、完成90分钟或战意：缺杯赛、欧战、国家队、低级别联赛；特殊比赛保留原日期进入索引，但进球标签仍排除。0只表示本地同源同联赛窗口未见记录。左截断与首次出现另记，不填造真实0场。",
        f"新增基础拟合{result['counts']['new_schedule_fits']}次；原参数复用{result['counts']['original_snapshots_reused']}份；新研究预测{result['counts']['new_schedule_predictions']}条。独立比赛数不因两臂或主客展开而翻倍。",
        "训练每行在自己的比赛日计算日期特征，不看该行比分／盘口；拟合标签严格早于截止日。同日全部排除。系数按8月1固定锚点每28天更新，日期输入到各预测日前更新，因此不是提前28天发布的预测。",
        "两季此前均已查看，这是探索性时间外研究，不是新增盲测。不替换正式API，不融合市场，不加入新球员先验／神经网络／强化学习。", "", "## 同场结果", "",
        "Log loss／NLL越低越好；各目标不同，不能把1X2和比分NLL的绝对值横比。", "",
        "| 季度 | 场数 | 原1X2 LL | 日期特征1X2 LL | 差值 | 4周探索性95%区间 |",
        "|---|---:|---:|---:|---:|---|"]
    for period, r in result["periods"].items():
        effect = r["bootstrap"]["4"]["features"]["schedule_minus_original_one_x_two"]
        ci = effect["interval_95"]
        lines.append(f"| {period} | {r['n']} | {r['scores']['original']['one_x_two']['log_loss']:.6f} | {r['scores']['schedule']['one_x_two']['log_loss']:.6f} | {r['mean_differences']['one_x_two']:+.6g} | "
                     + (f"[{ci[0]:+.6g}, {ci[1]:+.6g}]" if ci else "不可估计") + " |")
    for period, r in result["periods"].items():
        lines += ["", f"### {period}：其他目标与数据覆盖", "",
            "| 目标 | 原样 | 日期特征 | 调整−原样 |", "|---|---:|---:|---:|"]
        for domain in DOMAINS[1:]:
            field = "mean_nll" if domain == "exact_score" else "log_loss"
            lines.append(f"| {domain} | {r['scores']['original'][domain][field]:.6f} | {r['scores']['schedule'][domain][field]:.6f} | {r['mean_differences'][domain]:+.6g} |")
        lines += ["", f"期间{r['first_date']}至{r['last_date']}。所有2／4／8周同场联合日历块区间、联赛／月份／记录数比较描述及分箱校准均在JSON中保留，不按有利子组选方法。",
            f"主／客目标窗口左截断数：{r['date_feature_descriptives']['home']['left_censored_windows']}／{r['date_feature_descriptives']['away']['left_censored_windows']}。这不证明杯赛等不存在，原始档案覆盖和真实赛前可知性不同。"]
        market = r["markets"]["avg_preclosing"]
        if market["n"]:
            lines.append(f"同场Avg pre-closing去水LL {market['market_scores']['log_loss']:.6f}；采集时点未知，不视为开盘／T−24h／可成交报价。")
    lines += ["", "## 不确定性与后续", "",
        "每场比赛为配对单位；2／4／8周非环形联合重采样跨所有联赛／方案／指标，保留空周。区间以已拟合预测为条件，不包含重训、结构选择、跨季漂移或单场概率误差；多项探索区间没有同时覆盖保证。不据含零区间宣称等效，也不把单个不跨零区间当正式成功。",
        "真实有效大比分保留；精确比分NLL在log域计算，胜平负／总进球／让球由同一比分分布派生。客让二球三项的让胜要求净胜至少3球；净胜2是让平，不是亚洲盘走盘。",
        "实验设计技能限定两臂与固定时序；统计分析技能要求同场损失差、分布描述及条件区间。日期特征系数只是预测关联，不能解释为疲劳／伤停／赛事投入的因果效果。",
        "当前数据仍止于2026年5月；2026/27数据、实时伤停／首发、跨联赛实力桥梁、球员赛前衔接及半全场过程未完成。本研究不能直接支持当前比赛的可靠预测或投注收益结论。",
        "所有新增源代码、参数、CSV、测试临时文件与缓存均在G盘；本轮不下载数据／依赖，不新增hash、冻结contract或发布门禁。",
        "", "方法参考：[scikit-learn PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)。", ""]
    return "\n".join(lines)


def run(root, parent=None, *, output=None, repeats=REPEATS):
    root = Path(root)
    if parent is None:
        parent = sorted((root/"outputs").glob("更新频率研究_*"))[-1]
    parent = Path(parent).resolve()
    output = Path(output) if output else root/"outputs"/f"赛程日期特征研究_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    settings = {"parent_directory": str(parent), "arms": list(ARMS), "schedule_config": CONFIG,
        "feature_definition": SCHEDULE_FEATURE_DEFINITION, "update_interval_days": 28,
        "periods_previously_viewed": list(PERIODS), "model_selection": False,
        "production_api_changed": False, "market_fusion": False, "block_weeks": list(BLOCKS),
        "primary_block_weeks": 4, "repeats": repeats, "seed": SEED}
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output/"实验设置.json"
    if settings_path.exists():
        if _read_json(settings_path) != settings:
            raise ValueError("Existing experiment settings differ; use a new output directory")
    else:
        write_json(settings_path, settings)
    periods, sources, fit_counts, reused = {}, {}, 0, 0
    new_fits_this_call, trials_reloaded = 0, 0
    directories = season_directories(root)
    for period, (start, end) in PERIODS.items():
        # Complete earlier analysis before loading the later season's results.
        selected_directories = {s: p for s, p in directories.items() if int(s[:4]) <= start.year}
        records, source_reports = load_seasons(selected_directories)
        sources[period] = source_reports
        folder = output/period
        folder.mkdir(exist_ok=True)
        tracks = {arm: [] for arm in ARMS}
        for comp in sorted(LEAGUES):
            local = [r for r in records if r.competition == comp]
            original = _read_json(parent/period/f"{comp}_days_28.json")
            validate_original(original, local, start, end, comp, interval=28)
            old_path = folder/f"{comp}_original.json"
            if old_path.exists():
                _same(_read_json(old_path), original, "saved_original_trial")
            else:
                write_json(old_path, original)
            new_path = folder/f"{comp}_schedule.json"
            if new_path.exists():
                trial = load_trial(new_path, local, start, end, comp)
                trials_reloaded += 1
            else:
                trial = run_trial(local, start, end, comp)
                write_json(new_path, trial)
                new_fits_this_call += len(trial["fits"])
            fit_counts += len(trial["fits"])
            reused += len(original["fits"])
            tracks["original"].extend(original_rows(original, local))
            tracks["schedule"].extend(trial["predictions"])
            print(f"Completed {period} {comp}: {len(trial['fits'])} schedule fits, {len(trial['predictions'])} forecasts", flush=True)
        analysis, entries, stats = analyze_period(tracks, repeats=repeats)
        periods[period] = analysis
        write_json(folder/"同场汇总.json", analysis)
        write_csv(folder/"逐场概率与日期特征.csv", entries, stats)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "source_directories": {s: str(p) for s, p in directories.items()}, "source_loading": sources,
        "periods": periods, "counts": {"new_schedule_fits": fit_counts, "original_snapshots_reused": reused,
            "new_schedule_predictions": sum(r["n"] for r in periods.values()),
            "original_predictions_reused": sum(r["n"] for r in periods.values()),
            "unique_matches": sum(r["n"] for r in periods.values()),
            "new_fits_this_call": new_fits_this_call, "complete_schedule_trials_reloaded_this_call": trials_reloaded},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "production_api_changed": False, "model_adopted": False}
    write_json(output/RESULT_FILE, result)
    (output/REPORT_FILE).write_text(report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": result["counts"],
        "differences": {k: v["mean_differences"] for k, v in periods.items()}}, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        (args.report_only/REPORT_FILE).write_text(report(_read_json(args.report_only/RESULT_FILE)), encoding="utf-8")
    else:
        run(args.root, args.parent, output=args.resume)
