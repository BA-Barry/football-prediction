"""Retrain the existing score model: earlier policy selection, later comparison.

Nine half-life/L2 pairs, unchanged features and 28-day updates. The later CSVs
are opened only after saving the earlier choice. Both seasons have already been
viewed in other research, so this is not a new blind test or a default change.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn

from .core_tuning import candidate_configs, run_candidate, select_policy
from .evaluation import probability_scores
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import (BLOCKS, REPEATS, SEED, align_tracks,
    calibration_summary, losses, moving_block_means, write_json)
from .goal_rate_diagnostics import rates_from_parameters, row_statistics, summary as goal_summary
from .native_experiment import expected_target_ids, load_seasons, season_directories
from .probability import score_grid, summarize
from .refresh_experiment import prediction_arrays, scores_for
from .refresh_research import PERIODS, _same, validate_trial
from .team_model import team_features


ARMS = ("original", "selected")
DOMAINS = ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way", "exact_score")
ORIGINAL_ID = "regularized_poisson_h180_a0p01"
RESULT_FILE = "核心参数训练结果.json"
REPORT_FILE = "核心参数训练说明.md"
POLICY_FILE = "较早赛季选定参数.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_source(root, period):
    start, _ = PERIODS[period]
    # Discover directory names, but never open later-season CSVs here.
    directories = {season: path for season, path in season_directories(root).items()
                   if int(season[:4]) <= start.year}
    return load_seasons(directories)


def original_trial(records, parent, period, competition, config):
    start, end = PERIODS[period]
    old = read_json(Path(parent) / period / f"{competition}_days_28.json")
    validate_trial(old, records, start, end, competition, interval=28)
    fits = {fit["before_date"]: fit for fit in old["fits"]}
    rows = [{**row, "rates": list(rates_from_parameters(fits[row["fit_before_date"]], row))}
            for row in old["predictions"]]
    return {**old, "config": dict(config), "update_interval_days": 28,
            "predictions": rows, "reused_from": str(Path(parent) / period / f"{competition}_days_28.json")}


def replay_rates(trial, records):
    """Recompute raw rates from actual fitted coefficients, not grid means."""
    if trial["failures"]:
        raise ValueError("Failed fits/predictions retained; incomplete candidates cannot be compared")
    source = {r.match_id: r for r in records}
    fits = {p["before_date"]: p for p in trial["fits"]}
    if len(fits) != len(trial["fits"]):
        raise ValueError("Duplicate fitted cutoff")
    start, end = (datetime.fromisoformat(trial[k]).date() for k in ("start", "end_exclusive"))
    expected = expected_target_ids(records, start, end, 28)
    if len(trial["predictions"]) != len(expected) or {r["match_id"] for r in trial["predictions"]} != expected:
        raise ValueError("Trial must retain its full expected cohort")
    for row in trial["predictions"]:
        record, parameters = source[row["match_id"]], fits[row["fit_before_date"]]
        if (parameters["regularization"] != trial["config"]["regularization"]
                or parameters["half_life_days"] != trial["config"]["half_life_days"]
                or parameters["rho"] != 0):
            raise ValueError("Fitted parameters differ from candidate configuration")
        _same(row["actual_score"], record.full_time, "source_score")
        if row["competition"] != record.competition or row["date"] != record.match_date.isoformat():
            raise ValueError("Forecast/source identity differs")
        vectors = team_features(record.home, record.away, record.competition)
        rates = [math.exp(parameters["intercept"] + math.fsum(
            parameters["coefficients"].get(key, 0.) * value for key, value in vector.items()))
            for vector in vectors]
        _same(row["rates"], rates, "raw_rates")
        _same(row["forecast"], summarize(score_grid(*rates, rho=0)), "coherent_score_forecast")
        _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "one_x_two")
    return trial


def aggregate(trials, config, period):
    start, end = PERIODS[period]
    result = {"config": dict(config), "start": start.isoformat(), "end_exclusive": end.isoformat(),
              "update_interval_days": 28}
    for field in ("predictions", "fits", "failures", "skipped"):
        result[field] = [entry for trial in trials for entry in trial[field]]
    result["predictions"].sort(key=lambda row: (row["date"], row["match_id"]))
    return result


def loss_vectors(rows):
    values = {}
    for domain, (p, y) in prediction_arrays(rows).items():
        observed = p[np.arange(len(p)), y]
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise ValueError("Observed-outcome probability must be positive; no clipping in core comparison")
        values[domain] = -np.log(observed)
    values["exact_score"] = np.asarray([row_statistics(row, row["rates"])["joint_score_nll"] for row in rows])
    return values


def score_rows(rows):
    return {**scores_for(rows), "exact_score": {"n": len(rows),
        "mean_nll": float(loss_vectors(rows)["exact_score"].mean())}}


def analyze(raw_tracks, *, repeats=REPEATS):
    if set(raw_tracks) != set(ARMS):
        raise ValueError("Compare original and earlier-selected forecasts only")
    tracks, coverage = align_tracks(json.loads(json.dumps(raw_tracks, allow_nan=False)))
    if any(coverage["outside_common"][arm] for arm in ARMS):
        raise ValueError("No silent coverage shrinkage between model policies")
    rows = {arm: list(tracks[arm].values()) for arm in ARMS}
    reference = rows["original"]
    for arm in ARMS:
        for row in rows[arm]:
            row_statistics(row, row["rates"])
            _same(row["forecast"], summarize(score_grid(*row["rates"], rho=0)), "analysis_forecast")
            _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "analysis_probabilities")
    per_loss = {arm: loss_vectors(rows[arm]) for arm in ARMS}
    differences = {domain: per_loss["selected"][domain] - per_loss["original"][domain] for domain in DOMAINS}
    result = {"n": len(reference), "coverage": coverage, "first_date": reference[0]["date"],
        "last_date": reference[-1]["date"], "scores": {arm: score_rows(rows[arm]) for arm in ARMS},
        "selected_minus_original": {domain: float(values.mean()) for domain, values in differences.items()},
        "calibration": {arm: calibration_summary(*prediction_arrays(rows[arm])["one_x_two"]) for arm in ARMS},
        "goal_diagnostics": {arm: goal_summary(rows[arm], [row_statistics(r, r["rates"]) for r in rows[arm]])
                             for arm in ARMS}, "bootstrap": {}}
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in reference], np.column_stack(list(differences.values())),
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["differences"] = {domain: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                               for i, domain in enumerate(DOMAINS)}
        result["bootstrap"][str(weeks)] = boot
    for name, labels in (("by_league", [r["competition"] for r in reference]),
                         ("by_month", [r["date"][:7] for r in reference])):
        result[name] = {}
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            result[name][label] = {"n": len(indices),
                "scores": {arm: score_rows([rows[arm][i] for i in indices]) for arm in ARMS},
                "selected_minus_original": {domain: float(values[indices].mean()) for domain, values in differences.items()}}
    result["markets"] = {}
    y = np.asarray([r["actual_outcome"] for r in reference])
    for market in MARKET_COLUMNS:
        indices = [i for i, row in enumerate(reference) if market in row["market_probabilities"]]
        item = {"n": len(indices), "missing_ids": [r["match_id"] for r in reference if market not in r["market_probabilities"]]}
        if indices:
            p = np.asarray([reference[i]["market_probabilities"][market] for i in indices])
            market_loss = losses(p, y[indices])
            diff = np.column_stack([per_loss[arm]["one_x_two"][indices]-market_loss for arm in ARMS])
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


def write_csv(path, tracks):
    per_loss = {arm: loss_vectors(tracks[arm]) for arm in ARMS}
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals"]
    fields += [f"{arm}_{key}" for arm in ARMS for key in
               ("home_rate", "away_rate", "p_home", "p_draw", "p_away", *(f"{domain}_nll" for domain in DOMAINS))]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(tracks["original"]):
            item = {key: row[key] for key in fields[:5]}
            item.update({"home_goals": row["actual_score"][0], "away_goals": row["actual_score"][1]})
            for arm in ARMS:
                item.update({f"{arm}_home_rate": tracks[arm][i]["rates"][0],
                             f"{arm}_away_rate": tracks[arm][i]["rates"][1]})
                item.update({f"{arm}_p_{label}": tracks[arm][i]["probabilities"][j]
                             for j, label in enumerate(("home", "draw", "away"))})
                item.update({f"{arm}_{domain}_nll": float(per_loss[arm][domain][i]) for domain in DOMAINS})
            writer.writerow(item)


def report(result):
    choice, later = result["selection"]["config"], result["later"]
    lines = ["# 核心球队概率模型：联合衰减与收缩训练", "",
        f"实际新增球队参数拟合{result['counts']['new_fits']}次；未改变正式默认模型。"
        f"较早{result['earlier']['n']}场选定半衰期{choice['half_life_days']}天、alpha={choice['regularization']}；"
        f"较晚{later['n']}场同场比较。所有文件／临时文件位于项目目录。", "",
        "## 为什么这次直接作用于预测", "",
        "不是额外数据审计，也不是在旧概率上加固定修正。8个新候选实际重新估计球队攻击、防守、联赛进球环境及主场系数；原参数轨复用。"
        "半衰期决定历史赛果的记忆长度，L2系数决定球队参数向总体收缩的程度；两者会改变每场主客预期进球，"
        "再由同一个完整比分分布计算胜平负、总进球和客让两球三项概率。", "",
        "## 实验与选择", "",
        "仅搜索90/180/365天 × alpha 0.001/0.01/0.1九组；原模型为180天 × 0.01。"
        "各联赛各自拟合，五联赛共用一套超参数。逐场等权，以2024/25按28天前推预测的胜平负log loss选参数，"
        "平局按候选id排序；没有按联赛单独挑选，也未根据较晚赛季二次改参。"
        "2022/23、2023/24提供初始历史，后续快照仅使用严格更早日期赛果。保持原特征、独立Poisson比分分布、"
        "28天更新、未知队伍不填先验及特殊比赛排除规则。旧参数轨复用已核对的存档。", "",
        "先保存完整较早候选表与选择文件，再读取2025/26。两个赛季都已在其他研究中见过，"
        "因此本轮是探索性跨季检验，不是全新盲测；较早获选成绩尤其存在选择乐观偏差。", "",
        "| 半衰期 | alpha | 较早胜平负LL |", "|---:|---:|---:|"]
    for config in result["settings"]["candidates"]:
        value = result["selection"]["candidate_scores"][config["id"]]["log_loss"]
        lines.append(f"| {config['half_life_days']} | {config['regularization']} | {value:.6f} |")
    ranked = sorted(result["selection"]["candidate_scores"].items(), key=lambda pair: (pair[1]["log_loss"], pair[0]))
    gap = ranked[1][1]["log_loss"] - ranked[0][1]["log_loss"]
    lines += ["", f"较早前两名 {ranked[0][0]} 与 {ranked[1][0]} 的平均LL差仅{gap:.9f}；"
              "按预定规则获选不等于收缩强度已被稳健区分，也不等于找到了普适最优值。"]
    lines += ["", "## 较晚同场结果（损失越低越好）", "",
        "| 项目 | 原模型 | 较早选定模型 | 新减旧 | 四周区块95%区间 |", "|---|---:|---:|---:|---|"]
    for domain in DOMAINS:
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        a, b = (later["scores"][arm][domain][field] for arm in ARMS)
        ci = later["bootstrap"]["4"]["differences"][domain]["interval_95"]
        interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {domain} | {a:.6f} | {b:.6f} | {b-a:+.6f} | {interval} |")
    lines += ["", "| 联赛 | 场次 | 原胜平负LL | 新胜平负LL |", "|---|---:|---:|---:|"]
    for league, item in later["by_league"].items():
        lines.append(f"| {league} | {item['n']} | {item['scores']['original']['one_x_two']['log_loss']:.6f} | "
                     f"{item['scores']['selected']['one_x_two']['log_loss']:.6f} |")
    primary = later["bootstrap"]["4"]["differences"]["one_x_two"]
    interval = primary["interval_95"]
    lines += ["", "本次选择不是自动采纳理由。胜平负损失差为"
        f"{primary['mean']:+.6f}；"
        + ("四周区间跨零，不能确认稳定改善。" if interval and interval[0] <= 0 <= interval[1]
           else "仅凭本轮探索性区间不能确认未来表现。")
        + "总进球、让球和精确比分各自有不同评分目标，不能用胜平负的点改善覆盖其他目标的退步。"
        "正式模型仍保留原参数；这次优先回答联合调参是否已足够，而不是保证选出新赢家。"
        "结论限定于这9组，不能由此排除其他超参数范围仍有收益。"]
    lines += ["", "## 市场对照（仅比较，不融合）", "",
        "| 市场 | 可比场数 | 市场LL | 原模型LL | 新模型LL |", "|---|---:|---:|---:|---:|"]
    for market, item in later["markets"].items():
        if item["n"]:
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['original']['log_loss']:.6f} | "
                f"{item['model_scores_on_same_matches']['selected']['log_loss']:.6f} |")
    lines += ["", "## 如何理解结果及尚未解决的问题", "",
        "主评估为胜平负LL，Brier、RPS、准确率在JSON中一并保留；总进球分为0/1/2/3/4+，"
        "竞彩客让两球让胜为客队净胜至少3球，净胜2球为让平，不是亚洲盘走水。"
        "精确比分NLL由原始未截断λ解析计算，不以有限网格均值替代。", "",
        "同一比赛成对损失差，用共享五联赛日历的连续移动区块，4周为主、2/8周敏感性，"
        f"{result['settings']['bootstrap_repeats']}次、seed={SEED}。区间条件于已拟合预测及较早选参，"
        "不包含模型重训／参数选择不确定性，也不是未来盈利区间。结果同时报告全部领域，未只挑改善项。", "",
        "这仍是球队边际模型：没有验证为可靠的首发球员、伤停、跨赛事强度和重视程度输入，"
        "也未证明对欧战或当前比赛可直接外推。半全场还需要独立的上下半场/状态变化模型，不能据此冒充已完成。"
        "本轮不加入深度学习、不融合市场、不自动更换正式参数、不提供投注保证。"
        "历史市场报价仅作去水对照，其采集时点不等于已知开盘／T−24h；不据此宣称可交易收益。", "",
        "实验设计技能使本轮保持同特征、同比赛和同更新频率，只改变两个核心参数；统计分析技能用于成对损失与赛程相关性区间。", "",
        "方法来源：[scikit-learn PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)，"
        "[连续移动区块](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html)。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    configs = candidate_configs()
    original_config = next(c for c in configs if c["id"] == ORIGINAL_ID)
    settings = {"candidates": configs, "selection": "pooled 2024/25 chronological 1X2 log loss",
        "update_interval_days": 28, "bootstrap_blocks_weeks": list(BLOCKS), "bootstrap_repeats": repeats,
        "seed": SEED, "previously_viewed_seasons": True, "production_api_changed": False,
        "source_reference": str(parent), "raw_rates_not_grid_means": True}
    write_json(output / "实验设置.json", settings)
    earlier_records, earlier_sources = load_source(root, "2024-25")
    start, end = PERIODS["2024-25"]
    expected = set().union(*(expected_target_ids([r for r in earlier_records if r.competition == comp], start, end, 28)
                             for comp in sorted(LEAGUES)))
    validation, new_fits, reused_fits = [], 0, 0
    for config in configs:
        trials = []
        destination = output / "2024-25" / config["id"]
        destination.mkdir(parents=True)
        for comp in sorted(LEAGUES):
            records = [r for r in earlier_records if r.competition == comp]
            if config["id"] == ORIGINAL_ID:
                trial = original_trial(records, parent, "2024-25", comp, config)
                reused_fits += len(trial["fits"])
            else:
                trial = run_candidate(records, config, start, end)
                new_fits += len(trial["fits"])
            write_json(destination / f"{comp}.json", trial)
            replay_rates(trial, records)
            trials.append(trial)
            print(f"validation {config['id']} {comp}: fits={len(trial['fits'])} predictions={len(trial['predictions'])}", flush=True)
        validation.append(aggregate(trials, config, "2024-25"))
    selection = select_policy(validation, expected)
    table = {trial["config"]["id"]: score_rows(trial["predictions"]) for trial in validation}
    write_json(output / "较早候选完整成绩.json", table)
    write_json(output / POLICY_FILE, selection)
    print(f"earlier policy saved: {selection['config']}", flush=True)
    # Nothing above this line has loaded 2025/26 outcomes or forecast archives.
    later_records, later_sources = load_source(root, "2025-26")
    start, end = PERIODS["2025-26"]
    later_trials = {arm: [] for arm in ARMS}
    choice = selection["config"]
    for comp in sorted(LEAGUES):
        records = [r for r in later_records if r.competition == comp]
        old = original_trial(records, parent, "2025-26", comp, original_config)
        reused_fits += len(old["fits"])
        selected = old if choice == original_config else run_candidate(records, choice, start, end)
        if choice != original_config:
            new_fits += len(selected["fits"])
        for arm, trial in (("original", old), ("selected", selected)):
            destination = output / "2025-26" / arm
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / f"{comp}.json", trial)
            replay_rates(trial, records)
            later_trials[arm].append(trial)
        print(f"later selected {comp}: fits={len(selected['fits'])} predictions={len(selected['predictions'])}", flush=True)
    original_validation = next(r for r in validation if r["config"]["id"] == ORIGINAL_ID)
    selected_validation = next(r for r in validation if r["config"] == choice)
    earlier, earlier_tracks = analyze({"original": original_validation["predictions"],
                                      "selected": selected_validation["predictions"]}, repeats=repeats)
    later, later_tracks = analyze({arm: aggregate(trials, choice if arm == "selected" else original_config,
                                                "2025-26")["predictions"] for arm, trials in later_trials.items()}, repeats=repeats)
    result = {"settings": settings, "selection": selection, "candidate_scores_all_domains": table,
        "earlier": earlier, "later": later, "sources": {"earlier": earlier_sources, "later": later_sources},
        "counts": {"new_fits": new_fits, "reused_parameter_snapshots": reused_fits,
                   "new_validation_fits": sum(len(trial["fits"]) for trial in validation
                                               if trial["config"]["id"] != ORIGINAL_ID),
                   "new_later_fits": 0 if choice == original_config else sum(len(trial["fits"])
                                                                              for trial in later_trials["selected"]),
                   "new_validation_candidates": len(configs)-1, "later_policies_fitted": int(choice != original_config),
                   "later_forecasts_per_arm": later["n"]},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__, "elapsed_seconds": time.perf_counter()-started}}
    write_csv(output / "较早同场概率.csv", earlier_tracks)
    write_csv(output / "较晚同场概率.csv", later_tracks)
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    return result


def main():
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = args.parent or args.root / "outputs" / "更新频率研究_20261023T073047979805Z"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "outputs" / f"核心参数训练_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "config": result["selection"]["config"],
        "new_fits": result["counts"]["new_fits"], "later_differences": result["later"]["selected_minus_original"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
