"""Fixed v37 structure transferred to saved StatsBomb XI/role observations.

This is another already-viewed historical season, not a new blind test. No
coefficient, supplier identity, selection cohort or loss from the old stint
experiment is reused. Only the externally declared structure/settings carry.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .english_tiers_experiment import METRICS, summarize_rows
from .forecast_diagnostics import moving_block_means
from .gaussian_goal_mixture import score
from .hierarchical_xi_model import HierarchicalStartingXIModel, UnknownHierarchicalTeamError
from .hierarchical_xi_research import fixture_only, paired_comparison, _numerical_difference
from .statsbomb_data import write_json


TARGET_FROM, TARGET_THROUGH = date(2016, 2, 1), date(2016, 5, 17)
PRIOR_SD, PRIMARY_ORDER, CHECK_ORDER = .10, 15, 25
ARMS = ("team_laplace", "players_laplace")
BLOCKS, REPEATS, SEED = (2, 4, 8), 2000, 20261024


def _resolved(path):
    return Path(path).expanduser().resolve()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _day(text):
    if not isinstance(text, str) or len(text) != 10:
        raise ValueError("Use explicit calendar dates")
    value = date.fromisoformat(text)
    if value.isoformat() != text:
        raise ValueError("Use explicit calendar dates")
    return value


def weekly_windows(matches):
    """All fixed target fixtures; strictly earlier Monday histories only."""
    if not isinstance(matches, list) or not matches:
        raise ValueError("Nonempty role-aware match list required")
    seen, dates, grouped = set(), {}, defaultdict(list)
    for match in matches:
        mid = match["match_id"]
        if not isinstance(mid, str) or not mid.startswith("statsbomb:match:") or mid in seen:
            raise ValueError("Distinct StatsBomb match IDs required")
        seen.add(mid)
        day = _day(match["match_date"])
        dates[mid] = day
        if TARGET_FROM <= day <= TARGET_THROUGH:
            grouped[day - timedelta(days=day.weekday())].append(match)
    if not grouped:
        raise ValueError("No matches in the fixed transfer target interval")
    for cutoff, targets in sorted(grouped.items()):
        history = sorted((m for m in matches if dates[m["match_id"]] < cutoff),
                         key=lambda m: (m["match_date"], m["match_id"]))
        if not history:
            raise ValueError("No strictly earlier observations for weekly fit")
        yield cutoff, history, sorted(targets, key=lambda m: (m["match_date"], m["match_id"]))


def fixed_policy():
    return {
        "question": "Fixed v37 role/player structure transferred to StatsBomb EPL 2015/16",
        "target_from": TARGET_FROM.isoformat(), "target_through_inclusive": TARGET_THROUGH.isoformat(),
        "history": "same expanding history strictly before each Monday; no same-week updates",
        "prior_sd": PRIOR_SD, "candidate_scales": [PRIOR_SD], "hyperparameter_searches": 0,
        "components": "role and individual attack/defence each independent N(0, s^2/2); fixed not learned",
        "team_ridge": 1., "home_ridge": 1., "intercept_penalty": 0., "max_iterations": 2000,
        "loss": "summed two-sided complete full-match Poisson NLL; no minute offset or time decay",
        "reference": "per-team frequencies and role counts from the same strictly earlier history",
        "posterior": "full-Hessian MAP Laplace; unseen personal residuals integrated as Gaussian priors",
        "quadrature": {"primary_order": PRIMARY_ORDER, "check_order": CHECK_ORDER},
        "arms": list(ARMS), "primary_metric": "full_score_nll; players minus matched team, lower better",
        "source_identities": "StatsBomb only; never relabel as IMPECT or reuse foreign coefficients",
        "source_selection": "full catalogue denominator; explicit XI/role/source failures, not old stint-clock subset",
        "scope": "already-viewed season; known opening XI/roles condition; unknown historical publication times",
        "anchor_limit": "s=.10 traces to prior research including this supplier's earlier 2015/16 segment; not pure independent external parameter validation",
        "bootstrap": {"block_weeks": list(BLOCKS), "repeats": REPEATS, "seed": SEED,
                      "scope": "paired conditional exploration; not retraining/selection/shift uncertainty or simultaneous intervals"},
        "promotion": "none; saved current forecasts, LLM/market lines and production settings unchanged",
    }


def score_targets(model, targets):
    """Strip outcomes before predicting; attach actual goals only for scoring."""
    rows, numerical = [], []
    for match in targets:
        base = {"match_id": match["match_id"], "match_date": match["match_date"]}
        try:
            distribution = model.predict_distribution(fixture_only(match))
        except UnknownHierarchicalTeamError as error:
            rows.append({**base, "status": "missing", "reason": str(error)})
            continue
        primary = score(distribution["log_mean"], distribution["log_cov"], match["full_time"], order=PRIMARY_ORDER)
        check = score(distribution["log_mean"], distribution["log_cov"], match["full_time"], order=CHECK_ORDER)
        numerical.append({**base, **_numerical_difference(primary, check)})
        rows.append({**base, "status": "predicted", "actual": list(match["full_time"]),
                     "has_unknown_player": bool(distribution["unknown_player_ids"]),
                     "distribution": distribution, **primary})
    return rows, numerical


def compare_tracks(tracks):
    supported = {arm: [r for r in tracks[arm] if r["status"] == "predicted"] for arm in ARMS}
    unknown = {r["match_id"] for r in supported["players_laplace"] if r["has_unknown_player"]}
    known = {r["match_id"] for r in supported["players_laplace"]} - unknown
    return {name: paired_comparison(supported["team_laplace"], supported["players_laplace"], ids)
            for name, ids in (("all", None), ("known", known), ("unknown", unknown))}


def conditional_intervals(comparison, team_rows):
    paired = comparison["paired_matches"]
    if not paired:
        return {"status": "unavailable", "reason": "no common supported fixtures"}
    dates = {r["match_id"]: r["match_date"] for r in team_rows}
    values = [[r[key] for key in METRICS] for r in paired]
    outputs = {}
    for block in BLOCKS:
        result = moving_block_means([dates[r["match_id"]] for r in paired], values,
                                    block_weeks=block, repeats=REPEATS, seed=SEED)
        means, intervals = result.pop("mean"), result.pop("interval_95")
        result["metrics"] = {key: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                             for i, key in enumerate(METRICS)}
        outputs[str(block)] = result
    return outputs


def _descriptives(targets):
    totals = [sum(m["full_time"]) for m in targets]
    return {"n": len(totals), "total_goals_mean": statistics.mean(totals),
            "total_goals_sd": statistics.stdev(totals) if len(totals) > 1 else None,
            "total_goals_median": statistics.median(totals),
            "total_goals_0_1_2_3_4plus": dict(Counter(str(min(n, 4)) if n < 4 else "4+" for n in totals))}


def run(root, source, inspection):
    root, source, inspection = map(_resolved, (root, source, inspection))
    matches, receipt = _read(source), _read(inspection)
    windows = list(weekly_windows(matches))
    if receipt.get("included_matches") != len(matches):
        raise ValueError("Saved inspection and included matches disagree")
    observations = receipt.get("matches")
    if not isinstance(observations, list) or len(observations) != receipt.get("expected_catalogue_matches"):
        raise ValueError("Full original catalogue denominator required")
    if len({r["source_match_id"] for r in observations}) != len(observations):
        raise ValueError("Duplicate source catalogue IDs")
    included_ids = {f"statsbomb:match:{r['source_match_id']}" for r in observations if r["included"]}
    if included_ids != {m["match_id"] for m in matches}:
        raise ValueError("Source inspection and converted match identities disagree")
    targets = [m for _, _, rows in windows for m in rows]
    requested_targets = [r for r in observations if TARGET_FROM <= _day(r["match_date"]) <= TARGET_THROUGH]
    created = datetime.now(timezone.utc)
    output = root / "outputs" / created.strftime("首发固定结构迁移_%Y%m%dT%H%M%S%fZ")
    (output / "models").mkdir(parents=True, exist_ok=False)
    policy = {**fixed_policy(), "created_at_utc": created.isoformat(), "source": str(source),
              "inspection": str(inspection), "catalogue_matches": len(observations),
              "included_source_matches": len(matches), "catalogue_targets": len(requested_targets),
              "included_targets": len(targets)}
    write_json(output / "固定迁移设置.json", policy)
    write_json(output / "源检验说明副本.json", receipt)
    tracks, errors, weeks, fits = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, [], 0
    begun = time.perf_counter()
    try:
        for cutoff, history, target in windows:
            print(f"XI transfer {cutoff}: history={len(history)} targets={len(target)}", flush=True)
            week = {"week_start": cutoff.isoformat(), "history_matches": len(history),
                    "history_max_date": history[-1]["match_date"], "target_ids": [m["match_id"] for m in target], "models": {}}
            local = {}
            for arm in ARMS:
                model = HierarchicalStartingXIModel(include_players=arm == "players_laplace", prior_sd=PRIOR_SD).fit(history, cutoff)
                fits += 1
                path = output / "models" / f"{cutoff}_{arm}.json"
                model.save(path)
                restored = HierarchicalStartingXIModel.load(path)
                if model.parameters() != restored.parameters() or not np.array_equal(model.precision_cholesky, restored.precision_cholesky):
                    raise ValueError("Restored fixed model differs from saved fit")
                predicted, measured = score_targets(restored, target)
                tracks[arm].extend(predicted)
                errors[arm].extend(measured)
                local[arm] = predicted
                week["models"][arm] = {"parameters": str(path), "optimizer": restored.optimizer_diagnostics,
                                        "coverage": summarize_rows(predicted)["n"]}
                write_json(output / f"{arm}_predictions.json", tracks[arm])
            week["comparisons"] = compare_tracks(local)
            weeks.append(week)
            write_json(output / "逐周拟合与比较.json", weeks)
        comparisons = compare_tracks(tracks)
        numerical = {arm: {key: max(r[key] for r in rows) for key in rows[0] if key in METRICS or key.endswith("_probability")}
                     if rows else None for arm, rows in errors.items()}
        result = {"status": "complete", "policy": policy, "model_fits_completed": fits, "weeks": len(weeks),
                  "elapsed_seconds": time.perf_counter()-begun, "hyperparameter_searches": 0,
                  "catalogue_matches": len(observations), "included_source_matches": len(matches),
                  "catalogue_targets": len(requested_targets), "included_targets": len(targets),
                  "source_missing_targets": [r for r in requested_targets if not r["included"]],
                  "target_descriptives": _descriptives(targets), "comparisons": comparisons,
                  "tracks": {arm: {"requested": len(targets), "summary": summarize_rows(rows),
                                   "missing": [r for r in rows if r["status"] != "predicted"]} for arm, rows in tracks.items()},
                  "conditional_block_intervals": conditional_intervals(comparisons["all"], tracks["team_laplace"]),
                  "quadrature_max_absolute_difference_15_vs_25": numerical,
                  "limits": "already-viewed historical conditional study; not causal, new blind test, live promotion or validated posterior coverage"}
        write_json(output / "数值积分逐场误差.json", errors)
        write_json(output / "首发固定结构迁移结果.json", result)
        (output / "首发固定结构迁移说明.md").write_text(make_report(result, weeks), encoding="utf-8")
        print(f"XI transfer complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"status": "incomplete", "completed_fits": fits,
                                                  "completed_weeks": len(weeks), "error": f"{type(error).__name__}: {error}"})
        raise


def make_report(result, weeks):
    def fmt(value):
        return "未知" if value is None else f"{value:.6f}"
    comparison = result["comparisons"]["all"]
    lines = ["# 首发球员固定结构：StatsBomb跨来源／赛季复现", "",
             "数据来源：StatsBomb Open Data（本地保存的开场首发／位置），独立Football-Data全场比分。仅本地个人非商业研究；原始来源及署名资料保留，不再分发观测。", "",
             f"完整目录{result['catalogue_matches']}场，首发／位置纳入{result['included_source_matches']}场；固定目标2016-02-01至2016-05-17，目录{result['catalogue_targets']}场，纳入{result['included_targets']}场。",
             f"实际{result['weeks']}个比赛周、{result['model_fits_completed']}次拟合，参数搜索0次。主尺度只用原s=.10，不重复搜索其他尺度。", "",
             "两线在每周一使用完全相同、严格更早的本季历史，重估本来源系数，不搬用旧IMPECT球员或系数。球队／主场惩罚1、完整全场似然，非主统计线的alpha=.01／180天衰减口径。", "",
             "| 全部共同支持指标 | 球队Laplace | 角色＋首发Laplace | 后者−前者 |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        lines.append(f"| {metric} | {fmt(comparison['team'][metric])} | {fmt(comparison['players'][metric])} | {fmt(comparison['mean_players_minus_team'][metric])} |")
    lines += ["", "完整比分NLL为主指标；除top1越高越好，其余损失越低越好。不以多猜中一场或某次要指标改善覆盖主指标。", "",
              "| 按较早历史定义的球员身份子集 | 场数 | 平均比分NLL差 |", "|---|---:|---:|"]
    for name in ("known", "unknown"):
        item = result["comparisons"][name]
        lines.append(f"| {name} | {item['common_matches']} | {fmt(item['mean_players_minus_team']['full_score_nll'])} |")
    lines += ["", "未知不是已知零影响：仍积分个人残差先验。身份子集仅描述，不用来选择模型或归因；缺失场次全部在JSON明列。", "",
              "## 逐周主指标", "", "| 周起点 | 历史场数 | 共同目标场数 | 平均比分NLL差 |", "|---|---:|---:|---:|"]
    for week in weeks:
        item = week["comparisons"]["all"]
        lines.append(f"| {week['week_start']} | {week['history_matches']} | {item['common_matches']} | {fmt(item['mean_players_minus_team']['full_score_nll'])} |")
    lines += ["", "## 不确定性与限制", "",
              "| 日历区块长度 | 比分NLL差的条件95%区间 |", "|---|---:|"]
    for length, block in result["conditional_block_intervals"].items():
        if "metrics" in block:
            interval = block["metrics"]["full_score_nll"]["interval_95"]
            text = "不可用" if interval is None else f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
            lines.append(f"| {length}周 | {text} |")
    lines += ["", "复用2／4／8周非环形共同日历区块重采样，2000次、固定seed20261024，全部指标与区块长度均保留。探索区间以已拟合预测为条件，不包含重训、结构选择、赛季漂移或单场真实概率误差，也没有多指标同时覆盖保证；不作IID显著性／等效结论。非环形区块端点非均匀加权，尤其8周相对16周跨度很长，不能读作稳定跨季外推证据。", "",
              "15与25阶积分仅测数值误差，不证明Laplace是真实后验。各场分布、未知身份、精度矩阵与较早训练ID均保存，原始缺失不靠事后比分或红牌删去。", "",
              "该赛季已被旧研究查看，且固定尺度的旧锚点也来自此来源较早的在场研究，因此不是新的盲测或纯外部参数验证。历史首发／开场角色公布时间未知；初始事件的00:00是比赛时钟，不是公开时间。这是首发及角色已知的条件研究，不是T−24、真实首发发布回放或当前2026球员验证。", "",
              "原近期预测、市场与LLM设置不变，不自动上线，不继续围绕本期改先验。实验设计技能使同场／同历史／固定结构分开，统计分析技能使分母、效应幅度、条件区间及负结果完整保留。所有代码、模型、临时与报告在项目目录。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--inspection", required=True)
    args = parser.parse_args(argv)
    run(args.root, args.source, args.inspection)


if __name__ == "__main__":
    main()
