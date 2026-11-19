"""Bounded seasonal strength comparison on previously inspected EPL seasons.

Three innovation scales are compared on the earlier forward season only. The
saved selection is reused for two later forward seasons, not reselected there.
This is date-level exploratory research, not a timed live forecast service.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time
import warnings

from sklearn.exceptions import ConvergenceWarning

from .date_model import DateTeamGoalModel
from .english_tiers_experiment import load_inspections, score_forecast, summarize_rows, METRICS
from .season_state_model import SeasonStateTeamGoalModel, season_for_date


SCALES = {"q0": 0., "q05": .5, "q1": 1.}
EARLY, LATER = "2023-24", ("2024-25", "2025-26")


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_records(records):
    """Select EPL without altering the existing source inspection convention."""
    rows = sorted((r for r in records if r.competition == "E0"),
                  key=lambda r: (r.match_date, r.match_id))
    if not rows or len({r.match_id for r in rows}) != len(rows):
        raise ValueError("Need nonempty, uniquely identified EPL records")
    for r in rows:
        if type(r.match_date) is not date or season_for_date(r.match_date) != r.season:
            raise ValueError("EPL source season must agree with July-June dates")
        if r.season not in ("2022-23", EARLY, *LATER):
            raise ValueError("Research uses the declared four seasons only")
        if r.home == r.away or not r.home or not r.away:
            raise ValueError("Need distinct nonempty EPL teams")
        if len(r.full_time) != 2 or any(type(g) is not int or g < 0 for g in r.full_time):
            raise ValueError("Need two nonnegative integer goals")
    return rows


def restore_original(payload):
    """Restore the old coefficient predictor, without another optimization."""
    if payload.get("model_type") != "date_level_regularized_poisson":
        raise ValueError("Need original date-level model parameters")
    model = DateTeamGoalModel(payload["regularization"], payload["half_life_days"])
    model.coefficients = dict(payload["coefficients"])
    model.intercept = payload["intercept"]
    model.before_date = date.fromisoformat(payload["before_date"])
    model.training_min_date = date.fromisoformat(payload["training_min_date"])
    model.training_max_date = date.fromisoformat(payload["training_max_date"])
    if not model.training_min_date <= model.training_max_date < model.before_date:
        raise ValueError("Original saved training dates must precede cutoff")
    model.teams, model.competitions = set(payload["teams"]), set(payload["competitions"])
    model.n_matches = payload["n_matches"]
    model.reference_match_ids = list(payload["reference_match_ids"])
    model.optimizer_iterations = payload["optimizer_iterations"]
    if (len(set(model.reference_match_ids)) != model.n_matches
            or len(model.reference_match_ids) != model.n_matches
            or not math.isfinite(model.intercept)
            or any(not math.isfinite(x) for x in model.coefficients.values())):
        raise ValueError("Invalid saved original coefficients or reference IDs")
    return model


def _new_model(arm, q):
    return (DateTeamGoalModel(.01, 180) if arm == "original"
            else SeasonStateTeamGoalModel(.01, 180, innovation_scale=q))


def _restore_model(arm, parameters):
    return (restore_original(parameters) if arm == "original"
            else SeasonStateTeamGoalModel.from_parameters(parameters))


def _comparison(rows, arms):
    paired = [r for r in rows if all(r["arms"][a]["status"] == "predicted" for a in arms)]
    scores = {a: summarize_rows([r["arms"][a] for r in paired]) for a in arms}
    reference = arms[0]
    return {"n": len(paired), "scores": scores,
            "paired_match_ids": [r["match_id"] for r in paired],
            "differences_from_reference": {
                a: {m: scores[a][m]-scores[reference][m] if paired else None for m in METRICS}
                for a in arms[1:]}, "reference_arm": reference}


def summarize_season(rows, arms):
    result = {"target_count": len(rows), "arms": {}, "all_paired": _comparison(rows, arms),
              "by_window": {}, "by_month": {}, "by_opening_period": {},
              "by_prior_epl_status": {}, "by_previous_epl_presence": {}}
    for arm in arms:
        predicted = [r["arms"][arm] for r in rows if r["arms"][arm]["status"] == "predicted"]
        result["arms"][arm] = {"coverage": len(predicted),
            "all_supported_scores_secondary": summarize_rows(predicted),
            "missing": [{"match_id": r["match_id"], **r["arms"][arm]} for r in rows
                        if r["arms"][arm]["status"] != "predicted"]}
    fields = {"by_window": lambda r: r["cutoff"],
              "by_month": lambda r: r["match_date"][:7],
              "by_opening_period": lambda r: "aug_sep" if r["match_date"][5:7] in ("08", "09") else "rest",
              "by_prior_epl_status": lambda r: r["prior_epl_status"],
              "by_previous_epl_presence": lambda r: r["previous_epl_presence"]}
    for kind, key in fields.items():
        for label in sorted({key(r) for r in rows}):
            group = [r for r in rows if key(r) == label]
            result[kind][label] = {"target_count": len(group), **_comparison(group, arms)}
    return result


def run_season(records, season, arms, output):
    """Fit each declared arm once per nonempty 28-day window; no retries."""
    if season not in (EARLY, *LATER) or not arms:
        raise ValueError("Declared research season and arms required")
    year = int(season[:4])
    start, stop = date(year, 8, 1), date(year+1, 6, 1)
    targets = [r for r in records if r.season == season and start <= r.match_date < stop]
    if not targets:
        raise ValueError("No target fixtures for research season")
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate match identifiers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parameter_dir = output/"parameters"
    parameter_dir.mkdir()
    past = {t for r in records if r.match_date < date(year, 7, 1) for t in (r.home, r.away)}
    prior_label = f"{year-1}-{year % 100:02d}"
    previous = {t for r in records if r.season == prior_label for t in (r.home, r.away)}
    per_fixture, fits, cutoff = [], [], start
    while cutoff < stop:
        end = min(stop, cutoff+timedelta(days=28))
        window = sorted((r for r in targets if cutoff <= r.match_date < end),
                        key=lambda r: (r.match_date, r.match_id))
        if window:
            history = [r for r in records if r.match_date < cutoff and r.goal_fit_exclusion is None]
            if not history:
                raise ValueError("No strictly earlier eligible history")
            window_rows = {}
            for target in window:
                teams = {target.home, target.away}
                status = ("local_first_seen" if teams-past else
                          "returning" if teams-previous else "continuing")
                window_rows[target.match_id] = {
                    "match_id": target.match_id, "match_date": target.match_date.isoformat(),
                    "home": target.home, "away": target.away, "actual": list(target.full_time),
                    "cutoff": cutoff.isoformat(), "prior_epl_status": status,
                    "previous_epl_presence": "both_previous" if teams <= previous else "at_least_one_not_previous",
                    "arms": {}}
            for arm, q in arms.items():
                begun = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model = _new_model(arm, q).fit(history, cutoff)
                parameters = model.parameters()
                path = parameter_dir/f"{arm}_{cutoff.isoformat()}.json"
                _write(path, {"arm": arm, "innovation_scale": q,
                              "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                              "parameters": parameters})
                # Use the serialized algebra, not the still-live fitted object.
                saved = json.loads(path.read_text(encoding="utf-8"))
                restored = _restore_model(arm, saved["parameters"])
                fits.append({"arm": arm, "innovation_scale": q, "cutoff": cutoff.isoformat(),
                    "training_matches": len(history),
                    "training_max_date": max(r.match_date for r in history).isoformat(),
                    "training_current_season_matches": sum(r.season == season for r in history),
                    "optimizer_iterations": parameters["optimizer_iterations"],
                    "parameter_path": str(path), "elapsed_seconds": time.perf_counter()-begun})
                for target in window:
                    missing = sorted({target.home, target.away}-restored.teams)
                    if missing:
                        value = {"status": "missing", "reason": "unseen_team_no_estimated_prior",
                                 "missing_teams": missing, "rates": None, "forecast": None,
                                 "parameter_path": str(path)}
                    else:
                        rates = tuple(restored.predict(target.home, target.away, "E0", target.match_date))
                        value = {"status": "predicted", "rates": list(rates), "parameter_path": str(path),
                                 **score_forecast(rates, target.full_time)}
                    window_rows[target.match_id]["arms"][arm] = value
            per_fixture.extend(window_rows[t.match_id] for t in window)
            print(json.dumps({"season": season, "completed_window": cutoff.isoformat(),
                              "fixtures": len(window), "fits_so_far": len(fits)}), flush=True)
        cutoff = end
    if {r["match_id"] for r in per_fixture} != {r.match_id for r in targets}:
        raise ValueError("Every target must occur exactly once")
    result = {"season": season, "arm_scales": arms, "fit_count": len(fits), "fits": fits,
              "per_fixture": per_fixture, "summary": summarize_season(per_fixture, list(arms))}
    _write(output/"赛季结果.json", result)
    return result


def select_scale(early):
    if early["season"] != EARLY or early["arm_scales"] != SCALES:
        raise ValueError("Select only declared scales on the earlier forward season")
    common = early["summary"]["all_paired"]
    if not common["n"]:
        raise ValueError("No common early predictions; cannot choose a scale")
    losses = {arm: common["scores"][arm]["full_score_nll"] for arm in SCALES}
    if any(not math.isfinite(x) for x in losses.values()):
        raise ValueError("Early losses must be finite")
    selected = min(SCALES, key=lambda a: (losses[a], SCALES[a]))
    return {"selection_season": EARLY, "candidate_scales": SCALES, "losses": losses,
            "selected_arm": selected, "selected_scale": SCALES[selected],
            "objective": "same-fixture early forward mean full-score NLL",
            "n_common": common["n"], "tie_rule": "exact minimum tie -> smaller q",
            "training_match_ids": common["paired_match_ids"],
            "scope": "earlier selection, not independent evaluation; no later re-selection"}


def _report(result):
    selection = result["selection"]
    lines = ["# 跨赛季球队状态：固定尺度历史前推", "",
             f"早期{EARLY}三个候选q=0/.5/1，按同场完整比分NLL选择q={selection['selected_scale']:g}。"
             "选择先存盘重读，后两季固定；不是新盲测或自动上线。", "",
             "|阶段／赛季|目标|共同|线路|完整比分NLL|1X2 LL|Brier sum|Top1|",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for phase in (result["early"], *result["later"].values()):
        summary = phase["summary"]
        for arm, scores in summary["all_paired"]["scores"].items():
            if scores["n"]:
                lines.append(f"|{phase['season']}|{summary['target_count']}|{scores['n']}|{arm}|"
                    f"{scores['full_score_nll']:.6f}|{scores['one_x_two_log_loss']:.6f}|"
                    f"{scores['one_x_two_brier']:.6f}|{scores['top1']:.2%}|")
    combined = result["later_combined"]
    lines += ["", f"实际拟合{result['total_fits']}次；三个固定候选、早期选择1次、后期选择0次。"
              f"后两季合计目标{combined['target_count']}、共同{combined['all_paired']['n']}场。",
              "", "全部8指标、逐场概率／损失、参数路径、覆盖和缺失原因，以及按窗／月／开季／身份的描述分组保存在JSON。"
              "只在共同场次计算配对差，不拿覆盖不同的线路均值相减；早期只用于选尺度。",
              "", "## 模型与边界", "",
              "基础球队攻防＋q乘首次历史出现年以后逐年度创新；首次年只从严格较早history确定。"
              "alpha=.01、180天、28日和归一化加权损失沿原拟合器，独立Poisson不变。"
              "q=0为原特征端点；q改变允许变化与有效惩罚，不能因果归为真实季间信号。",
              "已知球队未观察的新季创新MAP条件均值0承接，不表示确定不变；本轮未积分状态不确定性。"
              "缺席EPL年度不压缩，同列创新由proper ridge分配，不是各年独立识别。"
              "2023/24没有缺席后回归球队，故早期选q没有验证gap演化；后两季回归组是明确外推。",
              "只用四季本地E0，不加入E1、球员、LLM、市场或新跨级折扣。完全未知球队明确缺失，"
              "不以全季成员构造训练效应或填0。全源读取只作解析／队列审计，早期函数仅接2024-07-01以前资料。",
              "较晚第二季可使用已结束前季作为history，这是扩展历史前推而非冻结两年静态留出。"
              "原资料已查看且首次发布时间未知；严格较早日期只是研究可用性约定，不冒充T−24或新盲测。",
              "各月／周／球队及多个指标相依，不声称显著、抽样区间、因果球员价值或投注收益。"
              "客让两球须净胜至少3球才让胜，评价不等于本轮推荐。",
              "原模型、当前9场与LLM设置均不更改；负结果不追加后季候选、改惩罚或删除异常比分。"
              "所有产物和缓存均在G盘，无新依赖、hash、冻结contract、baseline或gate。", ""]
    return "\n".join(lines)


def run(root, inspection):
    root, inspection = Path(root).resolve(), Path(inspection).resolve()
    started = datetime.now(timezone.utc)
    output = root/"outputs"/("跨赛季球队状态研究_"+started.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    _write(output/"研究安排.json", {"started_at_utc": started.isoformat(), "inspection": str(inspection),
        "source": "existing four-season English inspections; E0 only",
        "candidate_scales": SCALES, "selection_season": EARLY, "later_seasons": LATER,
        "alpha": .01, "half_life_days": 180, "refit_days": 28,
        "objective": "early common full-score NLL", "selection_count": 1, "later_selection_count": 0,
        "early_function_input": "only records before 2024-07-01; all source parsed for audit, no claim of no late file reads",
        "scope": "previously viewed date-level exploratory research; current forecasts unchanged"})
    all_records, audit = load_inspections(inspection)
    records = validate_records(all_records)
    _write(output/"输入资料说明.json", {"inspection_audit": audit, "epl_only_records": len(records),
        "not_used_other_division_records": len(all_records)-len(records),
        "season_counts": {s: sum(r.season == s for r in records) for s in ("2022-23", EARLY, *LATER)}})
    earlier = [r for r in records if r.match_date < date(2024, 7, 1)]
    early = run_season(earlier, EARLY, dict(SCALES), output/EARLY)
    _write(output/"早期尺度选择.json", select_scale(early))
    selection = json.loads((output/"早期尺度选择.json").read_text(encoding="utf-8"))
    fixed = selection["selected_scale"]
    later = {s: run_season(records, s, {"original": None, "seasonal": fixed}, output/s) for s in LATER}
    combined_rows = [row for detail in later.values() for row in detail["per_fixture"]]
    result = {"started_at_utc": started.isoformat(), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "output_directory": str(output), "selection": selection, "early": early, "later": later,
              "later_combined": summarize_season(combined_rows, ["original", "seasonal"]),
              "total_fits": early["fit_count"]+sum(x["fit_count"] for x in later.values()),
              "new_live_forecasts": 0, "later_selections": 0}
    _write(output/"跨赛季研究结果.json", result)
    (output/"跨赛季前推说明.md").write_text(_report(result), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.root, args.inspection))}, ensure_ascii=False))
