"""Fixed algebraic penalty-reference contrast; not a hyperparameter search.

PoissonRegressor divides the weighted data loss by total sample weight. Adding
E1 changes that denominator. If q = sum(E0 weights)/sum(all weights), alpha=.01*q
keeps the penalty coefficient .01 when the joint objective is expressed per E0
weight unit. This does NOT make the joint problem equivalent to the E0-only fit.
"""

import argparse
from copy import deepcopy
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from numbers import Real
from pathlib import Path
import time

from .batch_multiline import _paths, _read, _write
from .date_model import DateTeamGoalModel
from .english_tiers_experiment import (ARMS as OLD_ARMS, METRICS, load_inspections,
                                        score_forecast, summarize_rows)


NEW_ARM = "joint_epl_penalty_reference"
ARMS = (*OLD_ARMS, NEW_ARM)


def weight_share(history, cutoff, half_life_days=180):
    if type(cutoff) is not date:
        raise ValueError("Use an explicit calendar cutoff date")
    if (isinstance(half_life_days, bool) or not isinstance(half_life_days, Real)
            or not math.isfinite(half_life_days) or half_life_days <= 0):
        raise ValueError("Positive finite half-life required")
    weights = []
    for row in history:
        if row.match_date >= cutoff or row.goal_fit_exclusion is not None:
            continue
        if row.competition not in {"E0", "E1"}:
            raise ValueError("Unknown eligible training competition")
        weights.append((row.competition, math.exp(-math.log(2) *
                        (cutoff-row.match_date).days / half_life_days)))
    total = math.fsum(w for _, w in weights)
    epl = math.fsum(w for c, w in weights if c == "E0")
    if not total or not epl:
        raise ValueError("Eligible earlier EPL training weight required")
    return epl / total


def _validate_reference(records, reference):
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate source match identifier")
    settings = reference["parameters"]
    if settings["regularization"] != .01 or settings["half_life_days"] != 180:
        raise ValueError("Reference must use the stated .01/180 settings")
    interval = settings["refit_interval_days"]
    if type(interval) is not int or interval <= 0:
        raise ValueError("Positive reference update interval required")
    source = {r.match_id: r for r in records}
    rows = reference["per_fixture_rows"]
    ids = [r["match_id"] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or len(ids) != reference["target_count"]
            or ids != reference["target_ids"]):
        raise ValueError("Reference target identifiers/count disagree")
    cutoffs = {}
    for fit in reference["fits"]:
        cutoff = date.fromisoformat(fit["cutoff"])
        if fit["arm"] not in OLD_ARMS or fit["arm"] in cutoffs.setdefault(cutoff, set()):
            raise ValueError("Duplicate/unknown reference fit")
        cutoffs[cutoff].add(fit["arm"])
    if any(arms != set(OLD_ARMS) for arms in cutoffs.values()):
        raise ValueError("Both saved arms must represent each reference window")
    used_cutoffs = set()
    for row in rows:
        target = source.get(row["match_id"])
        if (target is None or target.competition != "E0" or target.season != "2025-26"
                or target.goal_fit_exclusion is not None
                or (row["match_date"], row["home"], row["away"], row["actual"]) !=
                   (target.match_date.isoformat(), target.home, target.away, list(target.full_time))):
            raise ValueError("Reference target identity/date/score differs from source")
        cutoff = date.fromisoformat(row["cutoff"])
        if cutoff not in cutoffs or not cutoff <= target.match_date < cutoff + timedelta(days=interval):
            raise ValueError("Reference target is outside its saved fit window")
        used_cutoffs.add(cutoff)
        if set(row["arms"]) != set(OLD_ARMS):
            raise ValueError("Reference must contain the original two arms only")
    if used_cutoffs != set(cutoffs):
        raise ValueError("Reference windows and target windows differ")
    return sorted(cutoffs)


def run_reference_comparison(records, reference, output_dir, model_factory=None):
    cutoffs = _validate_reference(records, reference)
    output = Path(output_dir)
    (output / "parameters").mkdir(parents=True, exist_ok=True)
    factory = model_factory or (lambda alpha: DateTeamGoalModel(alpha, 180))
    rows, fits = deepcopy(reference["per_fixture_rows"]), []
    started = time.perf_counter()
    for cutoff in cutoffs:
        history = [r for r in records if r.match_date < cutoff and r.goal_fit_exclusion is None]
        q = weight_share(history, cutoff)
        alpha = .01 * q
        model = factory(alpha).fit(history, cutoff)
        parameters = model.parameters()
        path = output / "parameters" / f"{NEW_ARM}_{cutoff.isoformat()}.json"
        saved_at = datetime.now(timezone.utc).isoformat()
        _write(path, {"arm": NEW_ARM, "fit_cutoff_date": cutoff.isoformat(),
                      "alpha": alpha, "epl_weight_share": q,
                      "saved_at": saved_at, "parameters": parameters})
        fits.append({"arm": NEW_ARM, "cutoff": cutoff.isoformat(), "alpha": alpha,
                     "epl_weight_share": q, "training_matches": len(history),
                     "training_max_date": max(r.match_date for r in history).isoformat(),
                     "parameter_path": str(path), "parameters_saved_at": saved_at})
        for row in rows:
            if row["cutoff"] != cutoff.isoformat():
                continue
            missing = sorted({row["home"], row["away"]} - model.teams)
            if missing:
                value = {"status": "missing", "reason": "unseen_team_no_estimated_prior",
                         "missing_teams": missing, "rates": None, "forecast": None,
                         "parameter_path": str(path)}
            else:
                rates = list(model.predict(row["home"], row["away"], "E0", date.fromisoformat(row["match_date"])))
                value = {"status": "predicted", "rates": rates, "parameter_path": str(path),
                         **score_forecast(rates, row["actual"])}
            row["arms"][NEW_ARM] = value
        print(json.dumps({"completed_window": cutoff.isoformat(), "alpha": alpha,
                          "epl_weight_share": q, "new_fits_so_far": len(fits)}), flush=True)
    paired = [r for r in rows if all(r["arms"][a]["status"] == "predicted" for a in ARMS)]
    groups = {"all_paired": paired,
              "promoted_paired": [r for r in paired if r["promoted_fixture"]],
              "other_paired": [r for r in paired if not r["promoted_fixture"]],
              "before_2026_paired": [r for r in paired if r["match_date"] < "2026-01-01"],
              "from_2026_paired": [r for r in paired if r["match_date"] >= "2026-01-01"]}
    comparisons = {}
    for name, group in groups.items():
        values = {a: summarize_rows([r["arms"][a] for r in group]) for a in ARMS}
        comparisons[name] = {"n": len(group), "arms": values,
            **{"reference_minus_" + old: {m: values[NEW_ARM][m]-values[old][m] if group else None
                                          for m in METRICS} for old in OLD_ARMS}}
    return {"target_count": len(rows), "target_ids": [r["match_id"] for r in rows],
            "per_fixture_rows": rows, "fits": fits, "fit_count": len(fits),
            "original_fit_count_reused": reference["fit_count"], "parameter_searches": 0,
            "elapsed_seconds": time.perf_counter()-started,
            "parameters": {"alpha_rule": ".01 * earlier EPL weight / earlier joint weight",
                           "half_life_days": 180, "refit_interval_days": reference["parameters"]["refit_interval_days"]},
            "paired_comparison": comparisons, "paired_target_ids": [r["match_id"] for r in paired],
            "arms": {a: {"target_count": len(rows),
                        "coverage": sum(r["arms"][a]["status"] == "predicted" for r in rows),
                        "metrics": summarize_rows([r["arms"][a] for r in rows]),
                        "missing": [{"match_id": r["match_id"], **r["arms"][a]} for r in rows if r["arms"][a]["status"] != "predicted"]}
                     for a in ARMS},
            "limitations": [
                "Selected after inspecting the previous joint comparison; exploratory sensitivity, not a fresh blind validation.",
                "This is a fixed penalty-reference convention, not a bug fix or an optimized alpha.",
                "Extra E1 losses, feature dimension and shared-club continuity assumptions still differ; not an isolated causal division-strength effect.",
                "Date-only historical availability assumed; publication times, transfer/coach effects and player effects not reconstructed.",
                "All original probabilities and current forecasts retained; no automatic adoption, significance or profitability claim."]}


def render_report(result):
    lines = ["# 跨级模型：固定英超惩罚参照的一次对照", "",
             "不搜索参数、不重训原两臂，只新增一条解析换算研究线；当前赛前预测不变。", "",
             "q为每窗严格较早训练资料中英超时间权重占联合权重的比例。alpha=.01×q，使联合目标按英超权重单位表达时L2系数仍为.01。", "",
             "这不是英超单独拟合的等价变换，也不证明之前的实现错误。英冠损失、参数数量及跨级连续性假设仍不同。", "",
             f"新增拟合{result['fit_count']}次，复用旧拟合{result['original_fit_count_reused']}份；参数搜索0次，新增拟合与评分{result['elapsed_seconds']:.2f}秒。", "",
             "| 共同样本指标 | 英超单独 | 原联合 | 惩罚参照联合 |", "|---|---:|---:|---:|"]
    common = result["paired_comparison"]["all_paired"]
    if common["n"]:
        for m in METRICS:
            values = " | ".join(f"{common['arms'][a][m]:.6f}" for a in ARMS)
            lines.append(f"| {m} | {values} |")
    lines += ["", f"目标{result['target_count']}场，共同{common['n']}场；覆盖分别为" +
              "／".join(str(result["arms"][a]["coverage"]) for a in ARMS) + "。覆盖增加不等于共同样本概率质量提高。", "",
              "| 分组 | n | 新臂−英超比分NLL | 新臂−原联合比分NLL |", "|---|---:|---:|---:|"]
    for name, group in result["paired_comparison"].items():
        if group["n"]:
            lines.append(f"| {name} | {group['n']} | {group['reference_minus_epl_only']['full_score_nll']:+.6f} | {group['reference_minus_joint']['full_score_nll']:+.6f} |")
    lines += ["", "完整比分NLL为解析Poisson；其余概率从同一比分分布计算。损失越低越好，top1越高越好；最高概率精确并列按1/k计分。", "",
              "以下为全部解释边界：", ""] + ["- " + x for x in result["limitations"]]
    return "\n".join(lines) + "\n"


def run_experiment(root, inspection, reference_path):
    root, inspection, reference_path = _paths(root, inspection, reference_path)
    records, audit = load_inspections(inspection)
    reference = _read(reference_path)
    _validate_reference(records, reference)
    started = datetime.now(timezone.utc)
    output = root / "outputs" / started.strftime("跨级惩罚参照研究_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "研究安排.json", {"started_at": started.isoformat(),
            "reference_path": str(reference_path), "inspection_path": str(inspection),
            "new_arm": NEW_ARM, "alpha_rule": ".01 * q; q derived only from earlier training weights",
            "parameter_searches": 0, "main_metric": "analytic full-score NLL",
            "exploratory_not_blind": True, "original_forecasts_unchanged": True})
    result = run_reference_comparison(records, reference, output)
    result.update(started_at=started.isoformat(), finished_at=datetime.now(timezone.utc).isoformat(),
                  reference_path=str(reference_path), input_audit=audit, output_directory=str(output))
    _write(output / "研究结果及逐场概率.json", result)
    with (output / "逐场对照.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        names = ["match_id", "match_date", "home", "away", "actual", "cutoff", "promoted_fixture",
                 "arm", "status", "rates", "forecast", "parameter_path", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in result["per_fixture_rows"]:
            for arm in ARMS:
                flat = {**{k: v for k, v in row.items() if k != "arms"}, "arm": arm, **row["arms"][arm]}
                for key in ("actual", "rates", "forecast"):
                    flat[key] = json.dumps(flat.get(key), ensure_ascii=False, allow_nan=False)
                writer.writerow(flat)
    (output / "惩罚参照对照说明.md").write_text(render_report(result), encoding="utf-8")
    return result, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(argv)
    result, output = run_experiment(args.root, args.inspection, args.reference)
    print(json.dumps({"output_directory": str(output), "fit_count": result["fit_count"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "all_paired": result["paired_comparison"]["all_paired"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
