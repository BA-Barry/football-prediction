"""Player-only exposure-scaled L2 versus fixed L2 on identical historical paths.

This bounded, previously-viewed-season study changes neither the production
model nor unknown-player handling. Base-ridge selection and scaling-policy
effects are separated with a same-cohort two-by-two comparison.
"""

import argparse
from datetime import date, datetime, timezone
import gc
import json
from math import fsum, isfinite
from pathlib import Path

from .forward_selection_experiment import _mapped, reuse_saved
from .regularization_comparison import _identities, _index, select_regularization
from .regularization_experiment import PLAYER_RIDGES, SEASONS, fit_saved, load_paths, selection_split
from .rolling_player_experiment import rolling_windows
from .scaling_comparison import compare_scaling
from .statsbomb_data import write_json
from .window_clock_experiment import append_score, empty_score


PARENT_RESULT = "多周前推收缩结果.json"
RESULT_FILE = "训练量缩放结果.json"
REPORT_FILE = "训练量缩放说明.md"


def find_parent(root, parent=None):
    if parent is not None:
        selected = Path(parent).resolve()
        if not (selected / PARENT_RESULT).is_file():
            raise ValueError("A completed weekly-forward experiment is required")
        return selected
    return next(p for p in sorted((Path(root) / "outputs").glob("多周前推收缩验证_*"), reverse=True)
                if (p / PARENT_RESULT).is_file())


def training_exposure(history):
    """Count each physical match-clock interval once, not once per player/side."""
    if not history or len({m["match_id"] for m in history}) != len(history):
        raise ValueError("Distinct nonempty history required for exposure")
    durations = [s["duration_seconds"] for m in history for s in m["stints"]]
    if not durations or any(isinstance(v, bool) or not isinstance(v, (int, float))
                            or not isfinite(v) or v <= 0 for v in durations):
        raise ValueError("Positive finite stint durations required")
    total = fsum(durations)
    if not isfinite(total):
        raise ValueError("Finite training exposure required")
    return total


def effective_ridge(base, exposure, reference):
    values = (base, exposure, reference)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not isfinite(v) or v <= 0 for v in values):
        raise ValueError("Positive finite base ridge and exposures required")
    value = base * (exposure / reference)
    if not isfinite(value) or value <= 0:
        raise ValueError("Scaled player ridge must remain finite and positive")
    return value


def validation_windows(paths, config):
    _, declared_ids, first_late = selection_split(paths, config["train_before"],
        config["selection_before"], config["target_from"])
    windows = []
    for cutoff, history, target in rolling_windows(paths[config["clocks"][0]], from_date=config["train_before"]):
        if cutoff >= config["selection_before"]:
            break
        target = [m for m in target if date.fromisoformat(m["match_date"]) < config["selection_before"]]
        if target:
            windows.append((cutoff, history, target))
    seen = [m["match_id"] for _, _, target in windows for m in target]
    if not windows or len(seen) != len(set(seen)) or set(seen) != set(declared_ids):
        raise ValueError("Weekly validation must cover declared earlier targets exactly once")
    if windows[-1][0] >= first_late:
        raise ValueError("Validation cannot reach the first later fit cutoff")
    return windows


def _counts(scores):
    return {"requested": scores["requested_matches"], "supported": len(scores["predictions"]),
            "skipped": len(scores["skipped"])}


def common_policy_selections(fixed, scaled):
    """Both policies select on one all-clock, all-candidate validation cohort."""
    if set(fixed) != set(scaled) or not fixed:
        raise ValueError("Fixed and scaled validation need the same declared clocks")
    all_indices, identity_indices = [], {c: {} for c in fixed}
    for label, policy in (("fixed", fixed), ("scaled", scaled)):
        # Existing selection checks numeric keys, duplicates and target/path identity.
        select_regularization(policy)
        for clock, candidates in policy.items():
            if set(candidates) != set(fixed[clock]):
                raise ValueError("Both policies require the same candidate base-ridge keys")
            all_indices.extend({p["match_id"] for p in score["predictions"]} for score in candidates.values())
            identity_indices[clock].update({f"{label}/{base}": _index(score) for base, score in candidates.items()})
    # Check every observed cross-policy overlap, including records outside the
    # eventual common cohort, and each clock's actual path partition.
    _identities(identity_indices)
    common = sorted(set.intersection(*all_indices))
    if not common:
        raise ValueError("No common validation matches across policies")
    def restricted(policy):
        return {c: {r: {"predictions": [p for p in score["predictions"] if p["match_id"] in common]}
                    for r, score in candidates.items()} for c, candidates in policy.items()}
    return {"fixed": select_regularization(restricted(fixed)),
            "scaled": select_regularization(restricted(scaled)), "common_match_ids": common}


def scaled_validation(paths, config, parent_season, destination, progress):
    destination.mkdir()
    models = destination / "models"
    models.mkdir()
    windows = validation_windows(paths, config)
    first_cutoff, first_history, _ = windows[0]
    reference_ids = [m["match_id"] for m in first_history]
    references = {c: training_exposure(h) for c, h in _mapped(paths, reference_ids).items()}
    reference_record = {"cutoff": first_cutoff.isoformat(), "history_match_ids": reference_ids,
                        "history_matches": len(reference_ids), "exposure_seconds_by_clock": references}
    write_json(destination / "暴露参考.json", reference_record)
    totals = {c: {f"{r:g}": empty_score() for r in PLAYER_RIDGES} for c in config["clocks"]}
    records = []
    for cutoff, history, target in windows:
        histories = _mapped(paths, [m["match_id"] for m in history])
        targets = _mapped(paths, [m["match_id"] for m in target])
        record = {"cutoff": cutoff.isoformat(), "history_matches": len(history), "target_matches": len(target), "clocks": {}}
        print(f"Scaled validation {cutoff}: history={len(history)} target={len(target)}", flush=True)
        for clock in config["clocks"]:
            exposure = training_exposure(histories[clock])
            record["clocks"][clock] = {}
            for base in PLAYER_RIDGES:
                key = f"{base:g}"
                ridge = effective_ridge(base, exposure, references[clock])
                stem = f"{cutoff}_{clock}_ridge{key}"
                cache = parent_season / "validation" / "models" / f"{stem}.json"
                if ridge == base and cache.is_file():
                    score, diag = reuse_saved(cache, histories[clock], targets[clock], cutoff, ridge, True,
                        models / f"{stem}.json", progress)
                else:
                    score, diag = fit_saved(histories[clock], targets[clock], cutoff, ridge, True,
                        models / f"{stem}.json", progress)
                record["clocks"][clock][key] = {**diag, "base_player_ridge": base,
                    "effective_player_ridge": ridge, "history_exposure_seconds": exposure,
                    "reference_exposure_seconds": references[clock], "policy": "player_exposure_scaled"}
                append_score(totals[clock][key], score)
                write_json(destination / f"{stem}_predictions.json", score)
        records.append(record)
        write_json(destination / "逐周缩放验证.json", records)
    fixed = {c: {f"{r:g}": json.loads((parent_season / "validation" / f"validation_{c}_ridge{r:g}_predictions.json").read_text(encoding="utf-8"))
                  for r in PLAYER_RIDGES} for c in config["clocks"]}
    for c in totals:
        for r, score in totals[c].items():
            write_json(destination / f"validation_{c}_ridge{r}_predictions.json", score)
    selection = common_policy_selections(fixed, totals)
    selection.update({"reference": reference_record, "active_weeks": len(windows),
        "requested_matches": sum(len(t) for _, _, t in windows),
        "coverage": {c: {r: _counts(s) for r, s in values.items()} for c, values in totals.items()}})
    write_json(destination / "固定与缩放较早选择.json", selection)
    return selection


def run_season(name, parent, parent_result, output, progress):
    config, prior = SEASONS[name], parent_result["seasons"][name]
    source = Path(prior["source_directory"])
    paths = load_paths(source, name)
    # Compare ordinary source identity/splits rather than adding hashes or contracts.
    prior_inputs = json.loads((parent / name / "分段与来源.json").read_text(encoding="utf-8"))
    train_ids, val_ids, first_late = selection_split(paths, config["train_before"], config["selection_before"], config["target_from"])
    if (Path(prior_inputs["source_directory"]).resolve() != source.resolve()
            or prior_inputs["source_file"] != config["source_file"]
            or prior_inputs["training_match_ids"] != train_ids or prior_inputs["validation_match_ids"] != val_ids
            or prior_inputs["first_weekly_cutoff"] != first_late.isoformat()
            or prior["source_matches"] != len(paths[config["clocks"][0]])):
        raise ValueError("Scaling must retain parent source cohort and chronological splits")
    destination = output / name
    destination.mkdir()
    models = destination / "models"
    models.mkdir()
    write_json(destination / "分段与来源.json", prior_inputs)
    before_counts = dict(progress)
    selected = scaled_validation(paths, config, parent / name, destination / "validation", progress)
    old_base = selected["fixed"]["selected_player_ridge"]
    new_base = selected["scaled"]["selected_player_ridge"]
    if old_base != prior["selection"]["selected_player_ridge"]:
        raise ValueError("Fixed-policy selection on common cohort must reproduce the parent")
    references = selected["reference"]["exposure_seconds_by_clock"]
    print(f"{name}: fixed base={old_base:g}; exposure-scaled base={new_base:g}", flush=True)
    tracks = ("team", "fixed100", "fixed_old", "fixed_new", "scaled_old", "scaled_new")
    totals = {c: {k: empty_score() for k in tracks} for c in config["clocks"]}
    weeks = []
    for cutoff, history, target in rolling_windows(paths[config["clocks"][0]], from_date=config["target_from"]):
        histories = _mapped(paths, [m["match_id"] for m in history])
        targets = _mapped(paths, [m["match_id"] for m in target])
        record = {"cutoff": cutoff.isoformat(), "history_matches": len(history), "target_matches": len(target), "clocks": {}}
        print(f"{name} later scaling {cutoff}: history={len(history)} target={len(target)}", flush=True)
        for clock in config["clocks"]:
            exposure = training_exposure(histories[clock])
            scores, diagnostics, computed = {}, {}, {}
            specs = (("team", False, 100.0, "fixed"), ("fixed100", True, 100.0, "fixed"),
                     ("fixed_old", True, old_base, "fixed"), ("fixed_new", True, new_base, "fixed"),
                     ("scaled_old", True, old_base, "scaled"), ("scaled_new", True, new_base, "scaled"))
            for track, players, base, policy in specs:
                ridge = base if policy == "fixed" else effective_ridge(base, exposure, references[clock])
                key = (players, ridge)
                if key in computed:
                    original = computed[key]
                    score, diag = scores[original], {**diagnostics[original], "alias_of_track": original}
                else:
                    cache_kind = "team" if not players else "fixed100" if ridge == 100 else "selected" if ridge == old_base else None
                    cache = parent / name / "models" / f"{cutoff}_{clock}_{cache_kind}.json" if cache_kind else None
                    model_path = models / f"{cutoff}_{clock}_{track}.json"
                    if cache is not None and cache.is_file():
                        score, diag = reuse_saved(cache, histories[clock], targets[clock], cutoff, ridge, players, model_path, progress)
                    else:
                        score, diag = fit_saved(histories[clock], targets[clock], cutoff, ridge, players, model_path, progress)
                    computed[key] = track
                scores[track] = score
                diagnostics[track] = {**diag, "base_player_ridge": base, "effective_player_ridge": ridge,
                    "history_exposure_seconds": exposure, "reference_exposure_seconds": references[clock],
                    "policy": policy, "player_penalty_applied": players}
                append_score(totals[clock][track], score)
            record["clocks"][clock] = diagnostics
        weeks.append(record)
        write_json(destination / "逐周缩放与覆盖.json", weeks)
    comparison = compare_scaling(totals)
    result = {"source_directory": str(source), "source_matches": len(paths[config["clocks"][0]]),
        "selection": selected, "comparison": comparison, "active_later_weeks": len(weeks),
        "coverage": {c: {k: _counts(s) for k, s in values.items()} for c, values in totals.items()},
        "last_week_diagnostics": weeks[-1],
        **{k: progress[k] - before_counts[k] for k in progress}}
    for clock, values in totals.items():
        for track, scores in values.items():
            write_json(destination / f"later_{clock}_{track}_predictions.json", scores)
    write_json(destination / RESULT_FILE, result)
    del paths, totals
    gc.collect()
    return result


def report(result):
    def fmt(value):
        return "—" if value is None else f"{value:+.6f}"
    lines = ["# 球员正则随训练暴露量缩放：同场对照", "",
        f"新增实际拟合{result['actual_model_fits']}次，复用已有参数重新评分{result['reused_model_evaluations']}次。"
        "别名轨道不重复训练、计数或保存同一份参数。正式赛前模型、市场双轨与未知球员整场跳过规则不变。", "",
        "## 研究规则", "",
        "固定政策 λ=基础值；缩放政策 λ=基础值×当前训练暴露秒数÷首验证周训练暴露秒数。"
        "暴露是比赛真实时段秒数之和，每段计一次，不是球员秒数、两队展开行数、时段数或未来进球。"
        "本研究固定45+45窗口下约为5400×训练场数。首验证周参考完全来自当时更早历史，不从全季总量定标。", "",
        "只缩放球员L2，球队／比分人数状态正则仍1，所以不是整个目标改为平均损失，也不保证每名球员收缩恒定。"
        "训练身份、参考阵容与球队／球员联合解释仍会变化。候选基础值沿用10／100／1000／10000，不扩网格。", "",
        "每个较早验证周一只用更早比赛；验证目标为12月1日至选择截止日前。2017/18选择必须早于2018-01-29，"
        "不使用1月29日至31日结果调参。全部候选、两政策、全部时钟在同一共同比赛集合选各政策的基础值；"
        "按比赛等权汇总，再对时钟等权，不按周均分等权，也不选择哪套时钟是真的。", "",
        "| 来源 | 首验证周一 | 参考训练场数 | 验证请求／共同 | 固定基础值 | 缩放基础值 |",
        "|---|---|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        s = row["selection"]
        lines.append(f"| {name} | {s['reference']['cutoff']} | {s['reference']['history_matches']} | "
            f"{s['requested_matches']}／{len(s['common_match_ids'])} | {s['fixed']['selected_player_ridge']:g} | {s['scaled']['selected_player_ridge']:g} |")
    lines += ["", "| 来源 | 政策 | 基础值 | 共同较早验证损失 |", "|---|---|---:|---:|"]
    for name, row in result["seasons"].items():
        for policy in ("fixed", "scaled"):
            for metric in row["selection"][policy]["candidate_metrics"].values():
                lines.append(f"| {name} | {policy} | {metric['player_ridge']:g} | {metric['mean_clock_match_loss']:.6f} |")
    lines += ["", "## 较晚同场2×2：分开缩放政策与基础值选择", "",
        "固定／缩放 × 原固定政策所选基础值／新缩放政策所选基础值均评分；只训练这两个基础值需要的后段模型，"
        "不根据后段再选政策。旧／新基础值相同时相关轨道是别名，不是独立验证。损失越低越好，负差较好。", "",
        "| 来源 | 时钟 | 共同较晚场数 | 固定旧值−球队 | 缩放新值−球队 | 缩放新值−固定旧值 | 同旧基础值：缩放−固定 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        compare = row["comparison"]
        for clock in SEASONS[name]["clocks"]:
            v = compare["per_clock"][clock]
            lines.append(f"| {name} | {clock} | {len(compare['common_match_ids'])} | {fmt(v['fixed_old_minus_team'])} | "
                f"{fmt(v['scaled_new_minus_team'])} | {fmt(v['scaled_new_minus_fixed_old'])} | {fmt(v['scaled_old_minus_fixed_old'])} |")
    lines += ["", "各来源时钟均值："]
    for name, row in result["seasons"].items():
        values = row["comparison"]["clock_averaged"]
        lines.append(f"{name}：缩放新值−固定旧值{fmt(values['scaled_new_minus_fixed_old'])}；"
            f"缩放新值−仅球队{fmt(values['scaled_new_minus_team'])}；缩放新值−原100 {fmt(values['scaled_new_minus_fixed100'])}。")
    unchanged = all(row["selection"]["fixed"]["selected_player_ridge"] == row["selection"]["scaled"]["selected_player_ridge"]
                    for row in result["seasons"].values())
    if unchanged:
        lines += ["", "本次两来源的基础值都未改变；因此上述总差来自同基础值下的缩放规则，不是换基础值。"
            "两条基数对照为别名，交互为0，不额外增加独立样本。"]
    if all(row["comparison"]["clock_averaged"]["scaled_new_minus_team"] is not None
           and row["comparison"]["clock_averaged"]["scaled_new_minus_team"] > 0 for row in result["seasons"].values()):
        lines += ["", "本次两来源的缩放方案仍都略／明显差于仅球队模型。相对原球员方案的改善不等于已超过更简单的模型，"
            "更不是赛前胜率提升。暂不将缩放规则或球员项接入正式赛前模型。"]
    lines += ["", "总差＝固定政策下换基础值的差＋新基础值下缩放政策的差；也等于旧基础值下政策差＋缩放政策下换基础值的差。"
        "两条分解、交互、所有同场单场损失及均值均在结果JSON中保留，不能只报总差就归因于缩放。", "",
        "| 来源 | 时钟 | 最后周暴露／参考 | 旧基础值 | 新基础值 | 缩放旧实际λ | 缩放新实际λ |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        for clock in SEASONS[name]["clocks"]:
            d = row["last_week_diagnostics"]["clocks"][clock]
            a, b = d["scaled_old"], d["scaled_new"]
            lines.append(f"| {name} | {clock} | {a['history_exposure_seconds']/a['reference_exposure_seconds']:.6f} | "
                f"{a['base_player_ridge']:g} | {b['base_player_ridge']:g} | {a['effective_player_ridge']:.6f} | {b['effective_player_ridge']:.6f} |")
    lines += ["", "| 来源 | 时钟 | 后段请求 | 缩放球员支持 | 跳过 |", "|---|---|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        for clock in SEASONS[name]["clocks"]:
            cov = row["coverage"][clock]["scaled_new"]
            lines.append(f"| {name} | {clock} | {cov['requested']} | {cov['supported']} | {cov['skipped']} |")
    lines += ["", "| 来源 | 缩放新值−固定旧值：中位数 | 描述性SD | 最小 | 最大 |", "|---|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        distribution = row["comparison"]["clock_averaged"]["contrast_distributions"]["scaled_new_minus_fixed_old"]
        if distribution:
            lines.append(f"| {name} | {fmt(distribution['median'])} | {fmt(distribution['sample_standard_deviation'])} | "
                f"{fmt(distribution['min'])} | {fmt(distribution['max'])} |")
    lines += ["", "分布先按同场平均时钟，再按比赛描述；SD不是标准误或置信区间，不假定比赛独立。"]
    lines += ["", "## 解释边界与执行记录", "",
        "两季后段此前已经查看，本轮仍是探索性敏感性对照，不是新盲测或因果试验。支持集非随机；未估计时间／球队相关抽样区间，"
        "不宣称显著或稳定改进。时钟、周数和拟合次数都不增加独立比赛数，分布中的胜／负场次也不是投注命中率。", "",
        "目标仍是固定45+45分钟、以真实在场／换人／比分／人数为条件的事后强度，不是完整比赛赛前1X2；"
        "不生成新xG、不添加新球员先验、伤停猜测或战意系数。未知球员不因λ很大就填零。"
        "原源排除、补时状态传递和近似时钟冲突处理保留，不删除不利进球。", "",
        "所有代码、模型、评分、缓存和源码包在项目目录。只读既有本地来源，无新依赖或下载；数据和模型不放入软件源码ZIP，原许可继续适用。", ""]
    return "\n".join(lines)


def run(root, parent=None):
    root = Path(root).resolve()
    parent = find_parent(root, parent)
    prior = json.loads((parent / PARENT_RESULT).read_text(encoding="utf-8"))
    if set(prior["seasons"]) != set(SEASONS):
        raise ValueError("Both declared source seasons required")
    output = root / "outputs" / ("训练量缩放对照_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    write_json(output / "实验设置.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_directory": str(parent), "player_base_ridges": list(PLAYER_RIDGES),
        "scaled_formula": "base_player_ridge * training_exposure_seconds / first_validation_week_training_exposure_seconds",
        "exposure_unit": "each physical match stint duration once, NOT model rows, player-seconds, or outcomes",
        "only_player_ridge_scaled": True, "team_ridge": 1, "context_ridge": 1,
        "reference_chosen_from": "strict earlier history before first weekly validation cutoff, separate each source/clock",
        "selection": "earlier pooled equal-match loss, clock average, common all-candidate/all-policy cohort; tie1e-12 larger base",
        "later_design": "fixed/scaled by fixed-selected/scaled-selected base 2x2, plus team and fixed100, all-common match cohort",
        "cache": "reuse only identical history IDs, cutoff, source, response and effective settings; aliases counted once",
        "exploratory_previously_viewed_seasons": True, "formal_prematch_probabilities_changed": False,
        "unknown_policy_changed": False, "market_fusion_added": False})
    progress = {"actual_model_fits": 0, "reused_model_evaluations": 0}
    try:
        result = {"parent_directory": str(parent), "seasons": {}}
        for name in SEASONS:
            result["seasons"][name] = run_season(name, parent, prior, output, progress)
        result.update(progress)
        write_json(output / RESULT_FILE, result)
        (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
        print(f"Exposure scaling complete: {output}; counts={progress}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {**progress, "error": f"{type(error).__name__}: {error}", "status": "incomplete"})
        raise


def refresh_summaries(output):
    """Recompute descriptive summaries from saved scores; no fitting/scoring."""
    output = Path(output)
    result = json.loads((output / RESULT_FILE).read_text(encoding="utf-8"))
    for name, row in result["seasons"].items():
        scores = {c: {track: json.loads((output / name / f"later_{c}_{track}_predictions.json").read_text(encoding="utf-8"))
                      for track in ("team", "fixed100", "fixed_old", "fixed_new", "scaled_old", "scaled_new")}
                  for c in SEASONS[name]["clocks"]}
        row["comparison"] = compare_scaling(scores)
        write_json(output / name / RESULT_FILE, row)
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    return {"output": str(output), "summary_only": True, "new_model_fits": 0, "new_predictions": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    parser.add_argument("--refresh-summaries", type=Path)
    arguments = parser.parse_args()
    if arguments.refresh_summaries:
        print(json.dumps(refresh_summaries(arguments.refresh_summaries), ensure_ascii=False))
    elif arguments.report_only:
        result = json.loads((arguments.report_only / RESULT_FILE).read_text(encoding="utf-8"))
        (arguments.report_only / REPORT_FILE).write_text(report(result), encoding="utf-8")
        print(json.dumps({"report_only": True, "new_model_fits": 0, "output": str(arguments.report_only)}, ensure_ascii=False))
    else:
        run(arguments.root, arguments.parent)
