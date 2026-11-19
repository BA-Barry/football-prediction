"""Weekly forward validation of the existing player-ridge candidate grid.

Previously viewed seasons remain exploratory. Clock alternatives are repeated
descriptions, not independent observations or probabilities over true clocks.
"""

import argparse
from datetime import date, datetime, timezone
import gc
import json
from math import fsum
from pathlib import Path

from .full_season_experiment import paired_comparison, score_matches
from .joint_stint_model import JointStintIntensityModel
from .regularization_comparison import compare_regularization, select_regularization
from .regularization_experiment import (
    FIXED_RIDGE, PLAYER_RIDGES, SEASONS, coefficient_summary, fit_saved,
    load_paths, selection_split,
)
from .rolling_player_experiment import rolling_windows
from .statsbomb_data import write_json
from .window_clock_experiment import append_score, empty_score


RESULT_FILE = "多周前推收缩结果.json"
REPORT_FILE = "多周前推收缩说明.md"
PARENT_RESULT = "球员收缩验证结果.json"


def find_parent(root, parent=None):
    if parent is not None:
        candidate = Path(parent).resolve()
        if not (candidate / PARENT_RESULT).is_file():
            raise ValueError("Parent must contain a completed regularization result")
        return candidate
    candidates = sorted((Path(root) / "outputs").glob("球员收缩验证_*"), reverse=True)
    return next(p for p in candidates if (p / PARENT_RESULT).is_file())


def _mapped(paths, ids):
    indices = {clock: {m["match_id"]: m for m in values} for clock, values in paths.items()}
    return {clock: [index[mid] for mid in ids] for clock, index in indices.items()}


def _coverage(scores):
    return {"requested": scores["requested_matches"], "supported": len(scores["predictions"]),
            "skipped": len(scores["skipped"])}


def weekly_validation(paths, config, destination, progress):
    """Select only from [train_before, selection_before) weekly predictions."""
    _, declared_ids, first_late_cutoff = selection_split(
        paths, config["train_before"], config["selection_before"], config["target_from"])
    destination = Path(destination)
    destination.mkdir()
    models = destination / "models"
    models.mkdir()
    before_fits = progress["actual_model_fits"]
    totals = {c: {f"{ridge:g}": empty_score() for ridge in PLAYER_RIDGES} for c in config["clocks"]}
    weeks, scored_ids = [], []
    first_path = paths[config["clocks"][0]]
    for cutoff, history, target in rolling_windows(first_path, from_date=config["train_before"]):
        if cutoff >= config["selection_before"]:
            break
        target = [m for m in target if date.fromisoformat(m["match_date"]) < config["selection_before"]]
        if not target:
            continue
        if cutoff >= first_late_cutoff:
            raise ValueError("Validation must finish before the first later weekly fit cutoff")
        history_ids = [m["match_id"] for m in history]
        target_ids = [m["match_id"] for m in target]
        histories, targets = _mapped(paths, history_ids), _mapped(paths, target_ids)
        record = {"cutoff": cutoff.isoformat(), "history_match_ids": history_ids,
                  "target_match_ids": target_ids, "clocks": {}}
        print(f"Weekly validation {cutoff}: train={len(history_ids)} target={len(target_ids)}", flush=True)
        for clock in config["clocks"]:
            if any(date.fromisoformat(m["match_date"]) >= cutoff for m in histories[clock]):
                raise ValueError("Every clock history must be strictly before its weekly cutoff")
            record["clocks"][clock] = {}
            for ridge in PLAYER_RIDGES:
                key = f"{ridge:g}"
                stem = f"{cutoff}_{clock}_ridge{key}"
                score, diagnostic = fit_saved(histories[clock], targets[clock], cutoff, ridge, True,
                    models / f"{stem}.json", progress)
                append_score(totals[clock][key], score)
                record["clocks"][clock][key] = diagnostic
                write_json(destination / f"{stem}_predictions.json", score)
        scored_ids.extend(target_ids)
        weeks.append(record)
        write_json(destination / "逐周验证拟合与覆盖.json", weeks)
    if len(scored_ids) != len(set(scored_ids)) or set(scored_ids) != set(declared_ids):
        raise ValueError("Weekly validation must score each declared earlier validation match once")
    selection = select_regularization(totals)
    selection.update({"validation_requested_matches": len(scored_ids), "active_weeks": len(weeks),
        "selection_before": config["selection_before"].isoformat(),
        "first_weekly_cutoff": first_late_cutoff.isoformat(),
        "validation_first_fit_cutoff": weeks[0]["cutoff"], "validation_last_fit_cutoff": weeks[-1]["cutoff"],
        "actual_model_fits": progress["actual_model_fits"] - before_fits,
        "coverage": {c: {r: _coverage(s) for r, s in values.items()} for c, values in totals.items()}})
    for clock, values in totals.items():
        for ridge, score in values.items():
            write_json(destination / f"validation_{clock}_ridge{ridge}_predictions.json", score)
    write_json(destination / "多周较早验证选择.json", selection)
    return selection


def validate_cached_model(model, history, cutoff, ridge, players):
    """Ordinary input/settings checks before reusing an already fitted model."""
    if type(cutoff) is not date or not history:
        raise ValueError("Cached model requires a calendar cutoff and nonempty history")
    if any(date.fromisoformat(m["match_date"]) >= cutoff for m in history):
        raise ValueError("Cached model history must be strictly earlier than cutoff")
    ordered = sorted(history, key=lambda m: (m["match_date"], m["match_id"]))
    ids = [m["match_id"] for m in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("Distinct history match IDs required for cached parameters")
    parameters = model.parameters()
    expected = {"before_date": cutoff.isoformat(), "include_players": players,
        "response_kind": "goals", "team_ridge": 1.0, "context_ridge": 1.0, "player_ridge": ridge,
        "reference_match_ids": ids, "n_matches": len(history),
        "n_stints": sum(len(m["stints"]) for m in history),
        "training_min_date": ordered[0]["match_date"], "training_max_date": ordered[-1]["match_date"]}
    for field, value in expected.items():
        if parameters.get(field) != value:
            raise ValueError(f"Cached model input/settings mismatch: {field}")


def reuse_saved(source, history, target, cutoff, ridge, players, destination, progress):
    model = JointStintIntensityModel.load(source)
    validate_cached_model(model, history, cutoff, ridge, players)
    score = score_matches(model, target)
    model.save(destination)
    restored = JointStintIntensityModel.load(destination)
    if restored.parameters() != model.parameters():
        raise ValueError("Reused saved parameters do not roundtrip")
    if score["predictions"]:
        mid = score["predictions"][0]["match_id"]
        witness = next(m for m in target if m["match_id"] == mid)
        if restored.predict_match(witness) != model.predict_match(witness):
            raise ValueError("Reused witness prediction does not roundtrip")
    progress["reused_model_evaluations"] += 1
    return score, {"player_ridge": ridge, "include_players": players,
        "optimizer": model.optimizer_diagnostics, "coefficient_summary": coefficient_summary(model, history),
        "reused_parent_model": str(source), "saved_parameter_and_witness_roundtrip": True,
        "supported": len(score["predictions"]), "requested": len(target), "skipped": score["skipped"]}


def compare_selected_tracks(previous, weekly):
    """Direct old/new selected comparison on all-clock common actual paths."""
    if set(previous) != set(weekly):
        raise ValueError("Previous and weekly selected scores require the same clocks")
    checked = compare_regularization({c: {"team": previous[c], "fixed100": previous[c], "selected": weekly[c]}
                                      for c in previous})
    common = checked["common_match_ids"]
    comparisons = {}
    for clock in checked["clocks"]:
        if not common:
            comparisons[clock] = {"previous_static_selected_on_common": None, "weekly_selected_on_common": None,
                                  "weekly_minus_previous_static_loss": None, "paired_matches": []}
            continue
        a = {p["match_id"]: p for p in previous[clock]["predictions"]}
        b = {p["match_id"]: p for p in weekly[clock]["predictions"]}
        paired = paired_comparison({"predictions": [a[mid] for mid in common]},
                                   {"predictions": [b[mid] for mid in common]})
        comparisons[clock] = {"previous_static_selected_on_common": paired["team_on_common"],
            "weekly_selected_on_common": paired["player_on_common"],
            "weekly_minus_previous_static_loss": paired["mean_player_minus_team_loss"],
            "paired_matches": [{"match_id": r["match_id"], "date": r["date"],
                "previous_static_selected_loss": r["team_loss"], "weekly_selected_loss": r["player_loss"],
                "weekly_minus_previous_static_loss": r["player_minus_team_loss"]} for r in paired["paired_matches"]]}
    return {"common_match_ids": common, "per_clock": comparisons,
        "mean_clock_weekly_minus_previous_static_loss": fsum(
            r["weekly_minus_previous_static_loss"] for r in comparisons.values()) / len(comparisons) if common else None,
        "lower_is_better": True, "sampling_uncertainty": "not estimated; repeated clock alternatives are not independent samples"}


def run_season(name, parent, parent_result, output, progress):
    config, old = SEASONS[name], parent_result["seasons"][name]
    source = Path(old["source_directory"]).resolve()
    destination = output / name
    destination.mkdir()
    models = destination / "models"
    models.mkdir()
    initial = dict(progress)
    paths = load_paths(source, name)
    train_ids, val_ids, first_cutoff = selection_split(paths, config["train_before"],
        config["selection_before"], config["target_from"])
    prior_inputs = json.loads((parent / name / "分段与来源.json").read_text(encoding="utf-8"))
    if (Path(prior_inputs["source_directory"]).resolve() != source
            or prior_inputs["source_file"] != config["source_file"]
            or prior_inputs["training_match_ids"] != train_ids or prior_inputs["validation_match_ids"] != val_ids
            or prior_inputs["first_weekly_cutoff"] != first_cutoff.isoformat()
            or old["source_matches"] != len(paths[config["clocks"][0]])):
        raise ValueError("Forward selection must retain the parent's exact source cohort and date splits")
    write_json(destination / "分段与来源.json", {**prior_inputs, "parent_directory": str(parent)})
    selection = weekly_validation(paths, config, destination / "validation", progress)
    selected = selection["selected_player_ridge"]
    prior_ridge = old["selection"]["selected_player_ridge"]
    write_json(destination / "多周较早验证选择.json", selection)
    print(f"{name}: weekly-selected={selected:g}, previous-static-selected={prior_ridge:g}", flush=True)
    totals = {c: {kind: empty_score() for kind in ("team", "fixed100", "selected")} for c in config["clocks"]}
    weeks = []
    for cutoff, history, target in rolling_windows(paths[config["clocks"][0]], from_date=config["target_from"]):
        histories = _mapped(paths, [m["match_id"] for m in history])
        targets = _mapped(paths, [m["match_id"] for m in target])
        record = {"cutoff": cutoff.isoformat(), "history_matches": len(history), "target_matches": len(target), "clocks": {}}
        print(f"{name} later {cutoff}: selected={selected:g}; history={len(history)} target={len(target)}", flush=True)
        for clock in config["clocks"]:
            scores, diagnostics = {}, {}
            for kind, players, ridge in (("team", False, FIXED_RIDGE), ("fixed100", True, FIXED_RIDGE), ("selected", True, selected)):
                if kind == "selected" and selected == FIXED_RIDGE:
                    scores[kind] = scores["fixed100"]
                    diagnostics[kind] = {**diagnostics["fixed100"], "reused_fixed100_alias": True}
                else:
                    cache_kind = kind if kind != "selected" or selected == prior_ridge else None
                    cached = parent / name / "models" / f"{cutoff}_{clock}_{cache_kind}.json" if cache_kind else None
                    destination_model = models / f"{cutoff}_{clock}_{kind}.json"
                    if cached is not None and cached.is_file():
                        scores[kind], diagnostics[kind] = reuse_saved(cached, histories[clock], targets[clock],
                            cutoff, ridge, players, destination_model, progress)
                    else:
                        scores[kind], diagnostics[kind] = fit_saved(histories[clock], targets[clock], cutoff,
                            ridge, players, destination_model, progress)
                append_score(totals[clock][kind], scores[kind])
            record["clocks"][clock] = diagnostics
        weeks.append(record)
        write_json(destination / "逐周较晚拟合与覆盖.json", weeks)
    previous = {}
    for clock, values in totals.items():
        previous[clock] = json.loads((parent / name / f"later_{clock}_selected_predictions.json").read_text(encoding="utf-8"))
        for kind, score in values.items():
            write_json(destination / f"later_{clock}_{kind}_predictions.json", score)
    prior_weeks = json.loads((parent / name / "逐周拟合与覆盖.json").read_text(encoding="utf-8"))
    result = {"source_directory": str(source), "source_matches": len(paths[config["clocks"][0]]),
        "selection": selection, "prior_selected_player_ridge": prior_ridge,
        "prior_validation_requested_matches": old["selection"]["validation_requested_matches"],
        "prior_validation_common_matches": len(old["selection"]["common_match_ids"]),
        "comparison": compare_regularization(totals), "selected_vs_prior": compare_selected_tracks(previous,
            {c: values["selected"] for c, values in totals.items()}),
        "active_later_weeks": len(weeks), **{key: progress[key] - initial[key] for key in progress},
        "coverage": {c: {k: _coverage(s) for k, s in values.items()} for c, values in totals.items()},
        "last_week_coefficient_summaries": {c: {k: v.get("coefficient_summary") for k, v in values.items()}
            for c, values in weeks[-1]["clocks"].items()},
        "prior_last_week_selected_coefficient_summaries": {c: values["selected"].get("coefficient_summary")
            for c, values in prior_weeks[-1]["clocks"].items()}}
    write_json(destination / "多周前推收缩结果.json", result)
    del paths, totals, previous
    gc.collect()
    return result


def report(result):
    def fmt(value):
        return "—" if value is None else f"{value:+.6f}"
    unchanged = all(row["selection"]["selected_player_ridge"] == row["prior_selected_player_ridge"]
                    for row in result["seasons"].values())
    lines = ["# 多周前推选择球员收缩强度", "",
        f"新增实际拟合{result['actual_model_fits']}次，复用已有参数重新评分{result['reused_model_evaluations']}次；"
        "复用不计为训练，选中100时与固定100共用模型也不重复计数。", "",
        "候选仍为10／100／1000／10000，球队／状态正则仍为1，真实进球为目标，未知球员仍整场跳过。"
        "没有根据较晚分数改候选、换时钟、新增先验或改动市场双轨。", "",
        "## 较早多周前推验证", "",
        "只评分12月1日起至选择截止日前的比赛；每个验证周一只用更早日期历史重训。第一周一可能在12月1日前，"
        "因此不能复用12月1日截点的静态训练模型。2017/18选择严格截止2018-01-29，1月29日至31日不参与选择。"
        "先汇总各周逐场损失，再按共同比赛等权评分，不是把各周均分等权；比赛很少的周不会被放大。"
        "三时钟对同一比赛的损失等权平均，不代表三场独立比赛或真实时钟各1/3概率。", "",
        "| 来源 | 验证周数 | 请求 | 原静态共同 | 多周共同 | 原静态正则 | 多周正则 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        selected = row["selection"]
        lines.append(f"| {name} | {selected['active_weeks']} | {selected['validation_requested_matches']} | "
            f"{row['prior_validation_common_matches']} | {len(selected['common_match_ids'])} | "
            f"{row['prior_selected_player_ridge']:g} | {selected['selected_player_ridge']:g} |")
    lines += ["", "| 来源 | 候选正则 | 多周共同验证损失 |", "|---|---:|---:|"]
    for name, row in result["seasons"].items():
        for metric in row["selection"]["candidate_metrics"].values():
            lines.append(f"| {name} | {metric['player_ridge']:g} | {metric['mean_clock_match_loss']:.6f} |")
    lines += ["", "共同验证集合跨所有候选及全部时钟取交集；损失越低越好，1e−12内平手选较大正则。"
        "原静态与多周验证支持集合可能不同，不能直接把二者平均验证损失相减当改进。边界候选胜出不证明最优值在边界之外。", "",
        "## 较晚每周同场对照", "",
        "| 来源 | 时钟 | 三轨共同场数 | 原100−球队 | 多周选择−球队 | 多周选择−原100 | 原静态／多周共同场数 | 多周选择−原静态选择 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        comparison, prior = row["comparison"], row["selected_vs_prior"]
        for clock, values in comparison["per_clock"].items():
            lines.append(f"| {name} | {clock} | {len(comparison['common_match_ids'])} | {fmt(values['fixed100_minus_team'])} | "
                f"{fmt(values['selected_minus_team'])} | {fmt(values['selected_minus_fixed100'])} | "
                f"{len(prior['common_match_ids'])} | {fmt(prior['per_clock'][clock]['weekly_minus_previous_static_loss'])} |")
    lines += ["", "每个差值都在其所列共同样本上计算；负差较好。后三轨和直接新旧选择对照的交集可以不同，不能跨集合拼接差值。", "",
        ("本次两季前推选择均未改变。后段模型／目标／路径一致时复用参数并得到相同结果是预期现象，不是第二份独立的成功验证。"
         if unchanged else "本次至少一季前推选择发生变化；其较晚效果只按表内共同比赛比较，不以验证均分变化代替后段效果。"),
        "目前不能据此部署新的球员项或宣称已找到通用最佳正则，也不在同一后段上继续扩网格追分。", "",
        "| 来源 | 时钟 | 轨道 | 请求 | 支持 | 未知跳过 |", "|---|---|---|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        for clock, values in row["coverage"].items():
            for kind, coverage in values.items():
                lines.append(f"| {name} | {clock} | {kind} | {coverage['requested']} | {coverage['supported']} | {coverage['skipped']} |")
    lines += ["", "| 来源 | 时钟 | 最后一周条件球员系数RMS：固定100 | 多周选择 | 原静态选择 |", "|---|---|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        for clock, values in row["last_week_coefficient_summaries"].items():
            def rms(summary):
                return None if summary is None else summary["player_group"]["rms"]
            lines.append(f"| {name} | {clock} | {fmt(rms(values['fixed100']))} | {fmt(rms(values['selected']))} | "
                f"{fmt(rms(row['prior_last_week_selected_coefficient_summaries'][clock]))} |")
    lines += ["", "逐周参数、评分、覆盖、优化器状态和训练期暴露分组系数诊断分别保留。固定数值L2施加于求和损失，"
        "历史增加时相对收缩会变化；联合拟合可重新分配球队和球员系数，不能称每个人按同一比例收缩。", "",
        "## 解释边界", "",
        "这些较晚赛季此前已查看，本轮仍为探索性复验，不是新盲测。未知球员导致选择性的支持子集；未估计时间／球队相关抽样区间，"
        "不声称显著、稳定提高。周数、拟合次数和时钟数都不是独立比赛样本量。", "",
        "验证覆盖增加同时伴随身份逐周积累和训练量变化，不是把一个验证段单纯切成多个独立重复。"
        "任何选择都只能称四档候选中的选择，不是全局最优；若落在网格边界，也不据此自动扩展候选。"
        "本轮未引入惩罚随训练量缩放、分层先验、新球员默认值或赛前阵容分布，后续需先讨论规则再验证。", "",
        "目标仍是每半场固定45分钟、已知真实换人／比分／人数路径的事后条件强度，不是完整比赛赛前胜平负或个人因果价值。"
        "不据此给出2026年比赛预测、可执行串关或中奖保证。补时进球不进入该窗口目标，上半场补时状态仍传至下半场。", "",
        "所有项目文件、模型与临时目录在项目目录；仅复用本地既有来源，不下载新数据、不新增依赖。原数据许可仍适用，"
        "数据与模型不随软件源码包分发。", ""]
    return "\n".join(lines)


def run(root, parent=None):
    root = Path(root).resolve()
    parent = find_parent(root, parent)
    prior = json.loads((parent / PARENT_RESULT).read_text(encoding="utf-8"))
    if set(prior["seasons"]) != set(SEASONS):
        raise ValueError("Parent must retain both declared source seasons")
    output = root / "outputs" / ("多周前推收缩验证_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    write_json(output / "实验设置.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_directory": str(parent), "sources": {n: v["source_directory"] for n, v in prior["seasons"].items()},
        "player_ridges": list(PLAYER_RIDGES), "fixed_comparator": FIXED_RIDGE, "team_ridge": 1, "context_ridge": 1,
        "seasons": {n: {k: v.isoformat() if type(v) is date else v for k, v in cfg.items()} for n, cfg in SEASONS.items()},
        "validation": "weekly strictly earlier history; targets train_before <= date < selection_before; common all-candidate all-clock IDs",
        "selection": "equal-match pooled loss, then equal clock mean; NOT equal-week average; tie1e-12 larger ridge; no later-result selection",
        "model_reuse": "same source cohort/reference IDs/cutoff/response/flags/ridge settings; copied params and freshly scored; counted separately",
        "unknown_identity": "skip whole match, no zero prior", "exploratory_previously_viewed_seasons": True,
        "formal_prematch_probabilities_changed": False})
    progress = {"actual_model_fits": 0, "reused_model_evaluations": 0}
    try:
        result = {"parent_directory": str(parent), "seasons": {}}
        for name in SEASONS:
            result["seasons"][name] = run_season(name, parent, prior, output, progress)
        result.update(progress)
        write_json(output / RESULT_FILE, result)
        (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
        print(f"Forward selection complete: {output}; counts={progress}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {**progress, "error": f"{type(error).__name__}: {error}", "status": "incomplete"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--report-only", type=Path)
    arguments = parser.parse_args()
    if arguments.report_only:
        result = json.loads((arguments.report_only / RESULT_FILE).read_text(encoding="utf-8"))
        (arguments.report_only / REPORT_FILE).write_text(report(result), encoding="utf-8")
        print(json.dumps({"report_only": True, "new_model_fits": 0, "output": str(arguments.report_only)}, ensure_ascii=False))
    else:
        run(arguments.root, arguments.parent)
