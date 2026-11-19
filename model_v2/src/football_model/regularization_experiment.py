"""Earlier-only player-ridge selection and later weekly descriptive comparisons.

All source paths have already been viewed. This is not a blinded test, causal
player valuation, or a pre-match probability model. Clock weights are a tuning
rule, not a distribution over true substitution times.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import gc
import json
from math import sqrt
from pathlib import Path

from .full_season_experiment import score_matches
from .joint_stint_model import JointStintIntensityModel
from .regularization_comparison import compare_regularization, select_regularization
from .rolling_player_experiment import rolling_windows
from .statsbomb_data import write_json
from .window_clock_experiment import append_score, empty_score


PLAYER_RIDGES = (10.0, 100.0, 1000.0, 10000.0)
FIXED_RIDGE = 100.0
SEASONS = {
    "statsbomb_2015_16": {"train_before": date(2015, 12, 1),
        "selection_before": date(2016, 2, 1), "target_from": date(2016, 2, 1),
        "clocks": ("exact",), "source_pattern": "同源窗口与时钟对照_*",
        "source_file": "按窗口与时钟派生时段.json", "completed_file": "同源窗口与时钟结果.json"},
    "wyscout_2017_18": {"train_before": date(2017, 12, 1),
        "selection_before": date(2018, 1, 29), "target_from": date(2018, 2, 1),
        "clocks": ("early", "middle", "late"), "source_pattern": "第二赛季固定窗口球员探索_*",
        "source_file": "固定90分钟派生时段.json", "completed_file": "固定窗口探索结果.json"},
}


def find_source(root, config):
    candidates = sorted((Path(root) / "outputs").glob(config["source_pattern"]), reverse=True)
    return next(p for p in candidates if (p / config["completed_file"]).is_file()
                and (p / config["source_file"]).is_file())


def load_paths(source, name):
    config = SEASONS[name]
    raw = json.loads((Path(source) / config["source_file"]).read_text(encoding="utf-8"))
    if name == "statsbomb_2015_16":
        paths = {"exact": raw["fixed45"]["exact"]}
    else:
        paths = {c: raw[c] for c in config["clocks"]}
    del raw
    gc.collect()
    indices = {c: {m["match_id"]: m for m in values} for c, values in paths.items()}
    if any(len(indices[c]) != len(paths[c]) for c in paths):
        raise ValueError("Distinct source match IDs required")
    first = set(indices[config["clocks"][0]])
    if any(set(index) != first for index in indices.values()):
        raise ValueError("All clock paths must use the same source cohort")
    return paths


def selection_split(paths, train_before, selection_before, target_from):
    """No result on/after the first late-fit cutoff can select a ridge."""
    if not all(type(d) is date for d in (train_before, selection_before, target_from)):
        raise ValueError("Calendar dates required")
    if not train_before < selection_before <= target_from:
        raise ValueError("Ordered training, selection and target dates required")
    first = next(iter(paths.values()))
    windows = list(rolling_windows(first, from_date=target_from))
    if not windows or selection_before > windows[0][0]:
        raise ValueError("Selection results must precede the first weekly fit cutoff")
    train, validation = [], []
    for m in first:
        day = date.fromisoformat(m["match_date"])
        if day < train_before:
            train.append(m["match_id"])
        elif day < selection_before:
            validation.append(m["match_id"])
    if not train or not validation:
        raise ValueError("Nonempty earlier training and validation required")
    return train, validation, windows[0][0]


def coefficient_summary(model, history):
    """Training-only group norms; no causal values or uniform shrinkage claim."""
    seconds = defaultdict(float)
    for m in history:
        for s in m["stints"]:
            for col in ("home_players", "away_players"):
                for pid in s[col]:
                    seconds[pid] += s["duration_seconds"]
    def group(values):
        return {"coefficients": len(values), "rms": sqrt(sum(v*v for v in values) / len(values)) if values else None,
                "l2": sqrt(sum(v*v for v in values)), "max_abs": max(map(abs, values), default=None)}
    coefficients = dict(zip(model.feature_names, map(float, model.coefficients)))
    bins = {"under_450_minutes": [], "450_to_under_1350_minutes": [], "at_least_1350_minutes": []}
    if model.include_players:
        for pid in model.players:
            minutes = seconds[pid] / 60
            key = "under_450_minutes" if minutes < 450 else "450_to_under_1350_minutes" if minutes < 1350 else "at_least_1350_minutes"
            bins[key].extend(coefficients[f"player_{kind}:{pid}"] for kind in ("attack", "defence"))
    return {"player_group": group([v for k, v in coefficients.items() if k.startswith("player_")]),
        "team_group": group([v for k, v in coefficients.items() if k.startswith("team_")]),
        "player_exposure_bins": {k: group(v) for k, v in bins.items()},
        "training_matches": len(history), "training_stints": sum(len(m["stints"]) for m in history),
        "training_exposure_seconds": sum(s["duration_seconds"] for m in history for s in m["stints"]),
        "known_players": len(model.players)}


def fit_saved(history, target, cutoff, ridge, players, destination, progress):
    model = JointStintIntensityModel(include_players=players, response_kind="goals",
        team_ridge=1.0, player_ridge=ridge, context_ridge=1.0).fit(history, cutoff)
    progress["actual_model_fits"] += 1
    model.save(destination)
    restored = JointStintIntensityModel.load(destination)
    if restored.parameters() != model.parameters():
        raise ValueError("Saved parameters do not roundtrip")
    score = score_matches(model, target)
    if score["predictions"]:
        mid = score["predictions"][0]["match_id"]
        witness = next(m for m in target if m["match_id"] == mid)
        if restored.predict_match(witness) != model.predict_match(witness):
            raise ValueError("Saved witness prediction does not roundtrip")
    return score, {"player_ridge": ridge, "include_players": players,
        "optimizer": model.optimizer_diagnostics, "coefficient_summary": coefficient_summary(model, history),
        "saved_parameter_and_witness_roundtrip": True,
        "supported": len(score["predictions"]), "requested": len(target), "skipped": score["skipped"]}


def run_season(name, source, output, progress):
    config = SEASONS[name]
    destination = output / name
    destination.mkdir()
    models = destination / "models"
    models.mkdir()
    initial_fits = progress["actual_model_fits"]
    paths = load_paths(source, name)
    by_id = {c: {m["match_id"]: m for m in values} for c, values in paths.items()}
    train_ids, val_ids, first_cutoff = selection_split(paths, config["train_before"],
        config["selection_before"], config["target_from"])
    write_json(destination / "分段与来源.json", {"source_directory": str(source), "source_file": config["source_file"],
        "source_matches": len(next(iter(paths.values()))), "training_match_ids": train_ids,
        "validation_match_ids": val_ids, "first_weekly_cutoff": first_cutoff.isoformat(),
        "selection_before": config["selection_before"].isoformat()})
    validation, diagnostics = {}, {}
    for clock in config["clocks"]:
        validation[clock], diagnostics[clock] = {}, {}
        train = [by_id[clock][mid] for mid in train_ids]
        target = [by_id[clock][mid] for mid in val_ids]
        for ridge in PLAYER_RIDGES:
            print(f"{name} validation fit {clock} ridge={ridge:g}: train={len(train)} val={len(target)}", flush=True)
            key = f"{ridge:g}"
            score, diag = fit_saved(train, target, config["train_before"], ridge, True,
                models / f"validation_{clock}_ridge{key}.json", progress)
            validation[clock][key], diagnostics[clock][key] = score, diag
            write_json(destination / f"validation_{clock}_ridge{key}_predictions.json", score)
    selection = select_regularization(validation)
    selected = selection["selected_player_ridge"]
    selection.update({"diagnostics": diagnostics, "selection_before": config["selection_before"].isoformat(),
        "first_weekly_cutoff": first_cutoff.isoformat(), "validation_requested_matches": len(val_ids)})
    write_json(destination / "较早验证选择.json", selection)
    print(f"{name}: selected ridge {selected:g} on {len(selection['common_match_ids'])}/{len(val_ids)} common validation games", flush=True)
    totals = {c: {k: empty_score() for k in ("team", "fixed100", "selected")} for c in config["clocks"]}
    weeks = []
    for cutoff, history, target in rolling_windows(next(iter(paths.values())), from_date=config["target_from"]):
        history_ids, target_ids = [m["match_id"] for m in history], [m["match_id"] for m in target]
        record = {"cutoff": cutoff.isoformat(), "history_matches": len(history_ids), "target_matches": len(target_ids), "clocks": {}}
        print(f"{name} weekly fit {cutoff}: history={len(history_ids)} target={len(target_ids)} selected={selected:g}", flush=True)
        for clock in config["clocks"]:
            history_path = [by_id[clock][mid] for mid in history_ids]
            target_path = [by_id[clock][mid] for mid in target_ids]
            scores, diag = {}, {}
            for kind, players, ridge in (("team", False, FIXED_RIDGE), ("fixed100", True, FIXED_RIDGE), ("selected", True, selected)):
                if kind == "selected" and selected == FIXED_RIDGE:
                    scores[kind], diag[kind] = scores["fixed100"], {**diag["fixed100"], "reused_fixed100_model": True}
                else:
                    scores[kind], diag[kind] = fit_saved(history_path, target_path, cutoff, ridge, players,
                        models / f"{cutoff}_{clock}_{kind}.json", progress)
                append_score(totals[clock][kind], scores[kind])
            record["clocks"][clock] = diag
        weeks.append(record)
        write_json(destination / "逐周拟合与覆盖.json", weeks)
    for clock in totals:
        for kind, score in totals[clock].items():
            write_json(destination / f"later_{clock}_{kind}_predictions.json", score)
    result = {"source_directory": str(source), "source_matches": len(next(iter(paths.values()))),
        "selection": selection, "comparison": compare_regularization(totals),
        "active_weeks": len(weeks), "actual_model_fits": progress["actual_model_fits"] - initial_fits,
        "coverage": {c: {k: {"requested": s["requested_matches"], "supported": len(s["predictions"]),
            "skipped": len(s["skipped"])} for k, s in values.items()} for c, values in totals.items()}}
    write_json(destination / "正则对照结果.json", result)
    del paths, by_id, validation, totals
    gc.collect()
    return result


def report(result):
    lines = ["# 球员正则：较早选择、较晚每周对照", "",
        f"实际完成{result['actual_model_fits']}次拟合。只比较固定数值的球员L2正则10／100／1000／10000；球队、状态正则仍为1。"
        "未知球员整场跳过，即使极强正则也不默认其效果为0。没有新增新球员先验、xG、市场融合或战意系数。", "",
        "## 较早参数选择", "",
        "两季独立用12月1日之前比赛拟合，后续较早验证段选参数。2015/16验证截止2016-02-01；"
        "2017/18必须截止2018-01-29，而非2月1日，因为第一轮较晚预测在1月29日训练。所有截止日期均不含当日。"
        "三套Wyscout时钟共同比赛的损失等权平均，只是预定选择准则，不代表真实时钟各有1/3概率或三份独立样本。", "",
        "| 来源 | 验证请求 | 共同可评分 | 选中正则 |", "|---|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        s = row["selection"]
        lines.append(f"| {name} | {s['validation_requested_matches']} | {len(s['common_match_ids'])} | {s['selected_player_ridge']:g} |")
    lines += ["", "| 来源 | 候选正则 | 共同验证场的平均损失 |", "|---|---:|---:|"]
    for name, row in result["seasons"].items():
        for candidate in row["selection"]["candidate_metrics"].values():
            lines.append(f"| {name} | {candidate['player_ridge']:g} | {candidate['mean_clock_match_loss']:.6f} |")
    lines += ["", "验证只在所有候选／全部时钟共同支持场次选最小损失；1e−12内平手取较强收缩。"
        "较晚只训练选中值与原100，不重搜候选；若选中100则复用同一模型，拟合次数不重复计算。", "",
        "## 较晚每周前向结果", "",
        "以下每行在同一时钟、同场、同进球及真实路径比较；跨时钟也使用总交集。损失越低越好，负差较好。"
        "每周一只用更早日期比赛，参数选择段也早于第一轮训练截止。", "",
        "| 来源 | 时钟 | 共同较晚场数 | 原100−仅球队 | 选中−仅球队 | 选中−原100 |",
        "|---|---|---:|---:|---:|---:|"]
    def fmt(x):
        return "—" if x is None else f"{x:+.6f}"
    for name, row in result["seasons"].items():
        compare = row["comparison"]
        for c in SEASONS[name]["clocks"]:
            values = compare["per_clock"][c]
            lines.append(f"| {name} | {c} | {len(compare['common_match_ids'])} | {fmt(values['fixed100_minus_team'])} | "
                f"{fmt(values['selected_minus_team'])} | {fmt(values['selected_minus_fixed100'])} |")
    lines += ["", "不能只凭选出候选就接入正式预测。选择是否优于原100或仅球队应分别看同场损失差，"
        "正差表示更差，负差表示更好；也不能把改善或恶化直接归为球员的因果价值。"
        "不因后段成绩回改选择，不继续扩展网格以追逐同一后段。"
        "固定窗口验证与早先完整目标验证的目标和共同集合不同，不将其选择差异称为同一试验的矛盾。", ""]
    for name, row in result["seasons"].items():
        averages = row["comparison"]["clock_averaged"]
        lines.append(f"{name}：较早选择{row['selection']['selected_player_ridge']:g}；较晚时钟均值的选中−原100为"
            f"{fmt(averages['selected_minus_fixed100'])}、选中−仅球队为{fmt(averages['selected_minus_team'])}。")
    lines += ["",
        "| 来源 | 时钟 | 较晚请求 | 原100／选中球员可评分 |",
        "|---|---|---:|---:|"]
    for name, row in result["seasons"].items():
        for c in SEASONS[name]["clocks"]:
            cov = row["coverage"][c]
            lines.append(f"| {name} | {c} | {cov['selected']['requested']} | {cov['fixed100']['supported']}／{cov['selected']['supported']} |")
    lines += ["", "上表是各时钟原覆盖；前一损失表统一采用各来源全部时钟／模型的共同交集，不将不同支持集合的均分混入比较。", "",
        "## 收缩量：仅较早训练数据", "",
        "下表为较早训练拟合的球员攻防系数组RMS，不是后段估计，更不是胜率或球员身价。", "",
        "| 来源 | 时钟 | ridge10 RMS | ridge100 RMS | ridge1000 RMS | ridge10000 RMS |",
        "|---|---|---:|---:|---:|---:|"]
    for name, row in result["seasons"].items():
        for c in SEASONS[name]["clocks"]:
            values = [row["selection"]["diagnostics"][c][f"{ridge:g}"]["coefficient_summary"]["player_group"]["rms"] for ridge in PLAYER_RIDGES]
            lines.append(f"| {name} | {c} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "各模型全部请求／支持／未知跳过、每场损失、全部参数、优化器状态及训练期球员系数组RMS/L2另存。"
        "球队／球员／状态联合拟合会重新分配解释量，不能称每名球员按同一比例收缩，更不能当作个人因果价值。", "",
        "## 解释边界", "",
        "所有赛季的较晚资料已经在前轮查看，本轮是探索性复验，不是新盲测。支持集合很小且非随机，未估计时间／球队相关的抽样区间，"
        "不声称显著、稳定提高，也不按本轮较晚分数再改选正则或挑时钟。"
        "三时钟不是独立比赛，两个来源不可简单合并成同质样本。", "",
        "正则施加于求和损失：Σ(暴露×exp(η)−进球×η)+Σ(ridge×系数²)/2。"
        "训练比赛增加时，固定数值ridge的相对约束会减弱；本轮记录训练场数／时段／暴露，但没有调整惩罚随样本量缩放。"
        "极大ridge接近无球员项不等于产生新的未知球员先验。", "",
        "研究目标仍是每半场固定45分钟内、以真实换人／比分／人数路径为条件的事后强度，不是完整比赛赛前1X2。"
        "补时退出目标但保留上半场补时对下半场状态的影响。原有源排除和近似时钟冲突保持不变，未修补或删除不利进球。"
        "市场双轨、正式赛前概率和原推荐均未修改。不能据本轮参数判断2026年球队实力或给出可执行串关。", "",
        "全部项目文件、模型与输出在G盘；只复用已保留本地派生数据，无下载或新依赖。"
        "原始StatsBomb／Wyscout／Football-Data许可和来源继续适用，数据及模型不随软件源码ZIP分发。", ""]
    return "\n".join(lines)


def run(root):
    root = Path(root).resolve()
    sources = {name: find_source(root, config) for name, config in SEASONS.items()}
    output = root / "outputs" / ("球员收缩验证_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    write_json(output / "实验设置.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "player_ridges": list(PLAYER_RIDGES), "fixed_comparator": FIXED_RIDGE, "team_ridge": 1, "context_ridge": 1,
        "seasons": {n: {k: v.isoformat() if type(v) is date else v for k, v in cfg.items()} for n, cfg in SEASONS.items()},
        "sources": {n: str(p) for n, p in sources.items()}, "target": "fixed45 goals, retrospective actual path conditional intensity",
        "candidate_selection": "earlier validation common IDs across every candidate and clock; equal-clock mean loss; tie1e-12 larger ridge",
        "later_comparisons": "selected vs fixed100 vs team, weekly strictly earlier history; selected100 reuses model",
        "coefficient_diagnostics": "RMS/L2 and training minutes bins <450, [450,1350), >=1350; descriptive only",
        "unknown_identity": "skip whole match, no zero prior", "penalty_scaling": "summed loss; fixed numerical ridge, not sample-size normalized",
        "exploratory_previously_viewed_seasons": True, "formal_prematch_probabilities_changed": False})
    progress = {"actual_model_fits": 0}
    try:
        result = {"seasons": {}}
        for name, source in sources.items():
            result["seasons"][name] = run_season(name, source, output, progress)
        result.update(progress)
        write_json(output / "球员收缩验证结果.json", result)
        (output / "球员收缩验证说明.md").write_text(report(result), encoding="utf-8")
        print(f"Regularization experiment complete: {output}; actual fits={progress['actual_model_fits']}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {**progress, "error": f"{type(error).__name__}: {error}", "status": "incomplete"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--report-only", type=Path)
    arguments = parser.parse_args()
    if arguments.report_only:
        result = json.loads((arguments.report_only / "球员收缩验证结果.json").read_text(encoding="utf-8"))
        (arguments.report_only / "球员收缩验证说明.md").write_text(report(result), encoding="utf-8")
        print(json.dumps({"report_only": True, "new_model_fits": 0, "output": str(arguments.report_only)}, ensure_ascii=False))
    else:
        run(arguments.root)
