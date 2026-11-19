"""One limited-budget dynamic attack/defense study, not a model search.

Select one shared update speed on 2024/25, save it, then open 2025/26.
The later season has already been viewed: this remains exploratory research.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .cmp_forward_experiment import write_histogram
from .core_tuning_experiment import analyze, loss_vectors, score_rows, write_csv
from .forecast_diagnostics import BLOCKS, REPEATS, SEED, write_json
from .rate_calibration_experiment import load_period
from .refresh_experiment import describe
from .snapshot_dynamic import fit_learning_rate, filter_rows


ARMS = ("original", "selected")
DOMAINS = ("exact_score", "one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way")
PARAMETER_FILE = "较早学习的唯一更新速度.json"
RESULT_FILE = "快照内动态攻防结果.json"
REPORT_FILE = "快照内动态攻防说明.md"


def tracks_for(rows, eta):
    tracks = {"original": deepcopy(rows), "selected": filter_rows(rows, eta)}
    ids = [row["match_id"] for row in tracks["original"]]
    if ids != [row["match_id"] for row in tracks["selected"]]:
        raise ValueError("Dynamic forecasts must retain every source match in order")
    for arm in ARMS:
        exact = loss_vectors(tracks[arm])["exact_score"]
        for row, value in zip(tracks[arm], exact):
            row["exact_score_nll"] = float(value)
    return tracks


def interval(later, domain, weeks=4):
    return later["bootstrap"][str(weeks)]["differences"][domain]["interval_95"]


def report(result):
    eta, later = result["parameter"]["eta"], result["later"]
    lines = ["# 限额实测：28天快照内动态攻防", "",
        f"只做1种动态结构、1个共用更新速度η={eta:.9f}。实际目标计算"
        f"{result['parameter']['n_evaluations']}次（上限14），不做大参数网格。"
        f"较早{result['earlier']['n']}场用于估计η，较晚{later['n']}场固定η前推对照。"
        "0次新增原球队拟合，正式默认未改；项目文件、缓存及临时文件均在项目目录。", "",
        "## 模型究竟改变了什么", "",
        "logλ主=log原μ主+a主−d客，客队对称；d越大代表防守越强。"
        "每天所有比赛先用日初状态生成预测，再用这些预测的残差e=实际进球−λ批量更新："
        "进球队伍a增加ηe/2，对方d减少ηe/2。只更新已发生比赛的球队；不额外对全联赛状态去均值。"
        "η=0精确退化为原μ。胜平负、总进球及让球均来自更新μ后的同一独立Poisson比分分布。", "",
        "原μ已来自180天衰减/.01收缩、28天重训；因此每当原fit_before_date更换，"
        "该联赛额外攻防残差归零，避免重复吸收新快照已经学过的赛果。"
        "归零的是给定已估计球队实力之后的额外短期残差，不是把未知球队/球员实力当零。"
        "本研究只检验两次重训之间新赛果的作用，不冒充完整长期随机状态空间模型，"
        "也没有状态后验方差或球员、阵容及赛事投入的新信息。", "",
        "方法借鉴[gasmodel的score-driven过滤框架](https://github.com/vladimirholy/gasmodel)。"
        "本项目自行实现上述简化Poisson score递推，不复制该R包代码、不安装它，"
        "也不是复刻其全部GAS模型或其引用论文的预测结果。", "",
        "## 一次估计，到此为止", "",
        "只允许η∈[0,.20]；优化器最多12次目标计算，另评估确切0与.20端点，总上限14次。"
        "上限意味着一个超额进球使单队攻/防log状态各最多改变.10，不是数值稳定保证。"
        "不按结果扩大区间、换衰减、分联赛学习或重启搜索；非有限率/优化失败明确报错，不截断或回退。"
        "有界数值估计不等于证明全局最优。选择0是有效候选，不是失败回退。", "",
        "训练目标是2024/25逐日前推完整比分NLL，η的选后较早成绩属于训练成绩，"
        "不能当成独立验证。参数文件保存后才加载2025/26；后季只更新状态，不再估η。"
        "所有源文件来自既有合格预测队列，只更新该队列内的赛果；未覆盖被原模型排除的未知球队/特殊比分比赛。"
        "源存档核验会读取整季赛果；过滤本身严格按日先预测后更新，首次公开时点仍未知。"
        "这两个赛季已被查看，本轮是探索研究，不是新的盲测或LLM同场评测。", "",
        "## 较晚同场结果（损失越低越好）", "",
        "| 目标 | 原模型 | 动态补充 | 新减旧 | 四周条件探索95%区间 |", "|---|---:|---:|---:|---|"]
    for domain in DOMAINS:
        field = "mean_nll" if domain == "exact_score" else "log_loss"
        old, new = (later["scores"][arm][domain][field] for arm in ARMS)
        ci = interval(later, domain)
        label = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {domain} | {old:.6f} | {new:.6f} | {new-old:+.6f} | {label} |")
    lines += ["", "| 目标 | 2周条件区间 | 8周条件区间 |", "|---|---|---|"]
    for domain in DOMAINS:
        labels = []
        for weeks in (2, 8):
            ci = interval(later, domain, weeks)
            labels.append("不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]")
        lines.append(f"| {domain} | " + " | ".join(labels) + " |")
    lines += ["", "## 市场仍独立", "",
        "| 市场 | 同场数 | 市场胜平负LL | 原模型 | 动态补充 |", "|---|---:|---:|---:|---:|"]
    for market, item in later["markets"].items():
        if item["n"]:
            scores = item["model_scores_on_same_matches"]
            lines.append(f"| {market} | {item['n']} | {item['market_scores']['log_loss']:.6f} | "
                f"{scores['original']['log_loss']:.6f} | {scores['selected']['log_loss']:.6f} |")
    primary = interval(later, "exact_score")
    delta = later["selected_minus_original"]["exact_score"]
    if eta == 0:
        conclusion = "较早估计选择η=0，动态轨与原模型相同，没有增加短期动态修正的证据。"
    elif delta >= 0:
        conclusion = "较晚完整比分点损失没有改善，不采用此候选。"
    elif primary is None or primary[0] <= 0 <= primary[1]:
        conclusion = "较晚完整比分点损失下降，但主区间跨零或不可估计，未确认稳定改善。"
    else:
        conclusion = "较晚完整比分有条件探索改善信号，仍需看其余目标及新数据，不自动替换正式模型。"
    lines += ["", "## 结论与工作边界", "", conclusion,
        "本轮到此为止，不再追加参数搜索。较早训练成绩、较晚全部四目标、联赛/月、市场同子集及逐场损失分布均保留。"
        "selected键仅表示较早学定η的动态候选，不表示选中了更优正式模型。", "",
        f"共享五联赛日历2/4/8周移动块，{result['settings']['bootstrap_repeats']}重复、seed={SEED}；"
        "区间条件于学定η及原预测，不含重新估η、球队重训、结构选择或所有长期依赖。"
        "没有作IID t检验、正态假设或多重比较后的确认性宣称。"
        "实验设计技能使比较限定为一种结构和一次有限估计；统计分析技能使全部目标及条件不确定性保留。", "",
        "尚未实测LLM直接预测是否更好。历史比赛可能已在LLM知识中，不能把隐藏赛果列当作消除了其记忆泄漏。"
        "后续LLM可独立提取有来源、有发布时间的伤停/轮换/赛事重视信息，"
        "若直接给概率，应与统计模型、市场同场前瞻比较，不凭文字解释或最高概率命中率认定更优。"
        "本轮未调用外部LLM、付费接口或安装大模型。", "",
        "球员/伤停、赛事投入、跨联赛欧战连接、半全场尚未完成，不据此发布当前串关。"
        "竞彩客让两球让胜需净胜至少3球，净胜2球是让平；不保证中奖。"
        "没有新hash、冻结contract、发布门禁或新依赖。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    root, parent, output = Path(root), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"structure": "snapshot_local_score_driven_attack_defense",
        "eta_bounds": [0., .20], "maximum_optimizer_evaluations": 12,
        "maximum_total_evaluations": 14, "eta_training_season": "2024-25",
        "fixed_eta_evaluation_season": "2025-26", "primary_domain": "exact_score",
        "base_half_life_days": 180, "base_regularization": .01, "base_update_days": 28,
        "same_day_policy": "all_predictions_before_any_score_update",
        "snapshot_change_policy": "reset_extra_states_to_zero",
        "state_input_cohort": "original_eligible_archived_predictions_only",
        "bootstrap_repeats": repeats, "bootstrap_blocks_weeks": list(BLOCKS), "seed": SEED,
        "source": str(parent), "previously_viewed_seasons": True,
        "production_defaults_changed": False, "new_model_families_this_run": 1,
        "no_more_hyperparameter_search_after_this_run": True}
    write_json(output / "实验设置.json", settings)
    stage = "earlier_source"
    try:
        earlier, early_source = load_period(root, parent, "2024-25")
        stage = "single_limited_estimation"
        parameter = fit_learning_rate(earlier)
        write_json(output / PARAMETER_FILE, parameter)
        # No future source rows enter update-speed estimation.
        stage = "later_source_after_parameter_save"
        later, late_source = load_period(root, parent, "2025-26")
        stage = "forecasts"
        early_tracks = tracks_for(earlier, parameter["eta"])
        late_tracks = tracks_for(later, parameter["eta"])
        write_json(output / "较早训练预测.json", early_tracks)
        write_json(output / "较晚动态预测.json", late_tracks)
        write_csv(output / "逐场较晚动态概率.csv", late_tracks)
        stage = "same_match_analysis"
        later_analysis, aligned = analyze(late_tracks, repeats=repeats)
        early_losses = {arm: loss_vectors(early_tracks[arm]) for arm in ARMS}
        earlier_analysis = {"n": len(earlier), "role": "eta_training_not_independent_validation",
            "scores": {arm: score_rows(early_tracks[arm]) for arm in ARMS},
            "selected_minus_original": {d: float(np.mean(early_losses["selected"][d]-early_losses["original"][d])) for d in DOMAINS}}
        per_loss = {arm: loss_vectors(aligned[arm]) for arm in ARMS}
        later_analysis["paired_loss_descriptives"] = {}
        for domain in DOMAINS:
            values = per_loss["selected"][domain]-per_loss["original"][domain]
            later_analysis["paired_loss_descriptives"][domain] = {**describe(values),
                "n_negative": int(np.sum(values < 0)), "n_positive": int(np.sum(values > 0)),
                "n_zero": int(np.sum(values == 0))}
        write_histogram(output / "动态逐场损失分布.svg", aligned, candidate="selected")
        result = {"settings": settings, "parameter": parameter, "earlier": earlier_analysis,
            "later": later_analysis, "sources": {"2024-25": early_source, "2025-26": late_source},
            "counts": {"new_shared_update_speed_fits": 1, "new_team_fits": 0,
                "objective_evaluations": parameter["n_evaluations"],
                "unique_training_matches": len(earlier), "unique_evaluation_matches": len(later),
                "new_dynamic_forecasts": len(earlier)+len(later),
                "reused_original_forecasts": len(earlier)+len(later),
                "later_changed_rates": sum(a["rates"] != b["rates"] for a, b in zip(later, late_tracks["selected"])),
                "reused_team_snapshots": early_source["parameter_snapshots"]+late_source["parameter_snapshots"]},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "elapsed_seconds": time.perf_counter()-started}}
        write_json(output / RESULT_FILE, result)
        (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
        return result
    except Exception as exc:
        write_json(output / "研究失败.json", {"stage": stage, "type": type(exc).__name__,
            "message": str(exc), "details": getattr(exc, "details", None),
            "history": getattr(exc, "history", None), "no_expanded_search_or_fallback": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = args.parent or args.root / "outputs" / "更新频率研究_20261023T073047979805Z"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "outputs" / f"限额动态攻防_{stamp}"
    result = run(args.root, parent, output)
    print(json.dumps({"output": str(output), "eta": result["parameter"]["eta"],
        "objective_evaluations": result["parameter"]["n_evaluations"],
        "later_n": result["later"]["n"], "differences": result["later"]["selected_minus_original"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
