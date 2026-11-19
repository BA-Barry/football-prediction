"""Mean-direction-separated score-shape diagnostics; no fit or new forecasts.

Use both policies from the core parameter experiment on the same matches.
Factorial residuals and first-order event projections diagnose the reference
independent Poisson; they are not replacement probabilities or proof of CMP.
"""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .core_tuning_experiment import (ORIGINAL_ID, POLICY_FILE, read_json,
    load_source, replay_rates)
from .core_tuning import candidate_configs
from .football_data import LEAGUES
from .forecast_diagnostics import (BLOCKS, REPEATS, SEED, align_tracks,
    moving_block_means, write_json)
from .goal_rate_diagnostics import RATE_EDGES, rate_bin_labels
from .probability import score_grid, summarize
from .refresh_experiment import describe
from .refresh_research import PERIODS, _same
from .score_shape import EVENTS, STAT_FIELDS, row_shape_statistics


ARMS = ("original", "selected")
DIAGNOSTICS = ("home_residual", "away_residual", "home_dispersion_residual",
    "away_dispersion_residual", "mean_centered_cross_product",
    *(f"{event}_residual" for event in EVENTS),
    *(f"{event}_mean_orthogonal_residual" for event in EVENTS))
RESULT_FILE = "比分形态诊断结果.json"
REPORT_FILE = "比分形态诊断说明.md"
BIN_NAMES = tuple(f"{a:g}-{b:g}" for a, b in zip(RATE_EDGES, RATE_EDGES[1:])) + ("3+",)


def load_tracks(root, parent, period):
    """Read readable parameter archives and compare their chronology with CSVs."""
    records, source_reports = load_source(root, period)
    selection = read_json(Path(parent) / POLICY_FILE)
    configurations = {config["id"]: config for config in candidate_configs()}
    if selection["config"] != configurations.get(selection["config"]["id"]):
        raise ValueError("Selected policy differs from the declared core candidate")
    tracks, snapshots = {}, {}
    start, end = PERIODS[period]
    for arm in ARMS:
        identifier = ORIGINAL_ID if arm == "original" else selection["config"]["id"]
        folder = Path(parent) / period / (identifier if period == "2024-25" else arm)
        tracks[arm], snapshots[arm] = [], 0
        for comp in sorted(LEAGUES):
            trial = read_json(folder / f"{comp}.json")
            history = [r for r in records if r.competition == comp]
            if (trial["config"]["id"] != identifier or trial["start"] != start.isoformat()
                    or trial["end_exclusive"] != end.isoformat() or trial["update_interval_days"] != 28):
                raise ValueError("Saved policy/period differs from the core experiment")
            expected_config = configurations[ORIGINAL_ID] if arm == "original" else selection["config"]
            if trial["config"] != expected_config:
                raise ValueError("Saved full configuration differs from the selected policy")
            for fit in trial["fits"]:
                cutoff = date.fromisoformat(fit["before_date"])
                earlier = sorted((r for r in history if r.match_date < cutoff and not r.goal_fit_exclusion),
                                 key=lambda r: (r.match_date, r.match_id))
                if not earlier:
                    raise ValueError("Saved fit requires nonempty strictly earlier history")
                _same(fit["reference_match_ids"], [r.match_id for r in earlier], "strict_history")
                _same(fit["n_matches"], len(earlier), "training_size")
                if (fit["training_min_date"] != earlier[0].match_date.isoformat()
                        or fit["training_max_date"] != earlier[-1].match_date.isoformat()):
                    raise ValueError("Saved training dates differ from source history")
            for row in trial["predictions"]:
                played = date.fromisoformat(row["date"])
                cutoff = start + timedelta(days=((played-start).days // 28) * 28)
                if row["fit_before_date"] != cutoff.isoformat():
                    raise ValueError("Saved cutoff does not follow fixed 28-day anchor")
            replay_rates(trial, history)
            snapshots[arm] += len(trial["fits"])
            tracks[arm].extend(trial["predictions"])
    return tracks, {"source_reports": source_reports, "parameter_snapshots_read": snapshots,
                    "selection": selection["config"], "new_fits": 0, "new_forecasts": 0}


def cohort(statistics, indices):
    if not indices:
        return {"n": 0, "summaries": None, "event_counts": None}
    return {"n": len(indices), "summaries": {arm: {
        field: describe([statistics[arm][i][field] for i in indices]) for field in DIAGNOSTICS} for arm in ARMS},
        "event_counts": {arm: {event: {
            "observed": int(round(sum(statistics[arm][i][f"{event}_residual"] +
                                      statistics[arm][i][f"{event}_probability"] for i in indices))),
            "expected": float(sum(statistics[arm][i][f"{event}_probability"] for i in indices))}
            for event in EVENTS} for arm in ARMS}}


def analyze(raw_tracks, *, repeats=REPEATS):
    if set(raw_tracks) != set(ARMS):
        raise ValueError("Need both original and selected policies")
    matched, coverage = align_tracks(json.loads(json.dumps(raw_tracks, allow_nan=False)))
    if any(coverage["outside_common"][arm] for arm in ARMS):
        raise ValueError("Shape diagnostics must preserve complete same-match coverage")
    tracks = {arm: list(rows.values()) for arm, rows in matched.items()}
    stats = {}
    for arm, rows in tracks.items():
        for row in rows:
            _same(row["forecast"], summarize(score_grid(*row["rates"], rho=0)), "same_raw_rate_distribution")
            _same(row["probabilities"], list(row["forecast"]["one_x_two"].values()), "same_one_x_two")
        stats[arm] = [row_shape_statistics(row, row["rates"]) for row in rows]
    rows = tracks["original"]
    result = {**cohort(stats, list(range(len(rows)))), "coverage": coverage,
              "first_date": rows[0]["date"], "last_date": rows[-1]["date"], "bootstrap": {}}
    # One row per match. Every arm and every field shares the same resample.
    features = {f"{arm}_{field}": np.asarray([s[field] for s in stats[arm]])
                for arm in ARMS for field in DIAGNOSTICS}
    matrix = np.column_stack(list(features.values()))
    for weeks in BLOCKS:
        boot = moving_block_means([r["date"] for r in rows], matrix,
                                  block_weeks=weeks, repeats=repeats, seed=SEED)
        means, intervals = boot.pop("mean"), boot.pop("interval_95")
        boot["features"] = {key: {"mean": means[i], "interval_95": None if intervals is None else intervals[i]}
                            for i, key in enumerate(features)}
        result["bootstrap"][str(weeks)] = boot
    for name, labels in (("by_league", [r["competition"] for r in rows]),
                         ("by_month", [r["date"][:7] for r in rows])):
        result[name] = {label: cohort(stats, [i for i, value in enumerate(labels) if value == label])
                        for label in sorted(set(labels))}
    result["fixed_original_rate_bins"] = {}
    for side in ("home", "away"):
        labels = rate_bin_labels([s[f"{side}_rate"] for s in stats["original"]])
        result["fixed_original_rate_bins"][side] = {
            label: cohort(stats, [i for i, value in enumerate(labels) if value == label]) for label in BIN_NAMES}
    return result, tracks, stats


def write_csv(path, tracks, statistics):
    fields = ["match_id", "date", "competition", "home", "away"]
    fields += [f"{arm}_{field}" for arm in ARMS for field in STAT_FIELDS]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(tracks["original"]):
            item = {key: row[key] for key in fields[:5]}
            item.update({f"{arm}_{field}": statistics[arm][i][field] for arm in ARMS for field in STAT_FIELDS})
            writer.writerow(item)


def ci(value):
    return "不可估计" if value is None else f"[{value[0]:+.6f}, {value[1]:+.6f}]"


def report(result):
    selected = result["source_metadata"]["2024-25"]["selection"]
    lines = ["# 比分分布形态：与均值方向分开的诊断", "",
        "本轮0次训练、0条新预测，不更换模型，不把诊断残差变成胜率。"
        "复用上轮真实重训存档，对原默认参数和较早选定参数均单列。所有新文件及临时文件位于项目目录。", "",
        "## 本次新增的核心问题", "",
        "旧Pearson平方减一同时混入了均值偏差。本轮采用D=((Y−μ)²−Y)/μ，"
        "其一般条件期望为{Var(Y)−m+(m−μ)²}/μ，m为真实条件均值。"
        "若真实条件分布为Poisson，均值错配只贡献非负项(m−μ)²/μ。"
        "因此负的平均D比旧指标更值得检查，但抽样误差、筛选和自适应选参仍不能忽略。", "",
        "对六个事件保存原始残差I−p，另计算z=I−p−cH(YH−μH)−cA(YA−μA)，"
        "c=Cov参考模型(I,Y)/μ=∂p/∂μ。参考独立Poisson下E[z]=Cov(z,YH)=Cov(z,YA)=0。"
        "这是局部一阶正交，不是去掉全部均值错误；z可以超过[−1,1]，不是概率、校准率或修正后的期望场数。", "",
        "## 分季结果", "",
        f"原模型为180天／alpha0.01；selected为上轮较早选定{selected['half_life_days']}天／alpha{selected['regularization']}。"
        "同一赛季两臂使用完全相同比赛，主客与所有指标共享同一日历区块抽样。"]
    shown = ("home_dispersion_residual", "away_dispersion_residual", "zero_zero_residual", "one_one_residual",
             "zero_zero_mean_orthogonal_residual", "one_one_mean_orthogonal_residual", "draw_mean_orthogonal_residual",
             "total_four_plus_mean_orthogonal_residual")
    labels = dict(zip(shown, ("主队校正离散D", "客队校正离散D", "0:0原始残差", "1:1原始残差",
                              "0:0一阶投影残差", "1:1一阶投影残差", "平局一阶投影残差", "总进球≥4一阶投影残差")))
    for period, local in result["periods"].items():
        lines += ["", f"### {period}：{local['n']}场", "",
            "| 指标 | 原模型均值 | 原四周95%区间 | selected均值 | selected四周95%区间 |",
            "|---|---:|---|---:|---|"]
        features = local["bootstrap"]["4"]["features"]
        for field in shown:
            a, b = (features[f"{arm}_{field}"] for arm in ARMS)
            lines.append(f"| {labels[field]} | {a['mean']:+.6f} | {ci(a['interval_95'])} | {b['mean']:+.6f} | {ci(b['interval_95'])} |")
        lines += ["", "| 原模型事件 | 实际场数 | 预测期望场数 |", "|---|---:|---:|"]
        for event, item in local["event_counts"]["original"].items():
            lines.append(f"| {event} | {item['observed']} | {item['expected']:.3f} |")
    lines += ["", "## 对下一步结构研究的含义", "",
        "本轮不根据这些诊断直接更改胜率。下面的量均来自原模型，selected完整结果也已保留："]
    for period, local in result["periods"].items():
        f = local["bootstrap"]["4"]["features"]
        raw, projected = f["original_zero_zero_residual"], f["original_zero_zero_mean_orthogonal_residual"]
        draw = local["event_counts"]["original"]["draw"]
        lines.append(f"- {period}：实际平局{draw['observed']}场，预测期望{draw['expected']:.3f}场；"
            f"0:0原始残差{raw['mean']:+.6f}，一阶投影后{projected['mean']:+.6f}。"
            "投影后不能再按原始频率残差直接压低0:0。")
    lines += ["", "两季主、客校正离散D点估计均负；较晚主队D在2/4/8周区间均低于零，"
        "但较早主队及两季客队的区间证据不完全一致。平局投影后仍同向偏正，"
        "区间是否跨零随区块长度变化，不称跨季稳定确认。总进球≥4投影残差跨季换向，"
        "不能把所有尾部统一上调或下调。", "",
        "建议下一步先讨论一个最小分布对照：保持每场主客预期进球μ不变，只用较早赛季学习一个跨主客共享的CMP离散参数ν，"
        "在较晚赛季固定该参数，保留原Poisson、四领域同场评分及失败记录。"
        "这只是一条值得检验的路线，不等于已经实施或证明有效；不与新均值、球员先验、相关项或神经网络同时引入。"]
    lines += ["", "## 可以与不可以得出的结论", "",
        "这里并未拟合新分布，不能给出新模型胜率或宣称预测提升。"
        "D若偏负，只提示离散程度问题值得作保持均值的研究；并未证明CMP优于其他分布。"
        "事件z偏离零只提示比分形态相对均值方向仍有残差，不能自动归因于主客依赖或战术因素。"
        "centered cross product还包含两侧均值偏差乘积，不称真实条件协方差。", "",
        "两个赛季此前已多轮查看，本轮全属探索。selected在较早全季选出，较早结果有选择自适应，"
        "不能把该臂当全程预定政策；两季共享球队和训练历史，并非独立验证。"
        "不跨季合并区间、不把两臂当双倍样本，分季合计为"
        f"{result['counts']['unique_match_ids']}场不同比赛。", "",
        f"区块长度2/4/8周、各{result['settings']['bootstrap_repeats']}次、seed={SEED}。"
        "区间为点态探索性区间，非多指标联合检验；依赖已保存预测及所选政策，"
        "不包括重训、选参、方向选择或未来漂移的不确定性。非循环区块可能低估首尾周权重。"
        "所有17个指标、5联赛、逐月及固定原模型μ分箱的均值、SD、中位数、范围和空箱均保留于JSON，"
        "不按子组好坏筛选。", "",
        "实验设计技能使本轮固定同场队列、保留两臂和两季；统计分析技能使均值方向投影与原始事件频率分开，"
        "并保留共同日历区块、不作正态／独立样本显著性筛选。结构变更、CMP或深度学习训练需再讨论。", "",
        "公式从条件矩和参考Poisson pmf直接推导；计算接口依据"
        "[SciPy Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)、"
        "[SciPy Skellam](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html)；"
        "区块方法参考[连续移动区块](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html)。", ""]
    return "\n".join(lines)


def run(root, parent, output, *, repeats=REPEATS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {"source": str(parent), "bootstrap_blocks_weeks": list(BLOCKS), "bootstrap_repeats": repeats,
        "seed": SEED, "fit_new_model": False, "new_forecasts": False, "production_changed": False,
        "already_viewed_seasons": True, "primary_policy": "original", "secondary_policy": "selected"}
    write_json(output / "诊断设置.json", settings)
    periods, metadata, all_ids = {}, {}, set()
    for period in PERIODS:
        raw, metadata[period] = load_tracks(root, parent, period)
        periods[period], tracks, statistics = analyze(raw, repeats=repeats)
        ids = set(periods[period]["coverage"]["common_match_ids"])
        if all_ids & ids:
            raise ValueError("Different seasons must not repeat match IDs")
        all_ids |= ids
        write_csv(output / f"{period}_逐场形态.csv", tracks, statistics)
        write_json(output / f"{period}_形态结果.json", periods[period])
        print(f"{period}: {len(ids)} same-match diagnostics, zero fits", flush=True)
    result = {"settings": settings, "periods": periods, "source_metadata": metadata,
        "counts": {"unique_match_ids": len(all_ids), "new_fits": 0, "new_forecasts": 0},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    write_json(output / RESULT_FILE, result)
    (output / REPORT_FILE).write_text(report(result), encoding="utf-8")
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = args.parent or args.root / "outputs" / "核心参数训练_20261023T115640092598Z"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "outputs" / f"比分形态诊断_{stamp}"
    run(args.root, parent, output)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
