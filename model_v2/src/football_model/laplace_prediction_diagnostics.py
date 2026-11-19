"""Saved-prediction MAP-centre/Laplace-expansion bookkeeping, without fitting.

The player MAP counterfactual suppresses *all* log-rate covariance, including
unseen-player priors, solely to define an algebraic comparison.  It is not a
deployable unknown-player policy, a causal attribution, or a new validation.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from numbers import Integral, Real
from pathlib import Path

from .gaussian_goal_mixture import METRICS, score as mixture_score
from .statsbomb_data import write_json


PRIOR_SCALES = (.05, .10, .20)
MAIN_SCALE = .10
ZERO_COV = ((0., 0.), (0., 0.))
TRACKS = ("team_map", "team_laplace", "player_map_counterfactual", "player_laplace")
PARTS = ("total_difference", "centre_difference", "player_expansion", "team_expansion",
         "expansion_contrast", "reconstructed_total", "identity_residual")


def _finite(value, name):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Finite numeric {name} required")
    return float(value)


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Two entries required for {name}")
    return [_finite(item, name) for item in value]


def _covariance(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"A 2 by 2 matrix required for {name}")
    matrix = [_vector(row, name) for row in value]
    if matrix[0][0] < 0 or matrix[1][1] < 0:
        raise ValueError(f"Nonnegative log variances required for {name}")
    return matrix


def _metrics(row):
    return {metric: _finite(row[metric], metric) for metric in METRICS}


def _distribution(row):
    value = row["distribution"]
    if value.get("match_id") != row["match_id"] or value.get("match_date") != row["match_date"]:
        raise ValueError("Distribution identity/date differs from prediction row")
    mean = _vector(value["log_mean"], "log_mean")
    covariance = _covariance(value["log_cov"], "log_cov")
    known = _covariance(value["known_parameter_log_cov"], "known_parameter_log_cov")
    unknown = _covariance(value["unknown_player_log_cov"], "unknown_player_log_cov")
    if row.get("log_mean") != value["log_mean"] or row.get("log_cov") != value["log_cov"]:
        raise ValueError("Saved scored log distribution differs from source distribution")
    if any(not math.isclose(covariance[i][j], known[i][j] + unknown[i][j], rel_tol=1e-12, abs_tol=1e-12)
           for i in range(2) for j in range(2)):
        raise ValueError("Known and unknown covariance components do not sum to log_cov")
    return mean, covariance, known, unknown


def _index(rows, name, laplace=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Nonempty prediction list required for {name}")
    indexed = {}
    for row in rows:
        mid = row["match_id"]
        if not isinstance(mid, str) or not mid:
            raise ValueError("Nonempty match_id required")
        if mid in indexed:
            raise ValueError(f"Duplicate match_id in {name}: {mid}")
        if row.get("status") != "predicted":
            raise ValueError(f"Only saved predicted targets accepted in {name}")
        day = date.fromisoformat(row["match_date"])
        if day.isoformat() != row["match_date"]:
            raise ValueError("Canonical calendar match_date required")
        actual = row["actual"]
        if (not isinstance(actual, list) or len(actual) != 2
                or any(not isinstance(goal, Integral) or isinstance(goal, bool) or goal < 0 for goal in actual)):
            raise ValueError("Two nonnegative integer actual goals required")
        _metrics(row)
        _finite(row["forecast"]["expected_goals"]["total"], "saved integrated total rate")
        if laplace:
            if row.get("old_v36_cohort") not in ("known", "unknown"):
                raise ValueError("Fixed old_v36_cohort required")
            _distribution(row)
        indexed[mid] = row
    return indexed


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _rate_diagnostics(row):
    mean, covariance, known, unknown = _distribution(row)
    map_total = math.fsum(math.exp(value) for value in mean)
    integrated_total = float(row["forecast"]["expected_goals"]["total"])
    if not math.isfinite(map_total) or map_total <= 0 or integrated_total <= 0:
        raise ValueError("Positive finite goal-rate totals required")
    return {"map_total_rate": map_total, "integrated_total_rate": integrated_total,
            "actual_total_goals": sum(row["actual"]),
            "jensen_total_rate_increase": integrated_total - map_total,
            **{f"{kind}_log_variance_{side}": matrix[i][i]
               for kind, matrix in (("known", known), ("unknown", unknown), ("total", covariance))
               for i, side in enumerate(("home", "away"))}}


def _summarize(rows):
    result = {"n": len(rows), "match_ids": [row["match_id"] for row in rows],
              "mean_metrics": {track: {metric: _mean(row["metrics"][track][metric] for row in rows)
                                        for metric in METRICS} for track in TRACKS},
              "mean_decomposition": {metric: {part: _mean(row["decomposition"][metric][part] for row in rows)
                                               for part in PARTS} for metric in METRICS},
              "maximum_absolute_identity_residual": {
                  metric: max((abs(row["decomposition"][metric]["identity_residual"]) for row in rows), default=None)
                  for metric in METRICS}, "mean_rates_and_variances": {}}
    rate_keys = ("map_total_rate", "integrated_total_rate", "actual_total_goals", "jensen_total_rate_increase",
                 "known_log_variance_home", "known_log_variance_away", "unknown_log_variance_home",
                 "unknown_log_variance_away", "total_log_variance_home", "total_log_variance_away")
    for track in ("team", "players"):
        result["mean_rates_and_variances"][track] = {
            key: _mean(row["rates_and_variances"][track][key] for row in rows) for key in rate_keys}
    return result


def _cohorts(rows):
    return {"all": _summarize(rows),
            "old_known": _summarize([row for row in rows if row["old_v36_cohort"] == "known"]),
            "old_unknown": _summarize([row for row in rows if row["old_v36_cohort"] == "unknown"])}


def diagnose(team_laplace, team_map, player_tracks):
    """Return summaries and ledgers for all fixed scales, without changing inputs."""
    if set(player_tracks) != set(PRIOR_SCALES):
        raise ValueError("Exactly the three fixed prior scales .05/.10/.20 are required")
    team = _index(team_laplace, "team_laplace", laplace=True)
    point = _index(team_map, "team_map")
    players = {scale: _index(player_tracks[scale], f"players_s{scale:.2f}", laplace=True)
               for scale in PRIOR_SCALES}
    for name, indexed in (("team_map", point), *((f"players_s{scale:.2f}", players[scale]) for scale in PRIOR_SCALES)):
        if set(indexed) != set(team):
            raise ValueError(f"Target match set differs in {name}")
        for mid, row in indexed.items():
            if row["match_date"] != team[mid]["match_date"] or row["actual"] != team[mid]["actual"]:
                raise ValueError(f"Target date/actual differs in {name}: {mid}")
            if name != "team_map" and row["old_v36_cohort"] != team[mid]["old_v36_cohort"]:
                raise ValueError(f"Fixed original cohort differs in {name}: {mid}")
    for mid, row in point.items():
        if row.get("log_mean") != team[mid]["log_mean"] or row.get("log_cov") != [[0., 0.], [0., 0.]]:
            raise ValueError("Saved team MAP must use the same centre and zero covariance")
    ids = sorted(team, key=lambda mid: (team[mid]["match_date"], mid))
    summaries, ledgers = {}, {}
    for scale in PRIOR_SCALES:
        ledger = []
        for mid in ids:
            t_lap, t_map, p_lap = team[mid], point[mid], players[scale][mid]
            # This intentionally suppresses unknown-player priors only in the
            # counterfactual ledger, never in an operational prediction policy.
            p_map = mixture_score(p_lap["log_mean"], ZERO_COV, p_lap["actual"], order=1)
            metrics = {"team_map": _metrics(t_map), "team_laplace": _metrics(t_lap),
                       "player_map_counterfactual": _metrics(p_map), "player_laplace": _metrics(p_lap)}
            decomposition = {}
            for metric in METRICS:
                total = p_lap[metric] - t_lap[metric]
                centre = p_map[metric] - t_map[metric]
                player_expansion = p_lap[metric] - p_map[metric]
                team_expansion = t_lap[metric] - t_map[metric]
                contrast = player_expansion - team_expansion
                reconstructed = centre + contrast
                decomposition[metric] = dict(zip(PARTS, (total, centre, player_expansion, team_expansion,
                                                          contrast, reconstructed, total - reconstructed)))
            day = date.fromisoformat(t_lap["match_date"])
            ledger.append({"match_id": mid, "match_date": day.isoformat(),
                           "week_start": (day - timedelta(days=day.weekday())).isoformat(),
                           "old_v36_cohort": t_lap["old_v36_cohort"], "actual": list(t_lap["actual"]),
                           "metrics": metrics, "decomposition": decomposition,
                           "rates_and_variances": {"team": _rate_diagnostics(t_lap), "players": _rate_diagnostics(p_lap)}})
        key = f"s{scale:.2f}"
        ledgers[key] = ledger
        summaries[key] = {"cohorts": _cohorts(ledger),
                          "weeks": {week: _cohorts([row for row in ledger if row["week_start"] == week])
                                    for week in sorted({row["week_start"] for row in ledger})}}
    result = {"status": "complete", "main_prior_sd": MAIN_SCALE, "fixed_prior_scales": list(PRIOR_SCALES),
              "target_matches": len(ids), "weeks": len(summaries["s0.10"]["weeks"]),
              "model_fits": 0, "hyperparameter_searches": 0, "new_prospective_predictions": 0,
              "laplace_metrics": "read directly from saved predictions, not recomputed",
              "player_map_counterfactual": "same saved log mean, all covariance zero, mixture_score order=1; bookkeeping only",
              "identity": "P_lap - T_lap = (P_map - T_map) + [(P_lap - P_map) - (T_lap - T_map)]",
              "aggregation": "equal match means; Monday calendar weeks; fixed original cohorts; empty means null",
              "rates": "MAP total is sum(exp(log_mean)); integrated total read from saved forecast; Jensen increase is their difference",
              "limits": ["algebraic decomposition, not causal attribution or independent validation",
                         "unknown-player zero-covariance counterfactual is not an operational zero-effect policy",
                         "fixed .10 main; all three scales retained, no winner or variance/prior adjustment",
                         "no refitting, new prospective prediction, posterior recalibration, significance or promotion"],
              "scales": summaries}
    return result, ledgers


def make_report(result):
    def fmt(value):
        return "未知" if value is None else f"{value:.6f}"
    main = result["scales"]["s0.10"]["cohorts"]
    lines = ["# MAP中心与Laplace扩张：已存预测的代数诊断", "",
             f"同场{result['target_matches']}场、{result['weeks']}个周；0次拟合、0次调参、0条新前瞻。"
             "s=.10为预先声明主尺度，.05/.20全部保留，不按结果选赢家。", "",
             "总差=P_lap−T_lap=中心差(P_map−T_map)＋扩张对比[(P_lap−P_map)−(T_lap−T_map)]。"
             "Laplace指标直接读取存档；球员MAP反事实用同一log中心、零协方差及1阶积分评分。", "",
             "## 主尺度：全部同场", "",
             "| 指标 | 总差 | 中心差 | 扩张对比 | 最大恒等式残差 |", "|---|---:|---:|---:|---:|"]
    for metric in METRICS:
        values = main["all"]["mean_decomposition"][metric]
        lines.append(f"| {metric} | {fmt(values['total_difference'])} | {fmt(values['centre_difference'])} | "
                     f"{fmt(values['expansion_contrast'])} | {main['all']['maximum_absolute_identity_residual'][metric]:.3g} |")
    lines += ["", "除top1越大越好外，其余损失越低越好；扩张可正可负，不预设其改善预测。", "",
              "## 三尺度与固定原分组：完整比分NLL", "",
              "| s | 分组 | 场数 | 总差 | 中心差 | 扩张对比 |", "|---|---|---:|---:|---:|---:|"]
    for scale in PRIOR_SCALES:
        for cohort, label in (("all", "全部"), ("old_known", "原已知"), ("old_unknown", "原未知")):
            group = result["scales"][f"s{scale:.2f}"]["cohorts"][cohort]
            values = group["mean_decomposition"]["full_score_nll"]
            lines.append(f"| {scale:.2f} | {label} | {group['n']} | {fmt(values['total_difference'])} | "
                         f"{fmt(values['centre_difference'])} | {fmt(values['expansion_contrast'])} |")
    lines += ["", "## 主尺度的均值与log方差：全部同场", "",
              "| 轨道 | MAP总率 | 积分总率 | 实际总球 | Jensen抬升 | 已知log方差(主/客) | 未知log方差(主/客) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for track, label in (("team", "球队"), ("players", "角色＋首发")):
        rates = main["all"]["mean_rates_and_variances"][track]
        lines.append(f"| {label} | {fmt(rates['map_total_rate'])} | {fmt(rates['integrated_total_rate'])} | "
                     f"{fmt(rates['actual_total_goals'])} | {fmt(rates['jensen_total_rate_increase'])} | "
                     f"{fmt(rates['known_log_variance_home'])}/{fmt(rates['known_log_variance_away'])} | "
                     f"{fmt(rates['unknown_log_variance_home'])}/{fmt(rates['unknown_log_variance_away'])} |")
    lines += ["", "全部8指标、三尺度、原已知／未知与按周描述汇总和逐场残差保存在JSON中；"
              "均按比赛等权，不把周、多个指标或多个尺度当独立实验。", "",
              "这是代数分解，不是因果归因或独立验证。MAP反事实连未知球员先验协方差也暂时关闭，"
              "只用于账本，不表示未知球员已知零效应，更不是上线建议。Jensen均值抬升也不能单独归因为损失变化原因。", "",
              "不据本结果修改方差、先验、模型结构或正式预测；不拟合，不生成新前瞻，"
              "不宣称Laplace后验已校准或预测获得提升。原来源许可与首发事后编码等限制继续适用；"
              "全部产物在G盘。", ""]
    return "\n".join(lines)


def run(root, research):
    root, research = Path(root).resolve(), Path(research).resolve()
    names = {"team_laplace": "team_laplace_predictions.json", "team_map": "team_map_predictions.json",
             **{f"players_s{scale:.2f}": f"players_s{scale:.2f}_predictions.json" for scale in PRIOR_SCALES}}
    saved = {key: json.loads((research / name).read_text(encoding="utf-8")) for key, name in names.items()}
    result, ledgers = diagnose(saved["team_laplace"], saved["team_map"],
                               {scale: saved[f"players_s{scale:.2f}"] for scale in PRIOR_SCALES})
    policy_path = research / "固定研究设置.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    actual_counts = {"target_matches": result["target_matches"],
                     "old_known": result["scales"]["s0.10"]["cohorts"]["old_known"]["n"],
                     "old_unknown": result["scales"]["s0.10"]["cohorts"]["old_unknown"]["n"]}
    if any(policy.get(key) != count for key, count in actual_counts.items()):
        raise ValueError("Saved targets/cohort counts differ from the original fixed research settings")
    result.update(created_at_utc=datetime.now(timezone.utc).isoformat(), research_directory=str(research),
                  sources={**{key: str(research / name) for key, name in names.items()},
                           "fixed_research_settings": str(policy_path)})
    output = root / "outputs" / datetime.now(timezone.utc).strftime("Laplace中心扩张诊断_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "诊断结果.json", result)
    for key, ledger in ledgers.items():
        write_json(output / f"players_{key}_逐场分解.json", ledger)
    (output / "中心扩张诊断说明.md").write_text(make_report(result), encoding="utf-8")
    print(f"Saved-prediction diagnostic complete, 0 fits: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--research", required=True)
    args = parser.parse_args()
    run(args.root, args.research)


if __name__ == "__main__":
    main()
