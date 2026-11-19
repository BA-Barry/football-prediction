"""Fixed weekly crossed-role/player Laplace research, not a live model update.

All three prior scales are reported as sensitivity, never ranked to select one.
The main scale is declared before fitting. Both tracks integrate parameter
uncertainty and use exactly the same earlier matches and held-forward fixtures.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

from .english_tiers_experiment import METRICS, summarize_rows
from .gaussian_goal_mixture import score as mixture_score
from .hierarchical_xi_model import HierarchicalStartingXIModel
from .impect_player_research import TARGET_FROM, TARGET_THROUGH, weekly_windows
from .starting_xi_research import FIXTURE_FIELDS
from .statsbomb_data import write_json


PRIOR_SCALES = (.05, .10, .20)
MAIN_SCALE = .10
PRIMARY_ORDER, CHECK_ORDER = 15, 25


def fixture_only(match):
    fixture = {key: list(match[key]) if key.endswith("starters") else match[key] for key in FIXTURE_FIELDS}
    fixture.update({f"{side}_roles": dict(match[f"{side}_roles"]) for side in ("home", "away")})
    return fixture


def fixed_policy():
    return {
        "question": "opening XI AND initial roles known, complete full-time goals",
        "target_from": TARGET_FROM.isoformat(), "target_through": TARGET_THROUGH.isoformat(),
        "history": "same expanding histories; Monday cutoff, strictly earlier match_date; no within-week refits",
        "main_prior_sd": MAIN_SCALE, "sensitivity_prior_sd": list(PRIOR_SCALES),
        "prior_scale_selection": "main .10 fixed externally from older ridge100 log-scale anchor; .05/.20 sensitivity only, no winner selection",
        "components": "independent role attack/defence and individual residual attack/defence normals, each variance s^2/2; fixed not learned",
        "centering": "current role counts and player presence minus the same team's earlier per-match references",
        "team_prior_precision": 1.0, "home_prior_precision": 1.0, "intercept_prior": "flat; positive historical goals required",
        "target": "summed complete full-match Poisson likelihood including normal stoppage; no exposure offset",
        "unknown_players": "integrate independent residual priors; prior mean zero is not known zero strength; k total unknown gives k*s^2/2 on BOTH log-rate variances",
        "unknown_teams": "unsupported, explicit failure; no new-team prior invented",
        "posterior": "full observed Hessian at joint MAP, Laplace normal; not MCMC or exact/full Bayes; no learned hypervariance",
        "prediction": "integrate conditional joint Poisson probabilities, not Poisson(E[lambda]); full home/away covariance retained",
        "quadrature": {"primary_order": PRIMARY_ORDER, "fixed_check_order": CHECK_ORDER,
                       "selection": "numerical error measurement only, not outcome-based order selection"},
        "tracks": "matched team-only Laplace, crossed-role/player Laplace at every fixed scale, team-only MAP as secondary",
        "primary_comparison": "main .10 versus team-only Laplace on all same supported target fixtures; full-score NLL lower is better",
        "cohorts": "all targets and old v36-known/unknown subsets are fixed before new fits; not new independent experiments",
        "searches": 0, "max_iterations": 2000,
        "limits": "already-viewed single season; opening-role labels may be posthoc, no historical announcement timestamps, no causal or live-2026 claim",
        "sampling_uncertainty": "descriptive weekly/paired results only; overlapping histories and teams not IID",
        "promotion": "none; old models and saved prospective cases unchanged",
        "source_terms": "IMPECT custom local personal noncommercial terms, attribution/logo preserved; no original or derived observations redistribution",
    }


def paired_comparison(team, players, ids=None):
    left = {row["match_id"]: row for row in team}
    right = {row["match_id"]: row for row in players}
    common = left.keys() & right.keys()
    if ids is not None:
        common &= set(ids)
    common = sorted(common, key=lambda mid: (left[mid]["match_date"], mid))
    differences = [{"match_id": mid, **{key: right[mid][key]-left[mid][key] for key in METRICS}}
                   for mid in common]
    return {"common_matches": len(common), "common_match_ids": common,
            "team": summarize_rows([left[mid] for mid in common]),
            "players": summarize_rows([right[mid] for mid in common]),
            "mean_players_minus_team": {key: statistics.mean(row[key] for row in differences) if differences else None
                                        for key in METRICS},
            "paired_matches": differences}


def _numerical_difference(primary, check):
    differences = {key: abs(primary[key]-check[key]) for key in METRICS}
    for category in ("one_x_two", "total_goals", "away_minus_two_jincai"):
        differences[f"{category}_probability"] = max(
            abs(value-check["forecast"][category][key])
            for key, value in primary["forecast"][category].items())
    return differences


def score_targets(model, targets, original_known):
    rows, errors = [], []
    for match in targets:
        distribution = model.predict_distribution(fixture_only(match))
        # Completed targets only enter the separate proper-scoring function.
        primary = mixture_score(distribution["log_mean"], distribution["log_cov"], match["full_time"], order=PRIMARY_ORDER)
        check = mixture_score(distribution["log_mean"], distribution["log_cov"], match["full_time"], order=CHECK_ORDER)
        errors.append({"match_id": match["match_id"], **_numerical_difference(primary, check)})
        rows.append({"match_id": match["match_id"], "match_date": match["match_date"], "status": "predicted",
                     "old_v36_cohort": "known" if match["match_id"] in original_known else "unknown",
                     "actual": list(match["full_time"]), "distribution": distribution, **primary})
    return rows, errors


def _cohorts(team, players, original_known):
    return {"all": paired_comparison(team, players),
            "old_known": paired_comparison(team, players, original_known),
            "old_unknown": paired_comparison(team, players, {r["match_id"] for r in team} - original_known)}


def run(root, source, inspection, old_research):
    root, source, inspection, old_research = (Path(p).resolve() for p in (root, source, inspection, old_research))
    matches, source_inspection = json.loads(source.read_text(encoding="utf-8")), json.loads(inspection.read_text(encoding="utf-8"))
    windows = list(weekly_windows(matches))
    old_players = json.loads((old_research / "full_match_players_predictions.json").read_text(encoding="utf-8"))
    old_team = json.loads((old_research / "full_match_team_predictions.json").read_text(encoding="utf-8"))
    original_known = {row["match_id"] for row in old_players if row["status"] == "predicted"}
    target_ids = {match["match_id"] for _, _, targets in windows for match in targets}
    if target_ids != {row["match_id"] for row in old_players} or target_ids != {row["match_id"] for row in old_team}:
        raise ValueError("Role-aware targets must match all original v36 fixtures, not a new selected subset")
    output = root / "outputs" / datetime.now(timezone.utc).strftime("分层首发固定前推研究_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    policy = {**fixed_policy(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "inspection": str(inspection), "old_research": str(old_research),
              "source_matches": len(matches), "target_matches": len(target_ids),
              "old_known": len(original_known), "old_unknown": len(target_ids-original_known)}
    write_json(output / "固定研究设置.json", policy)
    write_json(output / "源检验说明副本.json", source_inspection)
    tracks = {"team_laplace": [], "team_map": [], **{f"players_s{sd:.2f}": [] for sd in PRIOR_SCALES}}
    errors, weeks, fits = {}, [], 0
    try:
        for cutoff, history, targets in windows:
            print(f"Hierarchical XI cutoff {cutoff}: history={len(history)}, target={len(targets)}", flush=True)
            week = {"week_start": cutoff.isoformat(), "history_matches": len(history),
                    "history_min_date": history[0]["match_date"], "history_max_date": history[-1]["match_date"],
                    "target_match_ids": [match["match_id"] for match in targets], "models": {}}
            local = {}
            specifications = [("team_laplace", False, MAIN_SCALE)] + [(f"players_s{sd:.2f}", True, sd) for sd in PRIOR_SCALES]
            for kind, include_players, sd in specifications:
                print(f"  fitting {kind}", flush=True)
                model = HierarchicalStartingXIModel(include_players=include_players, prior_sd=sd).fit(history, cutoff)
                fits += 1
                path = output / "models" / f"{cutoff}_{kind}.json"
                model.save(path)
                restored = HierarchicalStartingXIModel.load(path)
                if model.parameters() != restored.parameters() or not np.array_equal(model.precision_cholesky, restored.precision_cholesky):
                    raise ValueError("Saved model/factor did not preserve the fitted state")
                scored, numerical = score_targets(restored, targets, original_known)
                if include_players:
                    for row in scored:
                        if bool(row["distribution"]["unknown_player_ids"]) != (row["old_v36_cohort"] == "unknown"):
                            raise ValueError("New unknown identities disagree with fixed old v36 coverage cohort")
                local[kind] = scored
                tracks[kind].extend(scored)
                errors.setdefault(kind, []).extend(numerical)
                week["models"][kind] = {"model_path": str(path), "optimizer": restored.optimizer_diagnostics,
                                         "prior_sd": sd, "supported_matches": len(scored),
                                         "parameters_saved_and_restored_before_scoring": True}
                write_json(output / f"{kind}_predictions.json", tracks[kind])
                if not include_players:
                    for match, predicted in zip(targets, scored):
                        distribution = predicted["distribution"]
                        point = mixture_score(distribution["log_mean"], [[0.,0.],[0.,0.]], match["full_time"], order=1)
                        tracks["team_map"].append({"match_id": match["match_id"], "match_date": match["match_date"],
                                                   "status": "predicted", "actual": list(match["full_time"]), **point})
                    write_json(output / "team_map_predictions.json", tracks["team_map"])
            week["comparisons"] = {f"s{sd:.2f}": _cohorts(local["team_laplace"], local[f"players_s{sd:.2f}"], original_known)
                                   for sd in PRIOR_SCALES}
            weeks.append(week)
            write_json(output / "逐周拟合与比较.json", weeks)
        numerical_summary = {kind: {key: max(row[key] for row in rows) for key in rows[0] if key != "match_id"}
                             for kind, rows in errors.items()}
        old = {row["match_id"]: row for row in old_team}
        point_reconstruction = {key: max(abs(row[key]-old[row["match_id"]][key]) for row in tracks["team_map"]) for key in METRICS}
        result = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "weeks": len(weeks), "model_fits_completed": fits, "source_matches": len(matches),
                  "target_matches": len(target_ids), "policy": policy,
                  "track_summaries": {key: summarize_rows(rows) for key, rows in tracks.items()},
                  "comparisons": {f"s{sd:.2f}": _cohorts(tracks["team_laplace"], tracks[f"players_s{sd:.2f}"], original_known)
                                  for sd in PRIOR_SCALES},
                  "quadrature_max_absolute_difference_15_vs_25": numerical_summary,
                  "team_map_reconstruction_max_absolute_metric_difference_from_v36": point_reconstruction,
                  "hyperparameter_searches": 0,
                  "limits": "fixed-scale crossed components; local Laplace and finite quadrature; no posterior calibration validation or live promotion"}
        write_json(output / "数值积分逐场误差.json", errors)
        write_json(output / "分层首发前推结果.json", result)
        (output / "分层首发前推说明.md").write_text(make_report(result), encoding="utf-8")
        print(f"Hierarchical XI research complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"status": "incomplete", "completed_fits": fits,
                                                  "completed_weeks": len(weeks), "error": f"{type(error).__name__}: {error}"})
        raise


def make_report(result):
    def fmt(value):
        return "未知" if value is None else f"{value:.6f}"
    main = result["comparisons"][f"s{MAIN_SCALE:.2f}"]
    lines = ["# 分层首发与未知球员：固定历史前推", "",
             "来源：[IMPECT Open Data](https://github.com/ImpectAPI/open-data)开场位置／首发，独立Football-Data全场进球。"
             "仅本地个人非商业研究，保留原署名logo，不再分发原始或派生观测。", "",
             f"接通{result['source_matches']}场，目标{result['target_matches']}场；{result['weeks']}周、"
             f"{result['model_fits_completed']}次拟合。预先声明主尺度s=.10，.05/.20只作先验敏感性，全部报告、不按成绩选赢家。", "",
             "角色组与个人残差的攻防先验各独立N(0,s²/2)，角色人数及个人首发相对本队较早训练均值中心化。"
             "这是固定方差交叉加性共享信息，不是从数据学到方差或个人因果价值。GK人数恒定时角色项不含证据。", "",
             "已知参数采用MAP处完整Hessian的Laplace近似；未知个人保留残差先验，双方合计k个未知会给双方对数进球率各加k·s²/2方差。"
             "零先验均值不等于已知零实力。对相关的两维对数进球率积分联合比分概率，不能拿平均进球率再套Poisson。", "",
             "## 预先声明主比较：s=.10与球队Laplace", "",
             "| 全部同场指标 | 球队Laplace | 角色＋首发Laplace | 后者−前者 |", "|---|---:|---:|---:|"]
    for key in METRICS:
        lines.append(f"| {key} | {fmt(main['all']['team'][key])} | {fmt(main['all']['players'][key])} | {fmt(main['all']['mean_players_minus_team'][key])} |")
    lines += ["", "完整比分NLL是主指标，越低越好；top1越高越好，其余损失越低越好。覆盖满并不等于效果改善。", "",
              "## 先验敏感性：不据此选尺度", "",
              "| s | 全部场数 | 全部平均比分NLL差 | 原已知子集场数／差 | 原未知子集场数／差 |", "|---|---:|---:|---:|---:|"]
    for sd in PRIOR_SCALES:
        c = result["comparisons"][f"s{sd:.2f}"]
        lines.append(f"| {sd:.2f} | {c['all']['common_matches']} | {fmt(c['all']['mean_players_minus_team']['full_score_nll'])} | "
                     f"{c['old_known']['common_matches']} / {fmt(c['old_known']['mean_players_minus_team']['full_score_nll'])} | "
                     f"{c['old_unknown']['common_matches']} / {fmt(c['old_unknown']['mean_players_minus_team']['full_score_nll'])} |")
    lines += ["", "原已知／未知是v36在新拟合前确定的固定子集，不能拼成独立实验，也不能直接互比平均值归因。"
              ".10来自旧研究尺度锚点，不是本目标验证最优；同位置替换的共享角色项可能抵消，未知个人新增方差并非s²。", "",
              "## 数值与使用限制", "",
              "每个模型先存参数和精度Cholesky，再载入评分。主积分阶数15与固定25阶差异逐场测量，"
              "检查的是数值积分误差，不是Laplace近似相对真实后验的误差。完整误差表与旧球队MAP重建差异见结果JSON。", "",
              "已查看单赛季、日期级赛果可用假设、首发及开场角色可能事后编码，缺原始公布时间；"
              "这是首发与角色已知的探索性条件研究，不是真实T−60或2026当前球员预测。"
              "不作显著性、稳定优势、后验覆盖已校准或保证中奖结论。没有新增深度学习、MCMC或学习超方差。", "",
              "原模型和9场前瞻保持不变，本轮不自动上线。实验设计技能限定同历史／同场及预先声明敏感性，"
              "统计分析技能使覆盖、近似、负结果与所有尺度如实报告。全部代码、模型、缓存与结果在项目目录。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--old-research", required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.inspection, args.old_research)


if __name__ == "__main__":
    main()
