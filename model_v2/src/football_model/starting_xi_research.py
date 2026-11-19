"""Fixed weekly full-match Poisson comparison, conditional on a known opening XI.

No source download, target-minute features, parameter search or live promotion.
This is a new full-score question, not the old path-conditional intensity loss.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .english_tiers_experiment import METRICS, score_forecast, summarize_rows
from .impect_player_research import TARGET_FROM, TARGET_THROUGH, weekly_windows
from .starting_xi_model import FullMatchStartingXIModel, UnknownStartingXIIdentityError
from .statsbomb_data import write_json


FIXTURE_FIELDS = ("match_id", "match_date", "home_team_id", "away_team_id",
                  "home_starters", "away_starters")


def fixture_only(match):
    """Create the prediction boundary without results or realized match paths."""
    return {key: list(match[key]) if key.endswith("starters") else match[key]
            for key in FIXTURE_FIELDS}


def fixed_policy():
    return {
        "question": "full-time goals conditional on opening XI being known",
        "target_from": TARGET_FROM.isoformat(), "target_through": TARGET_THROUGH.isoformat(),
        "cutoff": "calendar Monday; match_date strictly before cutoff; no same-week refits",
        "model": "joint full-match attack/defence Poisson; independent home/away conditional goals",
        "loss": "summed two-sided full-match Poisson NLL + 0.5 * sum(ridge * coefficient^2)",
        "penalties": {"intercept": 0.0, "home": 1.0, "team": 1.0, "players": 100.0},
        "penalty_origin": "fixed external 2015/16 EPL summed-loss research anchor, not an equivalent or optimal Bundesliga penalty",
        "max_iterations": 2000, "parameter_searches": 0,
        "history": "same expanding match set for both tracks; no decay or target-season tuning",
        "reference": "team starting frequencies from each training history only; no minute exposure",
        "target": "independent Football-Data full-time goals including normal stoppage time; no duration offset",
        "features": list(FIXTURE_FIELDS),
        "excluded_features": ["full_time", "actual minutes", "substitutions", "score paths", "red-card paths", "future registrations"],
        "unknown_identity_policy": "whole player-track fixture skipped; no zero imputation or team fallback; team track accepts new players",
        "primary_metric": "mean full_score_nll on common supported matches, lower is better",
        "secondary_metrics": list(METRICS[1:]),
        "sampling_uncertainty": "descriptive paired/weekly differences only; repeated teams and adjacent windows are not IID; no p-value or confidence claim",
        "availability": "date-level historical result availability assumed; original XI publication times missing; not a reconstructed T-60 test",
        "status": "exploratory development: this season has already been viewed, not a new blind test",
        "source_terms": "IMPECT custom terms; local personal noncommercial research, no original or derived data redistribution",
        "promotion": "none; preserve old models and nine saved prospective forecasts",
    }


def score_matches(model, targets):
    rows = []
    for match in targets:
        fixture = fixture_only(match)
        base = {"match_id": match["match_id"], "match_date": match["match_date"]}
        try:
            predicted = model.predict_match(fixture)
        except UnknownStartingXIIdentityError as error:
            rows.append({**base, "status": "skipped", "reason": str(error)})
            continue
        means = [predicted["home_mean"], predicted["away_mean"]]
        # Only the scoring function receives the target after prediction.
        score = score_forecast(means, match["full_time"])
        rows.append({**base, "status": "predicted", "means": means,
                     "actual": list(match["full_time"]), **score})
    return rows


def coverage(rows):
    n = sum(row["status"] == "predicted" for row in rows)
    return {"requested_matches": len(rows), "supported_matches": n,
            "skipped_matches": len(rows) - n,
            "supported_fraction": n / len(rows) if rows else None,
            "summary_on_supported": summarize_rows(rows),
            "skipped": [row for row in rows if row["status"] == "skipped"]}


def compare(team_rows, player_rows):
    """Compare common identities, never all-team against the known-player subset."""
    team = {r["match_id"]: r for r in team_rows if r["status"] == "predicted"}
    players = {r["match_id"]: r for r in player_rows if r["status"] == "predicted"}
    common = sorted(team.keys() & players.keys(), key=lambda mid: (team[mid]["match_date"], mid))
    deltas = [{"match_id": mid, "match_date": team[mid]["match_date"],
               **{metric: players[mid][metric] - team[mid][metric] for metric in METRICS}}
              for mid in common]
    values = [r["full_score_nll"] for r in deltas]
    return {
        "common_match_ids": common, "common_matches": len(common),
        "team_on_common": summarize_rows([team[mid] for mid in common]),
        "players_on_common": summarize_rows([players[mid] for mid in common]),
        "mean_players_minus_team": {metric: statistics.mean(r[metric] for r in deltas) if deltas else None
                                    for metric in METRICS},
        "paired_full_score_delta_descriptive": {
            "n": len(values), "median": statistics.median(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "note": "match-level descriptive spread, not an IID standard error or confidence interval",
        },
        "paired_matches": deltas,
    }


def source_denominators(inspection, matches):
    target_n = sum(TARGET_FROM.isoformat() <= m["match_date"] <= TARGET_THROUGH.isoformat() for m in matches)
    expected = inspection.get("expected_target_matches")
    return {"expected_catalogue_matches": inspection.get("expected_catalogue_matches"),
            "expected_target_matches": expected, "eligible_matches": len(matches),
            "eligible_target_matches": target_n,
            "source_ineligible_target_matches": expected - target_n if type(expected) is int else None,
            "note": "source exclusions and model unknown-identity skips are separate denominators"}


def run(root, matches_path, inspection_path):
    root, matches_path, inspection_path = (Path(p).resolve() for p in (root, matches_path, inspection_path))
    matches = json.loads(matches_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    if not isinstance(inspection, dict):
        raise ValueError("Saved source inspection must be an object")
    windows = list(weekly_windows(matches))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"首发全场固定前推研究_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    denominators = source_denominators(inspection, matches)
    policy = {**fixed_policy(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "matches_input": str(matches_path), "inspection_input": str(inspection_path),
              "source_denominators": denominators,
              "first_window_history_matches": len(windows[0][1])}
    write_json(output / "固定研究设置.json", policy)
    write_json(output / "源检验说明副本.json", inspection)
    totals, weeks, fits = {"team": [], "players": []}, [], 0
    try:
        for cutoff, history, targets in windows:
            print(f"Starting XI cutoff {cutoff}: history={len(history)}, target={len(targets)}", flush=True)
            week = {"week_start": cutoff.isoformat(), "history_matches": len(history),
                    "history_min_date": history[0]["match_date"], "history_max_date": history[-1]["match_date"],
                    "target_match_ids": [m["match_id"] for m in targets], "tracks": {}}
            scores = {}
            for kind, include_players in (("team", False), ("players", True)):
                model = FullMatchStartingXIModel(include_players=include_players).fit(history, cutoff)
                fits += 1
                path = output / "models" / f"{cutoff}_{kind}.json"
                model.save(path)
                restored = FullMatchStartingXIModel.load(path)
                if restored.parameters() != model.parameters():
                    raise ValueError("Saved model did not preserve parameters")
                scores[kind] = score_matches(restored, targets)
                totals[kind].extend(scores[kind])
                week["tracks"][kind] = {**coverage(scores[kind]), "model_path": str(path),
                                         "fitted_players": len(restored.players),
                                         "optimizer": restored.optimizer_diagnostics,
                                         "parameters_saved_and_restored_before_scoring": True}
                write_json(output / f"full_match_{kind}_predictions.json", totals[kind])
            week["comparison"] = compare(scores["team"], scores["players"])
            week["cumulative_comparison"] = compare(totals["team"], totals["players"])
            weeks.append(week)
            write_json(output / "逐周拟合与比较.json", weeks)
        result = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "model_fits_completed": fits, "weeks": len(weeks),
                  "coverage": {key: coverage(rows) for key, rows in totals.items()},
                  "comparison": compare(totals["team"], totals["players"]),
                  "weekly_primary_delta": [{"week_start": week["week_start"],
                                             "common_matches": week["comparison"]["common_matches"],
                                             "players_minus_team": week["comparison"]["mean_players_minus_team"]["full_score_nll"]}
                                            for week in weeks],
                  "source_denominators": denominators, "policy": policy}
        write_json(output / "首发全场前推结果.json", result)
        (output / "首发全场前推说明.md").write_text(make_report(result), encoding="utf-8")
        print(f"Starting XI research complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"status": "incomplete", "completed_fits": fits,
                                                  "completed_weeks": len(weeks),
                                                  "error": f"{type(error).__name__}: {error}"})
        raise


def make_report(result):
    c, source = result["comparison"], result["source_denominators"]
    def fmt(value):
        return "未知" if value is None else f"{value:.6f}"
    lines = ["# 已知首发→全场比分：固定每周前推研究", "",
             "数据：[IMPECT Open Data](https://github.com/ImpectAPI/open-data)，2023/24德甲首发；"
             "独立Football-Data全场进球。仅本地个人非商业研究，不再分发原始或派生观测。", "",
             f"源目录{source['expected_catalogue_matches']}场，接通{source['eligible_matches']}场；"
             f"原目标{source['expected_target_matches']}场，可用目标{source['eligible_target_matches']}场。"
             f"完成{result['weeks']}周、{result['model_fits_completed']}次拟合，搜索0次。", "",
             "仅用当周一之前的比赛扩窗训练；两条线相同历史、相同全场Poisson损失和球队结构。"
             "球队／主场ridge=1，球员ridge=100，截距不罚；目标是求和NLL，非平均loss。"
             "固定外部旧研究惩罚仅作锚点，不是本赛季最优选择。参数先保存、再载入预测。", "",
             "输入仅开场11人和球队历史，不输入本场分钟、换人、红牌、比分路径或终哨时长；"
             "全场进球含补时，无90分钟暴露offset。首发效应不是11人踢满全场的因果价值。", "",
             f"球队线覆盖{result['coverage']['team']['supported_matches']}/{result['coverage']['team']['requested_matches']}；"
             f"首发线覆盖{result['coverage']['players']['supported_matches']}/{result['coverage']['players']['requested_matches']}；"
             f"未知身份整场跳过，共同{c['common_matches']}场。球队全覆盖成绩不能与首发子集直接比较。", "",
             "| 同一批比赛指标 | 仅球队 | 球队＋首发 | 首发−球队 |",
             "|---|---:|---:|---:|"]
    for metric in METRICS:
        lines.append(f"| {metric} | {fmt(c['team_on_common'][metric])} | "
                     f"{fmt(c['players_on_common'][metric])} | {fmt(c['mean_players_minus_team'][metric])} |")
    lines += ["", "主指标完整比分NLL越低越好；除top1越高越好外，表内损失均越低越好。"
              "共同覆盖为零时损失差为未知，不填零。", "",
              "这是已查看赛季上的探索性条件研究，不是新的盲测。缺历史首发首次公布时间，"
              "不能声称真实T−60前瞻；重复球队、相邻周和未知球员筛选带来不确定性，"
              "不作显著性或稳定提升结论，不用本轮结果继续选参数或提前停止。", "",
              "本轮完整比分NLL不能与旧条件点过程loss直接横比。没有学习独立红牌效应、"
              "赛事重视程度或当前2026球员能力；保持原模型与9场已保存前瞻不变，不上线。", "",
              "实验设计技能用于同场／同历史／固定参数比较；统计分析技能用于完整分母、"
              "负结果与描述性不确定性报告。所有代码、模型、缓存和结果仍在项目目录。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--inspection", required=True)
    args = parser.parse_args()
    run(args.root, args.matches, args.inspection)


if __name__ == "__main__":
    main()
