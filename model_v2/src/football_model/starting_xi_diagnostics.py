"""Read-only diagnostics of archived starting-XI fits; no training or tuning.

Training resubstitution scores are descriptive, not out-of-sample accuracy.
Joint-team and XI log terms are separated: the joint model re-estimates team
effects, so its total change cannot be attributed to the XI term alone.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from .statsbomb_data import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def rate_parts(payload, match):
    """Recombine readable saved coefficients, for historical diagnosis only."""
    if len(payload["feature_names"]) != len(payload["coefficients"]):
        raise ValueError("Saved feature/coefficient lengths disagree")
    coefficients = dict(zip(payload["feature_names"], payload["coefficients"]))
    if len(coefficients) != len(payload["feature_names"]):
        raise ValueError("Duplicate saved feature names")
    home, away = match["home_team_id"], match["away_team_id"]
    core = [coefficients["intercept"] + coefficients["home_advantage"]
            + coefficients[f"team_attack:{home}"] - coefficients[f"team_defence:{away}"],
            coefficients["intercept"] + coefficients[f"team_attack:{away}"]
            - coefficients[f"team_defence:{home}"]]
    xi = [0.0, 0.0]
    if payload["include_players"]:
        known = set(payload["players"])
        for team, key, own, opponent in ((home, "home_starters", 0, 1), (away, "away_starters", 1, 0)):
            starters = set(match[key])
            if starters - known:
                raise ValueError("Unknown starting identity cannot be zero-imputed in diagnostics")
            reference = payload["reference_frequencies"][str(team)]
            for player in sorted(starters | reference.keys()):
                deviation = float(player in starters) - reference.get(player, 0.0)
                xi[own] += deviation * coefficients[f"player_attack:{player}"]
                xi[opponent] -= deviation * coefficients[f"player_defence:{player}"]
    means = [math.exp(a + b) for a, b in zip(core, xi)]
    if any(not math.isfinite(value) or value <= 0 for value in means):
        raise ValueError("Nonfinite diagnostic means")
    return {"means": means, "team_core_log": core, "xi_log": xi}


def full_nll(means, goals):
    return sum(mu - goal * math.log(mu) + math.lgamma(goal + 1) for mu, goal in zip(means, goals))


def history_context(history):
    global_starts, team_starts, last = Counter(), defaultdict(Counter), {}
    for row in sorted(history, key=lambda row: (row["match_date"], row["match_id"])):
        for side in ("home", "away"):
            team, starters = row[f"{side}_team_id"], row[f"{side}_starters"]
            global_starts.update(starters)
            team_starts[team].update(starters)
            last[team] = set(starters)
    return global_starts, team_starts, last


def fixture_context(match, context):
    """Context uses earlier XIs only; future goals/paths do not affect groups."""
    global_starts, team_starts, last = context
    starters = match["home_starters"] + match["away_starters"]
    counts = [global_starts[player] for player in starters]
    own_counts, changes, reference_distance = [], 0, 0.0
    for side in ("home", "away"):
        team, current = match[f"{side}_team_id"], set(match[f"{side}_starters"])
        if team not in last:
            raise ValueError("No earlier team XI for historical comparison")
        own_counts.extend(team_starts[team][player] for player in current)
        changes += 11 - len(current & last[team])
        n_team_matches = sum(team_starts[team].values()) / 11
        reference_distance += 11 - sum(team_starts[team][p] / n_team_matches for p in current)
    return {"minimum_earlier_global_starts": min(counts), "mean_earlier_global_starts": statistics.mean(counts),
            "minimum_earlier_own_team_starts": min(own_counts),
            "unknown_starting_ids": sorted(p for p in starters if not global_starts[p]),
            "sum_changes_from_latest_training_xi": changes,
            "sum_expected_replacements_from_training_reference": reference_distance}


def group_rows(rows, field, ranges):
    groups = []
    for label, low, high in ranges:
        subset = [r for r in rows if r[field] >= low and (high is None or r[field] < high)]
        supported = [r for r in subset if r["status"] == "predicted"]
        groups.append({"label": label, "requested_matches": len(subset), "common_matches": len(supported),
                       "primary_delta": describe(r["full_score_delta"] for r in supported),
                       "team_full_nll": describe(r["team_full_nll"] for r in supported),
                       "players_full_nll": describe(r["players_full_nll"] for r in supported)})
    return groups


def calibration_moments(rows, track):
    supported = [row for row in rows if row["status"] == "predicted"]
    pairs = [(mu, goal) for row in supported for mu, goal in zip(row[f"{track}_means"], row["actual"])]
    return {"matches": len(supported), "team_sides": len(pairs),
            "predicted_goals": sum(mu for mu, _ in pairs), "observed_goals": sum(goal for _, goal in pairs),
            "mean_pearson_square": statistics.mean((goal - mu) ** 2 / mu for mu, goal in pairs) if pairs else None,
            "note": "descriptive conditional-rate moment, no independence/df correction or distribution test"}


def run(root, source, research):
    root, source, research = (Path(p).resolve() for p in (root, source, research))
    matches = read(source)
    indexed = {row["match_id"]: row for row in matches}
    if len(indexed) != len(matches):
        raise ValueError("Duplicate source match IDs")
    weeks = read(research / "逐周拟合与比较.json")
    archived = {kind: {row["match_id"]: row for row in read(research / f"full_match_{kind}_predictions.json")}
                for kind in ("team", "players")}
    output = root / "outputs" / datetime.now(timezone.utc).strftime("首发全场机制诊断_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "诊断口径.json", {
        "source": str(source), "research": str(research), "fits": 0, "parameter_searches": 0,
        "scope": "descriptive archived-fit diagnosis; no altered model, new predictions or selection",
        "groups": "fixed before execution: minimum earlier starts 0/1-3/4-7/8+, XI changes 0-3/4-7/8+",
        "all_matches_retained": True, "inference": "exploratory; no p-values, IID confidence intervals, winner selection or causal claim",
        "terms": "IMPECT custom terms, local personal noncommercial, no observed or derived data redistribution"})
    rows, training = [], []
    max_mean_error = 0.0
    for week in weeks:
        cutoff = week["week_start"]
        history = sorted((m for m in matches if m["match_date"] < cutoff), key=lambda m: (m["match_date"], m["match_id"]))
        context = history_context(history)
        models = {kind: read(week["tracks"][kind]["model_path"]) for kind in ("team", "players")}
        ids = [m["match_id"] for m in history]
        if any(p["before_date"] != cutoff or p["reference_match_ids"] != ids for p in models.values()):
            raise ValueError("Archived model history/cutoff disagrees with strictly earlier source")
        train_losses = {kind: [full_nll(rate_parts(model, m)["means"], m["full_time"]) for m in history]
                        for kind, model in models.items()}
        training.append({"week_start": cutoff, "history_matches": len(history),
                         "team_resubstitution_full_nll": statistics.mean(train_losses["team"]),
                         "players_resubstitution_full_nll": statistics.mean(train_losses["players"]),
                         "training_delta": statistics.mean(p - t for t, p in zip(train_losses["team"], train_losses["players"])),
                         "common_target_matches": week["comparison"]["common_matches"],
                         "common_target_delta": week["comparison"]["mean_players_minus_team"]["full_score_nll"]})
        for mid in week["target_match_ids"]:
            match, team, player = indexed[mid], archived["team"][mid], archived["players"][mid]
            features = fixture_context(match, context)
            row = {"match_id": mid, "match_date": match["match_date"], "week_start": cutoff,
                   "home_team_id": match["home_team_id"], "away_team_id": match["away_team_id"],
                   "status": player["status"], **features}
            if bool(features["unknown_starting_ids"]) != (player["status"] == "skipped"):
                raise ValueError("Archived skip does not match independently counted unknown starters")
            if player["status"] == "predicted":
                parts = {kind: rate_parts(model, match) for kind, model in models.items()}
                for kind, saved in (("team", team), ("players", player)):
                    max_mean_error = max(max_mean_error, *(abs(a-b) for a,b in zip(parts[kind]["means"], saved["means"])))
                row.update(actual=list(match["full_time"]), team_means=team["means"], players_means=player["means"],
                           team_full_nll=team["full_score_nll"], players_full_nll=player["full_score_nll"],
                           full_score_delta=player["full_score_nll"] - team["full_score_nll"],
                           xi_log_adjustment=parts["players"]["xi_log"],
                           joint_team_core_log_shift=[a-b for a,b in zip(parts["players"]["team_core_log"], parts["team"]["team_core_log"])])
            rows.append(row)
    if len(rows) != len({r["match_id"] for r in rows}):
        raise ValueError("Target match duplicated across windows")
    supported = [r for r in rows if r["status"] == "predicted"]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "research": str(research),
        "source_matches": len(matches), "target_matches": len(rows), "common_matches": len(supported),
        "unknown_skips": len(rows)-len(supported), "fits": 0, "parameter_searches": 0, "new_forecasts": 0,
        "max_reconstructed_saved_mean_error": max_mean_error,
        "primary_delta": describe(r["full_score_delta"] for r in supported), "training": training,
        "training_delta_summary": describe(w["training_delta"] for w in training),
        "training_improved_windows": sum(w["training_delta"] < 0 for w in training),
        "common_target_improved_windows": sum(w["common_target_delta"] is not None and w["common_target_delta"] < 0 for w in training),
        "xi_log_adjustment_absolute": describe(abs(x) for r in supported for x in r["xi_log_adjustment"]),
        "joint_team_core_log_shift_absolute": describe(abs(x) for r in supported for x in r["joint_team_core_log_shift"]),
        "calibration_moments_common": {kind: calibration_moments(rows, kind) for kind in ("team", "players")},
        "groups_by_minimum_earlier_starts": group_rows(rows, "minimum_earlier_global_starts", [("0",0,1),("1-3",1,4),("4-7",4,8),("8+",8,None)]),
        "groups_by_xi_changes": group_rows(rows, "sum_changes_from_latest_training_xi", [("0-3",0,4),("4-7",4,8),("8+",8,None)]),
        "inference": "descriptive only, no significance, causal attribution, parameter selection or row removal",
        "unchanged": "original sources, 36 saved fits, all 123 common matches, original models and prospective cases",
    }
    write_json(output / "机制诊断结果.json", result)
    write_json(output / "逐场历史信息与误差.json", rows)
    (output / "首发全场机制诊断说明.md").write_text(make_report(result), encoding="utf-8")
    print(f"Starting-XI diagnostics complete: {output}", flush=True)
    return output


def make_report(result):
    first, last = result["training"][0], result["training"][-1]
    def fmt(value):
        return "未知" if value is None else f"{value:.6f}"
    lines = ["# 首发全场：学习与泛化机制诊断", "",
             "数据提供方：[IMPECT Open Data](https://github.com/ImpectAPI/open-data)。"
             "仅本地个人非商业，遵守自定义条款和原署名logo，不再分发原始或派生观测。", "",
             f"源{result['source_matches']}场、目标{result['target_matches']}场，共同{result['common_matches']}场，"
             f"未知跳过{result['unknown_skips']}场；本轮0拟合、0调参、0新前瞻。所有比赛保留。", "",
             "## 训练内不等于训练外", "",
             f"{len(result['training'])}周中，训练内完整比分NLL改善{result['training_improved_windows']}周；"
             f"共同目标改善{result['common_target_improved_windows']}周。训练内是重复使用拟合资料的重代入成绩，不能当预测成绩。", "",
             "| 截止周 | 训练场数 | 球队训练内NLL | 首发训练内NLL | 训练外共同场次增量 |",
             "|---|---:|---:|---:|---:|"]
    for week in (first, last):
        lines.append(f"| {week['week_start']} | {week['history_matches']} | {fmt(week['team_resubstitution_full_nll'])} | "
                     f"{fmt(week['players_resubstitution_full_nll'])} | {fmt(week['common_target_delta'])} |")
    lines += ["", "## 较早首发样本与阵容变化", "",
              "以下分组在诊断执行前明确，只描述而不用于新选择规则。最少历史首发按22人中的全联盟较早首发次数计算；"
              "阵容变更相对当周训练截止日前每队最后一场首发，非同周最新比赛。", "",
              "| 最少历史首发 | 目标场数 | 共同场数 | 平均首发−球队完整NLL |", "|---|---:|---:|---:|"]
    for group in result["groups_by_minimum_earlier_starts"]:
        lines.append(f"| {group['label']} | {group['requested_matches']} | {group['common_matches']} | {fmt(group['primary_delta']['mean'])} |")
    lines += ["", "| 双方合计换首发人数 | 目标场数 | 共同场数 | 平均首发−球队完整NLL |", "|---|---:|---:|---:|"]
    for group in result["groups_by_xi_changes"]:
        lines.append(f"| {group['label']} | {group['requested_matches']} | {group['common_matches']} | {fmt(group['primary_delta']['mean'])} |")
    lines += ["", "## 不能把联合差异全归因于首发修正", "",
              f"共同场次{2 * result['common_matches']}个球队侧，首发对数修正绝对值均值{fmt(result['xi_log_adjustment_absolute']['mean'])}；"
              f"联合估计后的球队核心对数率变化绝对值均值{fmt(result['joint_team_core_log_shift_absolute']['mean'])}。"
              "逐球队侧的有符号核心变化与首发项相加，才是完整对数率变化；上述两个绝对值均值不能直接相加。"
              "首发系数并非直接乘到原球队模型上。", "",
              "本轮只有探索性描述，无独立同分布假设、p值或置信区间；不删异常场、不把训练内改善当泛化证据，"
              "不根据分组挑参数。单赛季、队友共现、名单选择和未知首发等不足以唯一解释负增量的原因。"
              "缺原始首发发布时间，仍不是真实T−60验证；不改变现有模型或9场前瞻。", "",
              "统计分析技能促使训练内外、覆盖与探索性分组分开报告，不将诊断解释为因果证明。全部新文件在G盘。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--research", required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.research)


if __name__ == "__main__":
    main()
