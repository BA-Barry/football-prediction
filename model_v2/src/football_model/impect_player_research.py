"""Fixed-policy, goals-only weekly IMPECT conditional-intensity research.

The caller supplies a separately inspected set of converted matches. This
module does not fetch, repair, reselect or silently filter source paths. It
reuses the existing retrospective estimator; actual lineups, score and man
paths are not pre-match inputs. No parameter search or 1X2 output is made.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import json
from pathlib import Path

from .full_season_experiment import paired_comparison, score_matches, summarize
from .joint_stint_model import JointStintIntensityModel
from .rolling_player_experiment import week_start
from .statsbomb_data import write_json


TARGET_FROM = date(2024, 1, 1)
TARGET_THROUGH = date(2024, 5, 18)
PLAYER_RIDGE = 100.0
TEAM_RIDGE = CONTEXT_RIDGE = 1.0
PENALTY_SOURCE = (
    "2015/16 StatsBomb EPL: earlier 140 matches fitted, middle 90 matches "
    "validated player_ridge 10/100/1000; goals selected 100. "
    "No 2023/24 match selects or changes this penalty."
)


def weekly_windows(matches):
    """Calendar-Monday expanding windows; a Tuesday cannot inform its Sunday."""
    if not isinstance(matches, list) or not matches:
        raise ValueError("A nonempty inspected match list is required")
    dates, grouped, seen = {}, defaultdict(list), set()
    for match in matches:
        mid = match.get("match_id")
        if not isinstance(mid, str) or not mid or mid in seen:
            raise ValueError("Missing or duplicate match ID in inspected input")
        seen.add(mid)
        text = match.get("match_date")
        if not isinstance(text, str) or len(text) != 10:
            raise ValueError("match_date must be a calendar date")
        day = date.fromisoformat(text)
        if day.isoformat() != text:
            raise ValueError("match_date must be a calendar date")
        dates[mid] = day
        if TARGET_FROM <= day <= TARGET_THROUGH:
            grouped[week_start(day)].append(match)
    if not grouped:
        raise ValueError("No target matches in the fixed 2024-01-01..2024-05-18 interval")
    for cutoff, target in sorted(grouped.items()):
        history = sorted((m for m in matches if dates[m["match_id"]] < cutoff),
                         key=lambda m: (m["match_date"], m["match_id"]))
        if not history:
            raise ValueError("Weekly refit has no strictly earlier observations")
        yield cutoff, history, sorted(target, key=lambda m: (m["match_date"], m["match_id"]))


def supported_comparison(team_scores, player_scores):
    """Keep zero coverage explicit, rather than inventing an average of zero."""
    common = ({p["match_id"] for p in team_scores["predictions"]}
              & {p["match_id"] for p in player_scores["predictions"]})
    if common:
        return paired_comparison(team_scores, player_scores)
    return {"common_match_ids": [], "team_on_common": None, "player_on_common": None,
            "mean_player_minus_team_loss": None, "paired_matches": [],
            "lower_is_better": True,
            "sampling_uncertainty": "not estimated; no significance or confidence claim",
            "support_note": "no common supported matches; retain all skips, not a zero loss"}


def coverage_summary(scores):
    requested = scores["requested_matches"]
    supported = len(scores["predictions"])
    return {"requested_matches": requested, "supported_matches": supported,
            "skipped_matches": len(scores["skipped"]),
            "supported_fraction": supported / requested if requested else None,
            "summary_on_supported": summarize(scores["predictions"]) if supported else None}


def training_observations(matches):
    """Describe actual exposure once per stint, not once per team/player."""
    stints = [stint for match in matches for stint in match["stints"]]
    man_values = sorted({stint["home_man_difference"] for stint in stints})
    unequal = [stint for stint in stints if stint["home_man_difference"] != 0]
    return {
        "matches": len(matches), "stints": len(stints),
        "home_goals": sum(stint["home_goals"] for stint in stints),
        "away_goals": sum(stint["away_goals"] for stint in stints),
        "total_goals": sum(stint["home_goals"] + stint["away_goals"] for stint in stints),
        "total_actual_seconds": sum(stint["duration_seconds"] for stint in stints),
        "home_man_difference_observed_values": man_values,
        "home_man_difference_varies": len(man_values) > 1,
        "unequal_player_count_stints": len(unequal),
        "unequal_player_count_seconds": sum(stint["duration_seconds"] for stint in unequal),
        "matches_with_unequal_player_count": sum(
            any(stint["home_man_difference"] != 0 for stint in match["stints"]) for match in matches),
        "man_difference_limit": (
            "No unequal-player-count exposure: do not claim a red-card/man-disadvantage effect was learned"
            if not unequal else
            "Observed count imbalance is not itself a causal red-card effect; balanced double dismissals may have zero difference"
        ),
    }


def source_denominators(inspection, matches):
    """Retain raw catalogue and target denominators, without changing inclusion."""
    eligible_target = sum(TARGET_FROM.isoformat() <= match["match_date"] <= TARGET_THROUGH.isoformat()
                          for match in matches)
    expected_target = inspection.get("expected_target_matches")
    expected_catalogue = inspection.get("expected_catalogue_matches")
    return {
        "expected_catalogue_matches": expected_catalogue,
        "expected_target_matches": expected_target,
        "eligible_matches_in_parsed_input": len(matches),
        "eligible_target_matches_in_parsed_input": eligible_target,
        "inspection_declared_eligible_matches": inspection.get("eligible_matches"),
        "inspection_declared_eligible_target_matches": inspection.get("eligible_target_matches"),
        "source_ineligible_target_matches": (expected_target - eligible_target
                                               if type(expected_target) is int else None),
        "raw_red_card_matches": inspection.get("raw_red_card_matches"),
        "eligible_red_card_matches": inspection.get("eligible_red_card_matches"),
        "note": "raw source selection denominator, trusted parsed denominator, and model identity skips stay separate",
    }


def fixed_policy():
    return {
        "response_kind": "goals", "player_ridge": PLAYER_RIDGE,
        "team_ridge": TEAM_RIDGE, "context_ridge": CONTEXT_RIDGE,
        "max_iterations": 2000, "player_penalty_provenance": PENALTY_SOURCE,
        "target_from": TARGET_FROM.isoformat(), "target_through": TARGET_THROUGH.isoformat(),
        "cutoff": "each calendar Monday, strictly match_date < cutoff; no same-week updates",
        "history": "expanding inspected matches only; no decay, no parameter search",
        "new_player_policy": "unknown actual on-pitch ID excludes whole player-line match; never zero imputed",
        "source_selection": "caller-supplied inspected matches, no extra source repair/filtering in this runner",
        "availability": "date-level result-availability assumption; exact historical publication times unknown",
        "loss": "goals point-process NLL up to model-independent constants, conditional on actual paths",
        "limits": "retrospective nonrandom known-player subset; not pre-match score/1X2 or causal player value",
        "status": "exploratory development; this season's source quality and pilot have already been viewed",
        "sampling_uncertainty": "not estimated; no significance or confidence claim",
        "source_terms": "IMPECT open-data custom terms apply; local personal noncommercial research, no data redistribution",
    }


def _score_paths(output, totals):
    for kind, scores in totals.items():
        write_json(output / f"goals_weekly_{kind}_predictions.json", scores)


def run(root, parsed, inspection):
    """Execute only the explicitly provided local research inputs."""
    root, parsed, inspection = (Path(p).resolve() for p in (root, parsed, inspection))
    matches = json.loads(parsed.read_text(encoding="utf-8"))
    source_inspection = json.loads(inspection.read_text(encoding="utf-8"))
    if not isinstance(source_inspection, dict):
        raise ValueError("A saved source-inspection object is required")
    # Resolve chronology before starting any fit, not target identities/features.
    windows = list(weekly_windows(matches))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"较新球员固定前推研究_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    policy = fixed_policy()
    policy.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "parsed_input": str(parsed), "inspection_input": str(inspection),
                   "inspected_input_matches": len(matches),
                   "inspected_input_before_first_target": sum(m["match_date"] < TARGET_FROM.isoformat() for m in matches),
                   "target_matches_in_inspected_input": sum(len(target) for _, _, target in windows),
                   "inspected_input_after_target_interval": sum(m["match_date"] > TARGET_THROUGH.isoformat() for m in matches)})
    denominators = source_denominators(source_inspection, matches)
    policy["source_denominators"] = denominators
    write_json(output / "固定研究设置.json", policy)
    write_json(output / "源检验说明副本.json", source_inspection)
    totals = {kind: {"predictions": [], "skipped": [], "requested_matches": 0}
              for kind in ("team", "players")}
    weeks, fits = [], 0
    try:
        for cutoff, history, target in windows:
            print(f"IMPECT fixed weekly cutoff {cutoff}: history {len(history)}, target {len(target)}", flush=True)
            record = {"week_start": cutoff.isoformat(), "history_matches": len(history),
                      "history_min_date": history[0]["match_date"], "history_max_date": history[-1]["match_date"],
                      "training_observations": training_observations(history),
                      "target_match_ids": [m["match_id"] for m in target], "requested_matches": len(target),
                      "tracks": {}}
            scores = {}
            for kind, include_players in (("team", False), ("players", True)):
                model = JointStintIntensityModel(
                    include_players=include_players, response_kind="goals", team_ridge=TEAM_RIDGE,
                    player_ridge=PLAYER_RIDGE, context_ridge=CONTEXT_RIDGE, max_iterations=2000,
                ).fit(history, cutoff)
                fits += 1
                path = model_dir / f"{cutoff}_goals_{kind}.json"
                model.save(path)
                restored = JointStintIntensityModel.load(path)
                if restored.parameters() != model.parameters():
                    raise ValueError("Saved conditional-intensity model does not preserve its parameters")
                # All scoring uses the saved and restored artifact, not an
                # in-memory fit that has never been recorded.
                score = score_matches(restored, target)
                scores[kind] = score
                for key in ("predictions", "skipped"):
                    totals[kind][key].extend(score[key])
                totals[kind]["requested_matches"] += len(target)
                record["tracks"][kind] = {
                    **coverage_summary(score), "model_path": str(path),
                    "fitted_players": len(restored.players), "optimizer": restored.optimizer_diagnostics,
                    "parameters_saved_and_restored_before_scoring": True,
                    "skipped": score["skipped"],
                }
                # Preserve completed work even if the other line later fails.
                _score_paths(output, totals)
            record["comparison"] = supported_comparison(scores["team"], scores["players"])
            record["cumulative_comparison"] = supported_comparison(totals["team"], totals["players"])
            record["cumulative_coverage"] = {kind: coverage_summary(value) for kind, value in totals.items()}
            weeks.append(record)
            write_json(output / "逐周拟合与比较.json", weeks)
        result = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "model_fits_completed": fits, "weeks": len(weeks),
                  "coverage": {kind: coverage_summary(value) for kind, value in totals.items()},
                  "comparison": supported_comparison(totals["team"], totals["players"]),
                  "source_denominators": denominators,
                  "last_cutoff_training_observations": weeks[-1]["training_observations"],
                  "policy": policy}
        write_json(output / "固定前推结果.json", result)
        (output / "较新球员固定前推说明.md").write_text(make_report(result), encoding="utf-8")
        print(f"IMPECT fixed research complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {
            "error": f"{type(error).__name__}: {error}", "completed_fits": fits,
            "completed_weeks": len(weeks), "status": "incomplete; no completed-result claim",
        })
        raise


def make_report(result):
    coverage, comparison = result["coverage"], result["comparison"]
    common = len(comparison["common_match_ids"])
    source = result["source_denominators"]
    observations = result["last_cutoff_training_observations"]
    def count(value):
        return "未知" if value is None else str(value)
    lines = ["# 较新男足球员条件强度：固定规则每周前推", "",
             "来源：[IMPECT Open Data](https://github.com/ImpectAPI/open-data)，2023/24男足德甲。"
             "遵守自定义使用条款，仅本地个人非商业研究；不随代码再分发观测或派生数据。", "",
             f"完成{result['weeks']}个有目标比赛周、{result['model_fits_completed']}次拟合。"
             "目标日期固定为2024-01-01至2024-05-18；每周一只用严格更早日期的已解可信比赛扩窗训练，"
             "同周赛果不用于该周后续比赛。历史精确发布时间未知，日期级可用性是研究假设。", "",
             "真实进球单轨；球员ridge固定100，沿用2015/16英超早期训练／中间验证的既有选择。"
             "球队／状态ridge固定1、max_iterations固定2000；没有在本季比较候选、再选规则或改变收缩公式。"
             "所有参数先保存，再载入评分。", "",
             "## 覆盖与同场比较", "",
             f"原始目录{count(source['expected_catalogue_matches'])}场，传入可信比赛{source['eligible_matches_in_parsed_input']}场；"
             f"原始目标{count(source['expected_target_matches'])}场，可信目标{source['eligible_target_matches_in_parsed_input']}场，"
             f"源阶段未纳入目标{count(source['source_ineligible_target_matches'])}场。模型覆盖的可信目标分母不能取代原始目标分母。", "",
             f"仅球队支持{coverage['team']['supported_matches']}/{coverage['team']['requested_matches']}场；"
             f"球员线支持{coverage['players']['supported_matches']}/{coverage['players']['requested_matches']}场，"
             f"跳过{coverage['players']['skipped_matches']}场。未知球员不默认零：整场跳过且保留ID原因。"
             "源解析失败／未纳入比赛另见源检验说明副本，不与模型未知球员混为同一原因。", ""]
    if common:
        lines.extend(["| 同一支持队列 | 场数 | 平均条件点过程损失 |", "|---|---:|---:|",
                      f"| 仅球队 | {common} | {comparison['team_on_common']['mean_match_loss']:.6f} |",
                      f"| 加球员 | {common} | {comparison['player_on_common']['mean_match_loss']:.6f} |", "",
                      f"球员减球队的同场平均损失差：{comparison['mean_player_minus_team_loss']:+.6f}（越低越好）。"])
    else:
        lines.append("本轮没有共同支持比赛；损失差为未知，不报告0或拿不同队列比较。")
    lines.extend(["", "## 实际训练观测", "",
                  "逐周记录训练比赛、进球、实际暴露秒数与时段数，秒数按每个时段只计一次，不乘双队或球员人数。",
                  f"最后一个截止日的训练集合：{observations['matches']}场、{observations['stints']}段、"
                  f"{observations['total_goals']}球、{observations['total_actual_seconds']:.3f}实际秒。"
                  f"训练人数差取值{observations['home_man_difference_observed_values']}；"
                  f"非零人数差{observations['unequal_player_count_stints']}段、"
                  f"{observations['unequal_player_count_seconds']:.3f}秒。"])
    if not observations["unequal_player_count_stints"]:
        lines.extend(["没有观测到人数不等暴露，不能声称训练学到了红牌／人数劣势影响。"
                      "状态项保留在既有估计器中不等于有可识别的实测证据。"])
    else:
        lines.append("人数差有实际暴露也不能解释为红牌因果影响；双方同时减员可能仍为零人数差。")
    lines.extend(["", "两线均使用相同的实际在场、段初比分、人数与完整可解析比赛时长；"
                  "损失是去模型无关常数的进球条件点过程NLL，不是完整比分NLL、胜平负准确率或因果球员价值。"
                  "球队全覆盖均值单列存档，不拿它与球员非随机支持子集直接比较。", "",
                  "逐周／累计覆盖与成对损失、全部预测、跳过原因、所有模型及优化器信息都已保存。"
                  "本次未估计抽样区间，不作显著性或稳定提升结论；已查看的全季来源质量与三场实测使其属于探索性研发，"
                  "不是全新未见盲测。", "",
                  "## 尚未变成赛前预测", "",
                  "本轮不产生xG、1X2、总进球分布、半全场、市场融合或串关；没有新增神经网络、球员先验或"
                  "赛事投入假设。2023/24 IMPECT球员系数不能按姓名／整数ID拼进2026其他来源。"
                  "接入赛前需要预计在场情景与条件参考映射，仍需讨论，当前球队与LLM预测不被覆盖。", "",
                  "复现：`python -m football_model.impect_player_research --root <G盘项目> --parsed <parsed_matches.json> "
                  "--inspection <inspection.json>`。只读取显式本地输入，不隐式选最近目录或联网。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.parsed, arguments.inspection)
