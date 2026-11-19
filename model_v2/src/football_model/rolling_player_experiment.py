"""Expanding-window weekly refits; strict unknown-player handling retained.

The previously viewed later season is exploratory here, not a new untouched
test. Player penalties stay at the earlier validation choices. Real on-pitch
and score paths remain retrospective conditions, not pre-match inputs.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from .full_season_experiment import TEST_FROM, paired_comparison, score_matches, summarize
from .joint_stint_model import JointStintIntensityModel
from .statsbomb_data import write_json


def week_start(day):
    if type(day) is not date:
        raise ValueError("A calendar date is required")
    return day - timedelta(days=day.weekday())


def rolling_windows(matches, from_date=TEST_FROM):
    if type(from_date) is not date:
        raise ValueError("A calendar date is required")
    grouped = defaultdict(list)
    seen = set()
    for match in matches:
        if match["match_id"] in seen:
            raise ValueError("Duplicate match in weekly windows")
        seen.add(match["match_id"])
        day = date.fromisoformat(match["match_date"])
        if day >= from_date:
            grouped[week_start(day)].append(match)
    for cutoff in sorted(grouped):
        history = [m for m in matches if date.fromisoformat(m["match_date"]) < cutoff]
        target = sorted(grouped[cutoff], key=lambda m: (m["match_date"], m["match_id"]))
        if not history:
            raise ValueError("Weekly refit has no strictly earlier observations")
        yield cutoff, history, target


def static_vs_rolling(static_scores, rolling_scores):
    paired = paired_comparison(static_scores, rolling_scores)
    return {"common_match_ids": paired["common_match_ids"],
            "static_on_common": paired["team_on_common"],
            "rolling_on_common": paired["player_on_common"],
            "mean_rolling_minus_static_loss": paired["mean_player_minus_team_loss"],
            "interpretation": "same supported games only; viewed-season exploratory comparison"}


def compare_supported_week(team_scores, player_scores):
    common = ({p["match_id"] for p in team_scores["predictions"]}
              & {p["match_id"] for p in player_scores["predictions"]})
    if not common:
        return {"common_match_ids": [], "mean_player_minus_team_loss": None,
                "support_note": "no common supported games this week; retain skips and continue"}
    return paired_comparison(team_scores, player_scores)


def find_source(root):
    candidates = sorted((Path(root) / "outputs").glob("完整赛季球员强度训练_*"), reverse=True)
    return next(p for p in candidates if (p / "较晚检验结果.json").is_file()
                and (p / "验证期选择结果.json").is_file())


def run(root, source=None):
    root = Path(root).resolve()
    source = Path(source).resolve() if source else find_source(root)
    matches = json.loads((source / "真实在场时段.json").read_text(encoding="utf-8"))
    selection = json.loads((source / "验证期选择结果.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"每周滚动球员强度训练_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    penalties = {r: selection[r]["selected_player_ridge"] for r in ("goals", "xg")}
    write_json(output / "每周实验设置.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_experiment": str(source),
        "cutoff": "each calendar Monday; strictly match_date < cutoff; no within-week updates",
        "first_target_date": TEST_FROM.isoformat(), "window": "expanding history",
        "player_penalties_from_earlier_validation": penalties, "team_ridge": 1.0, "context_ridge": 1.0,
        "new_player_policy": "whole match excluded until actually observed before cutoff; no zero prior",
        "selection_using_later_matches": False,
        "status": "exploratory viewed-season development, not a new untouched test",
        "time_semantics": "date-level availability assumption; historical exact availability unknown",
        "usage": "retrospective actual-state intensity; not pre-match probabilities"})
    totals = {r: {k: {"predictions": [], "skipped": [], "requested_matches": 0}
                   for k in ("team", "players")} for r in penalties}
    weeks = []
    fits = 0
    try:
        for cutoff, history, target in rolling_windows(matches):
            record = {"week_start": cutoff.isoformat(), "history_matches": len(history),
                      "history_max_date": max(m["match_date"] for m in history),
                      "requested_matches": len(target), "tracks": {}}
            print(f"Weekly cutoff {cutoff}: history {len(history)}, target {len(target)}", flush=True)
            for response, penalty in penalties.items():
                models, scores = {}, {}
                for kind, players in (("team", False), ("players", True)):
                    model = JointStintIntensityModel(include_players=players, response_kind=response,
                                                    player_ridge=penalty).fit(history, cutoff)
                    fits += 1
                    path = model_dir / f"{cutoff}_{response}_{kind}.json"
                    model.save(path)
                    score = score_matches(model, target)
                    models[kind], scores[kind] = model, score
                    for key in ("predictions", "skipped"):
                        totals[response][kind][key].extend(score[key])
                    totals[response][kind]["requested_matches"] += len(target)
                shared = compare_supported_week(scores["team"], scores["players"])
                restored = JointStintIntensityModel.load(model_dir / f"{cutoff}_{response}_players.json")
                roundtrip = restored.parameters() == models["players"].parameters()
                if shared["common_match_ids"]:
                    witness_id = shared["common_match_ids"][0]
                    witness = next(m for m in target if m["match_id"] == witness_id)
                    roundtrip = roundtrip and restored.predict_match(witness) == models["players"].predict_match(witness)
                if not roundtrip:
                    raise ValueError("Weekly saved player model does not reproduce prediction")
                record["tracks"][response] = {"known_players": len(models["players"].players),
                    "supported_matches": len(shared["common_match_ids"]),
                    "player_minus_team_loss": shared["mean_player_minus_team_loss"],
                    "skipped_players": scores["players"]["skipped"],
                    "team_optimizer": models["team"].optimizer_diagnostics,
                    "player_optimizer": models["players"].optimizer_diagnostics,
                    "parameter_roundtrip": True}
            weeks.append(record)
            write_json(output / "逐周拟合明细.json", weeks)
        result = {}
        for response in penalties:
            team, player = totals[response]["team"], totals[response]["players"]
            write_json(output / f"{response}_weekly_team_predictions.json", team)
            write_json(output / f"{response}_weekly_players_predictions.json", player)
            static = json.loads((source / f"test_{response}_players_predictions.json").read_text(encoding="utf-8"))
            result[response] = {"player_ridge": penalties[response],
                "requested_matches": player["requested_matches"],
                "supported_matches": len(player["predictions"]), "skipped_matches": len(player["skipped"]),
                "comparison": paired_comparison(team, player), "team_full_coverage": summarize(team["predictions"]),
                "static_player_supported_matches": len(static["predictions"]),
                "static_vs_rolling_on_common": static_vs_rolling(static, player)}
        write_json(output / "每周滚动结果.json", {"model_fits_completed": fits, "weeks": len(weeks), "tracks": result})
        (output / "每周滚动训练说明.md").write_text(make_report(result, len(weeks), fits), encoding="utf-8")
        print(f"Weekly experiment complete: {output}", flush=True)
        for response, row in result.items():
            print(f"{response}: coverage {row['supported_matches']}/{row['requested_matches']}; "
                  f"player-minus-team {row['comparison']['mean_player_minus_team_loss']:+.6f}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"error": f"{type(error).__name__}: {error}",
                   "completed_fits": fits, "status": "incomplete; do not treat partial files as completed experiment"})
        raise


def make_report(result, weeks, fits):
    lines = ["# 每周滚动球员历史强度训练", "",
        "用户选择：先每周只用更早比赛重训，保留严格未知处理，暂不增加新球员先验。", "",
        f"实际完成{weeks}个有比赛周、{fits}次拟合。2016-02-02至2016-05-17共150场；每周一更新，"
        "训练仅使用日期早于本周一的比赛。同周周二的结果不会用于预测同周周日；下一周才可使用。"
        "日期级可用性是假设，不是掌握精确的历史结果发布时间。", "",
        "进球／xG球员正则分别固定为此前中间90场选出的100／10，球队与状态正则仍为1；没有根据后期成绩改选。"
        "窗口从较早230场逐周扩展；新球员在首次出现所在周仍为未知，不默认其系数为零。", "",
        "## 实际结果：每周模型在相同比赛上比较", "",
        "| 目标 | 严格球员覆盖 | 仅球队损失 | 加球员损失 | 加球员减仅球队 |",
        "|---|---:|---:|---:|---:|"]
    for response, label in (("goals", "真实进球"), ("xg", "xG单独对照")):
        row, compare = result[response], result[response]["comparison"]
        lines.append(f"| {label} | {row['supported_matches']}/{row['requested_matches']} | "
                     f"{compare['team_on_common']['mean_match_loss']:.6f} | "
                     f"{compare['player_on_common']['mean_match_loss']:.6f} | {compare['mean_player_minus_team_loss']:+.6f} |")
    lines.extend(["", "越低越好；进球点过程损失与xG准损失不是同一种尺度，不跨轨比较。没有估计抽样区间，"
        "不宣称统计显著或稳定提高。模型双方都以同样的实际比分、人数和在场路径为条件。", "",
        "## 固定窗口与滚动窗口：仅比较双方都支持的场次", "",
        "不能把原固定窗口20场均分与滚动模型更大样本均分直接比较。以下只用二者交集。", "",
        "| 目标 | 同场数 | 固定窗口球员损失 | 滚动球员损失 | 滚动减固定 |",
        "|---|---:|---:|---:|---:|"])
    for response, label in (("goals", "真实进球"), ("xg", "xG单独对照")):
        row = result[response]["static_vs_rolling_on_common"]
        lines.append(f"| {label} | {len(row['common_match_ids'])} | {row['static_on_common']['mean_match_loss']:.6f} | "
                     f"{row['rolling_on_common']['mean_match_loss']:.6f} | {row['mean_rolling_minus_static_loss']:+.6f} |")
    lines.extend(["", "## 解释边界与后续", "",
        "覆盖增长是身份数据逐周积累，不等于预测准确率增长。依然被排除的比赛有真实首次出场者，"
        "不存在ID丢失就自动补0的规则。当前结果仍是已知球员的非随机子集。", "",
        "这一赛季较晚结果已经在固定窗口实验中查看；更新频率又是根据覆盖问题随后决定，因此本轮是探索性研发，"
        "不是全新的未见检验。此前逐场较晚结果只有在其日期早于下一周时才能进入拟合。需要另一独立完整赛季验证稳定性。", "",
        "训练与评分仍使用实际历史在场、换人、比分状态和实际时长；尚无赛前预计路径、历史官方首发发布时间、"
        "伤停来源或赛事优先级训练，不输出1X2、半全场、价值赔率或串关。球员系数是收缩后的条件关联，不是个人因果价值。", "",
        "本轮没有增加新球员先验、改市场权重或更换目标。下一步建议先在另一完整赛季复核；"
        "新球员分层先验、参数不确定性和赛前阵容情景再分别讨论。", "",
        "## 复现与文件", "",
        "`run.ps1 -Action player-rolling`读取最近完成的完整赛季实验，不联网。逐周全部四个模型参数、"
        "预测与跳过原因保留在本目录；两个目标分开保存。所有项目文件／缓存／临时文件均在G盘。"
        "数据来源与许可、独立比分核对详见前一完整赛季实验；不重新分发原始或衍生数据。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-experiment", type=Path)
    args = parser.parse_args()
    run(args.root, args.source_experiment)
