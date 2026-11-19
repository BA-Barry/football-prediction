"""Reproducible source measurements, not a Wyscout training adapter.

Never synthesize a goal, repair a card, infer a true whistle or drop a
goal-dependent window into the existing point-process likelihood.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

from .statsbomb_data import write_json
from .wyscout_data import iter_match_events


FD_TEAMS = {1609: "Arsenal", 1610: "Chelsea", 1611: "Man United", 1612: "Liverpool",
    1613: "Newcastle", 1619: "Southampton", 1623: "Everton", 1624: "Tottenham",
    1625: "Man City", 1627: "West Brom", 1628: "Crystal Palace", 1631: "Leicester",
    1633: "West Ham", 1639: "Stoke", 1644: "Watford", 1646: "Burnley",
    1651: "Brighton", 1659: "Bournemouth", 1673: "Huddersfield", 10531: "Swansea"}
PERIOD = {"1H": 0, "2H": 1}


def tags(event):
    return {int(t["id"]) for t in event.get("tags", [])}


def goal_team(event, team_ids):
    """Count an original primary goal once; keeper tag 101 is NOT another goal."""
    marked = tags(event)
    team_id = event["teamId"]
    if 102 in marked:
        if team_id not in team_ids or len(team_ids) != 2:
            raise ValueError("Own goal needs two known match teams")
        return next(t for t in team_ids if t != team_id)
    if 101 in marked and event.get("eventName") != "Save attempt":
        if team_id not in team_ids:
            raise ValueError("Goal event team not in match")
        return team_id
    return None


def distribution(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values), "median": median(values), "max": max(values)}


def substitutions(formation):
    raw = formation.get("substitutions")
    if raw in (None, "null", 0):
        return []
    if not isinstance(raw, list):
        raise ValueError("Unexpected substitutions field")
    return raw


def minute_hypothesis(sub, actor):
    """Test ONE ceil-minute convention against actions; do not repair clocks.

    The observed actor bounds are diagnostics using future actions, NOT
    predictable censoring windows for a goal-intensity likelihood.
    """
    incoming = actor.get(sub["playerIn"], [])
    outgoing = actor.get(sub["playerOut"], [])
    first_in = min(incoming, default=None)
    last_out = max(outgoing, default=None)
    minute = sub["minute"]
    if type(minute) is not int or minute < 0:
        raise ValueError("Integer nonnegative substitution minute required")
    half = 0 if minute < 45 or (minute == 45 and first_in is not None and first_in[0] == 0) else 1
    base = 2700 * half
    low = max(0, (minute - 1) * 60 - base)
    high = max(0, minute * 60 - base)
    if last_out is not None:
        if last_out[0] == half:
            low = max(low, last_out[1])
        elif last_out[0] > half:
            low = math.inf
    if first_in is not None:
        if first_in[0] == half:
            high = min(high, first_in[1])
        elif first_in[0] < half:
            high = -math.inf
    conflict = low > high
    gap = low - high if conflict else 0
    return {"minute": minute, "player_in": sub["playerIn"], "player_out": sub["playerOut"],
        "assumed_period": "1H" if half == 0 else "2H", "first_in_action": first_in,
        "last_out_action": last_out, "no_incoming_action": first_in is None,
        "ceil_minute_action_conflict": conflict, "gap_seconds": gap if math.isfinite(gap) else None,
        "incompatible_period": not math.isfinite(gap),
        "actor_order_conflict": first_in is not None and last_out is not None and first_in < last_out}


def fd_index(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("Date"):
                continue
            key = (datetime.strptime(row["Date"], "%d/%m/%Y").date().isoformat(), row["HomeTeam"], row["AwayTeam"])
            if key in result:
                raise ValueError("Duplicate independent date/home/away key")
            # Do not load odds into this player-data inspection.
            result[key] = {name: int(row[name]) for name in ("FTHG", "FTAG", "HTHG", "HTAG", "HR", "AR")}
    return result


def inspect(root):
    root = Path(root).resolve()
    source = root / "data/wyscout/premier-league-2017-18"
    metadata = json.loads((source / "matches_England.json").read_text(encoding="utf-8"))
    matches = {m["wyId"]: m for m in metadata}
    if len(matches) != len(metadata):
        raise ValueError("Duplicate match metadata ID")
    independent = fd_index(source / "Football-Data_E0_2017-18.csv")
    fd_seen, seen = set(), set()
    counts, event_types, periods, minute_counts = Counter(), Counter(), Counter(), Counter()
    scores, red_details, sub_details, ends, structural = [], [], [], [], []
    half_ends = defaultdict(list)
    roster_counts, pair_counts = Counter(), Counter()
    team_names = {t["wyId"]: t["name"] for t in json.loads((source / "teams.json").read_text(encoding="utf-8"))}
    for mid, events in iter_match_events(source / "events_England.json"):
        if mid in seen or mid not in matches:
            raise ValueError("Repeated or unknown event match ID")
        seen.add(mid)
        match = matches[mid]
        teams = list(match["teamsData"].values())
        by_side = {t["side"]: t for t in teams}
        home, away = by_side["home"]["teamId"], by_side["away"]["teamId"]
        team_ids = {home, away}
        roster_counts.update(team_ids)
        pair_counts[(home, away)] += 1
        key = (match["dateutc"][:10], FD_TEAMS[home], FD_TEAMS[away])
        if key in fd_seen:
            raise ValueError("Repeated date/home/away key")
        fd_seen.add(key)
        fd = independent.get(key)
        goals, ht_goals, event_reds = Counter(), Counter(), Counter()
        actor, actor_events, event_ids, observed = defaultdict(list), defaultdict(list), set(), {}
        previous_clock = {}
        for event in events:
            counts["events"] += 1
            eid = event["id"]
            if eid in event_ids:
                counts["duplicate_event_ids_within_match"] += 1
            event_ids.add(eid)
            period, sec = event["matchPeriod"], float(event["eventSec"])
            if period not in PERIOD or not math.isfinite(sec) or sec < 0:
                raise ValueError("Unsupported period or invalid event clock")
            periods[period] += 1
            event_types[event["eventName"]] += 1
            if sec < previous_clock.get(period, -1):
                counts["raw_clock_inversions"] += 1
            previous_clock[period] = sec
            if period not in observed or sec > observed[period][0]:
                observed[period] = (sec, event["eventName"], event.get("subEventName"), eid)
            player = event.get("playerId")
            if player:
                actor[player].append((PERIOD[period], sec))
                actor_events[player].append(event)
            marked = tags(event)
            counts["tag101"] += 101 in marked
            counts["keeper_tag101"] += 101 in marked and event["eventName"] == "Save attempt"
            counts["own_goal_tag102"] += 102 in marked
            counts["simultaneous_goal_own_goal_tags"] += 101 in marked and 102 in marked
            awarded = goal_team(event, team_ids)
            if awarded is not None:
                goals[awarded] += 1
                ht_goals[awarded] += period == "1H"
                counts["primary_goals"] += 1
                counts["primary_goals_after_45_per_half"] += sec > 2700
            for red_tag in (1701, 1703):
                if red_tag in marked:
                    counts[f"tag{red_tag}"] += 1
            if marked & {1701, 1703}:
                event_reds[event["teamId"]] += 1
                red_details.append({"match_id": mid, "team_id": event["teamId"], "player_id": player,
                    "period": period, "seconds": sec, "tags": sorted(marked & {1701, 1703})})
        metadata_reds, metadata_red_players = Counter(), set()
        for team in teams:
            formation = team["formation"]
            lineup = formation["lineup"]
            if len(lineup) != 11 or len({r["playerId"] for r in lineup}) != 11:
                structural.append({"match_id": mid, "team_id": team["teamId"], "issue": "nonunique_or_non11_XI"})
            if formation.get("substitutions") == "null":
                counts["substitutions_string_null"] += 1
            for row in lineup + formation.get("bench", []):
                if str(row.get("redCards")) not in ("0", "null", "None", ""):
                    metadata_reds[team["teamId"]] += 1
                    metadata_red_players.add(row["playerId"])
            active = {r["playerId"] for r in lineup}
            for sub in substitutions(formation):
                counts["substitutions"] += 1
                minute_counts[sub["minute"]] += 1
                detail = {"match_id": mid, "team_id": team["teamId"], **minute_hypothesis(sub, actor)}
                sub_details.append(detail)
                if sub["playerOut"] not in active or sub["playerIn"] in active:
                    structural.append({"match_id": mid, "team_id": team["teamId"], "issue": "substitution_chain", "sub": sub})
                active.discard(sub["playerOut"])
                active.add(sub["playerIn"])
        for red in (r for r in red_details if r["match_id"] == mid):
            point = (PERIOD[red["period"]], red["seconds"])
            red["later_actor_events"] = sum((PERIOD[e["matchPeriod"]], e["eventSec"]) > point for e in actor_events[red["player_id"]])
            red["in_metadata_red_players"] = red["player_id"] in metadata_red_players
        for period, (sec, name, subname, eid) in observed.items():
            half_ends[period].append(sec)
            ends.append({"match_id": mid, "period": period, "last_record_seconds": sec, "event_name": name,
                "sub_event_name": subname, "event_id": eid, "is_true_end_known": False})
        counts["metadata_red_records"] += sum(metadata_reds.values())
        scores.append({"match_id": mid, "date_utc": match["dateutc"], "home": FD_TEAMS[home], "away": FD_TEAMS[away],
            "metadata_ft": [by_side["home"]["score"], by_side["away"]["score"]],
            "metadata_ht": [by_side["home"]["scoreHT"], by_side["away"]["scoreHT"]],
            "events_ft": [goals[home], goals[away]], "events_ht": [ht_goals[home], ht_goals[away]],
            "metadata_reds": [metadata_reds[home], metadata_reds[away]], "event_reds": [event_reds[home], event_reds[away]],
            "fd_ft": None if fd is None else [fd["FTHG"], fd["FTAG"]],
            "fd_ht": None if fd is None else [fd["HTHG"], fd["HTAG"]],
            "fd_reds": None if fd is None else [fd["HR"], fd["AR"]]})
        if len(seen) % 50 == 0:
            print(f"Measured {len(seen)} matches / {counts['events']} original events", flush=True)
    conflicts = [s for s in sub_details if s["ceil_minute_action_conflict"]]
    score_diff = {name: [s for s in scores if s["fd_ft"] is not None and s[name] != s[other]]
        for name, other in (("metadata_ft", "fd_ft"), ("metadata_ht", "fd_ht"), ("events_ft", "fd_ft"),
                            ("events_ht", "fd_ht"), ("metadata_reds", "fd_reds"), ("event_reds", "fd_reds"))}
    known_teams = set(roster_counts)
    missing_pairs = [(h, a) for h in known_teams for a in known_teams if h != a and (h, a) not in pair_counts]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_directory": str(source),
        "scope": "Original Wyscout 2017/18 EPL; data inspection only, zero new fits",
        "matches": len(matches), "event_matches": len(seen), "teams": len(known_teams),
        "date_min_utc": min(m["dateutc"] for m in metadata), "date_max_utc": max(m["dateutc"] for m in metadata),
        "team_match_counts": {str(k): {"name": team_names[k], "matches": v} for k, v in sorted(roster_counts.items())},
        "missing_home_away_pairs": missing_pairs,
        "duplicate_home_away_pairs": [(h, a, n) for (h, a), n in pair_counts.items() if n != 1],
        "metadata_without_events": sorted(set(matches) - seen), "independent_matches": len(independent),
        "unmatched_wyscout_keys": sorted(fd_seen - independent.keys()), "unmatched_independent_keys": sorted(independent.keys() - fd_seen),
        "counts": dict(counts), "event_types": dict(event_types), "period_events": dict(periods),
        "independent_goals_ft": sum(r["FTHG"] + r["FTAG"] for r in independent.values()),
        "independent_goals_ht": sum(r["HTHG"] + r["HTAG"] for r in independent.values()),
        "score_comparison_mismatches": score_diff, "substitution_minute_counts": dict(sorted(minute_counts.items())),
        "substitution_no_incoming_action": sum(s["no_incoming_action"] for s in sub_details),
        "ceil_minute_assumption_conflicts": len(conflicts), "ceil_conflict_matches": len({s["match_id"] for s in conflicts}),
        "ceil_conflict_gap_seconds": distribution([s["gap_seconds"] for s in conflicts if s["gap_seconds"] is not None]),
        "actor_order_conflicts": [s for s in sub_details if s["actor_order_conflict"]],
        "observed_last_action_seconds": {p: distribution(v) for p, v in half_ends.items()},
        "observed_tail_before_2700": [r for r in ends if r["last_record_seconds"] < 2700],
        "last_action_not_whistle_count": sum(r["sub_event_name"] != "Whistle" for r in ends),
        "red_event_metadata_conflicts": [r for r in red_details if not r["in_metadata_red_players"]],
        "structure_issues": structural,
        "new_model_fits": 0, "xg_generated": False, "parameters_changed": False,
        "identity_namespace": "wyscout: (not merged with StatsBomb player IDs)",
        "historical_lineup_first_available_at": None,
        "limits": ["max(eventSec) is last action, not true half-end; no reliable HalfEnd markers",
            "ceil-minute convention is a hypothesis, NOT a true uncertainty bound",
            "actor first/last actions are future-dependent diagnostics, NOT unbiased censoring masks",
            "data-provider shift confounds a new-season comparison; player coefficients cannot transfer by numeric ID",
            "no native xG, no fabricated goals/cards/clocks, no current pre-match probabilities"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"第二赛季真实数据接入_{stamp}"
    output.mkdir(parents=True)
    write_json(output / "全季质量实测.json", result)
    write_json(output / "逐场独立比分核对.json", scores)
    write_json(output / "换人时间假设逐条检查.json", sub_details)
    write_json(output / "红牌与半场末记录.json", {"red_events": red_details, "last_actions": ends})
    write_report(output, result)
    print(f"Full source measurement complete; no new training: {output}", flush=True)
    return output


def write_report(output, result):
    c, d = result["counts"], result["score_comparison_mismatches"]
    lines = ["# 2017/18英超：第二赛季真实数据接入与实测", "",
        "状态：原始数据已接入、全季检查已执行；**本探针不训练2017/18球员模型，不改变正式模型参数和概率。**", "",
        "## 实际覆盖", "",
        f"{result['matches']}场、{result['teams']}队、{c['events']:,}条事件；UTC日期从{result['date_min_utc']}到{result['date_max_utc']}。",
        f"与Football-Data按明确球队ID映射、日期及主客队独立连接：{result['independent_matches']}场，"
        f"两侧未匹配{len(result['unmatched_wyscout_keys'])}／{len(result['unmatched_independent_keys'])}场。",
        "", "| 核对项 | 不一致场次 |", "|---|---:|",
        *[f"| {label} | {len(d[key])} |" for key, label in (("metadata_ft", "比赛记录全场比分"),
            ("metadata_ht", "比赛记录半场比分"), ("events_ft", "主进球事件全场比分"),
            ("events_ht", "主进球事件半场比分"), ("metadata_reds", "阵容红牌计数"), ("event_reds", "原始红牌事件计数"))],
        "", f"独立全场总进球{result['independent_goals_ft']}，原始主进球事件{c['primary_goals']}。"
        f"门将失球标签{c['keeper_tag101']}条不能重复计球；乌龙{c['own_goal_tag102']}条记给对方。", "",
        "问题实例：2017-09-30切尔西—曼城，两个来源全场0–1，但曼城缺主进球标签；"
        "2018-03-05水晶宫—曼联，阵容来源的半场水晶宫为0，独立比分和进球事件均为1。"
        "不通过填补进球、改写动作或覆盖原始比分来凑齐。", "",
        "## 时间与红牌的不确定性", "",
        f"共{c['substitutions']}次换人，只有整分钟；其中{result['substitution_no_incoming_action']}次换入记录的入者整场没有动作记录。"
        f"按一个预先明确的ceil一分钟假设，有{result['ceil_minute_assumption_conflicts']}次／"
        f"{result['ceil_conflict_matches']}场与实际动作不兼容；这不是已知真实时间误差的上下界。",
        f"{result['last_action_not_whistle_count']}个半场最后记录都不是终场哨；"
        f"{len(result['observed_tail_before_2700'])}个半场的最后记录甚至不足45分钟。不能把最后动作当实际比赛结束。",
        f"原始红牌标签{c.get('tag1701',0)+c.get('tag1703',0)}条，阵容与独立比分表均支持{c['metadata_red_records']}次罚下；"
        f"{len(result['red_event_metadata_conflicts'])}条标签与球员后续动作及另外来源冲突，保留明细，不自动更正。", "",
        "## 为什么尚未训练", "",
        "直接把整分钟当秒级真实时间会错配进球与球员。按未来球员动作划窗并删除窗口内进球也不安全："
        "例如换下球员的最后动作恰好是进球，该球会决定自己被删除，却留下此前暴露时间，产生目标相关选择偏差。"
        "因此没有把这种剔窗子集喂给既有强度模型。", "",
        f"标准每半场45分钟窗口又会改变问题：{c['primary_goals_after_45_per_half']}个补时进球会被排除，"
        "不能仍称全场进球验证。分钟情景也不能恢复缺失的真实换人点或终场点。", "",
        "## 时间精度研究路线（与本探针分开执行）", "",
        "先在2015/16秒级已知数据上做时间精度退化实验：只使用较早训练段，"
        "把精确换人时间变成分钟级，在不删除进球的条件下，对照预先固定的近似时钟，测量影响。"
        "这只是测量误差实验，不能消除跨数据商差异，也不重新挑选已看过的后期测试结果。",
        "用户已选择先做上述时间误差测量；测量结果另行保存，不属于本探针的0次拟合计数。"
        "随后再确定2017/18近似规则与明确异常排除规则，先按时间划分较早训练／调参、较晚验证。"
        "Wyscout球员ID独立建模，不直接继承StatsBomb系数；xG原路线保留，但该来源不生成新xG。", "",
        "双轨市场概率、赛事重视程度、新球员先验均未调整；本报告不提供当前投注组合。", "",
        "## 来源与复用", "",
        "原始数据由Pappalardo等作者公开，CC BY 4.0；原ZIP、作者／许可元数据及下载时间留存项目目录。"
        "[原始数据集合](https://figshare.com/collections/Soccer_match_event_dataset/4415000)；"
        "[原始事件](https://figshare.com/articles/dataset/Events/7770599)；"
        "[原始赛程](https://figshare.com/articles/dataset/Matches/7770422)；"
        "[独立比分CSV](https://www.football-data.co.uk/mmz4281/1718/E0.csv)。",
        "参考[socceraction官方读取器](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/data/wyscout/loader.py)"
        "的字段和分钟处理概念，但未复制其代码、未安装该包或降级现有依赖。"
        "所有探针为本项目新增代码；本探针0次拟合，不把Wyscout数据接通表述为训练完成。", ""]
    (output / "第二赛季数据实测与待讨论方案.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    inspect(parser.parse_args().root)
