"""Earlier-window substitution-clock sensitivity, not a Wyscout timing rule.

Only Substitution clocks are rounded. Goal, dismissal, temporary off/on and
Half End clocks remain exact. Rounded streams are replayed independently;
off-pitch shooters are measured, not silently put back in an assumed lineup.
No later-season event file is needed by this experiment.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import json
from math import ceil, exp, floor, isfinite
from pathlib import Path
from statistics import mean, median

from .full_season_experiment import TRAIN_BEFORE
from .joint_stint_model import JointStintIntensityModel, UnknownStintIdentityError
from .statsbomb_data import write_json
from .statsbomb_stints import StatsBombParseError, parse_match as parse_verified_exact


SCENARIOS = ("exact", "floor", "nearest_half_up", "ceil")


class TimingReplayError(ValueError):
    """A specified hypothetical clock cannot be replayed without repair."""


def rounded_second(second, scenario, half_end):
    """Return the specified minute clock and whether true Half End clipped it."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown pre-specified timing scenario")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x)
           for x in (second, half_end)) or not 0 <= second <= half_end or half_end <= 0:
        raise ValueError("Finite within-half substitution and Half End clocks required")
    value = (second if scenario == "exact" else 60 * floor(second / 60)
             if scenario == "floor" else 60 * floor(second / 60 + .5)
             if scenario == "nearest_half_up" else 60 * ceil(second / 60))
    return min(float(value), float(half_end)), value > half_end


def _seconds(text):
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _prepare(metadata, events, lineups):
    # Validate the ORIGINAL stream only. The rounded stream deliberately does
    # not pass through the source parser's on-pitch shooter requirement.
    try:
        verified = parse_verified_exact(metadata, events, lineups)
    except StatsBombParseError as error:
        raise TimingReplayError(f"Original source is not replayable: {error}") from error
    teams = (verified["home_team_id"], verified["away_team_id"])
    roster = {entry["team_id"]: {p["player_id"]: p.get("player_name") or str(p["player_id"])
                                for p in entry["lineup"]} for entry in lineups}
    initial = {}
    rows = []
    for event in events:
        kind = event["type"]["name"]
        if kind == "Starting XI":
            initial[event["team"]["id"]] = {p["player"]["id"] for p in event["tactics"]["lineup"]}
        field = "foul_committed" if kind == "Foul Committed" else "bad_behaviour"
        red = kind in {"Foul Committed", "Bad Behaviour"} and event.get(field, {}).get("card", {}).get("name") in {
            "Red Card", "Second Yellow"}
        if kind in {"Shot", "Own Goal Against", "Substitution", "Player Off", "Player On"} or red:
            rows.append({"period": event["period"], "second": _seconds(event["timestamp"]),
                         "index": event["index"], "kind": kind, "red": red, "event": event})
    return {"verified": verified, "teams": teams, "roster": roster,
            "initial": initial, "rows": rows}


def _replay(prepared, scenario, exact_card_membership=None):
    source = prepared["verified"]
    h, a = prepared["teams"]
    active = {t: set(prepared["initial"][t]) for t in (h, a)}
    temporary, removed, dismissed = ({t: set() for t in (h, a)} for _ in range(3))
    score, stints, goals_attribution, conflicts, substitutions = [0, 0], [], [], [], []
    card_membership = {}
    for period in (1, 2):
        end = source["diagnostics"]["period_seconds"][str(period)]
        rows = []
        for original in prepared["rows"]:
            if original["period"] != period:
                continue
            row = dict(original)
            if row["kind"] == "Substitution":
                row["second"], clipped = rounded_second(row["second"], scenario, end)
                substitutions.append({"event_id": row["event"]["id"], "period": period,
                                      "original_second": original["second"], "scenario_second": row["second"],
                                      "clipped_at_true_half_end": clipped})
            rows.append(row)
        # Explicit tie convention, not a claim about true simultaneous order.
        rows.sort(key=lambda row: (row["second"], row["index"]))
        start, start_score, targets, goal_ids = 0.0, tuple(score), [0, 0], []

        def close(clock):
            nonlocal start, start_score, targets, goal_ids
            elapsed = clock - start
            if elapsed < 0 or elapsed == 0 and any(targets):
                raise TimingReplayError("Goal cannot be assigned to positive hypothetical exposure")
            if elapsed:
                if not all(7 <= len(active[t]) <= 11 for t in (h, a)):
                    raise TimingReplayError("Hypothetical on-pitch count outside 7..11")
                stints.append({"stint_id": f"{source['match_id']}:timing:{scenario}:stint:{len(stints)+1}",
                    "period": period, "start_second": start, "end_second": clock, "duration_seconds": elapsed,
                    "home_players": [f"statsbomb:player:{p}" for p in sorted(active[h])],
                    "away_players": [f"statsbomb:player:{p}" for p in sorted(active[a])],
                    "start_home_goals": start_score[0], "start_away_goals": start_score[1],
                    "home_score_difference": start_score[0] - start_score[1],
                    "home_man_difference": len(active[h]) - len(active[a]),
                    "home_goals": targets[0], "away_goals": targets[1], "goal_event_ids": list(goal_ids)})
            start, start_score, targets, goal_ids = clock, tuple(score), [0, 0], []

        for row in rows:
            event, kind, clock = row["event"], row["kind"], row["second"]
            team, pid = event["team"]["id"], event["player"]["id"]
            if team not in active or pid not in prepared["roster"][team]:
                raise TimingReplayError("Source actor missing from roster")
            if kind in {"Shot", "Own Goal Against"}:
                is_goal = kind == "Own Goal Against" or event.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                if pid not in active[team]:
                    conflicts.append({"event_id": event["id"], "period": period, "second": clock,
                                      "player_id": f"statsbomb:player:{pid}", "is_goal": is_goal,
                                      "kind": "actor_not_in_hypothetical_lineup"})
                if is_goal:
                    side = (1 if team == h else 0) if kind == "Own Goal Against" else (0 if team == h else 1)
                    goals_attribution.append({"event_id": event["id"], "period": period, "second": clock,
                        "scoring_team_id": (h, a)[side], "score_before": list(score),
                        "home_players": [f"statsbomb:player:{p}" for p in sorted(active[h])],
                        "away_players": [f"statsbomb:player:{p}" for p in sorted(active[a])],
                        "actor_on_assumed_pitch": pid in active[team]})
                    targets[side] += 1
                    goal_ids.append(event["id"])
                    close(clock)
                    score[side] += 1
                    start_score = tuple(score)
            elif kind == "Substitution":
                incoming = event["substitution"]["replacement"]["id"]
                if (pid not in active[team] | temporary[team] or incoming not in prepared["roster"][team]
                        or incoming in active[team] | temporary[team] | removed[team] | dismissed[team]):
                    raise TimingReplayError(f"{event['id']}: rounded substitution conflicts with player state")
                close(clock)
                active[team].discard(pid)
                temporary[team].discard(pid)
                removed[team].add(pid)
                active[team].add(incoming)
            elif kind == "Player Off":
                if pid in dismissed[team]:
                    continue
                if pid not in active[team]:
                    raise TimingReplayError(f"{event['id']}: Player Off interleaves illegally with rounded substitution")
                close(clock)
                active[team].remove(pid)
                (removed if event.get("player_off", {}).get("permanent", False) else temporary)[team].add(pid)
            elif kind == "Player On":
                if pid not in temporary[team] or pid in removed[team] | dismissed[team]:
                    raise TimingReplayError(f"{event['id']}: Player On interleaves illegally with rounded substitution")
                close(clock)
                temporary[team].remove(pid)
                active[team].add(pid)
            elif row["red"]:
                if pid in dismissed[team]:
                    continue
                on_pitch = pid in active[team] | temporary[team]
                card_membership[event["id"]] = on_pitch
                if exact_card_membership is not None and exact_card_membership[event["id"]] != on_pitch:
                    raise TimingReplayError(f"{event['id']}: rounded substitution changes on/off-pitch dismissal status")
                if on_pitch:
                    close(clock)
                    active[team].discard(pid)
                    temporary[team].discard(pid)
                dismissed[team].add(pid)
        close(end)
    if score != source["diagnostics"]["event_goals"]:
        raise TimingReplayError("Hypothetical replay did not retain all original goals")
    seconds = defaultdict(float)
    for stint in stints:
        for team, column in ((h, "home_players"), (a, "away_players")):
            for pid in stint[column]:
                seconds[(team, pid)] += stint["duration_seconds"]
    return {"match_id": source["match_id"], "match_date": source["match_date"],
        "source_match_id": source["source_match_id"], "home_team_id": h, "away_team_id": a,
        "stints": stints,
        "player_minutes": [{"team_id": team, "player_id": f"statsbomb:player:{pid}", "player_name": name,
            "started": pid in prepared["initial"][team], "on_pitch_seconds": seconds[(team, f"statsbomb:player:{pid}")],
            "minutes": seconds[(team, f"statsbomb:player:{pid}")] / 60}
            for team in (h, a) for pid, name in sorted(prepared["roster"][team].items())],
        "diagnostics": {"scenario": scenario, "event_goals": score, "all_source_goals_preserved": True,
            "period_seconds": dict(source["diagnostics"]["period_seconds"]),
            "total_duration_seconds": sum(s["duration_seconds"] for s in stints),
            "substitutions": substitutions, "clipped_substitutions": sum(s["clipped_at_true_half_end"] for s in substitutions),
            "actor_conflicts": conflicts, "goal_attribution": goals_attribution,
            "source_on_pitch_card_membership": card_membership,
            "tie_policy": "transformed period second, then unchanged original event index; an explicit convention",
            "usage": "hypothetical substitution-clock sensitivity, not pre-match probabilities or a Wyscout clock rule",
            "unchanged_information": "exact goals, score changes, dismissals, temporary off/on, true Half End; no xg target"}}


def replay_match(metadata, events, lineups, scenario="exact"):
    """Return goal-only model-compatible paths; reject illegal state interleaving."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown pre-specified timing scenario")
    prepared = _prepare(metadata, events, lineups)
    exact = _replay(prepared, "exact")
    return exact if scenario == "exact" else _replay(
        prepared, scenario, exact["diagnostics"]["source_on_pitch_card_membership"])


def measure_match(metadata, events, lineups):
    """Evaluate all four fixed scenarios without fitting or reading other games."""
    prepared = _prepare(metadata, events, lineups)
    exact = _replay(prepared, "exact")
    paths, failures = {"exact": exact}, {}
    for scenario in SCENARIOS[1:]:
        try:
            paths[scenario] = _replay(prepared, scenario, exact["diagnostics"]["source_on_pitch_card_membership"])
        except TimingReplayError as error:
            failures[scenario] = str(error)
    common = not failures
    result = {"match_id": exact["match_id"], "match_date": exact["match_date"],
              "common_supported": common, "failures": failures, "scenarios": paths, "comparisons": {}}
    if common:
        baseline = {p["player_id"]: p for p in exact["player_minutes"]}
        exact_goals = {g["event_id"]: g for g in exact["diagnostics"]["goal_attribution"]}
        exact_ids = {p for p, row in baseline.items() if row["on_pitch_seconds"] > 0}
        for scenario in SCENARIOS:
            path = paths[scenario]
            changed = []
            for goal in path["diagnostics"]["goal_attribution"]:
                original = exact_goals[goal["event_id"]]
                if (goal["period"], goal["second"], goal["score_before"]) != (
                        original["period"], original["second"], original["score_before"]):
                    raise TimingReplayError("A non-substitution goal clock or score was changed")
                if any(goal[k] != original[k] for k in ("home_players", "away_players")):
                    changed.append(goal["event_id"])
            minutes = [{"player_id": p["player_id"], "minute_difference": p["minutes"] - baseline[p["player_id"]]["minutes"]}
                       for p in path["player_minutes"]]
            ids = {p["player_id"] for p in path["player_minutes"] if p["on_pitch_seconds"] > 0}
            starters = {p["player_id"] for p in path["player_minutes"] if p["started"]}
            base_starters = {p for p, row in baseline.items() if row["started"]}
            result["comparisons"][scenario] = {
                "duration_difference_seconds": path["diagnostics"]["total_duration_seconds"] - exact["diagnostics"]["total_duration_seconds"],
                "period_duration_differences_seconds": {str(period): sum(s["duration_seconds"] for s in path["stints"] if s["period"] == period)
                    - sum(s["duration_seconds"] for s in exact["stints"] if s["period"] == period) for period in (1, 2)},
                "goal_difference": sum(path["diagnostics"]["event_goals"]) - sum(exact["diagnostics"]["event_goals"]),
                "changed_goal_lineup_event_ids": changed, "player_minute_differences": minutes,
                "max_abs_player_minute_difference": max(abs(row["minute_difference"]) for row in minutes),
                "sum_abs_player_minute_difference": sum(abs(row["minute_difference"]) for row in minutes),
                "positive_exposure_player_ids_lost": sorted(exact_ids - ids),
                "positive_exposure_player_ids_added": sorted(ids - exact_ids),
                "starter_ids_changed": starters != base_starters,
                "clip_count": path["diagnostics"]["clipped_substitutions"],
                "shot_actor_conflicts": len(path["diagnostics"]["actor_conflicts"]),
                "goal_actor_conflicts": sum(c["is_goal"] for c in path["diagnostics"]["actor_conflicts"])}
    return result


def training_path_intensities(model, exact_match):
    """Explicit IN-SAMPLE coefficient evaluation, not the prediction interface.

    Only fitted match IDs strictly before the cutoff are accepted. Missing
    fitted identities reject the entire match, including zero-exposure debuts
    created by rounding a late substitution to Half End.
    """
    if exact_match["match_id"] not in model.reference_match_ids or date.fromisoformat(exact_match["match_date"]) >= model.before_date:
        raise ValueError("Training-path evaluation requires a fitted earlier match")
    h, a = exact_match["home_team_id"], exact_match["away_team_id"]
    if {h, a} - model.teams:
        raise UnknownStintIdentityError("Unknown training-path team")
    current_ids = {p for s in exact_match["stints"] for col in ("home_players", "away_players") for p in s[col]}
    if model.include_players and current_ids - model.players:
        raise UnknownStintIdentityError(f"Unknown exact-path players: {sorted(current_ids - model.players)}")
    def coefficient(name):
        return float(model.coefficients[model.feature_index[name]])
    expected, rates = [0.0, 0.0], []
    for stint in exact_match["stints"]:
        pair = []
        for side, (own, opponent, own_col, other_col) in enumerate((
                (h, a, "home_players", "away_players"), (a, h, "away_players", "home_players"))):
            sign = 1 if side == 0 else -1
            eta = (coefficient("intercept") + (coefficient("home_advantage") if side == 0 else 0)
                   + coefficient(f"team_attack:{own}") - coefficient(f"team_defence:{opponent}")
                   + coefficient("own_score_difference") * sign * stint["home_score_difference"]
                   + coefficient("own_man_difference") * sign * stint["home_man_difference"])
            if model.include_players:
                for team, col, kind, multiplier in ((own, own_col, "attack", 1), (opponent, other_col, "defence", -1)):
                    current, reference = set(stint[col]), model.reference_fractions[team]
                    eta += sum(multiplier * (float(p in current) - reference.get(p, 0.0))
                               * coefficient(f"player_{kind}:{p}") for p in current | reference.keys())
            rate = exp(eta)
            if not isfinite(rate) or rate <= 0:
                raise FloatingPointError("Nonfinite training-path intensity, no clipping")
            pair.append(rate)
            expected[side] += stint["duration_seconds"] / 5400 * rate
        rates.append({"stint_id": stint["stint_id"], "duration_seconds": stint["duration_seconds"],
                      "home_intensity_per90": pair[0], "away_intensity_per90": pair[1]})
    return {"match_id": exact_match["match_id"], "path_integrated_home_intensity": expected[0],
            "path_integrated_away_intensity": expected[1], "stints": rates,
            "usage": "training-in-sample exact-state-path sensitivity only; not generalization or unconditional FT expectation"}


def fit_sensitivity(common_paths, destination):
    """Exactly four fixed-ridge goals fits. Root task may explicitly execute."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    evaluated, skipped = {}, {}
    for scenario in SCENARIOS:
        model = JointStintIntensityModel(include_players=True, response_kind="goals", player_ridge=10.0).fit(
            common_paths[scenario], before_date=TRAIN_BEFORE)
        model.save(destination / f"{scenario}_goals_players_ridge10.json")
        evaluated[scenario], skipped[scenario] = {}, []
        for match in common_paths["exact"]:
            try:
                evaluated[scenario][match["match_id"]] = training_path_intensities(model, match)
            except UnknownStintIdentityError as error:
                skipped[scenario].append({"match_id": match["match_id"], "reason": str(error)})
    shared = sorted(set.intersection(*(set(evaluated[s]) for s in SCENARIOS)))
    if not shared:
        raise ValueError("No exact training paths supported by all four fitted identity sets")
    results = {}
    for scenario in SCENARIOS:
        rows, relative_sum, duration_sum, maximum = [], 0.0, 0.0, 0.0
        for mid in shared:
            baseline, candidate = evaluated["exact"][mid], evaluated[scenario][mid]
            base_total = baseline["path_integrated_home_intensity"] + baseline["path_integrated_away_intensity"]
            total = candidate["path_integrated_home_intensity"] + candidate["path_integrated_away_intensity"]
            rows.append({"match_id": mid, "exact_path_integrated_intensity": base_total,
                         "scenario_model_on_exact_path_intensity": total, "difference": total - base_total})
            for a, b in zip(baseline["stints"], candidate["stints"]):
                for col in ("home_intensity_per90", "away_intensity_per90"):
                    relative = abs(b[col] / a[col] - 1)
                    relative_sum += a["duration_seconds"] * relative
                    duration_sum += a["duration_seconds"]
                    maximum = max(maximum, relative)
        results[scenario] = {"matches": rows,
            "mean_abs_integrated_intensity_difference": sum(abs(r["difference"]) for r in rows) / len(rows),
            "duration_weighted_mean_abs_relative_stint_intensity_change": relative_sum / duration_sum,
            "max_abs_relative_stint_intensity_change": maximum}
    return {"fixed_player_ridge": 10.0, "model_fits": 4, "common_exact_evaluation_match_ids": shared,
            "skipped_exact_path_evaluations": skipped, "scenarios": results,
            "usage": "training-in-sample same exact path; no held-out accuracy, likelihood comparison, 1X2 or Wyscout validity claim"}


def complete_report(root, output):
    """Complete an executed experiment's report without fitting again.

    A bounded second pass reads ONLY the recorded earlier event IDs for the
    whole-requested-window clock-shift denominator (including replay rejects).
    The source first pass and this report pass never inspect later events.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root / "data/statsbomb/premier-league-2015-16"
    settings = json.loads((output / "测量设置.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "分钟精度测量汇总.json").read_text(encoding="utf-8"))
    read_ids = json.loads((output / "读取事件比赛ID.json").read_text(encoding="utf-8"))
    metadata = {m["match_id"]: m for m in json.loads((source / "matches_2_27.json").read_text(encoding="utf-8"))}
    if len(set(read_ids)) != len(read_ids) or any(date.fromisoformat(metadata[mid]["match_date"]) >= TRAIN_BEFORE for mid in read_ids):
        raise ValueError("Clock-shift reporting accepts distinct recorded earlier IDs only")
    movements, clipped, event_count, substitution_count = {s: [] for s in SCENARIOS}, defaultdict(int), 0, 0
    for mid in read_ids:
        events = json.loads((source / f"events_{mid}.json").read_text(encoding="utf-8"))
        event_count += len(events)
        ends = {e["period"]: _seconds(e["timestamp"]) for e in events if e["type"]["name"] == "Half End"}
        for event in events:
            if event["type"]["name"] != "Substitution":
                continue
            substitution_count += 1
            second = _seconds(event["timestamp"])
            for scenario in SCENARIOS:
                value, was_clipped = rounded_second(second, scenario, ends[event["period"]])
                movements[scenario].append(abs(value - second))
                clipped[scenario] += was_clipped
    shifts = {"scope": "all recorded earlier source matches, before common-scenario exclusions",
        "source_matches": len(read_ids), "source_events": event_count, "source_substitutions": substitution_count,
        "date_min": min(metadata[mid]["match_date"] for mid in read_ids),
        "date_max": max(metadata[mid]["match_date"] for mid in read_ids),
        "additional_later_event_files_read": 0, "model_fits_in_report_pass": 0,
        "source_event_passes": "replay pass plus this earlier-only clock-summary pass; distinct IDs unchanged",
        "scenarios": {s: {"n": len(v), "mean_abs_shift_seconds": mean(v) if v else None,
            "median_abs_shift_seconds": median(v) if v else None, "max_abs_shift_seconds": max(v) if v else None,
            "changed": sum(x > 0 for x in v), "clipped": clipped[s]} for s, v in movements.items()}}
    write_json(output / "全部较早场次换人秒移动汇总.json", shifts)
    sensitivity_path = output / "训练内同精确路径强度敏感性.json"
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8")) if sensitivity_path.exists() else None
    labels = {"exact": "原秒级", "floor": "向下取整", "nearest_half_up": "四舍五入（半分向上）", "ceil": "向上取整"}
    later_metadata_count = sum(date.fromisoformat(m["match_date"]) >= TRAIN_BEFORE for m in metadata.values())
    def decimal_text(value):
        return "—" if value is None else f"{value:.3f}"
    lines = ["# 换人分钟精度：真实测量与强度敏感性", "",
        ("结论：分钟取整不是无影响操作；整体强度变化较小，但个别短时段变化明显。" if sensitivity is not None
         else "状态：已完成换人时钟与归属变化测量；本次没有进行模型拟合。") +
        "这些是较早训练数据上的测量与拟合敏感性，**不是预测准确率提升，也不能确定Wyscout采用哪套时钟规则。**", "",
        "## 范围和预先固定的规则", "",
        f"只读取{shifts['date_min']}至{shifts['date_max']}的{len(read_ids)}场（{event_count:,}条事件、{substitution_count}次换人）。"
        f"日期严格早于2015-12-01；后{later_metadata_count}场事件文件未读取。汇总二次扫描同一组较早文件，没有新增训练。",
        "仅把换人秒数按期内时钟变成分钟：向下、半分向上的四舍五入、向上。"
        "红牌、临时离返场、进球与比分改变、真实半场终点全部保持秒级；越过已知终点时裁到终点并记录。"
        "同秒按原事件索引排序，这是实验约定，不是恢复真实同时事件先后。", "",
        "| 时钟 | 平均绝对移动（秒） | 中位数（秒） | 最大（秒） | 终点裁剪 |",
        "|---|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        row = shifts["scenarios"][scenario]
        lines.append(f"| {labels[scenario]} | {decimal_text(row['mean_abs_shift_seconds'])} | {decimal_text(row['median_abs_shift_seconds'])} | {decimal_text(row['max_abs_shift_seconds'])} | {row['clipped']} |")
    lines += ["", "## 共同支持样本，不隐藏拒绝", "",
        f"原始{summary['requested_matches']}场解析失败{summary['original_source_failures']}场；"
        f"{summary['scenario_failed_matches']}场的某种取整与临离／换人等状态先后冲突，未自动修复。"
        f"四情景都能重放的共同样本为{summary['common_matches']}场；以下拟合只比较这一组，不能外推到被排除比赛。", "",
        "| 时钟 | 全部保留进球 | 进球时双方阵容改变次数 | 射门者不在假定阵容次数 | 最大球员分钟偏差 |",
        "|---|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        row = summary["scenarios"][scenario]
        lines.append(f"| {labels[scenario]} | {row['goals_preserved_in_common_matches']} | {row['changed_goal_lineups']} | {row['shot_actor_conflicts']} | {decimal_text(row['max_abs_player_minute_difference'])} |")
    lines += ["", "所有共同场的全场及逐半场暴露时间守恒（浮点尾差约1e-12秒），进球保留补时且不删球。"
        "射门者冲突是取整造成的假定阵容错配测量，不将射门者自动补入场上，也不把它说成原始事件错误。"
        "首发名册没有改变；场中首分钟内的阵容仍可能因取整改变。身份覆盖明细见JSON。", ""]
    if sensitivity is not None:
        shared = len(sensitivity["common_exact_evaluation_match_ids"])
        lines += ["## 真实拟合：固定设置的四套进球模型", "",
            f"已完成{sensitivity['model_fits']}次拟合，球员ridge固定10，球队／状态惩罚沿用既有默认；未搜索新超参数。"
            f"四模型统一在{shared}场原秒级路径上检查拟合后的条件强度。"
            "这是训练内敏感性，不经过赛前预测接口，不解除该接口的时间限制。", "",
            "| 取整训练模型 | 每场积分强度平均绝对变化（进球单位） | 时长加权平均绝对相对时段变化 | 最大绝对相对时段变化 |",
            "|---|---:|---:|---:|"]
        for scenario in SCENARIOS[1:]:
            row = sensitivity["scenarios"][scenario]
            lines.append(f"| {labels[scenario]} | {row['mean_abs_integrated_intensity_difference']:.6f} | {row['duration_weighted_mean_abs_relative_stint_intensity_change']:.3%} | {row['max_abs_relative_stint_intensity_change']:.3%} |")
        lines += ["", "积分强度是在已知实际比分和人员路径下对强度积分，不是无条件全场期望进球，也不是1X2胜率。"
            "最大值是所有训练时段中的极值，不是置信上界。不能根据这一表的较小变化选定Wyscout规则，"
            "也不能把训练内稳定性当时间外泛化能力。模型参数4份保存在本目录models/。", ""]
    lines += ["## 对2017/18训练的影响与下一步", "",
        "本实验支持把换人精度单独报告为误差来源，但没有解决Wyscout的整分钟语义、"
        "未知终场、缺主进球标签和矛盾红牌。Wyscout仍未做正式球员拟合；不能套用这里的误差百分比作为其置信区间。",
        "建议下一轮先明确两种层级：比分完整的球队级对照，与带计时假设的球员级探索。"
        "球员级用预先固定的近似情景分别报告，不按后期成绩择优拼接；"
        "异常整场明列原因，不能用未来进球决定的删窗规则直接构造正式似然。"
        "这些新来源时钟与异常规则需要与用户讨论后再启动第二赛季拟合。", "",
        "StatsBomb与Wyscout提供商不同、ID空间不同，不直接迁移数字ID或球员系数。"
        "第二赛季只能先称跨赛季／跨提供商方法复验，不能单独归因于年份。"
        "原xG分轨、市场双轨、新球员严格未知处理均保留；当前正式赛前概率未变。", "",
        "来源：[Hudl/StatsBomb Open Data](https://github.com/hudl/open-data)，"
        "原始数据、许可与历史下载记录仍留项目目录data/，不随源码包分发。"
        "测量设置、逐场拒绝原因、共同场路径、参数及逐场强度结果在本目录JSON中可复核。", ""]
    (output / "换人分钟精度测量说明.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def run(root, fit_models=False):
    root = Path(root).resolve()
    source = root / "data/statsbomb/premier-league-2015-16"
    metadata = json.loads((source / "matches_2_27.json").read_text(encoding="utf-8"))
    earlier = sorted((m for m in metadata if date.fromisoformat(m["match_date"]) < TRAIN_BEFORE),
                     key=lambda m: (m["match_date"], m["match_id"]))
    if not earlier:
        raise ValueError("No earlier training metadata")
    output = root / "outputs" / ("换人分钟精度测量_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    settings = {"before_date": TRAIN_BEFORE.isoformat(), "selected_metadata_matches": len(earlier),
        "scenarios": list(SCENARIOS), "rounding": "period-local seconds; floor, half-up, ceil to 60 seconds; clip only at true Half End",
        "ties": "unchanged original event index after transformed clock; convention only",
        "later_event_files_read": 0, "fit_models": fit_models, "fixed_player_ridge": 10.0,
        "limits": "only substitution precision changed; exact goal/red/off/on/HalfEnd data remain; not all Wyscout timing errors"}
    write_json(output / "测量设置.json", settings)
    paths = {s: [] for s in SCENARIOS}
    records, failures, read_ids = [], [], []
    try:
        for meta in earlier:
            mid = meta["match_id"]
            events = json.loads((source / f"events_{mid}.json").read_text(encoding="utf-8"))
            lineups = json.loads((source / f"lineups_{mid}.json").read_text(encoding="utf-8"))
            read_ids.append(mid)
            try:
                measured = measure_match(meta, events, lineups)
                records.append({k: v for k, v in measured.items() if k != "scenarios"})
                if measured["common_supported"]:
                    for scenario in SCENARIOS:
                        paths[scenario].append(measured["scenarios"][scenario])
            except TimingReplayError as error:
                failures.append({"source_match_id": mid, "date": meta["match_date"], "reason": str(error)})
            if len(read_ids) % 25 == 0:
                print(f"Measured earlier matches {len(read_ids)}/{len(earlier)}; common={len(paths['exact'])}", flush=True)
        write_json(output / "读取事件比赛ID.json", read_ids)
        write_json(output / "逐场分钟误差测量.json", records)
        write_json(output / "原始流拒绝.json", failures)
        write_json(output / "四情景共同场时段.json", paths)
        summary = {"requested_matches": len(earlier), "common_matches": len(paths["exact"]),
                   "original_source_failures": len(failures), "scenario_failed_matches": sum(not r["common_supported"] for r in records),
                   "later_event_files_read": 0, "scenarios": {}}
        for scenario in SCENARIOS:
            comparisons = [r["comparisons"][scenario] for r in records if r["common_supported"]]
            identities = {p for m in paths[scenario] for s in m["stints"] for col in ("home_players", "away_players") for p in s[col]}
            summary["scenarios"][scenario] = {
                "clip_count": sum(r["clip_count"] for r in comparisons),
                "changed_goal_lineups": sum(len(r["changed_goal_lineup_event_ids"]) for r in comparisons),
                "shot_actor_conflicts": sum(r["shot_actor_conflicts"] for r in comparisons),
                "goal_actor_conflicts": sum(r["goal_actor_conflicts"] for r in comparisons),
                "max_abs_player_minute_difference": max((r["max_abs_player_minute_difference"] for r in comparisons), default=None),
                "sum_abs_player_minute_difference": sum(r["sum_abs_player_minute_difference"] for r in comparisons),
                "positive_exposure_unique_players": len(identities), "positive_exposure_player_ids": sorted(identities),
                "starter_coverage_changed_matches": sum(r["starter_ids_changed"] for r in comparisons),
                "max_abs_duration_difference_seconds": max((abs(r["duration_difference_seconds"]) for r in comparisons), default=None),
                "goals_preserved_in_common_matches": sum(sum(m["diagnostics"]["event_goals"]) for m in paths[scenario])}
        write_json(output / "分钟精度测量汇总.json", summary)
        if fit_models:
            sensitivity = fit_sensitivity(paths, output / "models")
            write_json(output / "训练内同精确路径强度敏感性.json", sensitivity)
        complete_report(root, output)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"error": f"{type(error).__name__}: {error}",
                   "status": "incomplete; partial files are not a completed experiment"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--fit-models", action="store_true")
    args = parser.parse_args()
    print(run(args.root, fit_models=args.fit_models))
