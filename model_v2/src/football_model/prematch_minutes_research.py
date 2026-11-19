"""Date-forward research of history-only minute candidates, not goal forecasts.

Reuse previously inspected fixed-window paths. The target side's actual player
identities only enter scoring, after candidates have been constructed from
strictly earlier team records. No player effects or team goal rates are fitted.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import json
from math import fsum, isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean, median, stdev

from .prematch_minutes import METHODS, TeamMinutesObservation, estimate_minutes
from .statsbomb_data import write_json


STATS_ARCHIVE = "outputs/同源窗口与时钟对照_20261022T180549874996Z/按窗口与时钟派生时段.json"
WYSCOUT_ARCHIVE = "outputs/第二赛季固定窗口球员探索_20261022T174424730288Z/固定90分钟派生时段.json"
LOOKBACK = 5


def _number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def observations_from_paths(matches, source, namespace):
    """Sum complete fixed-window stints, preserving man-short exposure deficits.

    This measures the archived research path, not an independent verification of
    source substitutions. Existing parser exclusions are not undone or repaired.
    Both halves must be contiguous [0,2700]. Bench identities are not features.
    """
    observations, fixtures, seen = [], [], set()
    for match in matches:
        mid = match["match_id"]
        if not isinstance(mid, str) or not mid.startswith(namespace + ":match:") or mid in seen:
            raise ValueError("Distinct namespaced source match IDs required")
        seen.add(mid)
        match_date = date.fromisoformat(match["match_date"])
        teams = [match["home_team_id"], match["away_team_id"]]
        if any(type(t) is not int or t < 1 for t in teams) or teams[0] == teams[1]:
            raise ValueError("Two distinct positive source team IDs required")
        if match["diagnostics"].get("fixed_window_seconds_by_half") != {"1": 2700, "2": 2700}:
            raise ValueError("Only explicit fixed 45+45 paths are comparable")
        previous, last_period = {1: 0.0, 2: 0.0}, 1
        player_seconds = {t: defaultdict(list) for t in teams}
        for stint in match["stints"]:
            period = stint["period"]
            if type(period) is not int or period not in (1, 2) or period < last_period:
                raise ValueError("Two-half chronological fixed paths required")
            last_period = period
            start = _number(stint["start_second"], "start")
            end = _number(stint["end_second"], "end")
            duration = _number(stint["duration_seconds"], "duration")
            if (end > 2700 or duration <= 0
                    or not isclose(start, previous[period], rel_tol=0, abs_tol=1e-7)
                    or not isclose(end - start, duration, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("Fixed paths must cover each half without gaps or overlap")
            previous[period] = end
            lineups = []
            for team, column in zip(teams, ("home_players", "away_players")):
                players = stint[column]
                if (not isinstance(players, list) or not 7 <= len(players) <= 11
                        or any(not isinstance(p, str) or not p.startswith(namespace + ":player:") for p in players)
                        or len(set(players)) != len(players)):
                    raise ValueError("Distinct namespaced on-pitch player IDs required")
                lineups.append(set(players))
                for player in players:
                    player_seconds[team][player].append(duration)
            if lineups[0] & lineups[1]:
                raise ValueError("A player cannot be on both teams")
        if any(not isclose(previous[p], 2700, rel_tol=0, abs_tol=1e-7) for p in (1, 2)):
            raise ValueError("Never extend a short observed half to 45 minutes")
        if set(player_seconds[teams[0]]) & set(player_seconds[teams[1]]):
            raise ValueError("A source player cannot switch teams within one match")
        # Cross-check the existing derived exposure rather than quietly changing it.
        provided = {}
        for row in match["player_minutes"]:
            key = (row["team_id"], row["player_id"])
            if key in provided or row["team_id"] not in teams:
                raise ValueError("Duplicate or foreign derived player exposure")
            provided[key] = _number(row["on_pitch_seconds"], "derived on-pitch seconds")
        for team in teams:
            seconds = {p: fsum(values) for p, values in player_seconds[team].items()}
            for player, value in seconds.items():
                if (team, player) not in provided or not isclose(provided[(team, player)], value, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("Derived player exposure disagrees with archived stints")
            if any(t == team and p not in seconds and value != 0 for (t, p), value in provided.items()):
                raise ValueError("Positive derived exposure without an on-pitch path")
            observations.append(TeamMinutesObservation(mid, team, match_date,
                {p: value / 60 for p, value in seconds.items()}, source))
        fixtures.append({"match_id": mid, "match_date": match_date, "team_ids": teams})
    if not observations:
        raise ValueError("No complete fixed-window matches")
    return observations, sorted(fixtures, key=lambda f: (f["match_date"], f["match_id"]))


def describe(values):
    """Empirical spread, not an IID confidence interval or formal hypothesis test."""
    if not values:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": mean(values), "sd": stdev(values) if len(values) > 1 else None,
            "median": median(values), "min": min(values), "max": max(values)}


def score_team(candidates, actual_minutes, earlier_team_players):
    """One common evaluation player union; actual-only IDs never feed prediction.

    Treat an omitted candidate's minutes as an omitted prediction for loss only.
    This is not a zero player-effect prior, a claim of availability, or a squad
    selection decision. Never score just the players a method chose to include.
    """
    if set(candidates) != set(METHODS) or any(c["status"] != "estimated" for c in candidates.values()):
        raise ValueError("Both candidate methods must be available for a paired score")
    validated_actual = {p: _number(v, "actual minutes") for p, v in actual_minutes.items()}
    if any(v > 90 for v in validated_actual.values()):
        raise ValueError("Actual fixed-window minutes cannot exceed 90")
    actual = {p: v for p, v in validated_actual.items() if v > 0}
    if not actual:
        raise ValueError("Positive target exposure required")
    universe = sorted(set(actual) | set().union(*(set(c["predicted_minutes"]) for c in candidates.values())))
    actual_total = fsum(actual.values())
    unseen = sorted(set(actual) - set(earlier_team_players))
    metrics = {}
    for method, candidate in candidates.items():
        projected = candidate["predicted_minutes"]
        errors = [abs(projected.get(p, 0.0) - actual.get(p, 0.0)) for p in universe]
        actual_errors = [abs(projected.get(p, 0.0) - actual[p]) for p in actual]
        omitted = sorted(p for p in actual if projected.get(p, 0.0) == 0)
        retained_absent = sorted(p for p, minutes in projected.items() if minutes > 0 and p not in actual)
        metrics[method] = {
            "common_union_player_mae_minutes": mean(errors),
            "common_union_player_rmse_minutes": sqrt(mean([x*x for x in errors])),
            "total_absolute_error_minutes": fsum(errors),
            "actual_participant_mae_minutes": mean(actual_errors),
            "omitted_actual_player_ids": omitted,
            "omitted_actual_minutes": fsum(actual[p] for p in omitted),
            "omitted_actual_minute_fraction": fsum(actual[p] for p in omitted) / actual_total,
            "retained_but_not_on_target_pitch_ids": retained_absent,
            "retained_but_not_on_target_pitch_minutes": fsum(projected[p] for p in retained_absent),
            "predicted_total_minus_actual_minutes": candidate["total_predicted_minutes"] - actual_total,
        }
    return {"evaluation_player_ids": universe, "actual_positive_players": len(actual),
            "actual_team_minutes": actual_total, "actual_deficit_relative_to_990": 990 - actual_total,
            "actual_ids_not_in_earlier_team_history": unseen,
            "actual_minutes_not_in_earlier_team_history": fsum(actual[p] for p in unseen),
            "metrics": metrics}


def evaluate_track(observations, fixtures, from_date):
    """Build each pair before giving actual target records to the scoring helper."""
    by_team, by_match_team = defaultdict(list), {}
    for record in observations:
        key = (record.match_id, record.team_id)
        if key in by_match_team:
            raise ValueError("Duplicate match/team exposure")
        by_match_team[key] = record
        by_team[record.team_id].append(record)
    seen, rows, skipped = set(), [], []
    for fixture in fixtures:
        mid, target_date = fixture["match_id"], fixture["match_date"]
        if mid in seen:
            raise ValueError("Duplicate target fixture")
        seen.add(mid)
        if target_date < from_date:
            continue
        fixture_rows = []
        for team in fixture["team_ids"]:
            actual = by_match_team[(mid, team)]
            # The API receives only strictly earlier observations, no target roster.
            history = [r for r in by_team[team] if r.match_date < target_date and r.source == actual.source]
            candidates = {method: estimate_minutes(history, team_id=team,
                prediction_date=target_date, source=actual.source, method=method, lookback=LOOKBACK)
                for method in METHODS}
            if any(c["status"] != "estimated" for c in candidates.values()):
                fixture_rows.append({"match_id": mid, "team_id": team, "match_date": target_date.isoformat(),
                                     "status": "no_earlier_history", "candidates": candidates})
                continue
            earlier_players = {p for r in history for p, m in r.minutes.items() if m > 0}
            score = score_team(candidates, actual.minutes, earlier_players)
            fixture_rows.append({"match_id": mid, "team_id": team, "match_date": target_date.isoformat(),
                                 "status": "scored", "candidates": candidates, "score": score})
        if len(fixture_rows) != 2 or any(r["status"] != "scored" for r in fixture_rows):
            skipped.append({"match_id": mid, "match_date": target_date.isoformat(), "teams": fixture_rows,
                            "reason": "whole fixture omitted from paired score when either team lacks history"})
        else:
            rows.extend(fixture_rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["match_id"]].append(row)
    match_scores = []
    for mid, teams in sorted(grouped.items()):
        if len(teams) != 2:
            raise ValueError("Both teams must share a scored match")
        result = {"match_id": mid, "match_date": teams[0]["match_date"], "methods": {}}
        for method in METHODS:
            result["methods"][method] = {key: mean([r["score"]["metrics"][method][key] for r in teams])
                for key in ("common_union_player_mae_minutes", "total_absolute_error_minutes",
                            "actual_participant_mae_minutes", "omitted_actual_minute_fraction")}
        result["recent_mean_minus_last_match_mae"] = (
            result["methods"]["recent_mean"]["common_union_player_mae_minutes"]
            - result["methods"]["last_match"]["common_union_player_mae_minutes"])
        match_scores.append(result)
    summary = {"requested_matches": sum(f["match_date"] >= from_date for f in fixtures),
        "scored_matches": len(match_scores), "scored_team_rows": len(rows), "skipped_matches": len(skipped),
        "candidate_methods": list(METHODS), "recent_mean_lookback": LOOKBACK,
        "methods": {m: {k: describe([r["methods"][m][k] for r in match_scores])
            for k in ("common_union_player_mae_minutes", "total_absolute_error_minutes",
                      "actual_participant_mae_minutes", "omitted_actual_minute_fraction")} for m in METHODS},
        "paired_match_mae_difference_recent_minus_last": describe([r["recent_mean_minus_last_match_mae"] for r in match_scores]),
        "earlier_team_unseen_actual_minutes": fsum(r["score"]["actual_minutes_not_in_earlier_team_history"] for r in rows),
        "actual_total_minutes": fsum(r["score"]["actual_team_minutes"] for r in rows),
        "target_team_exposure_deficits": describe([r["score"]["actual_deficit_relative_to_990"] for r in rows]),
        "inference": "descriptive paired match-level spread only; no IID player tests, p-values, confidence intervals or automatic method selection"}
    return summary, rows, match_scores, skipped


def load_tracks(root, stats_archive=STATS_ARCHIVE, wyscout_archive=WYSCOUT_ARCHIVE):
    stats = json.loads((root / stats_archive).read_text(encoding="utf-8"))
    tracks = [("statsbomb_2015_16_exact", stats["fixed45"]["exact"], "statsbomb", date(2016, 2, 1), stats_archive)]
    del stats
    wyscout = json.loads((root / wyscout_archive).read_text(encoding="utf-8"))
    tracks += [(f"wyscout_2017_18_{clock}", wyscout[clock], "wyscout", date(2018, 2, 1), wyscout_archive)
               for clock in ("early", "middle", "late")]
    return tracks


def report(result):
    lines = ["# 历史分钟外推：球员赛前衔接的第一项实测", "",
        "仅比较最近一场球队分钟与最近5场球队比赛平均分钟，不用目标首发、替补名单或实际换人作预测输入。"
        "每场使用严格更早日期；历史第一次公开时点未知，日期次日可用是研究假设，不是已验证历史信息重建。", "",
        "## 全部预定对照", "",
        "| 来源/时钟 | 共同比赛 | 最近一场 MAE | 最近5场 MAE | 后者−前者 |",
        "|---|---:|---:|---:|---:|"]
    for name, track in result["tracks"].items():
        s = track["summary"]
        def number(value):
            return "—" if value is None else f"{value:.4f}"
        a = s["methods"]["last_match"]["common_union_player_mae_minutes"]["mean"]
        b = s["methods"]["recent_mean"]["common_union_player_mae_minutes"]["mean"]
        delta = s["paired_match_mae_difference_recent_minus_last"]["mean"]
        lines.append(f"| {name} | {s['scored_matches']} | {number(a)} | {number(b)} | {number(delta)} |")
    lines += ["", "单位为分钟。两方案共用历史候选与实际正曝光球员的并集，先逐队计算MAE，"
        "再将两队平均为每场指标；不只评价预测过的球员，也不把球员、两队或三套时钟当成独立比赛。"
        "每种方法的逐场均值、SD、中位数和范围以及遗漏分钟均保留；这里只作描述，不作显著性检验或自动选优。", "",
        "## 未知球员与阵容缺口", "",
        "| 来源/时钟 | 最近一场遗漏实际分钟比例 | 最近5场遗漏比例 | 实际分钟中此前未在本队历史出现 |",
        "|---|---:|---:|---:|"]
    for name, track in result["tracks"].items():
        s = track["summary"]
        def percentage(value):
            return "—" if value is None else f"{value*100:.2f}%"
        a = s["methods"]["last_match"]["omitted_actual_minute_fraction"]["mean"]
        b = s["methods"]["recent_mean"]["omitted_actual_minute_fraction"]["mean"]
        unseen = s["earlier_team_unseen_actual_minutes"] / s["actual_total_minutes"] if s["actual_total_minutes"] else None
        lines.append(f"| {name} | {percentage(a)} | {percentage(b)} | {percentage(unseen)} |")
    lines += ["", "前两列是逐场两队平均的遗漏比例；最后一列为实际分钟加权比例，口径不同。"
        "此前未在本队出现不等于现实新秀、从未参加职业比赛或球员效果未知。"
        "遗漏候选在误差计算中按未预测分钟计罚，不会生成零球员效果或已知当前名单。", "",
        "## 适用边界", "",
        "- 两季此前都已查看，本轮是新增问题的探索性测量，不是新的盲测。2017/18与2015/16的提供商、年份、时钟同时不同，不能将差异单独归因跨赛季。",
        "- 目标仅为每半场固定45分钟窗口，排除补时；StatsBomb按原秒级时钟，Wyscout三套只是近似时钟。后者不是精确真实分钟。",
        "- 只使用已有可解析路径；StatsBomb原380场中的2场窗口不足、Wyscout原380场中的6场源问题仍不补造。缺场会影响最近5场历史，选择性未知。",
        "- 阵容、转会、伤停、停赛、位置与守门员轮换没有接通。历史红牌或临时离场引起的球队分钟缺额不补到990；输出不是可直接进入Lineup的完整合法阵容。",
        "- 没有假定最近5场一定更好；没有按本次目标选择窗口、衰减或混合权重。下一步应补当前可用名单和预计首发/替补情景，再独立验证。",
        "- 未拟合球员效果、球队模型、条件参考阵容映射或进球率；未发布1X2、半全场或当前投注建议。分钟MAE改善不能直接当作进球或胜平负概率改善。", "",
        "## 来源与执行", "",
        f"去重比赛合计{result['distinct_matches']}场（同一Wyscout比赛的三时钟只计一次；不假定同队连续比赛统计独立）；"
        f"共{result['candidate_team_projections']}条球队分钟候选，0次新增优化器拟合。全部输入档案和逐场候选见结果JSON。", "",
        "[StatsBomb官方开放数据](https://github.com/hudl/open-data)；"
        "[Wyscout作者数据集合](https://figshare.com/collections/Soccer_match_event_dataset/4415000)。"
        "沿用本地已接入数据及原适用条款，未重新分发数据。", "",
        "实验设计技能使对照限定为同场、同队、相同评价球员并集且历史严格更早；"
        "统计分析技能使误差按比赛汇总、遗漏样本和全部候选保留，避免把球员行数当独立样本。", ""]
    return "\n".join(lines)


def run(root, *, tracks=None):
    root = Path(root).resolve()
    output = root / "outputs" / ("赛前分钟外推研究_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    settings = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "methods": list(METHODS),
        "lookback_team_matches": LOOKBACK, "prediction_inputs": "same team/source, strictly earlier dates only; no target roster",
        "evaluation": "common player union; team metrics averaged into match; all planned arms retained",
        "source_availability": "historical first publication unknown; next-day usability assumed for research",
        "new_player_effect_prior": False, "core_goal_model_changed": False, "new_optimizer_fits": 0}
    write_json(output / "研究设置.json", settings)
    result, distinct, projections = {"settings": settings, "tracks": {}}, set(), 0
    try:
        for name, matches, namespace, from_date, archive in (load_tracks(root) if tracks is None else tracks):
            if name in result["tracks"]:
                raise ValueError("Distinct source/clock track names required")
            observations, fixtures = observations_from_paths(matches, name, namespace)
            summary, rows, scored, skipped = evaluate_track(observations, fixtures, from_date)
            write_json(output / f"{name}_逐队候选与误差.json", {"rows": rows, "skipped_matches": skipped})
            write_json(output / f"{name}_逐场配对.json", scored)
            result["tracks"][name] = {"source_archive": str(root / archive), "from_date": from_date.isoformat(),
                "retained_source_matches": len(fixtures), "summary": summary}
            distinct.update(r["match_id"] for r in scored)
            projections += len(rows) * len(METHODS)
            print(f"Minute research {name}: retained={len(fixtures)}, scored={len(scored)}, skipped={len(skipped)}", flush=True)
        result.update(distinct_matches=len(distinct), candidate_team_projections=projections, new_optimizer_fits=0)
        write_json(output / "分钟研究结果.json", result)
        (output / "分钟外推研究说明.md").write_text(report(result), encoding="utf-8")
        print(f"Minute research complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"status": "incomplete", "error": f"{type(error).__name__}: {error}",
            "completed_tracks": list(result["tracks"]), "new_optimizer_fits": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    run(parser.parse_args().root)
