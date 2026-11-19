"""Local chronological experiment for retrospective team/player intensity.

No network calls, market fusion, pre-match probabilities or unseen-player
defaults. Test matches never choose the player penalty. All models use the
same actual historical state paths; comparisons use identical match cohorts.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import json
from math import exp
from pathlib import Path

from .joint_stint_model import JointStintIntensityModel, UnknownStintIdentityError
from .statsbomb_data import parse_season, write_json


TRAIN_BEFORE = date(2015, 12, 1)
TEST_FROM = date(2016, 2, 1)
PLAYER_RIDGES = (10.0, 100.0, 1000.0)
TEAM_NAMES = {1: "Arsenal", 22: "Leicester", 23: "Watford", 24: "Liverpool",
              25: "Southampton", 26: "Swansea", 27: "West Brom", 28: "Bournemouth",
              29: "Everton", 30: "Stoke", 31: "Crystal Palace", 33: "Chelsea",
              36: "Man City", 37: "Newcastle", 38: "Tottenham", 39: "Man United",
              40: "West Ham", 41: "Sunderland", 56: "Norwich", 59: "Aston Villa"}


def temporal_split(matches, train_before=TRAIN_BEFORE, test_from=TEST_FROM):
    if type(train_before) is not date or type(test_from) is not date or train_before >= test_from:
        raise ValueError("Two ordered calendar dates required")
    partitions = [[], [], []]
    seen = set()
    for match in sorted(matches, key=lambda m: (m["match_date"], m["match_id"])):
        if match["match_id"] in seen:
            raise ValueError("Duplicate match in chronological experiment")
        seen.add(match["match_id"])
        day = date.fromisoformat(match["match_date"])
        partitions[0 if day < train_before else 1 if day < test_from else 2].append(match)
    if not all(partitions):
        raise ValueError("Each chronological segment needs observations")
    return tuple(partitions)


def reconcile_football_data(parsed, csv_path):
    """Exact date and explicit team-ID mapping; scores must agree independently."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    index = {}
    for row in raw:
        if not row.get("Date"):
            continue
        key = (datetime.strptime(row["Date"], "%d/%m/%Y").date().isoformat(),
               row["HomeTeam"], row["AwayTeam"])
        if key in index:
            raise ValueError("Duplicate Football-Data date/home/away key")
        index[key] = row
    records, used = [], set()
    for match in parsed:
        key = (match["match_date"], TEAM_NAMES[match["home_team_id"]], TEAM_NAMES[match["away_team_id"]])
        row = index.get(key)
        if row is None:
            raise ValueError(f"No exact independent match join: {key}")
        score = [int(row["FTHG"]), int(row["FTAG"])]
        if score != match["diagnostics"]["event_goals"]:
            raise ValueError(f"Independent full-time score disagreement: {key}")
        used.add(key)
        records.append({"match_id": match["match_id"], "date": key[0], "home": key[1],
                        "away": key[2], "goals": score, "score_agrees": True})
    return {"football_data_rows": len(index), "exact_joins": len(records),
            "unmatched_football_data_rows": [list(k) for k in sorted(set(index) - used)],
            "identity_method": "explicit source team IDs to exact Football-Data labels; date/home/away",
            "matches": records, "odds_used_in_this_experiment": False}


def score_matches(model, matches):
    predictions, skipped = [], []
    for match in matches:
        try:
            predictions.append(model.predict_match(match))
        except UnknownStintIdentityError as error:
            skipped.append({"match_id": match["match_id"], "date": match["match_date"],
                            "reason": str(error), "action": "whole match excluded, no zero effect imputed"})
    return {"predictions": predictions, "skipped": skipped, "requested_matches": len(matches)}


def summarize(predictions):
    n = len(predictions)
    if not n:
        raise ValueError("No shared supported matches to score")
    return {"matches": n, "stints": sum(p["n_stints"] for p in predictions),
            "mean_match_loss": sum(p["point_process_loss"] for p in predictions) / n,
            "expected_total_target": sum(p["expected_home_target"] + p["expected_away_target"] for p in predictions),
            "observed_total_target": sum(p["actual_home_target"] + p["actual_away_target"] for p in predictions),
            "loss_kind": predictions[0]["loss_kind"]}


def paired_comparison(team_scores, player_scores):
    team = {p["match_id"]: p for p in team_scores["predictions"]}
    player = {p["match_id"]: p for p in player_scores["predictions"]}
    common = sorted(team.keys() & player.keys())
    if not common:
        raise ValueError("No supported matched comparison cohort")
    rows = []
    for mid in common:
        a, b = team[mid], player[mid]
        if (a["match_date"], a["actual_home_target"], a["actual_away_target"], a["n_stints"]) != (
                b["match_date"], b["actual_home_target"], b["actual_away_target"], b["n_stints"]):
            raise ValueError("Paired models must score the same response and actual state path")
        if a["loss_kind"] != b["loss_kind"]:
            raise ValueError("Cannot pair different target/loss tracks")
        rows.append({"match_id": mid, "date": a["match_date"], "team_loss": a["point_process_loss"],
                     "player_loss": b["point_process_loss"],
                     "player_minus_team_loss": b["point_process_loss"] - a["point_process_loss"]})
    return {"common_match_ids": common, "team_on_common": summarize([team[m] for m in common]),
            "player_on_common": summarize([player[m] for m in common]),
            "mean_player_minus_team_loss": sum(r["player_minus_team_loss"] for r in rows) / len(rows),
            "lower_is_better": True, "paired_matches": rows,
            "sampling_uncertainty": "not estimated in this first run; no significance or confidence claim"}


def choose_penalty(validation_records):
    """Validation only; don't compare candidates on different supported samples."""
    if not validation_records:
        raise ValueError("Validation candidates required")
    cohorts = [tuple(r["comparison"]["common_match_ids"]) for r in validation_records]
    if not cohorts[0] or any(c != cohorts[0] for c in cohorts):
        raise ValueError("Player penalty selection requires identical nonempty validation cohorts")
    return min(validation_records, key=lambda r: (r["comparison"]["player_on_common"]["mean_match_loss"],
                                                  -r["player_ridge"]))["player_ridge"]


def export_player_coefficients(model, history, destination):
    seconds, matches = defaultdict(float), defaultdict(set)
    names = {}
    for match in history:
        for row in match["player_minutes"]:
            names[row["player_id"]] = row["player_name"]
            if row["on_pitch_seconds"] > 0:
                seconds[row["player_id"]] += row["on_pitch_seconds"]
                matches[row["player_id"]].add(match["match_id"])
    fields = ["player_id", "player_name", "training_matches_on_pitch", "training_on_pitch_minutes",
              "conditional_attack_log_coefficient", "conditional_defence_log_coefficient",
              "attack_rate_ratio_one_indicator_unit", "opponent_rate_ratio_one_defence_unit",
              "interpretation"]
    with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pid in sorted(model.players):
            attack = float(model.coefficients[model.feature_index[f"player_attack:{pid}"]])
            defence = float(model.coefficients[model.feature_index[f"player_defence:{pid}"]])
            writer.writerow({"player_id": pid, "player_name": names.get(pid),
                "training_matches_on_pitch": len(matches[pid]), "training_on_pitch_minutes": seconds[pid] / 60,
                "conditional_attack_log_coefficient": attack, "conditional_defence_log_coefficient": defence,
                "attack_rate_ratio_one_indicator_unit": exp(attack),
                "opponent_rate_ratio_one_defence_unit": exp(-defence),
                "interpretation": "regularized conditional association, not causal value, ranking or pre-match probability"})


def _fit(matches, cutoff, response, players, penalty=10.0):
    print(f"Fitting {response}; players={players}; ridge={penalty}; cutoff={cutoff}", flush=True)
    return JointStintIntensityModel(include_players=players, response_kind=response,
                                   player_ridge=penalty).fit(matches, before_date=cutoff)


def run(root):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "outputs" / f"完整赛季球员强度训练_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    configuration = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_before": TRAIN_BEFORE.isoformat(), "validation_before": TEST_FROM.isoformat(),
        "test_from": TEST_FROM.isoformat(), "player_ridge_candidates": list(PLAYER_RIDGES),
        "team_ridge": 1.0, "context_ridge": 1.0, "response_tracks": ["goals", "xg"],
        "selection": "validation mean match point-process loss; identical known-player cohort; larger ridge breaks ties",
        "test_refit": "all matches strictly before test_from; no test-result parameter selection",
        "usage": "retrospective actual-state conditional intensity, not pre-match 1X2",
        "historical_lineup_announcement_at": None, "historical_source_first_available_at": None}
    write_json(output / "实验设置.json", configuration)
    try:
        matches, quality = parse_season(root / "data/statsbomb/premier-league-2015-16")
        write_json(output / "完整赛季解析质量.json", quality)
        write_json(output / "真实在场时段.json", matches)
        if quality["parse_failures"]:
            raise ValueError("Resolve recorded parse errors before this complete-season experiment")
        fd_path = root / "data/football-data/2015-16/20261022_full_player/E0.csv"
        independent = reconcile_football_data(matches, fd_path)
        write_json(output / "独立赛果逐场核对.json", independent)
        write_json(output / "Football-Data来源用途.json", {
            "source_url": "https://www.football-data.co.uk/mmz4281/1516/E0.csv", "local_path": str(fd_path),
            "observed_local_at_utc": datetime.now(timezone.utc).isoformat(), "retrieved_at_utc": None,
            "historical_first_available_at": None, "purpose": "personal noncommercial football statistics/entertainment",
            "raw_data_redistribution": False, "this_run_downloaded": False, "odds_used": False})
        train, validation, test = temporal_split(matches)
        anomalies = [{"match_id": m["match_id"], "date": m["match_date"], "record": row}
                     for m in matches for row in m["diagnostics"]["all_source_event_clock_anomalies"]]
        write_json(output / "源时钟异常明细.json", anomalies)
        coverage = {"season": quality["schedule_profile"],
            "segments": {name: {"matches": len(rows), "date_min": rows[0]["match_date"],
                                "date_max": rows[-1]["match_date"]}
                         for name, rows in (("train", train), ("validation", validation), ("test", test))},
            "stints": sum(len(m["stints"]) for m in matches),
            "events": sum(m["diagnostics"]["n_events"] for m in matches),
            "goals": sum(sum(m["diagnostics"]["event_goals"]) for m in matches),
            "xg": sum(m["diagnostics"]["home_xg"] + m["diagnostics"]["away_xg"] for m in matches),
            "source_clock_anomalies": len(anomalies),
            "clock_anomaly_kinds": dict(Counter(row["record"]["issue"] for row in anomalies)),
            "clock_policy": "no source reordering/repair; only non-model event anomalies tolerated and recorded",
            "independent_full_time_score_joins": independent["exact_joins"]}
        write_json(output / "覆盖与时间切分.json", coverage)
        print(f"Parsed {len(matches)} matches; train/validation/test={len(train)}/{len(validation)}/{len(test)}", flush=True)
        selection = {}
        # Complete both validation tracks and save choices before any test scoring.
        for response in ("goals", "xg"):
            team = _fit(train, TRAIN_BEFORE, response, False)
            team.save(output / f"validation_{response}_team.json")
            team_scores = score_matches(team, validation)
            write_json(output / f"validation_{response}_team_predictions.json", team_scores)
            candidates = []
            for penalty in PLAYER_RIDGES:
                player = _fit(train, TRAIN_BEFORE, response, True, penalty)
                label = f"validation_{response}_players_ridge{penalty:g}"
                player.save(output / f"{label}.json")
                scores = score_matches(player, validation)
                write_json(output / f"{label}_predictions.json", scores)
                comparison = paired_comparison(team_scores, scores)
                candidates.append({"player_ridge": penalty, "comparison": comparison,
                                   "skipped_matches": scores["skipped"]})
            selection[response] = {"selected_player_ridge": choose_penalty(candidates),
                                   "candidates": candidates, "team_full_coverage": summarize(team_scores["predictions"])}
        write_json(output / "验证期选择结果.json", selection)
        final = {}
        for response in ("goals", "xg"):
            penalty = selection[response]["selected_player_ridge"]
            team = _fit(train + validation, TEST_FROM, response, False)
            player = _fit(train + validation, TEST_FROM, response, True, penalty)
            team_path, player_path = output / f"test_{response}_team.json", output / f"test_{response}_players.json"
            team.save(team_path)
            player.save(player_path)
            a, b = score_matches(team, test), score_matches(player, test)
            write_json(output / f"test_{response}_team_predictions.json", a)
            write_json(output / f"test_{response}_players_predictions.json", b)
            compare = paired_comparison(a, b)
            # Verify persistence on an actually supported later match, not training observations.
            witness = next(m for m in test if m["match_id"] == compare["common_match_ids"][0])
            roundtrips = {"team": JointStintIntensityModel.load(team_path).predict_match(witness) == team.predict_match(witness),
                          "players": JointStintIntensityModel.load(player_path).predict_match(witness) == player.predict_match(witness)}
            if not all(roundtrips.values()):
                raise ValueError("Saved model does not reproduce actual later-match prediction")
            export_player_coefficients(player, train + validation, output / f"{response}_球员条件系数.csv")
            final[response] = {"selected_player_ridge": penalty, "test_requested": len(test),
                "comparison": compare, "team_full_coverage": summarize(a["predictions"]),
                "skipped_players": b["skipped"], "skipped_team": a["skipped"],
                "parameter_json_roundtrip": roundtrips, "team_optimizer": team.optimizer_diagnostics,
                "player_optimizer": player.optimizer_diagnostics,
                "fitted_players": len(player.players), "fitted_matches": player.n_matches}
            print(f"Later {response}: supported={len(compare['common_match_ids'])}/{len(test)}; "
                  f"player-minus-team loss={compare['mean_player_minus_team_loss']:.6f}", flush=True)
        write_json(output / "较晚检验结果.json", final)
        report = make_report(coverage, selection, final)
        (output / "完整赛季球员强度训练说明.md").write_text(report, encoding="utf-8")
        print(f"Experiment complete: {output}", flush=True)
        return output
    except Exception as error:
        write_json(output / "运行未完成说明.json", {"error": f"{type(error).__name__}: {error}",
                   "status": "incomplete; partial artifacts are not a successful experiment"})
        raise


def make_report(coverage, selection, final):
    lines = ["# 完整赛季球员历史强度：首轮真实训练", "",
        "这是历史实际在场／比分／人数条件下的攻防强度研究，不是赛前胜率或投注推荐。", "",
        f"数据：2015/16英超{coverage['season']['matches']}场、{coverage['stints']}个时段、{coverage['events']}条事件；"
        f"{coverage['goals']}个进球、xG合计{coverage['xg']:.6f}。Football-Data独立核对{coverage['independent_full_time_score_joins']}场。", "",
        "日期边界：2015-12-01之前训练，[2015-12-01,2016-02-01)验证，2016-02-01起检验。"
        + "；".join(f"{name}实际{row['date_min']}至{row['date_max']}、{row['matches']}场"
                    for name, row in coverage['segments'].items()) + "。球队／状态正则固定1，"
        "球员正则候选10、100、1000；验证期选定后，合并较早230场重拟合，再检验较晚150场。", "",
        f"验证期同场比较仅{len(selection['goals']['candidates'][0]['comparison']['common_match_ids'])}/"
        f"{coverage['segments']['validation']['matches']}场，后期覆盖见表；均为已知球员的非随机子集，不代表整季表现。", "",
        "## 较晚检验（同场比较）", "",
        "| 目标轨道 | 球员正则 | 覆盖 | 仅球队损失 | 加球员损失 | 球员减球队 |",
        "|---|---:|---:|---:|---:|---:|"]
    for response, label in (("goals", "真实进球"), ("xg", "xG单独对照")):
        result, compare = final[response], final[response]["comparison"]
        lines.append(f"| {label} | {result['selected_player_ridge']:g} | {len(compare['common_match_ids'])}/{result['test_requested']} | "
                     f"{compare['team_on_common']['mean_match_loss']:.6f} | {compare['player_on_common']['mean_match_loss']:.6f} | "
                     f"{compare['mean_player_minus_team_loss']:+.6f} |")
    lines.extend(["", "损失越低越好；进球是点过程负对数似然的模型相关部分，xG是独立准损失，二者数值不横向比大小。"
        "没有估计抽样区间，不宣称显著提高。无球员模型也使用同样的真实比分／人数状态。", "",
        "## 方法与解释", "",
        "每段预期目标 = 实际时长/5400 × exp(截距＋主场＋本队攻强−对方防强＋球员在场偏差＋段初比分／人数状态)。"
        "球队、球员和状态联合估计；球员参考阵容只来自本次训练窗口的在场暴露。没有把原有球队边际进球率改名后直接套用。", "",
        "目标函数为 Σ[时长/5400×强度 − 目标×log(强度)] 加分组L2惩罚。事件驱动时段不额外加入目标×log(时长)，"
        "也不把进球后的比分当该进球之前的条件。真实进球与xG分开拟合、分开选正则，不混分数。", "",
        "新球员／未知球队尚无显式先验，因此整场跳过并列明。公平比较限于双方都可预测的相同比赛；"
        "仅球队全覆盖成绩另存，不能拿不同样本的平均值宣称球员改进。伤停原因、赛事重视程度未从缺席凭空推断。", "",
        f"源时钟异常记录{coverage['source_clock_anomalies']}条：原始数据不改写、不重排；无关接球等非建模事件可记录后继续，"
        "但射门、进球、换人、红牌、离返场、半场边界仍要求严格关键顺序。逐场进球核对原始比分及独立来源；xG仅核对原始射门合计，没有独立xG来源。", "",
        "## 不确定性与下一步", "",
        "这只是一个旧赛季；共线性、出场选择、教练战术和队友组合使系数是收缩后的条件关联，不是球员个人因果价值。"
        "不能单看球员表排序或把单个系数当替换球员的确定影响。", "",
        "实际阵容、换人时刻、后续比分路径只在事后知道。预测时需要预计阵容／换人情景与权重；"
        "历史首发公布时点、来源首次可用时点和开球时区当前未知，因此没有T−24h或确认首发回测。", "",
        "本次较晚成绩已被查看。后续据此改动属于探索性研发，需要另一个较晚赛季验证；不能继续把同一150场称为未见过的最终检验。"
        "下一步先讨论未知球员先验与滚动更新，再扩充另一完整赛季、做稳定性与成组不确定性分析，最后才接赛前概率。", "",
        "## 来源、用途和复现", "",
        "[StatsBomb / Hudl Open Data](https://github.com/hudl/open-data)：原始事件、阵容、赛程与其数据协议保留在项目目录。"
        "[Football-Data官方CSV](https://www.football-data.co.uk/mmz4281/1516/E0.csv)：只作独立比分核验，本轮不使用赔率。"
        "仅个人非商业统计研究／娱乐，不重新分发原始或衍生数据。数据许可不等于软件MIT许可，没有代用户注册或提交联系资料。", "",
        "复现命令：在model_v2目录运行 `run.ps1 -Action player-train`。读取已下载本地数据，不联网。"
        "所有参数、明细、临时文件和报告在项目目录；源码压缩包不含原始或衍生数据。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    run(parser.parse_args().root)
