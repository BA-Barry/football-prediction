"""Replay a saved small IMPECT sample; no downloads, fits or pre-match forecasts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .impect_stints import ImpectParseError, parse_match


# Explicit, narrow source crosswalk for this pilot, not a fuzzy team resolver.
FOOTBALL_DATA_NAMES = {38: "Werder Bremen", 33: "Bayern Munich", 41: "Leverkusen",
                       37: "RB Leipzig", 29: "Dortmund", 36: "Darmstadt"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def independent_score(metadata, csv_path, *, team_names=None):
    """Match an independent saved score by exact date and explicit home/away IDs."""
    scheduled = datetime.fromisoformat(metadata["scheduledDate"].replace("Z", "+00:00"))
    if scheduled.tzinfo is None:
        raise ValueError("Catalogue scheduledDate must have an explicit timezone")
    day = scheduled.astimezone(timezone.utc).date()
    try:
        mapping = FOOTBALL_DATA_NAMES if team_names is None else team_names
        home, away = (mapping[metadata[k]] for k in ("homeSquadId", "awaySquadId"))
    except KeyError as exc:
        raise ValueError("No explicit pilot team-name mapping; do not infer one") from exc
    matches = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            if (datetime.strptime(row["Date"], "%d/%m/%Y").date() == day
                    and row["HomeTeam"] == home and row["AwayTeam"] == away):
                if row["Div"] != "D1":
                    raise ValueError("Expected German Bundesliga D1 score source")
                matches.append({"csv_file": str(csv_path), "csv_line": line,
                    "source_date": row["Date"], "home": home, "away": away,
                    "full_time": [int(row["FTHG"]), int(row["FTAG"])],
                    "half_time": [int(row["HTHG"]), int(row["HTAG"])],
                    "join": "exact UTC calendar date + explicit provider-ID/name crosswalk + home/away"})
    if len(matches) != 1:
        raise ValueError(f"Need exactly one independent score row, found {len(matches)}")
    return matches[0]


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run(root: Path, sample: Path):
    """Inspect only the preselected saved matches, preserving every raw file."""
    started = datetime.now(timezone.utc)
    timer = perf_counter()
    root, sample = Path(root), Path(sample)
    selection = _read(sample / "selection.json")
    if selection.get("documented_data_version") != "V3":
        raise ValueError("This pilot only supports documented IMPECT V3")
    selected = selection["selected_matches"]
    ids = [r["id"] for r in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A nonempty, distinct saved sample selection is required")
    catalogue_path = root / "data/impect/feasibility_20261023/matches_743.json"
    catalogue = {r["id"]: r for r in _read(catalogue_path)}
    score_path = root / "data/football-data/2023-24/20261022T113804786593Z/D1.csv"
    squads = {r["id"]: r["name"] for r in _read(sample / "squads_743.json")}
    parsed, measurements = [], []
    for selected_row in selected:
        match_id = selected_row["id"]
        metadata = catalogue[match_id]
        if any(metadata.get(k) != v for k, v in selected_row.items()):
            raise ValueError("Saved selection and catalogue match identity disagree")
        events = _read(sample / f"events_{match_id}.json")
        lineups = _read(sample / f"lineups_{match_id}.json")
        kpis = _read(sample / f"player_kpis_{match_id}.json")
        score = independent_score(metadata, score_path)
        item = {"source_match_id": match_id, "scheduled_date": metadata["scheduledDate"],
                "home": squads[metadata["homeSquadId"]], "away": squads[metadata["awaySquadId"]],
                "independent_score": score, "events": len(events)}
        try:
            match = parse_match(metadata, events, lineups, kpis, expected_score=score["full_time"])
        except ImpectParseError as exc:
            item.update(status="unreconciled", error=str(exc))
            measurements.append(item)
            continue
        parsed.append(match)
        halftime = [sum(s[f"{side}_goals"] for s in match["stints"] if s["period"] == 1)
                    for side in ("home", "away")]
        if halftime != score["half_time"]:
            raise ValueError(f"Independent half-time score disagrees for {match_id}")
        kpi_seconds = {}
        kpi_rows = 0
        for side in ("squadHome", "squadAway"):
            team = kpis[side]["id"]
            for row in kpis[side]["players"]:
                key = (team, f"impect:player:{row['id']}")
                kpi_seconds[key] = kpi_seconds.get(key, 0.0) + row["playDuration"]
                kpi_rows += 1
        errors = [abs(p["on_pitch_seconds"] - kpi_seconds[(p["team_id"], p["player_id"])])
                  for p in match["player_minutes"] if (p["team_id"], p["player_id"]) in kpi_seconds]
        missing_active = [p["player_id"] for p in match["player_minutes"]
                          if p["on_pitch_seconds"] > 0 and (p["team_id"], p["player_id"]) not in kpi_seconds]
        changes = [c for side in ("squadHome", "squadAway") for c in lineups[side]["substitutions"]]
        bank_crossings = sum((r["fromPosition"] == "BANK") != (r["toPosition"] == "BANK") for r in changes)
        period_seconds = {str(period): sum(s["duration_seconds"] for s in match["stints"] if s["period"] == period)
                          for period in (1, 2)}
        red_events = [e for e in events if e["actionType"] == "RED_CARD"]
        item.update(status="reconciled", stints=len(match["stints"]), period_seconds=period_seconds,
            total_seconds=sum(period_seconds.values()), half_time=halftime, full_time=score["full_time"],
            roster_player_matches=len(match["player_minutes"]),
            active_player_matches=sum(p["on_pitch_seconds"] > 0 for p in match["player_minutes"]),
            kpi_rows=kpi_rows, kpi_distinct_player_matches=len(kpi_seconds),
            compared_player_matches=len(errors), missing_active_duration_player_ids=missing_active,
            max_player_duration_absolute_difference_seconds=max(errors) if errors else None,
            source_change_rows=len(changes), bank_crossing_rows=bank_crossings,
            tactical_rows=len(changes) - bank_crossings,
            paired_substitutions=bank_crossings // 2 if not red_events else None,
            red_card_events=len(red_events),
            added_time_goals=sum(e["actionType"] in {"GOAL", "OWN_GOAL"}
                and e["gameTime"]["gameTimeInSec"] - 10000 * (e["periodId"] - 1) > 2700 for e in events))
        measurements.append(item)
    destination = root / "outputs" / started.strftime("球员在场实测_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    summary = {"started_at": started.isoformat(), "sample": str(sample),
        "selection": selection, "data_provider": "Impect", "catalogue": str(catalogue_path),
        "independent_score_source": str(score_path), "explicit_crosswalk": FOOTBALL_DATA_NAMES,
        "matches": measurements, "reconciled": len(parsed), "selected": len(selected),
        "fits": 0, "parameter_searches": 0, "new_forecasts": 0,
        "elapsed_seconds": perf_counter() - timer,
        "limits": ["Three matches measure an adapter, not prediction accuracy or whole-season completeness.",
            "Revised historical files do not reconstruct historical publication or pre-match availability.",
            "Realized on-pitch paths, score states and minutes must not be same-match pre-match inputs.",
            "No modern player effects, causal injury values, current English player mappings or pre-match rate corrections fitted.",
            "No red cards in the selected real sample; synthetic behavior is not real red-card validation.",
            "Goals only; vendor xG/other derived KPIs are not response targets or player strength.",
            "Original raw files retained locally for personal noncommercial research; no redistribution."]}
    _write(destination / "measurement.json", summary)
    _write(destination / "parsed_matches.json", parsed)
    lines = ["# 球员在场影响：较新男足数据实测", "",
        "本轮实现真实事件到既有球员条件强度研究格式的适配，未训练新球员参数、改动当前预测或搜索超参数。",
        f"固定抽取首、次、末场共{len(selected)}场，成功还原{len(parsed)}场；数据来源：[Impect Open Data](https://github.com/ImpectAPI/open-data)。",
        f"![Impect]({(sample / 'impect_logo_black.svg').as_posix()})", "",
        "| 比赛 | 实际秒数（含补时，不含半场间歇） | 原位置变更行／战术行／实际换人 | 全场／半场比分 | 在场秒数最大差 |",
        "|---|---:|---:|---|---:|"]
    for item in measurements:
        if item["status"] != "reconciled":
            lines.append(f"| {item['source_match_id']} | 未还原：{item['error']} | — | — | — |")
            continue
        ft, ht = ("–".join(map(str, item[k])) for k in ("full_time", "half_time"))
        error_text = (f"{item['max_player_duration_absolute_difference_seconds']:.3g}秒"
                      if item['max_player_duration_absolute_difference_seconds'] is not None else "未提供")
        if item["missing_active_duration_player_ids"]:
            error_text += f"（{len(item['missing_active_duration_player_ids'])}名出场球员缺源，未补零）"
        lines.append(f"| {item['home']}—{item['away']} | {item['total_seconds']:.3f} | "
            f"{item['source_change_rows']}/{item['tactical_rows']}/{item['paired_substitutions']} | {ft}／{ht} | "
            f"{error_text} |")
    lines += ["", "## 本轮补到了预测的哪一层", "",
        "原先只有较旧球员研究和当前伤停背景，不能可靠给当前球员扣分。本轮把更近年的真实在场名单、秒级换人、比分变化与真实进球接通到同一段落格式；按球员ID累加多位置playDuration作同源跨文件时长核对（不是独立时钟真值），并以早已保存的Football-Data逐半场比分独立核验。没有把供应商派生KPI当球员实力。",
        "时钟按两半场各自终哨还原：下半场编码减10000，绝不累加上半场终哨事件的duration（其中包含编码间隙），也不强截90分钟漏掉补时进球。纯战术换位不产生进出场。",
        "", "## 尚不能声称的结果", "",
        "这三场不是预测成绩，不能报告准确率提升。原始资料是后来修订的版本，不证明当年赛前已可得；同场真实在场／比分／分钟不能进入同场赛前模型。样本没有红牌，罚下逻辑只有普通合成测试，仍需真实样本研究。",
        "旧9场前瞻、参数、LLM回答及比分分布全部保留。本轮拟合0次、参数搜索0次、新预测0场；没有加入神经网络或给未知球员填零。",
        "", "## 下一步的有界方案（未在本轮执行）", "",
        "先扩大到完整同季并审计真实红牌、换人冲突及覆盖，再固定既有进球条件强度估计器，按较早训练／较晚同场的球队线与球员线比较增量，不增加参数网格。它仍然是历史条件强度验证；把条件强度变成真正赛前概率需要预计分钟情景与条件参考映射，属于重大模型结构，先与用户讨论。",
        "数据、原始下载和派生研究产物仅本地G盘个人非商用研究；使用范围依数据方原条款，不是无条件开源授权。",
        "", "来源：项目内Documentation.pdf第6、10–13、27–29页；LICENSE.pdf；原始download_metadata.json保留真实HTTP接收时点。"]
    report = destination / "球员在场实测说明.md"
    with report.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"run_directory": str(destination), "report": str(report), "reconciled": len(parsed),
            "selected": len(selected), "fits": 0, "new_forecasts": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.sample), ensure_ascii=False))


if __name__ == "__main__":
    main()
