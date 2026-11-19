"""Measure official match metadata before downloading large event archives.

A catalog entry is not proof of a complete league. No model fitting, no data
repair and no replacement of the existing season occur in this inspection.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .statsbomb_data import _fetch, season_profile, write_json


TOP_FIVE_IDS = {2, 7, 9, 11, 12}


def compact_profile(metadata):
    profile = season_profile(metadata)
    return {"matches": profile["matches"], "teams": profile["teams"],
            "date_min": profile["date_min"], "date_max": profile["date_max"],
            "complete_double_round_robin": profile["complete_double_round_robin"],
            "duplicate_ids": len(profile["duplicate_match_ids"]),
            "duplicate_home_away_pairs": len(profile["duplicate_home_away_pairs"]),
            "missing_home_away_pairs": len(profile["missing_home_away_pairs"]),
            "team_match_count_min": min(profile["team_match_counts"].values()),
            "team_match_count_max": max(profile["team_match_counts"].values()),
            "stages": dict(Counter(m.get("competition_stage", {}).get("name", "unknown") for m in metadata))}


def inspect_catalog(root):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = root / "data" / "statsbomb" / f"catalog-inspection_{stamp}"
    output = root / "outputs" / f"第二赛季数据覆盖核查_{stamp}"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    files = [_fetch(("data/competitions.json", "competitions.json"), source)]
    catalog = json.loads((source / "competitions.json").read_text(encoding="utf-8"))
    selected = [row for row in catalog if row["competition_id"] in TOP_FIVE_IDS
                and row["competition_gender"] == "male"]
    results = []
    for row in sorted(selected, key=lambda m: (m["competition_id"], m["season_id"])):
        comp, season = row["competition_id"], row["season_id"]
        entry = {k: row[k] for k in ("competition_id", "season_id", "competition_name", "season_name",
                                    "competition_gender")}
        try:
            filename = f"matches_{comp}_{season}.json"
            record = _fetch((f"data/matches/{comp}/{season}.json", filename), source)
            files.append(record)
            metadata = json.loads((source / filename).read_text(encoding="utf-8"))
            entry["profile"] = compact_profile(metadata)
            entry["source_url"] = record["source_url"]
            entry["error"] = None
            print(f"{entry['competition_name']} {entry['season_name']}: "
                  f"{len(metadata)} matches; complete={entry['profile']['complete_double_round_robin']}", flush=True)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        results.append(entry)
    write_json(output / "官方赛季覆盖.json", {"inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official catalog, all listed men's top-five league seasons; match metadata only",
        "results": results, "source_files": files, "event_lineup_files_downloaded": 0,
        "historical_first_available_at": None,
        "limits": "full match schedule does not establish complete valid events/lineups; data availability clocks not historical prediction clocks"})
    lines = ["# 第二赛季数据覆盖核查", "",
        "只核查官方目录与逐场赛程；未下载新事件大包、未训练新模型。所有目录和来源记录保留在项目目录。", "",
        "| 赛事 | 赛季 | 场数 | 球队数 | 每队最少／最多场 | 完整主客双循环 |",
        "|---|---|---:|---:|---:|---|"]
    for row in results:
        if row["error"]:
            lines.append(f"| {row['competition_name']} | {row['season_name']} | 读取失败 | — | — | 未确认 |")
        else:
            p = row["profile"]
            lines.append(f"| {row['competition_name']} | {row['season_name']} | {p['matches']} | {p['teams']} | "
                         f"{p['team_match_count_min']}／{p['team_match_count_max']} | {'是' if p['complete_double_round_robin'] else '否'} |")
    lines.extend(["", "判定依据是实际球队ID、主客组合和重复数，不根据目录名称猜测。"
        "这里只能确认赛程形状，事件／阵容质量仍需逐场验证。目录未列出某赛季不等于任何地方都没有该数据。", "",
        "[官方目录](https://raw.githubusercontent.com/hudl/open-data/master/data/competitions.json)；"
        "逐个赛季原始URL与下载时间见同目录JSON。没有生成历史首发公布时间。", ""])
    (output / "第二赛季数据覆盖说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Catalog inspection complete: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    inspect_catalog(parser.parse_args().root)
