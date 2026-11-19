"""Extract saved IMPECT V3 starting XIs and independent full-time targets.

Only the catalogue, selection, per-match receipt records, lineups and D1 CSV
are read. Substitutions, events, KPIs and players.currentSquadId are never
features or selection criteria. Historical announcement availability is unknown:
these are revised retrospective labels for a starting-XI-known comparison.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path

from .impect_pilot import independent_score
from .impect_season import TEAM_NAMES


CATALOGUE_MATCH_COUNT = 306
ITERATION_ID = 743
TARGET_FROM, TARGET_THROUGH = date(2024, 1, 1), date(2024, 5, 18)


class StartingXiSourceError(ValueError):
    """A saved starting-XI observation cannot be identified reliably."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise StartingXiSourceError(f"{name} must be an integer >= {minimum}")
    return value


def _utc(value, name):
    if not isinstance(value, str):
        raise StartingXiSourceError(f"{name} requires a timezone-aware timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise StartingXiSourceError(f"Invalid {name}") from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise StartingXiSourceError(f"{name} requires a timezone-aware timestamp")
    return stamp.astimezone(timezone.utc)


def _identity(metadata):
    if not isinstance(metadata, dict):
        raise StartingXiSourceError("Catalogue match must be an object")
    mid = _integer(metadata.get("id"), "match id")
    iteration = _integer(metadata.get("iterationId"), "iterationId")
    if iteration != ITERATION_ID:
        raise StartingXiSourceError("Only inspected iteration 743 / V3 is supported")
    home = _integer(metadata.get("homeSquadId"), "homeSquadId")
    away = _integer(metadata.get("awaySquadId"), "awaySquadId")
    if home == away:
        raise StartingXiSourceError("Distinct home and away squads required")
    return mid, iteration, home, away, _utc(metadata.get("scheduledDate"), "scheduledDate")


def extract_match(metadata, lineups, full_time):
    """Return only XI identities, UTC date and independent full-time response.

    Position labels validate starters but do not become features. Bench roster
    identities validate source membership only; current squad metadata and the
    source's later match path are neither inspected nor returned.
    """
    mid, iteration, home, away, kickoff = _identity(metadata)
    if not isinstance(lineups, dict):
        raise StartingXiSourceError("Lineups must be an object")
    if (_integer(lineups.get("id"), "lineup match id") != mid
            or _integer(lineups.get("iterationId"), "lineup iterationId") != iteration
            or _utc(lineups.get("dateTime"), "lineup dateTime") != kickoff):
        raise StartingXiSourceError("Lineup match/iteration/dateTime identity disagrees")
    if not isinstance(full_time, (list, tuple)) or len(full_time) != 2:
        raise StartingXiSourceError("Independent full_time must contain two goal counts")
    full_time = [_integer(value, "full_time goal", minimum=0) for value in full_time]
    all_roster, starters = set(), {}
    for key, team, side in (("squadHome", home, "home"), ("squadAway", away, "away")):
        squad = lineups.get(key)
        if not isinstance(squad, dict) or _integer(squad.get("id"), "lineup squad id") != team:
            raise StartingXiSourceError("Lineup squad identity disagrees")
        roster, positions = squad.get("players"), squad.get("startingPositions")
        if not isinstance(roster, list) or not isinstance(positions, list) or len(positions) != 11:
            raise StartingXiSourceError("Roster list and exactly 11 startingPositions required")
        current_roster = set()
        for row in roster:
            if not isinstance(row, dict):
                raise StartingXiSourceError("Roster player must be an object")
            player = _integer(row.get("id"), "roster player id")
            if player in current_roster or player in all_roster:
                raise StartingXiSourceError("Duplicate roster player or player on both squads")
            current_roster.add(player)
        all_roster.update(current_roster)
        current_starters = set()
        for row in positions:
            if not isinstance(row, dict):
                raise StartingXiSourceError("Starting position must be an object")
            player = _integer(row.get("playerId"), "starting player id")
            position = row.get("position")
            if (not isinstance(position, str) or not position.strip()
                    or position.strip().upper() == "BANK"):
                raise StartingXiSourceError("Starter requires a non-BANK source position")
            if player not in current_roster or player in current_starters:
                raise StartingXiSourceError("Unknown roster player or duplicate starter")
            current_starters.add(player)
        starters[side] = [f"impect:player:{player}" for player in sorted(current_starters)]
    if set(starters["home"]) & set(starters["away"]):
        raise StartingXiSourceError("Starting player cannot be on both teams")
    return {"match_id": f"impect:match:{mid}", "match_date": kickoff.date().isoformat(),
            "home_team_id": home, "away_team_id": away,
            "home_starters": starters["home"], "away_starters": starters["away"],
            "full_time": full_time}


def selected_catalogue(catalogue, selection):
    """Use the full explicit saved selection, never the old parsed subset."""
    if not isinstance(catalogue, list) or len(catalogue) != CATALOGUE_MATCH_COUNT:
        raise ValueError(f"Expected the existing {CATALOGUE_MATCH_COUNT}-match catalogue")
    if not isinstance(selection, dict) or selection.get("documented_data_version") != "V3":
        raise ValueError("Explicit saved V3 season selection required")
    selected = selection.get("selected_matches")
    if not isinstance(selected, list) or len(selected) != len(catalogue):
        raise ValueError("Season selection must contain the full catalogue exactly once")
    if any(not isinstance(row, dict) for row in catalogue + selected):
        raise ValueError("Catalogue and selection rows must be objects")
    source_by_id = {row.get("id"): row for row in catalogue}
    selected_by_id = {row.get("id"): row for row in selected}
    if len(source_by_id) != len(catalogue) or len(selected_by_id) != len(selected) or source_by_id != selected_by_id:
        raise ValueError("Season selection must equal the complete distinct source catalogue")
    for row in catalogue:
        _identity(row)
    return sorted(catalogue, key=lambda row: (_utc(row["scheduledDate"], "scheduledDate"), row["id"]))


def inspect_match(metadata, record, score_path):
    mid, _, home, away, kickoff = _identity(metadata)
    item = {"source_match_id": mid, "scheduled_kickoff_utc": kickoff.isoformat(),
            "source_scheduled_date_text": metadata["scheduledDate"],
            "home_team_id": home, "away_team_id": away,
            "target_period": TARGET_FROM <= kickoff.date() <= TARGET_THROUGH,
            "status": "lineups_missing", "included": False,
            "catalogue_last_calculation_date": metadata.get("lastCalculationDate"),
            "historical_lineup_announcement_at": None, "historical_first_available_at": None}
    if record is None:
        return item, None
    if not isinstance(record, dict) or record.get("match_id") != mid or not isinstance(record.get("files"), list):
        raise ValueError("Per-match source record has conflicting identity or files")
    lineup_files = [row for row in record["files"] if isinstance(row, dict) and row.get("kind") == "lineups"]
    if not lineup_files:
        return item, None
    if len(lineup_files) != 1:
        raise ValueError("Per-match source record must identify one unique lineups file")
    receipt = lineup_files[0]
    item["lineups_receipt"] = receipt
    if receipt.get("http_status") != 200 or receipt.get("error"):
        item.update(status="lineups_download_failed")
        return item, None
    path = Path(receipt["file"])
    if not path.is_file():
        item.update(error="Saved lineups response file is missing")
        return item, None
    try:
        lineups = read(path)
    except (OSError, json.JSONDecodeError) as error:
        item.update(status="lineups_read_failure", error=str(error))
        return item, None
    if isinstance(lineups, dict):
        item["lineups_last_calculation_date"] = lineups.get("lastCalculationDate")
    try:
        score = independent_score(metadata, score_path, team_names=TEAM_NAMES)
    except (KeyError, ValueError) as error:
        item.update(status="independent_score_join_failure", error=str(error))
        return item, None
    item["independent_score"] = score
    try:
        match = extract_match(metadata, lineups, score["full_time"])
    except StartingXiSourceError as error:
        item.update(status="lineups_identity_or_starters_failure", error=str(error))
        return item, None
    item.update(status="included", included=True)
    return item, match


def run(root, source):
    root, source = Path(root).expanduser().resolve(), Path(source).expanduser().resolve()
    catalogue_path = root / "data/impect/feasibility_20261023/matches_743.json"
    selection_path = source / "selection.json"
    catalogue = selected_catalogue(read(catalogue_path), read(selection_path))
    score_path = root / "data/football-data/2023-24/20261022T113804786593Z/D1.csv"
    matches, measurements = [], []
    for metadata in catalogue:
        record_path = source / "matches" / f"{metadata['id']}.json"
        item, match = inspect_match(metadata, read(record_path) if record_path.is_file() else None, score_path)
        item["source_match_record_file"] = str(record_path)
        measurements.append(item)
        if match is not None:
            matches.append(match)
    started = datetime.now(timezone.utc)
    output = root / "outputs" / started.strftime("IMPECT首发全场接通_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    logo = root / "data/impect/pilot_20261024T053746962994Z/impect_logo_black.svg"
    result = {
        "created_at_utc": started.isoformat(), "source_directory": str(source),
        "catalogue_file": str(catalogue_path), "selection_file": str(selection_path),
        "source_score_file": str(score_path), "explicit_team_mapping": TEAM_NAMES,
        "expected_catalogue_matches": len(catalogue), "included_matches": len(matches),
        "expected_target_matches": sum(item["target_period"] for item in measurements),
        "included_target_matches": sum(item["target_period"] and item["included"] for item in measurements),
        "statuses": dict(Counter(item["status"] for item in measurements)), "matches": measurements,
        "matches_schema": ["match_id", "match_date", "home_team_id", "away_team_id", "home_starters", "away_starters", "full_time"],
        "source_logo_file": str(logo) if logo.is_file() else None,
        "historical_lineup_announcement_at": None, "historical_first_available_at": None,
        "eligibility": "Full source catalogue; unique lineups response; exact match/iteration/scheduled instant/squad and 11-per-side roster membership; independent exact D1 full/half-time score join",
        "not_used": ["events", "substitutions", "player_kpis", "players.currentSquadId", "old 243-match parsed subset", "red-card outcomes", "observed final duration"],
        "interpretation": "Revised retrospective starting-XI labels and independent full-time goals, not verified historical pre-match announcement snapshots",
        "source_terms": "IMPECT custom terms; local personal noncommercial research; do not redistribute observed or derived data",
        "fits": 0, "parameter_searches": 0, "forecasts": 0,
    }
    _write(output / "matches.json", matches)
    _write(output / "inspection.json", result)
    problems = [item for item in measurements if not item["included"]]
    lines = ["# IMPECT首发已知／全场比分接通", "",
             "数据提供方：[IMPECT Open Data](https://github.com/ImpectAPI/open-data)。仅本地个人非商业研究，遵守自定义条款，不再分发观测或派生数据。", ""]
    if logo.is_file():
        lines += [f"![IMPECT](<{logo.as_posix()}>)", ""]
    lines += [f"完整原始目录{len(catalogue)}场，纳入{len(matches)}场；目标原始{result['expected_target_matches']}场，纳入{result['included_target_matches']}场。",
              f"逐项状态：{result['statuses']}。缺失与身份／首发异常均保留在inspection.json，不按后续模型损失或红牌筛选。", "",
              "每行仅比赛ID、UTC日历日期、主客球队ID、双方11名首发ID和独立全场比分。路径、源修订时间、HTTP接收记录、半场比分与拼接证据只保存在inspection.json，不是模型特征。", "",
              "首发来自事后修订历史文件，历史首次发布时间／首发公布时刻未知。因此它支持“首发已知条件下”的研究性全场比较，不证明真实T-24h或历史首发公布时信息集。", "",
              "不读取换人、事件、球员KPI、当前球队属性，不依赖原243场可解析时段，不根据最终红牌或比赛时长选择比赛。全场比分自然包含换人与红牌后的结果，但这些后续信息不作为输入；未来若建模其分布须另作结构讨论。", "",
              "本次0次拟合、0次调参、0个预测；不覆盖现有球队／LLM线路。", ""]
    if problems:
        lines += ["| 比赛ID | 状态 | 原因 |", "|---|---|---|"]
        lines += [f"| {item['source_match_id']} | {item['status']} | {item.get('error', '')} |" for item in problems]
    with (output / "首发全场接通说明.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"output_directory": str(output), "matches_file": str(output / "matches.json"),
            "included_matches": len(matches), "expected_matches": len(catalogue),
            "included_targets": result["included_target_matches"], "expected_targets": result["expected_target_matches"],
            "statuses": result["statuses"], "fits": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.source), ensure_ascii=False))


if __name__ == "__main__":
    main()
