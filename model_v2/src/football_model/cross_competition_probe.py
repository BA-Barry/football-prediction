"""Inspect a small CC0 European-cup corpus and explicit domestic-name bridges.

This is a coverage measurement, not a training adapter. No Fixture, timed
information set, global team identity or verified regulation-90 result is made.
Only the modern country-tagged Football.TXT shape is supported; rejected match
lines and unknown date/score semantics remain visible rather than disappearing.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping

from .domain import aware
from .football_data import LEAGUES, load_directory
from .openfootball_snapshot import Fetch, _download, _read
from .project_paths import project_root


BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master"
COMPETITIONS = {"cl": "UEFA Champions League", "el": "UEFA Europa League",
                "conf": "UEFA Conference League"}
COUNTRY_LEAGUE = {"ENG": "E0", "GER": "D1", "ESP": "SP1", "ITA": "I1",
                  "FRA": "F1", "MCO": "F1"}
MONTHS = dict(zip("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), range(1, 13)))
WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
DATE_LINE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+([0-9]{1,2})(?:\s+([0-9]{4}))?$")
MATCH_LINE = re.compile(
    r"^(?:(?P<clock>[0-9]{2}:[0-9]{2})\s+)?(?P<home>.+?)\s+\((?P<hc>[A-Z]{3})\)"
    r"\s+v\s+(?P<away>.+?)\s+\((?P<ac>[A-Z]{3})\)(?P<tail>.*)$"
)
SCORE_PAIR = re.compile(r"(?<![0-9])([0-9]+)-([0-9]+)(?![0-9])")
PLAIN_SCORE = re.compile(r"^([0-9]+)-([0-9]+)(?:\s+\(([0-9]+)-([0-9]+)\))?$")

# Small manually reviewed spelling table for these two historical seasons.
# Country tags are part of the key. Targets must actually occur in the supplied
# same-season domestic CSVs. This table is NOT automatic fuzzy identity matching.
ALIASES = {
    ("ENG", "Arsenal FC"): "Arsenal", ("ENG", "Aston Villa FC"): "Aston Villa",
    ("ENG", "Brighton & Hove Albion"): "Brighton", ("ENG", "Chelsea FC"): "Chelsea",
    ("ENG", "Liverpool FC"): "Liverpool", ("ENG", "Manchester City FC"): "Man City",
    ("ENG", "Manchester United FC"): "Man United", ("ENG", "Manchester United"): "Man United",
    ("ENG", "Newcastle United FC"): "Newcastle", ("ENG", "Tottenham Hotspur"): "Tottenham",
    ("ENG", "West Ham United"): "West Ham",
    ("GER", "1. FC Union Berlin"): "Union Berlin", ("GER", "Bayer Leverkusen"): "Leverkusen",
    ("GER", "Bayer 04 Leverkusen"): "Leverkusen", ("GER", "Borussia Dortmund"): "Dortmund",
    ("GER", "Eintracht Frankfurt"): "Ein Frankfurt", ("GER", "FC Bayern München"): "Bayern Munich",
    ("GER", "SC Freiburg"): "Freiburg", ("GER", "VfB Stuttgart"): "Stuttgart",
    ("GER", "1899 Hoffenheim"): "Hoffenheim", ("GER", "1. FC Heidenheim 1846"): "Heidenheim",
    ("ESP", "Athletic Club"): "Ath Bilbao", ("ESP", "Club Atlético de Madrid"): "Ath Madrid",
    ("ESP", "FC Barcelona"): "Barcelona", ("ESP", "Girona FC"): "Girona",
    ("ESP", "Real Betis"): "Betis", ("ESP", "Real Madrid CF"): "Real Madrid",
    ("ESP", "Real Sociedad"): "Sociedad", ("ESP", "Real Sociedad de Fútbol"): "Sociedad",
    ("ESP", "Sevilla FC"): "Sevilla", ("ESP", "Villarreal CF"): "Villarreal",
    ("ITA", "AC Milan"): "Milan", ("ITA", "ACF Fiorentina"): "Fiorentina",
    ("ITA", "AS Roma"): "Roma", ("ITA", "Atalanta BC"): "Atalanta",
    ("ITA", "Bologna FC 1909"): "Bologna", ("ITA", "FC Internazionale Milano"): "Inter",
    ("ITA", "Juventus FC"): "Juventus", ("ITA", "Lazio Roma"): "Lazio",
    ("ITA", "SS Lazio"): "Lazio", ("ITA", "SSC Napoli"): "Napoli",
    ("FRA", "Lille OSC"): "Lille", ("FRA", "OGC Nice"): "Nice",
    ("FRA", "Olympique Lyonnais"): "Lyon", ("FRA", "Olympique Marseille"): "Marseille",
    ("FRA", "Paris Saint-Germain FC"): "Paris SG", ("FRA", "RC Lens"): "Lens",
    ("FRA", "Racing Club de Lens"): "Lens", ("FRA", "Stade Brestois 29"): "Brest",
    ("FRA", "Stade Rennais"): "Rennes", ("FRA", "Toulouse FC"): "Toulouse",
    # Monaco is explicitly linked to Ligue 1 despite its MCO country tag.
    ("MCO", "AS Monaco FC"): "Monaco",
}


def _season_start(season: str) -> int:
    if not isinstance(season, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", season):
        raise ValueError("Season must use YYYY-YY")
    year = int(season[:4])
    if year < 1900 or year > 9998 or int(season[-2:]) != (year + 1) % 100:
        raise ValueError("Season must contain consecutive years")
    return year


def _date_context(match: re.Match, season: str) -> dict:
    year_start = _season_start(season)
    weekday, month_text, day_text, year_text = match.groups()
    if month_text not in MONTHS:
        raise ValueError("Unknown English source month")
    month = MONTHS[month_text]
    year = int(year_text) if year_text else year_start + (month < 7)
    day = date(year, month, int(day_text))
    if not date(year_start, 7, 1) <= day < date(year_start + 1, 7, 1):
        raise ValueError("Reported date lies outside the July-to-June season window")
    return {"match_date": day.isoformat(),
            "year_basis": "explicit_date_line" if year_text else "derived_season_month_window",
            "weekday_consistent": WEEKDAYS[day.weekday()] == weekday,
            "source_date_line": match.group(0)}


def _score_details(tail: str) -> dict:
    # Venue text is not treated as score annotation; its presence still does not
    # prove neutral/home venue or the timezone of a printed kickoff clock.
    score_text, separator, venue = tail.strip().partition(" @ ")
    pairs = [[int(a), int(b)] for a, b in SCORE_PAIR.findall(score_text)]
    extra = "a.e.t." in score_text
    penalty = "pen." in score_text
    plain = PLAIN_SCORE.fullmatch(score_text)
    if not score_text:
        kind = "missing_score"
    elif penalty:
        kind = "penalty_shootout_annotation"
    elif extra:
        kind = "extra_time_annotation"
    elif plain:
        kind = "reported_unmarked_score"
    else:
        kind = "unsupported_score_annotation"
    unmarked = pairs[0] if plain else None
    half = pairs[1] if plain and len(pairs) == 2 else None
    if half and any(h > f for h, f in zip(half, unmarked)):
        kind, unmarked, half = "inconsistent_unmarked_halftime", None, None
    penalty_match = re.match(r"^([0-9]+)-([0-9]+)\s+pen\.", score_text)
    return {"source_score_text": score_text, "score_kind": kind,
            "source_score_pairs_in_order": pairs,
            "reported_unmarked_score": unmarked, "reported_unmarked_half_time": half,
            "has_extra_time_annotation": extra, "has_penalty_annotation": penalty,
            "reported_penalty_shootout_score": list(map(int, penalty_match.groups())) if penalty_match else None,
            "regulation_90_score": None, "verified_regulation_90": False,
            "score_semantics": "not_verified_for_training; annotated brackets not assigned to intervals",
            "source_venue_text": venue or None if separator else None}


def parse_cup_text(text: str, *, competition: str, season: str,
                   source_url: str, observed_at: datetime) -> dict:
    """Inspect one modern country-tagged file, retaining every candidate line."""
    aware(observed_at)
    _season_start(season)
    if competition not in COMPETITIONS or not isinstance(text, str):
        raise ValueError("Known cup code and decoded text are required")
    records, rejected, context_issues, unclassified = [], [], [], []
    stage, date_info, source_date_line = "", None, None
    header_match_count, titles = None, []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("= "):
            titles.append(line)
        header_count = re.fullmatch(r"#\s+Matches\s+([0-9]+)", line)
        if header_count:
            header_match_count = int(header_count.group(1))
        if line.startswith("▪"):
            stage = line.lstrip("▪").strip()
            # A new round/group must not accidentally inherit another group's
            # date. Valid files report a new date before their first match.
            date_info, source_date_line = None, None
            continue
        match_date = DATE_LINE.fullmatch(line)
        if match_date:
            source_date_line = line
            try:
                date_info = _date_context(match_date, season)
                if not date_info["weekday_consistent"]:
                    context_issues.append({"line": line_number, "raw_line": raw, "issue": "weekday_mismatch"})
            except ValueError as error:
                date_info = None
                context_issues.append({"line": line_number, "raw_line": raw, "issue": str(error)})
            continue
        if line.startswith(tuple(WEEKDAYS)) and " v " not in line:
            # A malformed date must invalidate context, not inherit the last day.
            date_info, source_date_line = None, line
            context_issues.append({"line": line_number, "raw_line": raw, "issue": "unsupported_date_line"})
            continue
        if line.startswith(("#", "=")):
            continue
        parsed = MATCH_LINE.fullmatch(line)
        if parsed is None:
            item = {"line": line_number, "raw_line": raw, "issue": "unsupported_line_shape"}
            (rejected if " v " in line or SCORE_PAIR.search(line) else unclassified).append(item)
            continue
        home, away = parsed["home"].strip(), parsed["away"].strip()
        if not home or not away or (home, parsed["hc"]) == (away, parsed["ac"]):
            rejected.append({"line": line_number, "raw_line": raw, "issue": "need_distinct_nonempty_teams"})
            continue
        clock = parsed["clock"]
        if clock and (int(clock[:2]) > 23 or int(clock[-2:]) > 59):
            rejected.append({"line": line_number, "raw_line": raw, "issue": "invalid_clock_text"})
            continue
        leg = re.search(r"\b([12])(?:st|nd)\s+Leg\b", stage, re.IGNORECASE)
        row = {"source_line_number": line_number, "raw_line": raw, "source_url": source_url,
               "observed_at": observed_at.isoformat(), "competition": competition, "season": season,
               "stage": stage, "source_clock_text": clock, "kickoff_utc": None,
               "historical_first_published_at": None, "neutral_venue": None,
               "home_advantage_verified": False, "explicit_leg_number": int(leg[1]) if leg else None,
               "tie_format_status": "explicit_leg_label" if leg else "not_explicitly_labeled",
               "home_label": home, "away_label": away,
               "home_country": parsed["hc"], "away_country": parsed["ac"],
               "match_date": None, "year_basis": "unknown", "weekday_consistent": None,
               "source_date_line": source_date_line}
        if date_info:
            row.update(date_info)
        row.update(_score_details(parsed["tail"]))
        records.append(row)
    key_counts = Counter((r["match_date"], r["stage"], r["home_label"], r["home_country"],
                          r["away_label"], r["away_country"]) for r in records)
    expected_title = f"= {COMPETITIONS[competition]} {season.replace('-', '/')}"
    return {"competition": competition, "season": season, "source_url": source_url,
            "source_titles": titles, "expected_title_matches": titles == [expected_title],
            "source_header_match_count": header_match_count, "parsed_rows": len(records),
            "header_count_difference": len(records) - header_match_count if header_match_count is not None else None,
            "date_unknown_rows": sum(r["match_date"] is None for r in records),
            "weekday_conflict_rows": sum(r["weekday_consistent"] is False for r in records),
            "score_kinds": dict(Counter(r["score_kind"] for r in records)),
            "verified_regulation_90_rows": 0, "neutral_venue_known_rows": 0,
            "explicit_leg_rows": sum(r["explicit_leg_number"] is not None for r in records),
            "duplicate_keys": [list(k) for k, n in key_counts.items() if n > 1],
            "rejected_match_lines": rejected, "context_issues": context_issues,
            "unclassified_lines": unclassified, "records": records}


def resolve_domestic_team(label: str, country: str, domestic_teams: Mapping[str, set[str]]) -> dict:
    """Case/accent-sensitive equality or an explicit country-tagged alias only."""
    league = COUNTRY_LEAGUE.get(country)
    if league is None:
        return {"status": "outside_five_league_country_scope", "league": None, "team_key": None}
    target = label if label in domestic_teams.get(league, set()) else ALIASES.get((country, label))
    if target is None:
        return {"status": "unmapped_source_name", "league": league, "team_key": None}
    if target not in domestic_teams.get(league, set()):
        return {"status": "target_not_in_same_season_domestic_file", "league": league,
                "target_label": target, "team_key": None}
    return {"status": "exact_country_scoped_label" if target == label else "explicit_reviewed_alias",
            "league": league, "target_label": target, "team_key": f"football-data:{league}:{target}"}


def measure_bridges(inspection: dict, domestic_teams: Mapping[str, set[str]]) -> dict:
    """Count identity bridges, not comparable strengths or usable training rows."""
    result = {k: v for k, v in inspection.items() if k != "records"}
    rows, links, edges, link_methods = [], {}, Counter(), Counter()
    both, one, outside = 0, 0, 0
    for original in inspection["records"]:
        row = dict(original)
        for side in ("home", "away"):
            link = resolve_domestic_team(row[f"{side}_label"], row[f"{side}_country"], domestic_teams)
            row[f"{side}_domestic_link"] = link
            source_identity = (row[f"{side}_country"], row[f"{side}_label"])
            links[source_identity] = {"source_country": source_identity[0], "source_label": source_identity[1], **link}
            link_methods[link["status"]] += 1
        h, a = row["home_domestic_link"], row["away_domestic_link"]
        matched = int(h["team_key"] is not None) + int(a["team_key"] is not None)
        both += matched == 2
        one += matched == 1
        outside += matched == 0
        if matched == 2 and h["league"] != a["league"]:
            edges[tuple(sorted((h["league"], a["league"])))] += 1
        rows.append(row)
    result.update({"both_teams_linked_rows": both, "one_team_linked_rows": one,
                   "neither_team_linked_rows": outside, "cross_domestic_league_rows": sum(edges.values()),
                   "cross_domestic_league_edges": [{"leagues": list(k), "match_rows": v} for k, v in sorted(edges.items())],
                   "linked_team_appearances_by_method": dict(link_methods),
                   "team_links": sorted(links.values(), key=lambda r: (r["source_country"], r["source_label"])),
                   "records": rows})
    return result


def _markdown(summary: dict) -> str:
    lines = ["# 欧战与五大联赛：小规模跨赛事连接测量", "",
             "本轮只测量历史比赛的身份连接与字段覆盖；未训练、未增加评级或战意加分、未生成当前预测。", "",
             "来源是openfootball维护的公共数据库，不是UEFA官方发布接口。仓库声明CC0，逐场事实与原始来源仍未独立全部核验。", "",
             f"本次解析 {summary['totals']['parsed_rows']} 行；双方连入 {summary['totals']['both_teams_linked_rows']} 行，"
             f"其中跨五联赛 {summary['totals']['cross_domestic_league_rows']} 行。已核实90分钟比分和中立场均为0，不能直接进入球队强度训练。", "",
             "| 赛季 | 赛事 | 头部声明场次 | 解析 | 双方连入 | 跨五联赛 | 未标注加时比分 | 加时/点球标注 | 日期未知 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cup in summary["competitions"]:
        kinds = cup["score_kinds"]
        lines.append(f"| {cup['season']} | {cup['competition']} | {cup['source_header_match_count']} | {cup['parsed_rows']} | "
                     f"{cup['both_teams_linked_rows']} | {cup['cross_domestic_league_rows']} | "
                     f"{kinds.get('reported_unmarked_score', 0)} | "
                     f"{kinds.get('extra_time_annotation', 0) + kinds.get('penalty_shootout_annotation', 0)} | {cup['date_unknown_rows']} |")
    lines += ["", "## 可以和不可以据此得出什么", "",
              "- country+原始名称精确匹配，或小型显式人工别名表；不模糊匹配、不删重音、不自动去FC。别名目标必须出现在同季本地CSV；Monaco/MCO连法甲为显式例外。",
              "- 同季全部球队名单仅用于回溯身份覆盖测量，不是当年赛前已知名单，也没有把赛季结局/全季名单作为预测特征。跨联赛有连接边不等于已估计联赛强弱或验证可迁移性。",
              "- 欧战参赛队是被选择的强队群体，不能无层级区分地代表整条联赛；同队重复比赛、同一淘汰赛两回合相关，不能把行数当独立样本量。",
              "- 未标注加时的比分只叫reported_unmarked_score候选；regulation_90_score仍未知。加时/点球所有数字按原顺序保留，绝不把射点球比分或第一组数字直接当90分钟进球。",
              "- 中立场与实际主场优势均未知，不能把文本team1当已证实主场；阶段/显式leg标签保留，但不由双循环、小组对阵或对手反向组合擅自推断两回合第几场。",
              "- 缺年份日期按该欧洲赛季7月至次年6月窗口定位，并保留derived标签；不同小组反复回到9月，不按文件行序盲目递增年份。日期缺失、非法日期和星期冲突均保留。",
              "- observed_at是本次收到文件时点，不是历史赛前可得时点。无赔率、历史首发发布时间、伤停、完整跨赛事负荷或赛前战意。没有采用赛后阵容/排名作预测输入。",
              "- 本样本仅含指定两季正赛文本；没有证明资格赛、欧战所有赛季或当前2026/27完整，不能把source_header_match_count一致当独立事实验证。",
              "- 下载器不重试403、不轮换代理、不执行第三方仓库代码；错误及已下载原始文本保留，没有静默替代数据源。", "",
              "## 来源与本地材料", ""]
    for source in summary["sources"]:
        lines.append(f"- [{Path(source['local_path']).name}]({source['url']})：{source['status']}；{source['observed_at']}。")
    for season, source in summary["domestic_sources"].items():
        lines.append(f"- {season}本地Football-Data：`{source['input_directory']}`；{source['records']}条回溯赛果；"
                     f"逐联赛来源URL与原CSV路径保留在JSON。")
    if summary["inspection_errors"]:
        lines += ["", "## 解析/连接失败", "", *[f"- {e}" for e in summary["inspection_errors"]]]
    lines += ["", f"原始文本、LICENSE、README：`{summary['raw_directory']}`。", "",
              f"逐行原文、score语义、别名、跨联赛边及缺口：`{summary['result_json']}`。", ""]
    return "\n".join(lines)


def run_probe(task_root: Path, domestic_directories: Mapping[str, Path], *, fetch: Fetch = _read) -> tuple[dict, Path]:
    if not domestic_directories:
        raise ValueError("Supply at least one explicit same-season domestic CSV directory")
    root = Path(task_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw = root / "data" / "openfootball-cups" / stamp
    output = root / "outputs" / f"跨赛事连接测量_{stamp}"
    raw.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    sources, inspections, errors, domestic_sources = [], [], [], {}
    for season, path in sorted(domestic_directories.items()):
        _season_start(season)
        domestic_records, domestic_report = load_directory(Path(path), season.replace("-", "/"))
        if domestic_report["rejected_rows"] or domestic_report["duplicate_keys"]:
            raise ValueError("Domestic source contains rejected/duplicate rows; inspect ordinary parser report first")
        domestic_sources[season] = {key: domestic_report[key] for key in ("input_directory", "records", "leagues")}
        teams = {league: set() for league in LEAGUES}
        for row in domestic_records:
            teams[row.competition].update((row.home_label, row.away_label))
        for code in COMPETITIONS:
            path_out = raw / f"{season}_{code}.txt"
            source = _download(f"{BASE}/{season}/{code}.txt", path_out, fetch)
            sources.append(source)
            if source["status"] != "downloaded":
                continue
            try:
                inspection = parse_cup_text(path_out.read_text(encoding="utf-8-sig"), competition=code,
                                            season=season, source_url=source["url"],
                                            observed_at=datetime.fromisoformat(source["observed_at"]))
                if not inspection["expected_title_matches"]:
                    raise ValueError("Downloaded text does not identify the requested cup and season")
                if inspection["source_header_match_count"] is None or not inspection["parsed_rows"]:
                    raise ValueError("Historical cup text lacks a match-count header or any parsable match row")
                inspections.append(measure_bridges(inspection, teams))
            except (ValueError, TypeError, UnicodeError) as error:
                errors.append({"season": season, "competition": code, "error_type": type(error).__name__, "error": str(error)})
    for filename in ("README.md", "LICENSE.md"):
        sources.append(_download(f"{BASE}/{filename}", raw / filename, fetch))
    result_path = output / "跨赛事连接结果.json"
    summary = {"status": "complete" if not errors and all(s["status"] == "downloaded" for s in sources) else "partial",
               "trained": False, "predictions_generated": False, "source_repository": "https://github.com/openfootball/champions-league",
               "license_declaration": "CC0-1.0; not independent assurance of each record's original provenance",
               "raw_directory": str(raw), "result_json": str(result_path), "sources": sources,
               "domestic_sources": domestic_sources, "competitions": inspections, "inspection_errors": errors,
               "totals": {key: sum(c[key] for c in inspections) for key in (
                   "parsed_rows", "both_teams_linked_rows", "one_team_linked_rows", "neither_team_linked_rows",
                   "cross_domestic_league_rows", "date_unknown_rows", "weekday_conflict_rows",
                   "verified_regulation_90_rows", "neutral_venue_known_rows")},
               "rejected_match_lines": sum(len(c["rejected_match_lines"]) for c in inspections),
               "alias_entries": [{"country": country, "source_label": label, "target_label": target,
                                  "method": "manual_explicit_country_scoped"}
                                 for (country, label), target in sorted(ALIASES.items())]}
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = output / "跨赛事连接说明.md"
    report.write_text(_markdown(summary), encoding="utf-8")
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=project_root())
    parser.add_argument("--domestic", nargs=2, action="append", metavar=("YYYY-YY", "CSV_DIRECTORY"), required=True)
    args = parser.parse_args()
    if len({season for season, _ in args.domestic}) != len(args.domestic):
        parser.error("Each season must be supplied once")
    result, report = run_probe(args.task_root, {season: Path(path) for season, path in args.domestic})
    print(json.dumps({"status": result["status"], "totals": result["totals"], "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
