"""Read local Football-Data CSVs without inventing historical timestamps.

Records are date-level retrospective observations. Source clock text has no
verified timezone; pre-closing and closing prices have no individual observation
times. Neither is an OddsSnapshot or a precise T-24h information set.
"""

import csv
from dataclasses import dataclass
from datetime import date, datetime, time
import json
import math
from pathlib import Path
import re


LEAGUES = {"E0": "英超", "D1": "德甲", "SP1": "西甲", "I1": "意甲", "F1": "法甲"}
MARKET_COLUMNS = {
    "avg_preclosing": ("AvgH", "AvgD", "AvgA"),
    "avg_closing": ("AvgCH", "AvgCD", "AvgCA"),
    "b365_preclosing": ("B365H", "B365D", "B365A"),
    "b365_closing": ("B365CH", "B365CD", "B365CA"),
}
AWARDED_SOURCE = (
    "https://www.bundesliga.com/de/bundesliga/news/"
    "vfl-bochum-union-berlin-schiedsgericht-urteil-drei-punkte-feuerzeug-wurf-31928"
)
SPECIAL_MATCHES = {
    ("D1", date(2022, 3, 18), "Bochum", "M'gladbach"):
        ("administratively_awarded_incomplete_match",
         "https://www.dfb.de/fileadmin/_dfbdam/257416-220324_VfL_Bochum.pdf"),
    ("I1", date(2020, 9, 19), "Verona", "Roma"):
        ("administratively_awarded_result",
         "https://www.figc.it/it/federazione/news/respinto-il-ricorso-della-roma-confermata-la-sconfitta-per-3-0-a-tavolino-con-il-verona-ha0usqwf"),
    ("F1", date(2021, 10, 27), "Nice", "Marseille"):
        ("neutral_venue_not_represented",
         "https://www.lfp.fr/article/ogc-nice-olympique-de-marseille-le-27-octobre-a-21-h-au-stade-de-laube-a-troyes"),
    ("D1", date(2024, 12, 14), "Union Berlin", "Bochum"):
        ("administratively_awarded_result", AWARDED_SOURCE),
    ("I1", date(2024, 4, 25), "Udinese", "Roma"):
        ("resumed_partially_played_match",
         "https://www.udinese.it/news/squad/date-confirmed-for-resumption-of-udinese-v-roma"),
    ("SP1", date(2023, 12, 11), "Granada", "Ath Bilbao"):
        ("resumed_partially_played_match",
         "https://www.athletic-club.eus/noticias/2023/12/11/asi-se-reiniciara-el-granada-athletic-de-laliga/"),
    ("F1", date(2025, 3, 16), "Montpellier", "St Etienne"):
        ("administratively_confirmed_incomplete_match",
         "https://ligue1.com/fr/articles/l1_article_2269-le-point-sur-la-lutte-pour-le-maintien-020425-l1"),
    ("F1", date(2026, 5, 17), "Nantes", "Toulouse"):
        ("administratively_confirmed_incomplete_match",
         "https://ligue1.com/fr/articles/l1_article_5189-discipline-toutes-les-decisions-l1-270526"),
}


@dataclass
class HistoricalMatch:
    match_id: str
    competition: str
    season: str
    match_date: date
    source_time_text: str | None
    home: str
    away: str
    home_label: str
    away_label: str
    full_time: tuple[int, int]
    half_time: tuple[int, int] | None
    markets: dict[str, tuple[float, float, float]]
    source_path: str
    source_url: str
    goal_fit_exclusion: str | None = None


def _goal(value: str | None, column: str) -> int:
    text = (value or "").strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{column} must be a non-negative integer")
    return int(text)


def _result(score: tuple[int, int]) -> str:
    return "H" if score[0] > score[1] else "A" if score[0] < score[1] else "D"


def _date(text: str) -> date:
    # Restrict the shape so malformed or ambiguous non-European dates fail.
    if not re.fullmatch(r"[0-9]{2}/[0-9]{2}/(?:[0-9]{4}|[0-9]{2})", text):
        raise ValueError("Date must use dd/mm/yyyy or dd/mm/yy")
    return datetime.strptime(text, "%d/%m/%Y" if len(text) == 10 else "%d/%m/%y").date()


def parse_row(row: dict[str, str], *, competition: str, season: str,
              source_path: str, source_url: str) -> tuple[HistoricalMatch, list[dict]]:
    """Parse one row; report absent markets/HT without filling invented values."""
    if competition not in LEAGUES or (row.get("Div") or "").strip() != competition:
        raise ValueError("CSV division does not match its league filename")
    match_date = _date((row.get("Date") or "").strip())
    home, away = (row.get("HomeTeam") or "").strip(), (row.get("AwayTeam") or "").strip()
    if not home or not away or home == away:
        raise ValueError("Need two distinct nonempty team labels")
    clock = (row.get("Time") or "").strip() or None
    if clock is not None:
        if not re.fullmatch(r"[0-9]{2}:[0-9]{2}", clock):
            raise ValueError("Time must be HH:mm clock text, not a timezone assertion")
        time.fromisoformat(clock)
    full = (_goal(row.get("FTHG"), "FTHG"), _goal(row.get("FTAG"), "FTAG"))
    if (row.get("FTR") or "").strip() != _result(full):
        raise ValueError("FTR disagrees with full-time goals")
    special = SPECIAL_MATCHES.get((competition, match_date, home, away))
    exclusion = None if special is None else special[0]
    match_id = f"football-data:{competition}:{season}:{home}|{away}"
    issues = []
    half_values = [(row.get(c) or "").strip() for c in ("HTHG", "HTAG", "HTR")]
    if not all(half_values):
        half = None
        issues.append({"match_id": match_id, "field": "half_time",
                       "issue": "partial_missing" if any(half_values) else "missing"})
    else:
        half = (_goal(half_values[0], "HTHG"), _goal(half_values[1], "HTAG"))
        if half_values[2] != _result(half):
            raise ValueError("HTR disagrees with half-time goals")
        if exclusion is None and any(h > f for h, f in zip(half, full)):
            raise ValueError("Half-time score exceeds full-time score")
    markets = {}
    for market, columns in MARKET_COLUMNS.items():
        values = [(row.get(c) or "").strip() for c in columns]
        if not all(values):
            issues.append({"match_id": match_id, "field": market, "issue": "missing",
                           "columns": [c for c, value in zip(columns, values) if not value]})
            continue
        try:
            prices = tuple(float(v) for v in values)
            if any(not math.isfinite(p) or p <= 1 for p in prices):
                raise ValueError("Decimal odds must be finite and greater than one")
        except ValueError:
            issues.append({"match_id": match_id, "field": market, "issue": "invalid_prices"})
            continue
        markets[market] = prices
    return HistoricalMatch(
        match_id, competition, season, match_date, clock,
        f"football-data:{competition}:{home}", f"football-data:{competition}:{away}",
        home, away, full, half, markets, source_path, source_url, exclusion,
    ), issues


def load_directory(path: Path, season: str = "2024/25") -> tuple[list[HistoricalMatch], dict]:
    """Load existing known-league files; never download or modify source CSVs.

    Invalid rows and duplicate keys are explicit in the report. Duplicates remain
    visible in the returned list, not silently overwritten; callers must refuse
    fitting while ``rejected_rows`` or ``duplicate_keys`` is nonempty.
    """
    path = Path(path)
    if not re.fullmatch(r"[0-9]{4}/[0-9]{2}", season):
        raise ValueError("Season must use YYYY/YY")
    season_code = season[2:4] + season[-2:]
    source_urls = {}
    manifest = path / "下载来源.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
        for item in meta.get("files", []):
            source_urls[Path(item["path"]).name] = item["source"]
    records, issues, rejected, duplicate_keys, exclusions = [], [], [], [], []
    leagues = {}
    seen = set()
    for competition, label in LEAGUES.items():
        file_path = path / f"{competition}.csv"
        if not file_path.exists():
            continue
        source_url = source_urls.get(file_path.name,
            f"https://football-data.co.uk/mmz4281/{season_code}/{competition}.csv")
        league_records = []
        input_rows = 0
        with file_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{file_path.name}: missing required result columns")
            for line, row in enumerate(reader, start=2):
                if not any(value and str(value).strip() for value in row.values()):
                    continue
                input_rows += 1
                try:
                    record, row_issues = parse_row(row, competition=competition, season=season,
                        source_path=str(file_path.resolve()), source_url=source_url)
                except (ValueError, TypeError) as error:
                    rejected.append({"file": file_path.name, "line": line, "issue": str(error)})
                    continue
                if record.match_id in seen:
                    duplicate_keys.append(record.match_id)
                seen.add(record.match_id)
                league_records.append(record)
                records.append(record)
                issues.extend(row_issues)
                if record.goal_fit_exclusion:
                    exclusions.append({"match_id": record.match_id,
                        "date": record.match_date.isoformat(), "reason": record.goal_fit_exclusion,
                        "reported_official_score": list(record.full_time),
                        "handling": "Retain source result; exclude from goal fit and main scoring",
                        "source": SPECIAL_MATCHES[(competition, record.match_date,
                            record.home_label, record.away_label)][1]})
        dates = [r.match_date for r in league_records]
        leagues[competition] = {
            "label": label, "input_rows": input_rows, "parsed_rows": len(league_records),
            "goal_fit_rows": sum(r.goal_fit_exclusion is None for r in league_records),
            "teams": len({t for r in league_records for t in (r.home, r.away)}),
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
            "half_time_rows": sum(r.half_time is not None for r in league_records),
            "market_rows": {m: sum(m in r.markets for r in league_records) for m in MARKET_COLUMNS},
            "source_path": str(file_path.resolve()), "source_url": source_url,
        }
    if not leagues:
        raise ValueError("No known-league CSV files found")
    records.sort(key=lambda r: (r.match_date, r.competition, r.match_id))
    report = {
        "season": season, "input_directory": str(path.resolve()), "records": len(records),
        "goal_fit_rows": sum(r.goal_fit_exclusion is None for r in records),
        "date_min": min(r.match_date for r in records).isoformat() if records else None,
        "date_max": max(r.match_date for r in records).isoformat() if records else None,
        "leagues": leagues, "rejected_rows": rejected,
        "duplicate_keys": sorted(set(duplicate_keys)), "field_issues": issues,
        "excluded_matches": exclusions,
        "time_semantics": {
            "granularity": "source_calendar_date_only",
            "kickoff_timezone": "unverified",
            "historical_first_availability": "unknown",
            "odds_observation_times": "unknown",
            "prediction_scope": "retrospective date-level experiment, not T-24h/lineup-confirmed",
        },
        "notes": [
            "Pre-closing is not necessarily opening; closing is not a precise T-24h snapshot",
            "Source download time is not historical information availability",
            "Team identifiers are source/league-local, not cross-source or cross-league aliases",
            "Post-match shots/cards are not exposed as pre-match predictors by this adapter",
            "Administrative results are not treated as on-pitch goal observations",
            "Known partial resumptions are conservatively excluded from date-level fit and main scoring; source dates/scores are retained",
        ],
    }
    return records, report
