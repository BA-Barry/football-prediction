"""Small official openfootball snapshots and date-level coverage, never training.

The reference calendar date is only for classifying source-reported dates; it
does not reconstruct historical knowledge or a real-time prediction information
set. All project output is written below the explicitly supplied task root.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .openfootball import inspect_payload


COMPETITIONS = {"en.1": "英超", "de.1": "德甲", "es.1": "西甲", "it.1": "意甲", "fr.1": "法甲"}
BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
Fetch = Callable[[str], tuple[bytes, dict]]


def _read(url: str) -> tuple[bytes, dict]:
    # No browser impersonation, proxies, retry loop or 403/CAPTCHA bypass.
    request = Request(url, headers={"User-Agent": "football-research-snapshot/0.19"})
    with urlopen(request, timeout=30) as response:
        return response.read(), dict(response.headers)


def _calendar_coverage(inspection: dict, calendar_date: date) -> dict:
    records = inspection["records"]
    dated = [(date.fromisoformat(row["match_date"]), row["full_time"]) for row in records]
    scored = [day for day, ft in dated if ft is not None]
    scored_not_after = [day for day in scored if day <= calendar_date]
    latest = max(scored) if scored else None
    latest_not_after = max(scored_not_after) if scored_not_after else None
    return {
        "past_dated_rows": sum(day < calendar_date for day, _ in dated),
        "past_dated_without_result_rows": sum(day < calendar_date and ft is None for day, ft in dated),
        "on_calendar_date_without_result_rows": sum(day == calendar_date and ft is None for day, ft in dated),
        "future_dated_rows": sum(day > calendar_date for day, _ in dated),
        "future_scored_anomaly_rows": sum(day > calendar_date and ft is not None for day, ft in dated),
        "latest_scored_date": latest.isoformat() if latest else None,
        "latest_scored_date_gap_days": (calendar_date - latest).days if latest else None,
        "latest_scored_on_or_before_calendar_date": latest_not_after.isoformat() if latest_not_after else None,
        "latest_scored_on_or_before_gap_days": (calendar_date - latest_not_after).days if latest_not_after else None,
        "calendar_coverage_basis": "successfully parsed rows only; rejected rows reported separately",
    }


def _download(url: str, path: Path, fetch: Fetch) -> dict:
    metadata = {"url": url, "local_path": str(path)}
    try:
        body, headers = fetch(url)
        observed = datetime.now(timezone.utc)
        if not isinstance(body, bytes):
            raise ValueError("fetch must return raw bytes and HTTP headers")
        lower_headers = {str(key).lower(): value for key, value in headers.items()}
        # A source snapshot is never overwritten: its parent is unique per run.
        with path.open("xb") as handle:
            handle.write(body)
        metadata.update({
            "status": "downloaded", "observed_at": observed.isoformat(), "bytes": len(body),
            "http_last_modified_not_match_publication_time": lower_headers.get("last-modified"),
            "http_date_not_match_publication_time": lower_headers.get("date"),
        })
    except Exception as error:
        metadata.update({"status": "error", "observed_at": datetime.now(timezone.utc).isoformat(),
                         "error_type": type(error).__name__, "error": str(error)})
        if isinstance(error, HTTPError):
            metadata["http_status"] = error.code
    return metadata


def _markdown(summary: dict) -> str:
    lines = ["# openfootball 当前文件快照与覆盖测量", "",
             f"赛季：{summary['season']}；来源日历参考日：{summary['source_calendar_date']}。", "",
             "仅接入和检查真实源文件，不训练、不更新球队参数、不生成当前比赛预测。", "",
             "日历参考日不是真实历史 as-of 时点；选择更早参考日不会还原当时版本。"
             "日期比较也不等于已核实比赛状态：日期早于参考日且缺FT，只称“过去日期缺比分”。", "",
             "| 联赛 | 原始行 | 可解析 | FT | 过去日期缺比分 | 未来日期带比分 | 最近有比分日期 | 日期差/天 | 拒绝行 | 重复键 |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in summary["competitions"]:
        latest = row["latest_scored_date"] or "未知"
        gap = row["latest_scored_date_gap_days"]
        lines.append(f"| {row['label']} | {row['input_rows']} | {row['parsed_rows']} | {row['full_time_rows']} | "
                     f"{row['past_dated_without_result_rows']} | {row['future_scored_anomaly_rows']} | {latest} | "
                     f"{gap if gap is not None else '未知'} | {row['rejected_rows']} | {len(row['duplicate_keys'])} |")
    lines += ["", "日期差 = 参考日减最近有比分日期；负数表示未来日期带比分异常，不能解释为数据更新更及时。"
              "以上日期统计只包括可解析行；格式拒绝、重复键另外保留，不静默去重或补零。", "",
              "## 格式与缺失", ""]
    for row in summary["competitions"]:
        lines.append(f"- {row['label']}原始score形状：`{json.dumps(row['raw_score_formats'], ensure_ascii=False)}`；"
                     f"缺FT {row['missing_full_time_rows']}，缺HT {row['missing_half_time_rows']}，"
                     f"已核实UTC开球 {row['verified_utc_kickoff_rows']}，历史首次可用时点 {row['historical_first_availability_rows']}，"
                     f"赔率 {row['odds_rows']}，首发 {row['lineup_rows']}，伤停 {row['injury_rows']}。")
    lines += ["", "## 来源与观察时点", ""]
    for source in summary["sources"]:
        modified = source.get("http_last_modified_not_match_publication_time") or "未报告"
        lines.append(f"- [{Path(source['local_path']).name}]({source['url']})：{source['status']}；"
                     f"观察于 {source['observed_at']}；HTTP Last-Modified：{modified}。")
        if source["status"] == "error":
            lines.append(f"  失败原因：{source['error_type']}：{source['error']}。没有替代来源或绕过访问限制。")
    if summary["inspection_errors"]:
        lines += ["", "## 文件检查失败", ""]
        lines += [f"- {row['competition']}：{row['error_type']}：{row['error']}。原始下载文件保留。"
                  for row in summary["inspection_errors"]]
    lines += ["", "## 不能据此推断", "",
              "- observed_at仅是本次收到文件的观察时点；HTTP修改时间仅描述HTTP资源，不是比赛／首发／赔率历史发布时间。",
              "- 上游README说明JSON可每日生成，但Football.TXT尚非每日自动更新；生成新文件不证明比分新鲜、完整或正确。",
              "- score对象和数组均保留原始形状。ET／点球字段不替代缺失FT；FT在新增赛事上的90分钟语义仍需核实。",
              "- 来源时间文本没有已核实的时区；未重建UTC开球、历史赛前24小时信息、官方首发公布时间、伤停或盘口。",
              "- 来源本地球队键不是全球稳定ID；未建立跨联赛强度桥梁，未把当前赛季作为独立盲测。",
              "- 仓库声明CC0；逐场原始来源尚未完全核实，软件／仓库许可不自动替代第三方数据权利。", "",
              f"原始JSON及官方许可、README：`{summary['raw_directory']}`。", "",
              f"机器可读覆盖报告：`{summary['coverage_json']}`。", ""]
    return "\n".join(lines)


def run_snapshot(task_root: Path, season: str = "2026-27", calendar_date: date | None = None,
                 fetch: Fetch | None = None) -> tuple[dict, Path]:
    """Fetch seven small official files; calendar_date is a comparison date only.

    ``fetch(url) -> (raw_bytes, headers)`` can be injected for offline tests. No
    downloaded Python is executed, and no model fitting is imported or called.
    """
    if not isinstance(season, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", season):
        raise ValueError("season must use YYYY-YY")
    if int(season[5:]) != (int(season[:4]) + 1) % 100:
        raise ValueError("season must name consecutive years")
    if calendar_date is not None and type(calendar_date) is not date:
        raise ValueError("calendar_date must be a date, not a timestamp")
    task_root = Path(task_root).expanduser().resolve()
    started = datetime.now(timezone.utc)
    reference_date = calendar_date if calendar_date is not None else started.date()
    run_name = started.strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = task_root / "data" / "openfootball" / season / run_name
    output_dir = task_root / "outputs" / f"openfootball接入_{run_name}"
    raw_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    reader = fetch or _read
    requests = [(f"{BASE}/{season}/{code}.json", raw_dir / f"{code}.json") for code in COMPETITIONS]
    requests.extend((f"{BASE}/{name}", raw_dir / name) for name in ("LICENSE.md", "README.md"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(lambda item: _download(*item, reader), requests))
    inspections, errors = [], []
    for code, label in COMPETITIONS.items():
        source = next(source for source in sources if Path(source["local_path"]).name == f"{code}.json")
        if source["status"] != "downloaded":
            continue
        try:
            payload = json.loads(Path(source["local_path"]).read_bytes().decode("utf-8-sig"))
            inspection = inspect_payload(payload, competition=code, season=season,
                                         source_url=source["url"], observed_at=datetime.fromisoformat(source["observed_at"]))
            inspection.update({"label": label, **_calendar_coverage(inspection, reference_date)})
            inspection["http_last_modified_not_match_publication_time"] = source["http_last_modified_not_match_publication_time"]
            inspection_path = raw_dir / f"{code}.inspection.json"
            inspection_path.write_text(json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8")
            inspections.append({key: value for key, value in inspection.items() if key != "records"})
        except (ValueError, TypeError, UnicodeError) as error:
            errors.append({"competition": code, "source_url": source["url"],
                           "error_type": type(error).__name__, "error": str(error)})
    totals = {key: sum(row[key] for row in inspections) for key in (
        "input_rows", "parsed_rows", "rejected_rows", "full_time_rows", "missing_full_time_rows",
        "half_time_rows", "past_dated_without_result_rows", "future_scored_anomaly_rows")}
    totals.update({"inspected_competitions": len(inspections),
                   "duplicate_source_keys": sum(len(row["duplicate_keys"]) for row in inspections),
                   "download_failure_count": sum(source["status"] == "error" for source in sources)})
    coverage_json = output_dir / "覆盖报告.json"
    summary = {
        "status": "partial" if totals["download_failure_count"] or errors else "complete",
        "status_semantics": "download and inspection execution only; complete does not assert complete or valid match coverage",
        "purpose": "date-level source coverage inspection only", "trained": False,
        "predictions_generated": False, "season": season, "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "source_calendar_date": reference_date.isoformat(),
        "source_calendar_date_default": "UTC observation calendar day" if calendar_date is None else "explicit date",
        "source_calendar_date_semantics": "source-date comparison only, not historical as-of availability or filtering",
        "source_license": "Repository declares CC0-1.0; original per-record rights not fully verified",
        "raw_directory": str(raw_dir), "output_directory": str(output_dir), "coverage_json": str(coverage_json),
        "sources": sources, "competitions": inspections, "inspection_errors": errors, "totals": totals,
    }
    coverage_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = output_dir / "接入与覆盖报告.md"
    report.write_text(_markdown(summary), encoding="utf-8")
    return summary, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot official openfootball files; inspect only, never train")
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--source-calendar-date", type=date.fromisoformat,
                        help="YYYY-MM-DD reference date; NOT a historical as-of timestamp")
    args = parser.parse_args(argv)
    summary, report = run_snapshot(args.task_root, season=args.season, calendar_date=args.source_calendar_date)
    print(json.dumps({"report": str(report), "coverage_json": summary["coverage_json"],
                      "raw_directory": summary["raw_directory"], "status": summary["status"],
                      "trained": False, **summary["totals"]}, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
