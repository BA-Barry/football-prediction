"""Score saved forecasts using source-linked completion observations, no refit.

An official completed-match report and a provider's delayed administrative
final flag need not agree. Keep both observations and disclose which supports
settlement. Observation time is a completion upper bound, never a whistle time.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit

from .batch_multiline import LABELS, _fixture, _paths, _read, _write
from .multiline import TRACKS, score_multi
from .prospective import SETTLEMENT, _settled, _text, _time


def _identity(row, fixture, id_field, kickoff_field):
    if (row.get(id_field) != fixture.match_id or row.get("home") != fixture.home
            or row.get("away") != fixture.away
            or _time(row.get(kickoff_field), "source kickoff") != fixture.kickoff):
        raise ValueError("Result source must match saved fixture identity, teams and kickoff")


def _result(fixture, home_goals, away_goals, bound, available, source_url, observed, now):
    parsed = urlsplit(_text(source_url, "result source URL"))
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("Result source requires an HTTP(S) URL")
    received = _time(observed, "result source observed_at")
    bound, available = _time(bound, "finished_by"), _time(available, "available_at")
    if received > available or available > now or bound < received:
        raise ValueError("Result source/availability/bound times cannot be future or fabricated earlier observations")
    result = {"fixture_id": fixture.match_id, "home_goals": home_goals, "away_goals": away_goals,
              "finished_by": bound.isoformat(), "available_at": available.isoformat(),
              "settlement": SETTLEMENT, "source_url": source_url,
              "source_observed_at": received.isoformat(),
              "completion_time_note": "finished_by is an observed completion upper bound, not the actual whistle time"}
    _settled(result, fixture)
    return result


def source_results(path, fixtures, now):
    """Normalize the two already saved official observation formats.

    Source assertions are caller provenance, not cryptographic authentication.
    No source is fetched and no article text is executed or treated as a prompt.
    """
    source = _read(path)
    complete, observations = {}, []
    if "fixture_observations" in source:
        metadata = source["source_metadata"]["fixtures"]
        if metadata.get("status") != 200:
            raise ValueError("FPL result snapshot must have a successful source response")
        observed = metadata["received_at"]
        seen = set()
        for row in source["fixture_observations"]:
            fid = row["match_id"]
            if fid not in fixtures or fid in seen:
                raise ValueError("Unknown or duplicate source fixture")
            seen.add(fid)
            _identity(row, fixtures[fid], "match_id", "expected_kickoff")
            if row.get("verified") is not True:
                observations.append({"fixture_id": fid, "status": "unverified_source_identity", "source": str(path)})
                continue
            _identity(row, fixtures[fid], "match_id", "kickoff_time")
            if row.get("finished") is not True:
                observations.append({"fixture_id": fid, "status": "provisional" if row.get("finished_provisional") else "not_final",
                                     "source": str(path), "finished": row.get("finished"),
                                     "finished_provisional": row.get("finished_provisional"),
                                     "home_goals": row.get("home_goals"), "away_goals": row.get("away_goals")})
                continue
            complete[fid] = _result(fixtures[fid], row["home_goals"], row["away_goals"], observed,
                                    source["recorded_at"], metadata["source_url"], observed, now)
            complete[fid]["completion_basis"] = "official_fpl_final_flag"
            observations.append({"fixture_id": fid, "status": "confirmed_full_time", "source": str(path)})
    elif "completion_supported" in source:
        fid = source["match_id"]
        if fid not in fixtures:
            raise ValueError("Unknown completed-report fixture")
        fixture = fixtures[fid]
        _identity(source, fixture, "match_id", "kickoff")
        if source.get("completion_supported") is not True:
            return {}, [{"fixture_id": fid, "status": "report_does_not_confirm_completion", "source": str(path)}]
        if source.get("source_kind") not in {"official_home_club_match_report", "official_away_club_match_report"} or not source.get("body_completion_evidence"):
            raise ValueError("Completed club report requires explicit body completion evidence")
        raw, = _paths(source["original_raw_path"])
        if not raw.is_file():
            raise ValueError("Saved original report body is missing")
        result = _result(fixture, source["score"]["home"], source["score"]["away"],
                         source["finished_by_observation"], source["recorded_at"], source["source_url"],
                         source["original_received_at"], now)
        result.update(completion_basis="official_completed_match_report",
                      source_kind=source["source_kind"],
                      raw_evidence_path=str(raw), corroborating_provider=source.get("fpl_crosscheck"),
                      source_publication_time=source.get("published_at"))
        complete[fid] = result
        observations.append({"fixture_id": fid, "status": "confirmed_full_time_by_club_report",
                             "source": str(path), "provider_final_flag_may_still_be_false": True})
    else:
        raise ValueError("Unsupported saved result observation format")
    return complete, observations


def weekly_scores(records, results):
    """One fixture once, grouped by UTC kickoff ISO week; no inference/retuning."""
    groups = {}
    for record in records:
        iso = _time(record["fixture"]["kickoff"], "kickoff").isocalendar()
        groups.setdefault(f"{iso.year}-W{iso.week:02d}", []).append(record)
    return {week: score_multi(rows, results) for week, rows in sorted(groups.items())}


def report(payload, scores, periods):
    lines = ["# 已保存赛前预测的真实赛果评分", "",
             f"比赛记录{scores['n_records']}场，已录入完赛{scores['n_settled']}场，待结算{len(scores['pending_fixture_ids'])}场。",
             "只用原预测评分；本次球队拟合0次、LLM调用0次、参数搜索／变更0次。", "",
             "| 比赛 | 常规时间结果 | 线路 | 实际结果的赛前概率 | Log loss | 三类求和Brier | 最高概率是否命中 |",
             "|---|---|---|---:|---:|---:|---:|"]
    records = {row["fixture"]["match_id"]: row for row in payload["records"]}
    for item in scores["per_fixture"]:
        fid = item["fixture_id"]
        record, result = records[fid], payload["results"][fid]
        outcome = ("home", "draw", "away").index(item["outcome"])
        for arm in TRACKS:
            if arm not in item["scores"]:
                lines.append(f"| {record['fixture']['home']}—{record['fixture']['away']} | {result['home_goals']}–{result['away_goals']} | {LABELS[arm]} | 缺失 | — | — | — |")
                continue
            values = item["scores"][arm]
            loss = "∞" if values["log_loss_infinite"] else f"{values['log_loss']:.6f}"
            lines.append(f"| {record['fixture']['home']}—{record['fixture']['away']} | {result['home_goals']}–{result['away_goals']} | {LABELS[arm]} | {record['tracks'][arm]['probabilities'][outcome]:.2%} | {loss} | {values['brier']:.6f} | {values['top1_accuracy']:.1f} |")
    lines += ["", "Log loss = −ln(实际结果的原预测概率)，Brier为三类概率误差平方和，均越低越好。最高概率并列仍沿用均匀随机选取的期望命中率。", "",
              "## 来源与完赛时间", ""]
    for fid, result in payload["results"].items():
        lines += [f"- [{fid}]({result.get('source_url', '')})：{result.get('completion_basis', '原有赛果记录')}；"
                  f"{('finished_by=' + result['finished_by']) if 'finished_by' in result else ('finished_at=' + result['finished_at'])}；available_at={result['available_at']}。"]
    lines += ["", "finished_by只是该时点已确认比赛结束的上界，不是实际终场秒数。俱乐部明确完赛报道可支持结算；若FPL管理性final标记仍为false则原样保留，不把该字段改成true。后续若来源更正比分，须单独审阅，不静默覆盖。", "",
              "## 逐轮与累计解释", ""]
    for week, values in periods.items():
        lines.append(f"- UTC开球日历周{week}：已结算{values['n_settled']}场；全五线共同已结算{values['all_five_common']['n']}场。")
    lines += ["", "完整10组成对比较、覆盖和缺失均另存JSON。组合缺失的比赛不回填组合，也不因全五线不齐就丢掉现有成对比较。统计线与知情LLM相同不是两份独立证据。", "",
              "当前是探索性描述，单场没有可估计的跨比赛标准差或可靠校准结论，不做t检验、显著性检验或胜者选择。此轮不因一个结果调权重；以后只在多轮重复偏差下提出假设，在更晚比赛检验。", "",
              "原赛前记录、概率、模型和生成时间均未改动；新产物仅加入有来源的赛果。来源完整性是普通数据校验，不是新增发布门禁。", ""]
    return "\n".join(lines)


def run(root, case_path, evidence_paths):
    root, case_path, *evidence_paths = _paths(root, case_path, *evidence_paths)
    payload = deepcopy(_read(case_path))
    now = datetime.now(timezone.utc)
    fixtures = {row["fixture"]["match_id"]: _fixture(row["fixture"]) for row in payload["records"]}
    results = payload.setdefault("results", {})
    observations = []
    for path in evidence_paths:
        normalized, source_observations = source_results(path, fixtures, now)
        observations.extend(source_observations)
        for fid, value in normalized.items():
            if fid in results:
                if any(results[fid][key] != value[key] for key in ("home_goals", "away_goals", "settlement")):
                    raise ValueError("Conflicting completed results require explicit review, no silent overwrite")
            else:
                results[fid] = value
    for fid, value in results.items():
        if fid not in fixtures or _time(value["available_at"], "available_at") > now:
            raise ValueError("Unknown fixture or future result availability")
    scores, periods = score_multi(payload["records"], results), weekly_scores(payload["records"], results)
    payload["result_observations"] = observations
    payload["scored_at"] = now.isoformat()
    destination = root / "outputs" / now.strftime("五线赛果评分_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "原赛前记录及新增赛果.json", payload)
    _write(destination / "同场评分与覆盖.json", scores)
    _write(destination / "逐周评分.json", periods)
    _write(destination / "执行信息.json", {"source_case": str(case_path), "evidence": list(map(str, evidence_paths)),
                                           "scored_at": now.isoformat(), "model_fits": 0, "llm_calls": 0,
                                           "parameter_changes": 0, "original_forecasts_changed": False})
    report_path = destination / "真实赛果评分说明.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report(payload, scores, periods))
    return {"output_directory": str(destination), "case": str(destination / "原赛前记录及新增赛果.json"),
            "report": str(report_path), "n_records": scores["n_records"], "n_settled": scores["n_settled"],
            "pending": len(scores["pending_fixture_ids"]), "all_five_settled": scores["all_five_common"]["n"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.case, args.evidence), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
