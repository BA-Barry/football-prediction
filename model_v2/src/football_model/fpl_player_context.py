"""Saved FPL snapshot -> descriptive future-fixture player context, no forecast.

Only reads local observations. It neither fetches data, fits effects nor calls
an LLM. Source counters are not player contribution or expected playing time.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .batch_multiline import _fixture, _paths, _read, _write
from .multiline import build_packet
from .prospective import _player_context, _time


TEAM_NAMES = {"Bournemouth": "AFC Bournemouth", "Brighton": "Brighton & Hove Albion",
              "Leeds": "Leeds United", "Man City": "Manchester City",
              "Man Utd": "Manchester United", "Newcastle": "Newcastle United",
              "Nott'm Forest": "Nottingham Forest", "Spurs": "Tottenham Hotspur"}
LIMITS = [
    "news_added is API metadata, not injury onset or proof of when a paragraph first became public.",
    "Season minutes and starts are cumulative descriptive counters, not expected future minutes, causal effects, fatigue or replacement quality.",
    "The FPL status and percentages mentioned in raw news are source labels, not calibrated participation probabilities. No numeric chance field is supplied.",
    "A long-existing absence may already be reflected in team results; cumulative counters do not establish that all its effect is absorbed.",
    "Status u is unavailable/registration/transfer metadata, not an injury classification. A status a or absence of a flag is not proof of fitness.",
    "This is a previously saved cached snapshot, not a refreshed medical assessment or a confirmed lineup."
]


def _index(rows, key):
    output = {}
    for row in rows:
        ident = row[key]
        if ident in output:
            raise ValueError("Duplicate source identifier")
        output[ident] = row
    return output


def build_context(case, news, bootstrap, fixture_snapshot, metadata, observations, as_of):
    cutoff = _time(as_of, "as_of")
    for label in ("bootstrap", "fixtures"):
        source = metadata["sources"][label]
        if source["status"] != 200 or _time(source["received_at"], "source received_at") > cutoff:
            raise ValueError("Successful already-received snapshot required")
    received = _time(metadata["sources"]["bootstrap"]["received_at"], "bootstrap receipt")
    teams = _index(bootstrap["teams"], "id")
    players = _index(bootstrap["elements"], "id")
    live_fixtures = _index(fixture_snapshot, "id")
    observed_fixtures = _index(observations["fixture_observations"], "match_id")
    _index([row["fixture"] for row in case["records"]], "match_id")
    facts, fixtures, skipped, changes, summaries = {}, [], [], [], []
    for record in case["records"]:
        fixture = _fixture(record["fixture"])
        if fixture.kickoff <= cutoff:
            skipped.append(fixture.match_id)
            continue
        observed = observed_fixtures[fixture.match_id]
        raw = live_fixtures[observed["official_fixture_id"]]
        if (observed.get("verified") is not True or raw["started"] is not False
                or raw["finished"] is not False or raw.get("finished_provisional") is not False
                or _time(raw["kickoff_time"], "source kickoff") != fixture.kickoff
                or _time(observed["expected_kickoff"], "expected kickoff") != fixture.kickoff
                or observed["home"] != fixture.home or observed["away"] != fixture.away):
            raise ValueError("Future fixture identity/status differs from the saved case")
        old_facts = news["evidence_by_fixture"].get(fixture.match_id, [])
        selected = [deepcopy(row) for row in old_facts if row.get("source_team_id") is None]
        for side, team in (("h", fixture.home), ("a", fixture.away)):
            team_id = raw["team_" + side]
            if team_id != observed["home_team_id" if side == "h" else "away_team_id"]:
                raise ValueError("Fixture source team ID conflict")
            source_name = teams[team_id]["name"]
            if TEAM_NAMES.get(source_name, source_name) != team:
                raise ValueError("Explicit source team name does not match fixture")
            old = [r for r in old_facts if r.get("source_team_id") == team_id]
            if len(old) != 1 or old[0]["team"] != team:
                raise ValueError("One matching earlier team observation required")
            if _time(old[0]["observed_at"], "earlier observation") > received:
                raise ValueError("Earlier observation cannot be newer than this snapshot")
            source_players = []
            for p in players.values():
                if p["team"] != team_id or p["status"] == "a":
                    continue
                row = {"id": p["id"], "name": (p["first_name"] + " " + p["second_name"]).strip(),
                       "web_name": p["web_name"], "status": p["status"], "news": p["news"],
                       "news_added": p["news_added"], "source_team_id": team_id,
                       "position": p["element_type"]}
                for source_key, target_key in (("minutes", "season_minutes"), ("starts", "season_starts")):
                    if p.get(source_key) is not None:
                        row[target_key] = p[source_key]
                source_players.append(row)
            source_players.sort(key=lambda r: r["id"])
            _player_context(source_players, received)
            # Old intake kept status-u separately. Compare only its i/d/s cohort,
            # reporting new descriptive counters separately from source changes.
            def signature(rows):
                return {r["id"]: (r["status"], r.get("news"),
                        None if r.get("news_added") is None else _time(r["news_added"], "news metadata"))
                        for r in rows if r["status"] in {"i", "d", "s"}}
            before, after = signature(old[0].get("source_players", [])), signature(source_players)
            changed = sorted(i for i in set(before) | set(after) if before.get(i) != after.get(i))
            def serial(value):
                return None if value is None else [value[0], value[1], None if value[2] is None else value[2].isoformat()]
            changes.extend({"fixture_id": fixture.match_id, "team": team, "player_id": i,
                            "before": serial(before.get(i)), "after": serial(after.get(i))} for i in changed)
            selected.append({"id": f"fpl-player-context:{team_id}:{received.isoformat()}",
                "url": metadata["sources"]["bootstrap"]["source_url"], "published_at": None,
                "observed_at": received.isoformat(), "claim_type": "fact", "team": team,
                "category": "availability", "status": "official_source_assertion",
                "source_kind": "saved_official_fpl_json",
                "claim": f"Saved FPL availability metadata and season participation counters for {team}; source assertions only. See individual rows and limitations, not an estimated team-strength adjustment.",
                "retrieval_note": f"Already saved bootstrap HTTP receipt {received.isoformat()}, cache Age={metadata['sources']['bootstrap'].get('age_seconds')}; no new HTTP request or lineup confirmation.",
                "source_team_id": team_id, "source_players": source_players, "limits": LIMITS})
            summaries.append({"fixture_id": fixture.match_id, "team": team,
                              "flagged_players": sum(p["status"] in {"i", "d", "s"} for p in source_players),
                              "unavailable_other_players": sum(p["status"] == "u" for p in source_players),
                              "status_news_metadata_changed_ids": changed})
        facts[fixture.match_id] = selected
        fixtures.append(deepcopy(record["fixture"]))
    return {"recorded_at": cutoff.isoformat(), "evidence_by_fixture": facts,
            "unknowns_by_fixture": {f["match_id"]: [*news.get("unknowns_by_fixture", {}).get(f["match_id"], []), *LIMITS]
                                    for f in fixtures}}, {
                "as_of": cutoff.isoformat(), "future_fixtures": fixtures,
                "skipped_started_by_clock": skipped, "team_summaries": summaries,
                "changed_status_news_metadata": changes, "source_metadata": deepcopy(metadata),
                "scope": "Descriptive context enrichment of saved inputs, not new injury events or probability forecasts",
                "model_fits": 0, "llm_calls": 0, "new_http_requests": 0,
                "old_forecasts_changed": False, "limits": LIMITS}


def run(root, case_path, news_path, snapshot_directory):
    root, case_path, news_path, snapshot = _paths(root, case_path, news_path, snapshot_directory)
    case, news = _read(case_path), _read(news_path)
    metadata = _read(snapshot / "source_metadata.json")
    now = datetime.now(timezone.utc)
    enriched, audit = build_context(case, news, _read(snapshot / "bootstrap_raw.json"),
            _read(snapshot / "fixtures_raw.json"), metadata,
            _read(snapshot / "result_observations.json"), now)
    # Retain the existing raw-result descriptions and original statistical time,
    # but make genuinely new packets at NOW. No old response is attached to them.
    previous_packets = _read(case_path.parent / "independent_packets.json")["packets"]
    old_packets = {p["fixture"]["match_id"]: p for p in previous_packets}
    old_records = {r["fixture"]["match_id"]: r for r in case["records"]}
    packets = {"independent": [], "model_informed": []}
    for f in audit["future_fixtures"]:
        fixture, fid = _fixture(f), f["match_id"]
        old = old_packets[fid]
        if _fixture(old["fixture"]) != fixture:
            raise ValueError("Old descriptive packet identity differs")
        for mode in packets:
            packets[mode].append(build_packet(fixture, now, enriched["evidence_by_fixture"][fid],
                enriched["unknowns_by_fixture"][fid], descriptive_context=old["descriptive_context"],
                statistical=old_records[fid]["tracks"]["statistical"] if mode == "model_informed" else None,
                mode=mode))
    destination = root / "data/prematch-intelligence/20261024_batch" / now.strftime("player_context_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "news_with_player_context.json", enriched)
    _write(destination / "转换及输入审计.json", audit)
    for mode, rows in packets.items():
        _write(destination / f"{mode}_packets.json", {"prepared_at": now.isoformat(), "mode": mode,
                 "status": "prepared_only_no_new_llm_response_or_forecast", "packets": rows})
    lines = ["# 球员可用性与出场背景：真实输入修复", "",
             f"未来比赛{len(audit['future_fixtures'])}场；排除已到开球时点{len(audit['skipped_started_by_clock'])}场。",
             f"与原输入相比，i/d/s球员状态／新闻／元数据时间变化{len(audit['changed_status_news_metadata'])}人。没有把包生成时间较新当作新伤病。", "",
             "这次新增的是此前被投影丢失的逐人新闻和同来源出场背景；两种新资料包均实际生成，但尚未调用LLM，不是新预测。", "",
             "| 球队 | 球员 | 源状态 | 本季分钟 | 本季首发 | news_added源元数据 |", "|---|---|---|---:|---:|---|"]
    for fixture_facts in enriched["evidence_by_fixture"].values():
        for fact in fixture_facts:
            for p in fact.get("source_players", []):
                if p["status"] in {"i", "d", "s"}:
                    lines.append(f"| {fact['team']} | {p['name']} | {p['status']} | {p.get('season_minutes', '未知')} | {p.get('season_starts', '未知')} | {p.get('news_added')} |")
    lines += ["", "明确限制：", ""] + ["- " + s for s in LIMITS]
    (destination / "球员情报输入说明.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"output_directory": str(destination), "future_fixtures": len(audit["future_fixtures"]),
            "status_news_metadata_changed": len(audit["changed_status_news_metadata"]),
            "flagged_players": sum(r["flagged_players"] for r in audit["team_summaries"]),
            "llm_calls": 0, "model_fits": 0}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "case", "news", "snapshot"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.case, args.news, args.snapshot), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
