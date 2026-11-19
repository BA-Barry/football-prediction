"""Assemble a new pre-kickoff information version, without fitting or I/O services.

Existing statistical and market observations are reused, not refreshed. A
fixture remains one observation across revisions; score cases separately.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .batch_multiline import _fixture, _paths, _read, _responses, _write
from .multiline import _line, arithmetic_pool, build_packet, multi_record, score_multi
from .prospective import _time


WEIGHTS = {"statistical": .5, "llm": .5}


def _packets(document, mode, old_records, now):
    prepared = _time(document["prepared_at"], "packet prepared_at")
    if prepared > now or document.get("mode") != mode:
        raise ValueError("Packet mode/time differs from its declared preparation")
    if not isinstance(document.get("packets"), list) or not document["packets"]:
        raise ValueError("A nonempty packet list is required")
    output = {}
    for packet in document["packets"]:
        fixture = _fixture(packet["fixture"])
        fid = fixture.match_id
        if fid not in old_records or fid in output:
            raise ValueError("Unknown or duplicate packet fixture")
        if fixture != _fixture(old_records[fid]["fixture"]):
            raise ValueError("Packet fixture identity differs from old case")
        if now >= fixture.kickoff:
            raise ValueError("Revision assembly must precede every selected kickoff")
        if _time(packet["as_of"], "packet as_of") != prepared or packet.get("mode") != mode:
            raise ValueError("Each packet must match its declared mode and prepared_at")
        statistical = old_records[fid]["tracks"]["statistical"]
        if mode == "independent":
            if "statistical" in packet or "market" in packet:
                raise ValueError("Independent packet cannot contain prediction lines")
        elif (packet.get("statistical") != _line(statistical, "statistical", fixture, prepared)
              or "market" in packet):
            raise ValueError("Informed packet must contain the unchanged old statistical line only")
        rebuilt = build_packet(fixture, prepared, packet["evidence"], packet["unknowns"],
            descriptive_context=packet["descriptive_context"],
            statistical=statistical if mode == "model_informed" else None, mode=mode)
        # Do not silently discard evidence dated after a supposedly prepared packet.
        if (len(rebuilt["evidence"]) != len(packet["evidence"])
                or len(rebuilt["descriptive_context"]) != len(packet["descriptive_context"])):
            raise ValueError("Packet contains unavailable or duplicate source rows")
        output[fid] = packet
    return prepared, output


def run(root, old_case_path, packet_directory, independent_response=None, informed_response=None):
    independent_response = independent_response or Path(packet_directory) / "independent_response.json"
    informed_response = informed_response or Path(packet_directory) / "model_informed_response.json"
    root, old_case_path, packet_directory, independent_response, informed_response = _paths(
        root, old_case_path, packet_directory, independent_response, informed_response)
    old = _read(old_case_path)
    # Validate old probability/time semantics and duplicate identities without
    # importing old outcomes into this genuinely new future-only case.
    score_multi(old["records"], {})
    old_records = {r["fixture"]["match_id"]: r for r in old["records"]}
    policy = deepcopy(old["policy"])
    if policy.get("pool_weights") != WEIGHTS:
        raise ValueError("This revision reuses the existing fixed 50:50 policy only")
    policy["data_refresh"] = ("Reuse original statistical forecasts and old market observations with their original times; "
                              "only the LLM responses and declared 50:50 pool are newly generated in this revision.")
    now = datetime.now(timezone.utc)
    documents, packets, times = {}, {}, {}
    for mode in ("independent", "model_informed"):
        documents[mode] = _read(packet_directory / f"{mode}_packets.json")
        times[mode], packets[mode] = _packets(documents[mode], mode, old_records, now)
    if times["independent"] != times["model_informed"] or set(packets["independent"]) != set(packets["model_informed"]):
        raise ValueError("Both packet strategies must describe the same preparation and fixtures")
    prepared = times["independent"]
    if _time(old["preparation"]["prepared_at"], "original preparation") > prepared:
        raise ValueError("Existing pool policy must precede the new information packets")
    for fid, independent in packets["independent"].items():
        informed = packets["model_informed"][fid]
        for key in ("fixture", "as_of", "evidence", "unknowns", "descriptive_context"):
            if independent[key] != informed[key]:
                raise ValueError("LLM strategies differ in more than statistical exposure")
    preparation = {"prepared_at": prepared.isoformat(), "policy": policy,
                   "fixture_ids": sorted(packets["independent"]),
                   "old_case": str(old_case_path), "packet_directory": str(packet_directory),
                   "policy_origin": "Existing 50:50 policy; unchanged before this revision's responses"}
    responses = {mode: _responses(path, mode, preparation, set(packets[mode]), now)
                 for mode, path in (("independent", independent_response), ("model_informed", informed_response))}
    records = []
    for fid in preparation["fixture_ids"]:
        previous = old_records[fid]
        fixture = _fixture(previous["fixture"])
        tracks = {arm: deepcopy(previous["tracks"][arm]) for arm in ("statistical", "market")}
        missing = {arm: previous.get("missing_reasons", {}).get(arm, "not_supplied")
                   for arm, track in tracks.items() if track is None}
        for mode, arm in (("independent", "llm_independent"), ("model_informed", "llm_model_informed")):
            response = responses[mode].get(fid)
            if response is None or response["probabilities"] is None:
                tracks[arm] = None
                missing[arm] = "not_supplied" if response is None else "abstained: " + response["reasoning_summary"]
            else:
                at = _time(response["generated_at_utc"], "response generated_at")
                if at <= prepared or at >= now or at >= fixture.kickoff:
                    raise ValueError("Numerical response must strictly follow packet and precede assembly/kickoff")
                tracks[arm] = {"id": f"{arm}:{fid}:{response['generated_at_utc']}", "fixture_id": fid,
                    "generated_at": response["generated_at_utc"], "probabilities": response["probabilities"],
                    "model": "Codex isolated source-context revision; reported model_id=" + response["model_identifier"],
                    "exposed_to_statistical": response["exposed_to_statistical"],
                    "exposed_to_market": response["exposed_to_market"], "description": response["calibration_status"]}
        # Abstention timestamps must also be genuine pre-kickoff responses.
        for mode in responses:
            response = responses[mode].get(fid)
            if response is not None:
                at = _time(response["generated_at_utc"], "response generated_at")
                if not prepared < at < now or at >= fixture.kickoff:
                    raise ValueError("Response must strictly follow packet and precede assembly/kickoff")
        tracks["joint_pool"], reason = arithmetic_pool(tracks["statistical"], tracks["llm_independent"], WEIGHTS, fixture, now)
        if reason:
            missing["joint_pool"] = reason
        record = multi_record(fixture, now, tracks=tracks, missing_reasons=missing)
        # Preserve the source representation as well as IDs, times and values.
        for arm in ("statistical", "market"):
            record["tracks"][arm] = deepcopy(previous["tracks"][arm])
        record["information_origin"] = "new_source_context_llm; statistical_and_old_market_reused"
        records.append(record)
    scores = score_multi(records, {})
    metadata = {"assembled_at": now.isoformat(), "prepared_at": prepared.isoformat(),
        "old_case": str(old_case_path), "packet_directory": str(packet_directory),
        "response_paths": {"independent": str(independent_response), "model_informed": str(informed_response)},
        "model_fits": 0, "parameter_changes": 0, "old_market_refreshed": False,
        "old_forecasts_changed": False, "unit": "one fixture, not one line or revision",
        "revision_scope": "Information-version sensitivity, not independent extra fixture observations",
        "llm_invocation_note": "This assembler makes no LLM calls; source response generation is recorded separately"}
    payload = {"pilot_id": "source-context-revision:" + now.isoformat(), "assembled_at": now.isoformat(),
        "records": records, "results": {}, "policy": policy, "preparation": preparation,
        "projection_path": old.get("projection_path"), "llm_responses": responses,
        "market_source": deepcopy(old.get("market_source", {})), "revision": metadata,
        "source_packets": documents,
        "notes": ["Reuse original statistical probabilities and cached market observation, not retraining or new odds.",
            "Compare this version separately against the old version on matching fixture IDs; never concatenate records across versions.",
            "No new total-goals, handicap or half/full-time probabilities are inferred from LLM 1X2."]}
    destination = root / "data/prematch-intelligence/20261024_batch" / now.strftime("revision_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "batch_case.json", payload)
    _write(destination / "initial_scores.json", scores)
    _write(destination / "执行信息.json", metadata)
    lines = ["# 尚未开球比赛：新增来源资料版多线前瞻", "",
        f"同一批未来比赛{len(records)}场；准备{prepared.isoformat()}；真实合并{now.isoformat()}。",
        "本版只新增LLM回答与原定50:50组合，统计线及市场原ID、时间和概率原样复用；不是赔率刷新或球队重训。",
        "两种LLM上下文共享同一批资料，仅知情线看到旧统计参考；暴露及模型身份是调用方声明，不是代码能证明的盲法认证。",
        "所有比赛均未带入旧赛果，评分目前为0场；尚无准确度结论。",
        "旧版仍保留。同一fixture的两个版本应各自评分、同场对齐比较，不把版本当作额外比赛样本。", "",
        "| 比赛 | 原统计 | 原独立LLM | 新独立LLM | 原知情LLM | 新知情LLM | 原组合 | 新50:50组合 |", "|---|---|---|---|---|---|---|---|"]
    for record in records:
        row = [record["fixture"]["home"] + "—" + record["fixture"]["away"]]
        prior = old_records[record["fixture"]["match_id"]]
        for source, arm in ((record, "statistical"), (prior, "llm_independent"),
                (record, "llm_independent"), (prior, "llm_model_informed"),
                (record, "llm_model_informed"), (prior, "joint_pool"), (record, "joint_pool")):
            track = source["tracks"][arm]
            row.append("缺失／弃权" if track is None else "/".join(f"{p*100:.1f}" for p in track["probabilities"]))
        lines.append("| " + " | ".join(row) + " |")
    report = destination / "前瞻版本说明.md"
    with report.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"run_directory": str(destination), "case": str(destination / "batch_case.json"),
            "report": str(report), "records": len(records), "settled": 0,
            "available": {arm: value["available"] for arm, value in scores["coverage"].items()}}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", "--old-case", dest="old_case", type=Path, required=True)
    parser.add_argument("--packet-dir", "--packet-directory", dest="packet_directory", type=Path, required=True)
    parser.add_argument("--independent-response", type=Path)
    parser.add_argument("--informed-response", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.old_case, args.packet_directory,
                         args.independent_response, args.informed_response), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
