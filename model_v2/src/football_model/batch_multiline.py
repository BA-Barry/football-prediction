"""Prepare isolated LLM packets, then assemble saved one-shot batch responses.

No network, LLM service, model fitting or parameter search occurs here. Inputs
are actual saved observations; first-case research records remain untouched.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path

from .domain import Fixture, OddsSnapshot
from .multiline import TRACKS, arithmetic_pool, build_packet, multi_record, score_multi
from .probability import implied
from .prospective import _probabilities, _text, _time


LABELS = {"statistical": "统计", "llm_independent": "独立LLM", "market": "市场去水",
          "llm_model_informed": "模型知情LLM", "joint_pool": "50:50组合"}
POLICY = {
    "pool_weights": {"statistical": .5, "llm": .5},
    "pool_weight_status": "explicit_untrained_research; declared before both new batch responses",
    "data_refresh": "Update known results/availability and refit team coefficients with the existing settings; do not overwrite earlier forecasts.",
    "hyperparameters": "Keep alpha=.01/half_life180 for this batch; no tuning using one-match wins/losses.",
    "review": "Review all settled fixtures by round/calendar week and cumulatively. Look for repeated probability-loss and calibration patterns across multiple rounds, not an isolated result.",
    "change_validation": "Separate data/implementation repairs from modelling changes. Explain the hypothesised failure, change one dimension, retain the existing line, and evaluate the candidate on later unseen fixtures.",
    "evidence_limit": "No fixed small count proves superiority; exploratory repeated reviews are not confirmatory significance tests. No automatic parameter switch or efficacy claim.",
    "unit": "one fixture, not one line or one saved forecast version",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _paths(*values):
    paths = tuple(Path(value).resolve() for value in values)
    if os.name == "nt" and any(path.drive.upper() != "G:" for path in paths):
        raise ValueError("All batch project inputs/outputs must stay on G:")
    return paths


def _fixture(row):
    return Fixture(**(row | {"kickoff": _time(row["kickoff"], "kickoff")}))


def _compact(description):
    """Raw outcome summaries, not fitted strength or direct win probabilities."""
    result = {key: value for key, value in description.items() if key != "results"}
    games = description["results"]
    if description["played"] == 0:
        result["history_note"] = "No records for this team in this EPL season input; not zero ability or a statement about other competitions."
    else:
        result["last_five_results"] = games[-5:]
        for name, selected in (("home", [r for r in games if r["home"] == description["team"]]),
                               ("away", [r for r in games if r["away"] == description["team"]])):
            goals = [r["score"] if r["home"] == description["team"] else list(reversed(r["score"]))
                     for r in selected]
            result[name + "_descriptive"] = {"played": len(goals), "goals_for": sum(r[0] for r in goals),
                                               "goals_against": sum(r[1] for r in goals)}
    return result


def prepare(root, projection_path, news_path):
    root, projection_path, news_path = _paths(root, projection_path, news_path)
    projection, news = _read(projection_path), _read(news_path)
    cutoff = datetime.now(timezone.utc)
    if _time(projection["generated_at"], "projection generated_at") > cutoff:
        raise ValueError("Future statistical projection")
    independent, informed, ids = [], [], set()
    for case in projection["fixtures"]:
        fixture = _fixture(case["fixture"])
        if fixture.match_id in ids:
            raise ValueError("Duplicate fixture in projection")
        ids.add(fixture.match_id)
        context = [{"id": f"raw-results:{i}", "observed_at": projection["generated_at"],
                    "text": json.dumps(_compact(row), ensure_ascii=False),
                    "source": "Source-linked EPL results only; descriptive aggregates, not statistical-model probabilities"}
                   for i, row in enumerate(case["descriptive_team_results"])]
        unknowns = ["Official FPL flags are availability metadata, not confirmed lineups, minutes or calibrated participation probabilities.",
                    "Full confirmed first XI, substitution plans, true fatigue, tactics and competition priorities are unknown unless explicitly sourced.",
                    "Player-specific effect sizes, cross-competition strength and probability calibration are not supplied.",
                    "Source caches can lag. Missing reports do not establish fitness; no source flag does not mean full squad fitness."]
        unknowns += news.get("unknowns_by_fixture", {}).get(fixture.match_id, [])
        facts = news.get("evidence_by_fixture", {}).get(fixture.match_id, [])
        independent.append(build_packet(fixture, cutoff, facts, unknowns, descriptive_context=context))
        if case["track"] is not None:
            informed.append(build_packet(fixture, cutoff, facts, unknowns, descriptive_context=context,
                                         statistical=case["track"], mode="model_informed"))
    destination = root / "data/prematch-intelligence/20261024_batch" / cutoff.strftime("run_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    preparation = {"prepared_at": cutoff.isoformat(), "projection_path": str(projection_path),
                   "news_path": str(news_path), "policy": POLICY, "fixture_ids": sorted(ids)}
    _write(destination / "preparation.json", preparation)
    for name, packets in (("independent", independent), ("model_informed", informed)):
        _write(destination / f"{name}_packets.json", {"prepared_at": cutoff.isoformat(), "mode": name,
                                                     "packets": packets})
    return {"run_directory": str(destination), "independent_packets": len(independent),
            "model_informed_packets": len(informed), "prepared_at": cutoff.isoformat()}


def _responses(path, mode, preparation, fixture_ids, cutoff):
    rows = _read(path)["forecasts"]
    if not isinstance(rows, list):
        raise ValueError("Saved response forecasts must be a list")
    output = {}
    for row in rows:
        fid = _text(row.get("fixture_id"), "response fixture_id")
        if fid not in fixture_ids or fid in output:
            raise ValueError("Unknown or duplicate response fixture_id")
        at = _time(row.get("generated_at_utc"), "response generated_at")
        if not _time(preparation["prepared_at"], "packet prepared_at") <= at <= cutoff:
            raise ValueError("Response must follow its packet and precede assembly")
        if row.get("exposed_to_statistical") is not (mode == "model_informed") or row.get("exposed_to_market") is not False:
            raise ValueError("Contradictory LLM exposure declaration")
        _text(row.get("reasoning_summary"), "response reasoning")
        _text(row.get("decision"), "response decision")
        _text(row.get("calibration_status"), "response calibration_status")
        _text(row.get("model_identifier"), "response model_identifier")
        abstain = row.get("decision") == "abstain"
        if abstain != (row.get("probabilities") is None):
            raise ValueError("An abstention must have null probabilities; numerical forecasts must not claim abstention")
        if not abstain:
            _probabilities(row["probabilities"])
        output[fid] = row
    return output


def assemble(root, run_dir, market_path):
    root, run_dir, market_path = _paths(root, run_dir, market_path)
    preparation = _read(run_dir / "preparation.json")
    projection_path, = _paths(preparation["projection_path"])
    projection = _read(projection_path)
    market_source = _read(market_path)
    cutoff = datetime.now(timezone.utc)
    ids = set(preparation["fixture_ids"])
    if ids != {case["fixture"]["match_id"] for case in projection["fixtures"]}:
        raise ValueError("Projection fixtures differ from prepared batch")
    packets = {mode: _read(run_dir / f"{mode}_packets.json")["packets"]
               for mode in ("independent", "model_informed")}
    prepared_fixtures = {packet["fixture"]["match_id"]: packet["fixture"] for packet in packets["independent"]}
    for case in projection["fixtures"]:
        fid = case["fixture"]["match_id"]
        if fid not in prepared_fixtures or _fixture(prepared_fixtures[fid]) != _fixture(case["fixture"]):
            raise ValueError("Projection fixture identity differs from its prepared input packet")
    responses = {mode: _responses(run_dir / f"{mode}_response.json", mode, preparation,
                                 {packet["fixture"]["match_id"] for packet in packets[mode]}, cutoff)
                 for mode in ("independent", "model_informed")}
    quotes = {}
    for quote in market_source["market_observations"]:
        fid = quote["fixture_id"]
        if fid not in ids or fid in quotes:
            raise ValueError("Unknown or duplicate market fixture in batch observation")
        if quote.get("market") != "regulation_full_time_one_x_two":
            raise ValueError("Use standard regulation 1X2 prices")
        quotes[fid] = quote
    records = []
    for case in projection["fixtures"]:
        fixture = _fixture(case["fixture"])
        fid = fixture.match_id
        archival = cutoff >= fixture.kickoff
        packet_at = _time(preparation["prepared_at"], "packet prepared_at")
        if packet_at >= fixture.kickoff:
            raise ValueError("Original packet must precede kickoff, including for archival assembly")
        tracks, missing = {"statistical": case["track"]}, dict(case["missing_reasons"])
        if tracks["statistical"] is not None and _time(tracks["statistical"]["generated_at"], "statistical generated_at") >= fixture.kickoff:
            tracks["statistical"] = None
            missing["statistical"] = "not_prematch: statistical forecast generated at or after kickoff"
        for mode, arm in (("independent", "llm_independent"), ("model_informed", "llm_model_informed")):
            row = responses[mode].get(fid)
            if row is not None and _time(row["generated_at_utc"], "response generated_at") >= fixture.kickoff:
                tracks[arm] = None
                missing[arm] = "not_prematch: response generated at or after kickoff; raw response retained only for audit"
            elif row is None or row["probabilities"] is None:
                tracks[arm] = None
                missing[arm] = ("not_supplied" if row is None else "abstained: " + row["reasoning_summary"])
            else:
                tracks[arm] = {"id": f"{arm}:{fid}:{row['generated_at_utc']}", "fixture_id": fid,
                               "generated_at": row["generated_at_utc"], "probabilities": row["probabilities"],
                               "model": "Codex isolated batch agent; reported model_id=" + row["model_identifier"],
                               "exposed_to_statistical": row["exposed_to_statistical"],
                               "exposed_to_market": row["exposed_to_market"], "description": row["calibration_status"]}
        quote = quotes.get(fid)
        if quote is None:
            tracks["market"] = None
            missing["market"] = "Complete standard1X2 quote unavailable; no imputation"
        elif _time(quote["observed_at"], "quote observed_at") >= fixture.kickoff:
            tracks["market"] = None
            missing["market"] = "not_prematch: market observation at or after kickoff; raw quote retained only for audit"
        else:
            snapshot = OddsSnapshot(**quote["decimal_prices"], observed_at=_time(quote["observed_at"], "quote observed_at"),
                                    bookmaker=quote["bookmaker"], fixture_id=fid)
            tracks["market"] = {"id": f"market:{fid}:{quote['observed_at']}", "fixture_id": fid,
                                "observed_at": quote["observed_at"], "probabilities": list(map(float, implied(snapshot))),
                                "bookmaker": quote["bookmaker"], "source": quote["source"],
                                "description": "Single captured standard1X2 price; source event ID can be unknown, no inferred opening/history or executable quote claim"}
        if archival:
            # as_of describes the latest original information, NOT the actual
            # assembly time. Do not create a new prematch pool after kickoff.
            record_cutoff = max([packet_at] + [
                _time(track["observed_at"] if arm == "market" else track["generated_at"], "original track time")
                for arm, track in tracks.items() if track is not None])
            tracks["joint_pool"] = None
            reason = "not_prematch: assembly at or after kickoff; no new joint pool generated"
        else:
            record_cutoff = cutoff
            tracks["joint_pool"], reason = arithmetic_pool(tracks["statistical"], tracks["llm_independent"],
                                                           preparation["policy"]["pool_weights"], fixture, cutoff)
        if reason:
            missing["joint_pool"] = reason
        record = multi_record(fixture, record_cutoff, tracks=tracks, missing_reasons=missing)
        record["information_origin"] = ("archived_existing_prematch_tracks" if archival else "assembled_before_kickoff")
        records.append(record)
    # Existing scorer validates the assembled identity, time and probability semantics.
    scores = score_multi(records, {})
    payload = {"pilot_id": "five-line-batch:" + preparation["prepared_at"], "assembled_at": cutoff.isoformat(),
               "records": records, "results": {},
               "policy": preparation["policy"], "projection_path": preparation["projection_path"],
               "preparation": preparation, "llm_responses": responses, "market_source": market_source,
               "notes": [f"{len(records)} explicitly declared fixtures, future at packet/forecast time; no outcome-based selection.",
                         "Earlier records and revisions are retained separately; multiple versions of a fixture do not increase the unique fixture count.",
                         "Two isolated same-platform batch responses, not independent model vendors; exact execution context is retained in the saved response metadata.",
                         "Assembly time is separate from original track times. Elapsed fixtures retain only existing prematch tracks; as_of is their latest original information time, not a backdated assembly. No new pool for an elapsed fixture.",
                         "Listing-page quotes with unknown bookmaker event IDs use existing OddsSnapshot/implied only. No fake source event ID, changes to existing history safeguards, or market movement claim.",
                         f"No new hyperparameter search; refreshed team coefficients use{projection['training_matches']} earlier-date matches. Availability evidence is not numerically fitted."]}
    _write(run_dir / "batch_case.json", payload)
    _write(run_dir / "initial_scores.json", scores)
    return {"case": str(run_dir / "batch_case.json"), "records": scores["n_records"], "settled": scores["n_settled"],
            "available": {arm: values["available"] for arm, values in scores["coverage"].items()}}


def render_report(payload, projection, scores):
    archived = sum(record.get("information_origin") == "archived_existing_prematch_tracks" for record in payload["records"])
    market_source = payload["market_source"]
    cached_market = any(str(row.get("source_kind", "")).startswith("third_party_")
                        for row in market_source["market_observations"])
    market_label = "市场缓存去水" if cached_market else "市场去水"
    lines = [f"# {scores['n_records']}场真实五线前瞻批量比较", "",
             f"真实比赛{scores['n_records']}场，已结算{scores['n_settled']}场；不是五倍比赛样本。",
             f"实际合并时间：{payload.get('assembled_at', '未记录')}；其中{archived}场在合并时已到开球时点，只归档原赛前线路，不新生成联合概率。",
             f"球队数据更新后共享拟合{projection['fit_count']}次、参数搜索{projection['parameter_searches']}次；"
             f"训练{projection['training_matches']}场，其中本季{projection['current_season_matches']}场，源中更早日期结果仍缺{projection['current_missing_reported_results']}场。", "",
             "下表均为主／平／客百分比；不是精确真概率、置信区间或投注建议。所有行仍为研究预测，球员效应及赛事重视程度尚未数值拟合。", "",
             *( ["市场列包含第三方署名缓存报价，仅作描述性对照。记录取得时间不是博彩公司更新时间；未核实最新可成交价格，也未取得开盘变化。数值齐全不等于来源与时效已核实。", ""] if cached_market else []),
             f"| 北京时间 | 比赛 | 统计 | 独立LLM | {market_label} | 模型知情LLM | 50:50组合 |",
             "|---|---|---|---|---|---|---|"]
    for record in payload["records"]:
        fixture = record["fixture"]
        bj = _time(fixture["kickoff"], "kickoff").astimezone(timezone(timedelta(hours=8)))
        row = [bj.strftime("%m-%d %H:%M"), fixture["home"] + "—" + fixture["away"]]
        for arm in TRACKS:
            track = record["tracks"][arm]
            row.append("缺失／弃权" if track is None else "/".join(f"{p*100:.1f}" for p in track["probabilities"]))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 统计线完整进球输出", "",
              "以下仅来自统计比分分布，不代表LLM已预测了总进球；预期进球不是比分保证。", "",
              "| 比赛 | 主／客预期进球 | 总进球0/1/2/3/4+ | 大于2.5球 |",
              "|---|---|---|---:|"]
    for case in projection["fixtures"]:
        if case["forecast"] is not None:
            forecast, rates = case["forecast"], case["rates"]
            totals = "/".join(f"{forecast['total_goals'][k]*100:.1f}" for k in ("0", "1", "2", "3", "4+"))
            lines.append(f"| {case['fixture']['home']}—{case['fixture']['away']} | {rates['home']:.3f}/{rates['away']:.3f} | {totals} | {forecast['over_2_5']:.1%} |")
    lines += ["", "## LLM逐场理由与不确定性", ""]
    for record in payload["records"]:
        fid = record["fixture"]["match_id"]
        lines += [f"### {record['fixture']['home']}—{record['fixture']['away']}", ""]
        for mode, label in (("independent", "独立LLM"), ("model_informed", "模型知情LLM")):
            response = payload["llm_responses"].get(mode, {}).get(fid)
            if response is not None:
                lines += [f"{label}：{response['reasoning_summary']}", "",
                          f"生成：{response['generated_at_utc']}；决定：{response['decision']}；"
                          f"模型ID：{response['model_identifier']}。{response['calibration_status']}", ""]
        for arm, reason in record["missing_reasons"].items():
            lines += [f"{LABELS[arm]}缺失／弃权：{reason}", ""]
    lines += ["", "## 实际执行与信息边界", "",
              "统计线只更新结果数据，alpha=.01和180天衰减不变。来源缺失结果补齐后，球队拟合系数和预测数字可以变化；这不是根据预测输赢重新选参数。原单例预测保留，不覆盖、不把同场两个版本算两场。", "",
              "独立LLM读取原始表现摘要和有来源情报，看不到统计／市场数字；知情LLM多看到统计数字，但没有另一LLM回答或市场数字。"
              "本轮同平台两种上下文策略，模型精确ID若未知照实记录；各批次一次回答，不反复采样追求数字。未经校准的LLM判断不等于已训练球员影响。", "",
              "官方FPL状态含伤停／待定／停赛标签及缓存年龄，不是正式首发、预计分钟或真实出场概率。没有标签不代表无人受伤，不能把FPL chance字段直接当权重。"
              "休息天数和欧战赛程只是资料，赛事投入／疲劳方向和大小未确认；缺失因子并非已证明零影响。", "",
              "市场为单书商网页／索引快照，可能缓存或无法成交。部分列表页没有书商event ID，保留未知，用既有OddsSnapshot/implied比例去水；"
              "不伪造ID以通过历史接口、不放松原有保护，不拼造开盘变化。缺失报价保留缺失，冲突报价保留来源差异，不用看似有利的一组替换首次完整观察。", "",
              "50:50权重在这两批LLM回答前已写入准备记录，仅未训练候选，不是最优权重或默认升级。保留全部比赛的成对同场比较、全五线共同子集和各线缺失／弃权。",
              "已开赛行只整理原先保存、原时间严格早于开球的预测／报价；record.as_of是这些既有资料与原数据包的最晚时间，不是合并时间。实际assembled_at另记。任何开球后生成的回答／报价不能充当赛前线路，也不会把赛后新算的组合回填到过去。", "",
              "五线共同部分只比较1X2；统计完整比分网格、总进球0/1/2/3/4+、大小2.5及竞彩客让两球单独保留，不能从LLM三类概率补造这些玩法或半全场。", "",
              "## 趋势驱动的调整方针", "",
              "- 每次先更新真实数据和球队系数；数据错误、错队或时间错误及时修复，说明受影响记录，不重写原赛前预测。",
              "- 汇总已结算同场Log loss、三类求和Brier、覆盖及弃权，按轮次／日历周和累计查看；单场是否命中不直接触发超参数或融合权重改变。",
              "- 多轮反复出现同向概率偏差、损失恶化或明确数据分布变化时，先说明具体失败类型，再研究一个有解释的调整；没有机械最小场数就自动批准的门槛。",
              "- 用来发现或选择调整的比赛，不再宣称为该调整的独立验证；新候选与既有线路在更晚真实比赛上同场比较，重大结构／深度学习再与用户讨论。",
              "- 这是探索性滚动研究，不因反复查看找到一次有利区间就宣称显著或盈利。单轮小批不能代表大趋势。", "",
              "## 当前评分与产物", "",
              f"五线完整且已结算{scores['all_five_common']['n']}场；完整成对评分、逐场损失和覆盖另存JSON。"
              "代码测试和合成赛果不算真实预测成绩。", ""]
    if scores["n_settled"] == 0:
        lines += ["本轮尚未录入可结算的真实赛果，没有准确率、校准效果或优越性结论；未录入不表示比赛一定尚未结束。", ""]
    for observation in payload["market_source"]["market_observations"]:
        lines.append(f"- [{observation['fixture_id']}]({observation['source_url']})：{observation['decimal_prices']}，记录{observation['observed_at']}；开盘未知。")
    lines += ["", "导出不触发下载、球队拟合或LLM调用，不刷新原预测时间。所有新增工程文件、数据、模型及缓存／临时目录均在G盘。", ""]
    return "\n".join(lines)


def export(root, case_path):
    root, case_path = _paths(root, case_path)
    payload = _read(case_path)
    projection_path, = _paths(payload["projection_path"])
    projection = _read(projection_path)
    scores = score_multi(payload["records"], payload.get("results", {}))
    destination = root / "outputs" / datetime.now(timezone.utc).strftime("批量五线前瞻_%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "五线赛前记录及原始回答.json", payload)
    _write(destination / "同场评分与覆盖.json", scores)
    _write(destination / "球队完整比分分布与来源.json", projection)
    _write(destination / "导出信息.json", {"source": str(case_path), "exported_at": datetime.now(timezone.utc).isoformat(),
                                           "fit_count": 0, "llm_calls": 0, "prediction_times_refreshed": False})
    report = destination / "批量五线比较说明.md"
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_report(payload, projection, scores))
    return {"output_directory": str(destination), "report": str(report), "records": scores["n_records"],
            "settled": scores["n_settled"], "available": {arm: c["available"] for arm, c in scores["coverage"].items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "assemble", "export"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--case", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.projection is None or args.news is None:
            parser.error("prepare requires --projection and --news")
        result = prepare(args.root, args.projection, args.news)
    elif args.action == "assemble":
        if args.run_dir is None or args.market is None:
            parser.error("assemble requires --run-dir and --market")
        result = assemble(args.root, args.run_dir, args.market)
    else:
        if args.case is None:
            parser.error("export requires --case")
        result = export(args.root, args.case)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
