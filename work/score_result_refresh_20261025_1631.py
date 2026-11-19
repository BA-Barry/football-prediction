"""Score one new official snapshot against unchanged prior prospective records."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from football_model.batch_scoring import run
from diagnose_verified_old10_20261025 import individual, table

ROOT = REPOSITORY_ROOT
OBS = ROOT / "data/prematch-intelligence/20261024_batch/result_observation_20261024T163124853784Z"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write(path, obj):
    with path.open("x", encoding="utf-8") as target:
        json.dump(obj, target, ensure_ascii=False, indent=2)
        target.write("\n")

def main():
    snapshot_path = OBS / "result_observations.json"
    snapshot = load(snapshot_path)
    cases = {
        "original10": ROOT / "outputs/五线赛果评分_20261024T160739478761Z/原赛前记录及新增赛果.json",
        "player_revision9": ROOT / "outputs/五线赛果评分_20261024T160739530585Z/原赛前记录及新增赛果.json"
    }
    outputs, bundles = {}, {}
    for name, path in cases.items():
        before = load(path)
        ids = {r["fixture"]["match_id"] for r in before["records"]}
        source = deepcopy(snapshot)
        source["fixture_observations"] = [r for r in source["fixture_observations"] if r["match_id"] in ids]
        source["derived_from_full_snapshot"] = str(snapshot_path)
        source_path = OBS / f"{name}_result_observations.json"
        write(source_path, source)
        outputs[name] = run(ROOT, path, [source_path])
        after = load(Path(outputs[name]["case"]))
        assert before["records"] == after["records"]
        scores = load(Path(outputs[name]["output_directory"]) / "同场评分与覆盖.json")
        bundles[name] = {"case": after, "scores": scores, "own": individual(scores)}
    future_path = ROOT / "data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json"
    future = load(future_path)
    all_ids = {r["fixture"]["match_id"] for case in (bundles["original10"]["case"], future) for r in case["records"]}
    results = bundles["original10"]["case"]["results"]
    assert len(all_ids) == 20 and not future.get("results")
    lines = ["# 已保存20场：北京时间10月25日00:31后结果刷新", "",
             f"去重{len(all_ids)}场，确认结算{len(results)}场，待结算{len(all_ids)-len(results)}场。下一轮10场均未结算。",
             "原10与球员更新9分别评分；逐条比较证明原records均未改动。本轮模型拟合0次、参数变化0次、LLM调用0次。", "",
             "## 本次官方观察", "",
             f"FPL fixtures 实际接收：{snapshot['source_metadata']['fixtures']['received_at']}；HTTP Date={snapshot['source_metadata']['fixtures']['response_date']}，Age={snapshot['source_metadata']['fixtures']['age_seconds']}秒。",
             "| 比赛 | 观察比分 | started | finished | provisional | 本次动作 |", "|---|---|---|---|---|---|"]
    for row in snapshot["fixture_observations"]:
        if row["home"] not in ("Arsenal", "Coventry City", "Chelsea"):
            continue
        score = f"{row['home_goals']}–{row['away_goals']}" if row["home_goals"] is not None else "尚无"
        action = "已有官方赛后报道确认，保留原结算" if row["home"] == "Arsenal" else "继续待结算"
        lines.append(f"| {row['home']}–{row['away']} | {score} | {row['started']} | {row['finished']} | {row['finished_provisional']} | {action} |")
    lines += ["", "切尔西一场的FPL快照仍显示尚未开赛，且HTTP缓存已有225秒；不把该缓存字段解释为真实未开球，也绝不提前结算。",
              "富勒姆与考文垂官网各做一次有限读取：HTTP200但只有客户端页面壳，无可核验完赛正文。网页检索也未取得新的官方正文；非官方终场信号不纳入本次结算。没有改造FPL标记，也没有填造终场时刻。", "",
              "官方原始来源：[FPL fixtures](https://fantasy.premierleague.com/api/fixtures/)、[富勒姆官网尝试页面](https://www.fulhamfc.com/news/2026/october/24/coventry-0-1-fulham/)、[考文垂官网尝试页面](https://www.ccfc.co.uk/news/2026/october/24/report--coventry-city-0-1-fulham/)。后两者不作为完成证据。", "",
              "## 原10场：各线自身覆盖", ""]
    lines += table(bundles["original10"]["own"])
    lines += ["", "联合少一场有效赛前记录，不能将其1/2与其他线路1/3直接比较。全五线共同覆盖仍为维拉、阿森纳2场，指标与前次相同。", "",
              "## 原10场：全五线共同2场", ""]
    lines += table(bundles["original10"]["scores"]["all_five_common"]["metrics"], True)
    lines += ["", "## 球员更新9场：全五线共同2场", ""]
    lines += table(bundles["player_revision9"]["scores"]["all_five_common"]["metrics"], True)
    lines += ["", "本轮没有新增已确认赛果，因此没有新的失误归因或优化效果证据。小样本描述不作为稳定准确率；不根据单场或暂定比分调权重。", ""]
    destination = ROOT / "outputs" / datetime.now(timezone.utc).strftime("前瞻赛果有限刷新_%Y%m%dT%H%M%S%fZ")
    destination.mkdir()
    report = destination / "20场赛果刷新说明.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = {"unique_fixtures": len(all_ids), "confirmed_settled": len(results), "pending_unique": len(all_ids)-len(results),
               "new_confirmed_results": 0, "future_batch_settled": 0, "forecast_records_unchanged": True,
               "scoring_outputs": outputs, "source_observations": str(snapshot_path),
               "common_metrics": {k:v["scores"]["all_five_common"] for k,v in bundles.items()},
               "own_metrics": {k:v["own"] for k,v in bundles.items()}, "model_fits": 0, "parameter_changes": 0, "llm_calls": 0}
    write(destination / "20场赛果刷新数据.json", summary)
    write(OBS / "verified_scoring_outputs.json", outputs)
    print(json.dumps({"report": str(report), "summary": str(destination / "20场赛果刷新数据.json"),
                      "unique": len(all_ids), "confirmed": len(results), "pending": len(all_ids)-len(results),
                      "outputs": outputs}, ensure_ascii=False))

if __name__ == "__main__":
    main()
