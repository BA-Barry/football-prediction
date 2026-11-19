"""Describe scored prospective results without fitting or changing forecasts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from statistics import mean

ROOT = REPOSITORY_ROOT
OBS = ROOT / "data/prematch-intelligence/20261024_batch/result_observation_20261024T160147143875Z"
LABELS = {"statistical": "统计模型", "llm_independent": "独立LLM", "market": "市场去水", "llm_model_informed": "知情LLM", "joint_pool": "联合"}

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def individual(scores):
    output = {}
    for track in LABELS:
        rows = [item["scores"][track] for item in scores["per_fixture"] if track in item["scores"]]
        output[track] = {"n": len(rows), "top1_hits": sum(r["top1_accuracy"] for r in rows),
                         "mean_log_loss": mean(r["log_loss"] for r in rows), "mean_brier": mean(r["brier"] for r in rows)}
    return output

def table(metrics, common=False):
    lines = ["| 线路 | 已结算覆盖 | 最高概率命中 | Log loss | Brier（三类求和） |", "|---|---:|---:|---:|---:|"]
    for arm, value in metrics.items():
        n = value["n"]
        hits = value["top1_accuracy"] * n if common else value["top1_hits"]
        lines.append(f"| {LABELS[arm]} | {n} | {hits:g}/{n}（{hits/n:.1%}） | {value['mean_log_loss']:.6f} | {value['mean_brier']:.6f} |")
    return lines

def main():
    manifest = load(OBS / "verified_scoring_outputs.json")
    originals = {
        "original10": ROOT / "outputs/五线赛果评分_20261024T150438997728Z/原赛前记录及新增赛果.json",
        "player_revision9": ROOT / "outputs/五线赛果评分_20261024T150441350712Z/原赛前记录及新增赛果.json"
    }
    bundles = {}
    for name, info in manifest.items():
        directory = Path(info["output_directory"])
        case, scores = load(Path(info["case"])), load(directory / "同场评分与覆盖.json")
        assert case["records"] == load(originals[name])["records"]
        bundles[name] = {"case": case, "scores": scores, "own_metrics": individual(scores)}
    future_path = ROOT / "data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json"
    future = load(future_path)
    unique = {r["fixture"]["match_id"] for b in (bundles["original10"]["case"], future) for r in b["records"]}
    assert len(unique) == 20 and len(future["records"]) == 10 and not future.get("results")
    lines = ["# 前瞻结果与误差复核（北京时间2026-10-25）", "",
             "原10场新增确认阿森纳2–0埃弗顿：原10场现已结算3场、待7场；球员信息更新版9场已结算2场、待7场。",
             "加上下轮已存的10场，去重后仍为20场；其中3场已确认结算、17场待结算。更新版是同场赛前修订，不重复算新比赛。",
             "本次真实执行赛果评分；模型拟合0次、参数调整0次、LLM调用0次。逐条比较确认两版原赛前records均未改变。", "",
             "## 新赛果与证据", "",
             "阿森纳2–0埃弗顿由[埃弗顿官方赛后报道](https://www.evertonfc.com/news/2026/october/24/everton-defeated-at-arsenal/)确认。正文明确写0–2落败、赛后积分变化及下一场赛事。原始HTML、HTTP元数据及结构化证据均保存于本次观察目录。",
             "报道实际接收于2026-10-24T16:04:27.315860Z附近（精确值以HTTP元数据为准）；finished_by只使用实际接收时间上界，没有填造真实终场秒数。",
             "FPL快照接收于16:01:48.117921 UTC：阿森纳2–0仍为finished=false/provisional=false；考文垂0–1富勒姆为finished=false/provisional=true。字段原样保留。",
             "考文垂0–1已有比赛结束的非官方网页信号，但本次未取得符合现有结算格式的官方完赛正文；官网200响应只含客户端页面壳，不能把它当作完赛报道，故仍待结算。其余6场尚未开球。", "",
             "## 原10场：各线路自身覆盖", ""]
    lines += table(bundles["original10"]["own_metrics"])
    lines += ["", "联合缺少伊普斯维奇一场的有效赛前记录，因此1/2与其他线路1/3不能直接排名。以下同场比较仅维拉、阿森纳2场。", ""]
    lines += table(bundles["original10"]["scores"]["all_five_common"]["metrics"], True)
    lines += ["", "## 球员信息更新版9场：同样已结算2场", ""]
    lines += table(bundles["player_revision9"]["scores"]["all_five_common"]["metrics"], True)
    lines += ["", "## 误差解释实际改变了什么", "",
        "1. 新增阿森纳一场，所有可用线路的最高概率方向均命中。原统计、原独立LLM、市场、联合给主胜约62.23%、62.00%、66.24%、62.11%；这场市场的Log loss最低。但一场热门获胜不能证明市场长期校准更好。",
        "2. 更新版独立LLM把阿森纳主胜62%上调到67%，本场Log loss由0.478036降至0.400478；此前把维拉主胜22%下调到20%，维拉获胜使那场Log loss由1.514128增至1.609438。现在两场合看，更新版独立LLM的平均Log loss为1.004958，原版为0.996082：球员信息更新有得有失，尚无增量证据。",
        "3. 更新版联合的两场平均Log loss稍差（0.966502 vs 0.964390），Brier稍好（0.571498 vs 0.571921）。指标微小分歧加上样本仅2场，不支持选优、调融合权重或宣称球员信息有效。",
        "4. 伊普斯维奇1–1时，各线给平局约24%–28%，最高概率都没选平；维拉2–1时，各线偏向曼城。两场都是赛前有非零且不极小概率的结果，不能把一次未中直接归因为公式错误、伤停漏项或LLM失灵。现阶段只能记录实际结果得到的概率及失分，因果诊断要靠多轮重复偏差和分组实验。",
        "5. 知情LLM与统计线仍逐场给相同概率，因此没有新增概率信息；它们不能当两票独立支持。新增的下一轮双LLM路径是否有增量，要等该轮真实结果，不从本轮命中倒推。", "",
        "## 后续结算规则", "",
        "先补足明确官方完赛证据并继续结算已有20场，再按相同比赛覆盖看Log loss、Brier、热门/非热门与主客场偏差。此轮不因新增阿森纳命中调参，不因尚未确认的考文垂比分提前评分。", "",
        "这里只描述少量已完成的前瞻观察，不是稳定预测准确率、盈利能力或原因识别。", ""]
    # Render the real stored receipt rather than the explanatory draft timestamp.
    meta = load(OBS / "everton_completed_report_news_metadata.json")
    lines = [line.replace("2026-10-24T16:04:27.315860Z附近（精确值以HTTP元数据为准）", meta["received_at"]) for line in lines]
    report = ROOT / "outputs" / datetime.now(timezone.utc).strftime("前瞻结果复核_%Y%m%dT%H%M%S%fZ")
    report.mkdir()
    (report / "前瞻结果复核说明.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"recorded_at": datetime.now(timezone.utc).isoformat(), "unique_fixtures": len(unique),
        "confirmed_settled": 3, "pending_unique": 17, "future_batch_path": str(future_path),
        "future_batch_n": 10, "future_batch_settled": 0, "all_forecast_records_unchanged": True,
        "source_observation_directory": str(OBS), "scoring_outputs": manifest,
        "own_metrics": {name: b["own_metrics"] for name, b in bundles.items()},
        "common_metrics": {name: b["scores"]["all_five_common"] for name, b in bundles.items()},
        "model_fits": 0, "parameter_changes": 0, "llm_calls": 0}
    (report / "前瞻结果复核数据.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_directory": str(report), "unique_fixtures": 20, "confirmed_settled": 3, "pending": 17}, ensure_ascii=False))

if __name__ == "__main__":
    main()
