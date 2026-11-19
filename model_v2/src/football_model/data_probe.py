"""Fetch five small CC0-declared result files, inspect fields, never fit a model."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .openfootball import inspect_payload


SEASON = "2024-25"
COMPETITIONS = {"en.1": "英超", "de.1": "德甲", "es.1": "西甲", "it.1": "意甲", "fr.1": "法甲"}
BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"


def _read(url):
    request = Request(url, headers={"User-Agent": "football-research-data-inspection/0.3"})
    with urlopen(request, timeout=30) as response:
        return response.read(), dict(response.headers)


def run_probe(task_root: Path):
    observed = datetime.now(timezone.utc)
    run_name = observed.strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = task_root / "data" / "openfootball" / SEASON / run_name
    raw_dir.mkdir(parents=True, exist_ok=False)
    output_dir = task_root / "outputs" / "model-v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    license_bytes, _ = _read(f"{BASE}/LICENSE.md")
    (raw_dir / "LICENSE.md").write_bytes(license_bytes)

    def fetch(item):
        code, label = item
        url = f"{BASE}/{SEASON}/{code}.json"
        body, headers = _read(url)
        completed = datetime.now(timezone.utc)
        (raw_dir / f"{code}.json").write_bytes(body)
        inspection = inspect_payload(json.loads(body.decode("utf-8-sig")), competition=code,
                                     season=SEASON, source_url=url, observed_at=completed)
        inspection["label"] = label
        inspection["http_last_modified_not_match_publication_time"] = headers.get("Last-Modified")
        return inspection

    with ThreadPoolExecutor(max_workers=3) as pool:
        inspections = list(pool.map(fetch, COMPETITIONS.items()))
    result = {
        "status": "REAL DATA FIELD INSPECTION ONLY — NO MODEL TRAINING",
        "started_at": observed.isoformat(), "raw_directory": str(raw_dir),
        "source_license": "Repository declares CC0-1.0; per-record original provenance not fully verified",
        "excluded_source": "Football-Data.co.uk not downloaded because current automated/AI training restrictions require separate permission",
        "competitions": inspections,
    }
    (raw_dir / "inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {**result, "competitions": [{k: v for k, v in row.items() if k != "records"} for row in inspections]}
    (output_dir / "真实赛果样本_字段检查.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 真实赛果样本：字段与可用性检查", "",
             "仅下载并检查 openfootball 的 2024/25 五大联赛赛果；没有拟合模型，没有生成实战预测。", "",
             "| 联赛 | 原始记录 | 可解析 | FT | 缺FT | HT | 时间文本 | 队伍数 | 格式异常 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in inspections:
        lines.append(f"| {row['label']} | {row['input_rows']} | {row['parsed_rows']} | {row['full_time_rows']} | {row['missing_full_time_rows']} | {row['half_time_rows']} | {row['clock_text_rows']} | {row['teams']} | {len(row['issues'])} |")
    lines += ["", f"合计 {sum(r['input_rows'] for r in inspections):,} 条赛程、{sum(r['full_time_rows'] for r in inspections):,} 个完整 FT。格式异常为零不代表完整；缺失比分单独列示。", ""]
    for row in inspections:
        if row["missing_full_time_rows"]:
            lines.append(f"{row['label']}缺失 FT 的轮次：{json.dumps(row['missing_full_time_rounds'], ensure_ascii=False)}。不能据此计算完整赛季最终积分或完整赛季表现。")
    lines += ["", f"原始数据、声明许可及逐行解析保存在 `{raw_dir}`。", "",
              "所有时间文本均未在本轮核实其时区；UTC 开球时间、历史首次可用时间、赔率、首发和伤停字段均未重建。没有把缺失比分当成 0，也没有把观察时间当成历史首次公布时间。", "",
              "这些数据可用于进一步检查赛果层的覆盖，但不等于完整的赛前24小时／首发后训练集。国内联赛对手网络不连通，不能只靠这些数据推导欧战跨联赛强弱。", "",
              "来源：[openfootball 项目](https://github.com/openfootball/football.json)、[声明许可 CC0-1.0](https://github.com/openfootball/football.json/blob/master/LICENSE.md)。项目由 Football.TXT 上游生成；本轮未验证逐场原始来源，不代表完整法律审核。", "",
              "Football-Data 原候选来源未下载：其[当前说明](https://football-data.co.uk/data.php)限制自动化／AI 数据训练用途。不会通过其镜像绕过。", "",
              "下步需要核实开球时间语义、建立明确的球队别名映射，并补足有授权且带历史时间的阵容／赔率来源；真实训练尚未启动。"]
    destination = output_dir / "真实赛果样本_检查报告.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, destination


if __name__ == "__main__":
    summary, destination = run_probe(Path(__file__).resolve().parents[3])
    print(json.dumps({"report": str(destination), "raw_directory": summary["raw_directory"],
                      "parsed_rows": sum(r["parsed_rows"] for r in summary["competitions"]),
                      "issues": sum(len(r["issues"]) for r in summary["competitions"]),
                      "trained": False}, ensure_ascii=False))
