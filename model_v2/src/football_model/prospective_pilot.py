"""Export sourced news and separate prediction records; no LLM API or training.

Input is a user-reviewed JSON file. Re-running exports the recorded timestamps,
not a newly observed forecast. Files are ordinary research artifacts below the
explicitly supplied project root.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .domain import Fixture
from .prospective import ARMS, build_llm_packet, prediction_record, score_records


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def render_report(payload, packets, records, scores):
    lines = ["# LLM赛前情报：前瞻可行性样本", "",
             "这是接入和弃权行为的真实试跑，不是已完成的三轨准确率验证。没有训练、调参、自动融合或下注。", "",
             f"试验：{payload['pilot_id']}；实际记录 {len(records)} 场；已结算 {scores['n_settled']} 场；"
             f"三轨完整且已结算的同场比较 {scores['paired_n']} 场。", "",
             "一个比赛是一个样本，不因三条预测而算成三个样本。缺失不补零，LLM弃权不重提示直到出现概率。", ""]
    for case, packet, record in zip(payload["cases"], packets, records):
        fixture = record["fixture"]
        lines += [f"## {fixture['home']} — {fixture['away']}", "",
                  f"开球UTC：{fixture['kickoff']}；记录信息截止UTC：{record['as_of']}。", "",
                  "### 三轨记录", ""]
        for arm in ARMS:
            track = record["tracks"][arm]
            if track is None:
                lines.append(f"- {arm}：无概率。{record['missing_reasons'][arm]}")
            else:
                lines.append(f"- {arm}：主/平/客 {track['probabilities']}；"
                             f"时点 {track.get('generated_at', track.get('observed_at'))}。")
        response = case.get("llm_response")
        if response:
            lines += ["", f"本次LLM决定：{response['decision']}；{response['reasoning_summary']}", "",
                      "本次输出原样保存在输入样本中；不是多次采样后挑选的一份。"
                      "隔离上下文只排除了本项目其他轨概率，不能证明预训练知识完全无污染。"]
            if response.get("key_scenarios"):
                lines += ["", "### LLM提取的待检验情景", ""]
                lines += [f"- {item}" for item in response["key_scenarios"]]
        lines += ["", "### 截止时可用的来源陈述／推断", ""]
        for fact in packet["evidence"]:
            publication = fact["published_at"] or "未知（不以接收时点替代）"
            lines += [f"- [{fact['id']}] {fact['claim']}（{fact['claim_type']}；{fact['status']}）。"
                      f"[来源]({fact['url']})；收到 {fact['observed_at']}；精确发布时间：{publication}。"]
            if fact.get("retrieval_note"):
                lines.append(f"  访问限制：{fact['retrieval_note']}")
        lines += ["", "### 尚缺信息", ""]
        lines += [f"- {item}" for item in packet["unknowns"]]
        lines += ["", *case.get("notes", []), ""]
    lines += ["## 比较口径与下一步", "",
              "主目标是同一场90分钟（含补时，不含加时/点球）的胜平负Log loss；"
              "另报三类求和Brier与最高概率命中率。仅在同场三轨皆完整且已有赛果时比较；"
              "另报所有场的覆盖与弃权，不能靠只报成功输出的比赛证明LLM更好。", "",
              "当前paired_n=0时，所有准确性均值必须为空，不能把普通测试或样本导出计为有效性证据。"
              "未来完整样本仍需报告弃权造成的选择性，以及相同数据截止不等于相同信息覆盖。", "",
              "这1场是方便取到官网情报的流程样本，不是随机样本，也不是经功效计算的比较试验。"
              "补齐当前赛果/球队强度和完整1X2报价后，再选连续10场符合覆盖条件的比赛进行小批前瞻试跑，"
              "赛前记下名单及预测时点，不按赛果/看好方向选择；10场只能检查可行性，不能证明稳定盈利。", "",
              "新闻更新先改变可用阵容与情景描述；只有有历史验证的映射，才可改变球队进球率。"
              "本轮没有给伤停、战意、恢复或Kroupi回归添加手写概率百分点或假定情景权重。", "",
              "本入口不抓取新闻、不调用付费接口、不安装大模型、不刷新旧预测的生成时间；"
              "它导出经人工核对的真实情报样本，并支持之后同格式的预测与赛果记录。", ""]
    return "\n".join(lines)


def run_pilot(root, input_path):
    root, input_path = Path(root).expanduser().resolve(), Path(input_path).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload.get("pilot_id"), str) or not payload["pilot_id"].strip():
        raise ValueError("An explicit pilot_id is required")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("Supply at least one reviewed case")
    packets, records = [], []
    for case in payload["cases"]:
        data = dict(case["fixture"])
        data["kickoff"] = datetime.fromisoformat(data["kickoff"])
        fixture = Fixture(**data)
        packets.append(build_llm_packet(fixture, case["as_of"], case["evidence"], case["unknowns"]))
        records.append(prediction_record(fixture, case["as_of"], **case["tracks"],
                                         missing_reasons=case.get("missing_reasons")))
        response = case.get("llm_response")
        if response and response.get("probabilities") is None and records[-1]["tracks"]["llm"] is not None:
            raise ValueError("An abstained LLM response cannot be recorded as an available prediction")
    scores = score_records(records, payload.get("results", {}))
    exported_at = datetime.now(timezone.utc)
    output = root / "outputs" / f"LLM前瞻情报_{exported_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    _dump(output / "LLM新闻输入包.json", packets)
    _dump(output / "三轨赛前记录.json", records)
    _dump(output / "同场比较与覆盖.json", scores)
    _dump(output / "本次导出.json", {"input_path": str(input_path), "exported_at": exported_at.isoformat(),
                                   "timestamp_meaning": "export only, never a fresh forecast or source observation",
                                   "llm_api_calls_in_export": 0, "training_runs": 0})
    report = output / "前瞻样本说明.md"
    report.write_text(render_report(payload, packets, records, scores), encoding="utf-8")
    return {"output_directory": str(output), "report": str(report), "n_records": len(records),
            "paired_n": scores["paired_n"], "trained": False, "llm_api_calls_in_export": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    source = args.input or args.root / "data" / "prematch-intelligence" / "20261024_可行性样本.json"
    print(json.dumps(run_pilot(args.root, source), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
