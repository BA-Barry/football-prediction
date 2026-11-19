"""Export existing five-line predictions and compare settled same-match records.

No news fetching, model fitting, LLM resampling or prediction timestamp refresh.
The first real case is a research demonstration, not evidence of superiority.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .multiline import TRACKS, score_multi


LABELS = {"statistical": "统计模型", "llm_independent": "独立LLM",
          "market": "市场比例去水", "llm_model_informed": "模型知情LLM",
          "joint_pool": "模型＋独立LLM算术组合"}


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def render_report(payload, scores):
    lines = ["# 五线赛前研究：模型、LLM、市场与联合候选", "",
             f"研究：{payload['pilot_id']}；真实比赛记录 {scores['n_records']} 场；"
             f"已结算 {scores['n_settled']} 场；五线共同且已结算 {scores['all_five_common']['n']} 场。", "",
             "一场比赛仍只有一个观察单位。五条线路共享数据且部分直接依赖，不能当作五份独立证据或多数票置信度。", ""]
    for record in payload["records"]:
        fixture = record["fixture"]
        lines += [f"## {fixture['home']} — {fixture['away']}", "",
                  f"开球UTC：{fixture['kickoff']}；记录截止UTC：{record['as_of']}。", "",
                  "| 线路 | 主胜 | 平局 | 客胜 | 主胜无缓冲公平赔率 |", "|---|---:|---:|---:|---:|"]
        for arm in TRACKS:
            track = record["tracks"].get(arm)
            if track is None:
                lines.append(f"| {LABELS[arm]} | 缺失 | 缺失 | 缺失 | — |")
                continue
            p = track["probabilities"]
            fair = "无有限值" if p[0] == 0 else f"{1/p[0]:.3f}"
            lines.append(f"| {LABELS[arm]} | {p[0]:.2%} | {p[1]:.2%} | {p[2]:.2%} | {fair} |")
        lines += ["", "公平赔率只是1/p换算，不含概率误差、价格变化、费用或安全余量，不是推荐买入价。"
                  "单项概率最高不代表超过50%，模型与报价有差异也不等于发现可盈利价值。", ""]
        for arm, reason in record["missing_reasons"].items():
            lines.append(f"- {LABELS[arm]}缺失／弃权：{reason}")
        a, b = record["tracks"].get("statistical"), record["tracks"].get("llm_model_informed")
        if a is not None and b is not None and a["probabilities"] == b["probabilities"]:
            lines += ["模型知情LLM本次保留原模型数字。这是一次未作数值调整的审阅，不是额外的独立预测证据，亦不是已证明联合增益。", ""]
        lines += ["### 各线实际时点", ""]
        for arm, track in record["tracks"].items():
            if track is not None:
                lines.append(f"- {LABELS[arm]}：{track.get('generated_at', track.get('observed_at'))}；"
                             f"{track.get('model', track.get('bookmaker', 'unknown'))}。")
    lines += ["## LLM原始一次性回答", "",
              "两条新LLM路线收到相同的描述性赛果和情报；独立线没有统计概率或市场信息，模型知情线另收到统计概率。"
              "使用隔离新上下文，不给另一LLM的回答；本轮是同平台两种输入策略，不是不同厂商模型的比较。精确运行时模型ID未知，不能声称已记录具体版本。", ""]
    for arm, response in payload.get("llm_responses", {}).items():
        lines += [f"### {LABELS[arm]}", "", f"决定：{response['decision']}。{response['reasoning_summary']}", "",
                  f"校准状态：{response['calibration_status']}", ""]
    policy = payload.get("policy", {})
    lines += ["## 组合规则与证据边界", "",
              f"显式研究权重：{policy.get('pool_weights')}。权重在两条新LLM回答前写入输入准备记录；"
              "50:50只是未训练的算术候选，不是最优权重，不替换正式默认。任何组件缺失时不自动改成另一组件100%。", "",
              "本入口首例（2026-10-25曼联—伯恩茅斯）的统计来源：真实拟合1次既有alpha=.01、180天衰减的球队模型，历史英超1,520场，加当时已知54场，合计1,574场。"
              "两队各自7轮已补齐，但第6—7轮其他16场仍缺，所以对手近期强度可能滞后；没有用缺失当0。"
              "这些数量仅描述该首例，不代表后续输入的训练样本数。该来源是新的日期级当前研究适配器，不是把原回顾研究器直接宣称为已完成赛前时点／首发校准的生产模型。", "",
              "统计线不读取LLM、赔率或伤停补丁。LLM可讨论伤停、战意、疲劳和轮换情景，但这些效应未做数值训练；"
              "没有假设它们影响为零，也没有给新闻人工加减固定百分点。球员在场实力和赛事优先级的定量接入仍未完成。", "",
              "本轮五线共同输出只包括1X2。统计线的完整进球分布仍单独保留；LLM三类胜平负概率不能唯一确定比分、总进球或半全场分布，不强行补造。", ""]
    observation = payload.get("market_observation")
    if observation:
        lines += ["## 本次市场来源", "",
                  f"[{observation['bookmaker']}公开事件页]({observation['source_url']})："
                  f"主/平/客十进制报价 {observation['decimal_prices']}；收到页面后记录UTC {observation['observed_at']}。", "",
                  "使用标准Full Time Result，不使用相邻2 UP提前派奖价格。仅单书商单次网页观察，"
                  "不知道真实开盘或盘口变化；工具/网页缓存、报价延迟及可成交性未知，不称市场共识或保证实时。", ""]
    lines += ["## 赛后如何比较", "",
              "保留全部10组预先列明的成对同场比较，以及全五线共同子集；每个比较显示实际比赛ID和n。"
              "各对可能覆盖不同比赛，不把不同子集的均值直接排成总冠军。主目标胜平负Log loss，另报三类求和Brier、平手规则明确的top1。"
              "弃权与覆盖另列，不把输出失败藏起来；一条线给实际发生结果0概率时保留无限损失而非裁剪。", "",
              "代码测试不是预测样本；小批流程观察不能据此判断哪条线稳定更好或联合可盈利。尚不训练融合权重、选择厂商或微调LLM。", ""]
    if scores["n_settled"] == 0:
        lines += ["当前没有已结束的真实目标比赛，所有比较n=0、准确性指标为空。", ""]
    else:
        lines += [f"输入中已结算{scores['n_settled']}场，待结算{len(scores['pending_fixture_ids'])}场；"
                  "仅以下共同子集允许同场比较，完整逐场结果及10组成对比较另存JSON。", "",
                  "| 五线共同子集 | n | Log loss | 三类求和Brier | top1 |",
                  "|---|---:|---:|---:|---:|"]
        for arm, metric in scores["all_five_common"]["metrics"].items():
            loss = ("∞" if metric["mean_log_loss_is_infinite"] else
                    "—" if metric["mean_log_loss"] is None else f"{metric['mean_log_loss']:.6f}")
            brier = "—" if metric["mean_brier"] is None else f"{metric['mean_brier']:.6f}"
            top1 = "—" if metric["top1_accuracy"] is None else f"{metric['top1_accuracy']:.2%}"
            lines.append(f"| {LABELS[arm]} | {metric['n']} | {loss} | {brier} | {top1} |")
        lines.append("")
    lines += ["本入口只是重放并导出已经保存的记录，不刷新原预测时点，不重新训练或重采LLM。全部新增文件、数据、模型、缓存和临时目录留在项目目录。", ""]
    return "\n".join(lines)


def run_multiline(root, input_path):
    root, input_path = Path(root).expanduser().resolve(), Path(input_path).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload.get("pilot_id"), str) or not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError("Supply an explicit pilot_id and nonempty records")
    scores = score_multi(payload["records"], payload.get("results", {}))
    output = root / "outputs" / f"五线前瞻比较_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    _dump(output / "五线赛前记录.json", payload["records"])
    _dump(output / "同场评分与覆盖.json", scores)
    _dump(output / "输入与原始LLM回答.json", payload)
    _dump(output / "导出信息.json", {"input_path": str(input_path), "exported_at": datetime.now(timezone.utc).isoformat(),
                                   "export_time_is_not_prediction_time": True, "training_runs": 0, "llm_calls": 0})
    report = output / "五线比较说明.md"
    report.write_text(render_report(payload, scores), encoding="utf-8")
    return {"output_directory": str(output), "report": str(report), "records": scores["n_records"],
            "settled": scores["n_settled"], "all_five_common_scored": scores["all_five_common"]["n"],
            "available": {arm: c["available"] for arm, c in scores["coverage"].items()},
            "training_runs_in_export": 0, "llm_calls_in_export": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    source = args.input or args.root / "data/prematch-intelligence/20261024_multiline/multiline_case.json"
    print(json.dumps(run_multiline(args.root, source), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
