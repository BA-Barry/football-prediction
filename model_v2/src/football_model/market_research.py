"""Descriptive, same-quote margin sensitivity; no fitting or method selection.

The source seasons have already been examined. Results are exploratory and
conditional on archived quotes, not a new blind test, timed market backtest,
tradable-price claim, or uncertainty interval for a current match.
"""

import argparse
import csv
from datetime import date, datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .evaluation import calibration_bins
from .football_data import LEAGUES, MARKET_COLUMNS
from .forecast_diagnostics import validate_prediction
from .market_implied import IMPLIED_METHODS, dewater_1x2


SEASONS = ("2024-25", "2025-26")
NAMES = ("home", "draw", "away")


def _scores(probabilities, outcomes):
    """A zero probability for an observed outcome is infinite loss, not clipped."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    truth = np.eye(3)[y]
    observed = p[np.arange(len(y)), y]
    zeros = int(np.count_nonzero(observed == 0))
    return {
        "n": len(y),
        "log_loss": None if zeros else float(-np.log(observed).mean()),
        "log_loss_status": "positive_infinity" if zeros else "finite",
        "zero_observed_probability_count": zeros,
        "brier_sum": float(np.square(p-truth).sum(axis=1).mean()),
        "rps": float(np.square(np.cumsum(p, axis=1)[:, :2]-
                               np.cumsum(truth, axis=1)[:, :2]).sum(axis=1).mean()/2),
        "mean_probabilities": p.mean(axis=0).tolist(),
        "observed_frequencies": truth.mean(axis=0).tolist(),
        "calibration_5_bins": calibration_bins(p, y, bins=5),
    }


def analyse_quotes(records):
    """Keep every method and failure; compare only explicitly reported overlaps.

    Records must already be a resolved cohort of archived predictions. No price
    timestamps are generated. Missing markets and method failures remain visible;
    methods are never substituted and no successful subset is called full coverage.
    """
    if not records:
        raise ValueError("Need a nonempty, explicitly resolved match cohort")
    seen, rows, failures = set(), [], []
    for record in sorted(records, key=lambda row: row["match_id"]):
        mid = record["match_id"]
        if mid in seen:
            raise ValueError("Duplicate match in research cohort")
        seen.add(mid)
        y = record["actual_outcome"]
        if type(y) is not int or y not in (0, 1, 2):
            raise ValueError("Outcome must be 0=home, 1=draw, 2=away")
        for market in MARKET_COLUMNS:
            quote = record["market_prices"].get(market)
            row = {"match_id": mid, "date": record["date"],
                   "competition": record["competition"], "market": market,
                   "actual_outcome": y, "source_url": record.get("source_url"),
                   "quote_available": quote is not None,
                   "source_decimal_odds": None if quote is None else list(quote), "methods": {}}
            if quote is not None:
                for method in IMPLIED_METHODS:
                    try:
                        row["methods"][method] = dewater_1x2(quote, method)
                    except ValueError as error:
                        failures.append({"match_id": mid, "market": market,
                                         "method": method, "error": str(error)})
            rows.append(row)

    summaries = {}
    for market in MARKET_COLUMNS:
        pool = [r for r in rows if r["market"] == market]
        common = [r for r in pool if len(r["methods"]) == len(IMPLIED_METHODS)]
        common_ids = {r["match_id"] for r in common}
        available, paired = {}, {}
        for method in IMPLIED_METHODS:
            valid = [r for r in pool if method in r["methods"]]
            available[method] = None if not valid else _scores(
                [r["methods"][method]["probabilities"] for r in valid],
                [r["actual_outcome"] for r in valid])
            paired[method] = None if not common else _scores(
                [r["methods"][method]["probabilities"] for r in common],
                [r["actual_outcome"] for r in common])
        sensitivity = {}
        for method in IMPLIED_METHODS[1:]:
            if not common:
                sensitivity[method] = None
                continue
            old = np.asarray([r["methods"]["multiplicative"]["probabilities"] for r in common])
            new = np.asarray([r["methods"][method]["probabilities"] for r in common])
            diff = np.abs(new-old)
            a, b = paired[method], paired["multiplicative"]
            sensitivity[method] = {
                "log_loss_difference": None if a["log_loss"] is None or b["log_loss"] is None
                                       else a["log_loss"]-b["log_loss"],
                "brier_sum_difference": a["brier_sum"]-b["brier_sum"],
                "rps_difference": a["rps"]-b["rps"],
                "mean_absolute_probability_change": float(diff.mean()),
                "maximum_absolute_probability_change": float(diff.max()),
                "p95_match_max_absolute_probability_change": float(np.quantile(diff.max(axis=1), .95)),
                "matches_with_changed_top_outcome": int(np.count_nonzero(new.argmax(axis=1) != old.argmax(axis=1))),
            }
        quoted = [r for r in pool if r["quote_available"]]
        margins = [next(iter(r["methods"].values()))["overround"] for r in quoted if r["methods"]]
        summaries[market] = {
            "cohort_matches": len(pool), "quotes_available": len(quoted),
            "missing_quote_match_ids": [r["match_id"] for r in pool if not r["quote_available"]],
            "common_three_method_matches": len(common),
            "not_in_common_match_ids": [r["match_id"] for r in pool if r["match_id"] not in common_ids],
            "method_available_scores": available, "common_scores": paired,
            "difference_from_multiplicative_on_common": sensitivity,
            "overround": None if not margins else {
                "n": len(margins), "min": min(margins), "max": max(margins),
                "mean": float(np.mean(margins)), "median": float(np.median(margins)),
                "underround_quotes": sum(x < 0 for x in margins)},
        }
    return {"matches": len(records), "quote_method_failures": failures,
            "markets": summaries}, rows


def _csv_rows(rows):
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "methods"}
        flat["source_decimal_odds"] = json.dumps(row["source_decimal_odds"])
        for method in IMPLIED_METHODS:
            value = row["methods"].get(method)
            flat[f"{method}_available"] = value is not None
            for i, name in enumerate(NAMES):
                flat[f"{method}_p_{name}"] = None if value is None else value["probabilities"][i]
                flat[f"{method}_odds_{name}"] = None if value is None else value["odds"][i]
            flat[f"{method}_overround"] = None if value is None else value["overround"]
            flat[f"{method}_parameters"] = None if value is None else json.dumps(value["method_params"])
        yield flat


def load_season_records(parent: Path, season: str):
    """Read the declared season without accepting a misplaced valid trial.

    The expected retrospective interval is August 1 through next July 1
    (exclusive). Per-row identity/chronology checks alone do not establish that
    the row belongs to the directory's declared season or league.
    """
    if season not in SEASONS:
        raise ValueError("Unknown research season")
    start = date(int(season[:4]), 8, 1)
    end = date(start.year+1, 7, 1)
    records, sources, seen = [], [], set()
    for competition in LEAGUES:
        source = Path(parent)/season/f"{competition}_original.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("start") != start.isoformat() or data.get("end_exclusive") != end.isoformat():
            raise ValueError(f"Source trial interval disagrees with research season {season}: {source}")
        for row in data["predictions"]:
            validate_prediction(row)
            if row["competition"] != competition:
                raise ValueError("Source league disagrees with prediction")
            if not start <= date.fromisoformat(row["date"]) < end:
                raise ValueError(f"Prediction date outside declared research season {season}: {row['match_id']}")
            if row["match_id"] in seen:
                raise ValueError("Duplicate match in research season")
            seen.add(row["match_id"])
            records.append(row)
        sources.append(str(source.resolve()))
    return records, sources


def _log_loss_cell(score):
    """Distinguish an infinite computed loss from an absent common cohort."""
    if score is None:
        return "无共同报价"
    if score["log_loss_status"] == "positive_infinity":
        return "+∞（观测结果概率0）"
    return f"{score['log_loss']:.6f}"


def run_research(task_root: Path, parent: Path):
    task_root, parent = Path(task_root), Path(parent)
    started = datetime.now(timezone.utc)
    output = task_root/"outputs"/f"市场去水敏感性_{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": started.isoformat(), "parent_directory": str(parent.resolve()),
              "methods": list(IMPLIED_METHODS), "new_team_fits": 0,
              "implementation": {"package": "penaltyblog", "version": version("penaltyblog"),
                                 "api": "penaltyblog.implied.calculate_implied"},
              "default_market_method_changed": False, "seasons": {},
              "design": {
                  "unit": "one match, paired methods on the identical market quote",
                  "scope": "descriptive sensitivity on two already inspected historical seasons",
                  "missingness": "all failures and coverage retained; common-method cohort explicitly reported",
                  "selection": "no method selected, no fusion, no formal significance or ROI claims",
                  "time": "archive pre-closing/closing have unknown individual capture times; not openings or T-24h",
                  "uncertainty": "method differences are neither confidence intervals nor injury/motivation effects",
              }}
    report = ["# 市场去水方法：真实同场敏感性研究", "",
              "三法均实际调用已安装 penaltyblog；不新增基础训练，不替换默认比例法，不融合球队概率。",
              "两季此前已经查看，属于探索性描述，不是新盲测。单位为比赛，不把四组市场或三种方法当新增独立比赛。", "",
              "|赛季|市场|共同比赛|比例 LL|Power LL|Shin LL|", "|---|---|---:|---:|---:|---:|"]
    seen_matches = set()
    for season in SEASONS:
        records, sources = load_season_records(parent, season)
        ids = {row["match_id"] for row in records}
        if seen_matches & ids:
            raise ValueError("Duplicate match across research seasons")
        seen_matches.update(ids)
        summary, rows = analyse_quotes(records)
        summary["source_files"] = sources
        result["seasons"][season] = summary
        flat = list(_csv_rows(rows))
        with (output/f"{season}_逐报价三法.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        for market, detail in summary["markets"].items():
            values = [detail["common_scores"][m] for m in IMPLIED_METHODS]
            cells = [_log_loss_cell(v) for v in values]
            report.append(f"|{season}|{market}|{detail['common_three_method_matches']}|"+"|".join(cells)+"|")
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["output_directory"] = str(output.resolve())
    (output/"去水敏感性结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report += ["", "LL越低越好，但本轮不据已查看赛季挑选赢家；完整Brier sum、RPS、五箱校准、概率变化和失败记录见JSON及CSV。",
               "缺报价和方法求根失败逐项保留，分别报告各方法可用范围与三法共同范围，不悄悄缩样本或换回比例法。",
               "pre-closing不等于开盘，closing也不是相同时点的信息；Avg是聚合均价，未必可成交。没有盘口历史或临场伤停数据由此生成。",
               "方法之间的差异只反映分配水差的假设；不能称为真实概率、置信区间、预测准确率提升或投注盈利。",
               "实验设计技能限定三法、同场同报价；统计分析技能要求保留全部结果与缺失。正式方法/融合仍保持不变。", "",
               "来源：[penaltyblog官方API](https://penaltyblog.readthedocs.io/en/latest/implied/implied.html)。",
               "全部新增代码、CSV、结果和缓存位于项目目录，无新依赖、hash、冻结contract或gate。"]
    destination = output/"市场去水研究说明.md"
    destination.write_text("\n".join(report)+"\n", encoding="utf-8")
    return result, destination


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=root/"outputs"/"赛程日期特征研究_20261023T102135288039Z")
    args = parser.parse_args()
    result, report = run_research(root, args.parent)
    print(json.dumps({"report": str(report), "matches": {s: d["matches"] for s, d in result["seasons"].items()},
                      "failures": sum(len(d["quote_method_failures"]) for d in result["seasons"].values()),
                      "new_team_fits": 0}, ensure_ascii=False))
