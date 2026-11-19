"""Independent season refits of a complete-match, date-level team model.

This uses official FT scores, not player-event exposure, and never infers a
whistle or substitution second. Monday fits use strictly earlier dates; market
quotes are retrospective comparators with unknown individual observation times.
No hyperparameter search, market blending, player transfer or betting returns.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from .date_model import DateTeamGoalModel
from .evaluation import calibration_bins, probability_scores
from .football_data import parse_row
from .probability import score_grid
from .real_backtest import de_vig


MARKET = "b365_preclosing"
TOTAL_LABELS = tuple(map(str, range(7))) + ("7+",)
SEASON_SPECS = (
    ("2015/16", "data/football-data/2015-16/20261022_full_player/E0.csv", date(2016, 2, 1)),
    ("2017/18", "data/wyscout/premier-league-2017-18/Football-Data_E0_2017-18.csv", date(2018, 2, 1)),
)


def monday(day):
    if type(day) is not date:
        raise ValueError("Use a calendar date, not a fabricated timestamp")
    return day - timedelta(days=day.weekday())


def load_season_csv(path, season):
    """Read the existing independent CSV; do not download or fill other odds."""
    path = Path(path).resolve()
    season_code = season[2:4] + season[-2:]
    source_url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    records, issues, rejected, duplicates, seen = [], [], [], [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(columns):
            raise ValueError(f"{path.name}: missing required FT result columns")
        for line, row in enumerate(reader, 2):
            if not any(value and str(value).strip() for value in row.values()):
                continue
            try:
                match, row_issues = parse_row(row, competition="E0", season=season,
                    source_path=str(path), source_url=source_url)
            except (ValueError, TypeError) as error:
                rejected.append({"line": line, "reason": str(error)})
                continue
            # Same named comparator in both seasons; no synthetic Avg columns.
            match.markets = {MARKET: match.markets[MARKET]} if MARKET in match.markets else {}
            issues.extend(i for i in row_issues if i["field"] in {MARKET, "half_time"})
            if match.match_id in seen:
                duplicates.append(match.match_id)
            seen.add(match.match_id)
            records.append(match)
    records.sort(key=lambda r: (r.match_date, r.match_id))
    report = {"season": season, "source_path": str(path), "source_url": source_url,
        "records": len(records), "teams": len({t for r in records for t in (r.home, r.away)}),
        "date_min": min((r.match_date for r in records), default=None),
        "date_max": max((r.match_date for r in records), default=None),
        "full_time_goals": sum(sum(r.full_time) for r in records),
        "market_columns_present": [c for c in ("B365H", "B365D", "B365A") if c in columns],
        "market_rows": sum(MARKET in r.markets for r in records),
        "field_issues": issues, "rejected_rows": rejected, "duplicate_keys": sorted(set(duplicates)),
        "time_semantics": {"result_granularity": "source_calendar_date_only",
            "earlier_results_available": "assumed_available_at_next_calendar_date",
            "individual_odds_observed_at": None,
            "market_scope": "historical pre-closing, not verified opening or T-24h quotes"}}
    for key in ("date_min", "date_max"):
        if report[key] is not None:
            report[key] = report[key].isoformat()
    return records, report


def full_time_forecast(home_rate, away_rate):
    grid = score_grid(home_rate, away_rate, rho=0)
    total = grid.total_goals_distribution()
    h, a = np.indices(grid.grid.shape)
    return {"one_x_two": dict(zip(("home", "draw", "away"), map(float, grid.home_draw_away))),
        "expected_goals": {"home": float((grid.grid * h).sum()),
                           "away": float((grid.grid * a).sum())},
        "total_goals": {str(k): float(total[k]) for k in range(7)} | {"7+": float(total[7:].sum())}}


def weekly_walk_forward(records, *, evaluation_start):
    """One independent season; never update inside a Monday-to-Monday block."""
    first_cutoff = monday(evaluation_start)
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate source match IDs")
    if len({(r.season, r.competition) for r in records}) > 1:
        raise ValueError("Refit seasons independently; do not pool or transfer coefficients")
    predictions, skipped, fits, weeks = [], [], [], []
    ordered = sorted(records, key=lambda r: (r.match_date, r.match_id))
    if not ordered:
        return {"predictions": predictions, "skipped": skipped, "fits": fits, "weeks": weeks}
    cutoff = first_cutoff
    while cutoff <= ordered[-1].match_date:
        end = cutoff + timedelta(days=7)
        targets = [r for r in ordered if r.match_date >= evaluation_start and cutoff <= r.match_date < end]
        earlier = [r for r in ordered if r.match_date < cutoff and r.goal_fit_exclusion is None]
        block = {"week_start": cutoff.isoformat(), "week_end_exclusive": end.isoformat(),
                 "target_rows": len(targets), "eligible_training_rows": len(earlier)}
        eligible_targets = []
        for match in targets:
            if match.goal_fit_exclusion:
                skipped.append({"match_id": match.match_id, "season": match.season,
                    "date": match.match_date.isoformat(), "reason": match.goal_fit_exclusion,
                    "fit_before_date": cutoff.isoformat()})
            else:
                eligible_targets.append(match)
        if not eligible_targets:
            block["status"] = "no_eligible_targets_no_fit"
        elif not earlier:
            block["status"] = "no_earlier_history_no_fit"
            skipped.extend({"match_id": r.match_id, "season": r.season,
                "date": r.match_date.isoformat(), "fit_before_date": cutoff.isoformat(),
                "reason": "no_earlier_history"} for r in eligible_targets)
        else:
            model = DateTeamGoalModel().fit(earlier, before_date=cutoff)
            fits.append({"season": ordered[0].season, "week_start": cutoff.isoformat(),
                         "week_end_exclusive": end.isoformat(), **model.parameters()})
            block["status"] = "fitted_fixed_parameters"
            for match in eligible_targets:
                unknown = sorted({match.home, match.away} - model.teams)
                if unknown:
                    skipped.append({"match_id": match.match_id, "season": match.season,
                        "date": match.match_date.isoformat(), "fit_before_date": cutoff.isoformat(),
                        "reason": "unseen_team_no_estimated_prior", "unknown_teams": unknown})
                    continue
                rates = model.predict(match.home, match.away, match.competition, match.match_date)
                forecast = full_time_forecast(*rates)
                h, a = match.full_time
                label = 0 if h > a else 2 if h < a else 1
                market_prices = {MARKET: list(match.markets[MARKET])} if MARKET in match.markets else {}
                market_p = {MARKET: list(map(float, de_vig(match.markets[MARKET])))} if market_prices else {}
                predictions.append({"season": match.season, "match_id": match.match_id,
                    "competition": match.competition, "date": match.match_date.isoformat(),
                    "home": match.home_label, "away": match.away_label,
                    "home_team_id": match.home, "away_team_id": match.away,
                    "fit_before_date": cutoff.isoformat(), "training_matches": model.n_matches,
                    "training_max_date": model.training_max_date.isoformat(),
                    "actual_score": [h, a], "actual_outcome": label,
                    "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
                    "market_prices": market_prices, "market_probabilities": market_p,
                    "market_quote_observed_at": None, "source_url": match.source_url})
        weeks.append(block)
        cutoff = end
    return {"predictions": predictions, "skipped": skipped, "fits": fits, "weeks": weeks}


def _total_metrics(predictions):
    p = np.asarray([list(r["forecast"]["total_goals"].values()) for r in predictions])
    y = np.asarray([min(sum(r["actual_score"]), 7) for r in predictions])
    truth = np.eye(8)[y]
    calibration = []
    for c, label in enumerate(TOTAL_LABELS):
        assigned = np.minimum((p[:, c] * 5).astype(int), 4)
        for b in range(5):
            mask = assigned == b
            if mask.any():
                calibration.append({"total_goals": label, "bin": b, "n": int(mask.sum()),
                    "mean_probability": float(p[mask, c].mean()),
                    "observed_frequency": float((y[mask] == c).mean())})
    return {"n": len(predictions), "class_order": list(TOTAL_LABELS),
        "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny)).mean()),
        "brier_sum": float(np.square(p - truth).sum(axis=1).mean()),
        "rps": float((np.square(np.cumsum(p, axis=1)[:, :7] -
                                 np.cumsum(truth, axis=1)[:, :7]).sum(axis=1) / 7).mean()),
        "calibration": calibration}


def evaluate(predictions):
    """All-model diagnostics are separate from same-match market comparisons."""
    if not predictions:
        return {"model_all_supported": None, "total_goals": None, "market_comparisons": {}}
    labels = [r["actual_outcome"] for r in predictions]
    probabilities = [r["probabilities"] for r in predictions]
    result = {"model_all_supported": probability_scores(probabilities, labels),
        "model_all_calibration": calibration_bins(probabilities, labels, bins=5),
        "total_goals": _total_metrics(predictions), "market_comparisons": {}}
    paired = [r for r in predictions if MARKET in r["market_probabilities"]]
    if paired:
        y = [r["actual_outcome"] for r in paired]
        p = [r["probabilities"] for r in paired]
        q = [r["market_probabilities"][MARKET] for r in paired]
        result["market_comparisons"][MARKET] = {"n_matches": len(paired),
            "match_ids": [r["match_id"] for r in paired],
            "comparison_basis": "identical_match_ids_and_outcomes_only",
            "model_on_identical_matches": probability_scores(p, y), "market": probability_scores(q, y),
            "model_calibration": calibration_bins(p, y, bins=5),
            "market_calibration": calibration_bins(q, y, bins=5),
            "quote_timing": "individual historical observation times unknown; no T-24h claim"}
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path, predictions):
    fields = ["season", "match_id", "date", "home", "away", "fit_before_date", "training_matches",
        "actual_home_goals", "actual_away_goals", "p_home", "p_draw", "p_away",
        "expected_home_goals", "expected_away_goals"]
    fields += [f"p_goals_{i}" for i in range(7)] + ["p_goals_7_plus"]
    fields += [f"{MARKET}_{s}" for s in ("home", "draw", "away")]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            row = {k: item[k] for k in ("season", "match_id", "date", "home", "away",
                                        "fit_before_date", "training_matches")}
            row.update(dict(zip(("actual_home_goals", "actual_away_goals"), item["actual_score"])))
            row.update(dict(zip(("p_home", "p_draw", "p_away"), item["probabilities"])))
            row.update({f"expected_{s}_goals": item["forecast"]["expected_goals"][s] for s in ("home", "away")})
            row.update({f"p_goals_{i}": item["forecast"]["total_goals"][str(i)] for i in range(7)})
            row["p_goals_7_plus"] = item["forecast"]["total_goals"]["7+"]
            if MARKET in item["market_probabilities"]:
                row.update({f"{MARKET}_{s}": item["market_probabilities"][MARKET][i]
                            for i, s in enumerate(("home", "draw", "away"))})
            writer.writerow(row)


def write_report(path, result):
    """Describe the fixed experiment; do not select a model from its results."""
    def number(value):
        return "未支持" if value is None else f"{value:.6f}"

    lines = ["# 两赛季球队层完整比分验证", "",
        "这是一轮历史球队方法复验，不是当前比赛预测，不是球员模型验证，也不是跨赛季球员系数迁移。",
        "两个赛季分别从零拟合，目标为包含补时的官方全场比分；不需要假定真实终场秒数或换人秒数。", "",
        "## 固定方法与时间口径", "",
        "使用同一 DateTeamGoalModel：正则强度0.01、时间半衰期180天、rho=0；没有调参、"
        "市场融合、球员修正或按本轮结果自动择优。",
        "每个有合格目标比赛的日历周一扩窗重训，仅纳入严格早于该周一日期的赛果。"
        "同周已结束比赛不会用于同周后续预测，但可进入下个周一的训练。",
        "2015/16从2016-02-01起计分；2017/18从2018-02-01起计分，后者首个训练截止日是2018-01-29。"
        "这不是精确赛前24小时信息集：较早日期赛果次日可知只是实验假设，历史首次公开时间未知。", "",
        "## 数据与实际覆盖", "",
        "| 赛季 | 全场比分行数 | 目标场数 | 模型支持 | 与B365共同支持 | 实际拟合周数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for season, value in result["seasons"].items():
        cover, metrics = value["coverage"], value["metrics"]
        paired = metrics["market_comparisons"].get(MARKET, {})
        lines.append(f"| {season} | {result['sources'][season]['records']} | {cover['target_matches']} | "
            f"{cover['predicted_matches']} | {paired.get('n_matches', 0)} | {cover['fitted_weeks']} |")
        if cover["skipped_reasons"]:
            lines.extend(["", f"{season}明确跳过：" + "；".join(
                f"{reason}：{count}场" for reason, count in cover["skipped_reasons"].items()) + "。", ""])
    lines += ["", "未知球队不使用臆造先验，跳过原因与逐场记录保留在完整结果中。", "",
        "## 同一批比赛的胜平负对照", "",
        "这里只比较双方共同支持的同一批比赛；不同场数的模型全支持诊断不混入市场优劣比较。",
        "B365去水概率为三项倒数赔率除以倒数之和，仅作独立对照，不与模型融合。"
        "来源是历史pre-closing报价，不等于已核实开盘或赛前24小时报价，单条观测时间未知。", "",
        "| 赛季 | 共同场数 | 模型Log loss | B365 Log loss | 模型Brier | B365 Brier | 模型RPS | B365 RPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for season, value in result["seasons"].items():
        pair = value["metrics"]["market_comparisons"].get(MARKET, {})
        model, market = pair.get("model_on_identical_matches", {}), pair.get("market", {})
        lines.append(f"| {season} | {pair.get('n_matches', 0)} | " + " | ".join(
            number(scores.get(metric)) for metric in ("log_loss", "brier_sum", "rps")
            for scores in (model, market)) + " |")
    lines += ["", "Log loss、Brier和RPS均为越低越好；三类Brier为平方误差之和，RPS除以2。"
        "本报告仅列实测值，不据这些结果选择参数或宣称存在稳定优势。", "",
        "## 模型总进球分布", "",
        "与胜平负来自同一比分分布，分为0、1、2、3、4、5、6和7+八档。"
        "下表是模型全部支持比赛的总进球诊断，不与仅有胜平负赔率的市场强行比较。", "",
        "| 赛季 | 支持场数 | 总进球Log loss | 总进球Brier | 总进球RPS |",
        "|---|---:|---:|---:|---:|"]
    for season, value in result["seasons"].items():
        total = value["metrics"]["total_goals"] or {}
        lines.append(f"| {season} | {total.get('n', 0)} | " + " | ".join(
            number(total.get(metric)) for metric in ("log_loss", "brier_sum", "rps")) + " |")
    lines += ["", "总进球RPS除以7；完整结果保留胜平负及总进球的五箱描述性校准。"
        "分箱不是重新拟合的概率校准，也不是参数置信区间。", "",
        "## 解释边界与文件", "",
        "数据已经做过全季质量查看，因此本轮是按时间隔离拟合的回顾性验证，不是从未查看过的盲测。"
        "赛季之间的差异同时受到球队构成、赛程和年份变化影响，不能归因于球员系数迁移。"
        "本轮没有收益率、投注组合或中奖保证。", "",
        "逐场球队验证.csv及同名JSON保存每场1X2、总进球八档、市场概率与训练截止日；"
        "逐周完整参数.json保存每次拟合的全部可读系数和训练比赛ID；"
        "球队验证完整结果.json保存指标、校准、跳过记录与周块；数据接入与排除.json保存来源与读取问题。", "",
        "## 比分来源", ""]
    for season, source in result["sources"].items():
        lines.append(f"{season}：[Football-Data比分及历史报价]({source['source_url']})，"
            f"本地输入{source['records']}行，日期{source['date_min']}至{source['date_max']}。")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root, output_dir=None):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "outputs" / (
        "第二赛季球队验证_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    sources, seasons, fits, predictions = {}, {}, [], []
    for season, relative_path, start in SEASON_SPECS:
        records, source = load_season_csv(root / relative_path, season)
        if source["rejected_rows"] or source["duplicate_keys"]:
            raise ValueError(f"{season}: invalid FT rows or duplicate match IDs: "
                             f"{source['rejected_rows']} / {source['duplicate_keys']}")
        print(f"{season}: {len(records)} FT results; fixed Monday walk-forward from {start}", flush=True)
        experiment = weekly_walk_forward(records, evaluation_start=start)
        rows = experiment["predictions"]
        sources[season] = source
        seasons[season] = {"evaluation_start": start.isoformat(),
            "first_monday_cutoff": monday(start).isoformat(),
            "coverage": {"target_matches": sum(r.match_date >= start for r in records),
                "predicted_matches": len(rows), "fitted_weeks": len(experiment["fits"]),
                "skipped_reasons": dict(Counter(r["reason"] for r in experiment["skipped"]))},
            "metrics": evaluate(rows), "skipped": experiment["skipped"], "weeks": experiment["weeks"]}
        fits.extend(experiment["fits"])
        predictions.extend(rows)
        print(f"{season}: {len(rows)} supported predictions / {len(experiment['fits'])} Monday fits", flush=True)
    protocol = {"model": "DateTeamGoalModel", "regularization": 0.01, "half_life_days": 180,
        "rho": 0, "fit_schedule": "calendar Monday, only weeks with eligible targets",
        "training_information": "strictly earlier calendar date FT results only",
        "weekly_update": "expanding within each season; earlier evaluation results enter only later Monday fits",
        "market": MARKET, "de_vig": "inverse decimal prices divided by their sum",
        "market_quote_observed_at": None, "market_blending": False, "hyperparameter_search": False,
        "player_validation": False, "player_coefficient_transfer": False,
        "full_time_target": "complete official FT scores, including stoppage time; no event clock exposure",
        "probability_class_order": ["home", "draw", "away"], "total_goals_classes": list(TOTAL_LABELS),
        "score_definitions": "lower is better; Brier is class-square-error sum; RPS divides by K-1",
        "calibration": "five descriptive probability bins, not fitted probability recalibration",
        "comparison": "model and market scored only on identical supported fixtures; no cross-season pooling",
        "availability_caveat": "earlier-date results assumed available; first publication and quote times unknown"}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": {"scope": "historical complete-score team-method replication in two independently fitted seasons",
            "current_predictions": False, "player_validation": False,
            "cross_season_player_coefficient_migration": False,
            "claim_limit": "not blind untouched data, not current team strengths, no betting or ROI claim"},
        "protocol": protocol, "sources": sources, "seasons": seasons}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "球队验证完整结果.json", result)
    _write_json(output / "逐场球队验证.json", predictions)
    _write_json(output / "逐周完整参数.json", fits)
    _write_json(output / "数据接入与排除.json", sources)
    _write_csv(output / "逐场球队验证.csv", predictions)
    write_report(output / "球队层验证说明.md", result)
    print(json.dumps({"output_directory": str(output), "seasons": {
        name: {"coverage": value["coverage"], "model_all_supported": value["metrics"]["model_all_supported"],
            "paired_comparisons": {market: {key: paired[key] for key in
                ("n_matches", "model_on_identical_matches", "market")}
                for market, paired in value["metrics"]["market_comparisons"].items()}}
        for name, value in seasons.items()}}, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.root, args.output_dir)


if __name__ == "__main__":
    main()
