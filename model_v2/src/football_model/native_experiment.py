"""Installed penaltyblog models: chronological validation, then later-season test.

Model family and decay are selected only on 2024/25. 2025/26 is evaluated after
selection and never used to change it. This remains a date-level experiment.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from .date_model import DateTeamGoalModel
from .football_data import LEAGUES, load_directory
from .penaltyblog_adapter import NativeFitError, NativeGoalModel
from .probability import score_grid, summarize
from .real_backtest import de_vig, descriptive, evaluate, write_csv, _bar_svg


CANDIDATES = [
    {"id": f"{name}_h{half if half is not None else 'equal'}",
     "family": name, "half_life_days": half}
    for name in ("poisson", "dixon_coles") for half in (90, 180, 365, None)
]
ORIGINAL = {"id": "original_ridge_poisson_h180", "family": "original", "half_life_days": 180}
VALID_START, VALID_END = date(2024, 8, 1), date(2025, 7, 1)
TEST_START, TEST_END = date(2025, 8, 1), date(2026, 7, 1)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_model(config):
    if config["family"] == "original":
        return DateTeamGoalModel(0.01, config["half_life_days"])
    return NativeGoalModel(config["family"], config["half_life_days"])


def expected_target_ids(records, start, end, interval_days=28):
    """Target coverage determined by identities/dates, not predictions or scores."""
    output = set()
    cutoff = start
    while cutoff < end:
        stop = min(end, cutoff + timedelta(days=interval_days))
        teams = {t for r in records if r.match_date < cutoff and not r.goal_fit_exclusion
                 for t in (r.home, r.away)}
        output.update(r.match_id for r in records if cutoff <= r.match_date < stop
                      and not r.goal_fit_exclusion and {r.home, r.away} <= teams)
        cutoff = stop
    return output


def run_config(records, config, start, end, interval_days=28):
    if type(start) is not date or type(end) is not date or start >= end:
        raise ValueError("Use ordered calendar-date boundaries")
    if type(interval_days) is not int or interval_days < 1:
        raise ValueError("Update interval must be a positive integer")
    if len({r.competition for r in records}) != 1:
        raise ValueError("Run each league separately")
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate matches must be resolved before fitting")
    predictions, fits, failures, skipped = [], [], [], []
    cutoff = start
    while cutoff < end:
        stop = min(end, cutoff + timedelta(days=interval_days))
        targets = sorted((r for r in records if cutoff <= r.match_date < stop),
                         key=lambda r: (r.match_date, r.match_id))
        usable = []
        for match in targets:
            if match.goal_fit_exclusion:
                skipped.append({"match_id": match.match_id, "reason": match.goal_fit_exclusion})
            else:
                usable.append(match)
        if usable:
            earlier = [r for r in records if r.match_date < cutoff and not r.goal_fit_exclusion]
            known = {t for r in earlier for t in (r.home, r.away)}
            targets = []
            for match in usable:
                if not {match.home, match.away} <= known:
                    skipped.append({"match_id": match.match_id, "reason": "unseen_team_no_estimated_prior"})
                else:
                    targets.append(match)
            if targets:
                try:
                    model = make_model(config).fit(earlier, cutoff)
                except (NativeFitError, ValueError, FloatingPointError) as error:
                    failures.append({"cutoff": cutoff.isoformat(), "stage": "fit",
                                     "matches": [r.match_id for r in targets], "error": str(error)})
                    cutoff = stop
                    continue
                fits.append(model.parameters())
                for match in targets:
                    try:
                        grid = model.predict(match.home, match.away, match.competition, match.match_date)
                        if config["family"] == "original":
                            grid = score_grid(*grid, rho=0)
                        forecast = summarize(grid)
                    except (ValueError, FloatingPointError) as error:
                        failures.append({"cutoff": cutoff.isoformat(), "stage": "predict",
                                         "match_id": match.match_id, "error": str(error)})
                        continue
                    h, a = match.full_time
                    predictions.append({
                        "match_id": match.match_id, "competition": match.competition,
                        "date": match.match_date.isoformat(), "home": match.home_label,
                        "away": match.away_label, "fit_before_date": cutoff.isoformat(),
                        "training_max_date": model.training_max_date.isoformat(),
                        "training_matches": model.n_matches,
                        "actual_score": [h, a], "actual_outcome": 0 if h > a else 2 if h < a else 1,
                        "probabilities": list(forecast["one_x_two"].values()), "forecast": forecast,
                        "market_probabilities": {k: list(map(float, de_vig(v))) for k, v in match.markets.items()},
                        "market_prices": match.markets, "source_url": match.source_url,
                    })
        cutoff = stop
    return {"config": config, "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "update_interval_days": interval_days, "predictions": predictions, "fits": fits,
            "failures": failures, "skipped": skipped, "metrics": evaluate(predictions)}


def select_config(validation_runs, expected_ids):
    """Only validation results enter model selection; incomplete runs are reported."""
    for run in validation_runs:
        if (run["start"] != VALID_START.isoformat()
                or run["end_exclusive"] != VALID_END.isoformat()
                or any(not VALID_START <= date.fromisoformat(p["date"]) < VALID_END
                       for p in run["predictions"])):
            raise ValueError("Model selection accepts only the designated validation period, not test results")
    eligible = [r for r in validation_runs if not r["failures"]
                and {p["match_id"] for p in r["predictions"]} == set(expected_ids)
                and r["metrics"]["model"] is not None]
    if not eligible:
        raise ValueError("No native candidate completed the same validation match set")
    chosen = min(eligible, key=lambda r: (r["metrics"]["model"]["log_loss"], r["config"]["id"]))
    return {"config": chosen["config"], "validation_scores": chosen["metrics"]["model"],
            "selection_metric": "2024/25 chronological validation mean 1X2 log loss",
            "eligible_candidates": len(eligible), "total_candidates": len(validation_runs),
            "expected_matches": len(expected_ids)}


def season_directories(root):
    directories = {"2024/25": root / "outputs" / "Football-Data_2024-25_个人使用_20261022T111235Z"}
    for season in ("2022/23", "2023/24", "2025/26"):
        parent = root / "data" / "football-data" / season.replace("/", "-")
        available = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "下载来源.json").exists())
        if not available:
            raise ValueError(f"No downloaded source directory for {season}")
        directories[season] = available[-1]
    return directories


def load_seasons(directories):
    records, reports = [], {}
    for season, directory in sorted(directories.items()):
        matches, report = load_directory(directory, season=season)
        if report["rejected_rows"] or report["duplicate_keys"]:
            raise ValueError(f"Resolve invalid/duplicate rows in {season}")
        records.extend(matches)
        reports[season] = report
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate cross-season match IDs")
    return records, reports


def _report(path, result):
    table = result["matched_test_metrics"]
    selected = table["validation_selected"]["model"]
    lines = ["# penaltyblog 整包接入、训练与参数选择", "",
        "## 本轮实际完成", "",
        f"- 完整安装并调用 penaltyblog {result['runtime']['penaltyblog']} 的原生拟合、预测和保存接口，不再只复用比分容器。",
        f"- 四赛季共 {result['data_rows']} 场；可用 {result['eligible_goal_rows']} 场。原CSV不改写，排除已知行政改判及两场分段续赛。",
        "- 起步训练：2022/23、2023/24；调参与比较：2024/25；较晚赛季检验：2025/26。",
        "- 候选为 Poisson / Dixon–Coles × 半衰期90、180、365天或不衰减，共8组；每联赛独立选择。Dixon–Coles的rho由原生模型拟合，不手填。",
        "- 调参与检验均每28天更新，只使用更新日之前的赛果；同一时间块内不使用后续结果。调整更新频率是本轮预先设定的计算方案，所有模型一致。",
        "- 先完成2024/25选择，再运行2025/26；最终选择不按2025/26排名修改。检验期间可使用此前已结束的比赛作下一次参数更新。",
        "- 旧自建正则Poisson、原生Poisson180天、原生DC180天同时保留对照，市场单独去水，不融合。", "",
        "## 验证期选择", "",
        "| 联赛 | 选定模型 | 半衰期 | 验证场数 | 验证Log loss | 完整候选数 |",
        "|---|---|---:|---:|---:|---:|"]
    for comp, choice in result["selection"].items():
        cfg = choice["config"]
        half = "不衰减" if cfg["half_life_days"] is None else str(cfg["half_life_days"])
        lines.append(f"| {LEAGUES[comp]} | {cfg['family']} | {half} | {choice['validation_scores']['n']} | {choice['validation_scores']['log_loss']:.4f} | {choice['eligible_candidates']}/8 |")
    lines += ["", "## 较晚赛季：相同比赛上的比较", "",
        "Log loss、Brier、RPS越低越好。原生模型不一定比旧模型或市场更准确，按实测结果判断。", "",
        "| 方案 | 场数 | Log loss | Brier（三类和） | RPS | 命中率 |",
        "|---|---:|---:|---:|---:|---:|"]
    labels = {"original": "旧自建模型", "native_poisson_180": "原生Poisson，180天",
              "native_dc_180": "原生Dixon–Coles，180天", "validation_selected": "验证期选定方案"}
    for key, metrics in table.items():
        m = metrics["model"]
        if m:
            lines.append(f"| {labels[key]} | {m['n']} | {m['log_loss']:.4f} | {m['brier_sum']:.4f} | {m['rps']:.4f} | {m['accuracy']:.1%} |")
    for key in ("avg_preclosing", "avg_closing"):
        entry = table["validation_selected"]["markets"].get(key)
        if entry:
            m = entry["market"]
            lines.append(f"| {key} 市场去水 | {m['n']} | {m['log_loss']:.4f} | {m['brier_sum']:.4f} | {m['rps']:.4f} | {m['accuracy']:.1%} |")
    old_loss = table["original"]["model"]["log_loss"]
    delta = selected["log_loss"] - old_loss
    lines += ["", f"选定方案相对旧模型的平均Log loss差为 {delta:+.4f}（负数表示改善）。这不是保证盈利或因果证明。", "",
        "## 覆盖与不确定性", "",
        f"- 原始检验季 {result['test_target_rows']} 场；四方案共同完成预测 {len(result['common_test_match_ids'])} 场。",
        "- 不认识的新升班球队不填默认实力，先跳过；不同模型的拟合/预测失败也单独记录，未裁剪负概率或静默换模型。各方案完整覆盖与失败原因见JSON。",
        "- 原生比分网格覆盖0–29球，检查未归一化概率质量、负值和非有限值后才汇总；没有调用此前有疑点的外部lambda网格工厂。",
        "- 检验期校准分箱、期望/实际进球、平局、Pearson残差和进球相关见JSON；计数模型不套用正态残差检验。高比分有效比赛不删除。",
        "- 与市场的Log loss配对区间采用28天时间块重采样，属于探索性区间，不是球员/模型参数置信区间，不能消除全部时间依赖。",
        "- 这轮2025/26成绩被查看后，后续再据此调优就需要新的较晚时段或外部检验，不能继续称同一赛季完全未见。", "",
        "## 数据时间与特殊比赛", "",
        "- 原始数据日期、历史首次公开时间、赔率报价时间不同。本轮假设更早日历日期已结束的合格赛果在更新时已知，不能重建精确赛前24小时或首发后信息集。",
        "- 2024-12-14 柏林联合—波鸿：行政改判0–2，排除场上进球拟合。",
        "- 2023-12-11 格拉纳达—毕尔巴鄂及2024-04-25乌迪内斯—罗马：CSV日期对应部分续赛，保留原始比分，保守排除统一日历实验。",
        "- 原始分布图仅展示研发季，不利用检验季描述来选候选模型。四季数据来源和已知特殊比赛证据均列入数据接入检查JSON。", "",
        "## 交付与仍未完成", "",
        "- `较晚赛季逐场预测.csv`：验证期选定方案的逐场概率。",
        "- `验证期选择.json`、`验证期/`：所有候选的成绩、参数、预测和失败记录。",
        "- `整包训练完整结果.json`、`较晚赛季/`：同场对照、校准和诊断。",
        "- `训练模型/`：每联赛选定方案的原生pickle及可读参数JSON，只加载本机自己生成的可信pickle。模型训练止于2026年5月，并非2026/27当前球队实力。",
        "- 整包安装不等于全部模型均已验证：负二项、双变量Poisson、Weibull和贝叶斯模型在包内可用，但本轮未训练。",
        "- 球员、伤停、赛事重视程度仍未接入真实训练；球队边际率与球员条件率衔接、跨联赛实力桥梁、半全场和实时双时点快照仍待完成。", "",
        "参考：[penaltyblog原生模型说明](https://penaltyblog.readthedocs.io/en/latest/models/overview.html)。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root, output_dir):
    started = time.perf_counter()
    records, reports = load_seasons(season_directories(root))
    output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("验证期", "较晚赛季", "训练模型"):
        (output_dir / folder).mkdir()
    protocol = {"validation_start": VALID_START.isoformat(), "validation_end_exclusive": VALID_END.isoformat(),
        "test_start": TEST_START.isoformat(), "test_end_exclusive": TEST_END.isoformat(),
        "update_interval_days": 28, "candidates": CANDIDATES,
        "selection_metric": "validation_1X2_log_loss", "market_blending": False,
        "player_adjustment": False, "test_used_for_selection": False,
        "date_availability_assumption": "strictly_earlier_calendar_dates_assumed_known"}
    _json(output_dir / "实验设置.json", protocol)
    _json(output_dir / "数据接入检查.json", reports)
    # Raw descriptive plot is generated before fits and uses development seasons only.
    development = [r for r in records if r.match_date < VALID_END]
    (output_dir / "研发季原始进球分布.svg").write_text(
        _bar_svg(descriptive(development), "2022/23–2024/25 development goals; known special matches excluded"), encoding="utf-8")
    selection, validation_table = {}, {}
    for comp in sorted(LEAGUES):
        league = [r for r in records if r.competition == comp]
        expected = expected_target_ids(league, VALID_START, VALID_END)
        runs = []
        for config in CANDIDATES:
            trial = run_config(league, config, VALID_START, VALID_END)
            _json(output_dir / "验证期" / f"{comp}_{config['id']}.json", trial)
            runs.append(trial)
            m = trial["metrics"]["model"]
            score = None if m is None else round(m["log_loss"], 5)
            print(f"validation {comp} {config['id']}: n={len(trial['predictions'])}, loss={score}, failures={len(trial['failures'])}", flush=True)
        selection[comp] = select_config(runs, expected)
        validation_table[comp] = [{"config": r["config"], "metrics": r["metrics"],
                                   "failure_count": len(r["failures"]), "skipped": r["skipped"]}
                                  for r in runs]
    # This file is an ordinary result record, written before any test scoring.
    _json(output_dir / "验证期选择.json", selection)
    combined = {key: [] for key in ("original", "native_poisson_180", "native_dc_180", "validation_selected")}
    test_coverage, final_models = {}, {}
    for comp in sorted(LEAGUES):
        league = [r for r in records if r.competition == comp]
        choices = {"original": ORIGINAL,
                   "native_poisson_180": {"id": "poisson_h180", "family": "poisson", "half_life_days": 180},
                   "native_dc_180": {"id": "dixon_coles_h180", "family": "dixon_coles", "half_life_days": 180},
                   "validation_selected": selection[comp]["config"]}
        local = {}
        test_coverage[comp] = {}
        for label, config in choices.items():
            if config["id"] not in local:
                local[config["id"]] = run_config(league, config, TEST_START, TEST_END)
                _json(output_dir / "较晚赛季" / f"{comp}_{config['id']}.json", local[config["id"]])
            trial = local[config["id"]]
            combined[label].extend(trial["predictions"])
            test_coverage[comp][label] = {"predicted": len(trial["predictions"]),
                "skipped_reasons": dict(Counter(p["reason"] for p in trial["skipped"])),
                "skipped": trial["skipped"], "failures": trial["failures"]}
        cfg = selection[comp]["config"]
        before = max(r.match_date for r in league) + timedelta(days=1)
        try:
            model = make_model(cfg).fit(league, before)
            model.save(output_dir / "训练模型" / f"{comp}_{cfg['id']}.pkl")
            final_models[comp] = model.parameters()
            _json(output_dir / "训练模型" / f"{comp}_参数.json", final_models[comp])
        except (NativeFitError, ValueError) as error:
            final_models[comp] = {"config": cfg, "save_failed": str(error)}
        print(f"test {comp}: validation-selected n={test_coverage[comp]['validation_selected']['predicted']}", flush=True)
    common = set.intersection(*({p["match_id"] for p in rows} for rows in combined.values()))
    if not common:
        raise ValueError("No common completed test predictions")
    matched = {k: sorted((p for p in rows if p["match_id"] in common), key=lambda p: (p["date"], p["match_id"]))
               for k, rows in combined.items()}
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "runtime": {"python": sys.version, **{name: version(name) for name in ("penaltyblog", "numpy", "scipy", "scikit-learn")}},
        "elapsed_seconds": time.perf_counter() - started,
        "data_rows": len(records), "eligible_goal_rows": sum(not r.goal_fit_exclusion for r in records),
        "data_reports": reports, "selection": selection, "validation_table": validation_table,
        "test_target_rows": sum(TEST_START <= r.match_date < TEST_END for r in records),
        "test_coverage": test_coverage, "common_test_match_ids": sorted(common),
        "matched_test_metrics": {k: evaluate(rows) for k, rows in matched.items()},
        "by_league_selected": {c: evaluate([p for p in matched["validation_selected"] if p["competition"] == c]) for c in LEAGUES},
        "final_model_parameters": final_models,
    }
    _json(output_dir / "整包训练完整结果.json", result)
    write_csv(output_dir / "较晚赛季逐场预测.csv", matched["validation_selected"])
    _report(output_dir / "整包接入与训练报告.md", result)
    print(json.dumps({"output_dir": str(output_dir), "common_test_matches": len(common),
                      "test_scores": {k: v["model"] for k, v in result["matched_test_metrics"].items()},
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or root / "outputs" / ("整包模型训练_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run(root, output)


if __name__ == "__main__":
    main()
