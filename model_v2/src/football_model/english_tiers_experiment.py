"""One bounded exploratory English-tier comparison, never a forecast upgrade.

Both arms reuse DateTeamGoalModel(.01, 180). Historical scores were already
inspected; this is not a fresh blind test. No hyperparameter search is performed.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

from .date_model import DateTeamGoalModel
from .english_tiers_data import TEAM_ALIASES
from .football_data import HistoricalMatch
from .probability import score_grid, summarize


ARMS = ("epl_only", "joint")
METRICS = ("full_score_nll", "one_x_two_log_loss", "one_x_two_brier", "top1",
           "total_log_loss", "total_brier", "away_minus_two_log_loss", "away_minus_two_brier")


def _write(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_inspections(directory):
    directory = Path(directory).resolve()
    files = sorted(directory.glob("*.inspection.json"))
    if not files:
        raise ValueError("No English-tier inspection files")
    rows, excluded, sources, excluded_issues = [], [], [], []
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        source_competition, season = value["competition"], value["season"]
        if source_competition not in {"en.1", "en.2"}:
            raise ValueError("Unknown source competition")
        excluded_lines = {r["source_line"] for r in value["records"]
                          if r["stage"] != "regular" and type(r.get("source_line")) is int}
        source_issues = value.get("issues", [])
        blocking_issues = [issue for issue in source_issues if issue.get("line") not in excluded_lines]
        excluded_issues.extend({"source_path": str(path), **issue} for issue in source_issues if issue not in blocking_issues)
        if value.get("duplicate_keys") or blocking_issues:
            raise ValueError("Resolve source parse issues/duplicates before research fitting")
        competition = {"en.1": "E0", "en.2": "E1"}[source_competition]
        sources.append({"path": str(path), "season": season, "source_competition": source_competition})
        for row in value["records"]:
            if row["competition"] != source_competition or row["season"] != season:
                raise ValueError("Record and inspection season/division disagree")
            if row["stage"] != "regular":
                excluded.append({"match_id": row["match_id"], "reason": "non_regular_stage", "stage": row["stage"]})
                continue
            if row["eligible_regular_result"] is not True or row.get("extra_time") or row.get("penalties"):
                raise ValueError("Ineligible or extra-time regular source observation")
            day = date.fromisoformat(row["match_date"])
            if not isinstance(row["match_date"], str) or day.isoformat() != row["match_date"]:
                raise ValueError("Use unambiguous calendar dates")
            score = row["regulation_time_score"]
            if not isinstance(score, (list, tuple)) or len(score) != 2 or any(type(x) is not int or x < 0 for x in score):
                raise ValueError("Two nonnegative integer full-time goals required")
            labels, ids = [], []
            for side in ("home", "away"):
                info = row[side]
                canonical = TEAM_ALIASES.get(info["source_label"])
                if canonical is None or info["canonical_label"] != canonical or info["team_id"] != "england:" + canonical:
                    raise ValueError("Unknown or conflicting explicit team alias")
                labels.append(canonical)
                ids.append(info["team_id"])
            if ids[0] == ids[1]:
                raise ValueError("Distinct teams required")
            rows.append(HistoricalMatch(row["match_id"], competition, season, day, None,
                                        ids[0], ids[1], labels[0], labels[1], tuple(score), None, {},
                                        str(path), row.get("source_url", ""), None))
    duplicates = sorted(key for key, count in Counter(r.match_id for r in rows).items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate source match identifiers")
    rows.sort(key=lambda r: (r.match_date, r.competition, r.match_id))
    return rows, {"sources": sources, "regular_records": len(rows), "excluded": excluded,
                  "excluded_non_regular_records": len(excluded), "excluded_source_issues": excluded_issues,
                  "duplicate_keys": duplicates,
                  "time_semantics": "Strictly earlier source calendar dates assumed available; historical first-publication times unknown"}


def derive_promoted(records, target_season="2025-26", prior_season="2024-25"):
    target = {t for r in records if r.season == target_season and r.competition == "E0" for t in (r.home, r.away)}
    prior_second = {t for r in records if r.season == prior_season and r.competition == "E1" for t in (r.home, r.away)}
    return target & prior_second


def _categorical(probabilities, actual):
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-8):
        raise ValueError("Normalized categorical probabilities required")
    if probabilities[actual] <= 0:
        raise ValueError("Observed outcome has zero numerical probability")
    return (-math.log(probabilities[actual]),
            sum((p - int(i == actual)) ** 2 for i, p in enumerate(probabilities)))


def score_forecast(rates, actual):
    if len(rates) != 2 or any(not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError("Two positive finite goal rates required")
    if len(actual) != 2 or any(type(x) is not int or x < 0 for x in actual):
        raise ValueError("Two nonnegative integer goals required")
    h, a = actual
    forecast = summarize(score_grid(*rates, rho=0))
    p = list(forecast["one_x_two"].values())
    outcome = 0 if h > a else 1 if h == a else 2
    tied = [i for i, probability in enumerate(p) if probability == max(p)]
    loss, brier = _categorical(p, outcome)
    totals = [forecast["total_goals"][x] for x in ("0", "1", "2", "3", "4+")]
    total_loss, total_brier = _categorical(totals, min(h + a, 4))
    jc = forecast["away_minus_two_jincai"]
    handicap_actual = 0 if a-h > 2 else 1 if a-h == 2 else 2
    handicap_loss, handicap_brier = _categorical([jc[x] for x in ("win", "draw", "lose")], handicap_actual)
    nll = sum(rate - goals * math.log(rate) + math.lgamma(goals + 1)
              for rate, goals in zip(rates, actual))
    return {"forecast": forecast, "full_score_nll": nll, "one_x_two_log_loss": loss,
            "one_x_two_brier": brier, "top1": 1/len(tied) if outcome in tied else 0.,
            "top1_tied": len(tied) > 1,
            "total_log_loss": total_loss, "total_brier": total_brier,
            "away_minus_two_log_loss": handicap_loss, "away_minus_two_brier": handicap_brier}


def summarize_rows(rows):
    usable = [r for r in rows if r.get("status") == "predicted"]
    return {"n": len(usable), "top1_tie_count": sum(r.get("top1_tied", False) for r in usable),
            **{key: sum(r[key] for r in usable)/len(usable) if usable else None for key in METRICS}}


def run_comparison(records, output_dir, model_factory=None, start=date(2025, 8, 1),
                   stop=date(2026, 6, 1), interval_days=28):
    if type(start) is not date or type(stop) is not date or start >= stop or type(interval_days) is not int or interval_days < 1:
        raise ValueError("Ordered date boundaries and positive cadence required")
    if len({r.match_id for r in records}) != len(records):
        raise ValueError("Duplicate match identifiers")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters_dir = output / "parameters"
    parameters_dir.mkdir(exist_ok=True)
    factory = model_factory or (lambda: DateTeamGoalModel(.01, 180))
    begun = time.perf_counter()
    promoted = derive_promoted(records)
    targets = sorted((r for r in records if r.season == "2025-26" and r.competition == "E0"
                      and start <= r.match_date < stop), key=lambda r: (r.match_date, r.match_id))
    per_fixture, fits = [], []
    cutoff = start
    while cutoff < stop:
        window_end = min(stop, cutoff + timedelta(days=interval_days))
        window = [r for r in targets if cutoff <= r.match_date < window_end]
        if window:
            window_rows = {r.match_id: {"match_id": r.match_id, "match_date": r.match_date.isoformat(),
                            "home": r.home, "away": r.away, "actual": list(r.full_time),
                            "promoted_fixture": bool({r.home, r.away} & promoted),
                            "cutoff": cutoff.isoformat(), "arms": {}} for r in window}
            for arm in ARMS:
                history = [r for r in records if r.match_date < cutoff and not r.goal_fit_exclusion
                           and (arm == "joint" or r.competition == "E0")]
                fit_start = time.perf_counter()
                # Actual failures propagate. Never substitute a weaker/fitted-on-later model.
                model = factory().fit(history, cutoff)
                parameters = model.parameters()
                parameter_path = parameters_dir / f"{arm}_{cutoff.isoformat()}.json"
                saved_at = datetime.now(timezone.utc).isoformat()
                _write(parameter_path, {"arm": arm, "fit_cutoff_date": cutoff.isoformat(),
                                       "saved_at": saved_at, "parameters": parameters})
                weights = [(r.competition, math.exp(-math.log(2)*(cutoff-r.match_date).days/180)) for r in history]
                fit = {"arm": arm, "cutoff": cutoff.isoformat(), "training_matches": len(history),
                       "training_max_date": max(r.match_date for r in history).isoformat(),
                       "parameter_path": str(parameter_path), "parameters_saved_at": saved_at,
                       "elapsed_seconds": time.perf_counter()-fit_start,
                       "weighted_epl_objective_fraction": sum(w for c, w in weights if c == "E0")/sum(w for c, w in weights)}
                fits.append(fit)
                for target in window:
                    missing = sorted({target.home, target.away} - model.teams)
                    if missing:
                        value = {"status": "missing", "reason": "unseen_team_no_estimated_prior",
                                 "missing_teams": missing, "rates": None, "forecast": None,
                                 "parameter_path": str(parameter_path)}
                    else:
                        rates = tuple(model.predict(target.home, target.away, "E0", target.match_date))
                        value = {"status": "predicted", "rates": list(rates),
                                 "parameter_path": str(parameter_path), **score_forecast(rates, target.full_time)}
                    window_rows[target.match_id]["arms"][arm] = value
            per_fixture.extend(window_rows[target.match_id] for target in window)
            print(json.dumps({"completed_window": cutoff.isoformat(), "fixtures": len(window),
                              "fits_so_far": len(fits)}, ensure_ascii=False), flush=True)
        cutoff = window_end
    if len(per_fixture) != len(targets) or len({r["match_id"] for r in per_fixture}) != len(targets):
        raise ValueError("Every target must appear exactly once")
    paired = [r for r in per_fixture if all(r["arms"][a]["status"] == "predicted" for a in ARMS)]
    groups = {"all_paired": paired,
              "promoted_paired": [r for r in paired if r["promoted_fixture"]],
              "other_paired": [r for r in paired if not r["promoted_fixture"]],
              "before_2026_paired": [r for r in paired if r["match_date"] < "2026-01-01"],
              "from_2026_paired": [r for r in paired if r["match_date"] >= "2026-01-01"]}
    comparison = {}
    for name, group in groups.items():
        by_arm = {a: summarize_rows([r["arms"][a] for r in group]) for a in ARMS}
        comparison[name] = {"n": len(group), "arms": by_arm,
                            "joint_minus_epl_only": {m: by_arm["joint"][m]-by_arm["epl_only"][m] if group else None for m in METRICS}}
    arms = {arm: {"target_count": len(targets), "coverage": sum(r["arms"][arm]["status"] == "predicted" for r in per_fixture),
                  "missing": [{"match_id": r["match_id"], **r["arms"][arm]} for r in per_fixture if r["arms"][arm]["status"] != "predicted"],
                  "metrics": summarize_rows([r["arms"][arm] for r in per_fixture])} for arm in ARMS}
    return {"arms": arms, "per_fixture_rows": per_fixture, "fits": fits, "fit_count": len(fits),
            "elapsed_seconds": time.perf_counter()-begun, "parameter_searches": 0,
            "target_ids": [r.match_id for r in targets], "target_count": len(targets),
            "promoted_teams": sorted(promoted), "promoted_fixture_count": sum(r["promoted_fixture"] for r in per_fixture),
            "paired_target_ids": [r["match_id"] for r in paired], "paired_comparison": comparison,
            "parameters": {"regularization": .01, "half_life_days": 180, "refit_interval_days": interval_days},
            "scope": "Exploratory historical, already-inspected 2025/26 results; not a fresh blind test or current forecast update",
            "limitations": ["Strictly earlier calendar dates assumed available; no historical publication or UTC kickoff reconstruction.",
                            "Shared club effects are assumed transferable over time under decay and shrinkage, despite roster/coach/level changes.",
                            "Competition scoring environment and home advantage are not identified division-quality parameters.",
                            "Adding 552 E1 fixtures per season changes normalized loss weighting and shrinkage context even with alpha=.01 unchanged; this is a joint-fit package effect, not an isolated promotion-prior effect.",
                            "One fixture is one paired observational unit; two lines, multiple metrics and subgroup views do not create independent replications.",
                            "No parameter tuning, statistical significance claim, automatic adoption, LLM inference or current prediction change.",
                            "All losses are unscaled category-sum Brier or natural-log loss; lower is better except top1 accuracy."]}


def render_report(result):
    lines = ["# 英超／英冠共享球队身份：一次历史前推研究", "",
             "这是已看过历史数据后的探索性比较，不是新的盲测；不更新已记录的当前预测。", "",
             f"实际拟合 {result['fit_count']} 次，超参数搜索0次，耗时 {result['elapsed_seconds']:.2f} 秒。",
             "两线都用 alpha=.01、180天半衰期、28天更新；每窗参数先保存，随后只预测该窗较晚日期比赛。", "",
             "| 同场样本 | n | 英超单独：完整比分NLL | 共享：完整比分NLL | 共享−单独 |", "|---|---:|---:|---:|---:|"]
    for name, row in result["paired_comparison"].items():
        if row["n"]:
            lines.append(f"| {name} | {row['n']} | {row['arms']['epl_only']['full_score_nll']:.6f} | {row['arms']['joint']['full_score_nll']:.6f} | {row['joint_minus_epl_only']['full_score_nll']:+.6f} |")
    lines += ["", "| 同场全样本指标 | 英超单独 | 共享 | 共享−单独 |", "|---|---:|---:|---:|"]
    pair = result["paired_comparison"]["all_paired"]
    if pair["n"]:
        for metric in METRICS:
            lines.append(f"| {metric} | {pair['arms']['epl_only'][metric]:.6f} | {pair['arms']['joint'][metric]:.6f} | {pair['joint_minus_epl_only'][metric]:+.6f} |")
    lines += ["", f"目标{result['target_count']}场；英超单独覆盖{result['arms']['epl_only']['coverage']}场，共享覆盖{result['arms']['joint']['coverage']}场。", 
              f"事先按2024/25英冠身份与2025/26英超赛程球队交集定义升班马：{', '.join(result['promoted_teams'])}；涉及其至少一队的比赛{result['promoted_fixture_count']}场，一场只计一次。", "",
              "完整比分NLL用独立Poisson解析式，未用截断网格代替尾部；胜平负、总进球0/1/2/3/4+和客让两球来自同一比分分布。", "",
              "Top1若最高概率精确并列，按在并列项间等概率选择的期望准确率计分（命中给1/并列项数），与现有前瞻评分一致。", "",
              "客让两球为竞彩三项：净胜至少3球才让胜，净胜2球为让平。没有将低让胜概率转换为高信心推荐。", "",
              "## 解释边界", ""]
    lines += ["- " + note for note in result["limitations"]]
    lines += ["", "源数据：openfootball/england，仓库声明CC0；四季英超1520场、英冠2208场常规比赛。20场附加赛单列，不进入拟合。", "",
              "逐场JSON/CSV保留两线概率、解析损失、缺失原因、拟合窗口和参数文件路径。更低的本次历史损失不等于已验证今后的优势。"]
    return "\n".join(lines) + "\n"


def run_experiment(root, inspection):
    root, inspection = Path(root).expanduser().resolve(), Path(inspection).expanduser().resolve()
    records, audit = load_inspections(inspection)
    begun = datetime.now(timezone.utc)
    output = root / "outputs" / ("英超英冠共享身份研究_" + begun.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "研究安排.json", {"started_at": begun.isoformat(), "inspection_directory": str(inspection),
                                     "arms": ARMS, "alpha": .01, "half_life_days": 180, "cadence_days": 28,
                                     "target": "All 2025/26 EPL fixtures", "exploratory_not_blind": True,
                                     "parameter_searches": 0, "main_metric": "analytic independent-Poisson full-score NLL"})
    _write(output / "输入资料审计.json", audit)
    result = run_comparison(records, output)
    result.update({"started_at": begun.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                   "inspection_directory": str(inspection), "output_directory": str(output), "input_audit": audit})
    _write(output / "研究结果及逐场概率.json", result)
    with (output / "逐场对照.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        names = ["match_id", "match_date", "home", "away", "actual", "promoted_fixture", "cutoff", "arm", "status", "reason", "rates", "forecast", "parameter_path", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in result["per_fixture_rows"]:
            for arm in ARMS:
                flat = {**{k: v for k, v in row.items() if k != "arms"}, "arm": arm, **row["arms"][arm]}
                for key in ("actual", "rates", "forecast"):
                    flat[key] = json.dumps(flat.get(key), ensure_ascii=False, allow_nan=False)
                writer.writerow(flat)
    (output / "历史对照说明.md").write_text(render_report(result), encoding="utf-8")
    return result, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    args = parser.parse_args(argv)
    result, output = run_experiment(args.root, args.inspection)
    print(json.dumps({"output_directory": str(output), "fit_count": result["fit_count"],
                      "elapsed_seconds": result["elapsed_seconds"], "paired_comparison": result["paired_comparison"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
