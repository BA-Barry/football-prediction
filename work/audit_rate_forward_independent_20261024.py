"""Read-only numeric replay of a saved experiment, not another test sample.

Uses only standard-library arithmetic, no production fitting, probability,
loading or scoring function. Writes a separate audit result under work/.
"""

import csv
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT


ROOT = REPOSITORY_ROOT
OUT = ROOT / "outputs/进球率尺度多窗口前推_20261023T162450849975Z"
ARMS = ("original", "shared_scale", "side_scale")
DOMAINS = ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way", "exact_score")
ERRORS = {}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(label, actual, expected, limit=2e-11):
    error = abs(actual-expected)
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if not math.isfinite(error) or error > limit:
        raise AssertionError((label, actual, expected, error))


def exact_nll(goals, rates):
    return math.fsum(mu-y*math.log(mu)+math.lgamma(y+1) for y, mu in zip(goals, rates))


def marginal(mu, maximum):
    values = [math.exp(-mu)]
    for k in range(1, maximum+1):
        values.append(values[-1]*mu/k)
    return values


def grid_summary(rates, maximum):
    h, a = (marginal(mu, maximum) for mu in rates)
    normalizer = math.fsum(h)*math.fsum(a)
    cells = [(i, j, hi*aj/normalizer) for i, hi in enumerate(h) for j, aj in enumerate(a)]

    def probability(condition):
        return math.fsum(p for i, j, p in cells if condition(i, j))

    one_x_two = [probability(lambda i, j: i > j), probability(lambda i, j: i == j),
                 probability(lambda i, j: i < j)]
    totals = [probability(lambda i, j, k=k: i+j == k) for k in range(4)]
    totals.append(probability(lambda i, j: i+j >= 4))
    handicap = [probability(lambda i, j: j-i >= 3), probability(lambda i, j: j-i == 2),
                probability(lambda i, j: j-i < 2)]
    means = [math.fsum(i*p for i, j, p in cells), math.fsum(j*p for i, j, p in cells)]
    full_h, full_a = (marginal(mu, maximum+100) for mu in rates)
    ht, at = math.fsum(full_h[maximum+1:]), math.fsum(full_a[maximum+1:])
    return {
        "one_x_two": one_x_two, "total_goals_0_1_2_3_4plus": totals,
        "away_minus_two_three_way": handicap, "means": means,
        "over_2_5": probability(lambda i, j: i+j >= 3),
        "both_score": probability(lambda i, j: i >= 1 and j >= 1),
        "omitted_tail": ht+at-ht*at,
    }


def categorical_metrics(p, y):
    assert p[y] > 0
    cdf_error = []
    for k in range(len(p)-1):
        cdf_error.append((math.fsum(p[:k+1])-int(y <= k))**2)
    return {"log_loss": -math.log(p[y]),
            "brier_sum": math.fsum((value-int(i == y))**2 for i, value in enumerate(p)),
            "rps": math.fsum(cdf_error)/(len(p)-1),
            "accuracy": int(max(range(len(p)), key=p.__getitem__) == y)}


result = read(OUT / "进球率尺度多窗口前推结果.json")
settings = read(OUT / "实验设置.json")
assert settings == result["settings"]
source_parent = Path(settings["source_reference"])
source_rows, source_snapshots = {}, 0
for season in ("2024-25", "2025-26"):
    for competition in ("E0", "D1", "I1", "SP1", "F1"):
        trial = read(source_parent / season / f"{competition}_days_28.json")
        fits = {fit["before_date"]: fit for fit in trial["fits"]}
        source_snapshots += len(fits)
        for row in trial["predictions"]:
            model = fits[row["fit_before_date"]]
            assert model["training_max_date"] < model["before_date"] <= row["date"]
            assert model["regularization"] == .01 and model["half_life_days"] == 180 and model["rho"] == 0
            comp = row["competition"]
            home, away = [f"football-data:{comp}:{row[side]}" for side in ("home", "away")]
            c = model["coefficients"]
            rates = [math.exp(model["intercept"] + math.fsum([
                c[f"attack:{home}"], -c[f"defense:{away}"], c[f"competition:{comp}"], c[f"home:{comp}"]])),
                     math.exp(model["intercept"] + math.fsum([
                c[f"attack:{away}"], -c[f"defense:{home}"], c[f"competition:{comp}"]]))]
            assert row["match_id"] not in source_rows
            source_rows[row["match_id"]] = {**row, "rates": rates}

all_tracks = read(OUT / "完整前推预测.json")
expected_ids = [r["match_id"] for r in sorted(source_rows.values(), key=lambda r: (r["date"], r["match_id"]))
                if settings["windows"][0][0] <= r["date"] < settings["windows"][-1][1]]
assert len(expected_ids) == len(set(expected_ids)) == 2491
for arm in ARMS:
    assert [row["match_id"] for row in all_tracks[arm]] == expected_ids
    assert result["pooled"]["n"] == len(all_tracks[arm])
metrics = {arm: {} for arm in ARMS}
fold_summaries, scored = [], set()
identity_fields = ("match_id", "date", "competition", "home", "away", "actual_score", "actual_outcome",
                   "source_url", "fit_before_date", "training_max_date", "training_matches",
                   "market_prices", "market_probabilities")
for fold in result["folds"]:
    cutoff, end = fold["cutoff"], fold["end_exclusive"]
    folder = OUT / cutoff
    parameters = read(folder / "进球率尺度参数.json")
    assert parameters == fold["parameters"]
    training = sorted([row for row in source_rows.values() if settings["training_start"] <= row["date"] < cutoff],
                      key=lambda r: (r["date"], r["match_id"]))
    expected_target = [key for key in expected_ids if cutoff <= source_rows[key]["date"] < end]
    tracks = read(folder / "完整研究预测.json")
    exports = {row["match_id"]: row for row in read(folder / "比分网格与来源.json")}
    scales = {"original": [1., 1.]}
    for method in ARMS[1:]:
        model = parameters[method]
        assert model["training_match_ids"] == [row["match_id"] for row in training]
        assert model["n_matches"] == len(training)
        assert model["training_max_date"] < cutoff
        goal_sums = [sum(row["actual_score"][side] for row in training) for side in (0, 1)]
        rate_sums = [math.fsum(row["rates"][side] for row in training) for side in (0, 1)]
        expected_scales = ([math.fsum(goal_sums)/math.fsum(rate_sums)]*2 if method == "shared_scale"
                           else [goals/mu for goals, mu in zip(goal_sums, rate_sums)])
        scales[method] = expected_scales
        for actual, expected in zip(model["parameters"]["scales"], expected_scales):
            compare("closed_form_scale", actual, expected)
        raw_nll = math.fsum(exact_nll(row["actual_score"], row["rates"]) for row in training)/len(training)
        adjusted_nll = math.fsum(exact_nll(row["actual_score"], [mu*c for mu, c in zip(row["rates"], expected_scales)])
                                   for row in training)/len(training)
        compare("training_raw_nll", model["raw_training_analytic_joint_score_nll"]["mean"], raw_nll)
        compare("training_adjusted_nll", model["calibrated_training_analytic_joint_score_nll"]["mean"], adjusted_nll)
    assert not scored & set(expected_target)
    scored.update(expected_target)
    for arm in ARMS:
        assert [row["match_id"] for row in tracks[arm]] == expected_target
        for row in tracks[arm]:
            source = source_rows[row["match_id"]]
            assert all(row[field] == source[field] for field in identity_fields)
            assert row["calibration_before_date"] == cutoff
            assert row["calibration_target_end_exclusive"] == end
            for actual, expected in zip(row["base_rates"], source["rates"]):
                compare("original_rate_replay", actual, expected)
            expected_rates = [mu*c for mu, c in zip(source["rates"], scales[arm])]
            for actual, expected in zip(row["rates"], expected_rates):
                compare("adjusted_rate", actual, expected)
            nll = exact_nll(row["actual_score"], expected_rates)
            compare("per_match_full_support_nll", row["exact_score_nll"], nll)
            export = exports[row["match_id"]]
            assert export["research_forecasts"][arm] == row["forecast"]
            maximum = export["grid_metadata"][arm]["max_goal_inclusive"]
            independent = grid_summary(expected_rates, maximum)
            compare("tail_mass", export["grid_metadata"][arm]["omitted_tail_mass_before_normalization"],
                    independent["omitted_tail"])
            forecast = row["forecast"]
            probabilities = {"one_x_two": list(forecast["one_x_two"].values()),
                             "total_goals_0_1_2_3_4plus": list(forecast["total_goals"].values()),
                             "away_minus_two_three_way": list(forecast["away_minus_two_jincai"].values())}
            for domain, values in probabilities.items():
                for actual, expected in zip(values, independent[domain]):
                    compare("grid_probability", actual, expected)
            for actual, expected in zip(row["probabilities"], independent["one_x_two"]):
                compare("grid_probability", actual, expected)
            for actual, expected in zip(forecast["expected_goals"].values(), independent["means"]):
                compare("finite_grid_mean", actual, expected)
            for key in ("over_2_5", "both_score"):
                compare("extra_grid_event", forecast[key], independent[key])
            h, a = row["actual_score"]
            labels = {"one_x_two": 0 if h > a else 2 if h < a else 1,
                      "total_goals_0_1_2_3_4plus": min(h+a, 4),
                      "away_minus_two_three_way": 0 if a-h >= 3 else 1 if a-h == 2 else 2}
            metrics[arm][row["match_id"]] = {domain: categorical_metrics(independent[domain], labels[domain])
                                             for domain in labels}
            metrics[arm][row["match_id"]]["exact_score"] = {"mean_nll": nll}
    fold_summaries.append({"cutoff": cutoff, "training_n": len(training), "target_n": len(expected_target),
                           "scales": scales})
assert scored == set(expected_ids)


def check_summary(summary, ids):
    assert summary["n"] == len(ids)
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for domain in DOMAINS:
            for field in metrics[arm][ids[0]][domain]:
                value = math.fsum(metrics[arm][key][domain][field] for key in ids)/len(ids)
                compare("aggregated_scores", summary["scores"][arm][domain][field], value)
                if field in ("log_loss", "mean_nll"):
                    means[arm][domain] = value
    for method in ARMS[1:]:
        for domain in DOMAINS:
            delta = means[method][domain]-means["original"][domain]
            compare("aggregated_differences", summary["mean_differences"][method+"_minus_original"][domain], delta)


check_summary(result["pooled"], expected_ids)
for fold in result["folds"]:
    check_summary(fold["analysis"], [key for key in expected_ids if fold["cutoff"] <= source_rows[key]["date"] < fold["end_exclusive"]])
for season, analysis in result["by_season"].items():
    ids = [key for key in expected_ids if (source_rows[key]["date"] < "2025-08-01") == (season == "2024-25_partial")]
    check_summary(analysis, ids)
for grouping, field, transform in (("by_league", "competition", lambda x: x), ("by_month", "date", lambda x: x[:7])):
    for group, analysis in result["pooled"][grouping].items():
        ids = [key for key in expected_ids if transform(source_rows[key][field]) == group]
        check_summary(analysis, ids)
for arm in ARMS:
    for row in all_tracks[arm]:
        compare("combined_forecast_nll", row["exact_score_nll"], metrics[arm][row["match_id"]]["exact_score"]["mean_nll"])
with (OUT / "逐场前推概率.csv").open(encoding="utf-8-sig", newline="") as stream:
    csv_rows = list(csv.DictReader(stream))
assert [row["match_id"] for row in csv_rows] == expected_ids
for row in csv_rows:
    for arm in ARMS:
        for domain in DOMAINS:
            field = "mean_nll" if domain == "exact_score" else "log_loss"
            compare("csv_score", float(row[f"{arm}_{domain}_nll"]), metrics[arm][row["match_id"]][domain][field])
audit = {"status": "passed", "independent_numeric_replay_not_additional_validation_sample": True,
         "source_directory": str(OUT), "source_rows": len(source_rows), "source_snapshots": source_snapshots,
         "scored_unique_matches": len(expected_ids), "arms": list(ARMS), "closed_form_fits_checked": 12,
         "forecasts_checked": 3*len(expected_ids), "maximum_absolute_errors": ERRORS, "folds": fold_summaries,
         "method": "stdlib only: coefficient replay, math.fsum, math.lgamma full support NLL, recursive Poisson grid"}
audit_path = ROOT / "work/独立进球率尺度前推审核_20261024.json"
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"output": str(audit_path), **audit}, ensure_ascii=False), flush=True)
