"""Standalone arithmetic audit of coherent Poisson rate-scale research.

No production football_model module is imported. The adjacent standalone
independent_goal_rate_audit.py is loaded by file path to reuse its explicit
coefficient replay, Bessel/lgamma diagnostics, summaries and joint week blocks.
New offset-loss differences are checked as (c-1)*lambda-y*log(c); no base model
is trained, no probability output is repaired and source archives are read only.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import poisson


_SPEC = importlib.util.spec_from_file_location("standalone_goal_rate_audit_for_scaling",
    Path(__file__).with_name("independent_goal_rate_audit.py"))
independent = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(independent)

ARMS = ("original", "shared_scale", "side_scale")
METHODS = ARMS[1:]
CLASSES = ("home", "draw", "away")
NLL_FIELDS = independent.NLL_FIELDS
STAT_FIELDS = independent.STAT_FIELDS
DOMAIN_NAMES = ("one_x_two", "total_goals_0_1_2_3_4plus", "away_minus_two_three_way")


def load_source(parent, season, audit):
    combined, snapshots = {}, 0
    for comp in independent.COMPS:
        path = Path(parent)/season/f"{comp}_days_28.json"
        trial = independent.read_json(path)
        audit.check(trial["failures"] == [] and trial["update_interval_days"] == 28,
                    "source_complete_fixed_frequency", [season, comp])
        audit.tree({"id": "original_ridge_poisson_h180", "family": "original", "half_life_days": 180},
                   trial["config"], "source_original_configuration", str(path))
        models = {p["before_date"]: p for p in trial["fits"]}
        audit.check(len(models) == len(trial["fits"]), "unique_base_cutoffs", [season, comp])
        snapshots += len(models)
        for row in trial["predictions"]:
            mid = row["match_id"]
            audit.check(mid not in combined, "unique_source_prediction_id", mid)
            audit.check(row["competition"] == comp and trial["start"] <= row["date"] < trial["end_exclusive"],
                        "source_period_and_league", mid)
            fit = models[row["fit_before_date"]]
            audit.check(row["training_max_date"] == fit["training_max_date"] < row["fit_before_date"] <= row["date"],
                        "source_strictly_earlier_training_dates", mid)
            audit.check(fit["rho"] == 0 and fit["regularization"] == .01 and fit["half_life_days"] == 180,
                        "source_fixed_base_parameters", mid)
            audit.check(fit["n_matches"] == row["training_matches"], "source_training_match_count", mid)
            combined[mid] = (row, independent.replay_rates(fit, row))
    ids = sorted(combined, key=lambda k: (combined[k][0]["date"], k))
    return [combined[mid][0] for mid in ids], np.array([combined[mid][1] for mid in ids]), snapshots


def closed_form_parameters(rows, rates):
    if not rows or rates.shape != (len(rows), 2):
        raise ValueError("Need aligned nonempty rows and two-column rates")
    goals = [sum(row["actual_score"][side] for row in rows) for side in (0, 1)]
    totals = [math.fsum(rates[:, side]) for side in (0, 1)]
    if min(goals) <= 0 or not all(math.isfinite(x) and x > 0 for x in totals):
        raise ValueError("This audit expects positive finite goal and lambda totals")
    shared = sum(goals)/math.fsum(totals)
    scales = {"shared_scale": np.repeat(shared, 2), "side_scale": np.array(goals)/np.array(totals)}
    return goals, totals, scales


def offset_nll_change(score, rates, scales):
    """Independent algebraic check; factorial terms cancel, no clipping."""
    terms = [(c-1)*rate-y*math.log(c) for y, rate, c in zip(score, rates, scales)]
    if not all(math.isfinite(x) for x in terms):
        raise ValueError("Nonfinite offset-loss difference")
    return dict(zip(NLL_FIELDS, (terms[0], terms[1], math.fsum(terms))))


def poisson_forecast(rates):
    """Independent recurrence grid at the production-declared negligible tail."""
    lh, la = map(float, rates)
    maximum = max(15, int(poisson.ppf(1-1e-12, max(lh, la))))
    marginals = []
    for rate in (lh, la):
        probabilities = [math.exp(-rate)]
        for goal in range(1, maximum+1):
            probabilities.append(probabilities[-1]*rate/goal)
        marginals.append(probabilities)
    matrix = np.outer(*marginals)
    matrix /= matrix.sum()
    h, a = np.indices(matrix.shape)
    p = [float(matrix[h > a].sum()), float(matrix[h == a].sum()), float(matrix[h < a].sum())]
    jc = {"win": float(matrix[a-h >= 3].sum()), "draw": float(matrix[a-h == 2].sum()),
          "lose": float(matrix[a-h <= 1].sum())}
    forecast = {"one_x_two": dict(zip(CLASSES, p)),
        "expected_goals": {"home": float((matrix*h).sum()), "away": float((matrix*a).sum())},
        "total_goals": {**{str(g): float(matrix[h+a == g].sum()) for g in range(4)},
                        "4+": float(matrix[h+a >= 4].sum())},
        "over_2_5": float(matrix[h+a >= 3].sum()), "both_score": float(matrix[(h > 0) & (a > 0)].sum()),
        "away_minus_two_jincai": jc,
        "away_minus_two_asian": {"win": jc["win"], "push": jc["draw"], "lose": jc["lose"]},
        "fair_odds_unbuffered": {"away_win": 1/p[2], "away_minus_two_jincai_win": 1/jc["win"],
            "away_minus_two_asian": (1-jc["draw"])/jc["win"]}}
    return forecast


def domain_arrays(rows, forecasts):
    scores = [row["actual_score"] for row in rows]
    outcomes = [0 if h > a else 1 if h == a else 2 for h, a in scores]
    totals = [min(h+a, 4) for h, a in scores]
    handicap = [0 if a-h >= 3 else 1 if a-h == 2 else 2 for h, a in scores]
    return {
        "one_x_two": (np.array([[r["one_x_two"][k] for k in CLASSES] for r in forecasts]), np.array(outcomes)),
        "total_goals_0_1_2_3_4plus": (np.array([[r["total_goals"][k] for k in ("0", "1", "2", "3", "4+")]
                                                     for r in forecasts]), np.array(totals)),
        "away_minus_two_three_way": (np.array([[r["away_minus_two_jincai"][k] for k in ("win", "draw", "lose")]
                                             for r in forecasts]), np.array(handicap))}


def categorical_score(p, labels):
    truth = np.eye(p.shape[1])[labels]
    nll = -np.log(np.maximum(p[np.arange(len(labels)), labels], np.finfo(float).tiny))
    errors = np.cumsum(p-truth, axis=1)[:, :-1]
    return {"n": len(labels), "log_loss": float(nll.mean()),
        "brier_sum": float(np.square(p-truth).sum(axis=1).mean()),
        "rps": float(np.square(errors).sum(axis=1).mean()/(p.shape[1]-1)),
        "accuracy": float((p.argmax(axis=1) == labels).mean())}


def prepared_predictions(rows, raw_rates, scales, audit):
    rates = {"original": raw_rates, **{method: raw_rates*scales[method] for method in METHODS}}
    forecasts, stats = {}, {}
    for arm in ARMS:
        forecasts[arm], stats[arm] = [], []
        for row, rate in zip(rows, rates[arm]):
            forecast = poisson_forecast(rate)
            forecasts[arm].append(forecast)
            stats[arm].append(independent.statistics(row["actual_score"], rate))
            if arm == "original":
                audit.tree(forecast, row["forecast"], "raw_saved_forecast_replay", row["match_id"])
    offset_changes = {method: [offset_nll_change(row["actual_score"], rate, scales[method])
                     for row, rate in zip(rows, raw_rates)] for method in METHODS}
    return rates, forecasts, stats, offset_changes


def audit_fitted_parameters(saved_parameters, rows, rates, audit):
    goals, totals, scales = closed_form_parameters(rows, rates)
    raw_nll = math.fsum(independent.statistics(row["actual_score"], rate)["joint_score_nll"]
                       for row, rate in zip(rows, rates))
    ids = [row["match_id"] for row in rows]
    audit.check(len(ids) == len(set(ids)), "unique_calibrator_training_ids")
    audit.check(all(row["date"] < "2025-08-01" for row in rows), "training_strictly_before_calibration_cutoff")
    audit.check(set(saved_parameters) == set(METHODS), "both_predeclared_methods")
    for method in METHODS:
        factor = scales[method]
        differences = [offset_nll_change(row["actual_score"], rate, factor)["joint_score_nll"]
                       for row, rate in zip(rows, rates)]
        adjusted_nll = raw_nll+math.fsum(differences)
        residuals = [goals[i]-totals[i]*factor[i] for i in (0, 1)]
        expected = {"model_type": "independent_poisson_rate_scale_calibrator", "basis": "team_marginal", "rho": 0,
            "method": method, "parameters": {"scales": factor.tolist()}, "before_date": "2025-08-01",
            "n_matches": len(rows), "training_min_date": rows[0]["date"], "training_max_date": rows[-1]["date"],
            "training_match_ids": ids, "training_goal_sums": goals, "training_rate_sums": totals,
            "raw_training_analytic_joint_score_nll": {"sum": raw_nll, "mean": raw_nll/len(rows)},
            "calibrated_training_analytic_joint_score_nll": {"sum": adjusted_nll, "mean": adjusted_nll/len(rows)},
            "score_equation_residual_sums": {"home": residuals[0], "away": residuals[1], "shared": math.fsum(residuals)},
            "estimation": {"algorithm": "closed_form", "objective": "equal_match_weight_independent_poisson_joint_score_nll",
                "optimizer": None, "regularization": None, "market_inputs": False}}
        audit.tree(expected, saved_parameters[method], "earlier_closed_form_fit", method)
    return scales


def cohort(rows, forecasts, stats, changes, indices):
    indices = list(indices)
    selected = [rows[i] for i in indices]
    scores = {}
    for arm in ARMS:
        arrays = domain_arrays(selected, [forecasts[arm][i] for i in indices])
        scores[arm] = {domain: categorical_score(p, y) for domain, (p, y) in arrays.items()}
        scores[arm]["exact_score"] = {"n": len(indices), "mean_nll": float(np.mean([
            stats["original"][i]["joint_score_nll"]+(changes[arm][i]["joint_score_nll"] if arm != "original" else 0)
            for i in indices]))}
    differences = {method+"_minus_original": {domain: (
        float(np.mean([changes[method][i]["joint_score_nll"] for i in indices])) if domain == "exact_score" else
        scores[method][domain]["log_loss"]-scores["original"][domain]["log_loss"])
        for domain in (*DOMAIN_NAMES, "exact_score")} for method in METHODS}
    return {"n": len(indices), "scores": scores,
        "goal_summaries": {arm: independent.summarize([stats[arm][i] for i in indices]) for arm in ARMS},
        "mean_differences": differences}


def probability_calibration(p, y):
    truth = np.eye(3)[y]
    classes = {name: {"mean_probability": float(p[:, c].mean()),
        "observed_frequency": float(truth[:, c].mean()),
        "mean_probability_minus_frequency": float((p[:, c]-truth[:, c]).mean()),
        "binary_brier": float(np.square(p[:, c]-truth[:, c]).mean())} for c, name in enumerate(CLASSES)}
    bins = []
    for c, name in enumerate(CLASSES):
        membership = np.minimum(np.floor(p[:, c]*5).astype(int), 4)
        for index in range(5):
            mask = membership == index
            if mask.any():
                bins.append({"outcome": name, "bin": index, "n": int(mask.sum()),
                    "mean_probability": float(p[mask, c].mean()), "observed_frequency": float((y[mask] == c).mean())})
    return {"classes": classes, "equal_width_5_bins": bins}


def per_row_losses(rows, forecasts, stats, changes):
    losses, arrays = {}, {}
    for arm in ARMS:
        arrays[arm] = domain_arrays(rows, forecasts[arm])
        losses[arm] = {domain: -np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny))
                       for domain, (p, y) in arrays[arm].items()}
        raw = np.array([s["joint_score_nll"] for s in stats["original"]])
        losses[arm]["exact_score"] = raw if arm == "original" else raw+np.array([
            delta["joint_score_nll"] for delta in changes[arm]])
    return losses, arrays


def audit_bootstrap(dates, features, saved, audit, label):
    names = list(features)
    matrix = np.column_stack(list(features.values()))
    expected = independent.block_bootstrap(dates, matrix, saved["block_weeks"], saved["repeats"], saved["seed"])
    means, intervals = expected.pop("mean"), expected.pop("interval_95")
    audit.tree(expected, saved, "bootstrap_metadata", label)
    audit.check(set(saved["features"]) == set(names), "joint_bootstrap_features", label)
    for i, name in enumerate(names):
        audit.tree({"mean": means[i], "interval_95": intervals[i] if intervals is not None else None},
                   saved["features"][name], "joint_bootstrap", label+"."+name)


def audit_later(rows, forecasts, stats, changes, saved, audit):
    n = len(rows)
    audit.tree(cohort(rows, forecasts, stats, changes, range(n)), saved, "later_all_scores_and_statistics")
    ids = [r["match_id"] for r in rows]
    audit.tree({"match_ids": ids, "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "coverage": {"common_match_ids": ids, "counts": {arm: n for arm in ARMS},
                     "outside_common": {arm: [] for arm in ARMS}}}, saved, "later_full_same_match_cohort")
    loss, arrays = per_row_losses(rows, forecasts, stats, changes)
    features = {method+"_minus_original_"+domain: (
        np.array([s["joint_score_nll"] for s in changes[method]]) if domain == "exact_score" else
        loss[method][domain]-loss["original"][domain])
        for method in METHODS for domain in (*DOMAIN_NAMES, "exact_score")}
    for arm in ARMS:
        for field in independent.MOMENT_FIELDS:
            features[arm+"_"+field] = np.array([s[field] for s in stats[arm]])
        p, y = arrays[arm]["one_x_two"]
        for c, name in enumerate(CLASSES):
            features[arm+"_"+name+"_probability_minus_frequency"] = p[:, c]-(y == c)
        audit.tree(probability_calibration(p, y), saved["calibration"][arm], "one_x_two_calibration", arm)
    for weeks in (2, 4, 8):
        audit_bootstrap([r["date"] for r in rows], features, saved["bootstrap"][str(weeks)], audit, str(weeks))
    for key, labels in (("by_league", [r["competition"] for r in rows]), ("by_month", [r["date"][:7] for r in rows])):
        audit.check(set(saved[key]) == set(labels), "group_labels", key)
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            audit.tree(cohort(rows, forecasts, stats, changes, indices), saved[key][label], key, label)
    for name in ("avg_preclosing", "avg_closing", "b365_preclosing", "b365_closing"):
        indices = [i for i, row in enumerate(rows) if name in row["market_prices"]]
        missing = [row["match_id"] for row in rows if name not in row["market_prices"]]
        stored = saved["markets"][name]
        audit.tree({"n": len(indices), "missing_ids": missing}, stored, "market_missing_cohort", name)
        if not indices:
            continue
        q = []
        for i in indices:
            inv = 1/np.array(rows[i]["market_prices"][name])
            q.append(inv/inv.sum())
            audit.tree(q[-1].tolist(), rows[i]["market_probabilities"][name], "market_independent_devig", rows[i]["match_id"])
        q = np.array(q)
        y = arrays["original"]["one_x_two"][1][indices]
        market_losses = -np.log(q[np.arange(len(indices)), y])
        differences = {arm: loss[arm]["one_x_two"][indices]-market_losses for arm in ARMS}
        boot = independent.block_bootstrap([rows[i]["date"] for i in indices],
            np.column_stack(list(differences.values())), 4, stored["bootstrap_metadata"]["repeats"], stored["bootstrap_metadata"]["seed"])
        means, ci = boot.pop("mean"), boot.pop("interval_95")
        expected = {"match_ids": [ids[i] for i in indices], "market_scores": categorical_score(q, y),
            "arm_scores_on_same_matches": {arm: categorical_score(arrays[arm]["one_x_two"][0][indices], y) for arm in ARMS},
            "differences": {arm: {"mean": means[j], "interval_95": ci[j] if ci is not None else None}
                for j, arm in enumerate(ARMS)}, "bootstrap_metadata": boot}
        audit.tree(expected, stored, "market_scores_and_paired_intervals", name)
    return loss, arrays


def audit_forecast_export(saved, rows, rates, forecasts, audit):
    audit.check(len(saved) == len(rows), "forecast_export_row_count")
    audit.check([r["match_id"] for r in saved] == [r["match_id"] for r in rows], "forecast_export_identity_order")
    maximum_tails = {arm: 0. for arm in ARMS}
    for i, (row, actual) in enumerate(zip(rows, saved)):
        expected = {key: row[key] for key in ("match_id", "date", "competition", "home", "away",
            "actual_score", "actual_outcome", "source_url")}
        expected.update({"base_fit_before_date": row["fit_before_date"], "calibration_before_date": "2025-08-01",
            "production_api_changed": False, "rates": {arm: rates[arm][i].tolist() for arm in ARMS},
            "research_forecasts": {arm: forecasts[arm][i] for arm in ARMS}, "grid_metadata": {}})
        for arm in ARMS:
            maximum = max(15, int(poisson.ppf(1-1e-12, max(rates[arm][i]))))
            tails = [float(poisson.sf(maximum, r)) for r in rates[arm][i]]
            omitted = tails[0]+tails[1]-tails[0]*tails[1]
            maximum_tails[arm] = max(maximum_tails[arm], omitted)
            expected["grid_metadata"][arm] = {"max_goal_inclusive": maximum,
                "omitted_tail_mass_before_normalization": omitted, "normalized_finite_grid": True, "rho": 0}
        audit.tree(expected, actual, "forecast_export_complete", row["match_id"])
    return {arm: {"max_omitted_tail_mass_before_normalization": value} for arm, value in maximum_tails.items()}


def audit_csv(path, rows, stats, losses, arrays, audit):
    fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals",
              "base_fit_before_date", "calibration_before_date"]
    fields += [arm+"_"+key for arm in ARMS for key in (*STAT_FIELDS, "p_home", "p_draw", "p_away",
               *(domain+"_nll" for domain in (*DOMAIN_NAMES, "exact_score")))]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        audit.check(reader.fieldnames == fields, "csv_102_column_order")
        stored = list(reader)
    audit.check(len(stored) == len(rows), "csv_row_count")
    audit.check([r["match_id"] for r in stored] == [r["match_id"] for r in rows], "csv_identity_order")
    for i, (row, item) in enumerate(zip(rows, stored)):
        expected = {key: row[key] for key in fields[:5]}
        expected.update({"home_goals": row["actual_score"][0], "away_goals": row["actual_score"][1],
                         "base_fit_before_date": row["fit_before_date"], "calibration_before_date": "2025-08-01"})
        for arm in ARMS:
            expected.update({arm+"_"+key: value for key, value in stats[arm][i].items()})
            expected.update({arm+"_p_"+label: arrays[arm]["one_x_two"][0][i, c] for c, label in enumerate(CLASSES)})
            expected.update({arm+"_"+domain+"_nll": losses[arm][domain][i] for domain in (*DOMAIN_NAMES, "exact_score")})
        for key, value in expected.items():
            if isinstance(value, (str, bool)):
                audit.check(value == item.get(key), "csv_identity", [row["match_id"], key])
            else:
                audit.numeric(value, item.get(key), "csv_numeric", [row["match_id"], key])


def run(output, parent):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    result = independent.read_json(output/"进球率调整研究结果.json")
    parameters = independent.read_json(output/"进球率校准参数.json")
    audit = independent.Auditor()
    audit.check(Path(result["parent_directory"]).resolve() == parent, "explicit_parent_directory")
    audit.tree(result["settings"], independent.read_json(output/"实验设置.json"), "settings_file_identity")
    audit.tree(parameters, result["parameters"], "parameter_file_identity")
    audit.check(result["production_api_changed"] is False, "production_unchanged_metadata")
    earlier, earlier_rates, earlier_snapshots = load_source(parent, "2024-25", audit)
    scales = audit_fitted_parameters(parameters, earlier, earlier_rates, audit)
    audit.tree({"n": len(earlier), "match_ids": [r["match_id"] for r in earlier],
        "first_date": earlier[0]["date"], "last_date": earlier[-1]["date"],
        "raw_goal_summary": independent.summarize([independent.statistics(row["actual_score"], rate)
                                                   for row, rate in zip(earlier, earlier_rates)]),
        "calibrated_training_scores_not_temporal_validation": True}, result["earlier"], "earlier_cohort")
    later, raw_rates, later_snapshots = load_source(parent, "2025-26", audit)
    audit.check(not set(r["match_id"] for r in earlier) & set(r["match_id"] for r in later), "no_cross_season_identity_overlap")
    audit.check(all(row["date"] >= "2025-08-01" for row in later), "later_application_after_fitted_cutoff")
    rates, forecasts, stats, changes = prepared_predictions(later, raw_rates, scales, audit)
    losses, arrays = audit_later(later, forecasts, stats, changes, result["later"], audit)
    tails = audit_forecast_export(independent.read_json(output/"较晚研究预测.json"), later, rates, forecasts, audit)
    audit.tree(tails, result["grid_approximation"], "finite_grid_tail_summary")
    audit_csv(output/"较晚逐场概率与进球残差.csv", later, stats, losses, arrays, audit)
    audit.tree({"new_rate_calibrator_fits": 2, "new_scalar_rate_parameters": 3, "new_base_model_fits": 0,
        "source_parameter_snapshots_read": earlier_snapshots+later_snapshots,
        "source_prediction_rows": len(earlier)+len(later), "new_research_forecasts": 2*len(later),
        "reused_later_source_forecasts": len(later)}, result["counts"], "execution_counts")
    for key, count, n in (("earlier", earlier_snapshots, len(earlier)), ("later", later_snapshots, len(later))):
        audit.tree({"parameter_snapshots": count, "saved_prediction_rows": n, "saved_trials": 5},
                   result["source_loading"][key], "source_loading", key)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "passed" if not audit.errors else "failed", "error_count": len(audit.errors),
        "errors": audit.errors, "arithmetic_check_count": sum(audit.checks.values()),
        "check_counts": dict(audit.checks), "maximum_absolute_errors": audit.maximum_absolute_errors,
        "earlier_unique_matches": len(earlier), "later_unique_matches": len(later),
        "source_parameter_snapshots": earlier_snapshots+later_snapshots, "new_base_model_fits_in_audit": 0,
        "independent_closed_form_recomputations": 2,
        "independently_recomputed_scales": {key: value.tolist() for key, value in scales.items()},
        "implementation": {"production_package_imported": False,
            "reused_standalone_file": "model_v2/src/football_model/independent_goal_rate_audit.py",
            "new_checks": "Closed-form fsum goals/lambda scales; direct offset-loss identity; independent Poisson recurrence forecast",
            "scores": "All three categorical domains and exact score; all 24 diagnostics; all groups, market cohorts, JSON and 102 CSV columns",
            "bootstrap": "Joint noncircular 2/4/8 calendar weeks, shared all-arm/metric weights, matched 4-week market comparisons"},
        "not_covered": ["No base optimizer re-fit or optimum proof",
            "No fresh verification of raw vendor truth, historical publication availability, injuries, lineups or odds observation timestamps",
            "Loaded parent trials treated as date-level retrospective data, not a new blind test",
            "Conditional bootstrap excludes calibration/base estimation, selection, structural and cross-season uncertainty",
            "Arithmetic consistency does not establish model superiority, probability truth or profitability"]}
    destination = output/"独立进球率调整审计.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "audit_status": report["audit_status"],
        "error_count": report["error_count"], "arithmetic_check_count": report["arithmetic_check_count"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    args = parser.parse_args()
    audited = run(args.output, args.parent)
    raise SystemExit(0 if audited["audit_status"] == "passed" else 1)

