"""Standalone arithmetic audit of the source-league schedule-count research.

This script imports no production football_model module and never fits a model.
Raw CSV dates are read independently; every historical row's [d-14,d) counts,
explicit feature equations, rates and penalized Poisson gradient are recomputed.
The adjacent standalone rate/goal audit scripts are loaded by file path solely
for independent probability recurrence, scoring and calendar-block arithmetic.
Gradient magnitudes are measurements, not a new optimizer-acceptance threshold.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location("standalone_rate_audit_for_schedule",
    Path(__file__).with_name("independent_rate_calibration_audit.py"))
probability = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probability)
independent = probability.independent

ARMS = ("original", "schedule")
COMPS = independent.COMPS
PERIODS = independent.PERIODS
SELF_FEATURE = "context:self_fixture_count_prev14d"
OPPONENT_FEATURE = "context:opponent_fixture_count_prev14d"
DOMAINS = (*probability.DOMAIN_NAMES, "exact_score")
SPECIAL = {
    ("D1", "2024-12-14", "Union Berlin", "Bochum"): "administratively_awarded_result",
    ("I1", "2024-04-25", "Udinese", "Roma"): "resumed_partially_played_match",
    ("SP1", "2023-12-11", "Granada", "Ath Bilbao"): "resumed_partially_played_match",
}


def read_sources(directories):
    """Read all four raw CSV seasons without the production parser or model."""
    records = []
    seen = set()
    for season, directory in sorted(directories.items()):
        manifest = Path(directory)/"下载来源.json"
        urls = {}
        if manifest.exists():
            meta = independent.read_json(manifest)
            urls = {Path(item["path"]).name: item["source"] for item in meta.get("files", [])}
        for competition in COMPS:
            path = Path(directory)/f"{competition}.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    if row["Div"] != competition:
                        raise ValueError("Raw division differs from declared CSV filename")
                    day = datetime.strptime(row["Date"], "%d/%m/%Y" if len(row["Date"]) == 10 else "%d/%m/%y").date()
                    home, away = row["HomeTeam"].strip(), row["AwayTeam"].strip()
                    if not home or not away or home == away:
                        raise ValueError("Raw CSV requires distinct nonempty teams")
                    goals = tuple(int(row[k]) for k in ("FTHG", "FTAG"))
                    if min(goals) < 0:
                        raise ValueError("Raw CSV requires nonnegative goals")
                    result = "H" if goals[0] > goals[1] else "A" if goals[0] < goals[1] else "D"
                    if row["FTR"] != result:
                        raise ValueError("Raw goals and outcome disagree")
                    mid = f"football-data:{competition}:{season}:{home}|{away}"
                    if mid in seen:
                        raise ValueError("Duplicate raw source match identifier")
                    seen.add(mid)
                    markets = {}
                    for name, columns in (("avg_preclosing", ("AvgH", "AvgD", "AvgA")),
                                          ("avg_closing", ("AvgCH", "AvgCD", "AvgCA")),
                                          ("b365_preclosing", ("B365H", "B365D", "B365A")),
                                          ("b365_closing", ("B365CH", "B365CD", "B365CA"))):
                        values = [row.get(key, "").strip() for key in columns]
                        if all(values):
                            prices = [float(value) for value in values]
                            if all(math.isfinite(value) and value > 1 for value in prices):
                                markets[name] = prices
                    season_code = season[2:4]+season[-2:]
                    records.append({"match_id": mid, "competition": competition,
                        "date": day.isoformat(), "home": home, "away": away,
                        "actual_score": list(goals), "actual_outcome": (0 if result == "H" else 2 if result == "A" else 1),
                        "goal_fit_exclusion": SPECIAL.get((competition, day.isoformat(), home, away)),
                        "source_path": str(path), "market_prices": markets,
                        "source_url": urls.get(path.name, f"https://football-data.co.uk/mmz4281/{season_code}/{competition}.csv")})
    return sorted(records, key=lambda r: (r["date"], r["competition"], r["match_id"]))


def manual_context(records, competition, home, away, day):
    """Brute-force date-only counts; deliberately not the production index."""
    if type(day) is not date or home == away:
        raise ValueError("Use an explicit match calendar date and distinct teams")
    league = [r for r in records if r["competition"] == competition]
    if not league:
        raise ValueError("No source date coverage for competition")
    start = day-timedelta(days=14)
    source_start = min(date.fromisoformat(r["date"]) for r in league)
    sides = {}
    for side, team in (("home", home), ("away", away)):
        earlier = [r for r in league if r["date"] < day.isoformat() and team in (r["home"], r["away"])]
        window = [r for r in earlier if r["date"] >= start.isoformat()]
        last = max((r["date"] for r in earlier), default=None)
        sides[side] = {"count": len(window), "window_match_ids": [r["match_id"] for r in
            sorted(window, key=lambda r: (r["date"], r["match_id"]))],
            "last_prior_date": last,
            "days_since_prior": (day-date.fromisoformat(last)).days if last else None,
            "has_prior_source_record": bool(earlier)}
    return {"date": day.isoformat(), "window_start_inclusive": start.isoformat(),
        "source_first_date": source_start.isoformat(), "left_window_truncated": start < source_start,
        "home": sides["home"], "away": sides["away"]}


def export_context(row, context, records_by_id):
    """Express independent date counts in the saved research context fields."""
    result = {"competition": row["competition"], "on_date": row["date"], "window_days": 14,
              "scope": "source_recorded_same_competition_fixture_dates"}
    for side in ("home", "away"):
        values = context[side]
        result[side] = {"team": f"football-data:{row['competition']}:{row[side]}",
            "source_fixture_count_prev14d": values["count"], "previous_date": values["last_prior_date"],
            "gap_days": values["days_since_prior"], "previous_14d_ids": values["window_match_ids"],
            "special_previous_14d_ids": [mid for mid in values["window_match_ids"]
                if records_by_id[mid]["goal_fit_exclusion"] is not None],
            "window_left_censored": context["left_window_truncated"],
            "has_previous_source_record": values["has_prior_source_record"]}
    return result


def explicit_features(row, context=None):
    """Handwritten two goal-row equations, including provider-specific identity."""
    comp = row["competition"]
    home = f"football-data:{comp}:{row['home']}"
    away = f"football-data:{comp}:{row['away']}"
    features = [
        {"attack:"+home: 1., "defense:"+away: -1., "competition:"+comp: 1., "home:"+comp: 1.},
        {"attack:"+away: 1., "defense:"+home: -1., "competition:"+comp: 1.},
    ]
    if context is not None:
        counts = (context["home"]["count"], context["away"]["count"])
        for side, feature in enumerate(features):
            feature[SELF_FEATURE] = counts[side]
            feature[OPPONENT_FEATURE] = counts[1-side]
    return features


def replay_rates(parameters, row, context=None):
    coefficients = parameters["coefficients"]
    result = tuple(math.exp(parameters["intercept"]+math.fsum(
        coefficients[key]*value for key, value in terms.items()))
        for terms in explicit_features(row, context))
    if not all(math.isfinite(value) and value > 0 for value in result):
        raise ValueError("Nonfinite or nonpositive independently replayed rate")
    return result


def penalized_objective_gradient(intercept, coefficients, examples, alpha):
    """Normalized weighted Poisson objective and analytic gradient, no optimizer.

    examples holds (feature dictionary, integer goal, positive raw weight).
    Q=sum_i w_i*(exp(eta_i)-y_i*eta_i)/sum_i w_i + alpha/2*||beta||².
    This differs from half Poisson deviance only by a parameter-independent
    y-dependent constant. The intercept is NOT part of the L2 penalty.
    """
    if not examples or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Need nonempty weighted examples and a positive L2 strength")
    names = list(coefficients)
    positions = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(examples), len(names)))
    y, weights = [], []
    for i, (features, goals, weight) in enumerate(examples):
        if type(goals) is not int or goals < 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Need nonnegative integer targets and positive finite weights")
        for name, value in features.items():
            x[i, positions[name]] = value
        y.append(goals)
        weights.append(weight)
    y, weights, beta = np.array(y), np.array(weights), np.array([coefficients[k] for k in names])
    normalizer = float(weights.sum())
    normalized = weights/normalizer
    eta = intercept+x@beta
    rates = np.exp(eta)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Nonfinite or nonpositive training rates")
    residual = normalized*(rates-y)
    gradient = x.T@residual+alpha*beta
    intercept_gradient = float(residual.sum())
    penalty = float(alpha*.5*np.square(beta).sum())
    data_part = float(normalized@(rates-y*eta))
    full_gradient = dict(zip(names, map(float, gradient)))
    full_gradient["(intercept)"] = intercept_gradient
    largest = max(full_gradient, key=lambda key: abs(full_gradient[key]))
    return {"objective_parameter_dependent": data_part+penalty,
        "weighted_data_objective": data_part, "l2_penalty": penalty,
        "raw_weight_sum_goal_rows": normalizer, "n_goal_rows": len(examples),
        "max_absolute_gradient": abs(full_gradient[largest]), "largest_gradient_coordinate": largest,
        "intercept_gradient": intercept_gradient, "gradient": full_gradient}


def training_gradient(parameters, records_by_id, contexts):
    """Use saved training IDs, independent per-row contexts and explicit weights."""
    before = date.fromisoformat(parameters["before_date"])
    examples = []
    for mid in parameters["reference_match_ids"]:
        row = records_by_id[mid]
        day = date.fromisoformat(row["date"])
        if day >= before or row["goal_fit_exclusion"] is not None:
            raise ValueError("Saved training ID is not an eligible strictly earlier row")
        weight = math.exp(-math.log(2)*(before-day).days/parameters["half_life_days"])
        context = contexts[mid] if SELF_FEATURE in parameters["coefficients"] else None
        examples.extend((features, goals, weight)
            for features, goals in zip(explicit_features(row, context), row["actual_score"]))
    return penalized_objective_gradient(parameters["intercept"], parameters["coefficients"], examples,
                                         parameters["regularization"])


def expected_blocks(records, start, end):
    """Independent identity/date coverage, with no target-score selection."""
    blocks, skips = {}, {}
    cutoff = date.fromisoformat(start)
    stop_date = date.fromisoformat(end)
    while cutoff < stop_date:
        stop = min(stop_date, cutoff+timedelta(days=28))
        history = [r for r in records if r["date"] < cutoff.isoformat() and r["goal_fit_exclusion"] is None]
        known = {team for r in history for team in (r["home"], r["away"])}
        targets = []
        for row in records:
            if not cutoff.isoformat() <= row["date"] < stop.isoformat():
                continue
            if row["goal_fit_exclusion"] is not None:
                skips[row["match_id"]] = row["goal_fit_exclusion"]
            elif {row["home"], row["away"]}-known:
                skips[row["match_id"]] = "unseen_team_no_estimated_prior"
            else:
                targets.append(row)
        if targets:
            blocks[cutoff.isoformat()] = (history, targets)
        cutoff = stop
    return blocks, skips


def audit_trial(trial, records, contexts, arm, audit):
    """Replay every fit's provenance, every forecast and every new gradient."""
    source = {r["match_id"]: r for r in records}
    blocks, skips = expected_blocks(records, trial["start"], trial["end_exclusive"])
    audit.check(trial["failures"] == [], "no_recorded_trial_failures", arm)
    audit.check(trial["update_interval_days"] == 28, "fixed_28_day_interval", arm)
    audit.check({r["match_id"]: r["reason"] for r in trial["skipped"]} == skips,
                "all_exclusions_and_unknown_team_skips", arm)
    audit.check(len(trial["skipped"]) == len(skips), "unique_skipped_ids", arm)
    fit_by_cutoff = {p["before_date"]: p for p in trial["fits"]}
    audit.check(set(fit_by_cutoff) == set(blocks), "all_nonempty_anchored_fit_cutoffs", arm)
    audit.check(len(fit_by_cutoff) == len(trial["fits"]), "unique_fit_cutoffs", arm)
    gradients = []
    for cutoff, fit in fit_by_cutoff.items():
        history, targets = blocks[cutoff]
        comp = targets[0]["competition"]
        known = sorted({f"football-data:{comp}:{team}" for row in history for team in (row["home"], row["away"])})
        expected = {"before_date": cutoff, "training_min_date": history[0]["date"],
            "training_max_date": history[-1]["date"], "n_matches": len(history),
            "reference_match_ids": [r["match_id"] for r in history], "teams": known, "competitions": [comp],
            "regularization": .01, "half_life_days": 180, "basis": "team_marginal", "rho": 0,
            "availability_convention": "strictly_earlier_calendar_date_results_assumed_available"}
        expected["model_type"] = ("date_level_source_schedule_regularized_poisson" if arm == "schedule"
                                  else "date_level_regularized_poisson")
        if arm == "schedule":
            expected["schedule_feature_definition"] = {
                "scope": "source_recorded_same_competition_fixture_dates",
                "feature_names": [SELF_FEATURE, OPPONENT_FEATURE], "window_days": 14,
                "window": "[match_date-14_days,match_date)", "unit": "source_fixture_records",
                "numeric_transform": "none", "includes_goal_fit_excluded_source_records": True,
                "same_date_records_included": False, "result_or_market_fields_read_by_date_index": False,
                "date_context_updates_at_each_prediction_date": True,
                "coefficients_fixed_between_explicit_fits": True, "rest_gap_used_as_model_feature": False,
                "coverage_metadata_used_as_model_feature": False,
                "missing_other_competitions_or_sources_imputed": False,
                "zero_count_meaning": "no_same_competition_source_fixture_records_in_window",
                "special_date_semantics": "original_source_dates_retained_not_verified_full_match_load",
                "not_measured": "complete_workload_true_rest_or_fixture_publication_availability"}
        audit.tree(expected, fit, "fit_source_and_training_metadata", arm+"."+cutoff)
        names = {kind+":"+team for team in known for kind in ("attack", "defense")}
        names |= {"competition:"+comp, "home:"+comp}
        if arm == "schedule":
            names |= {SELF_FEATURE, OPPONENT_FEATURE}
        audit.check(set(fit["coefficients"]) == names, "explicit_trained_feature_names", [arm, cutoff])
        result = training_gradient(fit, source, contexts)
        gradients.append({"arm": arm, "competition": comp, "before_date": cutoff,
            "optimizer_iterations": fit["optimizer_iterations"],
            "optimizer_stopping_reason_saved": False, **result})
    predictions, rates, forecasts, stats = {}, {}, {}, {}
    for row in trial["predictions"]:
        mid = row["match_id"]
        audit.check(mid not in predictions, "unique_prediction_ids", [arm, mid])
        raw = source[mid]
        day = date.fromisoformat(raw["date"])
        start = date.fromisoformat(trial["start"])
        cutoff = (start+timedelta(days=((day-start).days//28)*28)).isoformat()
        fit = fit_by_cutoff[cutoff]
        audit.tree({**{key: raw[key] for key in ("match_id", "date", "competition", "home", "away",
                    "actual_score", "actual_outcome")}, "fit_before_date": cutoff,
                    "training_max_date": fit["training_max_date"], "training_matches": fit["n_matches"]},
                    row, "prediction_source_identity_and_cutoff", arm+"."+mid)
        if "market_prices" in raw:
            audit.tree(raw["market_prices"], row["market_prices"], "raw_source_market_prices", arm+"."+mid)
            audit.check(set(raw["market_prices"]) == set(row["market_prices"]), "raw_market_keys", mid)
            for name, prices in raw["market_prices"].items():
                inverse = 1/np.asarray(prices)
                audit.tree((inverse/inverse.sum()).tolist(), row["market_probabilities"][name], "independent_devig", mid)
            audit.check(raw["source_url"] == row["source_url"], "raw_source_url", mid)
        context = contexts[mid] if arm == "schedule" else None
        rate = replay_rates(fit, raw, context)
        forecast = probability.poisson_forecast(rate)
        audit.tree(forecast, row["forecast"], "complete_forecast_from_explicit_rates", arm+"."+mid)
        audit.tree([forecast["one_x_two"][key] for key in probability.CLASSES], row["probabilities"],
                   "one_x_two_vector", arm+"."+mid)
        if arm == "schedule":
            audit.tree(list(rate), row["rates"], "saved_true_rates", mid)
            audit.tree(export_context(raw, context, source), row["schedule_context"], "saved_date_context", mid)
            audit.check(row["context_before_date"] == row["date"], "context_updates_at_match_date", mid)
        predictions[mid], rates[mid], forecasts[mid] = row, rate, forecast
        stats[mid] = independent.statistics(raw["actual_score"], rate)
    expected_ids = {r["match_id"] for history, targets in blocks.values() for r in targets}
    audit.check(set(predictions) == expected_ids, "complete_prediction_coverage", arm)
    ordered = sorted(predictions, key=lambda mid: (predictions[mid]["date"], mid))
    if ordered:
        p, y = probability.domain_arrays([predictions[mid] for mid in ordered], [forecasts[mid] for mid in ordered])["one_x_two"]
        audit.tree(probability.categorical_score(p, y), trial["metrics"]["model"], "trial_one_x_two_scores", arm)
    return predictions, rates, forecasts, stats, gradients


def cohort(rows, forecasts, stats, indices):
    indices = list(indices)
    local = [rows[i] for i in indices]
    scores = {}
    for arm in ARMS:
        arrays = probability.domain_arrays(local, [forecasts[arm][i] for i in indices])
        scores[arm] = {domain: probability.categorical_score(p, y) for domain, (p, y) in arrays.items()}
        scores[arm]["exact_score"] = {"n": len(indices), "mean_nll": float(np.mean([
            stats[arm][i]["joint_score_nll"] for i in indices]))}
    return {"n": len(indices), "scores": scores,
        "goal_summaries": {arm: independent.summarize([stats[arm][i] for i in indices]) for arm in ARMS},
        "mean_differences": {domain: scores["schedule"][domain]["mean_nll" if domain == "exact_score" else "log_loss"]
            - scores["original"][domain]["mean_nll" if domain == "exact_score" else "log_loss"] for domain in DOMAINS}}


def describe(values):
    v = np.asarray(values, dtype=float)
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "min": float(v.min()), "q25": float(np.quantile(v, .25)), "median": float(np.median(v)),
        "q75": float(np.quantile(v, .75)), "max": float(v.max())}


def audit_period(rows, forecasts, stats, contexts, saved, audit):
    n = len(rows)
    audit.tree(cohort(rows, forecasts, stats, range(n)), saved, "all_scores_and_diagnostics")
    ids = [row["match_id"] for row in rows]
    audit.tree({"match_ids": ids, "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "coverage": {"common_match_ids": ids, "counts": {arm: n for arm in ARMS},
                     "outside_common": {arm: [] for arm in ARMS}}}, saved, "complete_matched_cohort")
    arrays, loss = {}, {}
    for arm in ARMS:
        arrays[arm] = probability.domain_arrays(rows, forecasts[arm])
        loss[arm] = {domain: -np.log(np.maximum(p[np.arange(n), y], np.finfo(float).tiny))
                     for domain, (p, y) in arrays[arm].items()}
        loss[arm]["exact_score"] = np.array([s["joint_score_nll"] for s in stats[arm]])
    features = {"schedule_minus_original_"+domain: loss["schedule"][domain]-loss["original"][domain] for domain in DOMAINS}
    for domain in DOMAINS:
        audit.tree(describe(features["schedule_minus_original_"+domain]), saved["loss_difference_descriptives"][domain], "loss_distribution", domain)
    for arm in ARMS:
        for field in independent.MOMENT_FIELDS:
            features[arm+"_"+field] = np.array([s[field] for s in stats[arm]])
        p, y = arrays[arm]["one_x_two"]
        audit.tree(probability.probability_calibration(p, y), saved["calibration"][arm], "probability_calibration", arm)
        for c, side in enumerate(probability.CLASSES):
            features[arm+"_"+side+"_probability_minus_frequency"] = p[:, c]-(y == c)
    for weeks in (2, 4, 8):
        probability.audit_bootstrap([r["date"] for r in rows], features, saved["bootstrap"][str(weeks)], audit, str(weeks))
    for key, labels in (("by_league", [r["competition"] for r in rows]),
                        ("by_month", [r["date"][:7] for r in rows]),
                        ("by_observed_count_comparison", ["equal" if c["home"]["count"] == c["away"]["count"] else
                            "home_more" if c["home"]["count"] > c["away"]["count"] else "away_more" for c in contexts])):
        audit.check(set(saved[key]) == set(labels), "descriptive_group_keys", key)
        for label in sorted(set(labels)):
            audit.tree(cohort(rows, forecasts, stats, [i for i, value in enumerate(labels) if value == label]),
                       saved[key][label], "descriptive_group", key+"."+label)
    for side in ("home", "away"):
        counts = [c[side]["count"] for c in contexts]
        expected = {"source_fixture_count_prev14d": describe(counts),
            "left_censored_windows": sum(c["left_window_truncated"] for c in contexts),
            "without_previous_source_record": sum(not c[side]["has_prior_source_record"] for c in contexts),
            "windows_containing_special_source_records": sum(bool(row["schedule_context"][side]["special_previous_14d_ids"]) for row in rows),
            "count_frequencies": {str(k): counts.count(k) for k in sorted(set(counts))}}
        audit.tree(expected, saved["date_feature_descriptives"][side], "date_feature_coverage", side)
    for name, stored in saved["markets"].items():
        indices = [i for i, row in enumerate(rows) if name in row["market_prices"]]
        missing = [row["match_id"] for row in rows if name not in row["market_prices"]]
        audit.tree({"n": len(indices), "missing_ids": missing}, stored, "market_coverage", name)
        if not indices:
            continue
        q = np.array([1/np.array(rows[i]["market_prices"][name]) for i in indices])
        q /= q.sum(axis=1, keepdims=True)
        y = arrays["original"]["one_x_two"][1][indices]
        ml = -np.log(q[np.arange(len(y)), y])
        delta = {arm: loss[arm]["one_x_two"][indices]-ml for arm in ARMS}
        boot = independent.block_bootstrap([rows[i]["date"] for i in indices], np.column_stack(list(delta.values())),
            4, stored["bootstrap_metadata"]["repeats"], stored["bootstrap_metadata"]["seed"])
        means, ci = boot.pop("mean"), boot.pop("interval_95")
        expected = {"match_ids": [ids[i] for i in indices], "market_scores": probability.categorical_score(q, y),
            "arm_scores_on_same_matches": {arm: probability.categorical_score(arrays[arm]["one_x_two"][0][indices], y) for arm in ARMS},
            "differences": {arm: {"mean": means[i], "interval_95": ci[i] if ci is not None else None} for i, arm in enumerate(ARMS)},
            "bootstrap_metadata": boot}
        audit.tree(expected, stored, "matched_market_comparison", name)
    return loss


def audit_csv(path, rows, stats, loss, audit):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    audit.check(len(saved) == len(rows), "csv_complete_row_count", str(path))
    audit.check(len(saved[0]) == 79, "csv_all_columns", str(path))
    for i, (row, actual) in enumerate(zip(rows, saved)):
        for field in ("match_id", "date", "competition", "fit_before_date"):
            audit.check(row[field] == actual[field], "csv_identity", row["match_id"])
        for side in ("home", "away"):
            for field in ("source_fixture_count_prev14d", "previous_date", "gap_days", "window_left_censored", "has_previous_source_record"):
                expected = row["schedule_context"][side][field]
                audit.check(actual[side+"_"+field] == (str(expected) if expected is not None else ""), "csv_date_context", row["match_id"])
        for field, value in (("training_matches", row["training_matches"]), ("actual_home_goals", row["actual_score"][0]),
                             ("actual_away_goals", row["actual_score"][1])):
            audit.numeric(value, actual[field], "csv_source_numbers", row["match_id"])
        for arm in ARMS:
            for field in independent.STAT_FIELDS:
                audit.numeric(stats[arm][i][field], actual[arm+"_"+field], "csv_diagnostics", row["match_id"])
            for domain in DOMAINS:
                audit.numeric(loss[arm][domain][i], actual[arm+"_"+domain+"_loss"], "csv_losses", row["match_id"])
            rate = (stats[arm][i]["home_rate"], stats[arm][i]["away_rate"])
            forecast = probability.poisson_forecast(rate)
            for side in probability.CLASSES:
                audit.numeric(forecast["one_x_two"][side], actual[arm+"_p_"+side], "csv_probabilities", row["match_id"])


def run(output):
    output = Path(output)
    saved = independent.read_json(output/"赛程日期特征研究结果.json")
    all_records = read_sources(saved["source_directories"])
    auditor = independent.Auditor()
    auditor.tree(saved["settings"], independent.read_json(output/"实验设置.json"), "experiment_settings")
    auditor.check(saved["model_adopted"] is False and saved["production_api_changed"] is False, "production_unchanged")
    gradients, counts = [], {"new_schedule_fits": 0, "original_snapshots_reused": 0,
                            "new_schedule_predictions": 0, "original_predictions_reused": 0, "unique_matches": 0}
    for period in PERIODS:
        first_year = int(period[:4])
        records = [r for r in all_records if int(r["match_id"].split(":")[2][:4]) <= first_year]
        # Compute all historical feature rows independently, not merely targets.
        contexts = {r["match_id"]: manual_context(records, r["competition"], r["home"], r["away"], date.fromisoformat(r["date"])) for r in records}
        combined = {arm: {} for arm in ARMS}
        rate_map, forecast_map, stat_map = ({arm: {} for arm in ARMS} for _ in range(3))
        for comp in COMPS:
            local = [r for r in records if r["competition"] == comp]
            for arm in ARMS:
                trial = independent.read_json(output/period/f"{comp}_{arm}.json")
                pred, rates, forecasts, stats, grad = audit_trial(trial, local, contexts, arm, auditor)
                combined[arm].update(pred)
                rate_map[arm].update(rates)
                forecast_map[arm].update(forecasts)
                stat_map[arm].update(stats)
                gradients.extend(grad)
                counts["new_schedule_fits" if arm == "schedule" else "original_snapshots_reused"] += len(trial["fits"])
                counts["new_schedule_predictions" if arm == "schedule" else "original_predictions_reused"] += len(pred)
        auditor.check(set(combined["original"]) == set(combined["schedule"]), "two_arm_exact_same_matches", period)
        ids = sorted(combined["original"], key=lambda k: (combined["original"][k]["date"], k))
        rows = []
        for mid in ids:
            row = dict(combined["original"][mid])
            row["schedule_context"] = export_context(row, contexts[mid], {r["match_id"]: r for r in records})
            rows.append(row)
        forecasts = {arm: [forecast_map[arm][mid] for mid in ids] for arm in ARMS}
        stats = {arm: [stat_map[arm][mid] for mid in ids] for arm in ARMS}
        counts["unique_matches"] += len(ids)
        auditor.tree(saved["periods"][period], independent.read_json(output/period/"同场汇总.json"), "period_saved_summary", period)
        loss = audit_period(rows, forecasts, stats, [contexts[mid] for mid in ids], saved["periods"][period], auditor)
        audit_csv(output/period/"逐场概率与日期特征.csv", rows, stats, loss, auditor)
    auditor.tree(counts, saved["counts"], "execution_counts")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "passed" if not auditor.errors else "failed", "error_count": len(auditor.errors), "errors": auditor.errors,
        "arithmetic_check_count": sum(auditor.checks.values()), "check_counts": dict(auditor.checks),
        "maximum_absolute_errors": auditor.maximum_absolute_errors,
        "training_gradients": gradients,
        "gradient_ranges": {arm: {"n_snapshots": sum(g["arm"] == arm for g in gradients),
            "min_maxabs": min(g["max_absolute_gradient"] for g in gradients if g["arm"] == arm),
            "max_maxabs": max(g["max_absolute_gradient"] for g in gradients if g["arm"] == arm)} for arm in ARMS},
        "new_fits_in_audit": 0, "unique_matches": counts["unique_matches"],
        "implementation": {"production_model_imported": False, "source_context": "brute_force_per_historical_calendar_date",
            "probability_and_bootstrap_reused_from": "adjacent independent_rate_calibration_audit.py and independent_goal_rate_audit.py",
            "gradient_objective": "weighted_mean(exp(eta)-y*eta)+alpha/2*sum(beta^2); intercept unpenalized",
            "new_gradient_threshold_or_gate": False},
        "not_covered": ["No optimizer re-fit or optimality proof", "No independent provider truth or historical publication availability verification",
            "No complete workload, injury, lineup or motivation reconstruction", "No fresh blind test or profitability claim",
            "Conditional intervals exclude training/selection uncertainty and cross-season structural change"]}
    destination = output/"独立赛程特征审计.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "audit_status": report["audit_status"], "error_count": report["error_count"],
                      "arithmetic_check_count": report["arithmetic_check_count"], "gradient_ranges": report["gradient_ranges"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    raise SystemExit(0 if result["audit_status"] == "passed" else 1)
