"""Independent arithmetic audit of saved goal-rate diagnostics.

This executable intentionally imports no football_model code. It replays rates
from explicit attack/defense coefficients and uses lgamma and scaled Bessel
formulas, not the production helper's gammaln/Skellam calculations. The joint
calendar-block implementation is adapted from the earlier independent audit at
outputs/较早校准研究_20261023T070259281026Z/independent_calibration_audit.py.
No model is fitted. Check counts are arithmetic checks, not independent samples.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.special import gammainc, i0e


ARMS = ("days_7", "days_14", "days_28")
COMPS = ("D1", "E0", "F1", "I1", "SP1")
PERIODS = ("2024-25", "2025-26")
STAT_FIELDS = (
    "home_rate", "away_rate", "home_goals", "away_goals",
    "home_residual", "away_residual", "home_pearson_square_minus_one",
    "away_pearson_square_minus_one", "pearson_cross_product",
    "home_zero_probability", "away_zero_probability", "total_four_plus_probability",
    "draw_probability", "zero_zero_probability", "one_one_probability",
    "home_zero_residual", "away_zero_residual", "total_four_plus_residual",
    "draw_residual", "zero_zero_residual", "one_one_residual",
    "home_poisson_nll", "away_poisson_nll", "joint_score_nll",
)
NLL_FIELDS = ("home_poisson_nll", "away_poisson_nll", "joint_score_nll")
MOMENT_FIELDS = ("home_residual", "away_residual", "home_pearson_square_minus_one",
    "away_pearson_square_minus_one", "pearson_cross_product", "home_zero_residual",
    "away_zero_residual", "total_four_plus_residual", "draw_residual",
    "zero_zero_residual", "one_one_residual")
BIN_EDGES = (0., .75, 1., 1.25, 1.5, 2., 3., math.inf)
BIN_NAMES = ("0-0.75", "0.75-1", "1-1.25", "1.25-1.5", "1.5-2", "2-3", "3+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Auditor:
    def __init__(self):
        self.checks = Counter()
        self.errors = []
        self.maximum_absolute_errors = {}

    def check(self, condition, label, context=None):
        self.checks[label] += 1
        if not condition:
            self.errors.append({"check": label, "context": context})

    def numeric(self, expected, actual, label, context=None):
        self.checks[label] += 1
        try:
            a, b = float(expected), float(actual)
            if isinstance(actual, bool) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Need finite non-boolean numeric scalars")
            error = abs(a-b)
            self.maximum_absolute_errors[label] = max(self.maximum_absolute_errors.get(label, 0), error)
            if not math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-10):
                self.errors.append({"check": label, "context": context, "expected": a,
                                    "actual": b, "absolute_error": error})
        except (ValueError, TypeError, OverflowError) as exc:
            self.errors.append({"check": label, "context": context, "error": str(exc)})

    def tree(self, expected, actual, label, context=""):
        """Check all expected fields without depending on production imports."""
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label+".mapping", context)
            if not isinstance(actual, dict):
                return
            self.check(set(expected) <= set(actual), label+".keys", [context, sorted(set(expected)-set(actual))])
            for key in set(expected) & set(actual):
                self.tree(expected[key], actual[key], label, context+"."+str(key))
        elif isinstance(expected, (tuple, list)):
            valid = isinstance(actual, (tuple, list))
            self.check(valid and len(expected) == len(actual), label+".length", context)
            if valid:
                for i, (a, b) in enumerate(zip(expected, actual)):
                    self.tree(a, b, label, context+f"[{i}]")
        elif isinstance(expected, Real) and not isinstance(expected, bool):
            self.numeric(expected, actual, label+".numeric", context)
        else:
            self.check(expected == actual, label+".value", [context, expected, actual])


def replay_rates(parameters, row):
    """Explicit provider keys and signs; do not call the model feature builder."""
    co = parameters["coefficients"]
    comp = row["competition"]
    home = f"football-data:{comp}:{row['home']}"
    away = f"football-data:{comp}:{row['away']}"
    base = parameters["intercept"] + co["competition:"+comp]
    lh = math.exp(base + co["attack:"+home] - co["defense:"+away] + co["home:"+comp])
    la = math.exp(base + co["attack:"+away] - co["defense:"+home])
    if not all(math.isfinite(x) and x > 0 for x in (lh, la)):
        raise ValueError("Nonfinite/nonpositive independently replayed rate")
    return lh, la


def statistics(score, rates):
    """24 scalar statistics computed independently of the project helpers."""
    h, a = score
    lh, la = rates
    if (any(type(g) is not int or g < 0 for g in score)
            or any(not math.isfinite(r) or r <= 0 for r in rates)):
        raise ValueError("Need a nonnegative integer score and positive finite rates")
    rh, ra = h-lh, a-la
    home_zero, away_zero = math.exp(-lh), math.exp(-la)
    four = float(gammainc(4, lh+la))
    sh, sa = math.sqrt(lh), math.sqrt(la)
    draw = math.exp(-(sh-sa)**2) * float(i0e(2*sh*sa))
    zz = math.exp(-(lh+la))
    oo = math.exp(math.log(lh)+math.log(la)-lh-la)
    hnll = lh-h*math.log(lh)+math.lgamma(h+1)
    anll = la-a*math.log(la)+math.lgamma(a+1)
    values = (lh, la, h, a, rh, ra, rh**2/lh-1, ra**2/la-1,
        (rh/math.sqrt(lh))*(ra/math.sqrt(la)), home_zero, away_zero,
        four, draw, zz, oo, int(h == 0)-home_zero, int(a == 0)-away_zero,
        int(h+a >= 4)-four, int(h == a)-draw, int(h == a == 0)-zz,
        int(h == a == 1)-oo, hnll, anll, hnll+anll)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Independent statistic is nonfinite")
    return dict(zip(STAT_FIELDS, values))


def bin_label(rate):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Need a positive finite rate")
    for i, (lo, hi) in enumerate(zip(BIN_EDGES, BIN_EDGES[1:])):
        if lo <= rate < hi:
            return BIN_NAMES[i]
    raise ValueError("Rate did not enter a bin")


def summarize(stats):
    n = len(stats)
    if not n:
        raise ValueError("Use a separate explicit empty-group result")
    matrix = np.array([[r[k] for k in STAT_FIELDS] for r in stats], dtype=float)
    descriptions = {}
    for j, key in enumerate(STAT_FIELDS):
        v = matrix[:, j]
        descriptions[key] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if n > 1 else None,
            "min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))}
    result = {"n": n, "scalar_descriptives": descriptions}
    for side in ("home", "away"):
        goals = [r[side+"_goals"] for r in stats]
        rates = [r[side+"_rate"] for r in stats]
        # Recurrences are independent of the production scipy.stats.poisson.pmf.
        pmfs = []
        for rate in rates:
            q = [math.exp(-rate)]
            for g in (1, 2, 3):
                q.append(q[-1]*rate/g)
            pmfs.append(q+[float(gammainc(4, rate))])
        expected = float(np.sum(rates))
        observed = sum(goals)
        result[side] = {"observed_sum": observed, "expected_sum": expected,
            "residual_per_match": (observed-expected)/n,
            "observed_to_expected_ratio": observed/expected,
            "goal_counts": {label: {"observed": sum(g == i for g in goals) if i < 4 else sum(g >= 4 for g in goals),
                "expected": float(np.array(pmfs)[:, i].sum())}
                for i, label in enumerate(("0", "1", "2", "3", "4+"))}}
    return result


def paired_summary(arm_stats, indices):
    indices = list(indices)
    if not indices:
        return {"n": 0, "summaries": None, "mean_nll_differences": None}
    summaries = {arm: summarize([arm_stats[arm][i] for i in indices]) for arm in ARMS}
    differences = {arm+"_minus_days_28": {field: float(np.mean([
        arm_stats[arm][i][field]-arm_stats["days_28"][i][field] for i in indices]))
        for field in NLL_FIELDS} for arm in ARMS[:2]}
    return {"n": len(indices), "summaries": summaries, "mean_nll_differences": differences}


def block_bootstrap(dates, matrix, length, repeats, seed):
    """Independent grouped-week replay, adapted from the earlier audit script."""
    days = [date.fromisoformat(x) for x in dates]
    monday = min(days)-timedelta(days=min(days).weekday())
    indices = [(d-monday).days//7 for d in days]
    weeks = max(indices)+1
    groups = [np.array([i for i, w in enumerate(indices) if w == week], dtype=int) for week in range(weeks)]
    counts = np.array([len(g) for g in groups])
    totals = np.array([matrix[g].sum(axis=0) for g in groups])
    metadata = {"block_weeks": length, "repeats": repeats, "seed": seed,
        "n_matches": len(days), "n_calendar_weeks": weeks, "n_nonempty_weeks": int((counts > 0).sum()),
        "calendar_week_starts": [(monday+timedelta(weeks=i)).isoformat() for i in range(weeks)],
        "weekly_match_counts": counts.tolist(), "mean": matrix.mean(axis=0).tolist(),
        "valid_repeats": 0, "zero_count_repeats": 0, "interval_95": None}
    if length >= weeks:
        metadata["unavailable_reason"] = "block_length_not_smaller_than_calendar_span"
        return metadata
    rng = np.random.default_rng(seed)
    starts = rng.integers(weeks-length+1, size=(repeats, math.ceil(weeks/length)))
    means = []
    for draw in starts:
        selected = [week for start in draw for week in range(int(start), int(start)+length)][:weeks]
        size = int(counts[selected].sum())
        if size:
            means.append(totals[selected].sum(axis=0)/size)
    metadata["valid_repeats"] = len(means)
    metadata["zero_count_repeats"] = repeats-len(means)
    if len(means) >= 2:
        metadata["interval_95"] = np.percentile(means, [2.5, 97.5], axis=0).T.tolist()
    else:
        metadata["unavailable_reason"] = "fewer_than_two_nonempty_resamples"
    return metadata


def load_period(parent, season, audit):
    tracks = {arm: {} for arm in ARMS}
    snapshots = 0
    for comp in COMPS:
        for arm in ARMS:
            days = int(arm[5:])
            trial = read_json(Path(parent)/season/f"{comp}_{arm}.json")
            audit.check(trial["failures"] == [], "trial_no_recorded_failure", [season, comp, arm])
            audit.tree({"id": "original_ridge_poisson_h180", "family": "original", "half_life_days": 180},
                       trial["config"], "original_config", [season, comp, arm].__str__())
            audit.check(trial["update_interval_days"] == days, "trial_interval", [season, comp, arm])
            models = {model["before_date"]: model for model in trial["fits"]}
            audit.check(len(models) == len(trial["fits"]), "unique_snapshot_cutoffs", [season, comp, arm])
            snapshots += len(models)
            for row in trial["predictions"]:
                mid = row["match_id"]
                audit.check(mid not in tracks[arm], "unique_trial_prediction_ids", [season, arm, mid])
                audit.check(row["competition"] == comp and trial["start"] <= row["date"] < trial["end_exclusive"],
                            "source_period_and_league", mid)
                model = models[row["fit_before_date"]]
                audit.check(model["training_max_date"] == row["training_max_date"] < row["fit_before_date"] <= row["date"],
                            "source_prediction_chronology", mid)
                audit.check(model["rho"] == 0 and model["regularization"] == .01 and model["half_life_days"] == 180,
                            "source_original_parameters", mid)
                stat = statistics(row["actual_score"], replay_rates(model, row))
                tracks[arm][mid] = (row, stat)
    identity_fields = ("competition", "date", "home", "away", "actual_score", "actual_outcome", "source_url")
    seen = {}
    for arm in ARMS:
        for mid, (row, stat) in tracks[arm].items():
            identity = {field: row[field] for field in identity_fields}
            if mid in seen:
                audit.tree(seen[mid], identity, "overlap_identity", mid)
            seen[mid] = identity
    common = set.intersection(*(set(track) for track in tracks.values()))
    ids = sorted(common, key=lambda mid: (tracks["days_28"][mid][0]["date"], mid))
    coverage = {"common_match_ids": ids, "counts": {arm: len(tracks[arm]) for arm in ARMS},
        "outside_common": {arm: sorted(set(tracks[arm])-common) for arm in ARMS}}
    rows = [tracks["days_28"][mid][0] for mid in ids]
    stats = {arm: [tracks[arm][mid][1] for mid in ids] for arm in ARMS}
    return rows, stats, coverage, snapshots, tracks


def feature_matrix(stats):
    features = {arm+"_"+field: [r[field] for r in stats[arm]] for arm in ARMS
                for field in (*MOMENT_FIELDS, *NLL_FIELDS)}
    for arm in ARMS[:2]:
        for field in NLL_FIELDS:
            features[arm+"_minus_days_28_"+field] = [a[field]-b[field]
                for a, b in zip(stats[arm], stats["days_28"])]
    return list(features), np.column_stack(list(features.values()))


def audit_csv(path, rows, stats, tracks, audit):
    expected_fields = ["match_id", "date", "competition", "home", "away", "home_goals", "away_goals",
                       "fixed_28_home_rate_bin", "fixed_28_away_rate_bin"]
    expected_fields += [arm+"_"+field for arm in ARMS for field in
                        ("fit_before_date", "training_matches", *STAT_FIELDS)]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        audit.check(reader.fieldnames == expected_fields, "csv_field_order", str(path))
        saved = list(reader)
    audit.check(len(saved) == len(rows), "csv_row_count", str(path))
    audit.check([r["match_id"] for r in saved] == [r["match_id"] for r in rows],
                "csv_order_and_unique_match_ids", str(path))
    for i, (row, stored) in enumerate(zip(rows, saved)):
        mid = row["match_id"]
        expected = {key: row[key] for key in expected_fields[:5]}
        expected.update({"home_goals": row["actual_score"][0], "away_goals": row["actual_score"][1],
            "fixed_28_home_rate_bin": bin_label(stats["days_28"][i]["home_rate"]),
            "fixed_28_away_rate_bin": bin_label(stats["days_28"][i]["away_rate"])})
        for arm in ARMS:
            source = tracks[arm][mid][0]
            expected.update({arm+"_fit_before_date": source["fit_before_date"],
                             arm+"_training_matches": source["training_matches"]})
            expected.update({arm+"_"+field: value for field, value in stats[arm][i].items()})
        for key, value in expected.items():
            if isinstance(value, Real):
                audit.numeric(value, stored.get(key), "csv_numeric", [mid, key])
            else:
                audit.check(value == stored.get(key), "csv_identity", [mid, key])
    return len(saved)


def audit_period(output, parent, season, saved, audit):
    rows, stats, coverage, snapshots, tracks = load_period(parent, season, audit)
    ids = coverage["common_match_ids"]
    audit.tree(coverage, saved["coverage"], "coverage", season)
    base = paired_summary(stats, range(len(rows)))
    base.update({"first_date": rows[0]["date"], "last_date": rows[-1]["date"]})
    audit.tree(base, saved, "whole_period_summary", season)
    for key, labels in (("by_league", [r["competition"] for r in rows]),
                        ("by_month", [r["date"][:7] for r in rows])):
        audit.check(set(saved[key]) == set(labels), "group_labels", [season, key])
        for label in sorted(set(labels)):
            expected = paired_summary(stats, [i for i, value in enumerate(labels) if value == label])
            audit.tree(expected, saved[key][label], key, season+"."+label)
    for side in ("home", "away"):
        labels = [bin_label(row[side+"_rate"]) for row in stats["days_28"]]
        groups = saved["fixed_28_rate_bins"][side]
        audit.check(set(groups) == set(BIN_NAMES), "fixed_bin_labels", [season, side])
        for label in BIN_NAMES:
            expected = paired_summary(stats, [i for i, value in enumerate(labels) if value == label])
            audit.tree(expected, groups[label], "fixed_28_rate_bins", season+"."+side+"."+label)
        audit.numeric(len(rows), sum(g["n"] for g in groups.values()), "fixed_bins_partition_n", [season, side])
    for arm in ARMS:
        outside = coverage["outside_common"][arm]
        # Source trials are stored league-by-league and date-ordered within each
        # league. Preserve that order as the runner does for extra-row summaries.
        extra_stats = [stat for mid, (_, stat) in tracks[arm].items() if mid in set(outside)]
        expected = {"match_ids": outside, "summary_separate_not_comparable_to_common":
                    summarize(extra_stats) if extra_stats else None}
        audit.tree(expected, saved["extra_coverage"][arm], "extra_coverage", season+"."+arm)
    names, matrix = feature_matrix(stats)
    for length in (2, 4, 8):
        stored = saved["bootstrap"][str(length)]
        boot = block_bootstrap([r["date"] for r in rows], matrix, length, stored["repeats"], stored["seed"])
        intervals, means = boot.pop("interval_95"), boot.pop("mean")
        audit.tree(boot, stored, "bootstrap_metadata", season+f".{length}")
        audit.check(set(stored["features"]) == set(names), "bootstrap_feature_set", [season, length])
        for i, name in enumerate(names):
            audit.tree({"mean": means[i], "interval_95": intervals[i] if intervals is not None else None},
                       stored["features"][name], "joint_bootstrap", season+f".{length}."+name)
    csv_rows = audit_csv(Path(output)/season/"逐场进球率与残差.csv", rows, stats, tracks, audit)
    return {"unique_common_matches": len(ids), "source_parameter_snapshots": snapshots,
        "source_saved_prediction_rows": sum(len(track) for track in tracks.values()),
        "common_arm_diagnostic_rows": len(ARMS)*len(rows), "csv_rows": csv_rows,
        "min_date": rows[0]["date"], "max_date": rows[-1]["date"]}


def run(output, parent):
    """Audit an existing result and write only a separate arithmetic report."""
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if not output.is_dir() or not parent.is_dir():
        raise ValueError("Both output and parent must be existing directories")
    result = read_json(output/"进球率诊断结果.json")
    audit = Auditor()
    audit.check(Path(result["parent_directory"]).resolve() == parent, "explicit_parent_directory")
    audit.tree({"new_model_fits": 0, "new_model_predictions": 0, "probabilities_changed": False},
               result, "no_fit_or_forecast_mutation")
    audit.tree(result["settings"], read_json(output/"分析设置.json"), "saved_settings_identity")
    audit.check(set(result["periods"]) == set(PERIODS), "both_declared_seasons")
    periods = {season: audit_period(output, parent, season, result["periods"][season], audit) for season in PERIODS}
    expected_counts = {"parameter_snapshots_read": sum(p["source_parameter_snapshots"] for p in periods.values()),
        "saved_prediction_rows_validated": sum(p["source_saved_prediction_rows"] for p in periods.values()),
        "common_arm_diagnostic_rows": sum(p["common_arm_diagnostic_rows"] for p in periods.values()),
        "unique_common_matches": sum(p["unique_common_matches"] for p in periods.values()), "saved_trials_read": 30}
    audit.tree(expected_counts, result["counts"], "execution_counts")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "passed" if not audit.errors else "failed", "error_count": len(audit.errors),
        "errors": audit.errors, "check_counts": dict(audit.checks),
        "arithmetic_check_count": sum(audit.checks.values()),
        "maximum_absolute_errors": audit.maximum_absolute_errors,
        "output_directory": str(output), "parent_directory": str(parent), "periods": periods,
        "new_base_model_fits": 0, "new_calibrator_fits": 0,
        "implementation": {"production_package_imported": False,
            "lambda": "Explicit attack/defense/competition/home coefficient additions and exponentials",
            "nll": "math.lgamma and direct log-domain independent-Poisson score",
            "draw": "exp(-(sqrt(lambda_h)-sqrt(lambda_a))^2)*scipy.special.i0e(2*sqrt(lambda_h)*sqrt(lambda_a))",
            "goal_count_expectations": "Independent 0-3 Poisson recurrence plus gammainc(4,lambda) tail, summed per match",
            "bootstrap": "Joint noncircular calendar-week groups with all arms and metrics; looped resample sums",
            "bootstrap_reference": "Adapted from outputs/较早校准研究_20261023T070259281026Z/independent_calibration_audit.py",
            "numeric_tolerance": {"absolute": 2e-10, "relative": 2e-11}},
        "not_covered": ["No base-model optimizer re-fit or optimum proof",
            "No new verification of raw data vendor truth, publication times, injury/lineup truth or pre-match availability",
            "Stored forecasts are treated as retrospective date-level experiments, not previously unseen blind tests",
            "Conditional bootstrap excludes model fitting, selection, future structural changes and cross-season uncertainty",
            "Passing arithmetic audit does not establish predictive superiority, probability truth or profitability"]}
    destination = output/"独立进球率算术审计.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "audit_status": report["audit_status"],
        "error_count": report["error_count"], "arithmetic_check_count": report["arithmetic_check_count"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    arguments = parser.parse_args()
    outcome = run(arguments.output, arguments.parent)
    raise SystemExit(0 if outcome["audit_status"] == "passed" else 1)

