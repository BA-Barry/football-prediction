"""Bounded, retrospective 1X2 statistical/market fusion research.

Only one scalar is fitted on the earlier season, then saved and reread before
loading the later season. Previously inspected archives are not a blind test;
pre-closing aggregation is not a verified T-24 quote or a tradable price. This
module neither reconstructs score distributions nor changes a production model.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from .evaluation import calibration_bins
from .forecast_diagnostics import probabilities, validate_prediction
from .market_research import load_season_records


EARLY, LATE = "2024-25", "2025-26"
MARKET = "avg_preclosing"
TRACKS = ("statistical", "market", "fusion")
PAIRS = (("statistical", "market"), ("fusion", "market"), ("fusion", "statistical"))
METRICS = ("log_loss", "brier_sum", "top1", "rps")
MAX_BISECTIONS = 32


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def _number(value):
    """JSON-safe extended-real result, distinguishing infinity from absence."""
    value = float(value)
    if math.isnan(value):
        return {"value": None, "status": "undefined_infinite_difference"}
    if math.isinf(value):
        return {"value": None, "status": "positive_infinity" if value > 0 else "negative_infinity"}
    return {"value": value, "status": "finite"}


def _arrays(statistical, market, outcomes):
    s, m, raw = np.asarray(statistical, dtype=float), np.asarray(market, dtype=float), np.asarray(outcomes)
    if s.ndim != 2 or s.shape[1:] != (3,) or m.shape != s.shape or not len(s):
        raise ValueError("Need paired, nonempty n x 3 statistical and market probabilities")
    for row in (*s, *m):
        probabilities(row)
    if raw.shape != (len(s),) or raw.dtype.kind not in "iu" or np.any((raw < 0) | (raw > 2)):
        raise ValueError("Outcomes must be paired integer labels 0=home, 1=draw, 2=away")
    return s, m, raw.astype(int)


def blend(statistical, market, weight):
    """Stable convex combination, retaining exact endpoints and tiny positives."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Statistical weight must be finite and in [0, 1]")
    s, m = np.asarray(statistical, dtype=float), np.asarray(market, dtype=float)
    if weight == 0:
        return m.copy()
    if weight == 1:
        return s.copy()
    # min + coefficient * (max-min) avoids endpoint cancellation and splitting
    # equal subnormal positive probabilities into two underflowing summands.
    return np.where(s >= m, m + weight*(s-m), s + (1-weight)*(m-s))


def _mean(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any() or (np.isposinf(values).any() and np.isneginf(values).any()):
        return math.nan
    if np.isposinf(values).any():
        return math.inf
    if np.isneginf(values).any():
        return -math.inf
    # Divide first: a finite mean must not overflow only because its sum does.
    return math.fsum((values / len(values)).tolist())


def _loss(observed):
    with np.errstate(divide="ignore"):
        return -np.log(observed)


def fit_weight(statistical, market, outcomes):
    """Minimize mean 1X2 LL by convex endpoint tests and <=32 bisections.

    The bracket width bounds weight resolution, not loss error or statistical
    uncertainty. A flat observed-class objective does not identify a weight;
    w=0 is then an explicit reference convention, not an estimated optimum.
    """
    s, m, y = _arrays(statistical, market, outcomes)
    s, m = s[np.arange(len(y)), y], m[np.arange(len(y)), y]
    impossible = np.flatnonzero((s == 0) & (m == 0)).tolist()
    if impossible:
        raise ValueError(f"No finite training objective: both lines give observed probability zero at rows {impossible}")
    delta = s-m

    def derivative(w):
        p = blend(s, m, w)
        if np.any(p == 0):
            return -math.inf if w == 0 else math.inf
        with np.errstate(over="ignore"):
            terms = -delta / p
        return _mean(terms)

    g0, g1 = derivative(0), derivative(1)
    identified = bool(np.any(delta != 0))
    iterations, lo, hi = 0, 0.0, 1.0
    if not identified:
        w, status = 0.0, "unidentifiable_flat_observed_probabilities"
    elif g0 >= 0:
        w, lo, hi, status = 0.0, 0.0, 0.0, "market_endpoint"
    elif g1 <= 0:
        w, lo, hi, status = 1.0, 1.0, 1.0, "statistical_endpoint"
    else:
        status = "interior_bisection"
        for _ in range(MAX_BISECTIONS):
            iterations += 1
            mid = (lo+hi)/2
            gradient = derivative(mid)
            if gradient == 0:
                lo = hi = mid
                break
            if gradient < 0:
                lo = mid
            else:
                hi = mid
        w = (lo+hi)/2
    return {"statistical_weight": w, "market_weight": 1-w,
            "identifiable": identified, "status": status, "n": len(y),
            "objective": "mean_1x2_log_loss", "iterations": iterations,
            "bracket": [lo, hi], "bracket_width": hi-lo,
            "endpoint_derivatives": {"w0": _number(g0), "w1": _number(g1)},
            "endpoint_log_loss": {"market": _number(_mean(_loss(m))),
                                  "statistical": _number(_mean(_loss(s)))},
            "fitted_log_loss": _number(_mean(_loss(blend(s, m, w)))),
            "unidentifiable_convention": None if identified else "w=0 market reference; not evidence for fusion"}


def _validate_records(records, season):
    if season not in (EARLY, LATE) or not records:
        raise ValueError("Need nonempty records for a declared research season")
    start, end = date(int(season[:4]), 8, 1), date(int(season[:4])+1, 7, 1)
    seen = set()
    for row in records:
        # Validate even rows with no primary quote; never salvage a valid subset.
        validate_prediction(row)
        if not start <= date.fromisoformat(row["date"]) < end:
            raise ValueError("Prediction date outside declared research season")
        if row["match_id"] in seen:
            raise ValueError("Duplicate match ID in research season")
        seen.add(row["match_id"])
    return sorted(records, key=lambda r: (r["date"], r["match_id"]))


def _components(p, y):
    truth = np.eye(3)[y]
    maxima = p == p.max(axis=1)[:, None]
    return {"log_loss": _loss(p[np.arange(len(y)), y]),
            "brier_sum": np.square(p-truth).sum(axis=1),
            "top1": maxima[np.arange(len(y)), y]/maxima.sum(axis=1),
            "rps": np.square(np.cumsum(p, axis=1)[:, :2]-np.cumsum(truth, axis=1)[:, :2]).sum(axis=1)/2}


def _score(p, y):
    c = _components(p, y)
    ll = _number(_mean(c["log_loss"]))
    return {"n": len(y), "log_loss": ll["value"], "log_loss_status": ll["status"],
            "zero_observed_probability_count": int(np.count_nonzero(p[np.arange(len(y)), y] == 0)),
            **{k: _mean(c[k]) for k in METRICS[1:]},
            "top1_tie_matches": int(np.count_nonzero((p == p.max(axis=1)[:, None]).sum(axis=1) > 1)),
            "mean_probabilities": p.mean(axis=0).tolist(),
            "observed_frequencies": np.eye(3)[y].mean(axis=0).tolist(),
            "calibration_5_bins": calibration_bins(p, y, bins=5)}


def _contrasts(components):
    output = {}
    for a, b in PAIRS:
        with np.errstate(invalid="ignore"):
            difference = {k: components[a][k]-components[b][k] for k in METRICS}
        output[f"{a}_minus_{b}"] = {
            "log_loss": _number(_mean(difference["log_loss"])),
            "undefined_infinite_log_loss_difference_count": int(np.isnan(difference["log_loss"]).sum()),
            **{k: _mean(difference[k]) for k in METRICS[1:]}}
    return output


def _summary(p, y):
    return {"n": len(y), "scores": {name: _score(p[name], y) for name in TRACKS},
            "paired_differences": _contrasts({name: _components(p[name], y) for name in TRACKS})}


def analyse_season(records, weight, season):
    """Describe fixed-w three-track comparisons only within quoted fixtures."""
    records = _validate_records(records, season)
    quoted = [r for r in records if MARKET in r["market_probabilities"]]
    missing = [r for r in records if MARKET not in r["market_probabilities"]]
    coverage = {"all_statistical_records": len(records), "same_fixture_quoted_records": len(quoted),
                "missing_quote_records": len(missing),
                "missing_quote_match_ids": [r["match_id"] for r in missing],
                "quoted_match_ids": [r["match_id"] for r in quoted],
                "missing_market_records": missing,
                "missingness_note": "Conditional quoted cohort; missing quotes not assumed random or substituted"}
    result = {"season": season, "statistical_weight": weight, "coverage": coverage,
              "all_statistical_scores_secondary_not_paired": _score(
                  np.asarray([r["probabilities"] for r in records]),
                  np.asarray([r["actual_outcome"] for r in records])),
              "same_fixture_comparison": None, "paired_matches": [],
              "by_league": {}, "by_calendar_month": {}, "by_utc_week": {}}
    if not quoted:
        result["comparison_status"] = "no_common_quotes"
        return result
    s, m, y = _arrays([r["probabilities"] for r in quoted],
                      [r["market_probabilities"][MARKET] for r in quoted],
                      [r["actual_outcome"] for r in quoted])
    p = {"statistical": s, "market": m, "fusion": blend(s, m, weight)}
    result["comparison_status"] = "available"
    result["same_fixture_comparison"] = _summary(p, y)
    difference = s-m
    result["probability_differences_descriptive"] = {
        "statistical_minus_market_mean_by_class": difference.mean(axis=0).tolist(),
        "mean_absolute_probability_difference": float(np.abs(difference).mean()),
        "maximum_absolute_probability_difference": float(np.abs(difference).max()),
        "top1_changed_statistical_vs_market": int(np.count_nonzero(s.argmax(axis=1) != m.argmax(axis=1))),
        "top1_changed_fusion_vs_market": int(np.count_nonzero(p["fusion"].argmax(axis=1) != m.argmax(axis=1))),
        "top1_changed_fusion_vs_statistical": int(np.count_nonzero(p["fusion"].argmax(axis=1) != s.argmax(axis=1)))}
    components = {name: _components(value, y) for name, value in p.items()}
    for i, row in enumerate(quoted):
        result["paired_matches"].append({
            **{k: row[k] for k in ("match_id", "competition", "date", "home", "away", "actual_outcome", "source_url")},
            "source_decimal_odds": row["market_prices"][MARKET],
            "probabilities": {name: value[i].tolist() for name, value in p.items()},
            "top1": {name: int(value[i].argmax()) for name, value in p.items()},
            "scores": {name: {"log_loss": _number(c["log_loss"][i]),
                              **{k: float(c[k][i]) for k in METRICS[1:]}}
                       for name, c in components.items()},
            "paired_differences": _contrasts({name: {k: v[i:i+1] for k, v in c.items()}
                                              for name, c in components.items()})})
    dates = [date.fromisoformat(r["date"]) for r in quoted]
    groups = {"by_league": [r["competition"] for r in quoted],
              "by_calendar_month": [r["date"][:7] for r in quoted],
              "by_utc_week": [(d-timedelta(days=d.weekday())).isoformat() for d in dates]}
    for kind, labels in groups.items():
        for label in sorted(set(labels)):
            mask = np.asarray([x == label for x in labels])
            result[kind][label] = _summary({name: value[mask] for name, value in p.items()}, y[mask])
    return result


def _cross_seasons(earlier, later):
    if {r["match_id"] for r in earlier} & {r["match_id"] for r in later}:
        raise ValueError("Duplicate match ID across research seasons")
    if max(r["date"] for r in earlier) >= min(r["date"] for r in later):
        raise ValueError("All fusion training dates must strictly precede later prediction dates")


def policy(parent):
    """Recorded choices, not a new contract, gate, or claim of preregistration."""
    return {"parent_directory": str(Path(parent).resolve()), "early_season": EARLY, "late_season": LATE,
            "statistical_source": "original row.probabilities only", "market": MARKET,
            "market_method": "proportional inverse-odds normalization; verify every saved row",
            "formula": "p=w*statistical+(1-w)*market", "parameter": "one global statistical weight w in [0,1]",
            "objective": "earlier quoted-cohort mean 1X2 log loss only",
            "solver": "convex endpoint derivatives then at most 32 bisections; no grid",
            "max_bisections": MAX_BISECTIONS, "maximum_interior_weight_bracket_width": 2**-32,
            "unidentifiable": "flat observed-class LL -> explicit status; w=0 market-reference convention",
            "zero_probability": "no clipping; infinity retained; both-zero observed training outcome raises",
            "execution": "write policy; load early; fit; save weight; reread weight; only then load late",
            "cohort": "all statistical rows validated; fitting and three-track comparison on identical quoted fixtures",
            "groups": ["league", "calendar month", "Monday-start UTC calendar week"],
            "date_note": "UTC-labelled calendar grouping uses archived date text, not verified kickoff timezone",
            "top1_ties": "expected hit under uniform selection among exact maxima; tie count reported; argmax label descriptive only",
            "calibration": "five equal-width bins by class, descriptive only; occupied bins retained",
            "scope": "exploratory previously inspected seasons, not blind or T-24 validation",
            "no_claims": ["formal p values", "sampling confidence intervals", "ROI", "tradable odds", "full scores", "handicaps"],
            "unchanged": ["original models", "prospective nine cases", "LLM settings", "production prediction API"],
            "team_model_fits": 0, "fusion_weight_fits": 1}


def _report(result):
    fit = result["weight_fit"]
    lines = ["# 统计＋市场：固定单权重历史对照", "",
             f"统计权重 w={fit['statistical_weight']:.10f}；{fit['status']}。只在2024/25共同报价队列按胜平负LL拟合，"
             "保存并重新读取权重后，才载入2025/26。后季不调权重、不按联赛重选。", "",
             "|赛季|全部统计|同场报价|缺报价|线路|LL|Brier sum|Top1|RPS|",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for season, detail in result["seasons"].items():
        coverage, comparison = detail["coverage"], detail["same_fixture_comparison"]
        if comparison is None:
            lines += [f"{season}：无共同报价，三线评分不可用；不替换市场。"]
            continue
        for name, score in comparison["scores"].items():
            ll = "+∞" if score["log_loss_status"] == "positive_infinity" else f"{score['log_loss']:.6f}"
            lines += [f"|{season}|{coverage['all_statistical_records']}|{coverage['same_fixture_quoted_records']}|"
                      f"{coverage['missing_quote_records']}|{name}|{ll}|{score['brier_sum']:.6f}|{score['top1']:.2%}|{score['rps']:.6f}|"]
    lines += ["", "早季评分是权重的拟合内描述，不是独立验证。两季此前已查看，后季是固定权重的历史对照，不称新盲测。",
              "若LL目标平坦，w=0只是明确的市场参考约定，不能声称识别出最优融合或支持引入统计线。",
              "LL、Brier sum、RPS越低越好；Top1为最高概率精确并列时均匀选择的期望命中率，越高越好。"
              "逐场argmax标签仅描述，命中值采用上述期望规则。逐场及联赛／月／周差值均为同场方案相减。"
              "无删异常赛果、抽样置信区间或显著性检验；分组不能视为独立重复试验。",
              "JSON完整保留缺报价记录、0概率造成的无穷损失、逐场概率差、三线配对差及五箱校准。"
              "∞−∞明确不可定义，不悄悄删场或改成0。全部统计记录的次要评分不可与报价子集混作三线比较。",
              "pre-closing不是核实过的开盘或T−24；聚合均价未必可成交。日期仅按原日期文本归入周一开始的UTC标记日历周，"
              "不是恢复了真实开球时区或报价时间。缺报价可能非随机，结果只适用于本次有报价队列。",
              "仅胜平负融合，不生成比分、进球数、半全场或让球，不推导收益或保证中奖。原模型、9场案例和LLM设置均不修改。",
              "统计分析技能用于同场比较、完整覆盖和结果边界；本轮无新依赖、hash、contract或gate。", ""]
    return "\n".join(lines)


def run(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    started = datetime.now(timezone.utc)
    output = root/"outputs"/("统计市场固定融合研究_"+started.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    settings = policy(parent)
    _write(output/"融合研究设置.json", settings)
    early, early_sources = load_season_records(parent, EARLY)
    early = _validate_records(early, EARLY)
    quoted = [r for r in early if MARKET in r["market_probabilities"]]
    if not quoted:
        raise ValueError("No earlier primary-market quotes; cannot fit or substitute another market")
    fit = fit_weight([r["probabilities"] for r in quoted],
                     [r["market_probabilities"][MARKET] for r in quoted],
                     [r["actual_outcome"] for r in quoted])
    fit.update({"training_season": EARLY, "training_match_ids": [r["match_id"] for r in quoted],
                "training_first_date": quoted[0]["date"], "training_last_date": quoted[-1]["date"]})
    weight_path = output/"早期拟合权重.json"
    _write(weight_path, fit)
    saved_fit = json.loads(weight_path.read_text(encoding="utf-8"))
    weight = saved_fit["statistical_weight"]
    # Intentionally no later-season read (even metadata) before this point.
    late, late_sources = load_season_records(parent, LATE)
    late = _validate_records(late, LATE)
    _cross_seasons(early, late)
    result = {"started_at_utc": started.isoformat(), "completed_at_utc": None,
              "output_directory": str(output), "settings": settings, "weight_fit": saved_fit,
              "source_files": {EARLY: early_sources, LATE: late_sources},
              "new_team_model_fits": 0, "fusion_weight_fits": 1, "late_weight_refits": 0,
              "seasons": {EARLY: analyse_season(early, weight, EARLY),
                          LATE: analyse_season(late, weight, LATE)}}
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(output/"统计市场融合结果.json", result)
    (output/"统计市场融合说明.md").write_text(_report(result), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.root, args.parent))}, ensure_ascii=False))
