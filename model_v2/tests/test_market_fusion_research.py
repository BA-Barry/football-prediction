from copy import deepcopy
import json
import math

import numpy as np
import pytest

from football_model import market_fusion_research as research
from football_model.football_data import LEAGUES


def row(mid="a", season=research.EARLY, statistical=(.6, .2, .2), odds=(2., 4., 4.), outcome=0):
    year = int(season[:4])
    inverse = 1/np.asarray(odds)
    return {"match_id": mid, "competition": "E0", "date": f"{year}-08-20", "home": "H", "away": "A",
            "actual_outcome": outcome, "actual_score": [[1, 0], [0, 0], [0, 1]][outcome],
            "probabilities": list(statistical), "market_probabilities": {research.MARKET: (inverse/inverse.sum()).tolist()},
            "market_prices": {research.MARKET: list(odds)}, "source_url": "https://example.test/history.csv",
            "fit_before_date": f"{year}-08-01", "training_max_date": f"{year}-05-20"}


def test_market_and_statistical_endpoints_are_retained_exactly():
    a, b = [[.8, .1, .1]], [[.5, .25, .25]]
    best_stat = research.fit_weight(a, b, [0])
    best_market = research.fit_weight(b, a, [0])
    assert best_stat["statistical_weight"] == 1 and best_stat["status"] == "statistical_endpoint"
    assert best_market["statistical_weight"] == 0 and best_market["status"] == "market_endpoint"
    assert best_stat["iterations"] == best_market["iterations"] == 0
    assert best_stat["bracket"] == [1, 1] and best_market["bracket"] == [0, 0]
    assert np.array_equal(research.blend(a, b, 0), b)
    assert np.array_equal(research.blend(a, b, 1), a)


def test_interior_optimum_analytic_bracket_and_swap_symmetry():
    # Observed probabilities .2+.6w and .7-.6w have their product maximized
    # where equal: w=5/12, without minimizing Brier or choosing a grid point.
    s, m = [[.8, .1, .1], [.1, .45, .45]], [[.2, .4, .4], [.7, .15, .15]]
    fitted = research.fit_weight(s, m, [0, 0])
    assert fitted["status"] == "interior_bisection"
    assert fitted["bracket"][0] <= 5/12 <= fitted["bracket"][1]
    assert fitted["statistical_weight"] == pytest.approx(5/12, abs=2**-33)
    assert fitted["iterations"] <= 32 and fitted["bracket_width"] <= 2**-32
    swapped = research.fit_weight(m, s, [0, 0])
    assert fitted["statistical_weight"] + swapped["statistical_weight"] == pytest.approx(1, abs=2**-32)
    repeated = research.fit_weight(s*7, m*7, [0]*14)
    assert fitted["statistical_weight"] == repeated["statistical_weight"]


@pytest.mark.parametrize("s,m", [
    ([[.5, .25, .25]], [[.5, .25, .25]]),
    ([[.5, .4, .1]], [[.5, .1, .4]]),
])
def test_flat_observed_objective_is_unidentifiable_even_if_other_classes_differ(s, m):
    fitted = research.fit_weight(s, m, [0])
    assert not fitted["identifiable"]
    assert fitted["status"] == "unidentifiable_flat_observed_probabilities"
    assert fitted["statistical_weight"] == 0
    assert fitted["unidentifiable_convention"] is not None
    assert fitted["fitted_log_loss"]["value"] == pytest.approx(math.log(2))
    assert fitted["iterations"] == 0


def test_zero_endpoint_probabilities_have_infinite_loss_but_finite_interior():
    s, m = [[.8, .1, .1], [0., .5, .5]], [[0., .5, .5], [.8, .1, .1]]
    fitted = research.fit_weight(s, m, [0, 0])
    assert fitted["statistical_weight"] == .5
    assert fitted["endpoint_log_loss"]["market"]["status"] == "positive_infinity"
    assert fitted["endpoint_log_loss"]["statistical"]["status"] == "positive_infinity"
    assert fitted["endpoint_derivatives"]["w0"]["status"] == "negative_infinity"
    assert fitted["endpoint_derivatives"]["w1"]["status"] == "positive_infinity"
    assert fitted["fitted_log_loss"]["value"] == pytest.approx(-math.log(.4))
    json.dumps(fitted, allow_nan=False)


def test_both_zero_observed_training_probability_is_error_not_exclusion():
    with pytest.raises(ValueError, match="No finite training objective.*rows \\[0\\]"):
        research.fit_weight([[0., .5, .5], [.6, .2, .2]], [[0., .2, .8], [.7, .2, .1]], [0, 0])


def test_small_positive_probabilities_and_endpoint_cancellation_are_not_clipped():
    tiny = np.nextafter(0., 1.)
    s, m = [[tiny, .5, .5]], [[1., 0., 0.]]
    assert research.blend(s, m, 1)[0, 0] == tiny
    assert research.blend(s, s, .5)[0, 0] == tiny
    fitted = research.fit_weight(s, m, [0])
    assert fitted["statistical_weight"] == 0
    assert fitted["endpoint_log_loss"]["statistical"]["value"] == pytest.approx(-math.log(tiny))
    assert research._mean([1e308, 1e308]) == 1e308


@pytest.mark.parametrize("s,m,y", [
    ([], [], []), ([[.5, .5]], [[.5, .5]], [0]),
    ([[.5, .25, .25]], [[.5, .25, .25]], [True]),
    ([[.5, .25, .25]], [[.5, .25, .25]], [1.5]),
    ([[.5, .25, .25]], [[.5, .25, .25]], [3]),
    ([[.5, .3, .3]], [[.5, .25, .25]], [0]),
    ([[math.nan, .5, .5]], [[.5, .25, .25]], [0]),
])
def test_fit_rejects_invalid_arrays(s, m, y):
    with pytest.raises(ValueError):
        research.fit_weight(s, m, y)


@pytest.mark.parametrize("weight", [-.1, 1.1, math.nan, math.inf, True])
def test_invalid_weight_not_silently_clipped(weight):
    with pytest.raises(ValueError):
        research.blend([.5, .3, .2], [.3, .4, .3], weight)


def test_missing_quote_fully_retained_and_all_record_validation_not_bypassed():
    missing = row("noquote")
    missing["market_probabilities"] = missing["market_prices"] = {}
    records = [row(), missing]
    before = deepcopy(records)
    result = research.analyse_season(records, .4, research.EARLY)
    assert records == before
    assert result["coverage"]["all_statistical_records"] == 2
    assert result["coverage"]["same_fixture_quoted_records"] == 1
    assert result["coverage"]["missing_market_records"] == [missing]
    assert result["same_fixture_comparison"]["n"] == 1
    assert all(s["n"] == 1 for s in result["same_fixture_comparison"]["scores"].values())
    assert result["all_statistical_scores_secondary_not_paired"]["n"] == 2
    missing["probabilities"] = [.6, .6, .6]
    with pytest.raises(ValueError, match="sum to one"):
        research.analyse_season(records, .4, research.EARLY)


def test_late_zero_quote_coverage_explicit_no_market_substitution():
    missing = row(season=research.LATE)
    missing["market_probabilities"] = missing["market_prices"] = {}
    result = research.analyse_season([missing], .4, research.LATE)
    assert result["comparison_status"] == "no_common_quotes"
    assert result["same_fixture_comparison"] is None and result["paired_matches"] == []
    assert result["coverage"]["missing_quote_match_ids"] == ["a"]


def test_proportional_market_consistency_and_all_other_quote_validation():
    bad = row()
    bad["market_probabilities"][research.MARKET] = [.6, .2, .2]
    with pytest.raises(ValueError, match="proportional de-vig"):
        research.analyse_season([bad], .4, research.EARLY)
    bad = row()
    bad["market_probabilities"]["avg_closing"] = [.5, .25, .25]
    with pytest.raises(ValueError, match="keys disagree"):
        research.analyse_season([bad], .4, research.EARLY)


def test_descriptive_metrics_paired_groups_probability_changes_and_ties():
    records = [row("a", statistical=(.25, .5, .25), outcome=1), row("b", outcome=2)]
    records[1]["date"] = "2024-09-01"
    result = research.analyse_season(records, .25, research.EARLY)
    p = np.array([[.25, .5, .25], [.6, .2, .2]])
    truth = np.eye(3)[[1, 2]]
    score = result["same_fixture_comparison"]["scores"]["statistical"]
    assert score["log_loss"] == pytest.approx(-np.log([.5, .2]).mean())
    assert score["brier_sum"] == pytest.approx(np.square(p-truth).sum(axis=1).mean())
    assert score["rps"] == pytest.approx(np.square(np.cumsum(p-truth, axis=1)[:, :2]).sum(axis=1).mean()/2)
    assert score["top1"] == .5 and score["n"] == 2
    assert len(score["calibration_5_bins"]) > 0
    assert set(result["by_calendar_month"]) == {"2024-08", "2024-09"}
    assert set(result["by_utc_week"]) == {"2024-08-19", "2024-08-26"}
    assert result["probability_differences_descriptive"]["top1_changed_statistical_vs_market"] == 1
    assert sum(x["n"] for x in result["by_utc_week"].values()) == 2
    rows = result["paired_matches"]
    delta = result["same_fixture_comparison"]["paired_differences"]["fusion_minus_market"]["log_loss"]
    assert delta["value"] == pytest.approx(np.mean([r["paired_differences"]["fusion_minus_market"]["log_loss"]["value"] for r in rows]))
    tied = research._score(np.array([[.5, .5, 0.]]), np.array([1]))
    assert tied["top1_tie_matches"] == 1 and tied["top1"] == .5
    tied3 = research._score(np.array([[1/3, 1/3, 1/3]]), np.array([2]))
    assert tied3["top1_tie_matches"] == 1 and tied3["top1"] == 1/3
    json.dumps(result, allow_nan=False)


def test_zero_observed_late_losses_keep_infinity_and_undefined_difference():
    scores = research._score(np.array([[0., .5, .5]]), np.array([0]))
    assert scores["log_loss"] is None and scores["log_loss_status"] == "positive_infinity"
    assert scores["zero_observed_probability_count"] == 1
    c = research._components(np.array([[0., .5, .5]]), np.array([0]))
    contrasts = research._contrasts({name: c for name in research.TRACKS})
    value = contrasts["fusion_minus_market"]
    assert value["log_loss"]["status"] == "undefined_infinite_difference"
    assert value["undefined_infinite_log_loss_difference_count"] == 1
    json.dumps(contrasts, allow_nan=False)


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update({"fit_before_date": "2024-05-20"}), "strictly earlier"),
    (lambda r: r.update({"date": "2025-07-01"}), "outside declared"),
    (lambda r: r.update({"actual_outcome": 2}), "disagrees with score"),
])
def test_invalid_chronology_interval_and_score_raise(change, match):
    bad = row()
    change(bad)
    with pytest.raises(ValueError, match=match):
        research.analyse_season([bad], .4, research.EARLY)


def test_duplicate_ids_and_cross_season_order_rejected():
    with pytest.raises(ValueError, match="Duplicate match ID"):
        research.analyse_season([row(), row()], .4, research.EARLY)
    with pytest.raises(ValueError, match="across research seasons"):
        research._cross_seasons([row()], [row(season=research.LATE)])
    with pytest.raises(ValueError, match="strictly precede"):
        research._cross_seasons([row("early")], [row("late")])


def write_archive(parent, late_outcome=0):
    for season in (research.EARLY, research.LATE):
        (parent/season).mkdir(parents=True)
        year = int(season[:4])
        for competition in LEAGUES:
            r = row(f"{season}:{competition}", season, outcome=late_outcome if season == research.LATE else 0)
            r["competition"] = competition
            data = {"start": f"{year}-08-01", "end_exclusive": f"{year+1}-07-01", "predictions": [r]}
            (parent/season/f"{competition}_original.json").write_text(json.dumps(data), encoding="utf-8")


def test_run_saves_and_rereads_early_weight_before_loading_late_and_sources_unchanged(tmp_path, monkeypatch):
    parent = tmp_path/"parent"
    write_archive(parent)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in parent.rglob("*.json")}
    loader, observations = research.load_season_records, []

    def watched_loader(p, season):
        directory = next((tmp_path/"run"/"outputs").iterdir())
        assert (directory/"融合研究设置.json").exists()
        if season == research.LATE:
            saved = json.loads((directory/"早期拟合权重.json").read_text(encoding="utf-8"))
            assert saved["training_season"] == research.EARLY
            observations.append(saved["statistical_weight"])
        return loader(p, season)

    monkeypatch.setattr(research, "load_season_records", watched_loader)
    output = research.run(tmp_path/"run", parent)
    result = json.loads((output/"统计市场融合结果.json").read_text(encoding="utf-8"))
    assert observations == [result["weight_fit"]["statistical_weight"]]
    assert result["late_weight_refits"] == result["new_team_model_fits"] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == original for p, original in before.items())
    assert result["seasons"][research.LATE]["same_fixture_comparison"]["n"] == len(LEAGUES)
    assert (output/"统计市场融合说明.md").exists()


def test_changing_late_outcomes_cannot_change_fitted_weight(tmp_path):
    weights = []
    for outcome in (0, 1, 2):
        parent = tmp_path/f"parent{outcome}"
        write_archive(parent, late_outcome=outcome)
        output = research.run(tmp_path/f"run{outcome}", parent)
        weights.append(json.loads((output/"早期拟合权重.json").read_text(encoding="utf-8")))
    assert weights[0] == weights[1] == weights[2]


def test_run_rejects_cross_season_duplicate_and_no_training_quote(tmp_path):
    parent = tmp_path/"parent"
    write_archive(parent)
    early = parent/research.EARLY/"E0_original.json"
    data = json.loads(early.read_text(encoding="utf-8"))
    data["predictions"][0]["match_id"] = f"{research.LATE}:E0"
    early.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="across research seasons"):
        research.run(tmp_path/"duplicates", parent)
    for path in (parent/research.EARLY).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["predictions"][0]["market_probabilities"] = {}
        data["predictions"][0]["market_prices"] = {}
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="No earlier primary-market quotes"):
        research.run(tmp_path/"missing", parent)
