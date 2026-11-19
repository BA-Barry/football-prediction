"""Synthetic checks for the English-tier retrospective comparison.

No actual historical results, network requests, fitted models or old forecasts
are read by this module.  The fake model makes time/order assertions observable.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import date
import json
import math

import pytest
from scipy.stats import poisson, skellam

from football_model.english_tiers_data import SEASONS, parse_text
from football_model.english_tiers_experiment import (
    derive_promoted,
    load_inspections,
    run_comparison,
    score_forecast,
    summarize_rows,
)
from football_model.football_data import HistoricalMatch


def historical(ident, day, home="Arsenal", away="Chelsea", *,
               competition="E0", season="2025-26", score=(1, 0)):
    return HistoricalMatch(
        ident, competition, season, day, None,
        "england:" + home, "england:" + away, home, away,
        score, None, {}, "synthetic-only.txt",
        "https://example.invalid/synthetic-only", None,
    )


def parsed(season, competition, *, home="Arsenal FC", away="Chelsea FC",
           stage="Matchday 1", score="1-0"):
    day = date(int(season[:4]), 8, 9)
    weekday = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[day.weekday()]
    text = f"▪ {stage}\n{weekday} Aug 9 {day.year}\n{home}  v {away}  {score}\n"
    return parse_text(text, season=season, competition=competition,
                      source_path=f"synthetic-{season}-{competition}.txt",
                      source_url="https://example.invalid/synthetic-only")


def inspections(tmp_path):
    """Small observations in all eight table slots, not a full-season claim."""
    result = {}
    for season in SEASONS:
        for competition in ("en.1", "en.2"):
            home, away = (("Arsenal FC", "Chelsea FC") if competition == "en.1"
                          else ("Hull City AFC", "Coventry City FC"))
            payload = parsed(season, competition, home=home, away=away)
            path = tmp_path / f"{season}_{competition}.inspection.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result[season, competition] = path
    return result


def mutate(path, fn):
    payload = json.loads(path.read_text(encoding="utf-8"))
    fn(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_regular_eight_tables_without_fabricating_markets_or_clocks(tmp_path):
    inspections(tmp_path)
    records, audit = load_inspections(tmp_path)
    assert len(records) == 8
    assert isinstance(audit, dict)
    assert {r.competition for r in records} == {"E0", "E1"}
    assert {r.season for r in records} == set(SEASONS)
    assert all(r.markets == {} and r.half_time is None for r in records)
    assert all(r.source_time_text is None for r in records)
    assert all(r.source_url == "https://example.invalid/synthetic-only" for r in records)
    assert all(r.home.startswith("england:") and r.away.startswith("england:") for r in records)
    assert records == sorted(records, key=lambda r: (r.match_date, r.competition, r.match_id))


def test_shared_team_id_survives_division_change(tmp_path):
    paths = inspections(tmp_path)
    for season, division in (("2024-25", "en.2"), ("2025-26", "en.1")):
        payload = parsed(season, division, home="Ipswich Town FC", away="Hull City AFC")
        paths[season, division].write_text(json.dumps(payload), encoding="utf-8")
    records, _ = load_inspections(tmp_path)
    ipswich = [r for r in records if r.home == "england:Ipswich"]
    assert len(ipswich) == 2
    assert {r.competition for r in ipswich} == {"E0", "E1"}


def test_load_does_not_promote_playoff_score_or_unknown_playoff_team(tmp_path):
    paths = inspections(tmp_path)
    # Parse one body so excluded and regular observations have distinct genuine
    # source-line numbers, as they do in a real source document.
    payload = parse_text(
        "▪ Matchday 1\nSat Aug 9 2025\nHull City AFC  v Coventry City FC  1-0\n"
        "▪ Playoffs\nSun Aug 10 2025\nHull City AFC  v Unresolved FC  5-6 pen. 1-1 a.e.t. (1-1, 0-1)\n",
        season="2025-26", competition="en.2",
        source_url="https://example.invalid/synthetic-only",
        source_path="synthetic-only.txt")
    paths["2025-26", "en.2"].write_text(json.dumps(payload), encoding="utf-8")
    records, audit = load_inspections(tmp_path)
    assert len(records) == 8
    assert all(r.away != "Unresolved FC" for r in records)
    assert all(r.full_time == (1, 0) for r in records)
    assert audit["excluded_non_regular_records"] == 1
    issue, = audit["excluded_source_issues"]
    assert issue["kind"] == "unknown_team_alias"
    assert issue["labels"] == ["Unresolved FC"]


def test_unrelated_parse_issue_is_not_hidden_by_an_excluded_playoff(tmp_path):
    paths = inspections(tmp_path)
    payload = parsed("2025-26", "en.2", home="Hull City AFC", away="Coventry City FC")
    playoff = parsed("2025-26", "en.2", home="Hull City AFC", away="Coventry City FC",
                     stage="Playoffs")
    playoff["records"][0]["source_line"] = 9
    payload["records"].extend(playoff["records"])
    payload["issues"].append({"line": 999, "kind": "unparsed_match_or_status", "raw": "unresolved"})
    paths["2025-26", "en.2"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_inspections(tmp_path)


def test_missing_excluded_line_number_does_not_exempt_an_unattributed_issue(tmp_path):
    paths = inspections(tmp_path)
    payload = parsed("2025-26", "en.2", home="Hull City AFC", away="Coventry City FC")
    playoff = parsed("2025-26", "en.2", home="Hull City AFC", away="Coventry City FC",
                     stage="Playoffs")
    playoff["records"][0]["source_line"] = None
    payload["records"].extend(playoff["records"])
    payload["issues"].append({"kind": "unparsed_match_or_status", "raw": "unattributed"})
    paths["2025-26", "en.2"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_inspections(tmp_path)


@pytest.mark.parametrize("damage", [
    lambda p: p["records"].append(deepcopy(p["records"][0])),
    lambda p: p["records"][0].update(eligible_regular_result=False),
    lambda p: p["records"][0]["home"].update(team_id=None),
    lambda p: p["records"][0]["home"].update(team_id="england:Invented"),
    lambda p: p["records"][0].update(regulation_time_score=[-1, 0]),
    lambda p: p["records"][0].update(regulation_time_score=[True, 0]),
    lambda p: p["records"][0].update(regulation_time_score=[1.5, 0]),
    lambda p: p["records"][0].update(regulation_time_score=None),
    lambda p: p["records"][0].update(match_date="2025-02-30"),
    lambda p: p["records"][0].update(competition="en.2"),
    lambda p: p["records"][0].update(season="2024-25"),
    lambda p: p.update(competition="unrecognised"),
])
def test_bad_regular_input_cannot_silently_enter_or_disappear(tmp_path, damage):
    paths = inspections(tmp_path)
    mutate(paths["2025-26", "en.1"], damage)
    with pytest.raises((ValueError, TypeError)):
        load_inspections(tmp_path)


def test_promoted_membership_uses_two_named_seasons_not_goals_or_future_results():
    records = [
        historical("prior-epl", date(2024, 8, 1), season="2024-25"),
        historical("prior-champ", date(2024, 8, 2), "Hull", "Coventry",
                   competition="E1", season="2024-25", score=(12, 0)),
        historical("target", date(2025, 8, 3), "Hull", "Arsenal"),
        historical("other-target", date(2025, 8, 4), "Newcomer", "Chelsea"),
        historical("irrelevant-future", date(2026, 8, 1), "Coventry", "Arsenal",
                   season="2026-27"),
    ]
    assert derive_promoted(records) == {"england:Hull"}
    changed = [replace(r, full_time=(0, 15)) for r in records]
    assert derive_promoted(changed) == derive_promoted(records)


@pytest.mark.parametrize("rates,actual", [
    ((1.4, 1.0), (0, 0)), ((1.4, 1.0), (2, 1)),
    ((0.3, 3.2), (1, 4)), ((2.0, 0.7), (24, 17)),
])
def test_full_score_nll_is_analytic_not_truncated_to_grid(rates, actual):
    scored = score_forecast(rates, actual)
    expected = sum(lam - goal * math.log(lam) + math.lgamma(goal + 1)
                   for lam, goal in zip(rates, actual))
    assert scored["full_score_nll"] == pytest.approx(expected, abs=1e-11)
    assert math.isfinite(scored["full_score_nll"])


@pytest.mark.parametrize("actual", [(2, 0), (1, 1), (0, 2)])
def test_outcome_and_total_losses_follow_the_same_poisson_forecast(actual):
    rates = (1.6, 0.8)
    scored = score_forecast(rates, actual)
    outcome = 0 if actual[0] > actual[1] else 1 if actual[0] == actual[1] else 2
    probabilities = [skellam.sf(0, *rates), skellam.pmf(0, *rates), skellam.cdf(-1, *rates)]
    expected_brier = sum((p - (i == outcome)) ** 2 for i, p in enumerate(probabilities))
    assert scored["one_x_two_log_loss"] == pytest.approx(-math.log(probabilities[outcome]), abs=1e-9)
    assert scored["one_x_two_brier"] == pytest.approx(expected_brier, abs=1e-9)
    totals = [poisson.pmf(i, sum(rates)) for i in range(4)] + [poisson.sf(3, sum(rates))]
    bucket = min(sum(actual), 4)
    assert scored["total_log_loss"] == pytest.approx(-math.log(totals[bucket]), abs=1e-9)
    assert scored["total_brier"] == pytest.approx(sum((p - (i == bucket)) ** 2
                                                    for i, p in enumerate(totals)), abs=1e-9)


@pytest.mark.parametrize("rates", [(0, 1), (-1, 1), (1, float("nan")), (float("inf"), 1)])
def test_invalid_goal_rates_raise(rates):
    with pytest.raises(ValueError):
        score_forecast(rates, (1, 0))


@pytest.mark.parametrize("actual", [(-1, 0), (True, 0), (1.5, 0), (1,), (1, 2, 3)])
def test_invalid_actual_score_raises(actual):
    with pytest.raises((ValueError, TypeError)):
        score_forecast((1.2, 1.0), actual)


def test_forecast_keeps_poisson_markets_and_three_way_minus_two_semantics():
    rates = (1.1, 2.4)
    forecast = score_forecast(rates, (0, 3))["forecast"]
    assert set(forecast["one_x_two"]) == {"home", "draw", "away"}
    assert sum(forecast["one_x_two"].values()) == pytest.approx(1)
    assert set(forecast["total_goals"]) == {"0", "1", "2", "3", "4+"}
    assert sum(forecast["total_goals"].values()) == pytest.approx(1)
    # Away minus two: >=3 margin wins; exactly2 is draw, not Asian push.
    jc = forecast["away_minus_two_jincai"]
    assert jc["win"] == pytest.approx(skellam.sf(2, rates[1], rates[0]), abs=1e-9)
    assert jc["draw"] == pytest.approx(skellam.pmf(2, rates[1], rates[0]), abs=1e-9)
    assert jc["lose"] == pytest.approx(skellam.cdf(1, rates[1], rates[0]), abs=1e-9)


def test_unique_top1_is_correctness_not_an_outcome_label():
    assert score_forecast((4, 0.3), (1, 0))["top1"] == 1
    assert score_forecast((4, 0.3), (0, 1))["top1"] == 0


def test_exact_two_way_top1_tie_splits_credit_without_order_bias(monkeypatch):
    forecast = score_forecast((1.2, 1.0), (1, 0))["forecast"]
    forecast["one_x_two"] = {"home": 0.45, "draw": 0.45, "away": 0.1}
    monkeypatch.setattr("football_model.english_tiers_experiment.summarize", lambda _: deepcopy(forecast))
    rows = [score_forecast((1.2, 1.0), actual) for actual in ((1, 0), (1, 1), (0, 1))]
    assert [row["top1"] for row in rows] == [0.5, 0.5, 0.0]
    assert all(row["top1_tied"] is True for row in rows)
    summary = summarize_rows([{"status": "predicted", **row} for row in rows]
                             + [{"status": "missing", "top1_tied": True}])
    assert summary["top1"] == pytest.approx(1 / 3)
    assert summary["top1_tie_count"] == 3


def test_uniform_three_way_top1_tie_gets_one_third(monkeypatch):
    forecast = score_forecast((1.2, 1.0), (1, 0))["forecast"]
    forecast["one_x_two"] = dict.fromkeys(("home", "draw", "away"), 1 / 3)
    monkeypatch.setattr("football_model.english_tiers_experiment.summarize", lambda _: deepcopy(forecast))
    for actual in ((1, 0), (0, 0), (0, 1)):
        row = score_forecast((1.2, 1.0), actual)
        assert row["top1"] == pytest.approx(1 / 3)
        assert row["top1_tied"] is True


def test_near_tie_is_not_treated_as_an_exact_top1_tie(monkeypatch):
    forecast = score_forecast((1.2, 1.0), (1, 0))["forecast"]
    forecast["one_x_two"] = {"home": 0.45 + 1e-13, "draw": 0.45 - 1e-13, "away": 0.1}
    monkeypatch.setattr("football_model.english_tiers_experiment.summarize", lambda _: deepcopy(forecast))
    home = score_forecast((1.2, 1.0), (1, 0))
    draw = score_forecast((1.2, 1.0), (0, 0))
    assert home["top1"] == 1
    assert draw["top1"] == 0
    assert home["top1_tied"] is draw["top1_tied"] is False
    summary = summarize_rows([{"status": "predicted", **home}, {"status": "predicted", **draw}])
    assert summary["top1_tie_count"] == 0


def test_summary_uses_predicted_rows_only_and_keeps_arithmetic_loss_means():
    first = {"status": "predicted", **score_forecast((1.5, 0.8), (1, 0))}
    second = {"status": "predicted", **score_forecast((0.8, 1.5), (1, 0))}
    missing = {"status": "missing", "reason": "unseen_team_no_estimated_prior",
               "one_x_two_log_loss": -999, "top1": -999}
    result = summarize_rows([first, missing, second])
    assert result["n"] == 2
    for key in ("full_score_nll", "one_x_two_log_loss", "one_x_two_brier",
                "top1", "total_log_loss", "total_brier"):
        assert result[key] == pytest.approx((first[key] + second[key]) / 2)


def test_empty_summary_has_no_invented_zero_accuracy_or_loss():
    result = summarize_rows([{"status": "missing", "reason": "no earlier data"}])
    assert result["n"] == 0
    assert all(result.get(key) is None for key in (
        "full_score_nll", "one_x_two_log_loss", "one_x_two_brier",
        "top1", "total_log_loss", "total_brier"))


def rolling_records():
    return [
        historical("old-epl", date(2024, 9, 1), season="2024-25"),
        historical("old-champ", date(2024, 9, 2), "Hull", "Coventry",
                   competition="E1", season="2024-25"),
        historical("old-champ-second", date(2024, 10, 1), "Coventry", "Hull",
                   competition="E1", season="2024-25"),
        historical("target-promoted-early", date(2025, 8, 9), "Hull", "Arsenal"),
        historical("target-known", date(2025, 8, 15)),
        historical("target-at-next-cutoff", date(2025, 8, 29), "Hull", "Chelsea"),
        historical("target-later", date(2025, 9, 6), "Chelsea", "Hull"),
        # Championship result inside the target period may update later fits,
        # but it is not one more Premier League scoring unit.
        historical("current-champ", date(2025, 8, 10), "Coventry", "Hull",
                   competition="E1"),
        # These dates are outside the requested evaluation bounds.
        historical("after-stop", date(2025, 9, 12), "Hull", "Arsenal"),
        historical("next-season", date(2026, 8, 8), season="2026-27"),
    ]


class FakeFactory:
    def __init__(self, output_dir, *, fit_error=None):
        self.output_dir = output_dir
        self.instances = []
        self.predictions = []
        self.fit_error = fit_error

    def __call__(self):
        factory = self
        token = f"synthetic-model-{len(self.instances)}"

        class FakeModel:
            def fit(self, records, before_date):
                if factory.fit_error is not None:
                    raise factory.fit_error
                self.records = list(records)
                self.before_date = before_date
                assert self.records
                assert all(r.match_date < before_date for r in self.records)
                assert len(self.records) == len({r.match_id for r in self.records})
                self.teams = {team for r in self.records for team in (r.home, r.away)}
                self.competitions = {r.competition for r in self.records}
                return self

            def parameters(self):
                return {
                    "_fake_model_token": token,
                    "before_date": self.before_date.isoformat(),
                    "n_matches": len(self.records),
                    "teams": sorted(self.teams),
                    "competitions": sorted(self.competitions),
                    "training_min_date": min(r.match_date for r in self.records).isoformat(),
                    "training_max_date": max(r.match_date for r in self.records).isoformat(),
                    "reference_match_ids": [r.match_id for r in self.records],
                    "not_for": "synthetic model, not any prediction",
                }

            def predict(self, home, away, competition, match_date):
                # An order assertion, not a post-hoc file-existence test.
                saved = [json.loads(p.read_text(encoding="utf-8"))
                         for p in (factory.output_dir / "parameters").glob("*.json")]
                assert any(p.get("parameters", {}).get("_fake_model_token") == token for p in saved), \
                    "parameters must be saved before the first prediction"
                assert home in self.teams and away in self.teams
                assert competition in self.competitions
                assert self.before_date <= match_date
                factory.predictions.append((token, home, away, competition, match_date))
                return (1.5, 0.8)

        model = FakeModel()
        self.instances.append(model)
        return model


def test_rolling_comparison_uses_earlier_dates_and_one_target_row_per_fixture(tmp_path):
    records = rolling_records()
    untouched = deepcopy(records)
    factory = FakeFactory(tmp_path)
    result = run_comparison(records, tmp_path, model_factory=factory,
                            start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=28)
    assert records == untouched
    expected_ids = {"target-promoted-early", "target-known", "target-at-next-cutoff", "target-later"}
    rows = result["per_fixture_rows"]
    assert len(rows) == len(expected_ids)
    assert {r["match_id"] for r in rows} == expected_ids
    assert all(set(r["arms"]) == {"epl_only", "joint"} for r in rows)
    assert result["fit_count"] == len(factory.instances) == 4
    assert {m.before_date for m in factory.instances} == {date(2025, 8, 1), date(2025, 8, 29)}
    assert len(result["fits"]) == 4
    for snapshot in result["fits"]:
        path = tmp_path / "parameters" / f"{snapshot['arm']}_{snapshot['cutoff']}.json"
        assert path.exists()
        snapshot_payload = json.loads(path.read_text(encoding="utf-8"))
        assert snapshot_payload["arm"] == snapshot["arm"]
        assert snapshot_payload["fit_cutoff_date"] == snapshot["cutoff"]
        assert snapshot_payload["saved_at"]
        params = snapshot_payload["parameters"]
        assert params["not_for"] == "synthetic model, not any prediction"
        assert date.fromisoformat(params["training_max_date"]) < date.fromisoformat(snapshot["cutoff"])
        assert "after-stop" not in params["reference_match_ids"]
        assert "next-season" not in params["reference_match_ids"]
        if snapshot["arm"] == "epl_only":
            assert params["competitions"] == ["E0"]
        else:
            assert params["competitions"] == ["E0", "E1"]
    assert len(factory.predictions) == 7


def test_unknown_team_has_explicit_missing_prediction_not_zero_strength(tmp_path):
    factory = FakeFactory(tmp_path)
    result = run_comparison(rolling_records(), tmp_path, model_factory=factory,
                            start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=28)
    by_id = {r["match_id"]: r for r in result["per_fixture_rows"]}
    early = by_id["target-promoted-early"]
    assert early["promoted_fixture"] is True
    missing = early["arms"]["epl_only"]
    assert missing["status"] == "missing"
    assert missing["reason"] == "unseen_team_no_estimated_prior"
    assert missing["missing_teams"] == ["england:Hull"]
    assert missing.get("rates") is None
    assert missing.get("forecast") is None
    assert early["arms"]["joint"]["status"] == "predicted"
    # EPL-only can learn Hull only in the subsequent window, not within Aug1.
    assert by_id["target-at-next-cutoff"]["arms"]["epl_only"]["status"] == "predicted"
    assert by_id["target-later"]["arms"]["epl_only"]["status"] == "predicted"


def test_fit_failure_is_not_replaced_by_another_arm_or_fake_prior(tmp_path):
    factory = FakeFactory(tmp_path, fit_error=RuntimeError("synthetic optimizer failure"))
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        run_comparison(rolling_records(), tmp_path, model_factory=factory,
                       start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=28)
    assert len(factory.instances) == 1
    assert factory.predictions == []


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_interval_is_rejected_before_fitting(tmp_path, interval):
    factory = FakeFactory(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        run_comparison(rolling_records(), tmp_path, model_factory=factory,
                       start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=interval)
    assert not factory.instances


def test_duplicate_target_id_is_not_counted_as_more_evaluation_data(tmp_path):
    rows = rolling_records()
    rows.append(deepcopy(rows[3]))
    factory = FakeFactory(tmp_path)
    with pytest.raises(ValueError):
        run_comparison(rows, tmp_path, model_factory=factory,
                       start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=28)
    assert not factory.instances

