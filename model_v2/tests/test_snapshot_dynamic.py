"""Small-budget dynamic residual behavior; synthetic data are not accuracy proof."""

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from football_model import snapshot_dynamic as dynamic
from football_model.forecast_diagnostics import validate_prediction
from football_model.probability import score_grid, summarize
from test_goal_rate_experiment import prediction


def row(key, day, score=(3, 0), rates=(1., 1.), comp="E0", home="A", away="B", cutoff=None):
    result = prediction(key, day, score, rates=rates, comp=comp)
    result.update(rates=list(rates), home=home, away=away)
    if cutoff is not None:
        result["fit_before_date"] = cutoff
    return result


def sample():
    return [row("a", "2024-08-02"), row("b", "2024-08-04", (1, 1)),
            row("c", "2024-08-06", (0, 2)), row("d", "2024-09-03", (2, 1))]


def test_zero_eta_preserves_exact_original_rates_full_probabilities_and_inputs():
    rows = sample()
    before = deepcopy(rows)
    result = dynamic.filter_rows(reversed(rows), 0)
    assert rows == before
    for source, generated in zip(rows, result):
        assert generated["rates"] == source["rates"]
        assert generated["forecast"] == source["forecast"]
        assert generated["probabilities"] == source["probabilities"]
        assert generated["base_rates"] == source["rates"]
        assert generated["base_fit_before_date"] == source["fit_before_date"]
        assert generated["base_training_max_date"] == source["training_max_date"]
        assert generated["fit_before_date"] == source["date"]
        validate_prediction(generated)
    result[0]["market_probabilities"]["changed"] = [.2, .3, .5]
    result[0]["actual_score"][0] = 99
    assert rows == before


def test_known_two_goal_residual_moves_attack_and_opponent_defense_half_steps():
    rows = [row("a", "2024-08-02", (3, 0)), row("b", "2024-08-04", (1, 1))]
    result = dynamic.filter_rows(rows, .1)
    state = result[1]["dynamic_state"]
    assert state["home_attack"] == pytest.approx(.1)
    assert state["away_defense"] == pytest.approx(-.1)
    assert state["away_attack"] == pytest.approx(-.05)
    assert state["home_defense"] == pytest.approx(.05)
    assert result[1]["rates"] == pytest.approx([math.exp(.2), math.exp(-.1)])
    assert state["dynamic_history_ids"] == ["a"]
    assert state["last_update_date"] == "2024-08-02"
    assert result[1]["training_matches"] == rows[1]["training_matches"] + 1
    assert result[1]["training_max_date"] == "2024-08-02"
    assert result[1]["forecast"] == summarize(score_grid(*result[1]["rates"], rho=0))
    nll = sum(rate - goal*math.log(rate) + math.lgamma(goal+1)
              for rate, goal in zip(result[1]["rates"], rows[1]["actual_score"]))
    assert result[1]["exact_score_nll"] == pytest.approx(nll)


def test_new_snapshot_resets_entire_competition_residual_not_only_teams_in_first_game():
    rows = [row("a", "2024-08-02"),
            row("b", "2024-08-30", home="C", away="D"),
            row("c", "2024-08-31", (1, 1))]
    result = dynamic.filter_rows(rows, .1)
    assert result[1]["dynamic_state"]["snapshot_reset"] is True
    assert result[1]["dynamic_state"]["dynamic_history_ids"] == []
    assert result[2]["rates"] == rows[2]["rates"]
    assert result[2]["dynamic_state"]["dynamic_history_ids"] == ["b"]
    assert result[2]["dynamic_state"]["snapshot_reset"] is False


def test_same_day_reordering_and_score_changes_cannot_affect_same_day_rates():
    rows = [row("a", "2024-08-02"), row("b", "2024-08-04"),
            row("c", "2024-08-04", (0, 4)), row("d", "2024-08-06")]
    original = dynamic.filter_rows(rows, .1)
    assert dynamic.filter_rows(reversed(rows), .1) == original
    changed = deepcopy(rows)
    changed[1]["actual_score"] = [8, 0]
    alternative = dynamic.filter_rows(changed, .1)
    assert [r["rates"] for r in alternative[:3]] == [r["rates"] for r in original[:3]]
    assert alternative[3]["rates"] != original[3]["rates"]
    for generated in original[1:3]:
        assert generated["dynamic_state"]["dynamic_history_ids"] == ["a"]
        assert generated["training_max_date"] == "2024-08-02"
    assert original[3]["dynamic_state"]["dynamic_history_ids"] == ["a", "b", "c"]


def test_cross_competition_shared_labels_and_snapshot_changes_are_isolated():
    rows = [row("a", "2024-08-02", comp="E0"),
            row("b", "2024-08-02", (0, 4), comp="D1"),
            row("c", "2024-08-04", comp="E0"), row("d", "2024-08-04", comp="D1")]
    result = dynamic.filter_rows(rows, .1)
    expected_home = dynamic.filter_rows([rows[0], rows[2]], .1)
    expected_away = dynamic.filter_rows([rows[1], rows[3]], .1)
    assert result[2] == expected_home[1]
    assert result[3] == expected_away[1]
    assert result[2]["rates"] != result[3]["rates"]


def test_later_score_perturbation_never_changes_any_earlier_prediction_or_history():
    rows = sample()
    result = dynamic.filter_rows(rows, .1)
    changed = deepcopy(rows)
    changed[-1]["actual_score"] = [9, 0]
    other = dynamic.filter_rows(changed, .1)
    assert other[:-1] == result[:-1]
    assert other[-1]["rates"] == result[-1]["rates"]
    assert other[-1]["dynamic_state"] == result[-1]["dynamic_state"]


def test_rate_only_mode_explicitly_omits_stale_original_forecasts():
    rows = sample()
    cheap = dynamic.filter_rows(rows, .1, forecasts=False)
    full = dynamic.filter_rows(rows, .1)
    for a, b in zip(cheap, full):
        assert a["rates"] == b["rates"]
        assert a["exact_score_nll"] == b["exact_score_nll"]
        assert "forecast" not in a and "probabilities" not in a
        assert a["dynamic_state"]["forecasts_rebuilt"] is False


@pytest.mark.parametrize("bad", [None, True, -.001, .201, float("nan"), float("inf"), "0.1"])
def test_invalid_learning_rate_is_not_coerced_or_clipped(bad):
    with pytest.raises(ValueError, match="eta"):
        dynamic.filter_rows(sample(), bad)


@pytest.mark.parametrize("problem", ["empty", "duplicate", "future_training", "future_fit",
    "rates_zero", "rates_bool", "rates_mismatch", "research", "noncanonical_date",
    "training_count", "backwards_snapshot", "conflicting_same_day_snapshot"])
def test_invalid_original_inputs_are_not_filtered_repaired_or_layered(problem):
    rows = sample()
    if problem == "empty":
        rows = []
    elif problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "future_training":
        rows[0]["training_max_date"] = rows[0]["date"]
    elif problem == "future_fit":
        rows[0]["fit_before_date"] = "2025-08-01"
    elif problem == "rates_zero":
        rows[0]["rates"][0] = 0
    elif problem == "rates_bool":
        rows[0]["rates"][0] = True
    elif problem == "rates_mismatch":
        rows[0]["rates"][0] += .1
    elif problem == "research":
        rows[0]["cmp_nu"] = None
    elif problem == "noncanonical_date":
        rows[0]["date"] = "20240802"
    elif problem == "training_count":
        rows[0]["training_matches"] = True
    elif problem == "backwards_snapshot":
        rows[1]["fit_before_date"] = "2024-07-31"
        rows[1]["training_max_date"] = "2024-07-30"
    elif problem == "conflicting_same_day_snapshot":
        rows[1]["date"] = rows[0]["date"]
        rows[1]["fit_before_date"] = "2024-07-31"
        rows[1]["training_max_date"] = "2024-07-30"
    before = deepcopy(rows)
    with pytest.raises(ValueError):
        dynamic.filter_rows(rows, .1)
    assert rows == before


@pytest.mark.parametrize("day", ["2024-07-31", "2025-07-01", "2025-08-01", "2026-01-01"])
def test_fit_rejects_rows_before_or_after_the_declared_training_season(day):
    with pytest.raises(ValueError, match="only 2024/25"):
        dynamic.fit_learning_rate([row("outside", day)])


def test_real_scalar_fit_records_all_evaluations_and_includes_exact_endpoints():
    rows = [row(f"m{i}", f"2024-08-{i+2:02d}", score=score)
            for i, score in enumerate(((3, 0), (2, 1), (1, 2)))]
    before = deepcopy(rows)
    fitted = dynamic.fit_learning_rate(rows)
    assert rows == before
    assert 0 < fitted["eta"] < .2
    assert fitted["selected_stage"] == "optimizer"
    assert fitted["selected_on_boundary"] is False
    assert fitted["fallback"] is False
    assert fitted["optimizer_success"] is True
    assert fitted["optimizer"]["nfev"] <= 12
    assert fitted["n_evaluations"] == len(fitted["evaluations"]) <= 14
    assert [v["eta"] for v in fitted["evaluations"][:2]] == [0., .2]
    assert all(v["numerical_status"] == "finite" for v in fitted["evaluations"])
    assert fitted["training_match_ids"] == [r["match_id"] for r in rows]
    assert fitted["training_min_date"] == rows[0]["date"]
    assert fitted["training_max_date"] == rows[-1]["date"] < fitted["before_date"]
    assert fitted["training_objective_not_blind_validation"] is True
    for evaluation in fitted["evaluations"]:
        applied = dynamic.filter_rows(rows, evaluation["eta"], forecasts=False)
        assert evaluation["nll"] == pytest.approx(sum(r["exact_score_nll"] for r in applied) / len(rows))


def test_exact_zero_endpoint_can_win_a_successful_search_without_being_a_fallback(monkeypatch):
    rows = [row(f"m{i}", f"2024-08-{i+2:02d}", score=(1, 1)) for i in range(3)]

    def controlled_search(objective, **kwargs):
        value = objective(.1)
        return SimpleNamespace(success=True, message="test-only successful search", x=.1, fun=value, nfev=1)

    monkeypatch.setattr(dynamic, "minimize_scalar", controlled_search)
    fitted = dynamic.fit_learning_rate(rows)
    assert fitted["eta"] == 0
    assert fitted["selected_stage"] == "lower_bound"
    assert fitted["selected_on_boundary"] is True
    assert fitted["fallback"] is False
    assert fitted["n_evaluations"] == 3


def test_real_flat_objective_may_exhaust_fixed_budget_but_is_not_retried():
    rows = [row(f"m{i}", f"2024-08-{i+2:02d}", score=(1, 1)) for i in range(3)]
    with pytest.raises(dynamic.DynamicFitError, match="did not converge") as caught:
        dynamic.fit_learning_rate(rows)
    assert caught.value.optimizer["nfev"] == 12
    assert len(caught.value.evaluations) == 14
    assert {entry["nll"] for entry in caught.value.evaluations} == {2.}


def test_nonconverged_optimizer_is_explicit_failure_not_zero_endpoint_fallback(monkeypatch):
    monkeypatch.setattr(dynamic, "minimize_scalar", lambda *a, **k:
        SimpleNamespace(success=False, message="budget exhausted", x=.01, fun=2., nfev=12))
    with pytest.raises(dynamic.DynamicFitError, match="did not converge") as caught:
        dynamic.fit_learning_rate(sample())
    assert [entry["eta"] for entry in caught.value.evaluations] == [0., .2]
    assert caught.value.optimizer["success"] is False


def test_numerical_error_retains_trace_and_never_clips_rates_or_restarts(monkeypatch):
    calls = []
    original = dynamic._run

    def fail(records, eta, **options):
        calls.append(eta)
        if eta > 0:
            raise OverflowError("test-only overflow")
        return original(records, eta, **options)

    monkeypatch.setattr(dynamic, "_run", fail)
    with pytest.raises(dynamic.DynamicFitError, match="evaluation failed") as caught:
        dynamic.fit_learning_rate(sample())
    assert calls == [0., .2]
    assert caught.value.evaluations[-1]["numerical_status"] == "error"
    assert caught.value.evaluations[-1]["nll"] is None
