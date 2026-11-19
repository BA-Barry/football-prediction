"""Synthetic, hand-calculated and failure-path tests of 1X2 calibration."""

from copy import deepcopy
from datetime import date, datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax

from football_model import prematch_calibration as calibration
from football_model.prematch_calibration import apply_calibrator, fit_calibrator


BEFORE = date(2024, 8, 1)


def row(mid, label, probabilities=(.2, .3, .5), day="2024-06-01"):
    return {"match_id": mid, "date": day, "probabilities": list(probabilities),
            "actual_outcome": label}


def rows():
    return [row("a", 0), row("b", 1), row("c", 2)]


def grouped_rows(probabilities, counts, prefix="g"):
    return [row(f"{prefix}-{label}-{j}", label, probabilities)
            for label, count in enumerate(counts) for j in range(count)]


def spec(method="class_bias", **parameters):
    if not parameters:
        parameters = {"bias": [0., 0., 0.]} if method == "class_bias" else {"beta": 1., "temperature": 1.}
    return {"method": method, "parameters": parameters}


@pytest.mark.parametrize("method", calibration.METHODS)
@pytest.mark.parametrize("p", [[.2, .3, .5], [[.2, .3, .5], [.7, .1, .2]]])
def test_identity_transform_preserves_probabilities_shape_and_input(method, p):
    original = deepcopy(p)
    q = apply_calibrator(p, spec(method))
    assert isinstance(q, np.ndarray)
    assert q.shape == np.asarray(p).shape
    np.testing.assert_allclose(q, p, rtol=1e-14, atol=1e-15)
    assert p == original


def test_bias_matches_hand_computed_multiplicative_odds_correction():
    p = [.2, .3, .5]
    expected = np.array([.4, .9, .5]) / 1.8
    actual = apply_calibrator(p, spec(bias=[np.log(2), np.log(3), 0.]))
    np.testing.assert_allclose(actual, expected)
    assert actual.sum() == pytest.approx(1.)
    assert actual[0] / actual[2] == pytest.approx(2 * p[0] / p[2])
    assert actual[1] / actual[2] == pytest.approx(3 * p[1] / p[2])


@pytest.mark.parametrize("beta", [0., .2, 1., 2.5])
def test_temperature_matches_power_formula_without_clipping(beta):
    p = np.array([[.1, .3, .6], [.8, .15, .05]])
    expected = p ** beta
    expected /= expected.sum(axis=1, keepdims=True)
    q = apply_calibrator(p, spec("temperature", beta=beta, temperature=1/beta if beta else None))
    np.testing.assert_allclose(q, expected, atol=1e-15)
    if beta:
        np.testing.assert_array_equal(q.argmax(axis=1), p.argmax(axis=1))
    else:
        np.testing.assert_allclose(q, np.full((2, 3), 1/3))


def test_temperature_beta_only_metadata_is_allowed():
    np.testing.assert_allclose(apply_calibrator([.2, .3, .5], spec("temperature", beta=0)), np.ones(3)/3)


def test_log_softmax_avoids_loss_of_normalization_from_large_common_offset():
    # Compare its general mathematical primitive, not an invalid away anchor.
    logits = np.array([[1e200, 1e200, 1e200]])
    np.testing.assert_allclose(np.exp(calibration._log_softmax(logits)), [[1/3]*3])


@pytest.mark.parametrize("objective,theta", [
    (calibration._class_bias_objective, np.array([.4, -.2])),
    (calibration._temperature_objective, np.array([.7])),
    (calibration._temperature_objective, np.array([0.])),
])
def test_analytic_gradient_matches_central_finite_difference(objective, theta):
    p = np.array([[.1, .4, .5], [.6, .1, .3], [.2, .6, .2], [.4, .4, .2]])
    labels = np.array([0, 1, 2, 0])
    log_p = np.log(p)
    loss, gradient = objective(theta, log_p, labels)
    expected = []
    for index in range(len(theta)):
        step = np.zeros(len(theta))
        step[index] = 1e-6
        expected.append((objective(theta+step, log_p, labels)[0] - objective(theta-step, log_p, labels)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-9)
    logits = log_p + [theta[0], theta[1], 0.] if len(theta) == 2 else theta[0] * log_p
    assert loss == pytest.approx(-np.log(softmax(logits, axis=1)[np.arange(4), labels]).mean())


def test_bias_recovers_known_offsets_for_several_probability_groups():
    training = (grouped_rows([1/3, 1/6, .5], [40, 30, 30], "one") +
                grouped_rows([.2, .4, .4], [20, 60, 20], "two") +
                grouped_rows([.6, .2, .2], [60, 30, 10], "three"))
    fitted = fit_calibrator(training, BEFORE, "class_bias")
    np.testing.assert_allclose(fitted["parameters"]["bias"], [np.log(2), np.log(3), 0.], atol=2e-7)
    p = np.array([r["probabilities"] for r in training])
    q = apply_calibrator(p, fitted)
    np.testing.assert_allclose(q.mean(axis=0), np.array([120, 120, 60]) / 300, atol=2e-8)
    assert fitted["calibrated_training_log_loss"] < fitted["raw_training_log_loss"]
    assert fitted["optimization"]["gradient_norm_inf"] < 2e-8


def test_temperature_recovers_known_beta_two():
    target = np.array([.2, .3, .5])
    p = np.sqrt(target)
    p /= p.sum()
    fitted = fit_calibrator(grouped_rows(p, [20, 30, 50]), BEFORE, "temperature")
    assert fitted["parameters"]["beta"] == pytest.approx(2., abs=2e-7)
    assert fitted["parameters"]["temperature"] == pytest.approx(.5, abs=1e-7)
    np.testing.assert_allclose(apply_calibrator(p, fitted), target, atol=2e-8)
    assert fitted["calibrated_training_log_loss"] < fitted["raw_training_log_loss"]


def test_temperature_can_reach_zero_beta_boundary_and_report_projected_gradient():
    training = []
    for label in range(3):
        p = np.full(3, .45)
        p[label] = .1
        training.extend(row(f"{label}-{i}", label, p) for i in range(10))
    fitted = fit_calibrator(training, BEFORE, "temperature")
    assert fitted["parameters"] == {"beta": 0., "temperature": None}
    assert fitted["calibrated_training_log_loss"] == pytest.approx(np.log(3))
    assert fitted["optimization"]["gradient"][0] > 0
    assert fitted["optimization"]["projected_gradient_norm_inf"] == 0
    json.dumps(fitted, allow_nan=False)


@pytest.mark.parametrize("method", calibration.METHODS)
def test_uniform_forecasts_balanced_labels_have_identity_optimum(method):
    training = [row(str(i), i % 3, (1/3, 1/3, 1/3)) for i in range(30)]
    fitted = fit_calibrator(training, BEFORE, method)
    assert fitted["parameters"] == spec(method)["parameters"]
    assert fitted["optimization"]["iterations"] == 0
    assert fitted["calibrated_training_log_loss"] == pytest.approx(fitted["raw_training_log_loss"])


@pytest.mark.parametrize("method", calibration.METHODS)
def test_saved_fit_metadata_is_chronological_complete_and_json_serializable(method):
    training = rows()
    training[0]["date"] = date(2024, 7, 1)
    training[1]["date"] = "2024-05-01"
    fitted = fit_calibrator(training, BEFORE, method)
    assert fitted["before_date"] == BEFORE.isoformat()
    assert fitted["training_min_date"] == "2024-05-01"
    assert fitted["training_max_date"] == "2024-07-01"
    assert fitted["training_match_ids"] == ["b", "c", "a"]
    assert fitted["class_counts"] == [1, 1, 1]
    assert fitted["n_matches"] == 3
    assert fitted["raw_training_nll"] == pytest.approx(fitted["raw_training_log_loss"] * 3)
    assert fitted["calibrated_training_nll"] == pytest.approx(fitted["calibrated_training_log_loss"] * 3)
    opt = fitted["optimization"]
    assert opt["algorithm"] == "L-BFGS-B"
    assert opt["regularization"] is None
    assert opt["objective"] == "mean_multiclass_negative_log_likelihood"
    assert opt["success"] is True
    assert opt["function_evaluations"] >= 1
    assert opt["gradient_evaluations"] >= 1
    assert opt["initial_objective"] == pytest.approx(fitted["raw_training_log_loss"])
    assert opt["final_objective"] == fitted["calibrated_training_log_loss"]
    assert opt["options"] == {"gtol": 1e-9, "ftol": 1e-13, "maxiter": 1000}
    assert len(opt["gradient"]) == (2 if method == "class_bias" else 1)
    json.dumps(fitted, allow_nan=False)


@pytest.mark.parametrize("method", calibration.METHODS)
def test_fit_does_not_mutate_rows_and_is_invariant_to_input_order(method):
    training = grouped_rows([.2, .3, .5], [7, 8, 15])
    original = deepcopy(training)
    result = fit_calibrator(training, BEFORE, method)
    assert training == original
    np.random.default_rng(73).shuffle(training)
    shuffled = fit_calibrator(training, BEFORE, method)
    assert result == shuffled


@pytest.mark.parametrize("method", calibration.METHODS)
def test_market_and_unrelated_fields_do_not_enter_fitting(method):
    training = rows()
    original = fit_calibrator(training, BEFORE, method)
    for r in training:
        r["market_probabilities"] = {"unrelated": [0., 0., 1.]}
        r["future_field"] = float("nan")
    assert fit_calibrator(training, BEFORE, method) == original


@pytest.mark.parametrize("bad", [[], {}, None, [None], [1], "not rows"])
def test_empty_or_invalid_training_rows_rejected(bad):
    with pytest.raises(ValueError):
        fit_calibrator(bad, BEFORE, "class_bias")


@pytest.mark.parametrize("bad", ["2024-08-01", datetime(2024, 8, 1), None, 20240801])
def test_before_date_requires_date_not_implicit_parse_or_timestamp(bad):
    with pytest.raises(ValueError, match="before_date"):
        fit_calibrator(rows(), bad, "class_bias")


@pytest.mark.parametrize("method", [None, "", "identity", "Temperature", 1, True])
def test_unknown_method_rejected_for_fit_and_apply(method):
    with pytest.raises(ValueError, match="method"):
        fit_calibrator(rows(), BEFORE, method)
    with pytest.raises(ValueError, match="method"):
        apply_calibrator([.2, .3, .5], spec(method, bias=[0., 0., 0.]))


@pytest.mark.parametrize("day", ["2024-08-01", "2024-08-02", date(2024, 9, 1)])
def test_future_or_cutoff_date_is_error_not_silently_filtered(day):
    training = rows() + [row("future", 0, day=day)]
    with pytest.raises(ValueError, match="strictly before"):
        fit_calibrator(training, BEFORE, "class_bias")


@pytest.mark.parametrize("day", [None, "20240601", "2024-6-1", "2024-06-01T00:00:00", "2024-02-30", 20240601,
                                 datetime(2024, 6, 1)])
def test_invalid_or_ambiguous_row_date_rejected(day):
    training = rows()
    training[0]["date"] = day
    with pytest.raises(ValueError, match="date"):
        fit_calibrator(training, BEFORE, "class_bias")


def test_duplicate_match_ids_rejected_even_on_different_dates():
    training = rows() + [row("a", 2, day="2024-07-01")]
    with pytest.raises(ValueError, match="duplicate match_id"):
        fit_calibrator(training, BEFORE, "temperature")


@pytest.mark.parametrize("mid", [None, "", "  ", 1, [], {}])
def test_invalid_match_id_rejected(mid):
    training = rows()
    training[0]["match_id"] = mid
    with pytest.raises(ValueError, match="match_id"):
        fit_calibrator(training, BEFORE, "class_bias")


@pytest.mark.parametrize("label", [True, np.bool_(False), 1.0, .5, -1, 3, float("nan"), "1", None])
def test_invalid_or_noninteger_labels_rejected(label):
    training = rows()
    training[0]["actual_outcome"] = label
    with pytest.raises(ValueError, match="integer label"):
        fit_calibrator(training, BEFORE, "class_bias")


@pytest.mark.parametrize("method", calibration.METHODS)
@pytest.mark.parametrize("missing", [0, 1, 2])
def test_missing_outcome_class_is_not_zero_effect_prior(method, missing):
    with pytest.raises(ValueError, match="all three"):
        fit_calibrator([r for r in rows() if r["actual_outcome"] != missing], BEFORE, method)


@pytest.mark.parametrize("bad", [[], [1/3, 2/3], [1/4]*4, [0., .5, .5], [-.1, .6, .5],
                                 [.2, .3, .6], [.2, .3, float("nan")], [.2, .3, float("inf")],
                                 [".2", ".3", ".5"], [True, False, False], [1j, .5, .5],
                                 [[[.2, .3, .5]]], None, [[.2, .3, .5], [.5, .5]]])
def test_invalid_probabilities_not_repaired_or_clipped(bad):
    with pytest.raises(ValueError):
        apply_calibrator(bad, spec())
    training = rows()
    training[0]["probabilities"] = bad
    with pytest.raises(ValueError):
        fit_calibrator(training, BEFORE, "class_bias")


def test_single_training_row_cannot_smuggle_a_matrix():
    training = rows()
    training[0]["probabilities"] = [[.2, .3, .5]]
    with pytest.raises(ValueError, match="one .* vector"):
        fit_calibrator(training, BEFORE, "class_bias")


@pytest.mark.parametrize("bad", [None, [], {}, {"method": "class_bias"},
                                 {"method": "class_bias", "parameters": []},
                                 spec(bias=[0., 0.]), spec(bias=[0., 0., 1.]),
                                 spec(bias=[float("nan"), 0., 0.]), spec(bias=[0., float("inf"), 0.]),
                                 spec(bias=["0", "0", "0"])])
def test_invalid_saved_bias_model_rejected(bad):
    with pytest.raises(ValueError):
        apply_calibrator([.2, .3, .5], bad)


@pytest.mark.parametrize("parameters", [{}, {"beta": -1.}, {"beta": float("nan")}, {"beta": float("inf")},
                                        {"beta": [1.]}, {"beta": True}, {"beta": "1"},
                                        {"beta": 0., "temperature": float("inf")},
                                        {"beta": 0., "temperature": 1.}, {"beta": 2., "temperature": None},
                                        {"beta": 2., "temperature": 2.}, {"beta": 2., "temperature": -.5},
                                        {"beta": 2., "temperature": float("nan")}])
def test_invalid_temperature_model_rejected(parameters):
    with pytest.raises(ValueError):
        apply_calibrator([.2, .3, .5], {"method": "temperature", "parameters": parameters})


@pytest.mark.parametrize("parameters", [spec(bias=[1000., 0., 0.]), spec("temperature", beta=1e308)])
def test_unrepresentable_transforms_fail_without_zero_replacement(parameters):
    with pytest.raises(ValueError, match="underflow|non-finite"):
        apply_calibrator([.2, .3, .5], parameters)


def test_strictly_positive_tiny_inputs_do_not_get_epsilon_clipped():
    p = [1e-280, .4, .6]
    q = apply_calibrator(p, spec())
    assert q[0] == pytest.approx(1e-280, rel=1e-12, abs=0)


def test_probability_upper_bound_is_not_relaxed_by_sum_tolerance():
    p = [1.00000000001, 1e-12, 1e-12]
    with pytest.raises(ValueError, match="at most one"):
        apply_calibrator(p, spec())
    training = rows()
    training[0]["probabilities"] = p
    with pytest.raises(ValueError, match="at most one"):
        fit_calibrator(training, BEFORE, "class_bias")


def test_exact_one_with_unrepresentably_small_positive_remainder_is_allowed():
    # IEEE addition rounds this sum to one. Enforce the component bound <=1,
    # rather than inventing an unrepresentable subtraction from its largest p.
    p = [1., 1e-280, 1e-280]
    q = apply_calibrator(p, spec())
    np.testing.assert_allclose(q, p, rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", calibration.METHODS)
def test_optimizer_unsuccessful_status_is_visible_not_raw_fallback(monkeypatch, method):
    monkeypatch.setattr(calibration, "minimize", lambda *a, **k: SimpleNamespace(success=False, status=2, message="line search failed"))
    with pytest.raises(RuntimeError, match=r"status=2.*line search failed"):
        fit_calibrator(rows(), BEFORE, method)


def test_optimizer_exception_is_visible_with_original_cause(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(calibration, "minimize", fail)
    with pytest.raises(RuntimeError, match="synthetic failure") as error:
        fit_calibrator(rows(), BEFORE, "class_bias")
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("field,value", [("x", [float("nan"), 0.]), ("x", [0.]),
                                         ("fun", float("inf")), ("jac", [float("nan"), 0.])])
def test_success_flag_does_not_hide_nonfinite_or_malformed_optimizer_output(monkeypatch, field, value):
    result = {"success": True, "x": np.zeros(2), "fun": 1., "jac": np.zeros(2)}
    result[field] = value
    monkeypatch.setattr(calibration, "minimize", lambda *a, **k: SimpleNamespace(**result))
    with pytest.raises(RuntimeError, match="invalid fit"):
        fit_calibrator(rows(), BEFORE, "class_bias")


def test_success_flag_does_not_hide_negative_beta(monkeypatch):
    fake = SimpleNamespace(success=True, x=[-1.], fun=1., jac=[0.])
    monkeypatch.setattr(calibration, "minimize", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="non-negative"):
        fit_calibrator(rows(), BEFORE, "temperature")
