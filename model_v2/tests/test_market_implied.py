"""Real dependency calls plus independent margin equations and failure cases."""

from datetime import datetime, timezone
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from football_model.domain import OddsSnapshot
from football_model.market_implied import IMPLIED_METHODS, dewater_1x2
import football_model.market_implied as market_implied
from football_model.probability import implied


@pytest.mark.parametrize("odds", [[2.7, 2.3, 4.4], [1.4, 4.5, 9.0], [3., 3., 3.]])
@pytest.mark.parametrize("method", IMPLIED_METHODS)
def test_real_library_and_independent_equations(odds, method):
    result = dewater_1x2(odds, method)
    raw = [1 / x for x in odds]
    total = math.fsum(raw)
    if method == "multiplicative":
        expected = [x / total for x in raw]
        assert result["method_params"] == {}
    elif method == "power":
        # Different root finder and a physical positive interval, not Ridder's
        # installed [0,100] implementation; the equation is independently coded.
        exponent = brentq(lambda k: math.fsum(x ** k for x in raw) - 1, .0001, 50)
        expected = [x ** exponent for x in raw]
        assert result["method_params"]["k"] == pytest.approx(exponent, abs=1e-9)
    else:
        # Rationalized Shin expression avoids subtracting nearby square roots.
        def probabilities(z):
            return [2 * x * x / total /
                    (math.sqrt(z*z + 4 * (1-z) * x*x / total) + z) for x in raw]
        if total == 1:
            z = 0.
        else:
            z = brentq(lambda value: math.fsum(probabilities(value)) - 1, 0, .999999)
        expected = probabilities(z)
        assert result["method_params"]["z"] == pytest.approx(z, abs=1e-9)
    assert result["probabilities"] == pytest.approx(expected, abs=1e-9)
    assert result["overround"] == pytest.approx(total - 1, abs=1e-15)
    assert result["odds"] == odds
    assert result["labels"] == ["home", "draw", "away"]
    assert result["probabilities_by_label"] == dict(zip(result["labels"], result["probabilities"]))
    assert result["method"] == method
    assert result["implementation"]["version"] == "1.13.1"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("odds", [[2.7, 2.3, 4.4], [1.01, 30, 80], [4., 4., 4.], [3., 3., 3.]])
def test_multiplicative_matches_original_exactly(odds):
    quote = OddsSnapshot(*odds, datetime(2024, 1, 1, tzinfo=timezone.utc), "test", "fixture")
    assert np.array_equal(np.asarray(dewater_1x2(odds)["probabilities"]), implied(quote))


@pytest.mark.parametrize("method", IMPLIED_METHODS)
def test_permuted_labels_and_original_objects_are_preserved(method):
    odds = [4.4, 2.7, 2.3]
    labels = ["away", "home", "draw"]
    result = dewater_1x2(odds, method, labels=labels)
    normal = dewater_1x2([2.7, 2.3, 4.4], method)
    assert result["labels"] == labels
    assert result["odds"] == odds
    assert result["probabilities_by_label"] == pytest.approx(normal["probabilities_by_label"])
    result["odds"][0] = 42
    result["labels"][0] = "changed"
    assert odds == [4.4, 2.7, 2.3]
    assert labels == ["away", "home", "draw"]


@pytest.mark.parametrize("odds", [None, "2.7,2.3,4.4", [], [2, 3], [2, 3, 4, 5],
    [True, 3, 4], [1, 3, 4], [0, 3, 4], [-2, 3, 4], [math.inf, 3, 4],
    [math.nan, 3, 4], ["2.7", 3, 4], [None, 3, 4]])
def test_rejects_invalid_prices(odds):
    with pytest.raises(ValueError, match="multiplicative"):
        dewater_1x2(odds)


@pytest.mark.parametrize("labels", [None, "HDA", [], ["H", "D"], ["H", "D", "H"],
    ["H", "D", ""], ["H", "D", "  "], ["H", "D", None]])
def test_rejects_invalid_outcome_names(labels):
    with pytest.raises(ValueError, match="labels"):
        dewater_1x2([2.7, 2.3, 4.4], labels=labels)


@pytest.mark.parametrize("method", [None, "additive", "unknown", "POWER", 1])
def test_only_prespecified_methods(method):
    with pytest.raises(ValueError, match="Unsupported"):
        dewater_1x2([2.7, 2.3, 4.4], method)


def test_real_shin_underround_failure_is_not_replaced():
    assert dewater_1x2([4., 4., 4.])["overround"] == -.25
    assert dewater_1x2([4., 4., 4.], "power")["method_params"]["k"] < 1
    with pytest.raises(ValueError, match="shin: penaltyblog margin calculation failed"):
        dewater_1x2([4., 4., 4.], "shin")


def test_real_power_finite_interval_failure_is_not_replaced():
    # The exponent required exceeds the installed solver's finite interval.
    odds = [1.0000001] * 3
    assert dewater_1x2(odds)["probabilities"] == [1/3] * 3
    with pytest.raises(ValueError, match="power: penaltyblog margin calculation failed"):
        dewater_1x2(odds, "power")


def _fake_result(**changes):
    values = {"probabilities": [.4, .3, .3], "method": SimpleNamespace(value="power"),
              "margin": sum(1/x for x in [2.7, 2.3, 4.4])-1,
              "market_names": ["home", "draw", "away"], "method_params": {"k": 1.03}}
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize("changes", [
    {"probabilities": [-.01, .5, .51]}, {"probabilities": [1.01, 0, -.01]},
    {"probabilities": [math.nan, .3, .3]}, {"probabilities": [.4, .3, .31]},
    {"probabilities": [.5, .5]}, {"method_params": {"k": math.inf}},
    {"method_params": {"k": 0}}, {"method_params": {}}, {"margin": .8},
    {"method": SimpleNamespace(value="multiplicative")},
    {"market_names": ["away", "draw", "home"]},
])
def test_invalid_library_results_are_never_clipped_or_relabelled(monkeypatch, changes):
    monkeypatch.setattr(market_implied.penaltyblog_implied, "calculate_implied",
                        lambda *args, **kwargs: _fake_result(**changes))
    with pytest.raises(ValueError, match="power:"):
        dewater_1x2([2.7, 2.3, 4.4], "power")


@pytest.mark.parametrize("z", [-.01, 1., math.nan])
def test_shin_parameter_domain(monkeypatch, z):
    monkeypatch.setattr(market_implied.penaltyblog_implied, "calculate_implied",
        lambda *args, **kwargs: _fake_result(method=SimpleNamespace(value="shin"),
                                            method_params={"z": z}))
    with pytest.raises(ValueError, match="shin:"):
        dewater_1x2([2.7, 2.3, 4.4], "shin")
