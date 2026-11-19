"""Check information timing and independently known offset effects."""

from copy import deepcopy
from datetime import date, timedelta
import math

import numpy as np
import pytest

from football_model.shot_increment import (
    OffsetShotModel, ShotObservation, history_features, poisson_offset_objective,
)


def observations():
    start = date(2024, 1, 1)
    return [ShotObservation(
        str(i), start + timedelta(days=i),
        "A" if i % 2 else "B", "B" if i % 2 else "A",
        (i, 30-i) if i % 2 else (30-i, i),
    ) for i in range(1, 13)]


def rows(n=30):
    start = date(2024, 1, 1)
    return [{
        "match_id": f"r-{i}", "date": (start+timedelta(days=i)).isoformat(),
        "fit_before_date": start.isoformat(),
        "shot_features": [[2+i % 5, 3+(i*3) % 7], [3+(i*2) % 5, 2+(i*5) % 7]],
        "rates": [1.1+(i % 2)*.2, 1.5],
        "actual_score": [i % 4, (i+2) % 3],
    } for i in range(n)]


def test_history_uses_last_ten_and_ignores_cutoff_day_and_future():
    source = observations()
    cutoff = date(2024, 1, 14)
    expected = history_features(source, "A", "B", cutoff)
    changed = source + [
        ShotObservation("same", cutoff, "A", "B", (999, 999)),
        ShotObservation("future", cutoff+timedelta(days=1), "B", "A", (900, 800)),
    ]
    assert history_features(changed, "A", "B", cutoff) == expected
    assert expected["history"]["A"]["match_ids"] == [str(i) for i in range(3, 13)]
    assert expected["features"] == [[7.5, 7.5], [22.5, 22.5]]
    assert expected["history"]["A"]["latest_gap_days"] == 1


def test_history_swapping_sides_swaps_prediction_rows():
    cutoff = date(2024, 1, 14)
    normal = history_features(observations(), "A", "B", cutoff)
    swapped = history_features(observations(), "B", "A", cutoff)
    assert swapped["features"] == normal["features"][::-1]
    assert swapped["history"] == normal["history"]


def test_missing_recent_shots_are_not_replaced_with_older_games_or_zero():
    source = observations()
    source[-1].shots_on_target = None
    result = history_features(source, "A", "B", date(2024, 1, 14))
    assert result["features"] is None
    assert result["history"]["A"]["n_matches"] == 10
    assert result["history"]["A"]["missing_match_ids"] == ["12"]
    assert result["history"]["A"]["for_mean"] is None
    source[-1].shots_on_target = (18, 12)
    source[0].shots_on_target = None
    assert history_features(source, "A", "B", date(2024, 1, 14))["features"] is not None


def test_insufficient_history_is_explicitly_unavailable():
    result = history_features(observations(), "A", "B", date(2024, 1, 4))
    assert result["features"] is None
    assert result["history"]["A"]["unavailable_reason"] == "insufficient_league_history"
    assert result["history"]["A"]["n_matches"] == 2


@pytest.mark.parametrize("field,bad", [
    ("actual_score", [True, 2]),
    ("rates", [1.5, False]),
    ("shot_features", [[True, 2], [3, 4]]),
])
def test_mixed_boolean_flags_cannot_be_silently_converted_to_counts(field, bad):
    data = rows()
    data[0][field] = bad
    with pytest.raises(ValueError, match="Boolean"):
        OffsetShotModel().fit(data, date(2024, 3, 1))


def test_objective_gradient_matches_independent_central_difference():
    x = np.array([[-1., .5], [.2, -.3], [2., 1.2], [-.4, -.8]])
    y = np.array([0., 1., 4., 2.])
    rates = np.array([1.1, 1.7, .8, 2.])
    weights = np.array([.1, .2, .3, .4])
    beta = np.array([.13, -.21])
    penalty = .07
    value, gradient = poisson_offset_objective(beta, x, y, rates, weights, penalty)
    def independent(b):
        eta = x @ b
        return sum(w*(r*math.exp(e)-g*e) for w, r, e, g in zip(weights, rates, eta, y)) + penalty*sum(b*b)/2
    eps = 1e-6
    numeric = np.array([(independent(beta+np.eye(2)[i]*eps)-independent(beta-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    assert value == pytest.approx(independent(beta), abs=1e-12)
    assert gradient == pytest.approx(numeric, rel=1e-8, abs=1e-9)


def test_fit_and_scaling_cannot_use_cutoff_day_or_future_rows():
    data = rows()
    cutoff = date(2024, 1, 21)
    early = OffsetShotModel().fit(data[:20], cutoff)
    mutated = deepcopy(data)
    for row in mutated[20:]:
        row["shot_features"] = [[1000, 900], [800, 700]]
        row["actual_score"] = [30, 40]
    with_future = OffsetShotModel().fit(mutated, cutoff)
    assert with_future.parameters() == early.parameters()
    weights = np.repeat([math.exp(-math.log(2)*(cutoff-date.fromisoformat(r["date"])).days/180) for r in data[:20]], 2)
    weights /= weights.sum()
    raw = np.array([r["shot_features"] for r in data[:20]], dtype=float).reshape(-1, 2)
    mean = np.average(raw, axis=0, weights=weights)
    scale = np.sqrt(np.average((raw-mean)**2, axis=0, weights=weights))
    assert with_future.mean == pytest.approx(mean)
    assert with_future.scale == pytest.approx(scale)
    assert with_future.parameters()["training_max_date"] == "2024-01-20"
    assert with_future.parameters()["n_matches"] == 20


def test_zero_coefficients_reproduce_offsets_and_constant_inputs_stay_identifiable():
    data = rows()
    for row in data:
        row["shot_features"] = [[5., 6.], [5., 6.]]
    model = OffsetShotModel().fit(data, date(2024, 2, 1))
    assert model.beta == pytest.approx([0., 0.], abs=0)
    assert model.scale.tolist() == [1., 1.]
    assert model.mean.tolist() == [5., 6.]
    assert model.predict([1.3, 2.1], [[99., 77.], [55., 33.]]) == (1.3, 2.1)


def test_known_two_feature_increment_can_be_learned_without_later_results():
    data = [
        dict(match_id="one", date="2024-01-01", fit_before_date="2023-12-31",
             shot_features=[[1., 2.], [3., 4.]], rates=[6., 6.], actual_score=[1, 36]),
        dict(match_id="two", date="2024-01-01", fit_before_date="2023-12-31",
             shot_features=[[1., 4.], [3., 2.]], rates=[6., 6.], actual_score=[9, 4]),
    ]
    cutoff = date(2024, 1, 2)
    later = deepcopy(data[0])
    later.update(match_id="later", date="2024-01-02", actual_score=[100, 200])
    model = OffsetShotModel(regularization=1e-8).fit(data+[later], cutoff)
    assert model.beta == pytest.approx([math.log(2), math.log(3)], rel=1e-6)
    assert model.predict(data[0]["rates"], data[0]["shot_features"]) == pytest.approx([1., 36.], rel=1e-6)
    assert model.parameters()["reference_match_ids"] == ["one", "two"]
    normal = model.predict([1.2, 1.4], [[2., 4.], [3., 2.]])
    swapped = model.predict([1.4, 1.2], [[3., 2.], [2., 4.]])
    assert swapped == normal[::-1]


@pytest.mark.parametrize("field,bad", [
    ("shot_features", [1., 2., 3., 4.]),
    ("rates", [[1., 2.]]),
    ("actual_score", [[1, 2]]),
    ("actual_score", [True, False]),
    ("actual_score", [1.2, 0.]),
    ("rates", [0., 1.]),
])
def test_malformed_match_cannot_be_flattened_into_a_different_target(field, bad):
    data = rows()
    data[0][field] = bad
    with pytest.raises(ValueError):
        OffsetShotModel().fit(data, date(2024, 2, 1))


def test_mutually_complementary_target_lengths_cannot_cross_fixture_boundaries():
    data = rows()
    data[0]["actual_score"] = [1]
    data[1]["actual_score"] = [0, 1, 2]
    with pytest.raises(ValueError):
        OffsetShotModel().fit(data, date(2024, 2, 1))


def test_offset_cutoff_cannot_follow_its_original_target():
    data = rows()
    data[0]["fit_before_date"] = "2024-01-02"
    with pytest.raises(ValueError, match="Offset"):
        OffsetShotModel().fit(data, date(2024, 2, 1))
