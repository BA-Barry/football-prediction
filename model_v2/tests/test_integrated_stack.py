from datetime import date, timedelta
from copy import deepcopy
import json

import joblib
import numpy as np
import pytest

from football_model.integrated_stack import (
    FEATURE_NAMES, ROUTES, IntegratedProbabilityStack, _features,
)


def row(i, *, future=False):
    observed = i % 3
    base = np.array([.48, .27, .25]) + np.array([.02, -.01, -.01]) * (i % 4)
    routes = {name: base.tolist() for name in ROUTES}
    shift = np.zeros(3)
    shift[observed] = .06
    shift[(observed+1) % 3] = -.06
    routes['xg'] = (base + shift).tolist()
    played = date(2025, 8, 10) + timedelta(days=i) if future else date(2024, 8, 10) + timedelta(days=i)
    return {'match_id': f'fixture-{i}', 'date': played.isoformat(),
            'forecast_before_date': (played-timedelta(days=3)).isoformat(),
            'competition': 'E0' if i % 2 else 'D1', 'probabilities_by_route': routes,
            'availability_control': None, 'availability_status': None,
            'availability_usage': None, 'actual_outcome': observed}


def test_real_sklearn_fit_outputs_positive_ordered_normalized_probabilities():
    model = IntegratedProbabilityStack().fit([row(i) for i in range(24)], date(2025, 8, 1))
    p = model.predict_many([row(i, future=True) for i in range(3)])
    assert p.shape == (3, 3)
    assert np.isfinite(p).all() and (p > 0).all()
    assert p.sum(axis=1) == pytest.approx(np.ones(3), abs=1e-14)
    assert model.estimator.classes_.tolist() == [0, 1, 2]
    assert model.parameters()['n_features'] == 29


def test_future_training_labels_and_predictions_before_fit_cutoff_are_rejected():
    rows = [row(i) for i in range(9)]
    future = row(99, future=True)
    future['actual_outcome'] = 'must not be read'
    with pytest.raises(ValueError, match='strictly before fit cutoff'):
        IntegratedProbabilityStack().fit(rows+[future], date(2025, 8, 1))
    model = IntegratedProbabilityStack().fit(rows, date(2025, 8, 1))
    with pytest.raises(ValueError, match='publication cannot precede'):
        model.predict_many([rows[0]])


def test_saved_restored_model_and_serializable_feature_parameters_are_identical(tmp_path):
    model = IntegratedProbabilityStack().fit([row(i) for i in range(18)], date(2025, 8, 1))
    targets = [row(1, future=True), row(2, future=True)]
    path = tmp_path/'stack.joblib'
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_many(targets), model.predict_many(targets))
    parameters = restored.parameters()
    assert parameters['feature_names'] == list(FEATURE_NAMES)
    assert len(parameters['coefficients']) == 3
    assert len(parameters['coefficients'][0]) == 29
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters


def test_missing_personnel_is_unavailable_branch_and_partial_probabilities_fail():
    absent = row(1)
    values, _, _ = _features(absent)
    residuals = [i for i,n in enumerate(FEATURE_NAMES) if n.startswith('personnel_') and n.endswith(('home_away', 'draw_away'))]
    assert values[residuals].tolist() == [0., 0., 0., 0.]
    assert values[FEATURE_NAMES.index('personnel_probability_available')] == 0
    observed = deepcopy(absent)
    observed.update(availability_control=[.5, .3, .2], availability_status=[.4, .3, .3],
                    availability_usage=[.35, .3, .35])
    known, _, _ = _features(observed)
    assert known[FEATURE_NAMES.index('personnel_probability_available')] == 1
    assert known[residuals[0]] == pytest.approx(np.log(.4/.3) - np.log(.5/.2))
    observed['availability_usage'] = None
    with pytest.raises(ValueError, match='observed together'):
        _features(observed)


def test_fixed_log_odds_and_balanced_interaction_have_expected_direction():
    observed = row(3)
    values, _, _ = _features(observed)
    base = observed['probabilities_by_route']['rich_performance']
    xg = observed['probabilities_by_route']['xg']
    expected = np.log(xg[0]/xg[2]) - np.log(base[0]/base[2])
    assert values[FEATURE_NAMES.index('delta_xg_log_home_away')] == pytest.approx(expected)
    assert values[FEATURE_NAMES.index('balance_delta_xg_log_home_away')] == pytest.approx(
        expected/(1+abs(np.log(base[0]/base[2]))))
