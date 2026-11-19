from copy import deepcopy
from datetime import date, timedelta
import json

import joblib
import numpy as np
import pytest

from football_model.anchored_residual import (
    AnchoredResidualStack, RESIDUAL_FEATURE_NAMES, _objective_and_gradient, fit_attenuation,
)
from football_model.integrated_stack import ROUTES


def row(i, *, future=False, zero=False):
    played = date(2025 if future else 2024, 8, 10) + timedelta(days=i)
    base = np.array([.48, .27, .25])
    routes = {route:base.tolist() for route in ROUTES}
    if not zero:
        shift = np.zeros(3)
        shift[i % 3] = .05
        shift[(i+1) % 3] = -.05
        routes['xg'] = (base+shift).tolist()
    return {'match_id':f'fixture-{i}', 'date':played.isoformat(),
            'forecast_before_date':(played-timedelta(days=3)).isoformat(),
            'competition':'E0', 'probabilities_by_route':routes,
            'availability_control':None, 'availability_status':None,
            'availability_usage':None, 'actual_outcome':i % 3}


def test_zero_residual_and_zero_attenuation_preserve_reference_exactly():
    model = AnchoredResidualStack().fit([row(i) for i in range(12)], date(2025,8,1))
    targets = [row(1,future=True,zero=True),row(2,future=True,zero=True)]
    reference = np.asarray([r['probabilities_by_route']['rich_performance'] for r in targets])
    assert len(RESIDUAL_FEATURE_NAMES) == 22
    np.testing.assert_array_equal(model.residuals_many(targets), np.zeros((2,3)))
    np.testing.assert_array_equal(model.predict_many(targets), reference)
    varied = [row(1,future=True),row(2,future=True)]
    np.testing.assert_array_equal(model.predict_many(varied,attenuation=0), reference)


def test_future_training_labels_and_earlier_prediction_dates_are_rejected():
    future = row(30,future=True)
    future['actual_outcome'] = 'future label must not be inspected'
    with pytest.raises(ValueError,match='strictly before fit cutoff'):
        AnchoredResidualStack().fit([row(i) for i in range(6)]+[future],date(2025,8,1))
    model = AnchoredResidualStack().fit([row(i) for i in range(6)],date(2025,8,1))
    with pytest.raises(ValueError,match='publication cannot precede'):
        model.predict_many([row(1)])
    with pytest.raises(ValueError,match='strictly before fit cutoff'):
        fit_attenuation([future],date(2025,8,1))


def test_prediction_ignores_target_labels_and_survives_save_restore(tmp_path):
    model = AnchoredResidualStack().fit([row(i) for i in range(15)],date(2025,8,1))
    target = row(1,future=True)
    expected = model.predict_many([target])
    target['actual_outcome'] = object()
    np.testing.assert_array_equal(model.predict_many([target]),expected)
    path = tmp_path/'anchored.joblib'
    joblib.dump(model,path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_many([target]),expected)
    assert expected.sum(axis=1) == pytest.approx([1.])
    assert (expected>0).all()
    parameters = restored.parameters()
    assert parameters['n_features'] == 22
    assert parameters['n_stored_coefficients'] == 66
    assert parameters['n_effective_logit_contrasts'] == 44
    np.testing.assert_allclose(restored.coefficients.sum(axis=1),0,atol=1e-10)
    assert parameters['intercept'] is None and not parameters['scaling']
    assert json.loads(json.dumps(parameters,allow_nan=False)) == parameters


def test_analytic_gradient_matches_finite_difference_with_scaled_l2():
    x = np.array([[.2,-.4],[.6,.1],[-.3,.5],[.4,.2]])
    log_reference = np.log(np.array([[.5,.3,.2],[.4,.3,.3],[.3,.4,.3],[.2,.3,.5]]))
    y = np.array([0,1,2,2])
    coefficients = np.array([.15,-.2,.05,.3,-.1,-.2])
    value, gradient = _objective_and_gradient(coefficients,x,log_reference,y)
    numerical=[]
    step=1e-6
    for i in range(len(coefficients)):
        plus,minus=coefficients.copy(),coefficients.copy()
        plus[i]+=step
        minus[i]-=step
        numerical.append((_objective_and_gradient(plus,x,log_reference,y)[0]
                          -_objective_and_gradient(minus,x,log_reference,y)[0])/(2*step))
    assert value>0
    np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-8)


def test_output_class_permutation_preserves_symmetric_penalty_and_gradient():
    x = np.array([[.2,-.4],[.6,.1],[-.3,.5],[.4,.2]])
    log_reference = np.log(np.array([[.5,.3,.2],[.4,.3,.3],[.3,.4,.3],[.2,.3,.5]]))
    y = np.array([0,1,2,2])
    coefficients = np.array([[.15,-.2,.05],[.3,-.1,-.2]])
    permutation = np.array([2,0,1])
    inverse = np.argsort(permutation)
    value,gradient = _objective_and_gradient(coefficients.ravel(),x,log_reference,y)
    permuted_value,permuted_gradient = _objective_and_gradient(
        coefficients[:,permutation].ravel(),x,log_reference[:,permutation],inverse[y])
    assert permuted_value == pytest.approx(value,abs=1e-14)
    np.testing.assert_allclose(permuted_gradient.reshape(2,3),
        gradient.reshape(2,3)[:,permutation],rtol=0,atol=1e-14)


def test_attenuation_explicit_zero_endpoint_wins_harmful_residuals():
    rows=[]
    for i in range(9):
        record=row(i)
        reference=np.array([.2,.2,.2])
        reference[i % 3]=.6
        residual=np.zeros(3)
        residual[i % 3]=-2.
        rows.append({'match_id':record['match_id'],'date':record['date'],
            'forecast_before_date':record['forecast_before_date'],'meta_fit_date':'2024-08-01',
            'reference_probabilities':reference.tolist(),'residual_logits':residual.tolist(),
            'actual_outcome':i % 3})
    fit=fit_attenuation(rows,date(2025,8,1))
    assert fit['attenuation']==0.
    assert fit['selected_mean_nll']==pytest.approx(-np.log(.6))
    assert fit['residual_model_refits']==0
