from copy import deepcopy
from datetime import date, timedelta
import json
import math

import joblib
import numpy as np
import pytest

from football_model.early_outcome_learners import EarlyOutcomeLearner


START = date(2022, 8, 1)
CUTOFF = date(2022, 12, 1)


def rows(n=90):
    output = []
    for i in range(n):
        played = START+timedelta(days=i+3)
        output.append({'match_id': f'game-{i:03d}', 'date': played.isoformat(),
            'fit_before_date': START.isoformat(), 'forecast_before_date': (played-timedelta(days=1)).isoformat(),
            'match_features': {'strength': (i % 11)/4., 'maybe': None if i % 4 else float(i % 5),
                               'empty': None, 'is_home': 1.}, 'actual_outcome': i % 3})
    return output


@pytest.mark.parametrize('kind', ['logistic', 'extra_trees'])
def test_presets_train_one_row_per_fixture_and_return_three_ordered_probabilities(kind):
    data = rows()
    model = EarlyOutcomeLearner(kind).fit(data, CUTOFF)
    p = model.predict(data[0]['match_features'], CUTOFF)
    assert len(p) == 3 and all(0 < v < 1 for v in p)
    assert sum(p) == pytest.approx(1., abs=1e-12)
    parameters = model.parameters()
    assert parameters['n_matches'] == parameters['n_training_rows'] == 90
    assert parameters['class_counts'] == {'home': 30, 'draw': 30, 'away': 30}
    assert parameters['classes'] == [0, 1, 2] and parameters['class_names'] == ['home', 'draw', 'away']
    assert parameters['imputation']['all_missing_training_features'] == ['empty']
    assert parameters['weighting']['n_fixture_weights'] == 90
    assert parameters['weighting']['sum'] == pytest.approx(90.)
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters
    if kind == 'extra_trees':
        assert parameters['configuration']['criterion'] == 'log_loss'
        assert parameters['configuration']['min_samples_leaf'] == 30
        assert len(model.estimator.estimators_) == 120
    else:
        assert parameters['configuration']['C'] == 1.
        assert parameters['configuration']['solver'] == 'lbfgs'
        assert len(parameters['coefficients']) == 3


@pytest.mark.parametrize('kind', ['logistic', 'extra_trees'])
def test_future_poison_labels_features_and_dates_are_not_read_by_earlier_fit(kind):
    data = rows()
    cutoff = START+timedelta(days=61)
    original = EarlyOutcomeLearner(kind).fit(data, cutoff)
    poison = deepcopy(data)
    for row in poison:
        if row['date'] >= cutoff.isoformat():
            row['actual_outcome'] = object()
            row['match_features'] = {'future': float('nan')}
            row['fit_before_date'] = 'not read'
            row['forecast_before_date'] = None
    changed = EarlyOutcomeLearner(kind).fit(poison, cutoff)
    assert changed.parameters() == original.parameters()
    assert changed.predict(data[0]['match_features'], cutoff) == original.predict(data[0]['match_features'], cutoff)


def test_training_only_medians_weight_normalization_and_scaler():
    data = rows()
    cutoff = START+timedelta(days=60)
    model = EarlyOutcomeLearner('logistic').fit(data, cutoff)
    parameters = model.parameters()
    selected = [r for r in data if r['date'] < cutoff.isoformat()]
    observed = [r['match_features']['maybe'] for r in selected if r['match_features']['maybe'] is not None]
    assert parameters['imputation']['statistics'][parameters['feature_names'].index('maybe')] == np.median(observed)
    raw = np.asarray([math.exp(-math.log(2)*(cutoff-date.fromisoformat(r['date'])).days/180) for r in selected])
    assert parameters['weighting']['raw_sum'] == pytest.approx(raw.sum())
    assert parameters['weighting']['normalizing_factor'] == pytest.approx(len(selected)/raw.sum())
    design = model.imputer.transform(model._matrix([r['match_features'] for r in selected], parameters['feature_names']))
    assert np.allclose(parameters['scaling']['mean'], np.average(design, axis=0, weights=raw))
    saved = model.parameters()
    model.predict({'strength': 2.}, cutoff)
    assert model.parameters() == saved


def test_multinomial_saved_coefficients_replay_softmax_without_goal_distribution():
    data = rows()
    model = EarlyOutcomeLearner('logistic').fit(data, CUTOFF)
    parameters = model.parameters()
    design = model.imputer.transform(model._matrix([data[8]['match_features']], model.feature_names))
    design = model.scaler.transform(design)[0]
    logits = np.asarray([parameters['intercepts'][name]+sum(
        parameters['coefficients'][name][feature]*value for feature, value in
        zip(parameters['expanded_feature_names'], design, strict=True)) for name in ('home', 'draw', 'away')])
    unnormalized = np.exp(logits-logits.max())
    assert np.allclose(model.predict(data[8]['match_features'], CUTOFF), unnormalized/unnormalized.sum(), atol=1e-12, rtol=0)
    assert parameters['output_scope'].startswith('1X2 only')


@pytest.mark.parametrize('outcome', [True, np.bool_(False), -1, 3, 1.5, 1., '1', None])
def test_invalid_outcomes_rejected_without_coercion(outcome):
    data = rows()
    data[0]['actual_outcome'] = outcome
    with pytest.raises(ValueError, match='Outcome'):
        EarlyOutcomeLearner('logistic').fit(data, CUTOFF)


@pytest.mark.parametrize('kind', ['logistic', 'extra_trees'])
def test_binary_class_history_cannot_silently_fabricate_missing_draw_or_away(kind):
    data = rows()
    for row in data:
        row['actual_outcome'] %= 2
    with pytest.raises(ValueError, match='All three'):
        EarlyOutcomeLearner(kind).fit(data, CUTOFF)


@pytest.mark.parametrize('value', ['1', True, np.bool_(True), float('nan'), float('inf'), object()])
def test_invalid_measurements_rejected(value):
    data = rows()
    data[0]['match_features']['strength'] = value
    with pytest.raises(ValueError, match='Features'):
        EarlyOutcomeLearner('logistic').fit(data, CUTOFF)


def test_parameter_and_publication_dates_must_precede_each_historical_fixture():
    data = rows()
    parameter_after_publication = deepcopy(data)
    parameter_after_publication[0]['fit_before_date'] = data[0]['date']
    with pytest.raises(ValueError, match='parameter date'):
        EarlyOutcomeLearner('logistic').fit(parameter_after_publication, CUTOFF)
    publication_same_day = deepcopy(data)
    publication_same_day[0]['forecast_before_date'] = data[0]['date']
    with pytest.raises(ValueError, match='publication'):
        EarlyOutcomeLearner('logistic').fit(publication_same_day, CUTOFF)
    with pytest.raises(ValueError, match='unique'):
        EarlyOutcomeLearner('logistic').fit(data+[data[0]], CUTOFF)


@pytest.mark.parametrize('kind', ['logistic', 'extra_trees'])
def test_saved_estimator_replay_and_batch_prediction_agree(tmp_path, kind):
    data = rows()
    model = EarlyOutcomeLearner(kind).fit(data, CUTOFF)
    features = [r['match_features'] for r in data[:7]]
    dates = [CUTOFF+timedelta(days=i) for i in range(7)]
    scalar = [model.predict(f, d) for f, d in zip(features, dates, strict=True)]
    assert np.allclose(model.predict_many(features, dates), scalar, atol=1e-12, rtol=0)
    path = tmp_path/'outcomes.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.parameters() == model.parameters()
    assert loaded.predict_many(features, dates) == model.predict_many(features, dates)
    copied = model.parameters()
    copied['reference_match_ids'].clear()
    assert len(model.parameters()['reference_match_ids']) == 90


def test_prediction_rejects_late_parameters_new_features_and_missing_corresponding_dates():
    data = rows()
    unfitted = EarlyOutcomeLearner('logistic')
    with pytest.raises(ValueError, match='Fit'):
        unfitted.predict(data[0]['match_features'], CUTOFF)
    model = unfitted.fit(data, CUTOFF)
    with pytest.raises(ValueError, match='publication'):
        model.predict(data[0]['match_features'], CUTOFF-timedelta(days=1))
    with pytest.raises(ValueError, match='Unseen'):
        model.predict({'strength': 2., 'unknown_future': 1.}, CUTOFF)
    with pytest.raises(ValueError, match='one publication'):
        model.predict_many([data[0]['match_features']], [])


def test_zero_probability_is_an_explicit_failure_without_smoothing(monkeypatch):
    data = rows()
    model = EarlyOutcomeLearner('extra_trees').fit(data, CUTOFF)
    monkeypatch.setattr(model.estimator, 'predict_proba', lambda design: np.tile([1., 0., 0.], (len(design), 1)))
    with pytest.raises(ValueError, match='Zero class probability'):
        model.predict(data[0]['match_features'], CUTOFF)


def test_class_order_is_explicitly_validated(monkeypatch):
    data = rows()
    model = EarlyOutcomeLearner('logistic').fit(data, CUTOFF)
    monkeypatch.setattr(model.estimator, 'classes_', np.asarray([2, 1, 0]))
    with pytest.raises(ValueError, match='class order'):
        model.predict(data[0]['match_features'], CUTOFF)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match='Unknown'):
        EarlyOutcomeLearner('neural')
