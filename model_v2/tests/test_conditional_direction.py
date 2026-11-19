"""Conditional H/A learning keeps its sample and probability meaning explicit."""
from copy import deepcopy
from datetime import date, timedelta
import json

import joblib
import numpy as np
import pytest

from football_model.conditional_direction import ConditionalDirectionLearner
from football_model.early_outcome_learners import EarlyOutcomeLearner


START = date(2022, 8, 1)
CUTOFF = date(2022, 12, 1)


def rows(n=90):
    data = []
    for i in range(n):
        played = START+timedelta(days=i+3)
        data.append({
            'match_id': f'conditional-{i:03d}', 'date': played.isoformat(),
            'fit_before_date': START.isoformat(),
            'forecast_before_date': (played-timedelta(days=1)).isoformat(),
            'actual_outcome': i % 3,
            'match_features': {'strength': float(i % 7),
                'maybe': None if i % 4 else float(i % 5), 'empty': None},
        })
    return data


@pytest.fixture(scope='module')
def fitted():
    data = rows()
    return ConditionalDirectionLearner().fit(data, CUTOFF), data


def test_conditional_population_and_exact_existing_tree_preset(fitted):
    model, data = fitted
    parameters = model.parameters()
    non_draw = [r for r in data if r['actual_outcome'] != 1]
    draws = [r for r in data if r['actual_outcome'] == 1]
    assert parameters['reference_match_ids'] == [r['match_id'] for r in non_draw]
    assert parameters['earlier_all_match_ids'] == [r['match_id'] for r in data]
    assert parameters['excluded_draw_match_ids'] == [r['match_id'] for r in draws]
    assert parameters['n_matches'] == 60
    assert parameters['n_earlier_all_matches'] == 90
    assert parameters['training_min_date'] == non_draw[0]['date']
    assert parameters['training_max_date'] == non_draw[-1]['date']
    assert parameters['before_date'] == CUTOFF.isoformat()
    assert parameters['half_life_days'] == 180
    assert parameters['configuration'] == EarlyOutcomeLearner('extra_trees')._estimator().get_params()
    assert model.estimator.classes_.tolist() == [0, 2]
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters


def test_sorting_is_by_earlier_fixture_and_id_not_input_order(fitted):
    model, data = fitted
    shuffled = ConditionalDirectionLearner().fit(list(reversed(data)), CUTOFF)
    assert shuffled.parameters() == model.parameters()
    features = [r['match_features'] for r in data[:5]]
    assert shuffled.predict_many(features, [CUTOFF]*5) == model.predict_many(features, [CUTOFF]*5)


def test_future_poison_labels_ids_dates_and_features_do_not_change_earlier_fit():
    data = rows()
    cutoff = START+timedelta(days=61)
    original = ConditionalDirectionLearner().fit(data, cutoff)
    poisoned = deepcopy(data)
    for row in poisoned:
        if row['date'] >= cutoff.isoformat():
            row['match_id'] = None
            row['actual_outcome'] = object()
            row['match_features'] = {'future_only': float('nan')}
            row['fit_before_date'] = 'not read'
            row['forecast_before_date'] = object()
    changed = ConditionalDirectionLearner().fit(poisoned, cutoff)
    assert changed.parameters() == original.parameters()
    assert changed.predict(data[0]['match_features'], cutoff) == original.predict(data[0]['match_features'], cutoff)


def test_earlier_draw_features_are_not_read_or_used_to_impute(fitted):
    original, data = fitted
    poisoned = deepcopy(data)
    for row in poisoned:
        if row['actual_outcome'] == 1:
            del row['match_features']
    changed = ConditionalDirectionLearner().fit(poisoned, CUTOFF)
    assert changed.parameters() == original.parameters()
    assert changed.predict(data[0]['match_features'], CUTOFF) == original.predict(data[0]['match_features'], CUTOFF)


def test_missing_measurements_use_only_earlier_non_draw_population():
    data = rows()
    for row in data:
        if row['actual_outcome'] == 1:
            row['match_features'] = {'maybe': 100000., 'empty': 999., 'draw_only': 4.}
    model = ConditionalDirectionLearner().fit(data, CUTOFF)
    parameters = model.parameters()
    assert 'draw_only' not in parameters['feature_names']
    observed = [r['match_features']['maybe'] for r in data
                if r['actual_outcome'] != 1 and r['match_features']['maybe'] is not None]
    names = parameters['feature_names']
    assert parameters['imputation']['statistics'][names.index('maybe')] == np.median(observed)
    assert parameters['imputation']['all_missing_training_features'] == ['empty']
    assert parameters['imputation']['statistics'][names.index('empty')] == 0.


@pytest.mark.parametrize('case', ['duplicate_draw', 'duplicate_draw_vs_non_draw'])
def test_earlier_ids_are_unique_even_for_excluded_draws(case):
    data = rows()
    if case == 'duplicate_draw':
        data.append(deepcopy(data[1]))
    else:
        data[1]['match_id'] = data[0]['match_id']
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().fit(data, CUTOFF)


@pytest.mark.parametrize('case', ['no_away', 'no_home', 'only_draw'])
def test_both_actual_non_draw_classes_are_required(case):
    data = rows()
    for row in data:
        if case == 'only_draw':
            row['actual_outcome'] = 1
        elif case == 'no_away' and row['actual_outcome'] == 2:
            row['actual_outcome'] = 0
        elif case == 'no_home' and row['actual_outcome'] == 0:
            row['actual_outcome'] = 2
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().fit(data, CUTOFF)


@pytest.mark.parametrize('change', ['draw_publication_same_day', 'parameter_after_publication'])
def test_every_earlier_label_requires_legal_own_information_dates(change):
    data = rows()
    if change == 'draw_publication_same_day':
        data[1]['forecast_before_date'] = data[1]['date']
    else:
        data[0]['fit_before_date'] = data[0]['date']
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().fit(data, CUTOFF)


@pytest.mark.parametrize('outcome', [True, 3, 1., '1', None])
def test_earlier_outcomes_are_not_coerced_or_mistaken_for_draws(outcome):
    data = rows()
    data[1]['actual_outcome'] = outcome
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().fit(data, CUTOFF)


@pytest.mark.parametrize('measurement', [True, '3', float('nan'), float('inf')])
def test_invalid_non_draw_measurements_are_rejected(measurement):
    data = rows()
    data[0]['match_features']['strength'] = measurement
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().fit(data, CUTOFF)


def test_q_is_home_conditional_column_and_prediction_never_requests_actual_result(fitted, monkeypatch):
    model, data = fitted
    monkeypatch.setattr(model.estimator, 'predict_proba', lambda design: np.tile([.17, .83], (len(design), 1)))
    features = [r['match_features'] for r in data[:3]]
    assert model.predict_many(features, [CUTOFF]*3) == [.17, .17, .17]
    assert model.predict(features[0], CUTOFF) == .17


def test_saved_round_trip_batch_scalar_and_deep_copy(fitted, tmp_path):
    model, data = fitted
    features = [r['match_features'] for r in data[:7]]
    forecasts = [CUTOFF+timedelta(days=i) for i in range(7)]
    q = model.predict_many(features, forecasts)
    assert all(type(value) is float and 0 < value < 1 for value in q)
    assert q == [model.predict(f, d) for f, d in zip(features, forecasts, strict=True)]
    filename = tmp_path/'conditional_direction.joblib'
    joblib.dump(model, filename)
    loaded = joblib.load(filename)
    assert loaded.parameters() == model.parameters()
    assert loaded.predict_many(features, forecasts) == q
    copy = model.parameters()
    copy['reference_match_ids'].clear()
    copy['imputation']['statistics'][0] = 100000.
    assert len(model.parameters()['reference_match_ids']) == 60
    assert model.parameters()['imputation']['statistics'][0] != 100000.


def test_prediction_rejects_unfitted_early_publication_unseen_fields_and_misaligned_dates(fitted):
    model, data = fitted
    features = data[0]['match_features']
    with pytest.raises(ValueError):
        ConditionalDirectionLearner().predict(features, CUTOFF)
    with pytest.raises(ValueError):
        model.predict(features, CUTOFF-timedelta(days=1))
    with pytest.raises(ValueError):
        model.predict({'unseen_target_result': 1.}, CUTOFF)
    with pytest.raises(ValueError):
        model.predict_many([features], [])


def test_reversed_class_order_is_rejected_instead_of_returning_away_probability(fitted, monkeypatch):
    model, data = fitted
    monkeypatch.setattr(model.estimator, 'classes_', np.asarray([2, 0]))
    with pytest.raises(ValueError):
        model.predict(data[0]['match_features'], CUTOFF)


@pytest.mark.parametrize('probabilities', [[1., 0.], [0., 1.], [.2, .2], [float('nan'), .5]])
def test_invalid_or_zero_probability_is_reported_without_smoothing(fitted, monkeypatch, probabilities):
    model, data = fitted
    monkeypatch.setattr(model.estimator, 'predict_proba', lambda design: np.tile(probabilities, (len(design), 1)))
    with pytest.raises(ValueError):
        model.predict(data[0]['match_features'], CUTOFF)
