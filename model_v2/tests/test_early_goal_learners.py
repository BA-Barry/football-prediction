from copy import deepcopy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest

from football_model.early_goal_learners import EarlyGoalLearner


def rows(n=90):
    start = date(2022, 8, 1)
    result = []
    for i in range(n):
        played = start+timedelta(days=i)
        result.append({'match_id': f'game-{i:03d}', 'date': played.isoformat(),
                       'fit_before_date': start.isoformat(),
                       'side_features': [
                           {'home': 1., 'strength': float(i % 7)/4, 'maybe': None if i % 4 else 2., 'empty': None},
                           {'home': 0., 'strength': float(i % 5)/4, 'maybe': None if i % 3 else 1., 'empty': None}],
                       'actual_score': [i % 4, (i+1) % 3]})
    return result


@pytest.mark.parametrize('kind', ['glm', 'hgb', 'extra_trees'])
def test_all_presets_produce_positive_full_goal_means_and_readable_metadata(kind):
    data = rows()
    cutoff = date(2022, 11, 1)
    model = EarlyGoalLearner(kind, 'rich').fit(data, cutoff)
    rates = model.predict(data[0]['side_features'], cutoff)
    assert len(rates) == 2 and all(np.isfinite(v) and v > 0 for v in rates)
    p = model.parameters()
    assert p['response'] == 'real_full_time_goals'
    assert p['n_matches'] == 90 and p['n_scoring_sides'] == 180
    assert p['feature_names'] == ['empty', 'home', 'maybe', 'strength']
    assert p['imputation']['all_missing_training_features'] == ['empty']
    assert p['half_life_days'] == 180
    if kind == 'hgb':
        assert p['configuration']['early_stopping'] is False
        assert p['fitted_boosting_iterations'] == 150
        assert 'impurity_feature_importance' not in p
    if kind == 'extra_trees':
        assert p['configuration']['criterion'] == 'poisson'
        assert len(model.estimator.estimators_) == 120
    if kind == 'glm':
        assert p['scaling'] is not None and p['configuration']['alpha'] == .01


@pytest.mark.parametrize('kind', ['glm', 'hgb', 'extra_trees'])
def test_same_day_future_labels_features_and_new_keys_cannot_change_past_fit(kind):
    data = rows()
    cutoff = date(2022, 10, 1)
    original = EarlyGoalLearner(kind, 'basic').fit(data, cutoff)
    changed = deepcopy(data)
    for row in changed:
        if row['date'] >= cutoff.isoformat():
            row['actual_score'] = [float('nan'), True]
            row['side_features'] = [{'future_key': object()}, None]
            row['fit_before_date'] = 'not read'
    model = EarlyGoalLearner(kind, 'basic').fit(changed, cutoff)
    assert model.parameters() == original.parameters()
    assert model.predict(data[0]['side_features'], cutoff) == original.predict(data[0]['side_features'], cutoff)
    assert model.parameters()['n_matches'] == 61


def test_imputation_and_weighted_scaling_use_training_rows_only():
    data = rows()
    cutoff = date(2022, 10, 1)
    model = EarlyGoalLearner('glm', 'rich').fit(data, cutoff)
    p = model.parameters()
    j = p['feature_names'].index('maybe')
    observed = [s['maybe'] for r in data if r['date'] < cutoff.isoformat()
                for s in r['side_features'] if s['maybe'] is not None]
    assert p['imputation']['statistics'][j] == float(np.median(observed))
    design = model.imputer.transform(model._matrix(
        [s for r in data if r['date'] < cutoff.isoformat() for s in r['side_features']], p['feature_names']))
    weights = np.repeat([np.exp(-np.log(2)*(cutoff-date.fromisoformat(r['date'])).days/180)
                         for r in data if r['date'] < cutoff.isoformat()], 2)
    assert np.allclose(p['scaling']['mean'], np.average(design, axis=0, weights=weights))
    before = deepcopy(p)
    model.predict([{}, {'maybe': 3.5}], cutoff)
    assert model.parameters() == before


@pytest.mark.parametrize('measurement', ['1', True, np.bool_(True), float('nan'), float('inf'), object()])
def test_invalid_feature_measurements_are_not_silently_coerced(measurement):
    data = rows()
    data[0]['side_features'][0]['strength'] = measurement
    with pytest.raises(ValueError, match='Features'):
        EarlyGoalLearner('glm', 'rich').fit(data, date(2022, 11, 1))


@pytest.mark.parametrize('counts', [[True, 1], [-1, 0], [1.2, 0], [float('nan'), 1], [1], ['1', 0]])
def test_invalid_full_goal_targets_rejected(counts):
    data = rows()
    data[0]['actual_score'] = counts
    with pytest.raises(ValueError, match='goal'):
        EarlyGoalLearner('glm', 'basic').fit(data, date(2022, 11, 1))


def test_chronology_duplicate_names_and_prediction_dates_checked():
    cutoff = date(2022, 11, 1)
    data = rows()
    with pytest.raises(ValueError, match='unique'):
        EarlyGoalLearner('glm', 'basic').fit(data+[data[0]], cutoff)
    future_features = deepcopy(data)
    future_features[0]['fit_before_date'] = '2022-08-02'
    with pytest.raises(ValueError, match='after'):
        EarlyGoalLearner('glm', 'basic').fit(future_features, cutoff)
    learner = EarlyGoalLearner('glm', 'basic')
    with pytest.raises(ValueError, match='Fit'):
        learner.predict(data[0]['side_features'], cutoff)
    learner.fit(data, cutoff)
    with pytest.raises(ValueError, match='cutoff'):
        learner.predict(data[0]['side_features'], cutoff-timedelta(days=1))
    with pytest.raises(ValueError, match='Unseen'):
        learner.predict([{'future_column': 1.}, {}], cutoff)
    p = learner.parameters()
    p['reference_match_ids'].clear()
    assert len(learner.parameters()['reference_match_ids']) == 90


def test_estimator_and_preprocessing_can_be_saved_and_replayed(tmp_path):
    data = rows()
    cutoff = date(2022, 11, 1)
    learner = EarlyGoalLearner('glm', 'rich').fit(data, cutoff)
    path = tmp_path/'learner.joblib'
    joblib.dump(learner, path)
    loaded = joblib.load(path)
    assert loaded.parameters() == learner.parameters()
    assert loaded.predict(data[0]['side_features'], cutoff) == learner.predict(data[0]['side_features'], cutoff)


@pytest.mark.parametrize('kind', ['glm', 'hgb', 'extra_trees'])
def test_batch_predictions_replay_scalar_and_validate_every_fixture(kind):
    data = rows()
    cutoff = date(2022, 11, 1)
    model = EarlyGoalLearner(kind, 'rich').fit(data, cutoff)
    features = [r['side_features'] for r in data[:8]]
    dates = [cutoff+timedelta(days=i) for i in range(len(features))]
    expected = [model.predict(f, played) for f, played in zip(features, dates)]
    assert np.allclose(model.predict_many(features, dates), expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='cutoff'):
        model.predict_many(features, dates[:-1]+[cutoff-timedelta(days=1)])
    with pytest.raises(ValueError, match='one calendar date'):
        model.predict_many(features, dates[:-1])
    changed = deepcopy(features)
    changed[-1][0]['unknown'] = 1.
    with pytest.raises(ValueError, match='Unseen'):
        model.predict_many(changed, dates)


@pytest.mark.parametrize('kind,feature_set', [('bad', 'rich'), ('glm', 'bad')])
def test_unknown_presets_rejected(kind, feature_set):
    with pytest.raises(ValueError):
        EarlyGoalLearner(kind, feature_set)
