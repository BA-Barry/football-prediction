from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pytest

from football_model.multi_rating_xgb import FixedMultiRatingXGB


START = date(2022, 8, 1)
CUTOFF = date(2022, 12, 1)


def rows(n=90):
    output = []
    for i in range(n):
        played = START + timedelta(days=i + 3)
        result = i % 3
        output.append({
            'match_id': f'game-{i:03d}', 'date': played.isoformat(),
            'fit_before_date': START.isoformat(),
            'forecast_before_date': (played - timedelta(days=1)).isoformat(),
            'actual_score': [[2, 0], [1, 1], [0, 2]][result],
            'match_features': {'signal': float(result), 'maybe': None if i % 4 else float(i % 5),
                               'empty': None, 'optional': np.nan if i % 3 else float(i % 7)},
        })
    return output


def test_actual_xgboost_hda_labels_native_missing_and_metadata():
    data = rows()
    model = FixedMultiRatingXGB().fit(data, CUTOFF)
    probabilities = model.predict_many([row['match_features'] for row in data[:3]], [CUTOFF] * 3)
    assert np.argmax(probabilities, axis=1).tolist() == [0, 1, 2]
    assert np.asarray(probabilities).shape == (3, 3)
    assert np.isfinite(probabilities).all()
    assert np.asarray(probabilities).sum(axis=1) == pytest.approx([1., 1., 1.], abs=1e-12)
    parameters = model.parameters()
    assert parameters['n_matches'] == parameters['n_training_rows'] == 90
    assert parameters['class_counts'] == {'home': 30, 'draw': 30, 'away': 30}
    assert parameters['missing']['all_missing_training_features'] == ['empty']
    assert parameters['weighting']['half_life_days'] is None
    assert parameters['boosting_rounds'] == 100
    assert parameters['configuration']['max_depth'] == 6
    assert parameters['resolved_booster_configuration']['learner']['objective']['name'] == 'multi:softprob'
    assert parameters['upstream_class_mapping'] == {'upstream_0_away': 2, 'upstream_1_draw': 1, 'upstream_2_home': 0}
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters


def test_future_features_and_labels_are_not_used_in_earlier_fit():
    data = rows()
    cutoff = START + timedelta(days=61)
    original = FixedMultiRatingXGB().fit(data, cutoff)
    changed = deepcopy(data)
    for row in changed:
        if row['date'] >= cutoff.isoformat():
            row['actual_score'] = object()
            row['fit_before_date'] = 'not inspected'
            row['forecast_before_date'] = object()
            row['match_features'] = {'future_only': object()}
    poisoned = FixedMultiRatingXGB().fit(changed, cutoff)
    assert original.parameters() == poisoned.parameters()
    sample = {'signal': 1., 'maybe': None}
    assert original.predict(sample, cutoff) == poisoned.predict(sample, cutoff)
    assert original.parameters()['training_max_date'] < cutoff.isoformat()


def test_historical_dates_and_prediction_cutoff_are_respected():
    data = rows()
    late_parameter = deepcopy(data)
    late_parameter[0]['fit_before_date'] = late_parameter[0]['date']
    with pytest.raises(ValueError, match='parameter date'):
        FixedMultiRatingXGB().fit(late_parameter, CUTOFF)
    same_day_forecast = deepcopy(data)
    same_day_forecast[0]['forecast_before_date'] = same_day_forecast[0]['date']
    with pytest.raises(ValueError, match='before fixture'):
        FixedMultiRatingXGB().fit(same_day_forecast, CUTOFF)
    model = FixedMultiRatingXGB().fit(data, CUTOFF)
    with pytest.raises(ValueError, match='at or after'):
        model.predict({'signal': 1.}, CUTOFF - timedelta(days=1))


def test_feature_key_missing_keys_and_entirely_missing_column():
    data = rows()
    for row in data:
        row['joint_features'] = row.pop('match_features')
    model = FixedMultiRatingXGB().fit(data, CUTOFF, feature_key='joint_features')
    sparse = model.predict({}, CUTOFF)
    explicit = model.predict({name: None for name in model.feature_names}, CUTOFF.isoformat())
    assert sparse == explicit
    assert sum(sparse) == pytest.approx(1.)
    copied = model.parameters()
    copied['reference_match_ids'].clear()
    assert len(model.parameters()['reference_match_ids']) == 90


def test_probability_roundoff_correction_preserves_ratios_and_rejects_large_error(monkeypatch):
    model = FixedMultiRatingXGB().fit(rows(), CUTOFF)
    raw = np.asarray([[0.2, 0.3, 0.5000001]])
    monkeypatch.setattr(model.estimator, 'predict_proba', lambda design: raw)
    result = model.predict({}, CUTOFF)
    assert result == pytest.approx((raw[0] / raw.sum()).tolist(), abs=1e-15)
    assert model.parameters()['prediction_roundoff']['rows_corrected'] == 1
    assert model.parameters()['prediction_roundoff']['maximum_raw_sum_deviation'] == pytest.approx(1e-7)
    monkeypatch.setattr(model.estimator, 'predict_proba', lambda design: np.asarray([[0.2, 0.3, 0.6]]))
    with pytest.raises(ValueError, match='sum exceeds'):
        model.predict({}, CUTOFF)


def test_score_labels_cannot_fabricate_an_absent_class():
    data = rows()
    for row in data:
        row['actual_score'] = [1, 0]
    with pytest.raises(ValueError, match='All three'):
        FixedMultiRatingXGB().fit(data, CUTOFF)
