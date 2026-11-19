from copy import deepcopy
from datetime import date

import numpy as np
import pytest
from scipy.special import gammaln

from football_model.goal_rate_scale import apply_goal_scale, fit_goal_scale


def rows():
    return [
        {'match_id': 'a', 'date': '2022-08-10', 'fit_before_date': '2022-08-01',
         'signal_rates': [1.2, .8], 'actual_score': [2, 1]},
        {'match_id': 'b', 'date': '2022-09-10', 'fit_before_date': '2022-08-29',
         'signal_rates': [2., 1.], 'actual_score': [0, 2]},
    ]


def test_scale_is_full_goal_likelihood_minimum_without_changing_side_ratio():
    source = rows()
    source[0]['actual_score'] = [3, 1]
    p = fit_goal_scale(source, date(2023, 8, 1), signal_label='NP mean')
    means = np.array([r['signal_rates'] for r in source])
    counts = np.array([r['actual_score'] for r in source])
    def nll(c):
        mu = c*means
        return np.mean(np.sum(mu-counts*np.log(mu)+gammaln(counts+1), axis=1))
    assert nll(p['scale']) < nll(p['scale']*.99)
    assert nll(p['scale']) < nll(p['scale']*1.01)
    assert abs(p['log_scale_gradient']) < 1e-12
    out = apply_goal_scale([2, 1], p, date(2023, 8, 1))
    assert out[0]/out[1] == 2


def test_future_results_cannot_change_past_scale_and_early_application_rejected():
    initial = rows()
    future = {'match_id': 'later', 'date': '2023-08-01', 'fit_before_date': '2023-08-01',
              'signal_rates': [100, 100], 'actual_score': [50, 50]}
    p = fit_goal_scale(initial, date(2023, 8, 1), signal_label='xG')
    q = fit_goal_scale(initial+[future], date(2023, 8, 1), signal_label='xG')
    assert p == q
    with pytest.raises(ValueError, match='cutoff'):
        apply_goal_scale([1, 1], p, date(2023, 7, 31))


@pytest.mark.parametrize('field,value', [
    ('actual_score', [1.1, 0]), ('actual_score', [True, 0]),
    ('signal_rates', [0, 1]), ('signal_rates', [float('nan'), 1]),
    ('signal_rates', ['1', 1]), ('fit_before_date', '2022-08-11'),
])
def test_invalid_or_future_fitted_signal_rejected(field, value):
    source = deepcopy(rows())
    source[0][field] = value
    with pytest.raises(ValueError):
        fit_goal_scale(source, date(2023, 8, 1), signal_label='xG')


def test_duplicate_calibration_fixture_rejected():
    source = rows()
    with pytest.raises(ValueError, match='unique'):
        fit_goal_scale(source+[source[0]], date(2023, 8, 1), signal_label='xG')
