"""Shared empirical conversion of earlier out-of-time means to full goals.

This is a single multiplicative Poisson-MLE scale, not a separate penalty or
own-goal mechanism. The caller must supply means fitted before each fixture.
"""
from datetime import date
import math

import numpy as np


def _pair(value, *, counts=False):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2,) or any(isinstance(v, (bool, np.bool_)) for v in raw):
        raise ValueError('Need two numeric values, not booleans')
    if any(not isinstance(v, (int, float, np.integer, np.floating)) for v in raw):
        raise ValueError('Need numeric counts or means')
    values = tuple(map(float, raw))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Need finite counts or means')
    if counts:
        if any(v < 0 or not v.is_integer() for v in values):
            raise ValueError('Full goals must be nonnegative integer counts')
    elif any(v <= 0 for v in values):
        raise ValueError('Signal means must be positive')
    return values


def fit_goal_scale(rows, before_date, *, signal_label):
    """Estimate c = sum(full goals)/sum(signal means) using earlier dates.

    Expected row fields: match_id, date, fit_before_date, signal_rates,
    actual_score. Same-date and later observations cannot train the scale.
    """
    if type(before_date) is not date:
        raise ValueError('Use a calendar-date cutoff')
    if not isinstance(signal_label, str) or not signal_label.strip():
        raise ValueError('Name the signal being calibrated')
    history = sorted((r for r in rows if date.fromisoformat(r['date']) < before_date),
                     key=lambda r: (r['date'], r['match_id']))
    ids = [r['match_id'] for r in history]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Need nonempty unique earlier fixtures')
    goals, means = [], []
    for row in history:
        played = date.fromisoformat(row['date'])
        if date.fromisoformat(row['fit_before_date']) > played:
            raise ValueError('Signal must not be fitted after its target fixture')
        goals.extend(_pair(row['actual_score'], counts=True))
        means.extend(_pair(row['signal_rates']))
    total_goals, total_means = math.fsum(goals), math.fsum(means)
    if total_goals <= 0:
        raise ValueError('No positive full-goal scale is estimable from all-zero goals')
    scale = total_goals/total_means
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid positive goal scale')
    return {
        'model_type': 'single_shared_multiplicative_full_goal_scale',
        'signal_label': signal_label, 'scale': scale, 'log_scale': math.log(scale),
        'n': len(history), 'before_date': before_date.isoformat(),
        'training_min_date': history[0]['date'], 'training_max_date': history[-1]['date'],
        'reference_match_ids': ids, 'sum_full_goals': total_goals,
        'sum_signal_means': total_means,
        'log_scale_gradient': (scale*total_means-total_goals)/len(history),
        'method': 'closed_form_full_score_Poisson_MLE',
        'scope': 'one scale for both scoring sides and all supplied leagues',
        'interpretation': 'empirical goal-scale conversion; no separate penalty or own-goal model',
        'rate_provenance': 'caller verifies strictly earlier training results for each signal mean',
    }


def apply_goal_scale(signal_rates, parameters, match_date):
    if type(match_date) is not date or match_date < date.fromisoformat(parameters['before_date']):
        raise ValueError('Apply the learned scale only at or after its cutoff')
    scale = parameters['scale']
    if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Need a positive finite scale')
    rates = tuple(scale*v for v in _pair(signal_rates))
    if not all(math.isfinite(v) and v > 0 for v in rates):
        raise ValueError('Invalid scaled goal rates')
    return rates
