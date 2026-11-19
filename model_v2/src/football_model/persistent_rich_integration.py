"""Time-aligned reuse of state information with the existing goal learner.

The planning and projection functions do not fit estimators. Only newly added
state columns that have observed values in an earlier training prefix can enter
that window. An all-missing original feature remains part of the original
learner's inputs; an all-missing new feature cannot alter the first reused fit.
"""

from copy import deepcopy
from datetime import date

from .early_goal_learners import _features
from .persistent_state_features import STATE_FEATURE_NAMES


def _calendar(value):
    if not isinstance(value, str):
        raise ValueError('Need an ISO calendar date')
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError('Need a canonical ISO calendar date')
    return result


def _validated(rows):
    records = list(rows)
    ids = [r['match_id'] for r in records]
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Need nonempty rows with unique fixture identities')
    names = set(records[0]['side_features'][0])
    if not set(STATE_FEATURE_NAMES) <= names or not names - set(STATE_FEATURE_NAMES):
        raise ValueError('Need the original features and all six named state fields')
    for row in records:
        played, parameter, forecast, feature = map(_calendar, (
            row['date'], row['fit_before_date'], row['forecast_before_date'], row['feature_before_date']))
        if not parameter <= forecast < played or feature != forecast:
            raise ValueError('Require P<=F<fixture date with feature cutoff equal to F')
        sides = _features(row['side_features'])
        if any(set(side) != names for side in sides):
            raise ValueError('Feature names must agree across original row sides')
        state_parameter = _calendar(row['state_parameter_training_max_date'])
        has_state = any(side[name] is not None for side in sides for name in STATE_FEATURE_NAMES)
        if has_state:
            if state_parameter >= forecast:
                raise ValueError('State parameters use observations at or after row F')
            observed = row.get('state_history_max_date')
            if observed is None or _calendar(observed) >= forecast:
                raise ValueError('State observations must be strictly earlier than row F')
    return records, names


def plan_windows(rows, target_periods=('2024-25', '2025-26')):
    """Select features from training prefixes, never from held fixture values."""
    records, names = _validated(rows)
    windows = []
    for period in target_periods:
        local = [r for r in records if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in local}):
            history = sorted((r for r in records if r['date'] < cutoff),
                             key=lambda r: (r['date'], r['match_id']))
            held = [r for r in local if r['fit_before_date'] == cutoff]
            if not history:
                raise ValueError('Need earlier training fixtures for every target window')
            active = [name for name in STATE_FEATURE_NAMES if any(
                side[name] is not None for row in history for side in row['side_features'])]
            inactive = [name for name in STATE_FEATURE_NAMES if name not in active]
            windows.append({'period': period, 'cutoff': cutoff,
                'training_ids': [r['match_id'] for r in history],
                'target_ids': [r['match_id'] for r in held],
                'active_state_features': active, 'inactive_state_features': inactive,
                'model_feature_names': sorted(names - set(inactive)),
                'reuse_original': not active})
    return windows


def window_rows(rows, window):
    """Copy the chosen prefix and held rows; strip only inactive NEW fields."""
    records = list(rows)
    by_id = {r['match_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate fixture identities')
    wanted = window['training_ids'] + window['target_ids']
    if len(set(wanted)) != len(wanted) or any(mid not in by_id for mid in wanted):
        raise ValueError('Window identities overlap or are missing')
    inactive = set(window['inactive_state_features'])
    if not inactive <= set(STATE_FEATURE_NAMES):
        raise ValueError('Only inactive added state fields can be omitted')

    def copy_row(identity):
        row = deepcopy(by_id[identity])
        row['side_features'] = [{key: value for key, value in side.items() if key not in inactive}
                                for side in row['side_features']]
        if any(sorted(side) != window['model_feature_names'] for side in row['side_features']):
            raise ValueError('Window feature projection changed an original feature')
        return row

    history = [copy_row(mid) for mid in window['training_ids']]
    held = [copy_row(mid) for mid in window['target_ids']]
    if any(r['date'] >= window['cutoff'] for r in history):
        raise ValueError('A training fixture is at or after its parameter cutoff')
    if any(r['period'] != window['period'] or r['fit_before_date'] != window['cutoff'] for r in held):
        raise ValueError('Held fixtures do not match the planned parameter window')
    return history, held
