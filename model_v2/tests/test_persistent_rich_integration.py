"""Time-safe window assembly without fitting another football predictor.

Tiny fixtures make future-only state values visible, so accidentally using the
held-out period to select columns cannot pass by coincidence. Original unknown
measurements remain in place; only unobserved NEW columns may be omitted.
"""
from copy import deepcopy

import pytest

from football_model.persistent_rich_integration import plan_windows, window_rows
from football_model.persistent_state_features import STATE_FEATURE_NAMES


ORIGINAL_NAMES = ['is_home'] + [f'old_{i:03d}' for i in range(120)]


def row(identity, played, parameter, forecast, *, period='earlier',
        home_state=None, away_state=None, state_parameter='2024-06-02',
        state_history='2024-07-01'):
    sides = []
    for is_home, state_values in ((1., home_state), (0., away_state)):
        original = {name: float(index) for index, name in enumerate(ORIGINAL_NAMES)}
        original['is_home'] = is_home
        # Original all-missing columns must not be dropped by a state adapter.
        original['old_003'] = None
        sides.append({**original, **dict.fromkeys(STATE_FEATURE_NAMES), **(state_values or {})})
    has_state = any(side[k] is not None for side in sides for k in STATE_FEATURE_NAMES)
    return {
        'match_id': identity, 'date': played, 'period': period,
        'competition': 'X', 'home': f'{identity}-H', 'away': f'{identity}-A',
        'fit_before_date': parameter, 'forecast_before_date': forecast,
        'feature_before_date': forecast, 'horizon_days': 3,
        'actual_score': [1, 0], 'actual_outcome': 0, 'side_features': sides,
        'persistent_state_available': has_state,
        'persistent_state_unavailable_reason': None if has_state else 'parameters_not_yet_fitted',
        'state_parameter_training_max_date': state_parameter,
        'state_history_max_date': state_history if has_state else None,
    }


def earlier(identity='past', **kwargs):
    return row(identity, '2024-07-10', '2024-07-01', '2024-07-07', **kwargs)


def held(identity='held', *, period='2024-25', **kwargs):
    return row(identity, '2024-08-12', '2024-08-01', '2024-08-09',
               period=period, **kwargs)


def only_window(rows):
    result = plan_windows(rows)
    assert len(result) == 1
    return result[0]


def test_held_values_cannot_activate_an_unobserved_training_column():
    rows = [earlier(), held(home_state={name: 99. for name in STATE_FEATURE_NAMES})]
    window = only_window(rows)
    assert window['training_ids'] == ['past']
    assert window['target_ids'] == ['held']
    assert window['active_state_features'] == []
    assert set(window['inactive_state_features']) == set(STATE_FEATURE_NAMES)
    assert window['reuse_original'] is True
    assert window['model_feature_names'] == sorted(ORIGINAL_NAMES)

    changed = deepcopy(rows)
    for side in changed[1]['side_features']:
        for name in STATE_FEATURE_NAMES:
            side[name] = -1000.
    assert plan_windows(changed) == plan_windows(rows)


def test_window_views_prune_only_inactive_new_columns_and_never_mutate_inputs():
    rows = [earlier(), held(home_state={STATE_FEATURE_NAMES[0]: .7})]
    before = deepcopy(rows)
    history, target = window_rows(rows, only_window(rows))
    assert rows == before
    for record in history + target:
        for side in record['side_features']:
            assert set(side) == set(ORIGINAL_NAMES)
            assert side['old_003'] is None
    history[0]['side_features'][0]['old_000'] = -999.
    target[0]['side_features'][1]['is_home'] = -1.
    assert rows == before


@pytest.mark.parametrize('side', ('home_state', 'away_state'))
def test_either_training_side_can_establish_an_active_state_column(side):
    active = STATE_FEATURE_NAMES[2]
    rows = [earlier(**{side: {active: .25}}), held()]
    window = only_window(rows)
    assert window['active_state_features'] == [active]
    assert window['reuse_original'] is False
    assert set(window['model_feature_names']) == set(ORIGINAL_NAMES) | {active}
    history, target = window_rows(rows, window)
    for record in history + target:
        for values in record['side_features']:
            assert active in values
            assert set(values) == set(ORIGINAL_NAMES) | {active}
    # Unknown held values remain unknown, despite the column being learned.
    assert all(values[active] is None for values in target[0]['side_features'])


def test_zero_is_an_observed_state_value_and_original_nulls_are_not_removed():
    active = STATE_FEATURE_NAMES[-1]
    window = only_window([earlier(away_state={active: 0.}), held()])
    assert window['active_state_features'] == [active]
    assert 'old_003' in window['model_feature_names']


def test_same_day_results_are_excluded_from_the_training_prefix():
    same_day = row('same-day', '2024-08-01', '2024-07-01', '2024-07-29',
                   home_state={STATE_FEATURE_NAMES[0]: 7.})
    later = row('later', '2024-08-02', '2024-07-01', '2024-07-30',
                away_state={STATE_FEATURE_NAMES[1]: 8.})
    rows = [later, held(), same_day, earlier()]
    window = only_window(rows)
    assert window['training_ids'] == ['past']
    assert window['reuse_original'] is True
    history, target = window_rows(rows, window)
    assert [r['match_id'] for r in history] == ['past']
    assert [r['match_id'] for r in target] == ['held']


def test_windows_use_the_original_period_and_parameter_dates_without_dropping_rows():
    first = held('first')
    second = row('second', '2024-09-10', '2024-08-29', '2024-09-07',
                 period='2024-25', home_state={STATE_FEATURE_NAMES[0]: .1})
    third = row('third', '2025-08-12', '2025-08-01', '2025-08-09',
                period='2025-26', home_state={STATE_FEATURE_NAMES[1]: .2})
    rows = [third, second, earlier(), first]
    windows = plan_windows(rows)
    assert [(w['period'], w['cutoff']) for w in windows] == [
        ('2024-25', '2024-08-01'), ('2024-25', '2024-08-29'), ('2025-26', '2025-08-01')]
    assert [mid for w in windows for mid in w['target_ids']] == ['first', 'second', 'third']
    assert windows[0]['reuse_original'] is True
    assert windows[1]['reuse_original'] is True
    assert windows[2]['active_state_features'] == [STATE_FEATURE_NAMES[0]]
    assert set(windows[2]['training_ids']) == {'past', 'first', 'second'}


def test_target_row_order_is_preserved_and_global_input_is_unchanged():
    rows = [held('z-held'), earlier(), held('a-held')]
    original_ids = [r['match_id'] for r in rows]
    window = only_window(rows)
    history, targets = window_rows(rows, window)
    assert [r['match_id'] for r in targets] == ['z-held', 'a-held']
    assert window['target_ids'] == ['z-held', 'a-held']
    assert [r['match_id'] for r in rows] == original_ids
    assert [r['match_id'] for r in history] == ['past']


@pytest.mark.parametrize('parameter_end', ('2024-08-09', '2024-08-10'))
def test_future_fitted_global_parameters_cannot_supply_nonmissing_target_values(parameter_end):
    rows = [earlier(), held(home_state={STATE_FEATURE_NAMES[0]: .1}, state_parameter=parameter_end)]
    with pytest.raises(ValueError):
        plan_windows(rows)


@pytest.mark.parametrize('observation_day', ('2024-08-09', '2024-08-10'))
def test_state_observation_on_or_after_feature_day_is_rejected(observation_day):
    rows = [earlier(), held(home_state={STATE_FEATURE_NAMES[0]: .1}, state_history=observation_day)]
    with pytest.raises(ValueError):
        plan_windows(rows)


@pytest.mark.parametrize('change', (
    {'fit_before_date': '2024-08-10'},
    {'forecast_before_date': '2024-08-12', 'feature_before_date': '2024-08-12'},
    {'feature_before_date': '2024-08-08'},
))
def test_invalid_parameter_publication_and_feature_date_order_is_rejected(change):
    invalid = held()
    invalid.update(change)
    with pytest.raises(ValueError):
        plan_windows([earlier(), invalid])
